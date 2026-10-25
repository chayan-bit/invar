//! ABI11 owned-numeric task checker, extracted from the CPU ABI11 host. Snapshots
//! are evidence: nothing here imports, restores or executes task bytes.
use super::{compiled, coverage, oracle};
use crate::{
    front::{Span, Type},
    mir,
};
use std::collections::BTreeMap;

pub(super) type TokenKey = (usize, usize, usize);
/// Pending output words: the ten-word packet is written only at termination.
pub(super) const PENDING: [i64; 10] = [i64::MIN; 10];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum TaskError {
    Malformed,
    Allocation,
}

/// Plan facts the checker reads, borrowed from a host plan.
pub(super) struct TaskShape<'a> {
    pub frame_bytes: usize,
    pub task_bytes: usize,
    pub states: usize,
    pub frame_values: usize,
    pub frame_places: usize,
    pub state_functions: &'a [usize],
    pub sites: &'a BTreeMap<TokenKey, (coverage::Rule, Span)>,
    pub program: &'a mir::Program,
    pub storage: &'a compiled::OwnedLayout,
}

/// One root's captured storage. `counters` are the 18 task tail words; the
/// checker requires the task bytes to agree with them.
#[derive(Clone, Copy)]
pub(super) struct TaskWords<'a> {
    pub input: &'a [i64],
    pub output: [i64; 10],
    pub trace: &'a [u64],
    pub task: &'a [u8],
    pub counters: [u64; 18],
    pub heap: &'a [i64],
    pub cleanup: &'a [u64],
}

/// How a root may have advanced between two snapshots.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Advance {
    /// CPU host: exactly one native resume call (`current == previous + 1`).
    NativeCall { previous: u64, current: u64 },
}

fn reserve<T>(n: usize) -> Result<Vec<T>, TaskError> {
    let mut v = Vec::new();
    v.try_reserve_exact(n).map_err(|_| TaskError::Allocation)?;
    Ok(v)
}

pub(super) type History = (u64, usize, usize, coverage::Rule, bool);
/// One validated trace row: Complete flag, frame id and site.
pub(super) type Row = (bool, u64, TokenKey);
pub(super) fn word(bytes: &[u8]) -> Result<u64, TaskError> {
    Ok(u64::from_ne_bytes(
        bytes.try_into().map_err(|_| TaskError::Malformed)?,
    ))
}
fn kind_of(ty: &Type) -> Option<i64> {
    match ty {
        Type::ArrayI32 => Some(0),
        Type::ArrayU32 => Some(1),
        Type::ArrayF32 => Some(258),
        _ => None,
    }
}
fn allocating(program: &crate::mir::Program, key: TokenKey) -> bool {
    let Some(i) = program
        .functions
        .get(key.0)
        .and_then(|f| f.blocks.get(key.1))
        .and_then(|b| b.instructions.get(key.2))
    else {
        return false;
    };
    matches!(i.kind, crate::mir::Operation::Array(_))
        || matches!(&i.kind, crate::mir::Operation::Call { name, .. }
            if matches!(name.as_str(), "filled_i32" | "filled_u32" | "filled_f32"))
}
fn cleanup_names(program: &crate::mir::Program, key: TokenKey) -> &[String] {
    let Some(block) = program
        .functions
        .get(key.0)
        .and_then(|f| f.blocks.get(key.1))
    else {
        return &[];
    };
    match block.instructions.get(key.2).map(|i| &i.kind) {
        Some(crate::mir::Operation::Drop(names)) => names,
        Some(
            crate::mir::Operation::Bind { name, .. } | crate::mir::Operation::Store { name, .. },
        ) => std::slice::from_ref(name),
        None => match &block.end {
            Some(crate::mir::Terminator::Return { cleanup, .. }) => cleanup,
            _ => &[],
        },
        _ => &[],
    }
}
/// Validate a newly captured task against the previous validated snapshot.
/// Pending packets are checked completely here and never decoded. Terminal
/// packets are checked for cross-snapshot consistency, then decoded by the owned-numeric decoder.
pub(super) fn validate(
    plan: &TaskShape<'_>,
    l: oracle::Limits,
    old: &TaskWords<'_>,
    new: &TaskWords<'_>,
    quantum: u64,
    advance: Advance,
) -> Result<(), TaskError> {
    let c = new.counters;
    let p = old.counters;
    let layout = plan.storage;
    let frame_bytes = plan.frame_bytes;
    let frame_len = (plan.frame_values + plan.frame_places + 4) * 8;
    let expected_task = plan.task_bytes;
    // The CPU host advances once per native call.
    let Advance::NativeCall { previous, current } = advance;
    let advanced = current == previous + 1;
    let pending = c[1] == 0;
    if new.task.len() != expected_task
        || expected_task != frame_bytes + 18 * 8
        || frame_len * 64 != frame_bytes
        || new.input != old.input
        || new.trace.len() != l.trace_events * 7
        || new.heap.len() != layout.heap_words()
        || new.cleanup.len() != layout.cleanup_words()
        || c[0] != 1
        || c[1] > 1
        || c[2] > 64
        || c[3] == 0
        || c[3] > 64
        || c[2] > c[3]
        || c[4] > l.fuel
        || c[5] > l.transitions
        || c[6] > l.trace_events as u64
        || c[7] == 0
        || c[7] > c[5] + 1
        || c[8] > layout.allocations() as u64
        || c[9] > layout.elements() as u64
        || c[10] > layout.cleanup_events() as u64
        || c[11..14] != [l.fuel, l.transitions, l.trace_events as u64]
        || c[14] > 13
        || c[14] == 8
        || (pending && (c[2] == 0 || c[14..18] != [0, 0, 0, 0]))
        || (!pending && c[14] == 0 && c[2] != 0)
        || !advanced
        || c[3] < p[3]
        || c[4] < p[4]
        || c[5] < p[5]
        || c[6] < p[6]
        || c[7] < p[7]
        || c[8] < p[8]
        || c[10] < p[10]
        || c[5] - p[5] > quantum
        || (pending && c[5] - p[5] != quantum)
    {
        return Err(TaskError::Malformed);
    }
    for (bytes, counter) in new.task[frame_bytes..].chunks_exact(8).zip(c) {
        if word(bytes)? != counter {
            return Err(TaskError::Malformed);
        }
    }
    let mut last_id = None;
    let mut active = reserve(64)?;
    for (index, frame) in new.task[..frame_bytes]
        .chunks_exact(frame_len)
        .take(c[2] as usize)
        .enumerate()
    {
        let mut m = [0; 4];
        for (slot, bytes) in m.iter_mut().zip(frame[frame_len - 32..].chunks_exact(8)) {
            *slot = word(bytes)?;
        }
        if m[0] >= plan.states as u64
            || m[1] > plan.frame_values as u64
            || m[2] >= c[7]
            || m[3] > c[5]
            || last_id.is_some_and(|id| m[2] <= id)
            || (last_id.is_none() && m[2] != 0)
        {
            return Err(TaskError::Malformed);
        }
        active.push((index, m[2], m[3], plan.state_functions[m[0] as usize]));
        last_id = Some(m[2]);
    }
    let used = c[6] as usize * 7;
    let previous = p[6] as usize * 7;
    if previous > old.trace.len()
        || new.trace[..previous] != old.trace[..previous]
        || new.trace[used..].iter().any(|w| *w != 0)
    {
        return Err(TaskError::Malformed);
    }
    let mut stack: Vec<(u64, u64, TokenKey, coverage::Rule)> = reserve(64)?;
    let mut history = reserve(c[7] as usize)?;
    history.resize(c[7] as usize, None::<History>);
    let mut rows: Vec<Row> = reserve(c[6] as usize)?;
    let mut completed_allocations = 0usize;
    for row in new.trace[..used].chunks_exact(7) {
        let key = (
            usize::try_from(row[3]).map_err(|_| TaskError::Malformed)?,
            usize::try_from(row[4]).map_err(|_| TaskError::Malformed)?,
            usize::try_from(row[5]).map_err(|_| TaskError::Malformed)?,
        );
        let (rule, _) = plan.sites.get(&key).ok_or(TaskError::Malformed)?;
        if row[0] > 1
            || row[1] >= c[7]
            || row[1] > c[5]
            || row[2] > c[5]
            || row[6] != *rule as u64
            || *rule == coverage::Rule::Tick
        {
            return Err(TaskError::Malformed);
        }
        let last = &mut history[row[1] as usize];
        if let Some((activation, function, block, previous_rule, complete)) = *last {
            let expected = activation
                + u64::from(
                    complete
                        && matches!(previous_rule, coverage::Rule::Goto | coverage::Rule::Branch),
                );
            if row[2] != expected
                || key.0 != function
                || (row[2] == activation && key.1 != block)
                || (complete && previous_rule == coverage::Rule::Return)
            {
                return Err(TaskError::Malformed);
            }
        } else if row[2] != 0 {
            return Err(TaskError::Malformed);
        }
        *last = Some((row[2], key.0, key.1, *rule, row[0] == 1));
        let event = (row[1], row[2], key, *rule);
        if row[0] == 0 {
            if stack.len() == 64
                || stack
                    .last()
                    .is_some_and(|parent| parent.3 != coverage::Rule::Call || row[1] <= parent.0)
            {
                return Err(TaskError::Malformed);
            }
            stack.push(event);
        } else {
            if stack.pop() != Some(event) {
                return Err(TaskError::Malformed);
            }
            if allocating(plan.program, key) {
                completed_allocations += 1;
            }
        }
        rows.push((row[0] == 1, row[1], key));
    }
    for &(_, frame, activation, function) in &active {
        if let Some((last_activation, last_function, _, rule, complete)) = history[frame as usize] {
            let expected = last_activation
                + u64::from(
                    complete && matches!(rule, coverage::Rule::Goto | coverage::Rule::Branch),
                );
            if activation != expected || function != last_function {
                return Err(TaskError::Malformed);
            }
        } else if activation != 0 {
            return Err(TaskError::Malformed);
        }
    }
    let owners = storage(plan, old, new, &rows)?;
    if pending {
        // Allocation and its Complete event execute without an intervening
        // yield, so a pending task has exactly one owner per Complete event.
        if new.output != PENDING
            || stack.iter().any(|e| e.3 != coverage::Rule::Call)
            || stack.len() + 1 != c[2] as usize
            || completed_allocations != c[8] as usize
        {
            return Err(TaskError::Malformed);
        }
        // Active array places hold zero or a live owner of the declared kind.
        // Loan aliases may share an owner; stale SSA values are unconstrained.
        for &(index, _, _, function) in &active {
            let frame = &new.task[index * frame_len..(index + 1) * frame_len];
            let f = plan
                .program
                .functions
                .get(function)
                .ok_or(TaskError::Malformed)?;
            if f.declaration_order.len() > plan.frame_places {
                return Err(TaskError::Malformed);
            }
            for (place, name) in f.declaration_order.iter().enumerate() {
                let Some(kind) = f.locals.get(name).and_then(|v| kind_of(&v.ty)) else {
                    continue;
                };
                let at = (plan.frame_values + place) * 8;
                let handle = word(&frame[at..at + 8])?;
                if handle == 0 {
                    continue;
                }
                let index = usize::try_from(handle - 1).map_err(|_| TaskError::Malformed)?;
                if handle > c[8] || owners.get(index) != Some(&Some(kind)) {
                    return Err(TaskError::Malformed);
                }
            }
        }
    } else {
        let expected = [
            c[14], c[15], c[16], c[17], c[4], c[5], c[3], c[6], c[10], c[8],
        ];
        if new.output.map(|w| w as u64) != expected {
            return Err(TaskError::Malformed);
        }
    }
    Ok(())
}
/// Check heap descriptors, pool, cleanup rows and their monotone relation to
/// the previous snapshot. Returns each issued descriptor's kind if live.
fn storage(
    plan: &TaskShape<'_>,
    old: &TaskWords<'_>,
    new: &TaskWords<'_>,
    rows: &[Row],
) -> Result<Vec<Option<i64>>, TaskError> {
    let layout = plan.storage;
    let (c, p) = (new.counters, old.counters);
    let allocated = c[8] as usize;
    let count = c[10] as usize;
    let slots = layout.allocations();
    let (descriptors, pool) = new.heap.split_at(slots * 4);
    if descriptors[allocated * 4..].iter().any(|w| *w != 0)
        || new.cleanup[count * 4..].iter().any(|w| *w != 0)
        || pool
            .iter()
            .any(|w| *w < i64::from(i32::MIN) || *w > i64::from(u32::MAX))
    {
        return Err(TaskError::Malformed);
    }
    let mut owners = reserve(allocated)?;
    let mut live = reserve(allocated)?;
    let mut dead_lengths = BTreeMap::<usize, usize>::new();
    let (mut ever, mut live_sum, mut dead) = (0usize, 0usize, 0usize);
    for (index, d) in descriptors[..allocated * 4].chunks_exact(4).enumerate() {
        let offset = usize::try_from(d[0]).map_err(|_| TaskError::Malformed)?;
        let len = usize::try_from(d[1]).map_err(|_| TaskError::Malformed)?;
        let end = offset.checked_add(len).ok_or(TaskError::Malformed)?;
        if len > crate::array_limits::MAX_ARRAY_LEN
            || end > pool.len()
            || !matches!(d[2], 0 | 1 | 258)
            || !matches!(d[3], 0 | 1)
        {
            return Err(TaskError::Malformed);
        }
        // Issued owners keep length and kind; a dead owner never resurrects.
        // Offsets may change through stable-ID compaction.
        if index < p[8] as usize {
            let o = &old.heap[index * 4..index * 4 + 4];
            if o[1] != d[1] || o[2] != d[2] || (o[3] == 0 && d[3] == 1) {
                return Err(TaskError::Malformed);
            }
        }
        ever = ever.saturating_add(len).min(pool.len());
        if d[3] == 1 {
            if pool[offset..end].iter().any(|w| {
                if d[2] == 0 {
                    i32::try_from(*w).is_err()
                } else {
                    u32::try_from(*w).is_err()
                }
            }) {
                return Err(TaskError::Malformed);
            }
            live_sum += len;
            live.push((offset, end));
            owners.push(Some(d[2]));
        } else {
            dead += 1;
            *dead_lengths.entry(len).or_default() += 1;
            owners.push(None);
        }
    }
    if pool[ever..].iter().any(|w| *w != 0) {
        return Err(TaskError::Malformed);
    }
    live.sort_unstable();
    if live.windows(2).any(|pair| pair[0].1 > pair[1].0)
        || live_sum != c[9] as usize
        || dead != count
    {
        return Err(TaskError::Malformed);
    }
    let kept = p[10] as usize * 4;
    if new.cleanup[..kept] != old.cleanup[..kept] {
        return Err(TaskError::Malformed);
    }
    // Rows follow eligible cleanup sites in trace order and reconcile with
    // dead owner lengths.
    let (mut event_at, mut name_at) = (0usize, 0usize);
    for row in new.cleanup[..count * 4].chunks_exact(4) {
        let function = usize::try_from(row[0]).map_err(|_| TaskError::Malformed)?;
        let place = usize::try_from(row[1]).map_err(|_| TaskError::Malformed)?;
        let len = usize::try_from(row[2]).map_err(|_| TaskError::Malformed)?;
        let f = plan
            .program
            .functions
            .get(function)
            .ok_or(TaskError::Malformed)?;
        let name = f.declaration_order.get(place).ok_or(TaskError::Malformed)?;
        if row[3] >= c[7] || f.locals.get(name).is_none_or(|v| kind_of(&v.ty).is_none()) {
            return Err(TaskError::Malformed);
        }
        let remaining = dead_lengths.get_mut(&len).ok_or(TaskError::Malformed)?;
        if *remaining == 0 {
            return Err(TaskError::Malformed);
        }
        *remaining -= 1;
        let mut found = false;
        while let Some(&(complete, frame, key)) = rows.get(event_at) {
            let names = if complete {
                &[][..]
            } else {
                cleanup_names(plan.program, key)
            };
            if frame == row[3] && key.0 == function {
                while name_at < names.len() {
                    let candidate = &names[name_at];
                    name_at += 1;
                    if candidate == name {
                        found = true;
                        break;
                    }
                }
                if found {
                    break;
                }
            }
            event_at += 1;
            name_at = 0;
        }
        if !found {
            return Err(TaskError::Malformed);
        }
    }
    if dead_lengths.values().any(|n| *n != 0) {
        return Err(TaskError::Malformed);
    }
    Ok(owners)
}
