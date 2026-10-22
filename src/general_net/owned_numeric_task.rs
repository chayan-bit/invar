//! Shared ABI11/ABI12 owned-numeric task checker. Extracted unchanged in meaning
//! from the CPU ABI11 host; the only parameter is the advance rule. Snapshots
//! are evidence: nothing here imports, restores or executes task bytes.
use super::{compiled, coverage, oracle};
use crate::{
    front::{Span, Type},
    mir,
};
use std::collections::BTreeMap;

pub(super) type TokenKey = (usize, usize, usize);
/// Pending output words: the ABI10 packet is written only at termination.
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
    /// Device epoch for a root that was pending before the epoch: it advanced
    /// exactly `quantum` transitions if still pending, or at most `quantum` if
    /// it terminated. Roots terminal before the epoch use
    /// [`validate_device_epoch`], which requires byte identity instead.
    #[allow(
        dead_code,
        reason = "ABI12 stage 2a: consumed by the stage-2c device host; exercised by owned tests now"
    )]
    DeviceEpoch,
}

fn reserve<T>(n: usize) -> Result<Vec<T>, TaskError> {
    let mut v = Vec::new();
    v.try_reserve_exact(n).map_err(|_| TaskError::Allocation)?;
    Ok(v)
}

/// Device advance rule for one root over one host-fenced epoch. A root that
/// was terminal before the epoch must be byte-identical in every slice (no
/// mutation or replay after `done`); a pending root must satisfy the full
/// ABI11 checker with [`Advance::DeviceEpoch`].
#[allow(
    dead_code,
    reason = "ABI12 stage 2a: consumed by the stage-2c device host; exercised by owned tests now"
)]
pub(super) fn validate_device_epoch(
    plan: &TaskShape<'_>,
    l: oracle::Limits,
    old: &TaskWords<'_>,
    new: &TaskWords<'_>,
    quantum: u64,
) -> Result<(), TaskError> {
    if old.counters[1] == 1 {
        let identical = old.input == new.input
            && old.output == new.output
            && old.trace == new.trace
            && old.task == new.task
            && old.counters == new.counters
            && old.heap == new.heap
            && old.cleanup == new.cleanup;
        return if identical {
            Ok(())
        } else {
            Err(TaskError::Malformed)
        };
    }
    validate(plan, l, old, new, quantum, Advance::DeviceEpoch)
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
/// packets are checked for cross-snapshot consistency, then decoded by ABI10.
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
    // The CPU host advances once per native call. A device epoch never runs a
    // lane that was already terminal (see `validate_device_epoch`).
    let advanced = match advance {
        Advance::NativeCall { previous, current } => current == previous + 1,
        Advance::DeviceEpoch => p[1] == 0,
    };
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

#[cfg(test)]
mod tests {
    //! Device advance rule on real ABI11 CPU executions: a C++ driver runs
    //! every root each epoch like a device launch (terminal roots return
    //! immediately) and dumps every root's storage after each epoch.
    use super::*;
    use crate::{
        eval,
        general_net::{numeric_owned_host, provenance},
        numeric_contract::F32Bits,
    };
    use std::{collections::BTreeSet, fmt::Write as _, fs, path::PathBuf};

    const SOURCE: &str = "fn mix(xs: mut Array<f32>, n: u32) -> Unit { if n > 0u32 { xs[0u32] = fma(xs[0u32], xs[1u32], xs[2u32]); mix(xs, n - 1u32); } } fn run(a: f32, b: f32, c: f32, n: u32) -> f32 { var xs: Array<f32> = [a, b, c]; mix(xs, n % 4u32); let t: f32 = xs[0u32]; if t < b { return t / b; } if t == t { return t - c + f32_from_u32(n) - f32_from_i32(0 - 3); } return xs[n]; } fn main() -> i32 { return 0; }";
    const QUANTUM: u64 = 3;
    const ROOTS: [[u32; 4]; 8] = [
        [0x3f80_0000, 0x4000_0000, 0x3f80_0000, 0],
        [0x3f80_0001, 0x3f7f_fffe, 0xbf80_0000, 3],
        [0x7fa1_2345, 0x3f80_0000, 0x0000_0001, 1],
        [0x7fa1_2345, 0x3f80_0000, 0x0000_0001, 4],
        [0x8000_0000, 0x0000_0000, 0x7f80_0000, 2],
        [0x4b80_0000, 0xcb80_0000, 0x3f80_0000, 5],
        [0x0000_0001, 0x7f7f_ffff, 0xff80_0000, 7],
        [0x3eaa_aaab, 0x4040_0000, 0x7fc0_0000, 6],
    ];
    fn limits() -> oracle::Limits {
        oracle::Limits {
            fuel: 1000,
            transitions: 10_000,
            trace_events: 400,
            array_elements: 64,
        }
    }
    fn storage() -> compiled::OwnedLimits {
        compiled::OwnedLimits {
            elements: 64,
            allocations: 16,
            cleanup_events: 32,
        }
    }

    /// Owned checker inputs derived from the artifact, as the ABI11 host does.
    struct Owned {
        artifact: provenance::ProvenancedArtifact,
        program: mir::Program,
        sites: BTreeMap<TokenKey, (coverage::Rule, Span)>,
        state_functions: Vec<usize>,
        frame_values: usize,
        frame_places: usize,
    }
    impl Owned {
        fn new() -> Self {
            let artifact = provenance::compile_resumable_owned_numeric(
                SOURCE.as_bytes(),
                "run",
                compiled::Target::Cpu,
                Default::default(),
                storage(),
            )
            .unwrap();
            let (net, proof) = artifact.checked_net();
            let program = net.compiled_mir(proof).unwrap();
            let (mut state_functions, mut frame_values, mut frame_places) = (Vec::new(), 1, 1);
            for (function_id, function) in program.functions.iter().enumerate() {
                let mut values = BTreeSet::new();
                for block in &function.blocks {
                    for (id, _) in block
                        .params
                        .iter()
                        .chain(block.instructions.iter().filter_map(|i| i.output.as_ref()))
                    {
                        values.insert(*id);
                    }
                    state_functions.extend(std::iter::repeat_n(
                        function_id,
                        block.instructions.len() + 1,
                    ));
                }
                frame_values = frame_values.max(values.len());
                frame_places = frame_places.max(function.locals.len());
            }
            let mut sites = BTreeMap::new();
            for (token, rule, span) in net
                .agents
                .iter()
                .map(|a| (a.token, a.rule, a.span))
                .chain(net.control.iter().map(|a| (a.token, a.rule, a.span)))
            {
                let operator = program
                    .functions
                    .get(token.block.function)
                    .and_then(|f| f.blocks.get(token.block.block))
                    .and_then(|b| b.instructions.get(token.position))
                    .and_then(|i| match i.kind {
                        mir::Operation::Binary { operator, .. }
                        | mir::Operation::Unary { operator, .. } => Some(operator),
                        _ => None,
                    });
                sites.insert(
                    (token.block.function, token.block.block, token.position),
                    (rule, operator.unwrap_or(span)),
                );
            }
            assert_eq!(state_functions.len(), artifact.artifact().states());
            Self {
                artifact,
                program,
                sites,
                state_functions,
                frame_values,
                frame_places,
            }
        }
        fn shape(&self) -> TaskShape<'_> {
            let a = self.artifact.artifact();
            TaskShape {
                frame_bytes: a.frame_bytes(),
                task_bytes: a.continuation_layout().unwrap().task_bytes(),
                states: a.states(),
                frame_values: self.frame_values,
                frame_places: self.frame_places,
                state_functions: &self.state_functions,
                sites: &self.sites,
                program: &self.program,
                storage: a.owned_layout().unwrap(),
            }
        }
    }

    #[derive(Clone)]
    struct Lane {
        input: Vec<i64>,
        output: [i64; 10],
        trace: Vec<u64>,
        heap: Vec<i64>,
        cleanup: Vec<u64>,
        task: Vec<u8>,
    }
    impl Lane {
        fn counters(&self) -> [u64; 18] {
            let tail = &self.task[self.task.len() - 144..];
            std::array::from_fn(|i| word(&tail[i * 8..i * 8 + 8]).unwrap())
        }
        fn words(&self) -> TaskWords<'_> {
            TaskWords {
                input: &self.input,
                output: self.output,
                trace: &self.trace,
                task: &self.task,
                counters: self.counters(),
                heap: &self.heap,
                cleanup: &self.cleanup,
            }
        }
    }

    /// Run all roots in device-style epochs and return the snapshot after each
    /// epoch (index 0 is the initial zero state).
    fn epochs(owned: &Owned) -> Vec<Vec<Lane>> {
        let a = owned.artifact.artifact();
        let layout = a.owned_layout().unwrap();
        let (h, c) = (layout.heap_words(), layout.cleanup_words());
        let task_bytes = a.continuation_layout().unwrap().task_bytes();
        let l = limits();
        let t = l.trace_events * 7;
        let r = ROOTS.len();
        let inputs = ROOTS
            .iter()
            .flatten()
            .map(|w| w.to_string())
            .collect::<Vec<_>>()
            .join(",");
        let mut cpp = a.source().to_owned();
        write!(
            cpp,
            r#"
#include <climits>
#include <cstdio>
#include <vector>
int main() {{
    const unsigned R = {r}, P = 4; const unsigned long long T = {t}, H = {h}, C = {c};
    const long long roots[] = {{{inputs}}};
    std::vector<gnet_w> in(R * P), out(R * 10), heap(R * H); std::vector<gnet_u> trace(R * T + 1), clean(R * C + 1);
    std::vector<GnetTask> task(R);
    for (unsigned i = 0; i < R * P; ++i) in[i] = roots[i];
    // ABI11 pending sentinel, as the native owner initializes it.
    for (auto& w : out) w = LLONG_MIN;
    FILE* f = std::fopen("epochs.bin", "wb"); if (!f) return 2;
    auto dump = [&]() {{ for (unsigned i = 0; i < R; ++i) {{
        std::fwrite(&in[i * P], 8, P, f); std::fwrite(&out[i * 10], 8, 10, f); std::fwrite(&trace[i * T], 8, T, f);
        std::fwrite(&heap[i * H], 8, H, f); std::fwrite(&clean[i * C], 8, C, f); std::fwrite(&task[i], 1, sizeof(GnetTask), f); }} }};
    static_assert(sizeof(GnetTask) == {task_bytes}, "task bytes");
    dump();
    for (unsigned e = 0; e < 100000; ++e) {{
        unsigned pending = 0; for (unsigned i = 0; i < R; ++i) pending += task[i].done ? 0 : 1;
        if (!pending) break;
        // Every lane each epoch, as a device launch: terminal roots return.
        for (unsigned i = 0; i < R; ++i)
            gnet_resume(&in[i * P], &out[i * 10], &trace[i * T], {fuel}, {transitions}, {cap}, &heap[i * H], &clean[i * C], &task[i], {QUANTUM});
        dump();
    }}
    return std::fclose(f) == 0 ? 0 : 3;
}}
"#,
            fuel = l.fuel,
            transitions = l.transitions,
            cap = l.trace_events,
        )
        .unwrap();
        let root = std::env::var_os("CARGO_TARGET_TMPDIR")
            .map(PathBuf::from)
            .unwrap_or_else(std::env::temp_dir);
        // Unique per invocation: tests run in parallel within one process.
        static NEXT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        let dir = root.join(format!(
            "abi12-advance-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
        ));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        fs::write(dir.join("driver.cpp"), cpp).unwrap();
        let compiler = std::env::var_os("CXX").unwrap_or_else(|| "c++".into());
        let built = crate::child_deadline::command(180)
            .arg(compiler)
            .args([
                "-std=c++17",
                "-O1",
                "-fsanitize=address,undefined",
                "-fno-sanitize-recover=all",
                "driver.cpp",
                "-o",
                "driver",
            ])
            .current_dir(&dir)
            .output()
            .unwrap();
        assert!(
            built.status.success(),
            "{}",
            String::from_utf8_lossy(&built.stderr)
        );
        let run = crate::child_deadline::command(120)
            .arg("./driver")
            .env("ASAN_OPTIONS", "detect_leaks=0")
            .current_dir(&dir)
            .output()
            .unwrap();
        assert!(
            run.status.success(),
            "{}",
            String::from_utf8_lossy(&run.stderr)
        );
        let bytes = fs::read(dir.join("epochs.bin")).unwrap();
        let _ = fs::remove_dir_all(&dir);
        let lane_bytes = (4 + 10 + t + h + c) * 8 + task_bytes;
        assert_eq!(bytes.len() % (lane_bytes * r), 0);
        let read_words =
            |b: &[u8]| -> Vec<u64> { b.chunks_exact(8).map(|w| word(w).unwrap()).collect() };
        bytes
            .chunks_exact(lane_bytes * r)
            .map(|epoch| {
                epoch
                    .chunks_exact(lane_bytes)
                    .map(|lane| {
                        let mut at = 0;
                        let mut take = |n: usize| {
                            let s = &lane[at..at + n];
                            at += n;
                            s
                        };
                        let input = read_words(take(32)).into_iter().map(|w| w as i64).collect();
                        let output: Vec<i64> =
                            read_words(take(80)).into_iter().map(|w| w as i64).collect();
                        let trace = read_words(take(t * 8));
                        let heap = read_words(take(h * 8))
                            .into_iter()
                            .map(|w| w as i64)
                            .collect();
                        let cleanup = read_words(take(c * 8));
                        let task = take(task_bytes).to_vec();
                        Lane {
                            input,
                            output: output.try_into().unwrap(),
                            trace,
                            heap,
                            cleanup,
                            task,
                        }
                    })
                    .collect()
            })
            .collect()
    }

    #[test]
    fn device_rule_accepts_real_epochs_and_matches_independent_abi11_outputs() {
        let owned = Owned::new();
        let shape = owned.shape();
        let snapshots = epochs(&owned);
        assert!(snapshots.len() > 10);
        let (mut pending_checked, mut terminal_checked) = (0, 0);
        for pair in snapshots.windows(2) {
            for (old, new) in pair[0].iter().zip(&pair[1]) {
                validate_device_epoch(&shape, limits(), &old.words(), &new.words(), QUANTUM)
                    .unwrap();
                if old.counters()[1] == 1 {
                    terminal_checked += 1;
                } else {
                    pending_checked += 1;
                }
            }
        }
        assert!(pending_checked > 50 && terminal_checked > 10);
        // Independent ABI11 output check: the unchanged ABI10 decoder on the
        // final packets equals the independent net oracle per root.
        let (net, proof) = owned.artifact.checked_net();
        let last = snapshots.last().unwrap();
        for (lane, root) in last.iter().zip(ROOTS) {
            assert_eq!(lane.counters()[1], 1, "every root terminal");
            let decoded = numeric_owned_host::decode(
                owned.artifact.artifact().result(),
                owned.artifact.source().len(),
                &owned.sites,
                &owned.program,
                owned.artifact.artifact().owned_layout().unwrap(),
                limits(),
                &invar_foreign::numeric_owned::Invocation {
                    output: lane.output,
                    trace: lane.trace.clone(),
                    heap: lane.heap.clone(),
                    cleanup: lane.cleanup.clone(),
                },
            )
            .unwrap();
            let args: Vec<_> = root[..3]
                .iter()
                .map(|w| eval::Value::F32(F32Bits::from_bits(*w)))
                .chain([eval::Value::U32(root[3])])
                .collect();
            let want = oracle::observe_call(net, proof, "run", &args, limits()).unwrap();
            assert_eq!(decoded.events, want.events, "{root:x?}");
            assert_eq!(decoded.transitions, want.transitions);
            match (&decoded.semantic.result, &want.semantic.result) {
                (Ok(eval::Value::F32(a)), Ok(eval::Value::F32(b))) => {
                    assert_eq!(a.to_bits(), b.to_bits(), "{root:x?}")
                }
                (a, b) => assert_eq!(a, b, "{root:x?}"),
            }
        }
        eprintln!(
            "device rule: {pending_checked} pending and {terminal_checked} terminal root-epochs accepted"
        );
    }

    /// First epoch index at which `lane` is pending before and after.
    fn pending_pair(s: &[Vec<Lane>], lane: usize) -> usize {
        (0..s.len() - 2)
            .find(|&e| s[e + 1][lane].counters()[1] == 0 && s[e][lane].counters()[0] == 1)
            .expect("lane pending over two epochs")
    }

    #[test]
    fn device_rule_refuses_invalid_advancement() {
        let owned = Owned::new();
        let shape = owned.shape();
        let s = epochs(&owned);
        let l = limits();
        let lane = 1;
        let e = pending_pair(&s, lane);
        let (old, new, next) = (&s[e][lane], &s[e + 1][lane], &s[e + 2][lane]);
        let check =
            |a: &Lane, b: &Lane, q| validate_device_epoch(&shape, l, &a.words(), &b.words(), q);
        check(old, new, QUANTUM).unwrap();
        // No advance by a pending root.
        assert_eq!(check(old, old, QUANTUM), Err(TaskError::Malformed));
        // Two quanta in one epoch.
        assert_eq!(check(old, next, QUANTUM), Err(TaskError::Malformed));
        // Advanced by a different quantum than submitted (either direction).
        assert_eq!(check(old, new, QUANTUM + 1), Err(TaskError::Malformed));
        assert_eq!(check(old, new, QUANTUM - 1), Err(TaskError::Malformed));
        // Going backwards.
        assert_eq!(check(new, old, QUANTUM), Err(TaskError::Malformed));
    }

    #[test]
    fn device_rule_refuses_terminal_mutation_and_replay() {
        let owned = Owned::new();
        let shape = owned.shape();
        let s = epochs(&owned);
        let l = limits();
        // A root terminal at epoch e and still present at e + 1.
        let (e, lane) = (0..s.len() - 1)
            .flat_map(|e| (0..ROOTS.len()).map(move |lane| (e, lane)))
            .find(|&(e, lane)| s[e][lane].counters()[1] == 1)
            .expect("an early terminal root");
        let (old, new) = (&s[e][lane], &s[e + 1][lane]);
        validate_device_epoch(&shape, l, &old.words(), &new.words(), QUANTUM).unwrap();
        type Mutation = (&'static str, fn(&mut Lane));
        let mutations: [Mutation; 7] = [
            ("output", |x| x.output[1] ^= 1),
            ("trace", |x| x.trace[0] ^= 1),
            ("heap", |x| x.heap[0] ^= 1),
            ("cleanup", |x| {
                let n = x.cleanup.len();
                x.cleanup[n - 1] ^= 1
            }),
            ("task frame", |x| x.task[0] ^= 1),
            ("task counter", |x| {
                let n = x.task.len();
                x.task[n - 144 + 4 * 8] ^= 1
            }),
            ("input", |x| x.input[0] ^= 1),
        ];
        for (name, mutate) in mutations {
            let mut bad = new.clone();
            mutate(&mut bad);
            assert_eq!(
                validate_device_epoch(&shape, l, &old.words(), &bad.words(), QUANTUM),
                Err(TaskError::Malformed),
                "terminal {name} mutation"
            );
        }
        // Replay: a terminal root reported with an earlier pending state.
        let earlier = &s[1][lane];
        assert_eq!(earlier.counters()[1], 0);
        assert_eq!(
            validate_device_epoch(&shape, l, &old.words(), &earlier.words(), QUANTUM),
            Err(TaskError::Malformed)
        );
        // `DeviceEpoch` itself refuses an already terminal predecessor.
        assert_eq!(
            validate(
                &shape,
                l,
                &old.words(),
                &new.words(),
                QUANTUM,
                Advance::DeviceEpoch
            ),
            Err(TaskError::Malformed)
        );
    }

    #[test]
    fn native_call_rule_is_unchanged_and_distinct() {
        let owned = Owned::new();
        let shape = owned.shape();
        let s = epochs(&owned);
        let l = limits();
        let lane = 1;
        let e = pending_pair(&s, lane);
        let (old, new) = (&s[e][lane].words(), &s[e + 1][lane].words());
        let native = |previous, current| Advance::NativeCall { previous, current };
        validate(&shape, l, old, new, QUANTUM, native(4, 5)).unwrap();
        assert_eq!(
            validate(&shape, l, old, new, QUANTUM, native(4, 4)),
            Err(TaskError::Malformed)
        );
        assert_eq!(
            validate(&shape, l, old, new, QUANTUM, native(4, 6)),
            Err(TaskError::Malformed)
        );
        validate(&shape, l, old, new, QUANTUM, Advance::DeviceEpoch).unwrap();
    }
}
