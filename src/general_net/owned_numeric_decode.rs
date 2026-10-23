//! Terminal decoder for the ten-word owned-numeric packet shared by the CPU
//! ABI11 host and the ABI12 CPU model. It validates every header word, trace
//! row, heap descriptor and cleanup record before publishing an observation.
use super::{BlockSite, Token, compiled, coverage, oracle};
use crate::{
    eval,
    front::{Span, Type},
    numeric_contract::F32Bits,
};
use invar_foreign::numeric_owned_resume as numeric;
use std::collections::BTreeMap;

#[derive(Debug, PartialEq, Eq)]
pub enum Error {
    Allocation,
    Malformed,
}
type TokenKey = (usize, usize, usize);
fn reserve<T>(n: usize) -> Result<Vec<T>, Error> {
    let mut v = Vec::new();
    v.try_reserve_exact(n).map_err(|_| Error::Allocation)?;
    Ok(v)
}
pub(super) fn decode(
    result: &Type,
    source_len: usize,
    sites: &BTreeMap<TokenKey, (coverage::Rule, Span)>,
    program: &crate::mir::Program,
    storage: &compiled::OwnedLayout,
    l: oracle::Limits,
    raw: &numeric::Invocation,
) -> Result<oracle::Observation, Error> {
    let h = raw.output;
    let [status, start, end, steps, transitions, frames, count] =
        [h[0], h[2], h[3], h[4], h[5], h[6], h[7]]
            .map(|n| u64::try_from(n).map_err(|_| Error::Malformed));
    let (status, start, end, steps, transitions, frames, count) =
        (status?, start?, end?, steps?, transitions?, frames?, count?);
    let used = usize::try_from(count)
        .ok()
        .and_then(|n| n.checked_mul(7))
        .ok_or(Error::Malformed)?;
    if status > 13
        || status == 8
        || frames == 0
        || frames > 64
        || steps > l.fuel
        || transitions > l.transitions
        || count > l.trace_events as u64
        || start > end
        || end > source_len as u64
        || raw.trace.len() != l.trace_events * 7
        || used > raw.trace.len()
        || raw.trace[used..].iter().any(|n| *n != 0)
    {
        return Err(Error::Malformed);
    }
    if (!matches!(status, 1 | 2 | 9) && (start != 0 || end != 0))
        || (matches!(status, 1 | 2 | 9) && start == end)
        || (status != 0 && h[1] != 0)
        || (status == 3 && steps != l.fuel)
        || (status == 4 && frames != 64)
        || (status == 5 && transitions != l.transitions)
        || (status == 6 && count != l.trace_events as u64)
    {
        return Err(Error::Malformed);
    }
    let mut events = reserve(count as usize)?;
    let mut pending: Vec<(u64, u64, TokenKey, coverage::Rule)> = reserve(64)?;
    for row in raw.trace[..used].chunks_exact(7) {
        let function = usize::try_from(row[3]).map_err(|_| Error::Malformed)?;
        let block = usize::try_from(row[4]).map_err(|_| Error::Malformed)?;
        let position = usize::try_from(row[5]).map_err(|_| Error::Malformed)?;
        let (rule, _) = sites
            .get(&(function, block, position))
            .ok_or(Error::Malformed)?;
        if row[0] > 1 || row[1] > transitions || row[2] > transitions || row[6] != *rule as u64 {
            return Err(Error::Malformed);
        }
        let key = (row[1], row[2], (function, block, position), *rule);
        if row[0] == 0 {
            if pending.len() == 64
                || pending
                    .last()
                    .is_some_and(|last| last.3 != coverage::Rule::Call || row[1] <= last.0)
            {
                return Err(Error::Malformed);
            }
            pending.push(key);
        } else if pending.pop() != Some(key) {
            return Err(Error::Malformed);
        }
        events.push(oracle::Event {
            kind: if row[0] == 0 {
                oracle::EventKind::Demand
            } else {
                oracle::EventKind::Complete
            },
            frame: row[1],
            activation: row[2],
            token: Token {
                block: BlockSite { function, block },
                position,
            },
            rule: *rule,
        });
    }
    if status == 0
        && (!pending.is_empty()
            || !events.last().is_some_and(|e| {
                e.kind == oracle::EventKind::Complete
                    && e.rule == coverage::Rule::Return
                    && e.frame == 0
            }))
    {
        return Err(Error::Malformed);
    }
    if matches!(status, 1 | 2 | 9) {
        let last = events.last().ok_or(Error::Malformed)?;
        let (_, span) = sites
            .get(&(
                last.token.block.function,
                last.token.block.block,
                last.token.position,
            ))
            .ok_or(Error::Malformed)?;
        if last.kind != oracle::EventKind::Demand
            || span.start as u64 != start
            || span.end as u64 != end
        {
            return Err(Error::Malformed);
        }
    }
    let failure = match status {
        5 => Some(oracle::ToolFailure::TransitionLimit),
        6 => Some(oracle::ToolFailure::TraceLimit),
        7 => Some(oracle::ToolFailure::InvalidState),
        11..=13 => Some(oracle::ToolFailure::Allocation),
        _ => None,
    };
    let (root, cleanup_trace) = decode_storage(result, program, storage, &events, status, raw)?;
    let outcome = match status {
        0 => Ok(if let Some(root) = root {
            root
        } else {
            match result {
                Type::F32 => eval::Value::F32(F32Bits::from_bits(
                    u32::try_from(h[1]).map_err(|_| Error::Malformed)?,
                )),
                Type::I32 => eval::Value::I32(i32::try_from(h[1]).map_err(|_| Error::Malformed)?),
                Type::U32 => eval::Value::U32(u32::try_from(h[1]).map_err(|_| Error::Malformed)?),
                Type::Bool if h[1] == 0 || h[1] == 1 => eval::Value::Bool(h[1] != 0),
                Type::Unit if h[1] == 0 => eval::Value::Unit,
                _ => return Err(Error::Malformed),
            }
        }),
        1 | 2 | 9 => Err(eval::EvalError::Trap {
            category: if status == 1 {
                eval::TrapCategory::IntegerOverflow
            } else if status == 2 {
                eval::TrapCategory::DivisionByZero
            } else {
                eval::TrapCategory::IndexOutOfBounds
            },
            span: Span {
                start: start as usize,
                end: end as usize,
            },
        }),
        3 => Err(eval::EvalError::EvaluationLimit {
            fuel_limit: l.fuel,
            steps_used: steps,
        }),
        4 => Err(eval::EvalError::Tool {
            code: "CALL_DEPTH_LIMIT",
            message: "compiled numeric call-depth limit".into(),
        }),
        5 | 6 => Err(eval::EvalError::Tool {
            code: "RESOURCE_LIMIT",
            message: "compiled numeric transition/trace limit".into(),
        }),
        7 => Err(eval::EvalError::Tool {
            code: "MIR_INVALID",
            message: "compiled numeric invalid state".into(),
        }),
        10 => Err(eval::EvalError::Tool {
            code: "ARRAY_SIZE_LIMIT",
            message: "compiled numeric source array-size limit".into(),
        }),
        11..=13 => Err(eval::EvalError::Tool {
            code: "RESOURCE_LIMIT",
            message: format!("compiled numeric storage limit {status}"),
        }),
        _ => return Err(Error::Malformed),
    };
    Ok(oracle::Observation {
        semantic: eval::Observation {
            result: outcome,
            steps_used: steps,
            cleanup_trace,
        },
        failure,
        events,
        transitions,
        max_frames: frames as usize,
    })
}

fn is_array(ty: &Type) -> bool {
    matches!(ty, Type::ArrayI32 | Type::ArrayU32 | Type::ArrayF32)
}
fn allocating(program: &crate::mir::Program, event: &oracle::Event) -> bool {
    let Some(i) = program.functions[event.token.block.function].blocks[event.token.block.block]
        .instructions
        .get(event.token.position)
    else {
        return false;
    };
    matches!(i.kind, crate::mir::Operation::Array(_))
        || matches!(&i.kind, crate::mir::Operation::Call { name, .. }
            if matches!(name.as_str(), "filled_i32" | "filled_u32" | "filled_f32"))
}
fn cleanup_names<'a>(program: &'a crate::mir::Program, event: &oracle::Event) -> &'a [String] {
    if event.kind != oracle::EventKind::Demand {
        return &[];
    }
    let block = &program.functions[event.token.block.function].blocks[event.token.block.block];
    match block
        .instructions
        .get(event.token.position)
        .map(|i| &i.kind)
    {
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
fn decode_storage(
    result: &Type,
    program: &crate::mir::Program,
    layout: &compiled::OwnedLayout,
    events: &[oracle::Event],
    status: u64,
    raw: &numeric::Invocation,
) -> Result<(Option<eval::Value>, Vec<eval::CleanupEvent>), Error> {
    use std::{cell::RefCell, rc::Rc};
    let allocated = usize::try_from(raw.output[9]).map_err(|_| Error::Malformed)?;
    let count = usize::try_from(raw.output[8]).map_err(|_| Error::Malformed)?;
    if allocated > layout.allocations()
        || count > layout.cleanup_events()
        || raw.heap.len() != layout.heap_words()
        || raw.cleanup.len() != layout.cleanup_words()
        || raw.cleanup[count * 4..].iter().any(|n| *n != 0)
        || raw.heap[allocated * 4..layout.allocations() * 4]
            .iter()
            .any(|n| *n != 0)
        || (status == 12 && allocated != layout.allocations())
        || (status == 13 && count != layout.cleanup_events())
    {
        return Err(Error::Malformed);
    }
    let mut allocations = events
        .iter()
        .filter(|e| e.kind == oracle::EventKind::Complete && allocating(program, e))
        .count();
    if status == 6
        && events
            .last()
            .is_some_and(|e| e.kind == oracle::EventKind::Demand && allocating(program, e))
    {
        allocations += 1;
    }
    if allocations != allocated {
        return Err(Error::Malformed);
    }
    let pool = &raw.heap[layout.allocations() * 4..];
    // Reclaimed pool words may remain as initialized stale values after stable-ID
    // compaction. They cannot publish a root; every word still has a 32-bit carrier.
    if pool
        .iter()
        .any(|w| *w < i64::from(i32::MIN) || *w > i64::from(u32::MAX))
    {
        return Err(Error::Malformed);
    }
    let mut live = reserve(allocated)?;
    let mut dead_lengths = BTreeMap::<usize, usize>::new();
    let mut ever_words = 0usize;
    for (index, d) in raw.heap[..allocated * 4].chunks_exact(4).enumerate() {
        let offset = usize::try_from(d[0]).map_err(|_| Error::Malformed)?;
        let len = usize::try_from(d[1]).map_err(|_| Error::Malformed)?;
        let end = offset.checked_add(len).ok_or(Error::Malformed)?;
        if len > crate::array_limits::MAX_ARRAY_LEN
            || end > pool.len()
            || !matches!(d[2], 0 | 1 | 258)
            || !matches!(d[3], 0 | 1)
        {
            return Err(Error::Malformed);
        }
        ever_words = ever_words.saturating_add(len).min(pool.len());
        if d[3] == 1 {
            if pool[offset..end].iter().any(|w| {
                if d[2] == 0 {
                    i32::try_from(*w).is_err()
                } else {
                    u32::try_from(*w).is_err()
                }
            }) {
                return Err(Error::Malformed);
            }
            live.push((offset, end, index, d[2]));
        } else {
            *dead_lengths.entry(len).or_default() += 1;
        }
    }
    // Compaction can leave stale words, but no write can reach beyond the sum
    // of every stable allocation length ever issued in this invocation.
    if pool[ever_words..].iter().any(|w| *w != 0) {
        return Err(Error::Malformed);
    }
    live.sort_unstable();
    if live.windows(2).any(|pair| pair[0].1 > pair[1].0) || allocated - live.len() != count {
        return Err(Error::Malformed);
    }
    let mut cleanup = reserve(count)?;
    let (mut event_at, mut name_at) = (0usize, 0usize);
    for row in raw.cleanup[..count * 4].chunks_exact(4) {
        let function = usize::try_from(row[0]).map_err(|_| Error::Malformed)?;
        let place = usize::try_from(row[1]).map_err(|_| Error::Malformed)?;
        let len = usize::try_from(row[2]).map_err(|_| Error::Malformed)?;
        let f = program.functions.get(function).ok_or(Error::Malformed)?;
        let name = f.declaration_order.get(place).ok_or(Error::Malformed)?;
        if !f.locals.get(name).is_some_and(|v| is_array(&v.ty)) {
            return Err(Error::Malformed);
        }
        let remaining = dead_lengths.get_mut(&len).ok_or(Error::Malformed)?;
        if *remaining == 0 {
            return Err(Error::Malformed);
        }
        *remaining -= 1;
        let mut found = false;
        while let Some(event) = events.get(event_at) {
            let names = cleanup_names(program, event);
            if event.frame == row[3] && event.token.block.function == function {
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
            return Err(Error::Malformed);
        }
        cleanup.push(eval::CleanupEvent {
            name: name.clone(),
            length: len,
        });
    }
    if dead_lengths.values().any(|n| *n != 0) {
        return Err(Error::Malformed);
    }
    if status != 0 {
        return Ok((None, cleanup));
    }
    if !is_array(result) {
        if !live.is_empty() {
            return Err(Error::Malformed);
        }
        return Ok((None, cleanup));
    }
    let handle = usize::try_from(raw.output[1])
        .ok()
        .and_then(|n| n.checked_sub(1))
        .ok_or(Error::Malformed)?;
    if live.len() != 1 || live[0].2 != handle {
        return Err(Error::Malformed);
    }
    let (start, end, _, kind) = live[0];
    let words = &pool[start..end];
    let value = match (result, kind) {
        (Type::ArrayI32, 0) => {
            let mut xs = reserve(words.len())?;
            for x in words {
                xs.push(i32::try_from(*x).map_err(|_| Error::Malformed)?);
            }
            eval::Value::ArrayI32(Rc::new(RefCell::new(xs)))
        }
        (Type::ArrayU32, 1) => {
            let mut xs = reserve(words.len())?;
            for x in words {
                xs.push(u32::try_from(*x).map_err(|_| Error::Malformed)?);
            }
            eval::Value::ArrayU32(Rc::new(RefCell::new(xs)))
        }
        (Type::ArrayF32, 258) => {
            let mut xs = reserve(words.len())?;
            for x in words {
                xs.push(F32Bits::from_bits(
                    u32::try_from(*x).map_err(|_| Error::Malformed)?,
                ));
            }
            eval::Value::ArrayF32(Rc::new(RefCell::new(xs)))
        }
        _ => return Err(Error::Malformed),
    };
    Ok((Some(value), cleanup))
}
