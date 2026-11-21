//! Reduction, pair-reduction and scan steps of the CPU scheduler.

use super::*;

/// Live `Array<u32>` elements for each owner, when every owner covers `len`.
fn borrow_live<'a>(
    live: &LiveLoop<'a>,
    names: &[String],
    len: usize,
) -> Option<Vec<std::cell::Ref<'a, Vec<u32>>>> {
    if len > laws::MAX_ELEMENTS
        || !names
            .iter()
            .all(|name| live.lengths.get(name).is_some_and(|n| *n >= len))
    {
        return None;
    }
    names.iter().map(|name| live.array(name)).collect()
}

fn slot(names: &[String], index: u32) -> Option<&String> {
    names.get(usize::try_from(index).ok()?)
}

pub(super) fn reduction_step(
    candidate: &laws::Reduction,
    live: &LiveLoop<'_>,
    workers: usize,
    hooks: &Hooks,
) -> Result<Option<Step>, Error> {
    let Plan::Reduce {
        law, seed, input, ..
    } = candidate.plan
    else {
        return Err(Error::Candidate);
    };
    let (arrays, scalars) = (
        std::slice::from_ref(&candidate.input),
        std::slice::from_ref(&candidate.state),
    );
    let (Some(input), Some(state)) = (slot(arrays, input), slot(scalars, seed)) else {
        return Err(Error::Candidate);
    };
    let len = live.iterations;
    let Some(seed) = live.scalars.get(state).copied() else {
        return Ok(None);
    };
    let Some(borrowed) = borrow_live(live, std::slice::from_ref(input), len) else {
        return Ok(None);
    };
    let values = &borrowed[0][..len];
    // Both laws are associative: seed then the ordered chunk totals equals the serial fold.
    let (total, chunks, joined) = reduction_output(law, values, workers, hooks)?;
    Ok(Some(Step {
        plan: LoopOverride {
            iterations: len,
            input_lengths: vec![(input.clone(), borrowed[0].len())],
            scalar: Some(vec![(state.clone(), seed, law.combine(seed, total))]),
            ..Default::default()
        },
        kind: PlanKind::Reduction,
        chunks,
        joined,
    }))
}

pub(super) fn pair_reduction_step(
    candidate: &crate::pair_reduction::Reduction,
    live: &LiveLoop<'_>,
    workers: usize,
    hooks: &Hooks,
) -> Result<Option<Step>, Error> {
    let len = live.iterations;
    let Some(borrowed) = borrow_live(live, &candidate.inputs, len) else {
        return Ok(None);
    };
    let seeds = candidate
        .states
        .iter()
        .map(|name| live.scalars.get(name).copied())
        .collect::<Option<Vec<_>>>();
    let Some(seeds) = seeds else {
        return Ok(None);
    };
    let (total, chunks, joined) =
        pair_reduction_output(&borrowed[0][..len], &borrowed[1][..len], workers, hooks)?;
    let final_pair = laws::Affine {
        a: seeds[0],
        b: seeds[1],
    }
    .then(total);
    Ok(Some(Step {
        plan: LoopOverride {
            iterations: len,
            input_lengths: candidate
                .inputs
                .iter()
                .zip(&borrowed)
                .map(|(name, a)| (name.clone(), a.len()))
                .collect(),
            scalar: Some(vec![
                (candidate.states[0].clone(), seeds[0], final_pair.a),
                (candidate.states[1].clone(), seeds[1], final_pair.b),
            ]),
            ..Default::default()
        },
        kind: PlanKind::Reduction,
        chunks,
        joined,
    }))
}

pub(super) fn scan_step(
    candidate: &crate::recurrence::Candidate,
    live: &LiveLoop<'_>,
    workers: usize,
    hooks: &Hooks,
) -> Result<Option<Step>, Error> {
    let Plan::AffineScan { prefix, seed, .. } = &candidate.plan else {
        return Err(Error::Candidate);
    };
    let Some(state) = slot(std::slice::from_ref(&candidate.state), *seed) else {
        return Err(Error::Candidate);
    };
    let len = live.iterations;
    let target = match (prefix, &candidate.output) {
        (Prefix::None, _) => None,
        (_, Some(name)) if live.lengths.get(name).is_some_and(|n| *n >= len) => Some(name),
        _ => return Ok(None),
    };
    let Some(seed) = live.scalars.get(state).copied() else {
        return Ok(None);
    };
    if target.is_some_and(|out| {
        candidate
            .arrays
            .iter()
            .any(|name| live.same_object(out, name))
    }) {
        return Ok(None);
    }
    let Some(borrowed) = borrow_live(live, &candidate.arrays, len) else {
        return Ok(None);
    };
    let slices = borrowed
        .iter()
        .map(|values| values.as_slice())
        .collect::<Vec<_>>();
    let coefficients = candidate
        .plan
        .affine_coefficients(&slices, len)
        .map_err(|_| Error::Candidate)?;
    let input_lengths = candidate
        .arrays
        .iter()
        .zip(&slices)
        .map(|(name, values)| (name.clone(), values.len()))
        .collect();
    drop(slices);
    drop(borrowed);
    let (states, chunks, joined) = scan_output(&coefficients, seed, workers, hooks)?;
    let final_state = states.last().copied().unwrap_or(seed);
    let published = if *prefix == Prefix::Exclusive {
        std::iter::once(seed)
            .chain(states.iter().copied())
            .take(len)
            .collect()
    } else {
        states
    };
    Ok(Some(Step {
        plan: LoopOverride {
            iterations: len,
            input_lengths,
            scalar: Some(vec![(state.clone(), seed, final_state)]),
            array: target.map(|name| (name.clone(), published)),
            ..Default::default()
        },
        kind: PlanKind::Scan,
        chunks,
        joined,
    }))
}

/// #204: a checked `i32`/`u32` sum on CPU workers. Each chunk computes its
/// carry; the ordered combine finds the first chunk whose prefix leaves the
/// range, and only that chunk reruns serially to find the trap index. The
/// result is published as a completed region, so MIR re-derives the fuel,
/// counts and trap site before anything is written.
pub(super) fn checked_fold_step(
    fold: &crate::checked_fold::Fold,
    live: &LiveLoop<'_>,
    workers: usize,
    hooks: &Hooks,
) -> Result<Option<Step>, Error> {
    use crate::{checked_fold::Int, mir_eval::region::Scalar};
    let len = live.iterations;
    // The 64-bit carry is exact for at most 2^28 elements of 2^32 magnitude.
    if len == 0
        || len > crate::array_limits::MAX_LIVE_ELEMENTS
        || live.lengths.get(&fold.input).is_none_or(|n| *n < len)
    {
        return Ok(None);
    }
    let folded = match (fold.ty, live.scalar(&fold.state)) {
        (Int::I32, Some(Scalar::I32(seed))) => match live.array_i32(&fold.input) {
            Some(values) => carry_fold(fold, seed.into(), &values[..len], workers, hooks)?,
            None => return Ok(None),
        },
        (Int::U32, Some(Scalar::U32(seed))) => match live.array(&fold.input) {
            Some(values) => carry_fold(fold, seed.into(), &values[..len], workers, hooks)?,
            None => return Ok(None),
        },
        _ => return Ok(None),
    };
    let Folded {
        seed,
        after,
        trap,
        chunks,
        joined,
    } = folded;
    let scalar = |value: i64| match fold.ty {
        Int::I32 => i32::try_from(value).ok().map(Scalar::I32),
        Int::U32 => u32::try_from(value).ok().map(Scalar::U32),
    };
    let taken = trap.map_or(len, |k| k + 1) as u64;
    // Out of range only when the injected mutation missed a trap: report an
    // unsound candidate instead of wrapping.
    let (Some(before), Some(after)) = (scalar(seed), scalar(after)) else {
        return Err(Error::Candidate);
    };
    let region = crate::mir_eval::region::CompletedRegion {
        function: live.function.clone(),
        span: fold.source_span,
        branches: vec![crate::mir_eval::region::Branch {
            span: fold.source_span,
            taken,
            skipped: u64::from(trap.is_none()),
        }],
        writes: Vec::new(),
        scalars: vec![crate::mir_eval::region::ScalarState {
            name: fold.state.clone(),
            before,
            after,
        }],
        cleanup: Vec::new(),
        fuel: fold.fuel(taken, trap.is_some()).ok_or(Error::Candidate)?,
        trap: trap.map(|_| crate::mir_eval::region::FirstTrap {
            category: crate::eval::TrapCategory::IntegerOverflow,
            span: fold.operator,
        }),
    };
    Ok(Some(Step {
        plan: LoopOverride::completed(region),
        kind: PlanKind::Reduction,
        chunks,
        joined,
    }))
}

/// A carried fold: the seed, the accumulator after the last completed
/// element, and the index of the trapping element, if any.
struct Folded {
    seed: i64,
    after: i64,
    trap: Option<usize>,
    chunks: Vec<std::ops::Range<usize>>,
    joined: usize,
}

fn carry_fold<T: Copy + Into<i64> + Sync>(
    fold: &crate::checked_fold::Fold,
    seed: i64,
    values: &[T],
    workers: usize,
    hooks: &Hooks,
) -> Result<Folded, Error> {
    use crate::checked_fold::{Carry, first_trap};
    let terms = |range: std::ops::Range<usize>| values[range].iter().map(|x| (*x).into());
    let (carries, chunks, joined) = parallel_chunks_with_limit(
        values.len(),
        workers,
        hooks,
        crate::array_limits::MAX_LIVE_ELEMENTS,
        |range| Ok(Carry::of(terms(range))),
    )?;
    let mut prefix = Carry::IDENTITY;
    for (carry, range) in carries.iter().zip(&chunks) {
        let mut next = prefix.then(*carry);
        if hooks.fault == Fault::LoseMinPrefix {
            next.low = prefix.low;
        }
        if next.leaves(seed, fold.ty) {
            let (k, before) = first_trap(seed + prefix.total, terms(range.clone()), fold.ty)
                .ok_or(Error::Candidate)?;
            return Ok(Folded {
                seed,
                after: before,
                trap: Some(range.start + k),
                chunks,
                joined,
            });
        }
        prefix = next;
    }
    Ok(Folded {
        seed,
        after: seed + prefix.total,
        trap: None,
        chunks,
        joined,
    })
}

/// #207: an L7 select loop on CPU workers. Each chunk keeps its local
/// records; the ordered combine yields the last update and how often the `if`
/// was taken. Published as a completed region, so MIR re-derives the fuel and
/// branch counts before anything is written. The body cannot trap once the
/// live input covers the trip count, which is checked here first.
pub(super) fn select_fold_step(
    select: &crate::select_fold::Select,
    live: &LiveLoop<'_>,
    workers: usize,
    hooks: &Hooks,
) -> Result<Option<Step>, Error> {
    use crate::{mir_eval::region::Scalar, select_fold::Elem};
    let len = live.iterations;
    if len == 0
        || len > crate::array_limits::MAX_LIVE_ELEMENTS
        || live.lengths.get(&select.input).is_none_or(|n| *n < len)
    {
        return Ok(None);
    }
    let (outcome, chunks, joined) = match (select.elem, live.scalar(&select.best)) {
        (Elem::I32, Some(Scalar::I32(seed))) => match live.array_i32(&select.input) {
            Some(values) => lift(
                select_records(select, seed, &values[..len], workers, hooks)?,
                Scalar::I32,
            ),
            None => return Ok(None),
        },
        (Elem::U32, Some(Scalar::U32(seed))) => match live.array(&select.input) {
            Some(values) => lift(
                select_records(select, seed, &values[..len], workers, hooks)?,
                Scalar::U32,
            ),
            None => return Ok(None),
        },
        (Elem::F32, Some(Scalar::F32(seed))) => match live.array_f32(&select.input) {
            Some(values) => lift(
                select_records(select, seed, &values[..len], workers, hooks)?,
                Scalar::F32,
            ),
            None => return Ok(None),
        },
        _ => return Ok(None),
    };
    let (taken, last) = outcome;
    let mut scalars = Vec::with_capacity(1 + select.companions.len());
    let Some(before) = live.scalar(&select.best) else {
        return Ok(None);
    };
    scalars.push(crate::mir_eval::region::ScalarState {
        name: select.best.clone(),
        before,
        after: last.map_or(before, |(_, value)| value),
    });
    for companion in &select.companions {
        let Some(before) = live.scalar(companion) else {
            return Ok(None);
        };
        let after = match last {
            Some((index, _)) => Scalar::U32(u32::try_from(index).map_err(|_| Error::Candidate)?),
            None => before,
        };
        scalars.push(crate::mir_eval::region::ScalarState {
            name: companion.clone(),
            before,
            after,
        });
    }
    let n = len as u64;
    let region = crate::mir_eval::region::CompletedRegion {
        function: live.function.clone(),
        span: select.source_span,
        branches: vec![
            crate::mir_eval::region::Branch {
                span: select.source_span,
                taken: n,
                skipped: 1,
            },
            crate::mir_eval::region::Branch {
                span: select.test_span,
                taken,
                skipped: n - taken,
            },
        ],
        writes: Vec::new(),
        scalars,
        cleanup: Vec::new(),
        fuel: select.fuel(n, taken).ok_or(Error::Candidate)?,
        trap: None,
    };
    Ok(Some(Step {
        plan: LoopOverride::completed(region),
        kind: PlanKind::Reduction,
        chunks,
        joined,
    }))
}

/// The if-taken count and last update, the chunks and the joined worker count.
type Selected<T> = (
    (u64, Option<(usize, T)>),
    Vec<std::ops::Range<usize>>,
    usize,
);

fn lift<T>(
    (outcome, chunks, joined): Selected<T>,
    scalar: impl Fn(T) -> crate::mir_eval::region::Scalar,
) -> Selected<crate::mir_eval::region::Scalar> {
    let (taken, last) = outcome;
    ((taken, last.map(|(k, v)| (k, scalar(v)))), chunks, joined)
}

fn select_records<T: crate::select_fold::Key>(
    select: &crate::select_fold::Select,
    seed: T,
    values: &[T],
    workers: usize,
    hooks: &Hooks,
) -> Result<Selected<T>, Error> {
    use crate::select_fold::{combine, records};
    let (order, strict) = (select.order, select.strict);
    let (chunks_records, chunks, joined) = parallel_chunks_with_limit(
        values.len(),
        workers,
        hooks,
        crate::array_limits::MAX_LIVE_ELEMENTS,
        |range| Ok(records(order, strict, &values[range.clone()], range.start)),
    )?;
    let outcome = combine(
        order,
        strict,
        seed,
        &chunks_records,
        hooks.fault == Fault::RightmostTie,
    );
    Ok(((outcome.taken, outcome.last), chunks, joined))
}
