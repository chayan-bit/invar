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
