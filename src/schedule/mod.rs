//! Bounded scoped CPU work, with staged publication and MIR serial fallback.

use std::{collections::VecDeque, ops::Range};

use crate::{
    admission::{self, Region},
    eval::{EvalError, Observation},
    front::Span,
    laws,
    loops::{Capture, Fused, FusedChain, LiveMap},
    mir,
    mir_eval::{self, LiveLoop, LoopOverride},
    plan::Execution,
    typed_net::{Plan, Prefix},
};

#[cfg(feature = "cuda")]
pub mod cuda;
mod families;
mod lowering;
#[cfg(all(feature = "metal", target_os = "macos"))]
pub mod metal;
pub mod normal_form;
pub mod profiled;
pub mod reductions;
pub mod tuning;
#[cfg(any(feature = "cuda", all(feature = "metal", target_os = "macos")))]
pub mod tuning_resident;
pub(crate) mod workers;

pub use crate::plan::PlanKind;
pub use lowering::{
    Dispatched, Gap, Selection, has_family_region, regions, selection, serial_only,
};
pub use workers::{Error, Fault, Hooks, LiveArrays, MAX_WORKERS, Snapshot, auto_workers};
use workers::{pair_reduction_output, reduction_output, scan_output};
pub(crate) use workers::{parallel_chunks, parallel_chunks_with_limit};

/// One region whose staged result was published by the scheduler.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Executed {
    pub function: String,
    pub span: Span,
    pub kind: PlanKind,
    /// The array owner whose elements this step wrote, if any.
    pub published_array: Option<String>,
    /// Actual implementation of an explicit builtin; absent for loop plans.
    pub implementation: Option<&'static str>,
}

#[derive(Debug)]
pub struct Run {
    pub observation: Observation,
    pub plan: PlanKind,
    pub workers_used: usize,
    pub chunks: Vec<Range<usize>>,
    pub joined: usize,
    pub fallback_reason: Option<&'static str>,
    /// Every published region, in execution order.
    pub executed: Vec<Executed>,
    /// Every reached float reduction, in execution order.
    pub reductions: Vec<reductions::Reduction>,
}

/// A staged fused output, valid only while the consumer's input still equals
/// the intermediate the producer published.
struct Pending {
    function: String,
    consumer: Span,
    stage: usize,
    /// Consumer operands and the element prefix each must still hold.
    operands: Vec<(String, Vec<u32>)>,
    output: Vec<u32>,
    /// The producer skipped writing its unobservable intermediate, so the consumer
    /// must publish this output; any refusal is an internal invariant failure.
    required: bool,
}

struct Step {
    plan: LoopOverride,
    kind: PlanKind,
    chunks: Vec<Range<usize>>,
    joined: usize,
}

struct Pass {
    captured: Vec<u32>,
    values: Vec<u32>,
    input_lengths: Vec<(String, usize)>,
    chunks: Vec<Range<usize>>,
    joined: usize,
}

/// Evaluate a live plan over the reached loop's current arrays. `None` when a
/// live length fact fails, so the loop runs serially before any worker starts.
fn live_pass(
    plan: &Plan,
    inputs: &[String],
    output: &str,
    live: &LiveLoop<'_>,
    capture: Option<Capture>,
    workers: usize,
    hooks: &Hooks,
) -> Result<Option<Pass>, Error> {
    let len = live.iterations;
    let covers = |name: &str| live.lengths.get(name).is_some_and(|n| *n >= len);
    if len > laws::MAX_ELEMENTS || !covers(output) || !inputs.iter().all(|name| covers(name)) {
        return Ok(None);
    }
    if inputs.iter().any(|name| live.same_object(output, name)) {
        return Ok(None);
    }
    let Some(borrowed) = inputs
        .iter()
        .map(|name| live.array(name))
        .collect::<Option<Vec<_>>>()
    else {
        return Ok(None);
    };
    let slices = borrowed
        .iter()
        .map(|values| values.as_slice())
        .collect::<Vec<_>>();
    let input_lengths = inputs
        .iter()
        .zip(&slices)
        .map(|(name, values)| (name.clone(), values.len()))
        .collect();
    let (parts, chunks, joined) = parallel_chunks(len, workers, hooks, |range| {
        plan.live_values(&slices, range, capture)
            .map_err(|_| Error::Candidate)
    })?;
    let mut captured = Vec::new();
    let mut values = Vec::new();
    captured
        .try_reserve_exact(if capture.is_some() { len } else { 0 })
        .and_then(|()| values.try_reserve_exact(len))
        .map_err(|_| Error::Allocation)?;
    for (part_captured, part_values) in parts {
        captured.extend(part_captured);
        values.extend(part_values);
    }
    Ok(Some(Pass {
        captured,
        values,
        input_lengths,
        chunks,
        joined,
    }))
}

fn map_step(
    map: &LiveMap,
    live: &LiveLoop<'_>,
    workers: usize,
    hooks: &Hooks,
) -> Result<Option<Step>, Error> {
    let Some(pass) = live_pass(
        &map.plan,
        &map.inputs,
        &map.output,
        live,
        None,
        workers,
        hooks,
    )?
    else {
        return Ok(None);
    };
    Ok(Some(Step {
        plan: LoopOverride {
            histogram: None,
            scatter: None,
            gather: None,
            summary: None,
            region: None,
            tensor: None,
            transpose: None,
            stencil: None,
            iterations: live.iterations,
            input_lengths: pass.input_lengths,
            scalar: None,
            array: Some((map.output.clone(), pass.values)),
        },
        kind: PlanKind::Map,
        chunks: pass.chunks,
        joined: pass.joined,
    }))
}

fn consumer_step(
    standalone: &LiveMap,
    live: &LiveLoop<'_>,
    pending: Option<Pending>,
    workers: usize,
    hooks: &Hooks,
) -> Result<Option<Step>, Error> {
    let required = pending.as_ref().is_some_and(|pending| pending.required);
    if let Some(step) = staged_consumer(standalone, live, pending) {
        return Ok(Some(step));
    }
    if required {
        return Err(Error::Candidate);
    }
    map_step(standalone, live, workers, hooks)
}

fn staged_consumer(
    standalone: &LiveMap,
    live: &LiveLoop<'_>,
    pending: Option<Pending>,
) -> Option<Step> {
    let len = live.iterations;
    if let Some(pending) = pending
        && pending.function == live.function
        && pending.consumer == live.span
        && pending.output.len() == len
        && live
            .lengths
            .get(&standalone.output)
            .is_some_and(|n| *n >= len)
        && !standalone
            .inputs
            .iter()
            .any(|name| live.same_object(&standalone.output, name))
        && pending.operands.iter().all(|(name, expected)| {
            live.array(name)
                .is_some_and(|current| current.get(..len) == Some(expected.as_slice()))
        })
    {
        let input_lengths = standalone
            .inputs
            .iter()
            .filter_map(|name| Some((name.clone(), *live.lengths.get(name)?)))
            .collect();
        return Some(Step {
            plan: LoopOverride {
                histogram: None,
                scatter: None,
                gather: None,
                summary: None,
                region: None,
                tensor: None,
                transpose: None,
                stencil: None,
                iterations: len,
                input_lengths,
                scalar: None,
                array: Some((standalone.output.clone(), pending.output)),
            },
            kind: PlanKind::FusedMap,
            chunks: Vec::new(),
            joined: 0,
        });
    }
    None
}

/// One pass computes the producer's output (published now) and the consumer's
/// output (staged until the consumer's operands are shown unchanged).
fn producer_step(
    fused: &Fused,
    live: &LiveLoop<'_>,
    pending: &mut VecDeque<Pending>,
    workers: usize,
    hooks: &Hooks,
) -> Result<Option<Step>, Error> {
    pending.clear();
    let len = live.iterations;
    let other = match &fused.consumer_other {
        Some(name) => match live.array(name) {
            Some(values) if values.len() >= len => Some((name.clone(), values[..len].to_vec())),
            _ => return Ok(None),
        },
        None => None,
    };
    let elide = fused.elide_intermediate;
    let capture = (!elide).then_some(fused.capture);
    let Some(pass) = live_pass(
        &fused.plan,
        &fused.inputs,
        &fused.intermediate,
        live,
        capture,
        workers,
        hooks,
    )?
    else {
        return Ok(None);
    };
    let mut operands = Vec::new();
    if !elide {
        operands.push((fused.consumer_input.clone(), pass.captured.clone()));
    }
    operands.extend(other);
    pending.push_back(Pending {
        function: live.function.clone(),
        consumer: fused.source_spans[1],
        stage: 1,
        operands,
        output: pass.values,
        required: elide,
    });
    Ok(Some(Step {
        plan: LoopOverride {
            histogram: None,
            scatter: None,
            gather: None,
            summary: None,
            region: None,
            tensor: None,
            transpose: None,
            stencil: None,
            iterations: len,
            input_lengths: pass.input_lengths,
            scalar: None,
            array: (!elide).then(|| (fused.intermediate.clone(), pass.captured)),
        },
        kind: PlanKind::FusedMap,
        chunks: pass.chunks,
        joined: pass.joined,
    }))
}

fn dag_producer_step(
    chain: &FusedChain,
    live: &LiveLoop<'_>,
    pending: &mut VecDeque<Pending>,
    workers: usize,
    hooks: &Hooks,
) -> Result<Option<Step>, Error> {
    pending.clear();
    let len = live.iterations;
    let Some(first_output) = chain.outputs.first() else {
        return Err(Error::Candidate);
    };
    if len > laws::MAX_ELEMENTS
        || live.lengths.get(first_output).is_none_or(|n| *n < len)
        || chain
            .inputs
            .iter()
            .any(|name| live.lengths.get(name).is_none_or(|n| *n < len))
        || chain
            .inputs
            .iter()
            .any(|name| live.same_object(first_output, name))
    {
        return Ok(None);
    }
    let Some(borrowed) = chain
        .inputs
        .iter()
        .map(|name| live.array(name))
        .collect::<Option<Vec<_>>>()
    else {
        return Ok(None);
    };
    let slices = borrowed
        .iter()
        .map(|array| array.as_slice())
        .collect::<Vec<_>>();
    let input_lengths = chain
        .inputs
        .iter()
        .zip(&slices)
        .map(|(name, values)| (name.clone(), values.len()))
        .collect::<Vec<_>>();
    let (parts, chunks, joined) = parallel_chunks(len, workers, hooks, |range| {
        chain
            .plan
            .run_dag(&slices, range)
            .map_err(|_| Error::Candidate)
    })?;
    let mut values = Vec::new();
    values
        .try_reserve_exact(chain.outputs.len())
        .map_err(|_| Error::Allocation)?;
    for _ in &chain.outputs {
        let mut stage = Vec::new();
        stage
            .try_reserve_exact(len)
            .map_err(|_| Error::Allocation)?;
        values.push(stage);
    }
    for part in parts {
        if part.len() != values.len() {
            return Err(Error::Candidate);
        }
        for (whole, chunk) in values.iter_mut().zip(part) {
            whole.extend(chunk);
        }
    }
    for (stage, link) in chain.links.iter().enumerate() {
        let operands = chain
            .link_operands(stage, &values, &slices, len)
            .ok_or(Error::Candidate)?;
        pending.push_back(Pending {
            function: live.function.clone(),
            consumer: link.span,
            stage: stage + 1,
            operands,
            output: values[stage + 1].clone(),
            required: false,
        });
    }
    Ok(Some(Step {
        plan: LoopOverride {
            histogram: None,
            scatter: None,
            gather: None,
            summary: None,
            region: None,
            tensor: None,
            transpose: None,
            stencil: None,
            iterations: len,
            input_lengths,
            scalar: None,
            array: Some((first_output.clone(), values.remove(0))),
        },
        kind: PlanKind::FusedMap,
        chunks,
        joined,
    }))
}

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

fn reduction_step(
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
            histogram: None,
            scatter: None,
            gather: None,
            summary: None,
            region: None,
            tensor: None,
            transpose: None,
            stencil: None,
            iterations: len,
            input_lengths: vec![(input.clone(), borrowed[0].len())],
            scalar: Some(vec![(state.clone(), seed, law.combine(seed, total))]),
            array: None,
        },
        kind: PlanKind::Reduction,
        chunks,
        joined,
    }))
}

fn pair_reduction_step(
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
            region: None,
            histogram: None,
            scatter: None,
            gather: None,
            summary: None,
            tensor: None,
            transpose: None,
            stencil: None,
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
            array: None,
        },
        kind: PlanKind::Reduction,
        chunks,
        joined,
    }))
}

fn scan_step(
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
            histogram: None,
            scatter: None,
            gather: None,
            summary: None,
            region: None,
            tensor: None,
            transpose: None,
            stencil: None,
            iterations: len,
            input_lengths,
            scalar: Some(vec![(state.clone(), seed, final_state)]),
            array: target.map(|name| (name.clone(), published)),
        },
        kind: PlanKind::Scan,
        chunks,
        joined,
    }))
}

fn region_step(
    region: &Region,
    live: &LiveLoop<'_>,
    pending: &mut VecDeque<Pending>,
    workers: usize,
    hooks: &Hooks,
) -> Result<Option<Step>, Error> {
    match region.execution() {
        Execution::Serial => Ok(None),
        Execution::Map(map) => map_step(map, live, workers, hooks),
        Execution::FusedProducer(fused) => producer_step(fused, live, pending, workers, hooks),
        Execution::FusedConsumer { standalone, .. } => {
            consumer_step(standalone, live, pending.pop_front(), workers, hooks)
        }
        Execution::DagProducer(chain) => dag_producer_step(chain, live, pending, workers, hooks),
        Execution::DagConsumer {
            stage, standalone, ..
        } => {
            let staged = pending.pop_front().filter(|item| item.stage == *stage);
            let step = consumer_step(standalone, live, staged, workers, hooks);
            if !matches!(
                &step,
                Ok(Some(Step {
                    kind: PlanKind::FusedMap,
                    ..
                }))
            ) {
                pending.clear();
            }
            step
        }
        Execution::Reduction(candidate) => reduction_step(candidate, live, workers, hooks),
        Execution::PairReduction(candidate) => pair_reduction_step(candidate, live, workers, hooks),
        Execution::Scan(candidate) => scan_step(candidate, live, workers, hooks),
    }
}

/// The offered steps MIR published, in offer order. A refused offer anywhere
/// in the run falls back to its serial loop and is dropped here, so later
/// publications keep their own labels.
fn keep_published<T>(offered: Vec<T>, published: &[bool]) -> Vec<T> {
    offered
        .into_iter()
        .zip(published)
        .filter_map(|(step, published)| published.then_some(step))
        .collect()
}

/// Whether any function calls a deterministic float reduction. The whole-program
/// f32 CPU path has no reduction hook, so such programs keep the MIR path, which
/// records a receipt for every reached call.
fn has_reduction(program: &mir::Program) -> bool {
    program.functions.iter().any(|function| {
        function.blocks.iter().any(|block| {
            block.instructions.iter().any(|instruction| {
                matches!(&instruction.kind, mir::Operation::Call { name, .. }
                    if crate::fsum::Kind::from_name(name).is_some())
            })
        })
    })
}

pub fn run_source(
    source: &[u8],
    program: &mir::Program,
    fuel: u64,
    workers: usize,
) -> Result<Run, Error> {
    run_source_with_hooks(source, program, fuel, workers, &Hooks::default())
}

/// Run `program` serially in MIR, publishing each admitted region's staged CPU
/// result at the moment the region's loop is reached with true live guards.
pub fn run_source_with_hooks(
    source: &[u8],
    program: &mir::Program,
    fuel: u64,
    workers: usize,
    hooks: &Hooks,
) -> Result<Run, Error> {
    if !(1..=MAX_WORKERS).contains(&workers) {
        return Err(Error::InvalidWorkers);
    }
    if crate::spatial::contains(program) {
        return crate::spatial::run_cpu(source, program, fuel, workers, hooks);
    }
    if crate::sorting::contains(program) {
        if hooks.fault != Fault::None {
            return Err(Error::Candidate);
        }
        return crate::sorting::run_cpu(source, program, fuel, workers);
    }
    let regions = families::dispatch(source, program, hooks.forge)
        .map_err(|_| Error::Candidate)?
        .selected;
    let mut receipts = Vec::new();
    let mut reduce_fault = None;
    let mut reducer = |call: &mir_eval::ReduceCall<'_>| {
        let bits = reductions::cpu(call, workers).map_err(|error| {
            reduce_fault = Some(error.clone());
            EvalError::Tool {
                code: error.code(),
                message: format!("CPU reduction failure: {error:?}"),
            }
        })?;
        let mut receipt = reductions::cpu_receipt(call, workers, "forced-cpu-parallel");
        if receipt.backend == reductions::SERIAL_BACKEND {
            receipt.reason = "measured-cpu-cost";
        }
        receipts.push(receipt);
        Ok(Some(bits))
    };
    if regions.is_empty() {
        if hooks.fault == Fault::None
            && !has_reduction(program)
            && program.functions.iter().any(|f| {
                f.locals
                    .values()
                    .any(|l| matches!(l.ty, crate::front::Type::F32 | crate::front::Type::ArrayF32))
            })
        {
            let floating = crate::f32_source::run_cpu_source(source, fuel, workers)
                .map_err(|_| Error::Candidate)?;
            if floating.completed != 0 {
                return Ok(Run {
                    observation: floating.observation,
                    plan: floating.kind,
                    workers_used: floating.workers_used,
                    chunks: floating.chunks,
                    joined: floating.joined,
                    fallback_reason: None,
                    executed: floating.executed,
                    reductions: Vec::new(),
                });
            }
        }
        let (observation, _) = mir_eval::observe_engine(program, fuel, None, Some(&mut reducer));
        if let Some(error) = reduce_fault {
            return Err(error);
        }
        return Ok(Run {
            observation,
            plan: PlanKind::Serial,
            workers_used: 0,
            chunks: Vec::new(),
            joined: 0,
            fallback_reason: receipts.is_empty().then_some("NO_ADMITTED_CANDIDATE"),
            executed: Vec::new(),
            reductions: receipts,
        });
    }
    let mut executed = Vec::new();
    let mut ranges = Vec::new();
    let mut joined = 0;
    let mut fault = None;
    let mut pending = VecDeque::new();
    let mut provider = |live: &LiveLoop<'_>| -> Result<Option<LoopOverride>, EvalError> {
        let Some(region) = regions
            .iter()
            .find(|region| region.function() == live.function && region.span() == live.span)
        else {
            return Ok(None);
        };
        let before = (hooks.fault != Fault::None).then(|| live.snapshot());
        let step = region.cpu(live, &mut pending, workers, hooks);
        if let (Some(before), Err(_)) = (before, &step)
            && let Ok(mut slot) = hooks.live_arrays.lock()
        {
            *slot = Some(LiveArrays {
                before,
                after: live.snapshot(),
            });
        }
        match step {
            Ok(Some(step)) => {
                executed.push(Executed {
                    function: live.function.clone(),
                    span: live.span,
                    kind: step.kind,
                    published_array: step.plan.published_array(),
                    implementation: None,
                });
                if !step.chunks.is_empty() {
                    ranges = step.chunks;
                }
                joined += step.joined;
                Ok(Some(step.plan))
            }
            Ok(None) => Ok(None),
            Err(error) => {
                fault = Some(error.clone());
                Err(EvalError::Tool {
                    code: error.code(),
                    message: format!("CPU worker failure: {error:?}"),
                })
            }
        }
    };
    let (observation, published) =
        mir_eval::observe_engine(program, fuel, Some(&mut provider), Some(&mut reducer));
    if let Some(error) = fault.or(reduce_fault) {
        return Err(error);
    }
    let executed = keep_published(executed, &published);
    Ok(Run {
        observation,
        plan: executed.last().map_or(PlanKind::Serial, |step| step.kind),
        workers_used: joined,
        chunks: ranges,
        joined,
        fallback_reason: (executed.is_empty() && receipts.is_empty())
            .then_some("UNREACHED_OR_GUARD_FALSE"),
        executed,
        reductions: receipts,
    })
}

/// Median wall milliseconds of the real cpu-parallel element executor on a
/// one-operation map over `elements` words, used to calibrate `auto`.
#[cfg(any(feature = "cuda", all(feature = "metal", target_os = "macos")))]
pub(crate) fn cpu_map_ms(elements: usize, workers: usize) -> Option<f64> {
    let plan = Plan::LiveDag {
        inputs: vec![0],
        ops: vec![
            crate::typed_net::DagOp::Input(0),
            crate::typed_net::DagOp::Map {
                input: 0,
                op: crate::typed_net::MapOp::WrappingAdd(1),
            },
        ],
        outputs: vec![1],
        source_identity: 1,
        graph_revision: 0,
    };
    let input = vec![7u32; elements];
    let hooks = Hooks::default();
    let mut samples = Vec::new();
    for trial in 0..6 {
        let started = std::time::Instant::now();
        let parts = parallel_chunks(elements, workers, &hooks, |range| {
            plan.run_dag(&[&input], range).map_err(|_| Error::Candidate)
        })
        .ok()?;
        std::hint::black_box(&parts);
        if trial > 0 {
            samples.push(started.elapsed().as_secs_f64() * 1000.0);
        }
    }
    samples.sort_by(f64::total_cmp);
    samples.get(samples.len() / 2).copied()
}

#[cfg(test)]
mod tests;
