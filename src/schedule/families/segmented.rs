//! Reached-source segmented map/reduce and two-pass stable compaction.
#![cfg_attr(
    not(any(test, feature = "cuda", all(feature = "metal", target_os = "macos"))),
    allow(dead_code)
)]
use super::super::*;
use crate::mir_eval::region::{Branch, CompletedRegion, Elements, Scalar, ScalarState, Write};
use crate::{compaction_source, segmented_device, segmented_source};
use invar_device::{Capabilities, segmented as seg};

pub(super) enum Candidate {
    Map(Box<segmented_device::MapContract>),
    Reduce(Box<segmented_source::Contract>),
    Compact(Box<compaction_source::Contract>, usize),
}

/// A PNF segmented map, segmented reduction or compaction admission binds the
/// function's segmented source contract at its span; a contract of another
/// family is no lowering. The recognizer this replaces is the oracle in
/// `tests/support/family_oracle.rs`.
pub(super) fn lower(
    admitted: &lowering::Admitted,
    live: &lowering::Live<'_>,
) -> Result<Option<Candidate>, admission::Error> {
    use crate::general_net::regions::family::Family::{Compaction, SegmentedMap, SegmentedReduce};
    if !admitted.admits(&[SegmentedMap, SegmentedReduce, Compaction]) {
        return Ok(None);
    }
    let candidate = contract(live.source, live.program, live.region)?;
    Ok(candidate.filter(|c| {
        admitted.admits(&[match c {
            Candidate::Map(_) => SegmentedMap,
            Candidate::Reduce(_) => SegmentedReduce,
            Candidate::Compact(..) => Compaction,
        }])
    }))
}

fn contract(
    source: &[u8],
    program: &mir::Program,
    region: &Region,
) -> Result<Option<Candidate>, admission::Error> {
    let invalid = |message| admission::Error {
        code: "ADMISSION_SOURCE",
        message,
    };
    let mismatch = || invalid("segmented contract differs from source MIR".into());
    let map = match segmented_device::MapContract::from_source(source, region.function()) {
        Ok(c) => Some(c),
        Err(segmented_device::Error::Unrecognized) => None,
        Err(error) => return Err(invalid(format!("segmented map contract: {error:?}"))),
    };
    if let Some(c) = map {
        if c.mir_contract() != program.contract_digest() {
            return Err(mismatch());
        }
        return Ok((c.loop_span() == region.span()).then(|| Candidate::Map(Box::new(c))));
    }
    let reduce = match segmented_source::Contract::from_source(source, region.function()) {
        Ok(c) => Some(c),
        Err(segmented_source::Error::Unrecognized) => None,
        Err(error) => return Err(invalid(format!("segmented reduction contract: {error:?}"))),
    };
    if let Some(c) = reduce {
        c.verify_mir(program).map_err(|_| mismatch())?;
        return Ok((c.loop_span() == region.span()).then(|| Candidate::Reduce(Box::new(c))));
    }
    let compact = match compaction_source::Contract::from_source(source, region.function()) {
        Ok(c) => Some(c),
        Err(compaction_source::Error::Unrecognized) => None,
        Err(error) => return Err(invalid(format!("compaction contract: {error:?}"))),
    };
    if let Some(c) = compact {
        c.verify_mir(program).map_err(|_| mismatch())?;
        if let Some(pass) = c
            .loop_spans()
            .iter()
            .position(|span| *span == region.span())
        {
            return Ok(Some(Candidate::Compact(Box::new(c), pass)));
        }
    }
    Ok(None)
}

pub(super) fn cpu(
    candidate: &Candidate,
    live: &LiveLoop<'_>,
    pending: &mut VecDeque<Pending>,
    _workers: usize,
    _hooks: &Hooks,
) -> Result<Option<Step>, Error> {
    // Only a previously completed device copy is consumed here. All other CPU
    // execution stays with serial MIR, including the intervening allocation.
    if let Candidate::Compact(c, 1) = candidate {
        return Ok(copy_step(c, live, pending));
    }
    Ok(None)
}

#[cfg(any(feature = "cuda", all(feature = "metal", target_os = "macos")))]
pub(super) fn is_consumer(candidate: &Candidate) -> bool {
    matches!(candidate, Candidate::Compact(_, 1))
}
#[cfg(any(feature = "cuda", all(feature = "metal", target_os = "macos")))]
pub(super) fn is_producer(candidate: &Candidate) -> bool {
    matches!(candidate, Candidate::Compact(_, 0))
}

fn step(region: CompletedRegion, kind: PlanKind) -> Step {
    Step {
        plan: LoopOverride::completed(region),
        kind,
        chunks: Vec::new(),
        joined: 0,
    }
}
fn branch(span: Span, taken: usize, skipped: usize) -> Option<Branch> {
    Some(Branch {
        span,
        taken: u64::try_from(taken).ok()?,
        skipped: u64::try_from(skipped).ok()?,
    })
}
fn compact_region(
    c: &compaction_source::Contract,
    pass: usize,
    live: &LiveLoop<'_>,
    output: Vec<u32>,
) -> Option<Step> {
    let count = *live.scalars.get(&c.names()[1])?;
    let retained = u32::try_from(output.len()).ok()?;
    if live.iterations != count as usize || *live.scalars.get(&c.states()[pass])? != 0 {
        return None;
    }
    let writes = if pass == 1 {
        if live.lengths.get(&c.names()[3]) != Some(&output.len()) {
            return None;
        }
        vec![Write {
            owner: c.names()[3].clone(),
            length: output.len(),
            range: 0..output.len(),
            values: Elements::U32(output),
        }]
    } else {
        Vec::new()
    };
    Some(step(
        CompletedRegion {
            function: live.function.clone(),
            span: c.loop_spans()[pass],
            branches: vec![
                branch(c.loop_spans()[pass], count as usize, 1)?,
                branch(
                    c.selection_spans()[pass],
                    retained as usize,
                    count.checked_sub(retained)? as usize,
                )?,
            ],
            writes,
            scalars: vec![ScalarState {
                name: c.states()[pass].clone(),
                before: Scalar::U32(0),
                after: Scalar::U32(retained),
            }],
            cleanup: Vec::new(),
            fuel: c.pass_fuel(pass, count, retained)?,
            trap: None,
        },
        PlanKind::Compaction,
    ))
}
fn copy_step(
    c: &compaction_source::Contract,
    live: &LiveLoop<'_>,
    pending: &mut VecDeque<Pending>,
) -> Option<Step> {
    let at = pending
        .iter()
        .position(|p| p.function == live.function && p.consumer == live.span)?;
    let p = pending.remove(at)?;
    let [input, count, cutoff, _] = c.names();
    let [(_, expected), (_, n), (_, threshold)] = p.operands.as_slice() else {
        return None;
    };
    if n.as_slice() != [*live.scalars.get(count)?]
        || threshold.as_slice() != [*live.scalars.get(cutoff)?]
        || live.array(input)?.get(..expected.len())? != expected
        || *live.scalars.get(&c.states()[0])? as usize != p.output.len()
    {
        return None;
    }
    compact_region(c, 1, live, p.output)
}

struct Bound<'a> {
    plan: seg::Verified,
    input: std::cell::Ref<'a, Vec<u32>>,
    offsets: Option<std::cell::Ref<'a, Vec<u32>>>,
}
impl Bound<'_> {
    fn input(&self) -> &[u32] {
        &self.input[..self.plan.ir().elements as usize]
    }
    fn offsets(&self) -> &[u32] {
        self.offsets.as_deref().map_or(&[], Vec::as_slice)
    }
}
fn bind<'a>(candidate: &Candidate, live: &LiveLoop<'a>, caps: Capabilities) -> Option<Bound<'a>> {
    if live.iterations == 0 {
        return None;
    }
    let names = match candidate {
        Candidate::Map(c) => c.names(),
        Candidate::Reduce(c) => c.names(),
        Candidate::Compact(c, 0) => c.names(),
        Candidate::Compact(_, _) => return None,
    };
    let input = live.array(&names[0])?;
    let mut offsets = None;
    let (ir, origin) = match candidate {
        Candidate::Map(c) => {
            let o = live.array(&names[1])?;
            if live.scalars.get(&names[2]).copied()? as usize != live.iterations
                || o.len() != live.iterations.checked_add(1)?
                || live.lengths.get(&names[3]) != Some(&input.len())
            {
                return None;
            }
            let ir = c.ir(&input, &o).ok()?;
            offsets = Some(o);
            ir
        }
        Candidate::Reduce(c) => {
            let o = live.array(&names[1])?;
            let segments = live.scalars.get(&names[2]).copied()?;
            if segments as usize != live.iterations
                || live.lengths.get(&names[3]) != Some(&(segments as usize))
            {
                return None;
            }
            c.prepare(
                &input,
                &o,
                segments,
                segmented_source::Budget {
                    max_elements: laws::MAX_ELEMENTS,
                    max_segments: laws::MAX_ELEMENTS,
                    max_output_bytes: u64::MAX,
                    max_work: u64::MAX,
                },
            )
            .ok()?;
            let ir = segmented_device::reduce_ir(c, &input, &o).ok()?;
            offsets = Some(o);
            ir
        }
        Candidate::Compact(c, 0) => {
            let count = *live.scalars.get(&names[1])?;
            if count as usize != live.iterations
                || input.len() < count as usize
                || live.scalars.get(&c.states()[0]) != Some(&0)
            {
                return None;
            }
            segmented_device::compact_ir(c, count, *live.scalars.get(&names[2])?).ok()?
        }
        Candidate::Compact(_, _) => return None,
    };
    if input.len() > laws::MAX_ELEMENTS
        || live.same_object(&names[0], &names[3])
        || offsets.is_some() && live.same_object(&names[1], &names[3])
    {
        return None;
    }
    let plan = seg::verify(ir, caps, origin).ok()?;
    let bound = Bound {
        plan,
        input,
        offsets,
    };
    seg::admit_inputs(&bound.plan, bound.input(), bound.offsets()).ok()?;
    Some(bound)
}
pub(crate) fn cost(
    candidate: &Candidate,
    live: &LiveLoop<'_>,
    caps: Capabilities,
    model: crate::plan::DeviceCostModel,
) -> Option<crate::plan::DeviceCost> {
    let b = bind(candidate, live, caps)?;
    let launches = b.plan.launches(b.input(), b.offsets()).ok()?.len();
    let bytes = b
        .plan
        .buffer_words()
        .into_iter()
        .try_fold(0usize, |n, words| {
            n.checked_add(words.max(1).checked_mul(4)?)
        })?;
    let elements = (b.plan.ir().elements as usize).checked_add(b.plan.ir().segments as usize)?;
    let operations = match &b.plan.ir().family {
        seg::Family::Map(ops) => ops.len(),
        seg::Family::Reduce(_) => 1,
        seg::Family::Compact { .. } => 4,
    };
    Some(model.estimate(bytes, elements, operations, launches))
}
fn complete(
    candidate: &Candidate,
    live: &LiveLoop<'_>,
    b: &Bound<'_>,
    output: Vec<u32>,
    pending: &mut VecDeque<Pending>,
) -> Option<Step> {
    let (names, spans, fuel) = match candidate {
        Candidate::Map(c) => (
            c.names(),
            [c.loop_span(), c.inner_span()],
            c.loop_fuel(live.iterations, b.input().len())?,
        ),
        Candidate::Reduce(c) => (
            c.names(),
            [c.loop_span(), c.inner_span()],
            c.prepare(
                b.input(),
                b.offsets(),
                b.plan.ir().segments,
                segmented_source::Budget {
                    max_elements: laws::MAX_ELEMENTS,
                    max_segments: laws::MAX_ELEMENTS,
                    max_output_bytes: u64::MAX,
                    max_work: u64::MAX,
                },
            )
            .ok()?
            .remaining_loop_ticks()?,
        ),
        Candidate::Compact(c, 0) => {
            let step = compact_region(c, 0, live, output.clone())?;
            pending.retain(|p| p.function != live.function || p.consumer != c.loop_spans()[1]);
            pending.push_back(Pending {
                function: live.function.clone(),
                consumer: c.loop_spans()[1],
                stage: 1,
                operands: vec![
                    (c.names()[0].clone(), b.input().to_vec()),
                    (c.names()[1].clone(), vec![b.plan.ir().elements]),
                    (
                        c.names()[2].clone(),
                        vec![*live.scalars.get(&c.names()[2])?],
                    ),
                ],
                output,
                required: false,
            });
            return Some(step);
        }
        Candidate::Compact(_, _) => return None,
    };
    let length = *live.lengths.get(&names[3])?;
    if output.len() != length || fuel > live.fuel_remaining {
        return None;
    }
    Some(step(
        CompletedRegion {
            function: live.function.clone(),
            span: spans[0],
            branches: vec![
                branch(spans[0], live.iterations, 1)?,
                branch(spans[1], b.input().len(), live.iterations)?,
            ],
            writes: vec![Write {
                owner: names[3].clone(),
                length,
                range: 0..length,
                values: Elements::U32(output),
            }],
            scalars: Vec::new(),
            cleanup: Vec::new(),
            fuel,
            trap: None,
        },
        PlanKind::Segmented,
    ))
}

#[cfg(all(feature = "metal", target_os = "macos"))]
pub(super) mod metal {
    pub(crate) use super::cost;
    use super::*;
    use crate::{
        device_cache::HintState,
        schedule::metal::{Error, Fault},
    };
    pub(crate) fn ineligible_reason(_: &Candidate, _: usize) -> &'static str {
        "segmented-live-guard-false"
    }
    pub(crate) fn kernels(
        candidate: &Candidate,
        live: &LiveLoop<'_>,
        caps: Capabilities,
    ) -> Vec<(String, Vec<String>)> {
        // A one-pass compaction is a host copy and loads no kernel.
        if let Candidate::Compact(_, 1) = candidate {
            return Vec::new();
        }
        bind(candidate, live, caps)
            .and_then(|b| {
                let msl = seg::emit_msl(&b.plan).ok()?;
                let names = b.plan.kernels().iter().map(|k| (*k).to_owned());
                Some(vec![(msl, names.collect())])
            })
            .unwrap_or_default()
    }
    pub(crate) fn step(
        _: &[u8],
        candidate: &Candidate,
        live: &LiveLoop<'_>,
        pending: &mut VecDeque<Pending>,
        device: &mut invar_metal::MetalDevice,
        fault: Fault,
    ) -> Result<(Step, bool, HintState), Error> {
        let step = if let Candidate::Compact(c, 1) = candidate {
            copy_step(c, live, pending)
        } else {
            let b = bind(candidate, live, device.capabilities())
                .ok_or_else(|| Error::unavailable("segmented live guard is false"))?;
            let result = device
                .run_segmented_with_fault(&b.plan, b.input(), b.offsets(), fault)
                .map_err(|e| Error::from_metal(e, device.is_poisoned()))?;
            complete(candidate, live, &b, result.output, pending)
        };
        step.map(|step| (step, false, HintState::Miss))
            .ok_or_else(|| Error::unavailable("segmented completion guard is false"))
    }
}
#[cfg(feature = "cuda")]
pub(super) mod cuda {
    pub(crate) use super::cost;
    use super::*;
    use crate::{
        device_cache::HintState,
        schedule::cuda::{Error, Fault},
    };
    pub(crate) fn step(
        _: &[u8],
        candidate: &Candidate,
        live: &LiveLoop<'_>,
        pending: &mut VecDeque<Pending>,
        device: &mut invar_cuda::cuda::CudaDevice,
        compiler: &std::path::Path,
        fault: Fault,
    ) -> Result<(Step, bool, HintState), Error> {
        let step = if let Candidate::Compact(c, 1) = candidate {
            copy_step(c, live, pending)
        } else {
            let b = bind(candidate, live, device.capabilities())
                .ok_or_else(|| Error::unavailable("segmented live guard is false"))?;
            let result = device
                .run_segmented_with_fault(
                    &b.plan,
                    b.input(),
                    b.offsets(),
                    compiler,
                    fault.collective(),
                )
                .map_err(|e| Error::from_cuda(e, device.is_poisoned()))?;
            complete(candidate, live, &b, result.output, pending)
        };
        step.map(|step| (step, false, HintState::Miss))
            .ok_or_else(|| Error::unavailable("segmented completion guard is false"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    const CAPS: Capabilities = Capabilities::Metal {
        apple_family: 9,
        msl_major: 3,
        msl_minor: 1,
    };
    const MAP: &str = "fn run(xs: read Array<u32>, offsets: read Array<u32>, count: u32) -> Array<u32> { var out = filled_u32(len(xs), 0u32); for s in 0u32..count { for i in offsets[s]..offsets[s + 1u32] { out[i] = wrapping_add(wrapping_mul(xs[i], 3u32), s); } } return move out; }";
    const REDUCE: &str = "fn run(xs: read Array<u32>, offsets: read Array<u32>, count: u32) -> Array<u32> { var out = filled_u32(count, 0u32); for s in 0u32..count { var acc = 0u32; for i in offsets[s]..offsets[s + 1u32] { acc = wrapping_add(acc, xs[i]); } out[s] = acc; } return move out; }";
    const COMPACT: &str = "fn run(xs: read Array<u32>, count: u32, cutoff: u32) -> Array<u32> { var selected = 0u32; for i in 0u32..count { if xs[i] < cutoff { selected = selected + 1u32; } } var out = filled_u32(selected, 0u32); var pos = 0u32; for j in 0u32..count { if xs[j] < cutoff { out[pos] = xs[j]; pos = pos + 1u32; } } return move out; }";

    fn program(source: &str) -> mir::Program {
        let checked =
            crate::check::check(crate::front::parse(source.as_bytes()).unwrap(), true).unwrap();
        mir::lower(&checked).unwrap()
    }
    fn source(function: &str, offsets: &str, groups: u32) -> String {
        format!(
            "{function} fn main() -> i32 {{ let xs: Array<u32> = [1u32, 2u32, 3u32, 4u32]; let offsets: Array<u32> = [{offsets}]; let out = run(xs, offsets, {groups}u32); if len(out) > 0u32 {{ if out[0u32] == 0u32 {{ return 10; }} }} return 0; }}"
        )
    }
    fn compact(function: &str) -> String {
        format!(
            "{function} fn main() -> i32 {{ let xs: Array<u32> = [4u32, 1u32, 3u32, 2u32]; let out = run(xs, 4u32, 3u32); if len(out) != 2u32 {{ return 10; }} if out[0u32] != 1u32 {{ return 11; }} if out[1u32] != 2u32 {{ return 12; }} return 0; }}"
        )
    }
    // The device model supplies words only. Publication and fuel accounting run
    // through the real MIR scheduler/validator, independently of this model.
    fn execute(source: &str, fuel: u64, damage: u8) -> (Observation, usize, usize) {
        let program = program(source);
        let regions = admission::analyze(source.as_bytes(), &program).unwrap();
        let candidates = regions
            .regions()
            .iter()
            .filter_map(|r| contract(source.as_bytes(), &program, r).unwrap())
            .collect::<Vec<_>>();
        let mut pending: VecDeque<Pending> = VecDeque::new();
        let mut submitted = 0;
        let mut provider = |live: &LiveLoop<'_>| -> Result<Option<LoopOverride>, EvalError> {
            let c = candidates.iter().find(|c| match c {
                Candidate::Map(c) => c.loop_span() == live.span,
                Candidate::Reduce(c) => c.loop_span() == live.span,
                Candidate::Compact(c, pass) => c.loop_spans()[*pass] == live.span,
            });
            let Some(c) = c else {
                return Ok(None);
            };
            let step = if let Candidate::Compact(c, 1) = c {
                if damage == 4
                    && let Some(staged) = pending.front_mut()
                {
                    staged.operands[2].1[0] ^= 1;
                }
                copy_step(c, live, &mut pending)
            } else if let Some(b) = bind(c, live, CAPS) {
                submitted += 1;
                if damage == 3 {
                    return Ok(None);
                }
                let mut output = seg::reference(&b.plan, b.input(), b.offsets()).unwrap();
                if damage == 1
                    && let Some(word) = output.first_mut()
                {
                    *word ^= 1;
                }
                complete(c, live, &b, output, &mut pending)
            } else {
                None
            };
            Ok(step.map(|mut step| {
                if damage == 2 {
                    step.plan.region.as_mut().unwrap().branches.pop();
                }
                step.plan
            }))
        };
        let (observed, applied) =
            mir_eval::observe_with_scheduler(&program, fuel, Some(&mut provider));
        (observed, applied, submitted)
    }
    fn agrees(source: &str, expected_regions: usize) {
        let program = program(source);
        let serial = mir_eval::observe(&program, 100_000);
        let (parallel, applied, _) = execute(source, 100_000, 0);
        assert_eq!(parallel.result, serial.result);
        assert_eq!(parallel.steps_used, serial.steps_used);
        assert_eq!(parallel.cleanup_trace, serial.cleanup_trace);
        assert_eq!(applied, expected_regions);
    }
    #[test]
    fn segmented_regions_match_mir_including_empty_segments_and_wrapping_product() {
        for f in [
            MAP.to_owned(),
            REDUCE.to_owned(),
            REDUCE
                .replace("acc = 0u32", "acc = 1u32")
                .replace("wrapping_add(acc", "wrapping_mul(acc"),
        ] {
            agrees(&source(&f, "0u32, 2u32, 2u32, 4u32", 3), 1);
            let empty = source(&f, "0u32, 0u32, 0u32", 2).replace("[1u32, 2u32, 3u32, 4u32]", "[]");
            agrees(&empty, 1);
        }
    }
    #[test]
    fn compact_two_regions_preserve_all_predicates_and_source_fuel() {
        for predicate in ["<", "<=", "==", "!=", ">=", ">"] {
            let source = compact(&COMPACT.replace(" < ", &format!(" {predicate} ")));
            agrees(&source, 2);
        }
        for fuel in 1..220 {
            agrees_with_fuel(&compact(COMPACT), fuel);
        }
        let mut spare = String::new();
        let mut remaining = crate::array_limits::MAX_LIVE_ELEMENTS - 1000;
        while remaining > 0 {
            let length = remaining.min(crate::array_limits::MAX_ARRAY_LEN);
            spare.push_str(&format!(
                "let spare_{remaining} = filled_u32({length}u32, 0u32); "
            ));
            remaining -= length;
        }
        let allocation_limit = format!(
            "{COMPACT} fn main() -> i32 {{ {spare} let xs = filled_u32(1000u32, 0u32); let out = run(xs, 1000u32, 1u32); return 0; }}"
        );
        let serial = mir_eval::observe(&program(&allocation_limit), 100_000);
        let (completed_count, applied, submitted) = execute(&allocation_limit, 100_000, 0);
        assert!(matches!(
            &serial.result,
            Err(EvalError::Tool {
                code: "ARRAY_SIZE_LIMIT",
                ..
            })
        ));
        assert_eq!(completed_count.result, serial.result);
        assert_eq!(completed_count.steps_used, serial.steps_used);
        assert_eq!((applied, submitted), (1, 1));
    }
    fn agrees_with_fuel(source: &str, fuel: u64) {
        let serial = mir_eval::observe(&program(source), fuel);
        let (device, _, _) = execute(source, fuel, 0);
        assert_eq!(device.result, serial.result, "fuel {fuel}");
        assert_eq!(device.steps_used, serial.steps_used, "fuel {fuel}");
        assert_eq!(device.cleanup_trace, serial.cleanup_trace, "fuel {fuel}");
    }
    #[test]
    fn malformed_offsets_zero_trip_unreached_and_failed_completion_publish_nothing() {
        let valid = source(MAP, "0u32, 4u32", 1);
        let p = program(&valid);
        let report = admission::analyze(valid.as_bytes(), &p).unwrap();
        assert!(contract(b"not valid source", &p, &report.regions()[0]).is_err());
        let changed = program(&valid.replace("return 0;", "return 1;"));
        assert!(contract(valid.as_bytes(), &changed, &report.regions()[0]).is_err());
        for offsets in [
            "0u32, 3u32, 2u32, 4u32",
            "1u32, 2u32, 4u32",
            "0u32, 2u32, 5u32",
        ] {
            agrees(
                &source(
                    MAP,
                    offsets,
                    if offsets.matches(',').count() == 3 {
                        3
                    } else {
                        2
                    },
                ),
                0,
            );
        }
        for f in [MAP, REDUCE] {
            let s = source(f, "0u32", 0);
            let (_, applied, submissions) = execute(&s, 100_000, 0);
            assert_eq!((applied, submissions), (0, 0));
            let s = format!("{f} fn main() -> i32 {{ return 0; }}");
            let (_, applied, submissions) = execute(&s, 100_000, 0);
            assert_eq!((applied, submissions), (0, 0));
        }
        for s in [source(MAP, "0u32, 4u32", 1), compact(COMPACT)] {
            let (failed, applied, _) = execute(&s, 100_000, 3);
            assert_eq!(applied, 0);
            assert_eq!(
                failed.result,
                mir_eval::observe(&program(&s), 100_000).result
            );
        }
    }
    #[test]
    fn corrupted_words_change_output_and_missing_counts_are_refused() {
        let s = compact(COMPACT);
        let (corrupt, _, _) = execute(&s, 100_000, 1);
        assert_ne!(
            corrupt.result,
            mir_eval::observe(&program(&s), 100_000).result
        );
        let (_, applied, _) = execute(&s, 100_000, 2);
        assert_eq!(applied, 0);
        let (changed_predicate, applied, _) = execute(&s, 100_000, 4);
        assert_eq!(applied, 1);
        assert_eq!(
            changed_predicate.result,
            mir_eval::observe(&program(&s), 100_000).result
        );
    }
    #[test]
    fn cost_uses_live_work_offsets_and_launches() {
        let s = source(MAP, "0u32, 2u32, 2u32, 4u32", 3);
        let program = program(&s);
        let report = admission::analyze(s.as_bytes(), &program).unwrap();
        let candidate = report
            .regions()
            .iter()
            .find_map(|r| contract(s.as_bytes(), &program, r).unwrap())
            .unwrap();
        let mut called = false;
        let mut provider = |live: &LiveLoop<'_>| -> Result<Option<LoopOverride>, EvalError> {
            if live.function == "run" && live.iterations == 3 {
                called = true;
                let fast = cost(
                    &candidate,
                    live,
                    CAPS,
                    crate::plan::DeviceCostModel {
                        cpu_bytes_per_ms: 1.0,
                        device_bytes_per_ms: 1e9,
                        device_fixed_ms: 0.0,
                    },
                )
                .unwrap();
                let slow = cost(
                    &candidate,
                    live,
                    CAPS,
                    crate::plan::DeviceCostModel {
                        cpu_bytes_per_ms: 1e9,
                        device_bytes_per_ms: 1.0,
                        device_fixed_ms: 1000.0,
                    },
                )
                .unwrap();
                assert!(fast.choose_device);
                assert!(!slow.choose_device);
            }
            Ok(None)
        };
        mir_eval::observe_with_scheduler(&program, 100_000, Some(&mut provider));
        assert!(called);
    }
}
