//! Reached grid loops, with source-bound plans and completion-only publication.
use super::super::*;
use crate::{check, front, stencil_source, transpose_source};
use std::rc::Rc;

pub(super) enum Candidate {
    Transpose(Rc<transpose_source::Contract>),
    Stencil(Rc<stencil_source::Contract>),
}

/// A PNF transpose or stencil admission binds the source grid contract at its
/// span; a contract of the other kind is no lowering. The recognizer this
/// replaces is the oracle in `tests/support/family_oracle.rs`.
pub(super) fn lower(
    admitted: &lowering::Admitted,
    live: &lowering::Live<'_>,
) -> Result<Option<Candidate>, admission::Error> {
    use crate::general_net::regions::family::Family::{Stencil, Transpose};
    if !admitted.admits(&[Transpose, Stencil]) {
        return Ok(None);
    }
    let candidate = bind(live.source, live.program, live.region)?;
    Ok(candidate.filter(|c| {
        admitted.admits(&[match c {
            Candidate::Transpose(_) => Transpose,
            Candidate::Stencil(_) => Stencil,
        }])
    }))
}

fn bind(
    source: &[u8],
    program: &mir::Program,
    region: &Region,
) -> Result<Option<Candidate>, admission::Error> {
    let checked = front::parse(source)
        .and_then(|p| check::check(p, true))
        .map_err(|e| admission::Error {
            code: "GRID_SOURCE",
            message: format!("{e:?}"),
        })?;
    let candidate = transpose_source::host::contracts(&checked)
        .into_iter()
        .find(|c| c.function() == region.function() && c.loop_span() == region.span())
        .map(Candidate::Transpose)
        .or_else(|| {
            stencil_source::host::contracts(&checked)
                .into_iter()
                .find(|c| c.function() == region.function() && c.loop_span() == region.span())
                .map(Candidate::Stencil)
        });
    if let Some(candidate) = &candidate {
        let fingerprint = match candidate {
            Candidate::Transpose(c) => c.function_contract(),
            Candidate::Stencil(c) => c.function_contract(),
        };
        if !program.functions.iter().any(|f| {
            f.name == region.function() && crate::tensor::function_fingerprint(f) == fingerprint
        }) {
            return Err(admission::Error {
                code: "GRID_MIR_MISMATCH",
                message: "grid source and MIR disagree".into(),
            });
        }
    }
    Ok(candidate)
}

pub(super) fn cpu(
    _candidate: &Candidate,
    _live: &LiveLoop<'_>,
    _pending: &mut VecDeque<Pending>,
    _workers: usize,
    _hooks: &Hooks,
) -> Result<Option<Step>, Error> {
    Ok(None)
}

#[cfg(any(feature = "cuda", all(feature = "metal", target_os = "macos")))]
pub(super) fn is_consumer(_: &Candidate) -> bool {
    false
}
#[cfg(any(feature = "cuda", all(feature = "metal", target_os = "macos")))]
pub(super) fn is_producer(_: &Candidate) -> bool {
    false
}

// Shared binding and receipt logic is also exercised on CPU with completed words.
#[cfg_attr(
    not(any(test, feature = "cuda", all(feature = "metal", target_os = "macos"))),
    allow(dead_code)
)]
enum Bound {
    Transpose(invar_device::transpose::Verified),
    Stencil(invar_device::stencil_u32::Verified),
}

#[cfg_attr(
    not(any(test, feature = "cuda", all(feature = "metal", target_os = "macos"))),
    allow(dead_code)
)]
impl Candidate {
    #[cfg(any(feature = "cuda", all(feature = "metal", target_os = "macos")))]
    fn refusal_reason(&self) -> &'static str {
        if matches!(self, Self::Stencil(c) if c.element() == front::Type::ArrayF32) {
            "f32 stencil device runtime is unavailable (u32 only)"
        } else {
            "grid live dimensions, layout, alias or source fuel guard is false"
        }
    }
    fn roles(&self) -> (&str, &str, front::Type) {
        match self {
            Self::Transpose(c) => (c.input(), c.output(), c.element()),
            Self::Stencil(c) => (c.input(), c.output(), c.element()),
        }
    }

    fn bind(&self, live: &LiveLoop<'_>, caps: invar_device::Capabilities) -> Option<Bound> {
        let (input, output, element) = self.roles();
        let (input_len, output_len) = match element {
            front::Type::U32 | front::Type::ArrayU32 => {
                (live.array(input)?.len(), live.array(output)?.len())
            }
            front::Type::F32 => (live.floating(input)?.len(), live.floating(output)?.len()),
            // The device stencil runtime performs wrapping u32 only.
            _ => return None,
        };
        match self {
            Self::Transpose(c) => {
                let (dimensions, _) = c.reached(live)?;
                let plan = c.device_plan(dimensions, input_len as u64, caps).ok()?;
                (plan.ir().output.elements == output_len as u64).then_some(Bound::Transpose(plan))
            }
            Self::Stencil(c) => {
                let (dimensions, _) = c.reached(live)?;
                let plan = c.device_u32(dimensions, 1, caps).ok()?;
                (plan.ir().input.elements <= input_len as u64
                    && plan.ir().output.elements == output_len as u64)
                    .then_some(Bound::Stencil(plan))
            }
        }
    }

    fn finish(&self, bound: &Bound, input: &[u32], output: Vec<u32>) -> Option<Step> {
        let (region, kind) = match (self, bound) {
            (Self::Transpose(c), Bound::Transpose(plan)) => (
                c.completed_region(plan.ir().dimensions, input, output)?,
                PlanKind::Transpose,
            ),
            (Self::Stencil(c), Bound::Stencil(plan)) => {
                (c.completed_region(plan, input, output)?, PlanKind::Stencil)
            }
            _ => return None,
        };
        Some(Step {
            plan: LoopOverride::completed(region),
            kind,
            chunks: Vec::new(),
            joined: 0,
        })
    }
}

#[cfg(any(feature = "cuda", all(feature = "metal", target_os = "macos")))]
pub(super) fn cost(
    candidate: &Candidate,
    live: &LiveLoop<'_>,
    caps: invar_device::Capabilities,
    model: crate::plan::DeviceCostModel,
) -> Option<crate::plan::DeviceCost> {
    let bound = candidate.bind(live, caps)?;
    let (input, output, cells, operations) = match &bound {
        Bound::Transpose(p) => (
            p.ir().input.elements,
            p.ir().output.elements,
            p.ir().output.elements,
            1,
        ),
        Bound::Stencil(p) => (
            p.ir().input.elements,
            p.ir().output.elements,
            p.ir().output.elements,
            4,
        ),
    };
    // Runtimes upload input and initialized output, then download output.
    let bytes = usize::try_from(input.checked_add(output.checked_mul(2)?)?.checked_mul(4)?).ok()?;
    Some(model.estimate(bytes, usize::try_from(cells).ok()?, operations, 1))
}

#[cfg(all(feature = "metal", target_os = "macos"))]
pub(super) mod metal {
    pub(in crate::schedule::families) use super::cost;
    use super::*;
    use crate::{
        device_cache::HintState,
        schedule::metal::{Error, Fault},
    };
    pub(in crate::schedule::families) fn ineligible_reason(
        candidate: &Candidate,
        _: usize,
    ) -> &'static str {
        candidate.refusal_reason()
    }
    pub(in crate::schedule::families) fn kernels(
        candidate: &Candidate,
        live: &LiveLoop<'_>,
        caps: invar_device::Capabilities,
    ) -> Vec<(String, Vec<String>)> {
        use invar_device::{stencil_u32, transpose};
        let kernel = match candidate.bind(live, caps) {
            Some(Bound::Transpose(p)) => transpose::emit_msl(&p)
                .ok()
                .map(|msl| (msl, transpose::KERNEL_NAME)),
            Some(Bound::Stencil(p)) => stencil_u32::emit_msl(&p)
                .ok()
                .map(|msl| (msl, stencil_u32::KERNEL_NAME)),
            None => None,
        };
        kernel
            .map(|(msl, name)| vec![(msl, vec![name.to_owned()])])
            .unwrap_or_default()
    }
    pub(in crate::schedule::families) fn step(
        _source: &[u8],
        candidate: &Candidate,
        live: &LiveLoop<'_>,
        _pending: &mut VecDeque<Pending>,
        device: &mut invar_metal::MetalDevice,
        fault: Fault,
    ) -> Result<(Step, bool, HintState), Error> {
        let bound = candidate
            .bind(live, device.capabilities())
            .ok_or_else(|| Error::unavailable(candidate.refusal_reason()))?;
        let (input, output, _) = candidate.roles();
        let _integer_output = live.array(output);
        let _floating_output = live.floating(output);
        let integer = live.array(input);
        let floating = live.floating(input);
        let bits = floating
            .as_ref()
            .map(|a| {
                let mut bits = Vec::new();
                bits.try_reserve_exact(a.len())
                    .map_err(|_| Error::unavailable("grid staging allocation failed"))?;
                bits.extend(a.iter().map(|v| v.to_bits()));
                Ok::<_, Error>(bits)
            })
            .transpose()?;
        let words = integer
            .as_deref()
            .map(Vec::as_slice)
            .or(bits.as_deref())
            .ok_or_else(|| Error::unavailable("grid input loan unavailable"))?;
        let (result, words) = match &bound {
            Bound::Transpose(p) => (device.run_transpose_with_fault(p, words, fault), words),
            Bound::Stencil(p) => {
                let words = &words[..p.ir().input.elements as usize];
                (device.run_stencil_u32_with_fault(p, words, fault), words)
            }
        };
        let result = result.map_err(|e| Error::from_metal(e, device.is_poisoned()))?;
        if result.phases != 1 || result.final_buffer != 1 {
            return Err(Error::unavailable("grid phase completion mismatch"));
        }
        let step = candidate
            .finish(&bound, words, result.output)
            .ok_or_else(|| Error::unavailable("grid completed words mismatch"))?;
        Ok((step, false, HintState::Miss))
    }
}

#[cfg(feature = "cuda")]
pub(super) mod cuda {
    pub(in crate::schedule::families) use super::cost;
    use super::*;
    use crate::{
        device_cache::HintState,
        schedule::cuda::{Error, Fault},
    };
    pub(in crate::schedule::families) fn step(
        _source: &[u8],
        candidate: &Candidate,
        live: &LiveLoop<'_>,
        _pending: &mut VecDeque<Pending>,
        device: &mut invar_cuda::cuda::CudaDevice,
        compiler: &std::path::Path,
        fault: Fault,
    ) -> Result<(Step, bool, HintState), Error> {
        let bound = candidate
            .bind(live, device.capabilities())
            .ok_or_else(|| Error::unavailable(candidate.refusal_reason()))?;
        let (input, output, _) = candidate.roles();
        let _integer_output = live.array(output);
        let _floating_output = live.floating(output);
        let integer = live.array(input);
        let floating = live.floating(input);
        let bits = floating
            .as_ref()
            .map(|a| {
                let mut bits = Vec::new();
                bits.try_reserve_exact(a.len())
                    .map_err(|_| Error::unavailable("grid staging allocation failed"))?;
                bits.extend(a.iter().map(|v| v.to_bits()));
                Ok::<_, Error>(bits)
            })
            .transpose()?;
        let words = integer
            .as_deref()
            .map(Vec::as_slice)
            .or(bits.as_deref())
            .ok_or_else(|| Error::unavailable("grid input loan unavailable"))?;
        use invar_cuda::cuda::StencilFault;
        let fault = match fault {
            Fault::None => StencilFault::None,
            Fault::OutOfMemory => StencilFault::OutOfMemory,
            Fault::LaunchFailure => StencilFault::LaunchFailure,
            Fault::PartialCompletion => StencilFault::PartialCompletion,
        };
        let (output, words) = match &bound {
            Bound::Transpose(p) => (
                device
                    .run_transpose_with_fault(p, words, compiler, fault)
                    .map_err(|e| Error::from_cuda(e, device.is_poisoned()))?
                    .output,
                words,
            ),
            Bound::Stencil(p) => {
                let words = &words[..p.ir().input.elements as usize];
                let result = device
                    .run_stencil_u32_with_fault(p, words, compiler, fault)
                    .map_err(|e| Error::from_cuda(e, device.is_poisoned()))?;
                if result.phases != 1 || result.final_buffer != 1 {
                    return Err(Error::unavailable("grid phase completion mismatch"));
                }
                (result.output, words)
            }
        };
        let step = candidate
            .finish(&bound, words, output)
            .ok_or_else(|| Error::unavailable("grid completed words mismatch"))?;
        Ok((step, false, HintState::Miss))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    mod fixtures {
        include!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/support/a1_grid_source.rs"
        ));
    }
    use crate::{eval, mir_eval};
    use invar_device::Capabilities;
    const FUEL: u64 = 10_000_000;

    fn completed(bound: &Bound, input: &[u32]) -> Vec<u32> {
        match bound {
            Bound::Transpose(p) => {
                let [rows, cols] = p.ir().dimensions.map(|n| n as usize);
                (0..rows * cols)
                    .map(|i| input[(i % rows) * cols + i / rows])
                    .collect()
            }
            Bound::Stencil(p) => {
                let [rows, cols] = p.ir().dimensions;
                (0..rows * cols)
                    .map(|i| {
                        p.reads([i / cols, i % cols])
                            .unwrap()
                            .into_iter()
                            .flatten()
                            .fold(0u32, |sum, at| sum.wrapping_add(input[at as usize]))
                    })
                    .collect()
            }
        }
    }

    // Inject completed CPU words at the exact adapter boundary. This is never
    // reported as device evidence; hardware tests use the actual runtimes.
    fn run(source: &str, fuel: u64, mutation: u8) -> (eval::Observation, usize, usize) {
        let checked = check::check(front::parse(source.as_bytes()).unwrap(), true).unwrap();
        let program = mir::lower(&checked).unwrap();
        let regions = families::analyze(source.as_bytes(), &program).unwrap();
        let mut submitted = 0;
        let mut provider = |live: &LiveLoop<'_>| {
            let Some(selected) = regions
                .iter()
                .find(|r| r.function() == live.function && r.span() == live.span)
            else {
                return Ok(None);
            };
            let families::Candidate::Grid(candidate) = &selected.candidate else {
                return Ok(None);
            };
            let Some(bound) = candidate.bind(live, Capabilities::cuda_12_9_sm89()) else {
                return Ok(None);
            };
            let (name, _, _) = candidate.roles();
            let input = if let Some(a) = live.array(name) {
                a.to_vec()
            } else {
                live.floating(name)
                    .unwrap()
                    .iter()
                    .map(|v| v.to_bits())
                    .collect()
            };
            let input = match &bound {
                Bound::Stencil(p) => &input[..p.ir().input.elements as usize],
                _ => &input,
            };
            #[cfg(feature = "cuda")]
            {
                let cpu = crate::plan::DeviceCostModel {
                    cpu_bytes_per_ms: 1e12,
                    device_bytes_per_ms: 1.0,
                    device_fixed_ms: 10.0,
                };
                let gpu = crate::plan::DeviceCostModel {
                    cpu_bytes_per_ms: 1.0,
                    device_bytes_per_ms: 1e12,
                    device_fixed_ms: 0.0,
                };
                assert!(
                    !cost(candidate, live, Capabilities::cuda_12_9_sm89(), cpu)
                        .unwrap()
                        .choose_device
                );
                assert!(
                    cost(candidate, live, Capabilities::cuda_12_9_sm89(), gpu)
                        .unwrap()
                        .choose_device
                );
            }
            submitted += 1;
            let mut output = completed(&bound, input);
            if mutation == 1 {
                output[0] ^= 1;
            }
            if mutation == 3 {
                output.pop();
            }
            let Some(mut step) = candidate.finish(&bound, input, output) else {
                return Ok(None);
            };
            if mutation == 2 {
                step.plan.region.as_mut().unwrap().branches.pop();
            }
            Ok(Some(step.plan))
        };
        let (observation, applied) =
            mir_eval::observe_with_scheduler(&program, fuel, Some(&mut provider));
        let serial = mir_eval::observe(&program, fuel);
        assert_eq!(observation.result, serial.result);
        assert_eq!(observation.steps_used, serial.steps_used);
        assert_eq!(observation.cleanup_trace, serial.cleanup_trace);
        assert_eq!(
            observation.result,
            crate::eval::observe(&checked, fuel).result
        );
        (observation, applied, submitted)
    }

    #[test]
    fn issue_ai_01_machine_diagnostics_grid_admission_faults() {
        let source = fixtures::transpose(false, 3, 5, false, true);
        let checked = check::check(front::parse(source.as_bytes()).unwrap(), true).unwrap();
        let program = mir::lower(&checked).unwrap();
        let report = admission::analyze(source.as_bytes(), &program).unwrap();
        let contract = transpose_source::host::contracts(&checked)
            .into_iter()
            .next()
            .unwrap();
        let region = report
            .regions()
            .iter()
            .find(|r| r.function() == contract.function() && r.span() == contract.loop_span())
            .unwrap();
        assert!(
            bind(source.as_bytes(), &program, region)
                .unwrap()
                .is_some()
        );
        let invalid_source = bind(b"!", &program, region).err().unwrap();
        let mut forged = program;
        forged
            .functions
            .iter_mut()
            .find(|f| f.name == region.function())
            .unwrap()
            .name
            .push('_');
        let mismatched = bind(source.as_bytes(), &forged, region).err().unwrap();
        let schema: serde_json::Value = serde_json::from_str(include_str!(
            "../../../schema/invar-diagnostics-v1.schema.json"
        ))
        .unwrap();
        for (error, expected, input) in [
            (invalid_source, "GRID_SOURCE", &b"!"[..]),
            (mismatched, "GRID_MIR_MISMATCH", source.as_bytes()),
        ] {
            assert_eq!(error.code, expected);
            let document = crate::diagnostics::document(
                "tool_error",
                vec![crate::diagnostics::diagnostic(
                    error.code,
                    &error.message,
                    "fixture.inv",
                    input,
                    None,
                )],
            );
            assert!(jsonschema::is_valid(&schema, &document), "{document}");
        }
    }

    #[test]
    fn a1_grid_completed_forms_and_word_count_mutations() {
        let mut sources = vec![
            fixtures::transpose(false, 3, 5, false, true),
            fixtures::transpose(true, 3, 5, false, true),
        ];
        for clamp in [false, true] {
            for shape in [[1, 1], [1, 5], [5, 1], [3, 5]] {
                for steps in [1, 2, 3] {
                    sources.push(fixtures::stencil(clamp, false, shape[0], shape[1], steps));
                }
            }
        }
        for source in sources {
            let (observation, applied, submitted) = run(&source, FUEL, 0);
            assert_eq!(observation.result, Ok(eval::Value::I32(7)));
            assert!(applied > 0);
            assert_eq!(applied, submitted);
            for mutation in [1, 2, 3] {
                let (_, applied, submitted) = run(&source, FUEL, mutation);
                assert_eq!(applied, 0);
                assert!(submitted > 0);
            }
        }
    }

    #[test]
    fn a1_grid_unreached_empty_f32_stencil_and_short_input_submit_nothing() {
        for source in [
            fixtures::transpose(false, 3, 5, false, false),
            fixtures::transpose(false, 0, 5, false, true),
            fixtures::transpose(false, 5, 0, false, true),
            fixtures::transpose(false, 3, 5, true, true),
            fixtures::stencil(false, true, 3, 5, 1),
            fixtures::stencil(false, false, 3, 5, 0),
            fixtures::stencil(true, false, 0, 5, 3),
            fixtures::stencil(true, false, 5, 0, 3),
        ] {
            let (_, applied, submitted) = run(&source, FUEL, 0);
            assert_eq!((applied, submitted), (0, 0));
        }
    }

    #[test]
    fn a1_grid_fuel_prefixes_never_publish_short_budget() {
        for source in [
            fixtures::transpose(false, 2, 3, false, true),
            fixtures::stencil(true, false, 2, 3, 1),
        ] {
            let (full, _, _) = run(&source, FUEL, 0);
            for fuel in 1..=full.steps_used {
                run(&source, fuel, 0);
            }
        }
    }
}
