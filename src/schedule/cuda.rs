//! Forced CUDA execution at the same reached MIR loop heads as the CPU scheduler.

use std::{
    cell::RefCell,
    collections::VecDeque,
    path::Path,
    time::{Duration, Instant},
};

use invar_cuda::Runtime;
use invar_cuda::cuda::{CollectiveFault, CudaDevice, FusedFault};

use super::*;
use crate::{
    device_cache::HintState,
    plan::{DeviceCost, DeviceCostModel},
    schedule::reductions::{self, Reduction},
};

pub(crate) mod reduce;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Error {
    pub code: &'static str,
    pub message: String,
}

/// Internal failure injection. Never exposed as a source or CLI option.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Fault {
    None,
    OutOfMemory,
    LaunchFailure,
    PartialCompletion,
}

impl Fault {
    pub(super) fn fused(self) -> FusedFault {
        match self {
            Self::None => FusedFault::None,
            Self::OutOfMemory => FusedFault::TransferOutOfMemory,
            Self::LaunchFailure => FusedFault::LaunchFailure,
            Self::PartialCompletion => FusedFault::PartialCompletion,
        }
    }

    pub(crate) fn collective(self) -> CollectiveFault {
        match self {
            Self::None => CollectiveFault::None,
            Self::OutOfMemory => CollectiveFault::OutOfMemory,
            Self::LaunchFailure => CollectiveFault::LaunchFailure,
            Self::PartialCompletion => CollectiveFault::PartialCompletion,
        }
    }
}

impl Error {
    pub(super) fn unavailable(message: impl Into<String>) -> Self {
        Self {
            code: "CUDA_UNAVAILABLE",
            message: message.into(),
        }
    }

    pub(crate) fn mismatch(message: impl Into<String>) -> Self {
        Self {
            code: "DEVICE_TOOLCHAIN_MISMATCH",
            message: message.into(),
        }
    }

    pub(crate) fn from_cuda(error: invar_cuda::Error, poisoned: bool) -> Self {
        if poisoned {
            Self {
                code: "CUDA_PENDING",
                message: format!("CUDA completion is unconfirmed: {error:?}"),
            }
        } else if matches!(error, invar_cuda::Error::Compile(_)) {
            Self::mismatch(format!("CUDA compilation failed: {error:?}"))
        } else {
            Self::unavailable(format!(
                "CUDA execution failed before publication: {error:?}"
            ))
        }
    }
}

pub struct Run {
    pub observation: Observation,
    pub executed: Vec<Executed>,
    /// Every reached float reduction, in execution order.
    pub reductions: Vec<Reduction>,
    pub cache_hits: usize,
    pub hint_hits: usize,
    pub hint_invalids: usize,
}

#[derive(Clone, Debug)]
pub struct Choice {
    pub function: String,
    pub span: crate::front::Span,
    pub backend: &'static str,
    pub reason: &'static str,
    pub cost: Option<DeviceCost>,
    pub record: Option<tuning::CostRecord>,
}

pub struct AutoRun {
    pub observation: Observation,
    pub executed: Vec<Executed>,
    pub choices: Vec<Choice>,
    /// Every reached float reduction and where it ran, in execution order.
    pub reductions: Vec<Reduction>,
    pub cache_hits: usize,
    pub hint_hits: usize,
    pub hint_invalids: usize,
}

/// Run the graph-extracted plan on a live CUDA context; every reached eligible
/// region must publish a completed device result or return a typed tool error.
pub fn run_source_with_device(
    source: &[u8],
    program: &mir::Program,
    fuel: u64,
    device: &mut CudaDevice,
    compiler: &Path,
) -> Result<Run, Error> {
    run_source_with_device_fault(source, program, fuel, device, compiler, Fault::None)
}

#[doc(hidden)]
pub fn run_source_with_device_fault(
    source: &[u8],
    program: &mir::Program,
    fuel: u64,
    device: &mut CudaDevice,
    compiler: &Path,
    fault: Fault,
) -> Result<Run, Error> {
    if !compiler.is_file() {
        return Err(Error::mismatch(
            "NVRTC helper is unavailable; build CUDA binaries with `cargo build --features cuda --bins`",
        ));
    }
    if crate::spatial::contains(program) {
        return crate::spatial::cuda::run(source, program, fuel, device, compiler, fault);
    }
    if crate::sorting::contains(program) {
        return crate::sorting::run_cuda(source, program, fuel, device, compiler, fault);
    }
    let regions = families::analyze(source, program)
        .map_err(|error| Error::mismatch(format!("admission failed: {error:?}")))?;
    if !regions.iter().any(|region| !region.is_consumer()) && !reductions::has_reduction(program) {
        return Err(Error::unavailable("program has no CUDA-eligible region"));
    }
    let device = RefCell::new(device);
    let mut receipts = Vec::new();
    let mut reducer = |call: &mir_eval::ReduceCall<'_>| {
        reduce::forced(call, &mut device.borrow_mut(), compiler, &mut receipts)
    };
    let mut pending = VecDeque::new();
    let mut executed = Vec::new();
    let mut cache_hits = 0;
    let mut hint_hits = 0;
    let mut hint_invalids = 0;
    let mut inject = fault;
    let mut provider = |live: &LiveLoop<'_>| -> Result<Option<LoopOverride>, EvalError> {
        let Some(region) = regions
            .iter()
            .find(|region| region.function() == live.function && region.span() == live.span)
        else {
            return Ok(None);
        };
        let fault = if region.is_consumer() {
            Fault::None
        } else {
            std::mem::replace(&mut inject, Fault::None)
        };
        let result = region.cuda_step(
            source,
            live,
            &mut pending,
            &mut device.borrow_mut(),
            compiler,
            fault,
        );
        let (step, hit, hint) = result.map_err(|error| EvalError::Tool {
            code: error.code,
            message: error.message,
        })?;
        cache_hits += usize::from(hit);
        hint_hits += usize::from(hint == HintState::Hit);
        hint_invalids += usize::from(hint == HintState::Invalid);
        if hint_invalids >= 3 {
            return Err(EvalError::Tool {
                code: "DEVICE_TOOLCHAIN_MISMATCH",
                message: "repeated schedule-hint validation failure".into(),
            });
        }
        executed.push(Executed {
            function: live.function.clone(),
            span: live.span,
            kind: step.kind,
            published_array: step.plan.published_array(),
            implementation: None,
        });
        Ok(Some(step.plan))
    };
    let (observation, published) =
        mir_eval::observe_engine(program, fuel, Some(&mut provider), Some(&mut reducer));
    let produced = executed.len();
    let executed = super::keep_published(executed, &published);
    if observation.result.is_ok() && executed.len() != produced {
        return Err(Error::unavailable(
            "CUDA output was rejected by the live MIR publication guard",
        ));
    }
    let on_device = receipts.iter().any(|receipt| receipt.backend == "cuda");
    if observation.result.is_ok() && executed.is_empty() && !on_device {
        return Err(Error::unavailable("no CUDA result was published"));
    }
    Ok(Run {
        observation,
        executed,
        reductions: receipts,
        cache_hits,
        hint_hits,
        hint_invalids,
    })
}

pub fn run_source(
    source: &[u8],
    program: &mir::Program,
    fuel: u64,
    compiler: &Path,
) -> Result<Run, Error> {
    let mut device = CudaDevice::open().map_err(|error| Error::from_cuda(error, false))?;
    run_source_with_device(source, program, fuel, &mut device, compiler)
}

/// Auto selects each reached region from the same graph plan, then uses a
/// fresh CPU result only before device submission or after confirmed quiescence.
pub fn run_auto_source(
    source: &[u8],
    program: &mir::Program,
    fuel: u64,
    compiler: &Path,
) -> Result<AutoRun, Error> {
    run_auto_source_inner(source, program, fuel, compiler, None, Fault::None)
}

#[doc(hidden)]
pub fn run_auto_source_with_fault(
    source: &[u8],
    program: &mir::Program,
    fuel: u64,
    compiler: &Path,
    model: DeviceCostModel,
    fault: Fault,
) -> Result<AutoRun, Error> {
    run_auto_source_inner(source, program, fuel, compiler, Some(model), fault)
}

/// The auto cost model: `model` when given, else a fresh calibration; a
/// failed calibration leaves auto on CPU unless the device is pending.
fn auto_calibration(
    model: Option<DeviceCostModel>,
    workers: usize,
) -> Result<Option<DeviceCostModel>, Error> {
    match model {
        Some(model) => Ok(Some(model)),
        None => match calibrate_with_workers(workers) {
            Ok(model) => Ok(Some(model)),
            Err(error) if error.code == "CUDA_PENDING" => Err(error),
            Err(_) => Ok(None),
        },
    }
}

fn run_auto_source_inner(
    source: &[u8],
    program: &mir::Program,
    fuel: u64,
    compiler: &Path,
    model: Option<DeviceCostModel>,
    fault: Fault,
) -> Result<AutoRun, Error> {
    let spatial = crate::spatial::contains(program);
    if spatial || crate::sorting::contains(program) {
        // ponytail: unmeasured spatial families stay on CPU; admit devices only
        // after per-operation, revision-matched hardware/baseline qualification.
        let workers = selected_workers()?;
        // Reached float reductions take the same auto choice as in a
        // program without intrinsics; only a program with one calibrates.
        let calibration = if reductions::has_reduction(program) {
            auto_calibration(model, workers)?
        } else {
            None
        };
        let device = RefCell::new(calibration.and_then(|_| CudaDevice::open().ok()));
        let (mut receipts, hooks) = (Vec::new(), Hooks::default());
        let auto = reduce::Auto {
            compiler,
            model: calibration,
            workers,
        };
        let mut reducer = |call: &mir_eval::ReduceCall<'_>| {
            auto.reduce(call, device.borrow_mut().as_mut(), &mut receipts)
        };
        let cpu = if spatial {
            crate::spatial::run_cpu_with(source, program, fuel, workers, &hooks, &mut reducer)
        } else {
            crate::sorting::run_cpu_with(source, program, fuel, workers, &mut reducer)
        }
        .map_err(|e| Error::mismatch(format!("intrinsic CPU fallback: {e:?}")))?;
        let choices = cpu
            .executed
            .iter()
            .map(|step| Choice {
                function: step.function.clone(),
                span: step.span,
                backend: "cpu-parallel",
                reason: if step.kind == crate::plan::PlanKind::Sort {
                    "sort-cost-unmeasured"
                } else {
                    crate::spatial::placement::UNMEASURED
                },
                cost: None,
                record: None,
            })
            .collect();
        return Ok(AutoRun {
            observation: cpu.observation,
            executed: cpu.executed,
            reductions: receipts,
            choices,
            cache_hits: 0,
            hint_hits: 0,
            hint_invalids: 0,
        });
    }
    let regions = families::analyze(source, program)
        .map_err(|error| Error::mismatch(format!("admission failed: {error:?}")))?;
    let workers = selected_workers()?;
    let calibration = auto_calibration(model, workers)?;
    let device = RefCell::new(calibration.and_then(|_| CudaDevice::open().ok()));
    let auto = reduce::Auto {
        compiler,
        model: calibration,
        workers,
    };
    let mut receipts = Vec::new();
    let mut reducer = |call: &mir_eval::ReduceCall<'_>| {
        auto.reduce(call, device.borrow_mut().as_mut(), &mut receipts)
    };
    let hooks = Hooks::default();
    let mut pending = VecDeque::new();
    let mut executed = Vec::new();
    let mut choices = Vec::new();
    let mut cache_hits = 0;
    let mut hint_hits = 0;
    let mut hint_invalids = 0;
    let mut pending_backend = "cpu-parallel";
    let mut inject = fault;
    let mut provider = |live: &LiveLoop<'_>| -> Result<Option<LoopOverride>, EvalError> {
        let Some(region) = regions
            .iter()
            .find(|region| region.function() == live.function && region.span() == live.span)
        else {
            return Ok(None);
        };
        let is_consumer = region.is_consumer();
        let mut device = device.borrow_mut();
        let estimate = device.as_ref().and_then(|device| {
            calibration.and_then(|model| region.cuda_cost(live, device.capabilities(), model))
        });
        let mut record = device.as_ref().and_then(|device| {
            calibration.and_then(|model| region.cost_record(live, device.capabilities(), model))
        });
        let estimate = record.as_ref().map(|record| record.estimate).or(estimate);
        let use_device = !is_consumer
            && compiler.is_file()
            && if let Some(record) = &record {
                tuning::decide(
                    tuning::Control::Auto,
                    Some(record),
                    0,
                    0,
                    true,
                    device.is_some(),
                )
                .is_ok_and(|choice| choice.engine == tuning::Engine::Gpu)
            } else {
                estimate.is_some_and(|estimate| estimate.choose_device)
            };
        let mut reason = if is_consumer {
            "staged-consumer"
        } else if device.is_none() {
            "device-unavailable"
        } else if !compiler.is_file() {
            "toolchain-unavailable"
        } else if estimate.is_none() {
            "device-plan-ineligible"
        } else if use_device {
            "measured-device-cost"
        } else {
            "measured-cpu-cost"
        };
        let mut backend = if is_consumer {
            pending_backend
        } else {
            "cpu-parallel"
        };
        let mut hit = false;
        let result = if use_device {
            let active = device.as_mut().ok_or_else(|| EvalError::Tool {
                code: "CUDA_UNAVAILABLE",
                message: "device disappeared before submission".into(),
            })?;
            let fault = std::mem::replace(&mut inject, Fault::None);
            match region.cuda_step(source, live, &mut pending, active, compiler, fault) {
                Ok((step, cache_hit, hint)) => {
                    backend = "cuda";
                    hit = cache_hit;
                    hint_hits += usize::from(hint == HintState::Hit);
                    hint_invalids += usize::from(hint == HintState::Invalid);
                    if hint_invalids >= 3 {
                        return Err(EvalError::Tool {
                            code: "DEVICE_TOOLCHAIN_MISMATCH",
                            message: "repeated schedule-hint validation failure".into(),
                        });
                    }
                    Ok(Some(step))
                }
                Err(error) if active.is_poisoned() => Err(EvalError::Tool {
                    code: error.code,
                    message: error.message,
                }),
                Err(_) => {
                    reason = "cuda-failed-before-publication";
                    region
                        .cpu(live, &mut pending, workers, &hooks)
                        .map_err(|error| EvalError::Tool {
                            code: error.code(),
                            message: format!("CPU fallback failed: {error:?}"),
                        })
                }
            }
        } else {
            region
                .cpu(live, &mut pending, workers, &hooks)
                .map_err(|error| EvalError::Tool {
                    code: error.code(),
                    message: format!("CPU execution failed: {error:?}"),
                })
        }?;
        if let Some(step) = result {
            if backend == "cuda"
                && let Some(record) = &mut record
            {
                record.completion = tuning::Completion::Completed;
            }
            if region.is_producer() {
                pending_backend = backend;
            }
            cache_hits += usize::from(hit);
            executed.push(Executed {
                function: live.function.clone(),
                span: live.span,
                kind: step.kind,
                published_array: step.plan.published_array(),
                implementation: None,
            });
            choices.push(Choice {
                function: live.function.clone(),
                span: live.span,
                backend,
                reason,
                cost: estimate,
                record: record.clone(),
            });
            Ok(Some(step.plan))
        } else {
            choices.push(Choice {
                function: live.function.clone(),
                span: live.span,
                backend: "mir-serial",
                // A device failure before publication falls back to serial MIR when
                // the family has no CPU step; keep that cause, not a false guard.
                reason: if reason == "cuda-failed-before-publication" {
                    reason
                } else {
                    "live-guard-false"
                },
                cost: estimate,
                record: record.clone(),
            });
            Ok(None)
        }
    };
    let (observation, published) =
        mir_eval::observe_engine(program, fuel, Some(&mut provider), Some(&mut reducer));
    let executed = super::keep_published(executed, &published);
    // Offered choices are exactly the non-serial ones, in offer order.
    let mut outcomes = published.iter();
    for choice in choices
        .iter_mut()
        .filter(|choice| choice.backend != "mir-serial")
    {
        if outcomes.next() != Some(&true) {
            choice.backend = "mir-serial";
            choice.reason = "publication-guard-false";
        }
    }
    Ok(AutoRun {
        observation,
        executed,
        choices,
        reductions: receipts,
        cache_hits,
        hint_hits,
        hint_invalids,
    })
}

fn selected_workers() -> Result<usize, Error> {
    auto_workers().map_err(|error| Error {
        code: error.code(),
        message: format!("CPU worker selection failed: {error:?}"),
    })
}

/// Measure a bounded host/device round trip and a CPU wrapping map on this host.
/// The model uses observed elapsed times, including staging and completion.
pub fn calibrate() -> Result<DeviceCostModel, Error> {
    calibrate_with_workers(selected_workers()?)
}

fn calibrate_with_workers(workers: usize) -> Result<DeviceCostModel, Error> {
    const SMALL: usize = 64 * 1024;
    const LARGE: usize = 8 * 1024 * 1024;
    let mut runtime = Runtime::new(
        CudaDevice::open().map_err(|error| Error::from_cuda(error, false))?,
        1,
    );
    let mut measure_gpu = |bytes: usize| -> Result<f64, Error> {
        let input = vec![0x5au8; bytes];
        let mut samples = Vec::new();
        for trial in 0..6 {
            let started = Instant::now();
            let mut submitted = runtime
                .prepare(&input)
                .map_err(|error| Error::from_cuda(error, false))?
                .submit()
                .map_err(|error| Error::from_cuda(error, true))?;
            let output = loop {
                if let Some(output) = submitted
                    .poll()
                    .map_err(|error| Error::from_cuda(error, true))?
                {
                    break output;
                }
                if started.elapsed() > Duration::from_secs(5) {
                    let error = submitted.timeout();
                    return Err(Error::from_cuda(error, true));
                }
                std::thread::sleep(Duration::from_millis(1));
            };
            if output != input {
                return Err(Error::mismatch(
                    "CUDA calibration copy disagreed with input",
                ));
            }
            if trial > 0 {
                samples.push(started.elapsed().as_secs_f64() * 1000.0);
            }
        }
        samples.sort_by(f64::total_cmp);
        Ok(samples[2])
    };
    let small_ms = measure_gpu(SMALL)?;
    let large_ms = measure_gpu(LARGE)?;
    let delta_ms = large_ms - small_ms;
    if !delta_ms.is_finite() || delta_ms <= 0.0 {
        return Err(Error::unavailable(
            "CUDA calibration did not resolve transfer bandwidth",
        ));
    }
    // The same element executor and worker count auto uses for CPU regions.
    let cpu_ms = cpu_map_ms(laws::MAX_ELEMENTS, workers).unwrap_or(f64::NAN);
    if !cpu_ms.is_finite() || cpu_ms <= 0.0 {
        return Err(Error::unavailable(
            "CPU calibration did not resolve map bandwidth",
        ));
    }
    Ok(DeviceCostModel {
        cpu_bytes_per_ms: (8 * laws::MAX_ELEMENTS) as f64 / cpu_ms,
        device_bytes_per_ms: (2 * (LARGE - SMALL)) as f64 / delta_ms,
        device_fixed_ms: small_ms,
    })
}

#[cfg(test)]
mod ai_01_fault_tests {
    #[test]
    fn issue_ai_01_machine_diagnostics_cuda_backend_faults() {
        let schema: serde_json::Value = serde_json::from_str(include_str!(
            "../../schema/invar-diagnostics-v1.schema.json"
        ))
        .unwrap();
        for (error, poisoned, code) in [
            (invar_cuda::Error::Poisoned, true, "CUDA_PENDING"),
            (
                invar_cuda::Error::Compile("injected compiler failure".into()),
                false,
                "DEVICE_TOOLCHAIN_MISMATCH",
            ),
        ] {
            let error = super::Error::from_cuda(error, poisoned);
            assert_eq!(error.code, code);
            let document = crate::diagnostics::document(
                "tool_error",
                vec![crate::diagnostics::diagnostic(
                    error.code,
                    &error.message,
                    "fixture.inv",
                    b"",
                    None,
                )],
            );
            assert!(jsonschema::is_valid(&schema, &document), "{document}");
        }
    }
}
