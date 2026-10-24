//! Forced Metal execution at the same reached MIR loop heads as the CPU scheduler.

use std::{cell::RefCell, collections::VecDeque};

pub use invar_metal::Fault;
use invar_metal::MetalDevice;

use super::*;
use crate::{device_cache::HintState, plan::DeviceCostModel, schedule::reductions};

pub(crate) mod reduce;

pub use super::device::{AutoRun, Choice, Run};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Error {
    pub code: &'static str,
    pub message: String,
}

impl Error {
    pub(super) fn unavailable(message: impl Into<String>) -> Self {
        Self {
            code: "METAL_UNAVAILABLE",
            message: message.into(),
        }
    }

    pub(crate) fn mismatch(message: impl Into<String>) -> Self {
        Self {
            code: "DEVICE_TOOLCHAIN_MISMATCH",
            message: message.into(),
        }
    }

    pub(crate) fn from_metal(error: invar_metal::Error, poisoned: bool) -> Self {
        if poisoned {
            Self {
                code: "METAL_PENDING",
                message: format!("Metal completion is unconfirmed: {error:?}"),
            }
        } else if let invar_metal::Error::Library(refusal) = &error {
            Self::mismatch(format!("compiled-ahead Metal library refused: {refusal:?}"))
        } else if matches!(error, invar_metal::Error::Toolchain(_)) {
            Self::mismatch(format!("Metal compilation failed: {error:?}"))
        } else {
            Self::unavailable(format!(
                "Metal execution failed before publication: {error:?}"
            ))
        }
    }
}

/// Developer flag: `INVAR_METAL_RUNTIME_SOURCE=1` lets a run without a
/// compiled-ahead library compile MSL source. Off by default; debugging only.
pub const RUNTIME_SOURCE_ENV: &str = "INVAR_METAL_RUNTIME_SOURCE";

/// Where a run's Metal pipelines come from.
#[derive(Clone, Copy)]
pub enum Kernels<'a> {
    /// Production: the program's compiled-ahead libraries. No compiler runs.
    Prebuilt(&'a crate::metal_aot::Image),
    /// Developer debugging only: compile generated MSL at run time.
    RuntimeSource,
    /// No library was supplied: forced Metal refuses, auto stays on the CPU.
    Missing,
}

impl Kernels<'_> {
    /// `RuntimeSource` when the developer flag is set, otherwise `Missing`.
    pub fn developer() -> Self {
        if std::env::var_os(RUNTIME_SOURCE_ENV).is_some_and(|v| v == "1") {
            Self::RuntimeSource
        } else {
            Self::Missing
        }
    }
}

fn refused(e: crate::metal_aot::Error) -> Error {
    Error::mismatch(format!("compiled-ahead Metal library refused: {e:?}"))
}

/// A supplied compiled-ahead library must be the build of exactly `source`.
/// Auto checks it before routes that never open the device.
fn verify_prebuilt(source: &[u8], kernels: Kernels<'_>) -> Result<(), Error> {
    use sha2::Digest;
    if let Kernels::Prebuilt(image) = kernels {
        image
            .header()
            .verify(
                crate::metal_aot::Route::Run,
                "main",
                sha2::Sha256::digest(source).into(),
            )
            .map_err(refused)?;
    }
    Ok(())
}

/// Open a device that holds exactly `kernels`. A prebuilt container must be
/// the build of exactly `source`; it is matched before Metal sees a byte.
pub fn open_device(source: &[u8], kernels: Kernels<'_>) -> Result<MetalDevice, Error> {
    verify_prebuilt(source, kernels)?;
    match kernels {
        Kernels::Prebuilt(image) => {
            let mut device =
                MetalDevice::open().map_err(|error| Error::from_metal(error, false))?;
            image.install(&mut device).map_err(refused)?;
            Ok(device)
        }
        Kernels::RuntimeSource => {
            MetalDevice::open_runtime_source().map_err(|error| Error::from_metal(error, false))
        }
        Kernels::Missing => Err(Error::unavailable(
            "Metal runs compiled-ahead libraries: build them with `invar build PATH --target metal --output FILE` and pass `--artifact FILE`",
        )),
    }
}

/// The compiled-ahead kernels of every family region (grid, segmented,
/// tensor) that a `run` at `fuel` reaches on a device with `caps`. Their MSL
/// depends on live dimensions, so an exact serial run discovers them: Invar
/// source has no external input, and the device publishes exactly the serial
/// result, so a Metal run at the same fuel reaches the same states.
pub fn family_kernels(
    source: &[u8],
    program: &mir::Program,
    fuel: u64,
    caps: invar_device::Capabilities,
) -> Result<Vec<(String, Vec<String>)>, Error> {
    let regions = families::analyze(source, program)
        .map_err(|error| Error::mismatch(format!("admission failed: {error:?}")))?;
    let mut kernels: Vec<(String, Vec<String>)> = Vec::new();
    let mut provider = |live: &LiveLoop<'_>| -> Result<Option<LoopOverride>, EvalError> {
        let region = regions
            .iter()
            .find(|region| region.function() == live.function && region.span() == live.span);
        for kernel in region
            .map(|region| region.metal_kernels(live, caps))
            .unwrap_or_default()
        {
            if !kernels.contains(&kernel) {
                kernels.push(kernel);
            }
        }
        Ok(None)
    };
    mir_eval::observe_engine(program, fuel, Some(&mut provider), None);
    Ok(kernels)
}

/// Run the graph-extracted plan on a live Metal context; every reached eligible
/// region must publish a completed device result or return a typed tool error.
pub fn run_source_with_device(
    source: &[u8],
    program: &mir::Program,
    fuel: u64,
    device: &mut MetalDevice,
) -> Result<Run, Error> {
    run_source_with_device_fault(source, program, fuel, device, Fault::None)
}

#[doc(hidden)]
pub fn run_source_with_device_fault(
    source: &[u8],
    program: &mir::Program,
    fuel: u64,
    device: &mut MetalDevice,
    fault: Fault,
) -> Result<Run, Error> {
    if crate::spatial::contains(program) {
        return crate::spatial::metal::run(source, program, fuel, device, fault);
    }
    if crate::sorting::contains(program) {
        return crate::sorting::run_metal(source, program, fuel, device, fault);
    }
    let regions = families::analyze(source, program)
        .map_err(|error| Error::mismatch(format!("admission failed: {error:?}")))?;
    if !regions.iter().any(|region| !region.is_consumer()) && !reductions::has_reduction(program) {
        return Err(Error::unavailable("program has no Metal-eligible region"));
    }
    let device = RefCell::new(device);
    let mut receipts = Vec::new();
    let workers = selected_workers()?;
    let mut reducer = |call: &mir_eval::ReduceCall<'_>| {
        reduce::forced(call, &mut device.borrow_mut(), workers, &mut receipts)
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
        let result = region.metal_step(source, live, &mut pending, &mut device.borrow_mut(), fault);
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
            "Metal output was rejected by the live MIR publication guard",
        ));
    }
    // A guard fallback still attempted the device, so it counts as Metal work.
    let on_device = receipts.iter().any(|receipt| {
        receipt.backend == "metal" || receipt.reason == "metal-failed-before-publication"
    });
    if observation.result.is_ok() && executed.is_empty() && !on_device {
        return Err(Error::unavailable("no Metal result was published"));
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
    kernels: Kernels<'_>,
) -> Result<Run, Error> {
    let mut device = open_device(source, kernels)?;
    run_source_with_device(source, program, fuel, &mut device)
}

/// Auto selects each reached region from the same graph plan, then uses a
/// fresh CPU result only before device submission or after confirmed quiescence.
pub fn run_auto_source(
    source: &[u8],
    program: &mir::Program,
    fuel: u64,
    kernels: Kernels<'_>,
) -> Result<AutoRun, Error> {
    run_auto_source_inner(source, program, fuel, kernels, None, Fault::None)
}

#[doc(hidden)]
pub fn run_auto_source_with_fault(
    source: &[u8],
    program: &mir::Program,
    fuel: u64,
    kernels: Kernels<'_>,
    model: DeviceCostModel,
    fault: Fault,
) -> Result<AutoRun, Error> {
    run_auto_source_inner(source, program, fuel, kernels, Some(model), fault)
}

fn run_auto_source_inner(
    source: &[u8],
    program: &mir::Program,
    fuel: u64,
    kernels: Kernels<'_>,
    model: Option<DeviceCostModel>,
    fault: Fault,
) -> Result<AutoRun, Error> {
    // A refused library is an error even where auto runs on the CPU alone:
    // the sort and spatial branch below, or a host without a device.
    verify_prebuilt(source, kernels)?;
    let spatial = crate::spatial::contains(program);
    if spatial || crate::sorting::contains(program) {
        // ponytail: unmeasured spatial families stay on CPU; admit devices only
        // after per-operation, revision-matched hardware/baseline qualification.
        let workers = selected_workers()?;
        // Reached float reductions take the same auto choice as in a
        // program without intrinsics; only a bound program with one calibrates.
        let (calibration, device) = match reductions::has_reduction(program) {
            true if crate::sorting::binds(source, program) => {
                reduce::auto_device(source, kernels, model, workers)?
            }
            _ => (None, None),
        };
        let (device, mut receipts, hooks) = (RefCell::new(device), Vec::new(), Hooks::default());
        let mut reducer = |call: &mir_eval::ReduceCall<'_>| {
            let mut device = device.borrow_mut();
            reduce::auto(call, device.as_mut(), calibration, workers, &mut receipts)
        };
        let cpu = if spatial {
            crate::spatial::run_cpu_with(source, program, fuel, workers, &hooks, &mut reducer)
        } else {
            crate::sorting::run_cpu_with(source, program, fuel, workers, &mut reducer)
        }
        .map_err(|e| Error::mismatch(format!("intrinsic CPU fallback: {e:?}")))?;
        return Ok(super::device::intrinsic_cpu_auto(cpu, receipts));
    }
    let regions = families::analyze(source, program)
        .map_err(|error| Error::mismatch(format!("admission failed: {error:?}")))?;
    let workers = selected_workers()?;
    let (calibration, device) = reduce::auto_device(source, kernels, model, workers)?;
    let device = RefCell::new(device);
    let mut receipts = Vec::new();
    let mut reducer = |call: &mir_eval::ReduceCall<'_>| {
        let mut device = device.borrow_mut();
        reduce::auto(call, device.as_mut(), calibration, workers, &mut receipts)
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
            calibration.and_then(|model| region.metal_cost(live, device.capabilities(), model))
        });
        let mut record = device.as_ref().and_then(|device| {
            calibration.and_then(|model| region.cost_record(live, device.capabilities(), model))
        });
        let estimate = record.as_ref().map(|record| record.estimate).or(estimate);
        let use_device = !is_consumer
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
        } else if estimate.is_none() {
            region.metal_ineligible_reason(live.iterations)
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
                code: "METAL_UNAVAILABLE",
                message: "device disappeared before submission".into(),
            })?;
            let fault = std::mem::replace(&mut inject, Fault::None);
            match region.metal_step(source, live, &mut pending, active, fault) {
                Ok((step, cache_hit, hint)) => {
                    backend = "metal";
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
                    reason = "metal-failed-before-publication";
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
            if backend == "metal"
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
                reason: if reason == "metal-failed-before-publication" {
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
    super::device::settle_choices(&mut choices, &published);
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

/// Measure dispatch, shared-memory copy and CPU map rates. Successful models
/// are cached per CPU worker count so placement uses the current CPU width.
pub fn calibrate(
    open: &mut dyn FnMut() -> Result<MetalDevice, Error>,
) -> Result<DeviceCostModel, Error> {
    calibrate_with_workers(selected_workers()?, open)
}

type ModelCache = [std::sync::OnceLock<DeviceCostModel>; MAX_WORKERS];
static MODELS: ModelCache = [const { std::sync::OnceLock::new() }; MAX_WORKERS];

fn calibrate_with_workers(
    workers: usize,
    open: &mut dyn FnMut() -> Result<MetalDevice, Error>,
) -> Result<DeviceCostModel, Error> {
    calibrate_cached(&MODELS, workers, &mut |workers| {
        measure_model(workers, open()?)
    })
}

/// The per-width cache around one measurement, which tests replace.
fn calibrate_cached(
    cache: &ModelCache,
    workers: usize,
    measure: &mut dyn FnMut(usize) -> Result<DeviceCostModel, Error>,
) -> Result<DeviceCostModel, Error> {
    let slot = &cache[workers - 1];
    if let Some(model) = slot.get() {
        return Ok(*model);
    }
    let model = measure(workers)?;
    Ok(*slot.get_or_init(|| model))
}

#[cfg(test)]
mod worker_calibration_tests {
    use super::*;

    /// A deterministic model that differs per width, standing in for the timed
    /// `measure_model`: the property under test is the cache, not Metal timing.
    fn model(workers: usize) -> DeviceCostModel {
        DeviceCostModel {
            cpu_bytes_per_ms: 1.0e6 * workers as f64,
            device_bytes_per_ms: 4.0e6,
            device_fixed_ms: 0.05,
        }
    }

    #[test]
    fn cpu_worker_calibration_cache_is_per_width() {
        let cache: ModelCache = [const { std::sync::OnceLock::new() }; MAX_WORKERS];
        let mut calls = Vec::new();
        let mut measure = |workers: usize| {
            calls.push(workers);
            Ok(model(workers))
        };
        let widest = MAX_WORKERS;
        let one = calibrate_cached(&cache, 1, &mut measure).unwrap();
        let many = calibrate_cached(&cache, widest, &mut measure).unwrap();
        assert!(one.cpu_bytes_per_ms > 0.0 && many.cpu_bytes_per_ms > 0.0);
        assert_ne!(one, many, "different widths must measure separate models");
        assert_eq!(one, calibrate_cached(&cache, 1, &mut measure).unwrap());
        assert_eq!(
            many,
            calibrate_cached(&cache, widest, &mut measure).unwrap()
        );
        assert_eq!(
            calls,
            [1, widest],
            "each width is measured once, then cached"
        );
    }

    #[test]
    fn a_failed_calibration_is_not_cached() {
        let cache: ModelCache = [const { std::sync::OnceLock::new() }; MAX_WORKERS];
        let attempts = std::cell::Cell::new(0);
        let mut measure = |workers: usize| {
            attempts.set(attempts.get() + 1);
            if attempts.get() == 1 {
                Err(Error::unavailable("noisy timing sample"))
            } else {
                Ok(model(workers))
            }
        };
        let error = calibrate_cached(&cache, 2, &mut measure).unwrap_err();
        assert_eq!(error.code, "METAL_UNAVAILABLE");
        assert_eq!(calibrate_cached(&cache, 2, &mut measure).unwrap(), model(2));
        assert_eq!(attempts.get(), 2);
        assert_eq!(calibrate_cached(&cache, 2, &mut measure).unwrap(), model(2));
        assert_eq!(attempts.get(), 2, "a successful model is cached");
    }
}

fn measure_model(workers: usize, mut device: MetalDevice) -> Result<DeviceCostModel, Error> {
    const SMALL: usize = 64 * 1024;
    const LARGE: usize = 8 * 1024 * 1024;
    const TRIALS: usize = 6;
    // Median of five after one warm trial, which also creates the probe pipeline.
    let mut median = |measure: &mut dyn FnMut(
        &mut MetalDevice,
    ) -> Result<f64, invar_metal::Error>|
     -> Result<f64, Error> {
        let mut samples = Vec::with_capacity(TRIALS);
        for trial in 0..TRIALS {
            let elapsed = measure(&mut device)
                .map_err(|error| Error::from_metal(error, device.is_poisoned()))?;
            if trial > 0 {
                samples.push(elapsed);
            }
        }
        samples.sort_by(f64::total_cmp);
        Ok(samples[samples.len() / 2])
    };
    let dispatch_ms = median(&mut |device| device.measure_dispatch())?;
    let small_ms = median(&mut |device| device.measure_roundtrip(SMALL))?;
    let large_ms = median(&mut |device| device.measure_roundtrip(LARGE))?;
    let delta_ms = large_ms - small_ms;
    if !delta_ms.is_finite() || delta_ms <= 0.0 || !dispatch_ms.is_finite() || dispatch_ms <= 0.0 {
        return Err(Error::unavailable(
            "Metal calibration did not resolve dispatch latency or bandwidth",
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
        // A real submit-to-completion of a trivial kernel, not a host copy.
        device_fixed_ms: dispatch_ms,
    })
}

#[cfg(test)]
mod ai_01_fault_tests {
    #[test]
    fn issue_ai_01_machine_diagnostics_metal_backend_faults() {
        let schema: serde_json::Value = serde_json::from_str(include_str!(
            "../../schema/invar-diagnostics-v3.schema.json"
        ))
        .unwrap();
        for (error, poisoned, code) in [
            (invar_metal::Error::Pending, true, "METAL_PENDING"),
            (
                invar_metal::Error::Toolchain("injected compiler failure".into()),
                false,
                "DEVICE_TOOLCHAIN_MISMATCH",
            ),
        ] {
            let error = super::Error::from_metal(error, poisoned);
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

#[cfg(test)]
mod prebuilt_binding_tests {
    //! Auto checks a supplied library against the program before its
    //! CPU-only sort and spatial branch, which never opens the device.
    use super::*;
    use crate::metal_aot::{self, Header, Library, Route, Toolchain};
    use sha2::Digest;

    const SORT: &str = "fn main() -> i32 { var xs: Array<u32> = [2u32,1u32]; sort(xs); if xs[0u32] == 1u32 { return 7; } return 0; }";

    fn lowered(source: &str) -> mir::Program {
        let program = crate::front::parse(source.as_bytes()).unwrap();
        mir::lower(&crate::check::check(program, true).unwrap()).unwrap()
    }

    /// A well-formed container bound to `source`, with one synthetic library.
    fn image_for(source: &[u8]) -> metal_aot::Image {
        let mut metallib = vec![0u8; 64];
        metallib[..4].copy_from_slice(b"MTLB");
        metallib[4..12].copy_from_slice(&[0x01, 0x80, 0x02, 0x00, 0x07, 0x00, 0x00, 0x81]);
        metallib[12..14].copy_from_slice(&14u16.to_le_bytes());
        metallib[16..24].copy_from_slice(&64u64.to_le_bytes());
        let msl = "kernel void a_v1(device uint* o [[buffer(0)]]) { o[0] = 1u; }";
        let header = Header {
            route: Route::Run,
            entry: "main".into(),
            source_sha256: sha2::Sha256::digest(source).into(),
            toolchain: Toolchain {
                metal: "Apple metal version 32023.921 (metalfe-32023.921.6)".into(),
                metallib: "AIR-LLD 32023.921 (metalfe-32023.921.6)".into(),
                sdk: "27.0 (26A425)".into(),
            },
            invar: "0.1.0".into(),
            libraries: vec![Library {
                msl_sha256: sha2::Sha256::digest(msl.as_bytes()).into(),
                functions: vec!["a_v1".into()],
                bytes: metallib.len(),
                sha256: sha2::Sha256::digest(&metallib).into(),
            }],
        };
        metal_aot::decode(&metal_aot::encode(&header, &[metallib]).unwrap()).unwrap()
    }

    #[test]
    fn auto_refuses_another_programs_library_before_the_cpu_only_sort_branch() {
        let program = lowered(SORT);
        assert!(crate::sorting::contains(&program));
        // Without a library the sort runs on the CPU, as before.
        let cpu = run_auto_source(SORT.as_bytes(), &program, 1_000_000, Kernels::Missing).unwrap();
        assert!(
            cpu.choices
                .iter()
                .all(|choice| choice.backend == "cpu-parallel")
        );
        let stale = image_for(b"fn main() -> i32 { return 0; }");
        let error = run_auto_source(
            SORT.as_bytes(),
            &program,
            1_000_000,
            Kernels::Prebuilt(&stale),
        )
        .err()
        .expect("a library built from another program is refused");
        assert_eq!(error.code, "DEVICE_TOOLCHAIN_MISMATCH");
        assert!(error.message.contains("source_sha256"), "{}", error.message);
        // This program's own library passes the check; the sort still runs on the CPU.
        let own = image_for(SORT.as_bytes());
        run_auto_source(
            SORT.as_bytes(),
            &program,
            1_000_000,
            Kernels::Prebuilt(&own),
        )
        .unwrap();
    }
}
