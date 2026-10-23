//! Profile-driven placement at reached loop heads. Unmeasured work stays serial.
use super::*;
#[cfg(any(feature = "cuda", all(feature = "metal", target_os = "macos")))]
use crate::placement::Engine;
use crate::placement::{self, Facts, PageState, Profile};
use serde_json::{Value as Json, json};
use std::path::Path;

pub struct Run {
    pub observation: Observation,
    pub choices: Vec<Json>,
}

/// Resource identity and plan verification remain independent of cost hints.
pub struct Resources<'a> {
    pub compiler: &'a Path,
    #[cfg(all(feature = "metal", target_os = "macos"))]
    pub kernels: metal::Kernels<'a>,
}

pub fn run(
    source: &[u8],
    program: &mir::Program,
    fuel: u64,
    profile: Option<&Profile>,
    trust: bool,
    resources: Resources<'_>,
) -> Result<Run, EvalError> {
    let _ = &resources;
    let report = super::regions(source, program).map_err(|e| EvalError::Tool {
        code: e.code,
        message: e.message,
    })?;
    #[cfg(any(feature = "cuda", all(feature = "metal", target_os = "macos")))]
    let regions = families::analyze(source, program).map_err(|e| EvalError::Tool {
        code: e.code,
        message: e.message,
    })?;
    #[cfg(feature = "cuda")]
    let mut cuda = if profile.is_some() && trust {
        invar_cuda::cuda::CudaDevice::open().ok()
    } else {
        None
    };
    #[cfg(all(feature = "metal", target_os = "macos"))]
    let mut metal = if profile.is_some() && trust {
        match metal::open_device(source, resources.kernels) {
            Ok(d) => Some(d),
            Err(e) if e.code == "DEVICE_TOOLCHAIN_MISMATCH" => {
                return Err(EvalError::Tool {
                    code: e.code,
                    message: e.message,
                });
            }
            Err(_) => None,
        }
    } else {
        None
    };
    let mut choices = Vec::new();
    #[cfg(any(feature = "cuda", all(feature = "metal", target_os = "macos")))]
    let mut pending = VecDeque::new();
    #[cfg(feature = "native")]
    let mut cpu = None;
    let mut provider = |live: &LiveLoop<'_>| -> Result<Option<LoopOverride>, EvalError> {
        let Some(region) = report
            .regions()
            .iter()
            .find(|r| r.function() == live.function && r.span() == live.span)
        else {
            return Ok(None);
        };
        let raw = u32::try_from(live.iterations)
            .ok()
            .and_then(|n| crate::device_ir::from_region(region, n).ok());
        // Never reuse the probe's coefficients for a different law or fusion.
        let map = match (region.execution(), raw.as_ref()) {
            (Execution::Map(m), Some(ir))
                if ir.ops == [invar_device::Op::WrappingAdd(1)]
                    && ir.dag.is_none()
                    && ir.collective.is_none()
                    && ir.numeric == invar_device::Numeric::WrappingU32
                    && m.inputs.len() == 1 =>
            {
                Some(m)
            }
            _ => None,
        };
        let input = map.and_then(|m| live.array(&m.inputs[0]));
        let guard = map.is_some_and(|m| {
            input.as_ref().is_some_and(|a| a.len() >= live.iterations)
                && live
                    .lengths
                    .get(&m.output)
                    .is_some_and(|n| *n >= live.iterations)
                && !live.same_object(&m.inputs[0], &m.output)
        });
        #[allow(unused_mut)]
        let mut caps = None;
        #[allow(unused_mut)]
        let mut capacity = 0;
        #[allow(unused_mut)]
        let mut cuda_legal = false;
        #[allow(unused_mut)]
        let mut metal_legal = false;
        #[cfg(feature = "cuda")]
        if let (Some(ir), Some(d)) = (raw.as_ref(), cuda.as_ref()) {
            cuda_legal = guard
                && resources.compiler.is_file()
                && verified(ir.clone(), d.capabilities()).is_some();
            caps = Some(d.capabilities());
            capacity = d.available_capacity().unwrap_or(0);
        }
        #[cfg(all(feature = "metal", target_os = "macos"))]
        if let (Some(ir), Some(d)) = (raw.as_ref(), metal.as_ref()) {
            metal_legal = guard && verified(ir.clone(), d.capabilities()).is_some();
            caps = Some(d.capabilities());
            capacity = d.available_capacity();
        }
        let identity = raw.as_ref().map(|ir| json!({"source_identity":ir.source_identity,"graph_revision":ir.graph_revision,"region_identity":region.identity()}));
        let physical = raw
            .as_ref()
            .and_then(|ir| caps.and_then(|c| verified(ir.clone(), c)))
            .and_then(|p| p.cost_facts());
        let bytes = live.iterations.saturating_mul(4);
        let facts = Facts {
            family: if map.is_some() {
                placement::MAP_FAMILY
            } else {
                "unmeasured"
            },
            numeric: if map.is_some() {
                "u32-wrapping"
            } else {
                "unknown"
            },
            numeric_contract_version: if map.is_some() {
                placement::MAP_NUMERIC_CONTRACT
            } else {
                "unknown"
            },
            cpu_boundary: if map.is_some() && cfg!(feature = "native") {
                "llvm-host-array-v1"
            } else {
                "mir-serial-v1"
            },
            device_boundary: "host-map-v1",
            region: identity.as_ref(),
            work: live.iterations,
            input_bytes: physical.map_or(bytes, |p| p.input_bytes),
            output_bytes: physical.map_or(bytes, |p| p.output_bytes),
            temporary_bytes: physical.map_or(0, |p| p.temporary_bytes),
            launches: physical.map_or(1, |p| p.launches),
            // The backend checks live device memory again before allocation.
            capacity_bytes: capacity.min(invar_device::MAX_GNET_STORAGE_BYTES),
            residency: tuning::Residency::HOST,
            pages: input.as_ref().map_or(PageState::Unknown, |a| {
                invar_cuda::pages::observe(&a[..live.iterations.min(a.len())])
            }),
            cuda_legal: guard && cuda_legal && cfg!(feature = "native"),
            metal_legal: guard && metal_legal && cfg!(feature = "native"),
        };
        let decision = placement::select(profile, trust, &facts);
        let mut receipt = decision.receipt;
        receipt["advisory_engine"] = receipt["engine"].clone();
        receipt["function"] = json!(live.function);
        receipt["span"] = json!({"start":live.span.start,"end":live.span.end});
        receipt["published"] = json!(false);
        #[cfg(any(feature = "cuda", all(feature = "metal", target_os = "macos")))]
        let selected = regions
            .iter()
            .find(|r| r.function() == live.function && r.span() == live.span);
        #[allow(unused_mut)]
        let mut step = None;
        #[cfg(any(feature = "cuda", all(feature = "metal", target_os = "macos")))]
        if let Some(selected) = selected {
            #[cfg(feature = "cuda")]
            if decision.engine == Engine::Cuda
                && let Some(d) = cuda.as_mut()
            {
                match selected.cuda_step(
                    source,
                    live,
                    &mut pending,
                    d,
                    resources.compiler,
                    cuda::Fault::None,
                ) {
                    Ok((s, _, _)) => step = Some(s.plan),
                    Err(e) if d.is_poisoned() => {
                        return Err(EvalError::Tool {
                            code: e.code,
                            message: e.message,
                        });
                    }
                    Err(_) => receipt["execution_refusal"] = json!("TUNING_LAUNCH_FAILED"),
                }
            }
            #[cfg(all(feature = "metal", target_os = "macos"))]
            if decision.engine == Engine::Metal
                && let Some(d) = metal.as_mut()
            {
                match selected.metal_step(source, live, &mut pending, d, metal::Fault::None) {
                    Ok((s, _, _)) => step = Some(s.plan),
                    Err(e) if d.is_poisoned() => {
                        return Err(EvalError::Tool {
                            code: e.code,
                            message: e.message,
                        });
                    }
                    Err(e) if e.code == "DEVICE_TOOLCHAIN_MISMATCH" => {
                        return Err(EvalError::Tool {
                            code: e.code,
                            message: e.message,
                        });
                    }
                    Err(_) => receipt["execution_refusal"] = json!("TUNING_LAUNCH_FAILED"),
                }
            }
        }
        if step.is_none() {
            receipt["engine"] = json!("cpu");
            #[cfg(feature = "native")]
            if guard && let (Some(m), Some(input)) = (map, input.as_ref()) {
                use crate::eval::host_read::{
                    HostArgument as A, HostMutArray as M, HostReadArray as R,
                };
                use crate::exports::{ExportEngine, compile_exports};
                if cpu.is_none() {
                    cpu = Some(
                        compile_exports(
                            placement::calibration::MAP,
                            "placement-map.inv",
                            &["map"],
                            ExportEngine::NativeSerial,
                        )
                        .map_err(|e| EvalError::Tool {
                            code: "NATIVE_BACKEND",
                            message: format!("{e:?}"),
                        })?,
                    );
                }
                let mut values = vec![0; live.iterations];
                if let Some(lib) = cpu.as_mut() {
                    lib.invoke_arrays(
                        "map",
                        &mut [
                            A::Read(R::U32(&input[..live.iterations])),
                            A::Mut(M::U32(&mut values)),
                        ],
                        100_000_000,
                    )
                    .map_err(|e| EvalError::Tool {
                        code: "NATIVE_BACKEND",
                        message: format!("{e:?}"),
                    })?;
                }
                step = Some(LoopOverride {
                    iterations: live.iterations,
                    input_lengths: vec![(m.inputs[0].clone(), input.len())],
                    array: Some((m.output.clone(), values)),
                    ..Default::default()
                });
            }
        }
        receipt["offered"] = json!(step.is_some());
        choices.push(receipt);
        Ok(step)
    };
    let (observation, published) = mir_eval::observe_offers(program, fuel, Some(&mut provider));
    let mut outcomes = published.iter();
    for choice in &mut choices {
        if choice["offered"] == true {
            let accepted = outcomes.next() == Some(&true);
            choice["published"] = json!(accepted);
            if !accepted {
                choice["engine"] = json!("cpu");
                choice["execution_refusal"] = json!("TUNING_GUARD_FALSE");
            }
        }
    }
    Ok(Run {
        observation,
        choices,
    })
}

fn verified(
    mut ir: invar_device::Ir,
    caps: invar_device::Capabilities,
) -> Option<invar_device::Verified> {
    if matches!(caps, invar_device::Capabilities::Metal { .. }) {
        crate::device_ir::plan_metal(&mut ir).ok()?;
    }
    let origin = invar_device::Provenance {
        source_identity: ir.source_identity,
        graph_revision: ir.graph_revision,
    };
    invar_device::verify(ir, origin, caps).ok()
}
