//! Shared CLI/Python engine names and guarded whole-program dispatch.
use crate::eval::{EvalError, Value};

/// Engine names are exactly the names accepted by `invar run --engine`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Engine {
    Reference,
    NativeSerial,
    CpuParallel,
    Auto,
    Cuda,
    Metal,
}

impl Engine {
    pub fn parse(name: &str) -> Option<Self> {
        Some(match name {
            "reference" => Self::Reference,
            "native-serial" => Self::NativeSerial,
            "cpu-parallel" => Self::CpuParallel,
            "auto" => Self::Auto,
            "cuda" => Self::Cuda,
            "metal" => Self::Metal,
            _ => return None,
        })
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::Reference => "reference",
            Self::NativeSerial => "native-serial",
            Self::CpuParallel => "cpu-parallel",
            Self::Auto => "auto",
            Self::Cuda => "cuda",
            Self::Metal => "metal",
        }
    }
}

#[cfg(not(feature = "cuda"))]
fn auto_workers() -> Result<usize, EvalError> {
    crate::schedule::auto_workers().map_err(|error| EvalError::Tool {
        code: error.code(),
        message: format!("CPU worker selection failed: {error:?}"),
    })
}

/// Run the CLI's guarded MIR scheduler once and return its execution receipt.
/// Reference/native artifacts have separate entry points and are refused here.
/// The CPU scheduler retains its documented serial guard path; `fallback_reason`
/// discloses it. Callers that require a published parallel result must refuse it.
/// `metal_artifact` names a compiled-ahead Metal library; without one, Metal
/// compiles runtime source only under the developer flag.
pub fn run_mir(
    source: &[u8],
    program: &crate::mir::Program,
    engine: Engine,
    fuel: u64,
    workers: usize,
    metal_artifact: Option<&std::path::Path>,
) -> Result<serde_json::Value, EvalError> {
    let mut receipt = match engine {
        Engine::CpuParallel => run_cpu(
            source,
            program,
            engine.name(),
            fuel,
            workers,
            "requested-cpu",
        ),
        Engine::Metal => run_metal(source, program, engine.name(), fuel, metal_artifact),
        Engine::Auto if cfg!(all(feature = "metal", target_os = "macos")) => {
            run_metal(source, program, engine.name(), fuel, metal_artifact)
        }
        Engine::Auto | Engine::Cuda => run_cuda(source, program, engine.name(), fuel),
        _ => Err(EvalError::Tool {
            code: "NATIVE_ABI",
            message: "reference and native-serial use their own execution entry points".into(),
        }),
    }?;
    // No published region means that the ordinary MIR evaluator executed main.
    let mut backends = receipt["choices"]
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(|choice| choice["backend"].as_str())
        .collect::<Vec<_>>();
    backends.sort_unstable();
    backends.dedup();
    if receipt["normal_form"]["status"] == "routed" {
        backends.push("normal-form");
    } else if backends.is_empty() {
        backends.push("mir-serial");
    }
    receipt["backends"] = serde_json::json!(backends);
    Ok(receipt)
}

#[cfg(feature = "cuda")]
fn run_cuda(
    source: &[u8],
    program: &crate::mir::Program,
    engine: &str,
    fuel: u64,
) -> Result<serde_json::Value, EvalError> {
    let compiler = nvrtc_helper();
    if engine == "cuda" {
        let run = crate::schedule::cuda::run_source(source, program, fuel, &compiler).map_err(
            |error| EvalError::Tool {
                code: error.code,
                message: error.message,
            },
        )?;
        let value = observed_i32(run.observation)?;
        let mut choices = run
            .executed
            .iter()
            .map(|step| step_choice(step, "cuda", "forced-device"))
            .collect::<Vec<_>>();
        choices.extend(run.reductions.iter().map(reduction_choice));
        Ok(serde_json::json!({
            "requested_engine": "cuda",
            "choices": choices,
            "result": value,
            "module_cache_hits": run.cache_hits,
            "schedule_hint_hits": run.hint_hits,
            "schedule_hint_invalids": run.hint_invalids,
        }))
    } else {
        let run = crate::schedule::cuda::run_auto_source(source, program, fuel, &compiler)
            .map_err(|error| EvalError::Tool {
                code: error.code,
                message: error.message,
            })?;
        let value = observed_i32(run.observation)?;
        let mut choices = run
            .choices
            .iter()
            .map(|choice| {
                serde_json::json!({
                    "function": choice.function,
                    "span": {"start": choice.span.start, "end": choice.span.end},
                    "backend": choice.backend,
                    "reason": choice.reason,
                    "cost_ms": choice.cost.map(|cost| serde_json::json!({
                        "cpu": cost.cpu_ms, "device": cost.device_ms,
                    })),
                })
            })
            .collect::<Vec<_>>();
        annotate_sort_implementations(&mut choices, &run.executed);
        choices.extend(run.reductions.iter().map(reduction_choice));
        Ok(serde_json::json!({
            "requested_engine": "auto",
            "choices": choices,
            "result": value,
            "module_cache_hits": run.cache_hits,
            "schedule_hint_hits": run.hint_hits,
            "schedule_hint_invalids": run.hint_invalids,
        }))
    }
}

#[cfg(not(feature = "cuda"))]
fn run_cuda(
    source: &[u8],
    program: &crate::mir::Program,
    engine: &str,
    fuel: u64,
) -> Result<serde_json::Value, EvalError> {
    if engine == "cuda" {
        return Err(EvalError::Tool {
            code: "CUDA_UNAVAILABLE",
            message: "CUDA engine requires the cuda Cargo feature and a compatible CUDA device"
                .into(),
        });
    }
    run_cpu(
        source,
        program,
        engine,
        fuel,
        auto_workers()?,
        "cuda-feature-unavailable",
    )
}

/// The compiled-ahead container named by `artifact`, read privately and
/// validated.
#[cfg(all(feature = "metal", target_os = "macos"))]
pub fn metal_image(
    artifact: Option<&std::path::Path>,
) -> Result<Option<crate::metal_aot::Image>, EvalError> {
    artifact
        .map(|file| {
            crate::metal_aot::build::read(file).map_err(|e| EvalError::Tool {
                code: "DEVICE_TOOLCHAIN_MISMATCH",
                message: format!("compiled-ahead Metal library refused: {e:?}"),
            })
        })
        .transpose()
}

/// The kernels a Metal run uses: the compiled-ahead image when one is given,
/// otherwise the developer source route.
#[cfg(all(feature = "metal", target_os = "macos"))]
pub fn metal_kernels(
    image: Option<&crate::metal_aot::Image>,
) -> crate::schedule::metal::Kernels<'_> {
    image.map_or_else(
        crate::schedule::metal::Kernels::developer,
        crate::schedule::metal::Kernels::Prebuilt,
    )
}

#[cfg(all(feature = "metal", target_os = "macos"))]
fn run_metal(
    source: &[u8],
    program: &crate::mir::Program,
    engine: &str,
    fuel: u64,
    artifact: Option<&std::path::Path>,
) -> Result<serde_json::Value, EvalError> {
    let image = metal_image(artifact)?;
    let kernels = metal_kernels(image.as_ref());
    if engine == "metal" {
        let run = crate::schedule::metal::run_source(source, program, fuel, kernels).map_err(
            |error| EvalError::Tool {
                code: error.code,
                message: error.message,
            },
        )?;
        let value = observed_i32(run.observation)?;
        let mut choices = metal_no_f64_choices(program);
        choices.extend(
            run.executed
                .iter()
                .map(|step| step_choice(step, "metal", "forced-device")),
        );
        choices.extend(run.reductions.iter().map(reduction_choice));
        Ok(serde_json::json!({
            "requested_engine": "metal", "choices": choices, "result": value,
            "module_cache_hits": run.cache_hits,
            "schedule_hint_hits": run.hint_hits,
            "schedule_hint_invalids": run.hint_invalids,
        }))
    } else {
        let run = crate::schedule::metal::run_auto_source(source, program, fuel, kernels).map_err(
            |error| EvalError::Tool {
                code: error.code,
                message: error.message,
            },
        )?;
        let value = observed_i32(run.observation)?;
        let mut choices = metal_no_f64_choices(program);
        // The metal-no-f64 entries come first; the scheduler's choices pair with run.executed by position after them.
        let scheduled = choices.len();
        choices.extend(run.choices.iter().map(|choice| serde_json::json!({
            "function": choice.function,
            "span": {"start": choice.span.start, "end": choice.span.end},
            "backend": choice.backend, "reason": choice.reason,
            "cost_ms": choice.cost.map(|cost| serde_json::json!({"cpu": cost.cpu_ms, "device": cost.device_ms})),
        })));
        annotate_sort_implementations(&mut choices[scheduled..], &run.executed);
        choices.extend(run.reductions.iter().map(reduction_choice));
        Ok(serde_json::json!({
            "requested_engine": engine, "choices": choices, "result": value,
            "module_cache_hits": run.cache_hits,
            "schedule_hint_hits": run.hint_hits,
            "schedule_hint_invalids": run.hint_invalids,
        }))
    }
}

#[cfg(not(all(feature = "metal", target_os = "macos")))]
fn run_metal(
    _: &[u8],
    _: &crate::mir::Program,
    _: &str,
    _: u64,
    _: Option<&std::path::Path>,
) -> Result<serde_json::Value, EvalError> {
    Err(EvalError::Tool {
        code: "METAL_UNAVAILABLE",
        message: "Metal engine requires the metal Cargo feature and an Apple7+ GPU on macOS".into(),
    })
}

fn observed_i32(observation: crate::eval::Observation) -> Result<i32, EvalError> {
    match observation.result {
        Ok(Value::I32(value)) => Ok(value),
        Ok(_) => Err(EvalError::Tool {
            code: "INTERNAL_MAIN",
            message: "checked main returned non-i32".into(),
        }),
        Err(error) => Err(error),
    }
}

fn run_cpu(
    source: &[u8],
    program: &crate::mir::Program,
    engine: &str,
    fuel: u64,
    workers: usize,
    reason: &str,
) -> Result<serde_json::Value, EvalError> {
    use crate::schedule::normal_form;
    // An invalid worker count reaches `run_source`, which rejects it.
    let route = ((1..=crate::schedule::MAX_WORKERS).contains(&workers)
        && crate::schedule::serial_only(source, program))
    .then(|| normal_form::route(program, fuel));
    let route = match route {
        Some(normal_form::Route::Routed(routed)) => {
            let receipt = routed.document();
            let value = routed.result?;
            return Ok(serde_json::json!({
                "requested_engine": engine, "choices": [], "result": value,
                "fallback_reason": null,
                "module_cache_hits": 0, "schedule_hint_hits": 0, "schedule_hint_invalids": 0,
                "normal_form": receipt,
            }));
        }
        other => other,
    };
    let run = crate::schedule::run_source(source, program, fuel, workers).map_err(|error| {
        EvalError::Tool {
            code: error.code(),
            message: format!("CPU execution failed: {error:?}"),
        }
    })?;
    let value = observed_i32(run.observation)?;
    let mut choices = run
        .executed
        .iter()
        .map(|step| step_choice(step, "cpu-parallel", reason))
        .collect::<Vec<_>>();
    choices.extend(run.reductions.iter().map(|receipt| {
        let mut choice = reduction_choice(receipt);
        // The CPU cost model's own refusal to split stays visible.
        if receipt.backend != crate::schedule::reductions::SERIAL_BACKEND {
            choice["reason"] = serde_json::json!(reason);
        }
        choice
    }));
    let mut receipt = serde_json::json!({
        "requested_engine": engine, "choices": choices, "result": value,
        "fallback_reason": run.fallback_reason,
        "module_cache_hits": 0, "schedule_hint_hits": 0, "schedule_hint_invalids": 0,
    });
    if let Some(document) = route.as_ref().and_then(normal_form::Route::document) {
        receipt["normal_form"] = document;
    } else if let Some(gap) = (!crate::sorting::contains(program)
        && !crate::spatial::contains(program))
    .then(|| crate::schedule::selection(source, program, &Default::default()).ok())
    .flatten()
    .and_then(|dispatched| dispatched.gaps.into_iter().next())
    {
        // A certified loop no family lowers ran on this path (#54). Sort and
        // spatial programs explain under v3/v4, which have no `normal_form`.
        receipt["normal_form"] = serde_json::json!({"status": "fallback", "reason": gap.reason});
    }
    Ok(receipt)
}

#[cfg(feature = "cuda")]
/// Resolve the bounded NVRTC helper from the environment or executable sibling.
pub fn nvrtc_helper() -> std::path::PathBuf {
    std::env::var_os("INVAR_NVRTC_HELPER")
        .map(std::path::PathBuf::from)
        .or_else(|| {
            std::env::current_exe().ok().map(|path| {
                path.with_file_name(format!("invar-nvrtc{}", std::env::consts::EXE_SUFFIX))
            })
        })
        .unwrap_or_default()
}

/// Run the existing native executable protocol, with source-bound diagnostics.
pub fn run_native(
    program: &crate::mir::Program,
    fuel: u64,
    path: &str,
    source: &[u8],
    library: Option<&std::path::Path>,
) -> Result<(i32, serde_json::Value), EvalError> {
    run_native_with(program, fuel, path, source, library, false)
}

/// [`run_native`], with DWARF in the executable when `debug` is set (DX-20).
#[cfg(feature = "native")]
pub fn run_native_with(
    program: &crate::mir::Program,
    fuel: u64,
    path: &str,
    source: &[u8],
    library: Option<&std::path::Path>,
    debug: bool,
) -> Result<(i32, serde_json::Value), EvalError> {
    let ran = if debug {
        // DWARF names the source by absolute path (DX-20).
        std::path::absolute(path)
            .map_err(|failure| failure.to_string())
            .and_then(|absolute| {
                let source = crate::native::DebugSource {
                    path: &absolute,
                    text: source,
                };
                crate::native::run_program_debug(program, fuel, library, &source)
            })
    } else {
        crate::native::run_program_with_library(program, fuel, library)
    };
    let (status, mut document) = ran.map_err(|message| EvalError::Tool {
        code: "NATIVE_BACKEND",
        message,
    })?;
    annotate_native_document(&mut document, path, source);
    document["schema"] = serde_json::json!(if document["schema_version"] == 2 {
        "invar-diagnostics-v2"
    } else {
        crate::diagnostics::SCHEMA
    });
    Ok((status, document))
}

#[cfg(not(feature = "native"))]
pub fn run_native_with(
    _: &crate::mir::Program,
    _: u64,
    _: &str,
    _: &[u8],
    _: Option<&std::path::Path>,
    _: bool,
) -> Result<(i32, serde_json::Value), EvalError> {
    Err(EvalError::Tool {
        code: "NATIVE_UNAVAILABLE",
        message: "native engine requires the native Cargo feature and LLVM 21".into(),
    })
}

#[cfg(feature = "native")]
fn annotate_native_document(document: &mut serde_json::Value, path: &str, source: &[u8]) {
    if document["status"] == "tool_error" {
        let code = document["tool_code"].as_str().unwrap_or("NATIVE_TOOL");
        document["diagnostics"] = serde_json::json!([crate::diagnostics::diagnostic(
            code,
            "native runtime tool error",
            path,
            source,
            None,
        )]);
    } else if document["status"] == "trap" {
        let code = document["trap"]["category"]
            .as_str()
            .unwrap_or("NATIVE_TOOL");
        let span = document["trap"]["span"]["start"]
            .as_u64()
            .zip(document["trap"]["span"]["end"].as_u64())
            .and_then(|(start, end)| {
                Some(crate::front::Span {
                    start: usize::try_from(start).ok()?,
                    end: usize::try_from(end).ok()?,
                })
            });
        document["diagnostics"] = serde_json::json!([crate::diagnostics::diagnostic(
            code,
            "native execution trapped",
            path,
            source,
            span,
        )]);
    } else if document["status"] == "evaluation_limit" {
        document["diagnostics"] = serde_json::json!([crate::diagnostics::diagnostic(
            "EVALUATION_LIMIT",
            "evaluation fuel limit exceeded",
            path,
            source,
            None,
        )]);
    }
}

/// Metal has no binary64: each function computing f64 stays on the CPU
/// (`mir-serial`) with reason `metal-no-f64`, never emulated.
#[cfg(all(feature = "metal", target_os = "macos"))]
fn metal_no_f64_choices(program: &crate::mir::Program) -> Vec<serde_json::Value> {
    crate::math::f64_sites(program)
        .into_iter()
        .map(|(function, span)| {
            serde_json::json!({
                "function": function,
                "span": {"start": span.start, "end": span.end},
                "backend": "mir-serial", "reason": crate::math::METAL_NO_F64,
            })
        })
        .collect()
}

/// One published region as an explain device choice; sort sites name their implementation.
fn step_choice(step: &crate::schedule::Executed, backend: &str, reason: &str) -> serde_json::Value {
    let mut choice = serde_json::json!({
        "function": step.function,
        "span": {"start": step.span.start, "end": step.span.end},
        "backend": backend,
        "kind": format!("{:?}", step.kind),
        "reason": reason,
    });
    if let Some(implementation) = step.implementation {
        choice["implementation"] = serde_json::json!(implementation);
    }
    choice
}

/// Auto choices pair with the executed regions by position; a sort site kept on
/// the CPU records the implementation that ran it.
#[cfg(any(feature = "cuda", all(feature = "metal", target_os = "macos")))]
fn annotate_sort_implementations(
    choices: &mut [serde_json::Value],
    executed: &[crate::schedule::Executed],
) {
    for (choice, step) in choices.iter_mut().zip(executed) {
        if choice["reason"] == "sort-cost-unmeasured"
            && let Some(implementation) = step.implementation
        {
            choice["implementation"] = serde_json::json!(implementation);
        }
    }
}

/// One reached float reduction as an explain device choice.
fn reduction_choice(receipt: &crate::schedule::reductions::Reduction) -> serde_json::Value {
    serde_json::json!({
        "function": receipt.function,
        "span": {"start": receipt.span.start, "end": receipt.span.end},
        // `kind` carries the calibration family id; the device schema is closed.
        "backend": receipt.backend, "kind": receipt.family, "reason": receipt.reason,
        "cost_ms": receipt.cost.map(|cost| serde_json::json!({"cpu": cost.cpu_ms, "device": cost.device_ms})),
    })
}

/// Project a receipt onto the frozen AI-01 device-explain schema.
pub fn device_document(receipt: &serde_json::Value) -> serde_json::Value {
    let mut document = receipt.clone();
    if let Some(fields) = document.as_object_mut() {
        fields.remove("backends");
        fields.remove("fallback_reason");
    }
    document
}

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(feature = "native")]
    #[test]
    fn issue_ai_01_machine_diagnostics_native_backend_faults() {
        let schema: serde_json::Value =
            serde_json::from_str(include_str!("../schema/invar-diagnostics-v1.schema.json"))
                .unwrap();
        let path = "fixture.inv";
        for (tool_code, expected) in [
            (Some("NATIVE_ABI"), "NATIVE_ABI"),
            (Some("NATIVE_TRACE_LIMIT"), "NATIVE_TRACE_LIMIT"),
            (None, "NATIVE_TOOL"),
        ] {
            let mut document =
                serde_json::json!({"schema_version":1,"status":"tool_error","diagnostics":[]});
            if let Some(code) = tool_code {
                document["tool_code"] = serde_json::json!(code);
            }
            annotate_native_document(&mut document, path, b"");
            document["schema"] = serde_json::json!(crate::diagnostics::SCHEMA);
            assert_eq!(document["diagnostics"][0]["code"], expected);
            assert!(jsonschema::is_valid(&schema, &document), "{document}");
        }
    }

    #[test]
    fn issue_ai_01_machine_diagnostics_internal_main_fault() {
        let observation = crate::eval::Observation {
            result: Ok(Value::Bool(true)),
            steps_used: 0,
            cleanup_trace: Vec::new(),
        };
        let error = observed_i32(observation).unwrap_err();
        let document = error.to_diagnostics_json("fixture.inv", b"");
        assert_eq!(document["diagnostics"][0]["code"], "INTERNAL_MAIN");
        let schema: serde_json::Value =
            serde_json::from_str(include_str!("../schema/invar-diagnostics-v1.schema.json"))
                .unwrap();
        assert!(jsonschema::is_valid(&schema, &document));
    }
}
