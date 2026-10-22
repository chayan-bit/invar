//! The ordinary CLI's shared run/explain placement boundary.
use super::*;
use invar::{
    engine::Engine,
    placement::{self, Profile, calibration},
};
use serde_json::json;
use std::path::{Path, PathBuf};

/// Placement owns `--engine auto` only when asked for: a placement flag, or a
/// profile at the live host's default path that is not a matching provisional
/// one. A frozen, mismatched or malformed default goes through placement so its
/// refusal or typed error is reported. A matching provisional profile alone
/// never switches auto; otherwise the established auto path runs and its
/// documents stay byte-identical.
pub fn placement_requested(flags: &[OsString]) -> bool {
    if flags
        .iter()
        .any(|f| f == "--profile" || f == "--host-class" || f == "--trust-provisional")
    {
        return true;
    }
    let host = format!("{}-{}", std::env::consts::OS, std::env::consts::ARCH);
    let path = PathBuf::from(format!("calibration/{host}.json"));
    match Profile::load_default(&path, false, &calibration::live_identity(&host)) {
        Ok(None) => false,
        Ok(Some(profile)) => profile.document()["provisional"] == false,
        Err(_) => true,
    }
}

/// The explain document version. Every `--engine auto` explain is v5 (#50),
/// so one reaching here records the established path; the others keep the
/// version of the intrinsic sites they carry.
pub fn explain_version(document: &mut serde_json::Value, engine: Option<&str>) -> u32 {
    if engine == Some("auto") {
        document["auto_path"] = json!("established");
        5
    } else if document.get("explicit_spatial").is_some() {
        4
    } else if document.get("explicit_sorts").is_some() {
        3
    } else {
        1
    }
}

/// A scheduled run's JSON receipt. The established auto path's is the unified
/// `invar-diagnostics-v3` recording `auto_path: "established"` (#50).
pub fn receipt(mut document: serde_json::Value, path: &OsString, engine: Engine) -> i32 {
    if engine == Engine::Auto {
        document["schema_version"] = json!(3);
        document["auto_path"] = json!("established");
    }
    print_run_json(document, path)
}

/// A scheduled run's failure, reported like [`receipt`] reports success.
pub fn report(error: EvalError, path: &OsString, json: bool, engine: Engine) -> i32 {
    if !json || engine != Engine::Auto {
        return report_eval(error, path, json);
    }
    let status = match error {
        EvalError::Trap { .. } => 3,
        EvalError::EvaluationLimit { .. } => 4,
        EvalError::Tool { .. } => 2,
    };
    let document = error.to_diagnostics_json(&path.to_string_lossy(), &diagnostic_source(path));
    match receipt(document, path, engine) {
        2 => 2,
        _ => status,
    }
}

pub fn command(args: &[OsString]) -> i32 {
    let path = &args[2];
    let explain = args[1] == "explain";
    let mut json_output = explain;
    let mut trust = false;
    let mut profile_path = None;
    let mut host = format!("{}-{}", std::env::consts::OS, std::env::consts::ARCH);
    let mut fuel = 1_000_000;
    let mut artifact = None;
    let mut seen = std::collections::HashSet::new();
    let mut i = 3;
    while i < args.len() {
        let Some(flag) = args[i].to_str() else {
            return usage();
        };
        if !seen.insert(flag) {
            return usage();
        }
        if flag == "--trust-provisional" {
            trust = true;
            i += 1;
            continue;
        }
        let Some(value) = args.get(i + 1) else {
            return usage();
        };
        match flag {
            "--engine" if value == "auto" => {}
            "--format" if value == "json" => json_output = true,
            "--profile" => profile_path = Some(PathBuf::from(value)),
            "--host-class" => {
                let Some(v) = value.to_str().filter(|v| {
                    !v.is_empty()
                        && v.len() <= 128
                        && v.bytes()
                            .all(|b| b.is_ascii_alphanumeric() || b == b'-' || b == b'_')
                }) else {
                    return usage();
                };
                host = v.into();
            }
            "--artifact" => artifact = Some(value),
            "--fuel" if !explain => {
                let Some(v) = value
                    .to_str()
                    .filter(|v| !v.is_empty() && v.bytes().all(|b| b.is_ascii_digit()))
                    .and_then(|v| v.parse::<u64>().ok())
                    .filter(|v| *v > 0)
                else {
                    return usage();
                };
                fuel = v;
            }
            _ => return usage(),
        }
        i += 2;
    }
    if let Some(e) = unhonored_artifact(artifact) {
        return report_eval(e, path, json_output);
    }
    let (source, checked) = match load_checked(path, true) {
        Ok(v) => v,
        Err(e) => return report_front(e, path, json_output),
    };
    let program = match invar::mir::lower(&checked) {
        Ok(v) => v,
        Err(e) => return report_front(FrontError::tool(e.code, e.message), path, json_output),
    };
    // Placement run receipts record numeric contracts like every other run.
    if let Some(contracts) = invar::math::contracts(&program) {
        let _ = RUN_CONTRACTS.set(contracts);
    }
    let fold = invar::math::fold_receipt(&program);
    if let Some(fold) = &fold {
        let _ = RUN_FOLD.set(fold.clone());
    }
    #[cfg(all(feature = "metal", target_os = "macos"))]
    let image = match invar::engine::metal_image(artifact.map(Path::new)) {
        Ok(v) => v,
        Err(e) => return report_eval(e, path, json_output),
    };
    #[cfg(all(feature = "metal", target_os = "macos"))]
    if let Some(image) = image.as_ref()
        && let Err(e) = invar::schedule::metal::open_device(
            &source,
            invar::schedule::metal::Kernels::Prebuilt(image),
        )
    {
        return report_eval(
            EvalError::Tool {
                code: e.code,
                message: e.message,
            },
            path,
            json_output,
        );
    }
    let identity = calibration::live_identity(&host);
    let explicit = profile_path.is_some();
    let profile_path =
        profile_path.unwrap_or_else(|| PathBuf::from(format!("calibration/{host}.json")));
    let (profile, refusal) = match Profile::load_default(&profile_path, explicit, &identity) {
        Ok(_) if identity["host_id"].is_null() => (None, Some("TUNING_HINT_INVALID")),
        Ok(p) => (p, None),
        Err(placement::Error::Identity) => (None, Some("TUNING_HINT_INVALID")),
        Err(placement::Error::FreezeRequired) => (None, Some("TUNING_FREEZE_REQUIRED")),
        Err(e) => {
            return report_eval(
                EvalError::Tool {
                    code: e.code(),
                    message: "placement profile rejected".into(),
                },
                path,
                json_output,
            );
        }
    };
    let route = placement::Route::select(cfg!(feature = "native"), profile.is_some() && trust);
    let summary = json!({"profile_id":profile.as_ref().map(Profile::id),"profile_schema_version":profile.as_ref().map(|_|placement::VERSION),
        "profile_identity":profile.as_ref().map(|p|&p.document()["identity"]),"resource_identity":identity,
        "trust_provisional":trust,"engine":"cpu","cpu_cost":{"fixed_ms":null,"work_ms":null,"page_in_ms":null,"transfer_ms":null,"launch_ms":null,"total_ms":null},"variants":[],"choices":[],"refusal_code":refusal.or_else(||if profile.is_none() {Some("TUNING_NO_CANDIDATES")} else if !trust {Some("TUNING_FREEZE_REQUIRED")} else {None}),
        "legality":placement::legality(&source),"route":route.receipt()});
    let run = || -> Result<(i32, serde_json::Value), EvalError> {
        // D1(a): absent/mismatched/untrusted profiles retain the compiled CPU.
        if route == placement::Route::Native {
            #[cfg(feature = "native")]
            {
                let (status, mut doc) = invar::engine::run_native(
                    &program,
                    fuel,
                    &path.to_string_lossy(),
                    &source,
                    None,
                )?;
                doc["placement"] = summary.clone();
                doc["auto_path"] = json!("placement");
                doc["placement"]["choices"] = json!([]);
                return Ok((status, doc));
            }
        }
        let compiler = {
            #[cfg(feature = "cuda")]
            {
                nvrtc_helper()
            }
            #[cfg(not(feature = "cuda"))]
            {
                PathBuf::new()
            }
        };
        let resources = invar::schedule::profiled::Resources {
            compiler: Path::new(&compiler),
            #[cfg(all(feature = "metal", target_os = "macos"))]
            kernels: invar::engine::metal_kernels(image.as_ref()),
        };
        let run = invar::schedule::profiled::run(
            &source,
            &program,
            fuel,
            profile.as_ref(),
            trust,
            resources,
        )?;
        let (status, mut doc) = match run.observation.result {
            Ok(Value::I32(v)) => (
                0,
                json!({"schema_version":1,"status":"ok","diagnostics":[],"result":{"type":"i32","value":v}}),
            ),
            Ok(_) => {
                return Err(EvalError::Tool {
                    code: "INTERNAL_MAIN",
                    message: "checked main returned non-i32".into(),
                });
            }
            Err(e) => {
                let status = match e {
                    EvalError::Trap { .. } => 3,
                    EvalError::EvaluationLimit { .. } => 4,
                    EvalError::Tool { .. } => 2,
                };
                (
                    status,
                    e.to_diagnostics_json(&path.to_string_lossy(), &source),
                )
            }
        };
        let engine = if run
            .choices
            .iter()
            .any(|v| v["engine"] == "cuda" && v["published"] == true)
        {
            "cuda"
        } else if run
            .choices
            .iter()
            .any(|v| v["engine"] == "metal" && v["published"] == true)
        {
            "metal"
        } else {
            "cpu"
        };
        doc["placement"] = summary.clone();
        doc["auto_path"] = json!("placement");
        doc["placement"]["engine"] = if doc["status"] == "tool_error"
            && doc["diagnostics"]
                .as_array()
                .into_iter()
                .flatten()
                .any(|d| d["code"] == "CUDA_PENDING" || d["code"] == "METAL_PENDING")
        {
            json!("none")
        } else {
            json!(engine)
        };
        doc["placement"]["choices"] = json!(run.choices);
        Ok((status, doc))
    };
    match run() {
        Err(e) => {
            let status = match e {
                EvalError::Trap { .. } => 3,
                EvalError::EvaluationLimit { .. } => 4,
                EvalError::Tool { .. } => 2,
            };
            let mut doc = e.to_diagnostics_json(&path.to_string_lossy(), &source);
            doc["placement"] = summary.clone();
            doc["auto_path"] = json!("placement");
            doc["placement"]["engine"] = json!("none");
            doc["schema_version"] = json!(3);
            if json_output {
                if print_run_json(doc, path) == 2 {
                    return 2;
                }
            } else {
                eprintln!("{doc}")
            }
            status
        }
        Ok((status, mut doc)) => {
            doc["schema_version"] = json!(3);
            doc["schema"] = json!("invar-diagnostics-v3");
            if explain {
                let mut report = match invar::schedule::regions(&source, &program) {
                    Ok(r) => r.to_json(),
                    Err(e) => return report_front(FrontError::tool(e.code, e.message), path, true),
                };
                // v5 extends v4: the intrinsic sites stay beside the placement receipt.
                explain_intrinsics(&mut report, &program);
                if let Some(fold) = &fold {
                    report["numeric_contracts"]["fold"] = fold.clone();
                }
                report["schema"] = json!("invar-explain-v5");
                report["schema_version"] = json!(5);
                report["placement"] = doc["placement"].clone();
                report["auto_path"] = json!("placement");
                // Retain the established device-explain schema; detailed profile
                // costs live in the shared placement receipt beside it.
                let choices=doc["placement"]["choices"].as_array().into_iter().flatten().map(|c|json!({
                    "function":c["function"],"span":c["span"],"backend":c["engine"],
                    "reason":if c["engine"]=="cpu" {"measured-cpu-cost"} else {"measured-device-cost"}
                })).collect::<Vec<_>>();
                if status == 0 {
                    report["device_schema"] = json!("invar-explain-device-v5");
                    report["device_execution"] = json!({"requested_engine":"auto","choices":choices,"result":doc["result"]["value"],
                    "module_cache_hits":0,"schedule_hint_hits":0,"schedule_hint_invalids":0});
                }
                report["execution"] = doc;
                if report.to_string().len() > 1_000_000 {
                    return report_front(
                        FrontError::tool(
                            "DIAGNOSTIC_DOCUMENT_LIMIT",
                            "explain document exceeds one million bytes",
                        ),
                        path,
                        true,
                    );
                }
                println!("{report}");
                return status;
            }
            if json_output {
                if print_run_json(doc, path) == 2 {
                    return 2;
                }
            } else if status == 0 {
                println!("Result: {}", doc["result"]["value"])
            } else {
                eprintln!("{doc}")
            }
            status
        }
    }
}
