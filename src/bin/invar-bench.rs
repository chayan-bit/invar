#[path = "../../bench/baseline.rs"]
mod baseline;
#[path = "../../bench/baseline_families.rs"]
mod baseline_families;
#[path = "../../bench/corpus.rs"]
mod corpus;
#[path = "../../bench/domains/mod.rs"]
mod domains;
#[path = "../../bench/family_references.rs"]
mod family_references;
#[path = "../../bench/general-net-v01/cli.rs"]
mod general_net;
#[path = "../../bench/hash.rs"]
mod hash;
#[path = "../../bench/heldout.rs"]
#[allow(dead_code)]
mod heldout;
#[path = "../../bench/manifest.rs"]
mod manifest;

#[path = "invar-bench/cold.rs"]
mod cold;
#[cfg(test)]
#[path = "invar-bench/cold_tests.rs"]
mod cold_tests;
#[path = "invar-bench/generated.rs"]
mod generated;
#[path = "invar-bench/report.rs"]
mod report;

use serde_json::{Value, json};
use std::{
    env, fs,
    io::{Read, Write},
    path::{Path, PathBuf},
    process::{Child, Command, ExitCode, ExitStatus, Stdio},
    thread,
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

const MAX_MANIFEST: u64 = 256 * 1024;
const MAX_CORPUS: u64 = 2 * 1024 * 1024;
const MAX_REPORT: u64 = 256 * 1024;
const COLD_TIMEOUT: Duration = Duration::from_secs(30);
const COLD_TOTAL_TIMEOUT: Duration = Duration::from_secs(300);
const COLD_HOST_BUDGET: u64 = 1024 * 1024 * 1024;
#[cfg(any(test, all(feature = "metal", target_os = "macos"), feature = "cuda"))]
const MEASURED_SEED: u32 = 17;
#[cfg(any(test, all(feature = "metal", target_os = "macos"), feature = "cuda"))]
const MEASURED_INPUT_RECIPE: &str = "lcg1664525_1013904223_plus_slot17_v1";

use cold::*;
use generated::*;
use report::*;

fn run() -> Result<(), String> {
    if matches!(
        env::args_os()
            .nth(1)
            .as_deref()
            .and_then(|arg| arg.to_str()),
        Some("--gnet-validate" | "--gnet-evaluate")
    ) {
        return general_net::run(env::args_os().skip(1).collect());
    }
    if matches!(
        env::args_os()
            .nth(1)
            .as_deref()
            .and_then(|arg| arg.to_str()),
        Some("--domain-report" | "--write-domain-proposal")
    ) {
        return domains::cli::run(env::args_os().skip(1).collect());
    }
    let mut manifest_path = PathBuf::from("bench/manifest-v1.json");
    let mut report_path: Option<PathBuf> = None;
    let mut track: Option<String> = None;
    let mut baseline_executable: Option<PathBuf> = None;
    let mut baseline_build_record: Option<PathBuf> = None;
    let mut baseline_family: Option<String> = None;
    let mut baseline_elements: Option<usize> = None;
    let mut generated_case: Option<String> = None;
    let mut generated_backend: Option<String> = None;
    let mut baseline_report: Option<PathBuf> = None;
    let mut associate_generated: Option<PathBuf> = None;
    let mut generated_build_record: Option<PathBuf> = None;
    let (
        mut validate_only,
        mut require_gpu,
        mut preflight,
        mut write_artifacts,
        mut write_v2_proposal,
        mut metal_smoke,
        mut cold_child,
        mut baseline_cold_child,
    ) = (false, false, false, false, false, false, false, false);
    let mut args = env::args_os().skip(1);
    while let Some(arg) = args.next() {
        match arg.to_str() {
            Some("--manifest") => {
                manifest_path = PathBuf::from(args.next().ok_or("missing --manifest value")?)
            }
            Some("--report") => {
                report_path = Some(PathBuf::from(args.next().ok_or("missing --report value")?))
            }
            Some("--track") => {
                track = Some(
                    args.next()
                        .ok_or("missing --track value")?
                        .into_string()
                        .map_err(|_| "non-UTF8 track")?,
                )
            }
            Some("--baseline-executable") => {
                baseline_executable = Some(PathBuf::from(
                    args.next().ok_or("missing --baseline-executable value")?,
                ))
            }
            Some("--baseline-build-record") => {
                baseline_build_record = Some(PathBuf::from(
                    args.next().ok_or("missing --baseline-build-record value")?,
                ))
            }
            Some("--baseline-family") => {
                baseline_family = Some(
                    args.next()
                        .ok_or("missing --baseline-family value")?
                        .into_string()
                        .map_err(|_| "non-UTF8 baseline family")?,
                )
            }
            Some("--baseline-elements") => {
                baseline_elements = Some(
                    args.next()
                        .ok_or("missing --baseline-elements value")?
                        .into_string()
                        .map_err(|_| "non-UTF8 element count")?
                        .parse::<usize>()
                        .map_err(|_| "invalid baseline element count")?,
                )
            }
            Some("--generated-case") => {
                generated_case = Some(
                    args.next()
                        .ok_or("missing --generated-case value")?
                        .into_string()
                        .map_err(|_| "non-UTF8 generated case")?,
                )
            }
            Some("--generated-backend") => {
                generated_backend = Some(
                    args.next()
                        .ok_or("missing --generated-backend value")?
                        .into_string()
                        .map_err(|_| "non-UTF8 generated backend")?,
                )
            }
            Some("--baseline-report") => {
                baseline_report = Some(PathBuf::from(
                    args.next().ok_or("missing --baseline-report value")?,
                ))
            }
            Some("--associate-generated") => {
                associate_generated = Some(PathBuf::from(
                    args.next().ok_or("missing --associate-generated value")?,
                ))
            }
            Some("--generated-build-record") => {
                generated_build_record = Some(PathBuf::from(
                    args.next()
                        .ok_or("missing --generated-build-record value")?,
                ))
            }
            Some("--validate-only") => validate_only = true,
            Some("--require-gpu") => require_gpu = true,
            Some("--preflight") => preflight = true,
            Some("--write-artifacts") => write_artifacts = true,
            Some("--write-v2-proposal") => write_v2_proposal = true,
            Some("--metal-held-out-smoke") => metal_smoke = true,
            Some("--cold-child") => cold_child = true,
            Some("--baseline-cold-child") => baseline_cold_child = true,
            _ => return Err(format!("unknown argument: {}", arg.to_string_lossy())),
        }
    }
    if !write_artifacts
        && !write_v2_proposal
        && associate_generated.is_none()
        && [validate_only, require_gpu, preflight]
            .iter()
            .filter(|x| **x)
            .count()
            != 1
    {
        return Err("choose exactly one of --validate-only, --require-gpu, or --preflight".into());
    }
    if write_artifacts
        && (validate_only
            || require_gpu
            || preflight
            || track.is_some()
            || report_path.is_some()
            || baseline_executable.is_some()
            || baseline_build_record.is_some()
            || generated_case.is_some()
            || generated_backend.is_some()
            || baseline_report.is_some()
            || associate_generated.is_some()
            || generated_build_record.is_some()
            || metal_smoke
            || cold_child
            || baseline_cold_child)
    {
        return Err("--write-artifacts takes no other mode options".into());
    }
    if write_v2_proposal
        && (write_artifacts
            || validate_only
            || require_gpu
            || preflight
            || report_path.is_some()
            || baseline_executable.is_some()
            || baseline_build_record.is_some()
            || baseline_report.is_some()
            || associate_generated.is_some()
            || generated_build_record.is_some()
            || generated_case.is_some()
            || generated_backend.is_some()
            || metal_smoke
            || cold_child
            || baseline_cold_child)
    {
        return Err("--write-v2-proposal takes only --manifest and optional --track".into());
    }
    if associate_generated.is_some()
        && (track.as_deref() != Some("generated")
            || baseline_report.is_none()
            || report_path.is_none()
            || validate_only
            || require_gpu
            || preflight
            || metal_smoke
            || generated_case.is_some()
            || generated_backend.is_some()
            || cold_child
            || baseline_cold_child
            || baseline_executable.is_some()
            || baseline_build_record.is_some())
    {
        return Err(
            "--associate-generated requires --track generated, --baseline-report and --report only"
                .into(),
        );
    }
    if generated_build_record.is_some() && associate_generated.is_none() {
        return Err("--generated-build-record requires --associate-generated".into());
    }
    let corpus = corpus::generate()?;
    if corpus["programs"]
        .as_array()
        .is_none_or(|p| p.len() != corpus::COUNT)
    {
        return Err("generated corpus count mismatch".into());
    }
    let corpus_path = manifest_path
        .parent()
        .unwrap_or(std::path::Path::new("."))
        .join("corpus-v1.json");
    if write_artifacts {
        if manifest_path
            .file_name()
            .is_some_and(|name| name == "corpus-v1.json")
            || corpus_path.exists()
            || manifest_path.exists()
        {
            return Err("artifact output exists or paths collide; choose new paths".into());
        }
        let bytes = json_bytes(&corpus, MAX_CORPUS)?;
        let manifest = manifest::build(&corpus, &bytes)?;
        write_json(&corpus_path, &corpus, MAX_CORPUS)?;
        write_json(&manifest_path, &manifest, MAX_MANIFEST)?;
        return Ok(());
    }
    let track = if write_v2_proposal {
        track.unwrap_or_else(|| "generated".into())
    } else {
        track.ok_or("--track is required")?
    };
    if !["generated", "library-assisted", "same-structure"].contains(&track.as_str()) {
        return Err("invalid track".into());
    }
    let report_path = if write_v2_proposal {
        report_path
    } else {
        Some(report_path.ok_or("--report is required")?)
    };
    if preflight
        && (track != "generated"
            || baseline_executable.is_some()
            || baseline_build_record.is_some()
            || generated_build_record.is_some()
            || baseline_family.is_some()
            || baseline_elements.is_some()
            || generated_case.is_some()
            || generated_backend.is_some()
            || baseline_report.is_some()
            || metal_smoke
            || cold_child
            || baseline_cold_child)
    {
        return Err("--preflight requires --track generated and no benchmark options".into());
    }
    let raw = read_bounded(&manifest_path, MAX_MANIFEST)?;
    let observed: Value =
        serde_json::from_slice(&raw).map_err(|e| format!("manifest JSON: {e}"))?;
    let corpus_raw = read_bounded(&corpus_path, MAX_CORPUS)?;
    let observed_corpus: Value =
        serde_json::from_slice(&corpus_raw).map_err(|e| format!("corpus JSON: {e}"))?;
    if observed_corpus != corpus {
        return Err("corpus content, seed, source hash, or split differs from generator".into());
    }
    let v2 = observed["schema_version"] == 2;
    let v1 = manifest::build(&corpus, &corpus_raw)?;
    let expected = if v2 {
        manifest::propose_v2(&manifest::base(&corpus, &corpus_raw)?)?
    } else {
        v1
    };
    if observed != expected {
        return Err(
            "manifest fields, arithmetic, baseline, cases, track, or hashes differ from contract"
                .into(),
        );
    }
    if observed["cases"].as_array().is_none_or(|c| c.is_empty())
        || manifest::family_counts(&observed)
            .as_object()
            .is_none_or(|c| c.values().any(|v| v.as_u64() == Some(0)))
    {
        return Err("zero-case family".into());
    }
    if write_v2_proposal {
        let proposal_path = manifest_path
            .parent()
            .unwrap_or(std::path::Path::new("."))
            .join("manifest-v2-proposal.json");
        if proposal_path.exists() {
            return Err("v2 proposal path exists; preserve the prior artifact".into());
        }
        write_json(
            &proposal_path,
            // The pending proposal stays derived from the measured base only.
            &manifest::propose_v2(&manifest::base(&corpus, &corpus_raw)?)?,
            MAX_MANIFEST,
        )?;
        return Ok(());
    }
    let report_path = report_path.ok_or("--report is required")?;
    if report_path.exists() {
        return Err("report path exists; choose a new path to preserve it".into());
    }
    let mut report = json!({
        "schema_version":observed["report_schema_version"],"kind":"manifest_validation","status":"passed",
        "track":track,"checkout_at_invocation":checkout_at_invocation(),"manifest_sha256":hash::sha256(&raw),"corpus_sha256":hash::sha256(&corpus_raw),
        "held_out_sha256":corpus::held_out_hash(&corpus)?,"generator_seed":corpus::SEED,
        "program_counts":{"total":corpus::COUNT,"held_out":60,"tuning":180},
        "case_counts":manifest::family_counts(&observed),
        "freeze_status":observed["freeze_status"],"gpu_performance":"not_run",
        "hardware_targets":"pending_preflight","library_versions":"pending_preflight",
    });
    if let Some(generated_path) = associate_generated {
        let associated = (|| -> Result<Value, String> {
            let generated_raw = read_bounded(&generated_path, MAX_REPORT)?;
            let baseline_path = baseline_report.as_ref().ok_or("missing baseline report")?;
            let baseline_raw = read_bounded(baseline_path, MAX_REPORT)?;
            let mut generated: Value = serde_json::from_slice(&generated_raw)
                .map_err(|e| format!("generated report JSON: {e}"))?;
            let baseline: Value = serde_json::from_slice(&baseline_raw)
                .map_err(|e| format!("baseline report JSON: {e}"))?;
            // An unavailable case or baseline is never associated as a measurement.
            if let Some(case) = observed["cases"].as_array().and_then(|cases| {
                cases
                    .iter()
                    .find(|case| case["id"] == generated["generated_result"]["case_id"])
            }) {
                for backend in [None, Some("cuda"), Some("metal")] {
                    manifest::require_measurable(&observed, case, backend)?;
                }
            }
            if generated["schema_version"] != report["schema_version"]
                || baseline["schema_version"] != report["schema_version"]
                || baseline["baseline_result"]["schema_version"] != report["schema_version"]
                || generated["manifest_sha256"] != report["manifest_sha256"]
                || baseline["manifest_sha256"] != report["manifest_sha256"]
                || generated["corpus_sha256"] != report["corpus_sha256"]
                || baseline["corpus_sha256"] != report["corpus_sha256"]
                || generated["held_out_sha256"] != report["held_out_sha256"]
                || baseline["held_out_sha256"] != report["held_out_sha256"]
                || generated["track"] != "generated"
                || baseline["track"] != "library-assisted"
                || baseline["kind"] != "independent_library_baseline"
            {
                return Err("association schema or manifest lineage mismatch".into());
            }
            if generated["kind"] != "generated_case"
                || generated["status"] != "partial"
                || generated["generated_result"]["full_output_correct"] != true
            {
                return Err("generated report is not a verified partial case".into());
            }
            let build_record = if v2 {
                let path = generated_build_record
                    .as_ref()
                    .ok_or("v2 generated build record required")?;
                let raw = read_bounded(path, 16 * 1024)?;
                let record: Value = serde_json::from_slice(&raw)
                    .map_err(|e| format!("generated build record JSON: {e}"))?;
                if record["schema_version"] != 1
                    || record["runner_executable_sha256"]
                        != generated["generated_result"]["runner_executable_sha256"]
                    || record["source_head"] != generated["checkout_at_invocation"]["head"]
                    || record["worktree_at_build"]
                        != generated["checkout_at_invocation"]["worktree"]
                    || record["rustc_version"].as_str().is_none_or(str::is_empty)
                    || record["build_command"]
                        .as_array()
                        .is_none_or(|items| items.is_empty())
                {
                    return Err("v2 generated build record mismatch".into());
                }
                generated["generated_result"]["build_record"] = record.clone();
                json!({"record_sha256":hash::sha256(&raw),"record":record})
            } else {
                Value::Null
            };
            let case_id = generated["generated_result"]["case_id"]
                .as_str()
                .ok_or("generated case id missing")?;
            let case = observed["cases"]
                .as_array()
                .and_then(|cases| cases.iter().find(|case| case["id"] == case_id))
                .ok_or("generated case is outside manifest")?;
            let comparison = baseline_eligibility(&generated, &baseline, case);
            Ok(
                json!({"generated_report":generated_path,"generated_report_sha256":hash::sha256(&generated_raw),
                "baseline_report":baseline_path,"baseline_report_sha256":hash::sha256(&baseline_raw),
                "generated_build_record":build_record,
                "comparison":comparison}),
            )
        })();
        report["kind"] = json!("benchmark_association");
        match associated {
            Ok(value) => {
                report["status"] = value["comparison"]["status"].clone();
                report["association"] = value;
            }
            Err(reason) => {
                report["status"] = json!("pending");
                report["reason"] = json!(reason);
            }
        }
        write_json(&report_path, &report, MAX_REPORT)?;
        if report["status"] != "eligible" {
            return Err("benchmark association pending".into());
        }
        return Ok(());
    }
    if preflight {
        let checks = gpu_preflight();
        let available = checks.as_array().is_some_and(|items| {
            !items.is_empty() && items.iter().all(|item| item["status"] == "available")
        });
        report["kind"] = json!("gpu_preflight");
        report["status"] = json!(if available { "available" } else { "pending" });
        report["preflight_scope"] = json!("backend_feature_and_device_open_only");
        report["gpu_preflight"] = checks;
        write_json(&report_path, &report, MAX_REPORT)?;
        if !available {
            return Err("GPU preflight pending: an enabled backend is unavailable".into());
        }
        return Ok(());
    }
    if let Some(executable) = baseline_executable {
        if !require_gpu || track != "library-assisted" || metal_smoke || cold_child {
            return Err(
                "baseline executable requires --require-gpu --track library-assisted".into(),
            );
        }
        let family = baseline_family.ok_or("--baseline-family is required")?;
        let n = baseline_elements.ok_or("--baseline-elements is required")?;
        if baseline_cold_child && !v2 {
            return Err("baseline cold child requires v2".into());
        }
        if family == "f32gemm" && n > 128 {
            return Err("GEMM dimension limit exceeded".into());
        }
        let build_record = if v2 {
            let path = baseline_build_record
                .as_ref()
                .ok_or("v2 baseline build record required")?;
            let raw = read_bounded(path, 16 * 1024)?;
            let record: Value = serde_json::from_slice(&raw)
                .map_err(|e| format!("baseline build record JSON: {e}"))?;
            if record["schema_version"] != 1
                || record["fixture_source_sha256"]
                    != observed["baseline_fixture_sources"]["metal_mpsgraph_reduction_v2_sha256"]
                || record["executable_sha256"]
                    != hash::sha256(&read_bounded(&executable, 64 * 1024 * 1024)?)
                || record["source_head"] != report["checkout_at_invocation"]["head"]
                || !["clean", "dirty"].contains(&record["worktree_at_build"].as_str().unwrap_or(""))
                || record["compiler_version"]
                    .as_str()
                    .is_none_or(str::is_empty)
                || record["build_command"]
                    .as_array()
                    .is_none_or(|items| items.is_empty())
            {
                return Err("v2 baseline build record mismatch".into());
            }
            Some((raw, record))
        } else {
            if baseline_build_record.is_some() {
                return Err("baseline build record is v2 only".into());
            }
            None
        };
        let mut result = if baseline_families::FAMILIES.contains(&family.as_str()) {
            if v2 || baseline_cold_child {
                return Err("family baselines have no v2 or cold-child protocol".into());
            }
            baseline_families::run(&executable, &family, n, baseline_families::Backend::host())?
        } else {
            baseline::run(&executable, &family, n, v2, baseline_cold_child)?
        };
        if let Some((raw, record)) = build_record {
            if record["fixture_source_sha256"] != result["fixture_source_sha256"]
                || record["executable_sha256"] != result["executable_sha256"]
            {
                return Err("v2 baseline changed during execution".into());
            }
            result["build_record_sha256"] = json!(hash::sha256(&raw));
            result["build_record"] = record;
            result["build_provenance"] =
                json!("matching source/executable hashes and unreviewed build record");
            result["build_review_status"] = json!("pending");
        }
        if v2
            && result["fixture_source_sha256"]
                != observed["baseline_fixture_sources"]["metal_mpsgraph_reduction_v2_sha256"]
        {
            return Err("v2 fixture source differs from manifest".into());
        }
        let candidate = match family.as_str() {
            "map" => Some(("map", "wrapping_map")),
            "sum" => Some(("reduction", "wrapping_sum")),
            "affine" => Some(("scan", "affine_composition")),
            _ => None,
        }
        .and_then(|(manifest_family, law)| {
            observed["cases"].as_array()?.iter().find(|case| {
                case["family"] == manifest_family && case["law"] == law && case["elements"] == n
            })
        });
        let quality_control = result["kind"] == "handwritten_quality_control";
        result["candidate_case_id"] = candidate.map_or(Value::Null, |case| case["id"].clone());
        result["gate_status"] = json!(if quality_control {
            "quality_control_only"
        } else if candidate.is_some() {
            "pending_semantic_quality_build_review"
        } else {
            "outside_current_manifest_or_law"
        });
        report["kind"] = if baseline_cold_child {
            json!("baseline_cold_child")
        } else {
            result["kind"].clone()
        };
        report["status"] = json!(match result["full_output_correctness"].as_str() {
            Some("exact_full_output_pass") if baseline_cold_child => "passed",
            Some("exact_full_output_pass") => "pending_review",
            Some("pending_m4_numeric_policy") => "pending_numeric_policy",
            _ => "failed",
        });
        if !baseline_cold_child {
            report["gpu_preflight"] = gpu_preflight();
        }
        report["baseline_result"] = result;
        if v2 && !baseline_cold_child && report["status"] == "pending_review" {
            let cold = env::current_exe()
                .map_err(|e| e.to_string())
                .and_then(|exe| {
                    let case = candidate.ok_or("v2 baseline case outside manifest")?;
                    let build_path = baseline_build_record
                        .as_ref()
                        .ok_or("missing baseline build record")?;
                    cold_process_samples(
                        &report_path,
                        case,
                        &report,
                        "fixture_process_open_compile_run_verify_one_library_call",
                        |child_report| {
                            let mut command = Command::new(&exe);
                            command
                                .arg("--manifest")
                                .arg(&manifest_path)
                                .args([
                                    "--track",
                                    "library-assisted",
                                    "--require-gpu",
                                    "--baseline-executable",
                                ])
                                .arg(&executable)
                                .arg("--baseline-build-record")
                                .arg(build_path)
                                .arg("--baseline-family")
                                .arg(&family)
                                .arg("--baseline-elements")
                                .arg(n.to_string())
                                .arg("--baseline-cold-child")
                                .arg("--report")
                                .arg(child_report);
                            command
                        },
                        validate_baseline_cold_child,
                    )
                });
            report["cold_process"] =
                cold.unwrap_or_else(|reason| json!({"status":"pending","reason":reason}));
        }
        write_json(&report_path, &report, MAX_REPORT)?;
        if baseline_cold_child && report["status"] == "passed" {
            return Ok(());
        }
        return Err(format!(
            "baseline gate status: {}",
            report["status"].as_str().unwrap_or("invalid")
        ));
    }
    if baseline_family.is_some() || baseline_elements.is_some() {
        return Err("baseline family/elements require an executable".into());
    }
    if baseline_cold_child {
        return Err("--baseline-cold-child requires a v2 baseline executable".into());
    }
    if generated_backend.is_some() && generated_case.is_none() {
        return Err("--generated-backend requires --generated-case".into());
    }
    if baseline_report.is_some() && (generated_case.is_none() || cold_child) {
        return Err("--baseline-report requires a parent generated case".into());
    }
    if cold_child && generated_case.is_none() {
        return Err("--cold-child requires --generated-case".into());
    }
    if let Some(case_id) = generated_case {
        if !require_gpu || track != "generated" || metal_smoke {
            return Err("--generated-case requires --require-gpu --track generated".into());
        }
        let case = observed["cases"]
            .as_array()
            .and_then(|cases| cases.iter().find(|case| case["id"] == case_id))
            .ok_or("unknown generated case")?;
        manifest::require_measurable(&observed, case, None)?;
        let backend = generated_backend.as_deref().unwrap_or("metal");
        let result = match (backend, cold_child) {
            ("metal", false) => metal_generated_case(case, &corpus),
            ("metal", true) => metal_generated_case_once(case, &corpus),
            ("cuda", false) => cuda_generated_case(case, &corpus),
            ("cuda", true) => cuda_generated_case_once(case, &corpus),
            _ => return Err("generated backend must be metal or cuda".into()),
        };
        report["kind"] = json!(if cold_child {
            "generated_cold_child"
        } else {
            "generated_case"
        });
        if !cold_child {
            report["gpu_preflight"] = gpu_preflight();
        }
        match result {
            Ok(result) => {
                report["status"] = json!(if cold_child { "passed" } else { "partial" });
                report["gpu_performance"] = json!(if cold_child {
                    "one_verified_cold_child"
                } else {
                    "raw_generated_samples_recorded"
                });
                report["generated_result"] = result;
                if !cold_child {
                    let case_name = case["id"].as_str().unwrap_or("");
                    let cold = env::current_exe()
                        .map_err(|e| e.to_string())
                        .and_then(|exe| {
                            cold_process_samples(
                                &report_path,
                                case,
                                &report,
                                "parse_check_lower_compile_run_verify_one_generated_call",
                                |child_report| {
                                    let mut command = Command::new(&exe);
                                    command
                                        .arg("--manifest")
                                        .arg(&manifest_path)
                                        .args([
                                            "--track",
                                            "generated",
                                            "--require-gpu",
                                            "--generated-backend",
                                        ])
                                        .arg(backend)
                                        .arg("--generated-case")
                                        .arg(case_name)
                                        .arg("--cold-child")
                                        .arg("--report")
                                        .arg(child_report);
                                    command
                                },
                                validate_cold_child,
                            )
                        });
                    report["cold_process"] =
                        cold.unwrap_or_else(|reason| json!({"status":"pending","reason":reason}));
                    report["comparison"] = match baseline_report {
                        Some(path) => match read_bounded(&path, MAX_REPORT).and_then(|bytes| {
                            serde_json::from_slice::<Value>(&bytes)
                                .map_err(|e| format!("baseline report JSON: {e}"))
                        }) {
                            Ok(baseline) => baseline_eligibility(&report, &baseline, case),
                            Err(reason) => {
                                json!({"status":"pending","reasons":[format!("baseline_report_invalid: {reason}")],
                                "device_resident_ratio":null,"host_to_host_ratio":null,"cold_process_ratio":null})
                            }
                        },
                        None => {
                            json!({"status":"pending","reasons":["independent_baseline_report_not_supplied"],
                            "device_resident_ratio":null,"host_to_host_ratio":null,"cold_process_ratio":null})
                        }
                    };
                }
            }
            Err(reason) => {
                report["status"] = json!("pending");
                report["reason"] = json!(reason);
            }
        }
        write_json(&report_path, &report, MAX_REPORT)?;
        if cold_child && report["status"] == "passed" {
            return Ok(());
        }
        return Err(format!(
            "generated case qualification: {}",
            report["status"]
        ));
    }
    if metal_smoke {
        if !require_gpu || track != "generated" {
            return Err("Metal held-out smoke requires --require-gpu --track generated".into());
        }
        let smoke = metal_held_out_smoke(&corpus).unwrap_or_else(|reason| {
            json!({"schema_version":1,"kind":"held_out_generated_correctness_smoke",
                "status":"pending","reason":reason,"performance_gate":"not_run"})
        });
        report["kind"] = json!("held_out_generated_correctness_smoke");
        report["status"] = smoke["status"].clone();
        report["gpu_preflight"] = gpu_preflight();
        report["held_out_smoke"] = smoke;
        write_json(&report_path, &report, MAX_REPORT)?;
        if report["status"] != "passed" {
            return Err(format!(
                "Metal held-out correctness smoke: {}",
                report["status"].as_str().unwrap_or("invalid")
            ));
        }
        return Ok(());
    }
    if require_gpu {
        report["kind"] = json!("gpu_qualification");
        report["status"] = json!("pending");
        report["gpu_performance"] = json!("not_run");
        report["gpu_preflight"] = gpu_preflight();
        report["reason"] =
            json!("M5 manifest-to-generated-run adapter and M4-01 f32 kernels are pending");
    }
    write_json(&report_path, &report, MAX_REPORT)?;
    if require_gpu {
        return Err("GPU qualification pending; no generated kernel was timed".into());
    }
    Ok(())
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("invar-bench: {e}");
            ExitCode::FAILURE
        }
    }
}
