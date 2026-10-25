#[path = "../../bench/acceptance.rs"]
mod acceptance;
use crate::hash;
use serde_json::{Value, json};

// All timings and hosts here are synthetic, with no device execution or speed claim.
fn fixture() -> (Vec<u8>, Value, Value) {
    let manifest = json!({"freeze_status":"frozen","corpus_sha256":"synthetic-corpus","held_out_sha256":"synthetic-held-out","targets":[{"name":"synthetic-cuda","host":"synthetic-host","device_id":"synthetic-device","compute_capability":"synthetic-sm","compiler_flags":"synthetic-flags","stream_policy":"synthetic-stream","driver":"synthetic","toolchain":"synthetic"}],"cases":[{"id":"synthetic-map","seed":17,"family":"map","shape":[64],"layout":"contiguous","required_bytes":768,"numeric_policy":"u32-wrapping-v1"}]});
    let raw = serde_json::to_vec(&manifest).unwrap();
    let digest = hash::sha256(&raw);
    let plan = json!({"status":"frozen","manifest_sha256":digest,"statistics":"paired-session-cluster-bootstrap-v1","bootstrap_seed":17,"baselines":[{"case_id":"synthetic-map","target":"synthetic-cuda","kind":"handwritten","source_sha256":"synthetic-baseline","revision":"synthetic-baseline-revision","executable_sha256":"synthetic-executable"},{"case_id":"synthetic-map","target":"synthetic-cuda","kind":"library","source_sha256":"synthetic-baseline","revision":"synthetic-baseline-revision","executable_sha256":"synthetic-executable"}],"expected":[{"id":"synthetic-parity","case_id":"synthetic-map","target":"synthetic-cuda","gate":"parity","scope":"device_resident"}]});
    let environment = json!({"host":"synthetic-host","device_id":"synthetic-device","compute_capability":"synthetic-sm","compiler_flags":"synthetic-flags","stream_policy":"synthetic-stream","driver":"synthetic","toolchain":"synthetic","cpu":"synthetic","os":"synthetic","clock_policy":"synthetic-default","power_policy":"synthetic-default","occupancy":"synthetic-empty-snapshot","temperature":"synthetic-30C","memory_preflight":"synthetic-pass","runtime_bound":"synthetic-60s","gpu_foreign_activity":{"record":"synthetic-gpu-activity-record","max_foreign_util_percent":0,"foreign_processes":0}});
    let build = json!({"worktree_at_build":"clean","source_head":"synthetic-revision","rustc_version":"synthetic","executable_sha256":"synthetic-executable"});
    let mut generated = json!({"manifest_sha256":digest,"corpus_sha256":"synthetic-corpus","freeze_status":"frozen","checkout_at_invocation":{"head":"synthetic-revision","worktree":"clean"},"environment":environment,"generated_result":{"case_id":"synthetic-map","device":"synthetic-cuda","build_review_status":"accepted","build_record":build,"measurement_scope":"checked_source_program","full_output_correct":true,"qualification_status":"passed","safety_status":"passed","numeric_policy_status":"passed","source_sha256":"synthetic-source","generated_code_hash":"synthetic-code","runner_executable_sha256":"synthetic-executable","warmups":10,"seed":17,"shape":[64],"layout":"contiguous","required_bytes":768,"numeric_policy":"u32-wrapping-v1","transfer_bytes":768,"launches":1,"workspace_bytes":0,"compile_ms":1,"peak_compiler_bytes":1024,"binary_bytes":100,"cache_bytes":100,"variants":1,"tuning_ms":1,"input_sha256":"synthetic-input","source_observable_output_sha256":"synthetic-output","input_recipe":"synthetic-recipe","allocation_policy":"synthetic-reuse","device_resident_boundary":"synthetic-resident","host_to_host_boundary":"synthetic-warm","device_resident_ms":vec![0.8;30],"host_to_host_ms":vec![0.9;30]}});
    generated["cold_process"] = json!({"status":"passed","contract_review":"accepted","samples_ms":vec![2.;30],"boundary":"synthetic-fresh-process"});
    let mut baseline = generated.clone();
    baseline.as_object_mut().unwrap().remove("generated_result");
    baseline["baseline_result"] = json!({"candidate_case_id":"synthetic-map","device":"synthetic-cuda","build_review_status":"accepted","build_record":build,"gate_status":"accepted","target_compatibility":"accepted","quality_review":"accepted","semantic_review":"accepted","full_output_correctness":"exact_full_output_pass","fixture_source_sha256":"synthetic-baseline","executable_sha256":"synthetic-executable","baseline_revision":"synthetic-baseline-revision","warmups":10,"seed":17,"transfer_bytes":768,"launches":1,"workspace_bytes":0,"compile_ms":1,"peak_compiler_bytes":1024,"binary_bytes":100,"cache_bytes":100,"variants":1,"tuning_ms":1,"input_sha256":"synthetic-input","output_sha256":"synthetic-output","input_recipe":"synthetic-recipe","numeric_policy":"u32-wrapping-v1","allocation_policy":"synthetic-reuse","device_resident_boundary":"synthetic-resident","host_to_host_boundary":"synthetic-warm","device_resident_samples_ms":vec![1.;30],"host_to_host_samples_ms":vec![1.2;30]});
    let sessions:Vec<_>=(0..3).map(|i|json!({"id":format!("synthetic-process-{i}"),"measurement_protocol":"randomized-interleaved-paired-blocks","ordering_seed":17,"generated":generated,"baselines":[{"kind":"handwritten","report":baseline}]})).collect();
    let bundle = json!({"synthetic":true,"qualification":{"status":"passed","manifest_sha256":digest},"prerequisites":{"m4_06":"passed","gnet":"passed","baseline_review":"passed","held_out_freeze":"passed"},"rows":[{"id":"synthetic-parity","track":"generated","status":"measured","sessions":sessions}]});
    (raw, plan, bundle)
}
fn result(raw: &[u8], plan: &Value, bundle: &Value) -> Value {
    acceptance::report(raw, plan, bundle).unwrap()["rows"][0]["result"].clone()
}

#[test]
fn issue_m5_03_cuda_performance_acceptance_gate_arithmetic_and_no_claim() {
    let (raw, plan, bundle) = fixture();
    let report = acceptance::report(&raw, &plan, &bundle).unwrap();
    assert_eq!(report["status"], "synthetic_no_claim");
    assert_eq!(report["passed"], 1);
    let stats = &report["rows"][0]["result"]["statistics"];
    assert!((stats["ratio"].as_f64().unwrap() - 0.8).abs() < 1e-12);
    assert!((stats["upper95"].as_f64().unwrap() - 0.8).abs() < 1e-12);
    assert_eq!(stats["generated_ms"]["iqr"], 0.0);
    assert!(acceptance::markdown(&report).contains("synthetic_no_claim"));
}
#[test]
fn issue_m5_03_cuda_performance_acceptance_point_failure_and_uncertainty() {
    let (raw, plan, mut bundle) = fixture();
    for s in bundle["rows"][0]["sessions"].as_array_mut().unwrap() {
        s["generated"]["generated_result"]["device_resident_ms"] = json!(vec![1.01; 30]);
    }
    assert_eq!(result(&raw, &plan, &bundle)["status"], "failed");
    for (i, s) in bundle["rows"][0]["sessions"]
        .as_array_mut()
        .unwrap()
        .iter_mut()
        .enumerate()
    {
        s["generated"]["generated_result"]["device_resident_ms"] =
            json!(vec![if i == 2 { 1.2 } else { 0.8 }; 30]);
    }
    let r = result(&raw, &plan, &bundle);
    assert_eq!(r["status"], "pending");
    assert!(r["statistics"]["ratio"].as_f64().unwrap() <= 1.0);
    assert!(r["statistics"]["upper95"].as_f64().unwrap() > 1.05);
}
#[test]
fn issue_m5_03_cuda_performance_acceptance_refusal_mutations() {
    let (raw, plan, bundle) = fixture();
    let paths = [
        "/environment/host",
        "/environment/occupancy",
        "/manifest_sha256",
        "/freeze_status",
        "/generated_result/measurement_scope",
        "/generated_result/full_output_correct",
        "/generated_result/build_review_status",
        "/generated_result/safety_status",
        "/generated_result/numeric_policy_status",
    ];
    for path in paths {
        let mut b = bundle.clone();
        for s in b["rows"][0]["sessions"].as_array_mut().unwrap() {
            *s["generated"].pointer_mut(path).unwrap() = if path.ends_with("occupancy") {
                json!("")
            } else {
                json!("mutated")
            };
        }
        assert_ne!(
            result(&raw, &plan, &b)["status"],
            "passed",
            "survived {path}"
        );
    }
    for unavailable in ["unavailable", "unrecorded", "unknown", "pending"] {
        let mut b = bundle.clone();
        b["rows"][0]["sessions"][0]["generated"]["environment"]["occupancy"] = json!(unavailable);
        assert_ne!(result(&raw, &plan, &b)["status"], "passed");
    }
    let mut missing = bundle.clone();
    missing["rows"][0]["sessions"][0]["baselines"] = json!([]);
    assert_ne!(result(&raw, &plan, &missing)["status"], "passed");
    let mut provisional: Value = serde_json::from_slice(&raw).unwrap();
    provisional["freeze_status"] = json!("provisional");
    let r = acceptance::report(&serde_json::to_vec(&provisional).unwrap(), &plan, &bundle).unwrap();
    assert!(
        r["reasons"]
            .as_array()
            .unwrap()
            .iter()
            .any(|s| s == "M5-01 manifest provisional")
    );
    assert_eq!(r["passed"], 0);
    let mut stale = plan.clone();
    stale["manifest_sha256"] = json!("stale");
    assert!(
        acceptance::report(&raw, &stale, &bundle).unwrap()["reasons"]
            .as_array()
            .unwrap()
            .iter()
            .any(|s| s == "report plan provisional or stale manifest digest")
    );
}
#[test]
fn issue_m5_03_cuda_performance_acceptance_unavailable_cannot_mutate_into_pass() {
    let (raw, plan, mut bundle) = fixture();
    bundle["rows"][0]["status"] = json!("unavailable");
    bundle["rows"][0]["sessions"] = json!([]);
    let report = acceptance::report(&raw, &plan, &bundle).unwrap();
    assert_eq!(report["denominator"], 1);
    assert_eq!(report["passed"], 0);
    assert_eq!(report["counts"]["unavailable"], 1);
    bundle["rows"][0]["status"] = json!("measured");
    bundle["rows"][0]["pass"] = json!(true);
    let mutant = acceptance::report(&raw, &plan, &bundle).unwrap();
    assert_eq!(mutant["passed"], 0);
    assert_eq!(mutant["counts"]["pending"], 1);
    bundle["rows"] = json!([]);
    assert_eq!(
        acceptance::report(&raw, &plan, &bundle).unwrap()["denominator"],
        1
    );
}
#[test]
fn issue_m5_03_cuda_performance_acceptance_fastest_baseline_and_library_tracks() {
    let (raw, mut plan, mut bundle) = fixture();
    plan["expected"][0]["gate"] = json!("library_ratio");
    for s in bundle["rows"][0]["sessions"].as_array_mut().unwrap() {
        s["baselines"][0]["kind"] = json!("library");
        let mut stronger = s["baselines"][0].clone();
        stronger["report"]["baseline_result"]["device_resident_samples_ms"] = json!(vec![0.5; 30]);
        s["baselines"].as_array_mut().unwrap().push(stronger);
    }
    let r = result(&raw, &plan, &bundle);
    assert!((r["statistics"]["ratio"].as_f64().unwrap() - 1.6).abs() < 1e-12);
    assert_eq!(r["status"], "failed");
    bundle["rows"][0]["track"] = json!("library-assisted");
    assert_ne!(result(&raw, &plan, &bundle)["status"], "passed");
}
#[test]
fn issue_m5_03_cuda_performance_acceptance_warm_cold_and_budgets() {
    let (raw, mut plan, bundle) = fixture();
    plan["expected"][0]["scope"] = json!("host_to_host");
    assert!(
        (result(&raw, &plan, &bundle)["statistics"]["ratio"]
            .as_f64()
            .unwrap()
            - 0.75)
            .abs()
            < 1e-12
    );
    plan["expected"][0]["scope"] = json!("cold_process");
    plan["expected"][0]["gate"] = json!("cold");
    assert_eq!(result(&raw, &plan, &bundle)["status"], "passed");
    plan["expected"][0]["scope"] = json!("device_resident");
    plan["expected"][0]["gate"] = json!("resources");
    plan["expected"][0]["workspace_max"] = json!(1024);
    plan["expected"][0]["compile_ms_max"] = json!(2);
    plan["expected"][0]["tuning_ms_max"] = json!(2);
    assert_eq!(result(&raw, &plan, &bundle)["status"], "passed");
    let mut bad = bundle.clone();
    bad["rows"][0]["sessions"][0]["generated"]["generated_result"]["variants"] = json!(33);
    assert_eq!(result(&raw, &plan, &bad)["status"], "failed");
}

#[test]
fn issue_m5_03_cuda_performance_acceptance_bandwidth_compute_heldout_dispatch() {
    let (raw, mut plan, mut bundle) = fixture();
    for s in bundle["rows"][0]["sessions"].as_array_mut().unwrap() {
        s["generated"]["generated_result"]["acceptance_metrics"] = json!({"consecutive_runs":{"bandwidth":vec![0.8;10],"compute":vec![0.5;10],"dispatch":vec![1.1;10]},"copy_bytes_per_ms":1200,"measured_peak_ops_per_ms":1000,"auto_ms":vec![1.1;30],"forced_cpu_ms":vec![2.;30],"forced_gpu_ms":vec![1.;30],"dispatch_overhead_ms":vec![0.100;30],"held_out_sha256":"synthetic-held-out","held_out_bandwidth_fractions":vec![0.8;30],"worst_five_profiler":["synthetic1","synthetic2","synthetic3","synthetic4","synthetic5"]});
    }
    for gate in [
        "bandwidth",
        "held_out_median",
        "held_out_p10",
        "dispatch",
        "dispatch_small",
    ] {
        plan["expected"][0]["gate"] = json!(gate);
        let r = result(&raw, &plan, &bundle);
        assert_eq!(r["status"], "passed", "{gate}: {r}");
    }
    let mut manifest: Value = serde_json::from_slice(&raw).unwrap();
    manifest["cases"][0]["family"] = json!("matrix");
    manifest["cases"][0]["required_operations"] = json!(400);
    let raw = serde_json::to_vec(&manifest).unwrap();
    let digest = hash::sha256(&raw);
    plan["manifest_sha256"] = json!(digest);
    plan["expected"][0]["gate"] = json!("compute");
    for s in bundle["rows"][0]["sessions"].as_array_mut().unwrap() {
        s["generated"]["manifest_sha256"] = json!(digest);
    }
    assert_eq!(result(&raw, &plan, &bundle)["status"], "passed");
    bundle["rows"][0]["sessions"][0]["generated"]["generated_result"]["device_resident_ms"] =
        json!(vec![1.; 30]);
    assert_eq!(result(&raw, &plan, &bundle)["status"], "failed");
}
#[test]
fn issue_m5_03_cuda_performance_acceptance_cli_preserves_outputs() {
    use std::{fs, process::Command};
    let (raw, plan, bundle) = fixture();
    let dir = std::env::temp_dir().join(format!("m5-03-synthetic-{}", std::process::id()));
    fs::create_dir(&dir).unwrap();
    fs::write(dir.join("manifest.json"), &raw).unwrap();
    fs::write(
        dir.join("plan.json"),
        serde_json::to_vec_pretty(&plan).unwrap(),
    )
    .unwrap();
    fs::write(
        dir.join("results.json"),
        serde_json::to_vec_pretty(&bundle).unwrap(),
    )
    .unwrap();
    let run = || {
        Command::new(env!("CARGO_BIN_EXE_invar-performance-report"))
            .current_dir(&dir)
            .args([
                "--manifest",
                "manifest.json",
                "--plan",
                "plan.json",
                "--results",
                "results.json",
                "--report",
                "report.json",
                "--markdown",
                "report.md",
            ])
            .output()
            .unwrap()
    };
    let output = run();
    assert_eq!(
        output.status.code(),
        Some(2),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let first = fs::read(dir.join("report.json")).unwrap();
    let r: Value = serde_json::from_slice(&first).unwrap();
    assert_eq!(r["status"], "synthetic_no_claim");
    assert_eq!(run().status.code(), Some(2));
    assert_eq!(fs::read(dir.join("report.json")).unwrap(), first);
    fs::remove_file(dir.join("report.json")).unwrap();
    assert_eq!(run().status.code(), Some(2));
    assert!(!dir.join("report.json").exists());
    if let Ok(export) = std::env::var("M5_03_SYNTHETIC_EXPORT") {
        fs::create_dir_all(&export).unwrap();
        for file in ["manifest.json", "plan.json", "results.json", "report.md"] {
            fs::copy(dir.join(file), std::path::Path::new(&export).join(file)).unwrap();
        }
    }
    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn issue_m5_03_cuda_performance_acceptance_missing_near_threshold_and_track_counts() {
    let (raw, mut plan, mut bundle) = fixture();
    plan["expected"][0]["gate"] = json!("bandwidth");
    for s in bundle["rows"][0]["sessions"].as_array_mut().unwrap() {
        s["generated"]["generated_result"]["acceptance_metrics"] =
            json!({"copy_bytes_per_ms":1200});
    }
    assert_eq!(result(&raw, &plan, &bundle)["status"], "pending");
    let r = acceptance::report(&raw, &plan, &bundle).unwrap();
    assert_eq!(r["library_assisted"]["denominator"], 1);
    assert_eq!(r["library_assisted"]["unavailable"], 1);
    assert_eq!(r["library_assisted"]["reported"], 0);
    assert_eq!(r["counts"]["ignored"], 0);
    assert_eq!(r["counts"]["failed"], 0);
    for gate in [
        "compute",
        "held_out_p10",
        "dispatch_small",
        "tc_library_ratio",
    ] {
        let mut omitted = plan.clone();
        omitted["expected"][0]["gate"] = json!(gate);
        assert_ne!(
            acceptance::report(&raw, &omitted, &bundle).unwrap()["status"],
            "passed"
        );
    }
}

#[test]
fn issue_m5_03_cuda_performance_acceptance_typed_dispatch_exception_has_no_parity_credit() {
    let (raw, mut plan, mut bundle) = fixture();
    let mut manifest: Value = serde_json::from_slice(&raw).unwrap();
    manifest["cases"][0]["performance_class"] = json!("placement_control");
    let raw = serde_json::to_vec(&manifest).unwrap();
    let digest = hash::sha256(&raw);
    plan["manifest_sha256"] = json!(digest);
    plan["expected"][0]["gate"] = json!("dispatch");
    plan["expected"][0]["admitted_typed_reasons"] = json!(["DEPENDENCY_LIMITED"]);
    for s in bundle["rows"][0]["sessions"].as_array_mut().unwrap() {
        s["generated"]["manifest_sha256"] = json!(digest);
        s["generated"]["generated_result"]["placement_status"] = json!("passed");
        s["generated"]["generated_result"]["acceptance_metrics"]["typed_reason"] = json!({"code":"DEPENDENCY_LIMITED","review":"accepted","evidence_sha256":"synthetic-reason-evidence"});
    }
    let report = acceptance::report(&raw, &plan, &bundle).unwrap();
    assert_eq!(report["passed"], 0);
    assert_eq!(report["counts"]["exception"], 1);
    assert_eq!(report["rows"][0]["result"]["parity_credit"], false);
    plan["expected"][0]["gate"] = json!("parity");
    assert_ne!(result(&raw, &plan, &bundle)["status"], "exception");
    plan["expected"][0]["gate"] = json!("dispatch");
    plan["expected"][0]["admitted_typed_reasons"] = json!([]);
    assert_eq!(result(&raw, &plan, &bundle)["status"], "pending");
}
#[test]
fn issue_m5_03_cuda_performance_acceptance_prints_gpu_conditions_beside_every_timing() {
    let (raw, plan, bundle) = fixture();
    let report = acceptance::report(&raw, &plan, &bundle).unwrap();
    let conditions = &report["rows"][0]["gpu_conditions"];
    assert_eq!(
        conditions["label"],
        "exclusive GPU (no foreign activity recorded)"
    );
    assert_eq!(conditions["records"].as_array().unwrap().len(), 6);
    assert!(acceptance::markdown(&report).contains("exclusive GPU (no foreign activity recorded)"));

    // Foreign activity in one baseline packet makes the whole row shared.
    let mut shared = bundle.clone();
    shared["rows"][0]["sessions"][1]["baselines"][0]["report"]["environment"]["gpu_foreign_activity"] =
        json!({"record":"synthetic-shared","max_foreign_util_percent":37,"foreign_processes":2});
    let report = acceptance::report(&raw, &plan, &shared).unwrap();
    assert_eq!(
        report["rows"][0]["gpu_conditions"]["label"],
        "shared GPU, max foreign util 37%"
    );
    assert!(acceptance::markdown(&report).contains("shared GPU, max foreign util 37%"));

    // Foreign processes without utilization are still not exclusive.
    let mut idle = bundle.clone();
    idle["rows"][0]["sessions"][0]["generated"]["environment"]["gpu_foreign_activity"]["foreign_processes"] =
        json!(1);
    let report = acceptance::report(&raw, &plan, &idle).unwrap();
    assert_eq!(
        report["rows"][0]["gpu_conditions"]["label"],
        "shared GPU, max foreign util 0%"
    );

    // A timing without the record, or with an invalid one, never passes.
    for mutate in [
        json!(null),
        json!({"record":null,"max_foreign_util_percent":0,"foreign_processes":0}),
        json!({"record":"r","max_foreign_util_percent":101,"foreign_processes":0}),
        json!({"record":"r","max_foreign_util_percent":0}),
    ] {
        let mut b = bundle.clone();
        b["rows"][0]["sessions"][2]["generated"]["environment"]["gpu_foreign_activity"] =
            mutate.clone();
        let report = acceptance::report(&raw, &plan, &b).unwrap();
        assert_ne!(report["rows"][0]["result"]["status"], "passed", "{mutate}");
        assert_eq!(report["rows"][0]["gpu_conditions"]["label"], "unrecorded");
    }
}
