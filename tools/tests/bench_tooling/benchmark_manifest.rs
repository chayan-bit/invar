use serde_json::{Value, json};
use std::{
    fs,
    path::PathBuf,
    process::Command,
    sync::atomic::{AtomicUsize, Ordering},
    time::{SystemTime, UNIX_EPOCH},
};

static FIXTURE_ID: AtomicUsize = AtomicUsize::new(0);

fn fixture() -> PathBuf {
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let dir = std::env::temp_dir().join(format!(
        "invar-m5-01-{}-{nonce}-{}",
        std::process::id(),
        FIXTURE_ID.fetch_add(1, Ordering::Relaxed)
    ));
    fs::create_dir(&dir).unwrap();
    let output = Command::new(env!("CARGO_BIN_EXE_invar-bench"))
        .current_dir(&dir)
        .arg("--write-artifacts")
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    dir
}

fn validate(dir: &PathBuf, track: &str, gpu: bool) -> std::process::Output {
    let mut command = Command::new(env!("CARGO_BIN_EXE_invar-bench"));
    command.current_dir(dir).args([
        "--manifest",
        "bench/manifest-v1.json",
        "--track",
        track,
        "--report",
        "report.json",
    ]);
    command.arg(if gpu {
        "--require-gpu"
    } else {
        "--validate-only"
    });
    command.output().unwrap()
}

#[test]
fn v2_byte_accounting_proposal_preserves_v1_and_corpus() {
    let dir = fixture();
    let manifest = dir.join("bench/manifest-v1.json");
    let corpus = dir.join("bench/corpus-v1.json");
    let old_manifest = fs::read(&manifest).unwrap();
    let old_corpus = fs::read(&corpus).unwrap();
    let command = || {
        Command::new(env!("CARGO_BIN_EXE_invar-bench"))
            .current_dir(&dir)
            .args([
                "--manifest",
                "bench/manifest-v1.json",
                "--write-v2-proposal",
            ])
            .output()
            .unwrap()
    };
    let first = command();
    assert!(
        first.status.success(),
        "{}",
        String::from_utf8_lossy(&first.stderr)
    );
    let proposal_path = dir.join("bench/manifest-v2-proposal.json");
    let proposal_raw = fs::read(&proposal_path).unwrap();
    assert_eq!(
        proposal_raw,
        include_bytes!("../../bench/manifest-v2-proposal.json")
    );
    let proposal: Value = serde_json::from_slice(&proposal_raw).unwrap();
    assert_eq!(proposal["schema_version"], 2);
    assert_eq!(proposal["report_schema_version"], 2);
    let v1: Value = serde_json::from_slice(&old_manifest).unwrap();
    assert_eq!(proposal["corpus_sha256"], v1["corpus_sha256"]);
    assert_eq!(proposal["held_out_sha256"], v1["held_out_sha256"]);
    let cases = proposal["cases"].as_array().unwrap();
    assert_eq!(
        cases.iter().find(|c| c["id"] == "reduction-n33").unwrap()["required_bytes"],
        136
    );
    assert_eq!(
        cases.iter().find(|c| c["id"] == "scan-n33").unwrap()["required_bytes"],
        396
    );
    assert_eq!(
        proposal["baseline_fixture_sources"]["metal_mpsgraph_reduction_v2_sha256"]
            .as_str()
            .unwrap()
            .len(),
        64
    );
    let validate_v2 = Command::new(env!("CARGO_BIN_EXE_invar-bench"))
        .current_dir(&dir)
        .args([
            "--manifest",
            "bench/manifest-v2-proposal.json",
            "--track",
            "generated",
            "--validate-only",
            "--report",
            "v2-validation.json",
        ])
        .output()
        .unwrap();
    assert!(validate_v2.status.success());
    let v2_report: Value =
        serde_json::from_slice(&fs::read(dir.join("v2-validation.json")).unwrap()).unwrap();
    assert_eq!(v2_report["schema_version"], 2);
    assert_eq!(v2_report["freeze_status"], "proposal_pending_m5_review");
    let mut altered = proposal.clone();
    altered["cases"][0]["required_bytes"] = json!(123);
    fs::write(&proposal_path, serde_json::to_vec_pretty(&altered).unwrap()).unwrap();
    let reject_v2 = Command::new(env!("CARGO_BIN_EXE_invar-bench"))
        .current_dir(&dir)
        .args([
            "--manifest",
            "bench/manifest-v2-proposal.json",
            "--track",
            "generated",
            "--validate-only",
            "--report",
            "v2-rejected.json",
        ])
        .output()
        .unwrap();
    assert!(!reject_v2.status.success());
    fs::write(&proposal_path, &proposal_raw).unwrap();
    assert!(!command().status.success());
    assert_eq!(fs::read(proposal_path).unwrap(), proposal_raw);
    assert_eq!(fs::read(manifest).unwrap(), old_manifest);
    assert_eq!(fs::read(corpus).unwrap(), old_corpus);
    fs::remove_dir_all(dir).unwrap();
}

/// Provisional revision: the seven contract-table families are appended with
/// `unavailable` cases, baselines and targets; the held-out hash, freeze status
/// and the pending v2 proposal are untouched; no unavailable entry is measured.
#[test]
fn benchmark_manifest_provisional_families_are_unavailable_and_never_measured() {
    let dir = fixture();
    let path = dir.join("bench/manifest-v1.json");
    let raw = fs::read(&path).unwrap();
    assert_eq!(raw, include_bytes!("../../bench/manifest-v1.json"));
    let manifest: Value = serde_json::from_slice(&raw).unwrap();
    let v2: Value =
        serde_json::from_slice(include_bytes!("../../bench/manifest-v2-proposal.json")).unwrap();
    assert_eq!(manifest["freeze_status"], "provisional-pre-m4-01");
    assert_eq!(manifest["held_out_sha256"], v2["held_out_sha256"]);
    assert_eq!(manifest["held_out_freeze"]["status"], "deferred");
    assert!(
        manifest["held_out_freeze"]["condition"]
            .as_str()
            .unwrap()
            .contains("after the M4-02..M4-05 device paths exist")
    );
    assert_eq!(v2["freeze_status"], "proposal_pending_m5_review");
    assert_eq!(v2["families"], json!(["map", "reduction", "scan"]));
    assert!(manifest.get("pending_families").is_none());
    let provisional = [
        "compaction",
        "transpose",
        "stencil",
        "matrix",
        "segmented",
        "gather_scatter",
        "frontier",
    ];
    let mut families = vec!["map", "reduction", "scan"];
    families.extend(provisional);
    assert_eq!(manifest["families"], json!(families));
    let cases = manifest["cases"].as_array().unwrap();
    let reasoned = |entry: &Value| {
        entry["status"] == "unavailable" && entry["reason"].as_str().is_some_and(|r| r.len() > 20)
    };
    for family in provisional {
        let family_cases: Vec<_> = cases.iter().filter(|c| c["family"] == family).collect();
        assert!(!family_cases.is_empty(), "{family}");
        assert!(family_cases.iter().all(|c| reasoned(c)), "{family}");
        for backend in ["cuda", "metal"] {
            assert!(
                reasoned(&manifest["baselines"][backend][family]),
                "{family} {backend}"
            );
        }
        let targets = manifest["provisional_families"][family]["targets"]
            .as_array()
            .unwrap();
        assert_eq!(targets.len(), 2);
        assert!(targets.iter().all(reasoned), "{family}");
    }
    // Measured base families keep their earlier, non-unavailable entries.
    assert!(
        cases
            .iter()
            .filter(|c| ["map", "reduction", "scan"].contains(&c["family"].as_str().unwrap()))
            .all(|c| c.get("status").is_none())
    );
    // Shapes come only from the frozen contract table.
    let shapes = |family: &str| -> Vec<Value> {
        cases
            .iter()
            .filter(|c| c["family"] == family)
            .map(|c| c["shape"].clone())
            .collect()
    };
    assert_eq!(
        shapes("compaction"),
        [0, 1, 50, 99, 100].map(|p| json!({"selectivity_percent": p}))
    );
    assert_eq!(
        shapes("transpose"),
        [
            json!({"rows":31,"cols":33}),
            json!({"rows":127,"cols":257}),
            json!({"rows":1024,"cols":1024}),
            json!({"class":"larger bounded cases"})
        ]
    );
    assert!(
        cases
            .iter()
            .any(|c| c["id"] == "matrix-dim-4096"
                && c["shape"]["condition"] == "where budget permits")
    );

    let valid = validate(&dir, "generated", false);
    assert!(
        valid.status.success(),
        "{}",
        String::from_utf8_lossy(&valid.stderr)
    );
    let report: Value =
        serde_json::from_slice(&fs::read(dir.join("report.json")).unwrap()).unwrap();
    assert_eq!(report["freeze_status"], "provisional-pre-m4-01");
    assert_eq!(report["case_counts"]["transpose"], 4);
    assert_eq!(report["case_counts"].as_object().unwrap().len(), 10);

    // Any edit that makes an unavailable entry look measurable is refused.
    assert_eq!(cases[117]["id"], "compaction-selectivity-0pct");
    let mutations: [fn(&mut Value); 4] = [
        |v| v["cases"][117]["status"] = json!("available"),
        |v| v["baselines"]["cuda"]["transpose"]["status"] = json!("measured"),
        |v| v["provisional_families"]["stencil"]["targets"][0]["status"] = json!("available"),
        |v| v["held_out_freeze"]["status"] = json!("frozen"),
    ];
    for (index, mutate) in mutations.into_iter().enumerate() {
        let mut changed = manifest.clone();
        mutate(&mut changed);
        assert_ne!(changed, manifest);
        fs::write(&path, serde_json::to_vec(&changed).unwrap()).unwrap();
        let report = format!("mutated-{index}.json");
        let output = Command::new(env!("CARGO_BIN_EXE_invar-bench"))
            .current_dir(&dir)
            .args([
                "--manifest",
                "bench/manifest-v1.json",
                "--track",
                "generated",
            ])
            .args(["--validate-only", "--report", &report])
            .output()
            .unwrap();
        assert!(!output.status.success(), "mutation {index}");
        assert!(!dir.join(report).exists());
    }
    fs::write(&path, &raw).unwrap();

    // An unavailable case is never run as a generated measurement.
    let measured = Command::new(env!("CARGO_BIN_EXE_invar-bench"))
        .current_dir(&dir)
        .args([
            "--manifest",
            "bench/manifest-v1.json",
            "--track",
            "generated",
        ])
        .args(["--require-gpu", "--generated-case", "transpose-31x33"])
        .args(["--report", "measured.json"])
        .output()
        .unwrap();
    assert!(!measured.status.success());
    assert!(String::from_utf8_lossy(&measured.stderr).contains("unavailable"));
    assert!(!dir.join("measured.json").exists());

    // Nor associated with a baseline report as a measured comparison.
    fs::write(
        dir.join("generated.json"),
        serde_json::to_vec(&json!({"generated_result":{"case_id":"stencil-tiny-domains"}}))
            .unwrap(),
    )
    .unwrap();
    fs::write(dir.join("baseline.json"), b"{}").unwrap();
    let associated = Command::new(env!("CARGO_BIN_EXE_invar-bench"))
        .current_dir(&dir)
        .args([
            "--manifest",
            "bench/manifest-v1.json",
            "--track",
            "generated",
        ])
        .args(["--associate-generated", "generated.json"])
        .args(["--baseline-report", "baseline.json"])
        .args(["--report", "association.json"])
        .output()
        .unwrap();
    assert!(!associated.status.success());
    let association: Value =
        serde_json::from_slice(&fs::read(dir.join("association.json")).unwrap()).unwrap();
    assert_eq!(association["status"], "pending");
    assert!(
        association["reason"]
            .as_str()
            .unwrap()
            .starts_with("manifest entry for stencil-tiny-domains is unavailable")
    );
    assert!(association["association"].is_null());
    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn rejects_mismatched_saved_report_lineage() {
    let dir = fixture();
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let generated = root
        .join("tests/fixtures/archived-evidence/parallel-benchmark-baselines/generated-metal-scan-n33-cold.json");
    let baseline = root
        .join("tests/fixtures/archived-evidence/parallel-benchmark-baselines/metal-mpsgraph-affine-n33.report.json");
    let command = || {
        Command::new(env!("CARGO_BIN_EXE_invar-bench"))
            .current_dir(&dir)
            .arg("--manifest")
            .arg("bench/manifest-v1.json")
            .arg("--track")
            .arg("generated")
            .arg("--associate-generated")
            .arg(&generated)
            .arg("--baseline-report")
            .arg(&baseline)
            .arg("--report")
            .arg("association.json")
            .output()
            .unwrap()
    };
    let first = command();
    assert!(!first.status.success());
    let path = dir.join("association.json");
    let raw = fs::read(&path).unwrap();
    let report: Value = serde_json::from_slice(&raw).unwrap();
    assert_eq!(report["kind"], "benchmark_association");
    assert_eq!(report["status"], "pending");
    assert_eq!(
        report["reason"],
        "association schema or manifest lineage mismatch"
    );
    assert!(report["association"].is_null());
    assert!(!command().status.success());
    assert_eq!(fs::read(path).unwrap(), raw);
    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn v2_rejects_reports_from_prior_proposal_digest() {
    let dir = fixture();
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let proposal = Command::new(env!("CARGO_BIN_EXE_invar-bench"))
        .current_dir(&dir)
        .args([
            "--manifest",
            "bench/manifest-v1.json",
            "--write-v2-proposal",
        ])
        .output()
        .unwrap();
    assert!(proposal.status.success());
    let generated =
        root.join("tests/fixtures/archived-evidence/parallel-benchmark-baselines/v2-generated-reduction-n33.json");
    let baseline = root
        .join("tests/fixtures/archived-evidence/parallel-benchmark-baselines/v2-mpsgraph-reduction-n33-metal.json");
    let build =
        root.join("tests/fixtures/archived-evidence/parallel-benchmark-baselines/v2-generated-build-record.json");
    let output = Command::new(env!("CARGO_BIN_EXE_invar-bench"))
        .current_dir(&dir)
        .arg("--manifest")
        .arg("bench/manifest-v2-proposal.json")
        .arg("--track")
        .arg("generated")
        .arg("--associate-generated")
        .arg(&generated)
        .arg("--baseline-report")
        .arg(&baseline)
        .arg("--generated-build-record")
        .arg(&build)
        .arg("--report")
        .arg("association.json")
        .output()
        .unwrap();
    assert!(!output.status.success());
    let report: Value =
        serde_json::from_slice(&fs::read(dir.join("association.json")).unwrap()).unwrap();
    assert_eq!(report["schema_version"], 2);
    assert_eq!(report["status"], "pending");
    assert_eq!(
        report["reason"],
        "association schema or manifest lineage mismatch"
    );
    assert!(report["association"].is_null());
    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn v2_clean_cold_pair_remains_review_pending() {
    let dir = fixture();
    let prepare = Command::new(env!("CARGO_BIN_EXE_invar-bench"))
        .current_dir(&dir)
        .args([
            "--manifest",
            "bench/manifest-v1.json",
            "--write-v2-proposal",
        ])
        .output()
        .unwrap();
    assert!(prepare.status.success());
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/archived-evidence/parallel-benchmark-baselines/v2-clean-n33-25c5083");
    let generated = root.join("generated-n33.json");
    let baseline = root.join("baseline-n33.json");
    let build = root.join("generated-build-record.json");
    let g: Value = serde_json::from_slice(&fs::read(&generated).unwrap()).unwrap();
    let b: Value = serde_json::from_slice(&fs::read(&baseline).unwrap()).unwrap();
    assert_eq!(g["checkout_at_invocation"]["worktree"], "clean");
    assert_eq!(b["checkout_at_invocation"]["worktree"], "clean");
    assert_eq!(
        g["generated_result"]["input_sha256"],
        b["baseline_result"]["input_sha256"]
    );
    assert_eq!(
        g["generated_result"]["source_observable_output_sha256"],
        b["baseline_result"]["output_sha256"]
    );
    for report in [&g, &b] {
        assert_eq!(report["cold_process"]["status"], "passed");
        assert_eq!(report["cold_process"]["attempts_completed"], 40);
        assert_eq!(
            report["cold_process"]["samples_ms"]
                .as_array()
                .unwrap()
                .len(),
            30
        );
    }
    let associate = |build_path: &std::path::Path, output: &str| {
        Command::new(env!("CARGO_BIN_EXE_invar-bench"))
            .current_dir(&dir)
            .arg("--manifest")
            .arg("bench/manifest-v2-proposal.json")
            .arg("--track")
            .arg("generated")
            .arg("--associate-generated")
            .arg(&generated)
            .arg("--baseline-report")
            .arg(&baseline)
            .arg("--generated-build-record")
            .arg(build_path)
            .arg("--report")
            .arg(output)
            .output()
            .unwrap()
    };
    assert!(!associate(&build, "association.json").status.success());
    let associated: Value =
        serde_json::from_slice(&fs::read(dir.join("association.json")).unwrap()).unwrap();
    let comparison = &associated["association"]["comparison"];
    assert_eq!(comparison["candidate_case_id"], "reduction-n33");
    assert_eq!(comparison["status"], "pending");
    assert!(comparison["device_resident_ratio"].is_null());
    assert!(comparison["host_to_host_ratio"].is_null());
    assert!(comparison["cold_process_ratio"].is_null());
    let reasons = comparison["reasons"].as_array().unwrap();
    for unexpected in [
        "clean_build_lineage_missing",
        "cold_process_boundary_or_samples_mismatch",
        "raw_cold_timing_samples_missing_or_invalid",
        "matched_input_and_source_output_digests_missing",
    ] {
        assert!(
            !reasons.iter().any(|reason| reason == unexpected),
            "{unexpected}"
        );
    }
    assert!(
        reasons
            .iter()
            .any(|reason| reason == "cold_process_scope_review_pending")
    );
    assert!(
        reasons
            .iter()
            .any(|reason| reason == "parameterized_extracted_plan_not_source_program")
    );
    let mut wrong_build: Value = serde_json::from_slice(&fs::read(&build).unwrap()).unwrap();
    wrong_build["runner_executable_sha256"] = json!("wrong");
    fs::write(
        dir.join("wrong-build.json"),
        serde_json::to_vec(&wrong_build).unwrap(),
    )
    .unwrap();
    assert!(
        !associate(&dir.join("wrong-build.json"), "rejected.json")
            .status
            .success()
    );
    let rejected: Value =
        serde_json::from_slice(&fs::read(dir.join("rejected.json")).unwrap()).unwrap();
    assert_eq!(rejected["reason"], "v2 generated build record mismatch");
    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn repeated_report_path_preserves_measurement_and_raw_artifact() {
    let dir = fixture();
    let report = dir.join("baseline.json");
    let raw = dir.join("attempt-00.json");
    fs::write(&report, b"original measurement").unwrap();
    fs::write(&raw, b"original raw attempt").unwrap();
    let output = Command::new(env!("CARGO_BIN_EXE_invar-bench"))
        .current_dir(&dir)
        .args([
            "--manifest",
            "bench/manifest-v1.json",
            "--track",
            "library-assisted",
            "--require-gpu",
            "--baseline-executable",
            "missing-fixture",
            "--baseline-family",
            "sum",
            "--baseline-elements",
            "33",
            "--report",
            "baseline.json",
        ])
        .output()
        .unwrap();
    assert!(!output.status.success());
    assert!(String::from_utf8_lossy(&output.stderr).contains("report path exists"));
    for mode in ["--validate-only", "--preflight"] {
        let output = Command::new(env!("CARGO_BIN_EXE_invar-bench"))
            .current_dir(&dir)
            .args([
                "--manifest",
                "bench/manifest-v1.json",
                "--track",
                "generated",
                mode,
                "--report",
                "baseline.json",
            ])
            .output()
            .unwrap();
        assert!(!output.status.success(), "{mode}");
        assert!(String::from_utf8_lossy(&output.stderr).contains("report path exists"));
        assert_eq!(fs::read(&report).unwrap(), b"original measurement");
        assert_eq!(fs::read(&raw).unwrap(), b"original raw attempt");
    }
    let regenerate = |manifest: &str| {
        Command::new(env!("CARGO_BIN_EXE_invar-bench"))
            .current_dir(&dir)
            .args(["--manifest", manifest, "--write-artifacts"])
            .output()
            .unwrap()
    };
    assert!(!regenerate("baseline.json").status.success());
    assert_eq!(fs::read(&report).unwrap(), b"original measurement");
    assert_eq!(fs::read(&raw).unwrap(), b"original raw attempt");
    assert!(!dir.join("corpus-v1.json").exists());
    let sibling = dir.join("corpus-v1.json");
    fs::write(&sibling, b"existing corpus").unwrap();
    assert!(!regenerate("new-manifest.json").status.success());
    assert_eq!(fs::read(&sibling).unwrap(), b"existing corpus");
    assert!(!dir.join("new-manifest.json").exists());
    fs::remove_dir_all(dir).unwrap();
}

#[cfg(all(feature = "metal", target_os = "macos"))]
#[test]
fn generated_metal_collective_records_verified_raw_samples() {
    if invar_metal::MetalDevice::open().is_err() {
        return;
    }
    let dir = fixture();
    let output = Command::new(env!("CARGO_BIN_EXE_invar-bench"))
        .current_dir(&dir)
        .args([
            "--manifest",
            "bench/manifest-v1.json",
            "--track",
            "generated",
            "--require-gpu",
            "--generated-backend",
            "metal",
            "--generated-case",
            "scan-n33",
            "--report",
            "generated.json",
        ])
        .output()
        .unwrap();
    assert!(!output.status.success(), "full M5 gate remains pending");
    let report: Value =
        serde_json::from_slice(&fs::read(dir.join("generated.json")).unwrap()).unwrap();
    assert_eq!(
        report["status"],
        "partial",
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let result = &report["generated_result"];
    assert_eq!(result["case_id"], "scan-n33");
    assert_eq!(result["full_output_correct"], true);
    assert_eq!(result["device_resident_ms"].as_array().unwrap().len(), 30);
    assert_eq!(result["host_to_host_ms"].as_array().unwrap().len(), 30);
    assert_eq!(result["source_sha256"].as_str().unwrap().len(), 64);
    fs::remove_dir_all(dir).unwrap();
}

/// The generated-track map sampler checks every output word before timing,
/// and refuses (never times) a plan whose output differs from the reference.
#[cfg(all(feature = "metal", target_os = "macos"))]
#[test]
fn generated_metal_map_records_verified_raw_samples() {
    let Ok(mut device) = invar_metal::MetalDevice::open_runtime_source() else {
        return;
    };
    const MAP: &str = "fn main() -> i32 { let xs: Array<u32> = [1u32,2u32,3u32]; \
        var ys: Array<u32> = filled_u32(len(xs),0u32); \
        for i in 0u32 .. len(xs) { ys[i] = wrapping_add(xs[i],5u32); } return 0; }";
    let checked = invar::check::check(invar::front::parse(MAP.as_bytes()).unwrap(), true).unwrap();
    let program = invar::mir::lower(&checked).unwrap();
    let report = invar::admission::analyze(MAP.as_bytes(), &program).unwrap();
    let ir = invar::device_ir::from_region(&report.regions()[0], 3).unwrap();
    let origin = invar_device::Provenance {
        source_identity: ir.source_identity,
        graph_revision: ir.graph_revision,
    };
    let plan = invar_device::verify(ir, origin, device.capabilities()).unwrap();
    let result = device
        .benchmark_generated_map(&plan, &[&[1, 2, 3]], &[vec![6, 7, 8]])
        .unwrap();
    assert_eq!(result.device_resident_ms.len(), 30);
    assert_eq!(result.host_to_host_ms.len(), 30);
    assert!(
        result
            .device_resident_ms
            .iter()
            .chain(&result.host_to_host_ms)
            .all(|ms| ms.is_finite() && *ms > 0.0)
    );
    assert_eq!((result.transfer_bytes, result.launches), (24, 1));
    assert_ne!(result.source_hash, 0);
    assert!(matches!(
        device.benchmark_generated_map(&plan, &[&[1, 2, 3]], &[vec![6, 7, 9]]),
        Err(invar_metal::Error::Device(_))
    ));
    assert!(matches!(
        device.benchmark_generated_map(&plan, &[&[1, 2]], &[vec![6, 7]]),
        Err(invar_metal::Error::Limit)
    ));
}

#[test]
fn benchmark_manifest_validation_and_rejections() {
    let dir = fixture();
    let valid = validate(&dir, "generated", false);
    assert!(
        valid.status.success(),
        "{}",
        String::from_utf8_lossy(&valid.stderr)
    );
    let report: Value =
        serde_json::from_slice(&fs::read(dir.join("report.json")).unwrap()).unwrap();
    assert_eq!(report["kind"], "manifest_validation");
    assert_eq!(report["gpu_performance"], "not_run");
    let revision = &report["checkout_at_invocation"];
    assert!(matches!(
        revision["head"].as_str().map(str::len),
        Some(40 | 64)
    ));
    assert!(["clean", "dirty"].contains(&revision["worktree"].as_str().unwrap()));
    #[cfg(unix)]
    {
        let without_git = Command::new(env!("CARGO_BIN_EXE_invar-bench"))
            .current_dir(&dir)
            .env("PATH", "")
            .args([
                "--manifest",
                "bench/manifest-v1.json",
                "--track",
                "generated",
                "--validate-only",
                "--report",
                "without-git.json",
            ])
            .output()
            .unwrap();
        assert!(without_git.status.success());
        let no_git: Value =
            serde_json::from_slice(&fs::read(dir.join("without-git.json")).unwrap()).unwrap();
        assert_eq!(no_git["checkout_at_invocation"]["head"], Value::Null);
        assert_eq!(no_git["checkout_at_invocation"]["worktree"], "unavailable");
    }
    assert_eq!(report["program_counts"]["total"], 240);
    assert_eq!(report["program_counts"]["held_out"], 60);
    assert!(
        report["case_counts"]
            .as_object()
            .unwrap()
            .values()
            .all(|v| v.as_u64().unwrap() > 0)
    );
    let preflight = Command::new(env!("CARGO_BIN_EXE_invar-bench"))
        .current_dir(&dir)
        .args([
            "--manifest",
            "bench/manifest-v1.json",
            "--track",
            "generated",
            "--preflight",
            "--report",
            "preflight.json",
        ])
        .output()
        .unwrap();
    let preflight_report: Value =
        serde_json::from_slice(&fs::read(dir.join("preflight.json")).unwrap()).unwrap();
    assert_eq!(preflight_report["kind"], "gpu_preflight");
    assert_eq!(
        preflight_report["preflight_scope"],
        "backend_feature_and_device_open_only"
    );
    assert_eq!(preflight_report["gpu_performance"], "not_run");
    assert_eq!(
        preflight_report["manifest_sha256"],
        report["manifest_sha256"]
    );
    assert_eq!(
        preflight.status.success(),
        preflight_report["status"] == "available"
    );
    assert!(preflight_report["gpu_preflight"].as_array().is_some());
    #[cfg(not(any(feature = "metal", feature = "cuda")))]
    assert_eq!(preflight_report["status"], "pending");
    let invalid_preflight = Command::new(env!("CARGO_BIN_EXE_invar-bench"))
        .current_dir(&dir)
        .args([
            "--track",
            "generated",
            "--preflight",
            "--require-gpu",
            "--report",
            "invalid-preflight.json",
        ])
        .output()
        .unwrap();
    assert!(!invalid_preflight.status.success());
    assert!(!dir.join("invalid-preflight.json").exists());
    let report_path = dir.join("report.json");
    let mut stale = report.clone();
    stale["manifest_sha256"] = json!("previous-manifest");
    fs::write(&report_path, serde_json::to_vec(&stale).unwrap()).unwrap();
    assert!(!validate(&dir, "generated", false).status.success());
    let preserved: Value = serde_json::from_slice(&fs::read(&report_path).unwrap()).unwrap();
    assert_eq!(preserved["manifest_sha256"], "previous-manifest");
    fs::write(&report_path, serde_json::to_vec(&report).unwrap()).unwrap();
    fs::write(&report_path, b"{").unwrap();
    assert!(!validate(&dir, "generated", false).status.success());
    assert_eq!(fs::read(&report_path).unwrap(), b"{");
    fs::write(&report_path, serde_json::to_vec(&report).unwrap()).unwrap();

    let path = dir.join("bench/manifest-v1.json");
    let original: Value = serde_json::from_slice(&fs::read(&path).unwrap()).unwrap();
    let mutations: [fn(&mut Value); 6] = [
        |v| {
            v["cases"]
                .as_array_mut()
                .unwrap()
                .retain(|c| c["family"] != "scan");
        },
        |v| {
            let first = v["cases"][0].clone();
            v["cases"].as_array_mut().unwrap().push(first);
        },
        |v| {
            v["seed"] = json!(1);
        },
        |v| {
            v["held_out_sha256"] = json!("tampered");
        },
        |v| {
            v["cases"][0]["numeric_policy"] = json!("strict-f32");
        },
        |v| {
            v["baselines"]["cuda"]["scan"]["independent"] = json!(false);
        },
    ];
    for mutate in mutations {
        let mut changed = original.clone();
        mutate(&mut changed);
        fs::write(&path, serde_json::to_vec(&changed).unwrap()).unwrap();
        assert!(!validate(&dir, "generated", false).status.success());
    }
    fs::write(&path, serde_json::to_vec(&original).unwrap()).unwrap();
    assert!(!validate(&dir, "not-a-track", false).status.success());
    let invalid_smoke = Command::new(env!("CARGO_BIN_EXE_invar-bench"))
        .current_dir(&dir)
        .args([
            "--manifest",
            "bench/manifest-v1.json",
            "--track",
            "generated",
            "--validate-only",
            "--metal-held-out-smoke",
            "--report",
            "invalid-smoke.json",
        ])
        .output()
        .unwrap();
    assert!(!invalid_smoke.status.success());
    assert!(!dir.join("invalid-smoke.json").exists());
    #[cfg(not(feature = "metal"))]
    {
        let unavailable = Command::new(env!("CARGO_BIN_EXE_invar-bench"))
            .current_dir(&dir)
            .args([
                "--manifest",
                "bench/manifest-v1.json",
                "--track",
                "generated",
                "--require-gpu",
                "--metal-held-out-smoke",
                "--report",
                "unavailable-smoke.json",
            ])
            .output()
            .unwrap();
        assert!(!unavailable.status.success());
        let pending: Value =
            serde_json::from_slice(&fs::read(dir.join("unavailable-smoke.json")).unwrap()).unwrap();
        assert_eq!(pending["status"], "pending");
        assert_eq!(pending["gpu_performance"], "not_run");
    }
    fs::remove_file(&report_path).unwrap();
    let gpu = validate(&dir, "generated", true);
    assert!(!gpu.status.success());
    let pending: Value =
        serde_json::from_slice(&fs::read(dir.join("report.json")).unwrap()).unwrap();
    assert_eq!(pending["status"], "pending");
    assert_eq!(pending["gpu_performance"], "not_run");

    let corpus_path = dir.join("bench/corpus-v1.json");
    let mut corpus: Value = serde_json::from_slice(&fs::read(&corpus_path).unwrap()).unwrap();
    corpus["programs"][0]["source"] = json!("fn main() -> i32 { return 0; }");
    fs::write(corpus_path, serde_json::to_vec(&corpus).unwrap()).unwrap();
    assert!(!validate(&dir, "generated", false).status.success());
    fs::remove_dir_all(dir).unwrap();
}

#[cfg(unix)]
#[test]
fn benchmark_manifest_rejects_wrong_baseline_output() {
    use std::os::unix::fs::PermissionsExt;
    let dir = fixture();
    let fake = dir.join("wrong-baseline.sh");
    let packet = json!({"schema_version":1,"baseline":"CUB","vendor_versions":{"cub":101700,"cuda_runtime":11080,"driver":12000},"family":"sum","n":1,"warmups":10,"samples":30,
        "input_bytes":4,"output_bytes":4,"vendor_temporary_bytes":0,"explicit_intermediate_bytes":0,"total_scratch_bytes":0,"workspace_bytes":0,"workspace_scope":"vendor temporary allocation plus explicit fixture intermediate","cold_process_ms":1.0,
        "device_samples_ms":vec![1.0;30],"host_samples_ms":vec![1.0;30]});
    let source = format!(
        "#!/bin/sh\nprintf '\\000\\000\\000\\000' > \"$3\"\nprintf '%s\\n' '{}'\n",
        packet
    );
    fs::write(&fake, source).unwrap();
    fs::set_permissions(&fake, fs::Permissions::from_mode(0o700)).unwrap();
    let output = Command::new(env!("CARGO_BIN_EXE_invar-bench"))
        .current_dir(&dir)
        .args([
            "--manifest",
            "bench/manifest-v1.json",
            "--track",
            "library-assisted",
            "--require-gpu",
            "--baseline-executable",
            fake.to_str().unwrap(),
            "--baseline-family",
            "sum",
            "--baseline-elements",
            "1",
            "--report",
            "report.json",
        ])
        .output()
        .unwrap();
    assert!(!output.status.success());
    let report: Value =
        serde_json::from_slice(&fs::read(dir.join("report.json")).unwrap()).unwrap();
    assert_eq!(
        report["baseline_result"]["full_output_correctness"],
        "exact_full_output_fail"
    );
    assert_eq!(report["status"], "failed");
    assert_eq!(
        report["baseline_result"]["candidate_case_id"],
        "reduction-n1"
    );
    assert_eq!(
        report["baseline_result"]["gate_status"],
        "pending_semantic_quality_build_review"
    );
    let mut packet = packet;
    packet["family"] = json!("scan");
    fs::write(
        &fake,
        format!(
            "#!/bin/sh\nprintf '\\003\\000\\000\\000' > \"$3\"\nprintf '%s\\n' '{}'\n",
            packet
        ),
    )
    .unwrap();
    let scan = Command::new(env!("CARGO_BIN_EXE_invar-bench"))
        .current_dir(&dir)
        .args([
            "--manifest",
            "bench/manifest-v1.json",
            "--track",
            "library-assisted",
            "--require-gpu",
            "--baseline-executable",
            fake.to_str().unwrap(),
            "--baseline-family",
            "scan",
            "--baseline-elements",
            "1",
            "--report",
            "scan-report.json",
        ])
        .output()
        .unwrap();
    assert!(!scan.status.success());
    let scan_report: Value =
        serde_json::from_slice(&fs::read(dir.join("scan-report.json")).unwrap()).unwrap();
    assert_eq!(scan_report["status"], "pending_review");
    assert_eq!(
        scan_report["baseline_result"]["full_output_correctness"],
        "exact_full_output_pass"
    );
    assert_eq!(
        scan_report["baseline_result"]["candidate_case_id"],
        Value::Null
    );
    assert_eq!(
        scan_report["baseline_result"]["gate_status"],
        "outside_current_manifest_or_law"
    );
    packet["baseline"] = json!("Apple MPSGraph reductionSum");
    packet["vendor_versions"] =
        json!({"macos":[27,0,0],"framework":"MetalPerformanceShadersGraph system framework"});
    packet["compilation_ms"] = json!(1.0);
    packet["workspace_scope"] = json!("explicit buffers only; framework internals unknown");
    fs::write(
        &fake,
        format!(
            "#!/bin/sh\nprintf '\\003\\000\\000\\000' > \"$3\"\nprintf '%s\\n' '{}'\n",
            packet
        ),
    )
    .unwrap();
    let spoofed = Command::new(env!("CARGO_BIN_EXE_invar-bench"))
        .current_dir(&dir)
        .args([
            "--manifest",
            "bench/manifest-v1.json",
            "--track",
            "library-assisted",
            "--require-gpu",
            "--baseline-executable",
            fake.to_str().unwrap(),
            "--baseline-family",
            "scan",
            "--baseline-elements",
            "1",
            "--report",
            "spoofed-scan.json",
        ])
        .output()
        .unwrap();
    assert!(!spoofed.status.success());
    assert!(!dir.join("spoofed-scan.json").exists());
    packet["baseline"] = json!("Handwritten Metal affine quality control");
    packet["family"] = json!("affine");
    packet["vendor_versions"] = json!({"macos":[27,0,0],"framework":"Metal system framework"});
    packet["input_bytes"] = json!(8);
    packet["workspace_bytes"] = json!(16);
    packet["launches_per_call"] = json!(2);
    packet["workspace_scope"] = json!("explicit pair prefixes and block totals");
    fs::write(
        &fake,
        format!(
            "#!/bin/sh\nprintf '\\047\\000\\000\\000' > \"$3\"\nprintf '%s\\n' '{}'\n",
            packet
        ),
    )
    .unwrap();
    let control = Command::new(env!("CARGO_BIN_EXE_invar-bench"))
        .current_dir(&dir)
        .args([
            "--manifest",
            "bench/manifest-v1.json",
            "--track",
            "library-assisted",
            "--require-gpu",
            "--baseline-executable",
            fake.to_str().unwrap(),
            "--baseline-family",
            "affine",
            "--baseline-elements",
            "1",
            "--report",
            "quality-control.json",
        ])
        .output()
        .unwrap();
    assert!(!control.status.success());
    let control_report: Value =
        serde_json::from_slice(&fs::read(dir.join("quality-control.json")).unwrap()).unwrap();
    assert_eq!(control_report["kind"], "handwritten_quality_control");
    assert_eq!(
        control_report["baseline_result"]["gate_status"],
        "quality_control_only"
    );
    assert_eq!(
        control_report["baseline_result"]["full_output_correctness"],
        "exact_full_output_pass"
    );
    assert_eq!(
        control_report["baseline_result"]["vendor_temporary_bytes"],
        0
    );
    assert_eq!(
        control_report["baseline_result"]["explicit_intermediate_bytes"],
        16
    );
    assert_eq!(control_report["baseline_result"]["total_scratch_bytes"], 16);
    packet["baseline"] = json!("CUB");
    packet["vendor_versions"] = json!({"cub":200802,"cuda_runtime":12090,"driver":13000});
    packet["workspace_scope"] =
        json!("vendor temporary allocation plus explicit fixture intermediate");
    packet["vendor_temporary_bytes"] = json!(1791);
    packet["explicit_intermediate_bytes"] = json!(8);
    packet["total_scratch_bytes"] = json!(1799);
    packet["workspace_bytes"] = json!(1799);
    let run_affine = |report: &str, packet: &Value| {
        fs::write(
            &fake,
            format!(
                "#!/bin/sh\nprintf '\\047\\000\\000\\000' > \"$3\"\nprintf '%s\\n' '{}'\n",
                packet
            ),
        )
        .unwrap();
        Command::new(env!("CARGO_BIN_EXE_invar-bench"))
            .current_dir(&dir)
            .args([
                "--manifest",
                "bench/manifest-v1.json",
                "--track",
                "library-assisted",
                "--require-gpu",
                "--baseline-executable",
                fake.to_str().unwrap(),
                "--baseline-family",
                "affine",
                "--baseline-elements",
                "1",
                "--report",
                report,
            ])
            .output()
            .unwrap()
    };
    let accepted = run_affine("scratch-ok.json", &packet);
    assert!(!accepted.status.success()); // Review is pending even when the fixture is correct.
    let accepted_report: Value =
        serde_json::from_slice(&fs::read(dir.join("scratch-ok.json")).unwrap()).unwrap();
    assert_eq!(
        accepted_report["baseline_result"]["full_output_correctness"],
        "exact_full_output_pass"
    );
    assert_eq!(
        accepted_report["baseline_result"]["vendor_temporary_bytes"],
        1791
    );
    assert_eq!(
        accepted_report["baseline_result"]["explicit_intermediate_bytes"],
        8
    );
    assert_eq!(
        accepted_report["baseline_result"]["total_scratch_bytes"],
        1799
    );
    packet["explicit_intermediate_bytes"] = json!(0);
    packet["total_scratch_bytes"] = json!(1791);
    packet["workspace_bytes"] = json!(1791);
    assert!(!run_affine("scratch-bad.json", &packet).status.success());
    assert!(!dir.join("scratch-bad.json").exists());
    fs::remove_dir_all(dir).unwrap();
}
