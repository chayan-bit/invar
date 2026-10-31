//! Issue label GNET-05.
use crate::{hash, invar_bin};
use serde_json::{Value, json};
use std::{
    fs,
    path::{Path, PathBuf},
    process::Command,
    time::{SystemTime, UNIX_EPOCH},
};

fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .to_path_buf()
}

fn revision() -> String {
    if let Some(revision) = option_env!("GNET_SOURCE_REVISION") {
        return revision.to_owned();
    }
    let output = Command::new("git")
        .arg("rev-parse")
        .arg("HEAD")
        .current_dir(root())
        .output()
        .unwrap();
    assert!(output.status.success());
    String::from_utf8(output.stdout).unwrap().trim().to_owned()
}

fn temp() -> PathBuf {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let dir = std::env::temp_dir().join(format!("invar-gnet-{}-{nanos}", std::process::id()));
    fs::create_dir(&dir).unwrap();
    dir
}

fn paths() -> (PathBuf, PathBuf) {
    let base = root().join("bench/general-net-v01");
    (base.join("manifest.json"), base.join("parity-spec.json"))
}

fn invoke(args: &[&Path]) -> std::process::Output {
    let mut command = Command::new(env!("CARGO_BIN_EXE_invar-bench"));
    for arg in args {
        command.arg(arg);
    }
    command.output().unwrap()
}

#[test]
fn gnet_validates_all_current_source_observations_with_fresh_binary() {
    let dir = temp();
    let (manifest, spec) = paths();
    let report = dir.join("reference.json");
    let revision = revision();
    let output = invoke(&[
        Path::new("--gnet-validate"),
        Path::new("--manifest"),
        &manifest,
        Path::new("--spec"),
        &spec,
        Path::new("--invar"),
        &invar_bin::invar_bin("invar"),
        Path::new("--source-revision"),
        Path::new(&revision),
        Path::new("--report"),
        &report,
    ]);
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let value: Value = serde_json::from_slice(&fs::read(&report).unwrap()).unwrap();
    assert_eq!(value["case_count"], 15);
    assert_eq!(value["status"], "passed");
    assert_eq!(
        value["validation_scope"],
        "source_literal_corpus_semantics_only"
    );
    assert_eq!(value["gpu_acceptance"], "pending");
    let cases = value["results"].as_array().unwrap();
    assert_eq!(
        cases
            .iter()
            .find(|c| c["id"] == "ordered_trap_first")
            .unwrap()["reference_run"]["trap"]["category"],
        "DIVISION_BY_ZERO"
    );
    assert_eq!(
        cases
            .iter()
            .find(|c| c["id"] == "divergent_prefix")
            .unwrap()["reference_run"]["status"],
        "evaluation_limit"
    );
    assert_eq!(
        cases.iter().find(|c| c["id"] == "alias_conflict").unwrap()["check"]["diagnostic_codes"],
        json!(["OWN_BORROW_CONFLICT"])
    );
    assert_eq!(value["invar_executable_sha256"].as_str().unwrap().len(), 64);
    assert_eq!(
        value["source_revision_provenance"],
        "caller_supplied_unverified"
    );
    assert_eq!(value["executable_source_binding"], "unverified");
    assert!(
        !invoke(&[
            Path::new("--gnet-validate"),
            Path::new("--manifest"),
            &manifest,
            Path::new("--spec"),
            &spec,
            Path::new("--invar"),
            &invar_bin::invar_bin("invar"),
            Path::new("--source-revision"),
            Path::new(&revision),
            Path::new("--report"),
            &report,
        ])
        .status
        .success()
    );
    fs::remove_dir_all(dir).unwrap();
}

fn fixture(ratios: [f64; 3]) -> Value {
    let (manifest_path, spec_path) = paths();
    let manifest_bytes = fs::read(manifest_path).unwrap();
    let spec_bytes = fs::read(spec_path).unwrap();
    let manifest: Value = serde_json::from_slice(&manifest_bytes).unwrap();
    let source = manifest["cases"]
        .as_array()
        .unwrap()
        .iter()
        .find(|c| c["id"] == "branch_tree_a")
        .unwrap();
    let manifest_sha = hash::sha256(&manifest_bytes);
    let spec_sha = hash::sha256(&spec_bytes);
    let mut cases = Vec::new();
    for scope in [
        "device_resident_complete_operation",
        "warm_host_input_to_host_output",
    ] {
        let binding = json!({
            "manifest_sha256":manifest_sha,"parity_spec_sha256":spec_sha,
            "source_sha256":source["source_sha256"],"input_sha256":"a".repeat(64),
            "baseline_sha256":"b".repeat(64),"device_sha256":"c".repeat(64),
            "numeric_contract_sha256":"d".repeat(64),"scope_sha256":hash::sha256(scope.as_bytes())
        });
        let sessions = ratios
            .iter()
            .enumerate()
            .map(|(i, ratio)| {
                json!({
                    "id":format!("session-{i}"),"warmups_invar":10,"warmups_baseline":10,
                    "order_seed":20260928,
                    "pairs":(0..30).map(|block| json!({"block_id":block,
                        "order":if block%2==0 {"invar_first"} else {"baseline_first"},
                        "invar_ms":ratio,"baseline_ms":1.0})).collect::<Vec<_>>()
                })
            })
            .collect::<Vec<_>>();
        cases.push(json!({"case_id":"branch_tree_a","size":"n=7","work_items":7,"device":"cuda",
            "scope":scope,"track":"B/D","binding":binding,"invar_binding":binding,
            "baseline_binding":binding,"evidence":{"kind":"synthetic_fixture"},"sessions":sessions}));
    }
    json!({"schema_version":1,"kind":"gnet_paired_raw_v1",
        "manifest_sha256":manifest_sha,"parity_spec_sha256":spec_sha,"cases":cases})
}

fn evaluate(dir: &Path, raw: &Value, name: &str) -> (std::process::Output, PathBuf) {
    let (manifest, spec) = paths();
    let input = dir.join(format!("{name}-input.json"));
    let report = dir.join(format!("{name}-report.json"));
    let revision = revision();
    fs::write(&input, serde_json::to_vec(raw).unwrap()).unwrap();
    (
        invoke(&[
            Path::new("--gnet-evaluate"),
            Path::new("--manifest"),
            &manifest,
            Path::new("--spec"),
            &spec,
            Path::new("--raw"),
            &input,
            Path::new("--source-revision"),
            Path::new(&revision),
            Path::new("--report"),
            &report,
        ]),
        report,
    )
}

#[test]
fn gnet_statistics_and_provenance_are_bounded_and_never_claim_fixture_parity() {
    let dir = temp();
    for (name, ratios, status) in [
        ("pass", [0.9; 3], "pass"),
        ("fail", [1.1; 3], "fail"),
        ("inconclusive", [0.9, 1.0, 1.1], "inconclusive"),
    ] {
        let (output, report) = evaluate(&dir, &fixture(ratios), name);
        assert!(
            output.status.success(),
            "{name}: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        let value: Value = serde_json::from_slice(&fs::read(&report).unwrap()).unwrap();
        assert_eq!(value["cases"].as_array().unwrap().len(), 2);
        assert_eq!(value["cases"][0]["statistical_status"], status);
        assert_eq!(value["cases"][0]["qualification_status"], "pending");
        assert_eq!(
            value["cases"][0]["qualification_reason"],
            "synthetic_fixture_not_performance_evidence"
        );
        assert_eq!(value["status"], "pending_gpu_acceptance");
        assert_eq!(
            value["source_revision_provenance"],
            "caller_supplied_unverified"
        );
        assert_eq!(value["executable_source_binding"], "unverified");
    }
    let valid = fixture([0.9; 3]);
    let mut altered = valid.clone();
    altered["cases"][0]["scope"] = json!("wrong_scope");
    assert!(!evaluate(&dir, &altered, "bad-scope").0.status.success());
    altered = valid.clone();
    altered["cases"][0]["baseline_binding"]["input_sha256"] = json!("e".repeat(64));
    assert!(!evaluate(&dir, &altered, "bad-contract").0.status.success());
    altered = valid.clone();
    altered["manifest_sha256"] = json!("f".repeat(64));
    assert!(!evaluate(&dir, &altered, "bad-digest").0.status.success());
    altered = valid.clone();
    altered["cases"].as_array_mut().unwrap().pop();
    assert!(!evaluate(&dir, &altered, "missing-scope").0.status.success());
    altered = valid.clone();
    altered["cases"][0]["work_items"] = json!(100_000_001);
    assert!(
        !evaluate(&dir, &altered, "oversized-work")
            .0
            .status
            .success()
    );
    altered = valid.clone();
    altered["cases"]
        .as_array_mut()
        .unwrap()
        .push(valid["cases"][0].clone());
    assert!(
        !evaluate(&dir, &altered, "duplicate-case")
            .0
            .status
            .success()
    );
    altered = valid.clone();
    let other_id = altered["cases"][0]["sessions"][1]["id"].clone();
    altered["cases"][0]["sessions"][0]["id"] = other_id;
    assert!(
        !evaluate(&dir, &altered, "duplicate-session")
            .0
            .status
            .success()
    );
    altered = valid.clone();
    altered["cases"][0]["sessions"][0]["pairs"][1]["block_id"] = json!(0);
    assert!(
        !evaluate(&dir, &altered, "duplicate-block")
            .0
            .status
            .success()
    );
    altered = valid.clone();
    altered["cases"][0]["sessions"]
        .as_array_mut()
        .unwrap()
        .pop();
    assert!(
        !evaluate(&dir, &altered, "missing-session")
            .0
            .status
            .success()
    );
    altered = valid.clone();
    altered["cases"][0]["sessions"][0]["pairs"]
        .as_array_mut()
        .unwrap()
        .pop();
    assert!(!evaluate(&dir, &altered, "missing-block").0.status.success());
    for (i, bad) in [json!(0), json!(f64::NAN), json!(f64::INFINITY)]
        .into_iter()
        .enumerate()
    {
        altered = valid.clone();
        altered["cases"][0]["sessions"][0]["pairs"][0]["invar_ms"] = bad;
        assert!(
            !evaluate(&dir, &altered, &format!("bad-time-{i}"))
                .0
                .status
                .success()
        );
    }
    let (output, report) = evaluate(&dir, &valid, "immutable");
    assert!(output.status.success());
    let original = fs::read(&report).unwrap();
    let (second, _) = evaluate(&dir, &valid, "immutable");
    assert!(!second.status.success());
    assert_eq!(fs::read(report).unwrap(), original);
    fs::remove_dir_all(dir).unwrap();
}
