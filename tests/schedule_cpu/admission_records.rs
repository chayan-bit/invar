//! Issue label M2-01.
use invar::{admission, check::check, eval, front::parse, mir};
use std::process::Command;

const SOURCE: &str = r#"
fn main() -> i32 {
    let xs: Array<u32> = [1u32, 2u32, 3u32];
    var ys: Array<u32> = filled_u32(len(xs), 0u32);
    for i in 0u32 .. len(xs) {
        ys[i] = wrapping_add(xs[i], 7u32);
    }
    return 0;
}
"#;

fn lowered(source: &str) -> mir::Program {
    let checked = check(parse(source.as_bytes()).unwrap(), true).unwrap();
    mir::lower(&checked).unwrap()
}

#[test]
fn admission_records_round_trip_and_invalidation() {
    let program = lowered(SOURCE);
    let report = admission::analyze(SOURCE.as_bytes(), &program).unwrap();
    assert_eq!(report.regions().len(), 1);
    let region = &report.regions()[0];
    assert_eq!(region.status(), admission::Status::Admitted);
    assert_eq!(region.fallback_plan(), "mir-serial");
    let encoded = report.to_json();
    let obligations = encoded["regions"][0]["obligations"].as_array().unwrap();
    for code in ["TERMINATION", "FRESH_OUTPUT", "PHYSICAL_BOUNDS"] {
        assert!(
            obligations.iter().any(|item| item["name"] == code),
            "{code}"
        );
    }
    assert!(region.guard().unwrap().evaluate(&[("xs", 3), ("ys", 3)]));
    assert!(!region.guard().unwrap().evaluate(&[("xs", 3), ("ys", 2)]));
    assert_eq!(
        region.guard().unwrap().select_plan(&[("xs", 3), ("ys", 2)]),
        "mir-serial"
    );

    admission::validate_serialized(&encoded, SOURCE.as_bytes(), &program).unwrap();
    assert!(admission::validate_serialized(&encoded, b"changed source", &program).is_err());
    let mut changed_rule = encoded.clone();
    changed_rule["regions"][0]["rule_revision"] = "old-rule".into();
    assert!(admission::validate_serialized(&changed_rule, SOURCE.as_bytes(), &program).is_err());
    for index in 0..encoded["regions"][0]["obligations"]
        .as_array()
        .unwrap()
        .len()
    {
        let mut missing = encoded.clone();
        missing["regions"][0]["obligations"]
            .as_array_mut()
            .unwrap()
            .remove(index);
        assert!(
            admission::validate_serialized(&missing, SOURCE.as_bytes(), &program).is_err(),
            "obligation {index}"
        );
    }
}

#[test]
fn admission_records_rejects_checked_arithmetic() {
    let source = SOURCE.replace("wrapping_add(xs[i], 7u32)", "xs[i] + 7u32");
    let report = admission::analyze(source.as_bytes(), &lowered(&source)).unwrap();
    assert_eq!(report.regions().len(), 1);
    assert_eq!(report.regions()[0].status(), admission::Status::Rejected);
    assert_eq!(
        report.regions()[0].first_rejected_obligation(),
        Some("TOTALITY")
    );
}

#[test]
fn admission_records_rejects_unrelated_index_and_nonfresh_output() {
    let extra = SOURCE.replace(
        "ys[i] = wrapping_add(xs[i], 7u32);",
        "let z: u32 = xs[99u32]; ys[i] = wrapping_add(xs[i], 7u32);",
    );
    let report = admission::analyze(extra.as_bytes(), &lowered(&extra)).unwrap();
    assert_eq!(report.regions()[0].status(), admission::Status::Rejected);

    let nonfresh = SOURCE.replace(
        "var ys: Array<u32> = filled_u32(len(xs), 0u32);",
        "let zs: Array<u32> = filled_u32(len(xs), 0u32); var ys: Array<u32> = move zs;",
    );
    let report = admission::analyze(nonfresh.as_bytes(), &lowered(&nonfresh)).unwrap();
    assert_eq!(report.regions()[0].status(), admission::Status::Rejected);
    assert_eq!(
        report.regions()[0].first_rejected_obligation(),
        Some("FRESH_OUTPUT")
    );
}

#[test]
fn admission_records_explain_cli() {
    let path = std::env::temp_dir().join(format!("invar-m2-01-{}.inv", std::process::id()));
    std::fs::write(&path, SOURCE).unwrap();
    let output = Command::new(env!("CARGO_BIN_EXE_invar"))
        .args(["explain", path.to_str().unwrap(), "--format", "json"])
        .output()
        .unwrap();
    std::fs::remove_file(&path).unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let doc: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(doc["status"], "ok");
    assert_eq!(doc["regions"][0]["status"], "admitted");
    assert_eq!(doc["regions"][0]["fallback_plan"], "mir-serial");
}

#[test]
fn admission_records_reports_unbounded_loop_as_unknown() {
    let source =
        "fn main() -> i32 { var n: u32 = 0u32; while n < 2u32 { n = n + 1u32; } return 0; }";
    let report = admission::analyze(source.as_bytes(), &lowered(source)).unwrap();
    assert_eq!(report.regions().len(), 1);
    assert_eq!(report.regions()[0].status(), admission::Status::Rejected);
    assert_eq!(
        report.regions()[0].first_rejected_obligation(),
        Some("RANGE")
    );
    assert_eq!(
        report.to_json()["regions"][0]["obligations"][0]["evidence"],
        "unknown"
    );
}

#[test]
fn admission_records_rejects_near_matches() {
    let variants = [
        (
            SOURCE.replace("0u32 .. len(xs)", "1u32 .. len(xs)"),
            "RANGE",
        ),
        (SOURCE.replace("ys[i] =", "ys[0u32] ="), "FOOTPRINT"),
        (SOURCE.replace("let xs:", "var xs:"), "OWNERSHIP"),
        (
            SOURCE.replace(
                "ys[i] = wrapping_add(xs[i], 7u32);",
                "ys[i] = wrapping_add(ys[i], 7u32);",
            ),
            "ALIAS",
        ),
        (
            SOURCE.replace(
                "ys[i] = wrapping_add(xs[i], 7u32);",
                "let x: u32 = xs[0u32]; ys[i] = wrapping_add(xs[i], 7u32);",
            ),
            "EFFECT_ORDER",
        ),
        (
            format!(
                "fn step(x: u32) -> u32 {{ return x; }} {}",
                SOURCE.replace("wrapping_add(xs[i], 7u32)", "step(xs[i])")
            ),
            "UNKNOWN_EFFECT",
        ),
    ];
    for (source, reason) in variants {
        let report = admission::analyze(source.as_bytes(), &lowered(&source)).unwrap();
        assert_eq!(
            report.regions()[0].first_rejected_obligation(),
            Some(reason),
            "{source}"
        );
    }
}

#[test]
fn admission_records_allows_read_read_sharing() {
    let source = SOURCE.replace("wrapping_add(xs[i], 7u32)", "wrapping_add(xs[i], xs[i])");
    let report = admission::analyze(source.as_bytes(), &lowered(&source)).unwrap();
    assert_eq!(report.regions()[0].status(), admission::Status::Admitted);
    let facts = &report.to_json()["regions"][0]["facts"];
    assert_eq!(facts["reads"].as_array().unwrap().len(), 2);
    assert_eq!(facts["writes"].as_array().unwrap().len(), 1);
}

#[test]
fn admission_records_false_guard_selects_serial_before_write() {
    let source = SOURCE
        .replace(
            "var ys: Array<u32>",
            "let zs: Array<u32> = [4u32]; var ys: Array<u32>",
        )
        .replace("wrapping_add(xs[i], 7u32)", "wrapping_add(xs[i], zs[i])");
    let report = admission::analyze(source.as_bytes(), &lowered(&source)).unwrap();
    let region = &report.regions()[0];
    assert_eq!(region.status(), admission::Status::Admitted);
    let guard = region.guard().unwrap();
    assert_eq!(
        guard.select_plan(&[("xs", 3), ("zs", 1), ("ys", 3)]),
        "mir-serial"
    );
    assert_eq!(
        guard.select_plan(&[("xs", 3), ("zs", 3), ("ys", 3)]),
        "wrapping-map-candidate"
    );
    let checked = check(parse(source.as_bytes()).unwrap(), true).unwrap();
    assert!(matches!(
        eval::observe(&checked, 10_000).result,
        Err(eval::EvalError::Trap {
            category: eval::TrapCategory::IndexOutOfBounds,
            ..
        })
    ));
}

#[test]
fn admission_records_reports_unsupported_nested_body() {
    let source = SOURCE.replace(
        "ys[i] = wrapping_add(xs[i], 7u32);",
        "if true { ys[i] = wrapping_add(xs[i], 7u32); }",
    );
    let report = admission::analyze(source.as_bytes(), &lowered(&source)).unwrap();
    assert_eq!(report.regions().len(), 1);
    assert_eq!(report.regions()[0].status(), admission::Status::Rejected);
}

#[test]
fn admission_records_offset_indexes_report_dependence() {
    for body in [
        "ys[i] = wrapping_add(xs[i + 1u32], 7u32);",
        "ys[i] = wrapping_add(ys[i - 1u32], 7u32);",
    ] {
        let source = SOURCE.replace("ys[i] = wrapping_add(xs[i], 7u32);", body);
        let program = lowered(&source);
        let report = admission::analyze(source.as_bytes(), &program).unwrap();
        assert_eq!(
            report.regions()[0].first_rejected_obligation(),
            Some("DEPENDENCE"),
            "{body}"
        );
        assert_eq!(report.to_json()["regions"][0]["chosen_plan"], "mir-serial");
    }
    let checked = SOURCE.replace("wrapping_add(xs[i], 7u32)", "xs[i] + 7u32");
    let program = lowered(&checked);
    let report = admission::analyze(checked.as_bytes(), &program).unwrap();
    assert_eq!(
        report.regions()[0].first_rejected_obligation(),
        Some("TOTALITY")
    );
}

#[test]
fn admission_records_cleanup_is_a_trusted_rule() {
    let program = lowered(SOURCE);
    let report = admission::analyze(SOURCE.as_bytes(), &program).unwrap();
    let json = report.to_json();
    let cleanup = json["regions"][0]["obligations"]
        .as_array()
        .unwrap()
        .iter()
        .find(|obligation| obligation["name"] == "CLEANUP")
        .unwrap();
    assert_eq!(cleanup["evidence"], "trusted_rule");
}
