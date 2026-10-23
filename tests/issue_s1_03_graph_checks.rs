use std::{path::Path, process::Command};

fn run(command: &str, file: &Path) -> std::process::Output {
    let mut process = Command::new(env!("CARGO_BIN_EXE_invar"));
    process.args(["system", command, file.to_str().unwrap()]);
    if command == "check" {
        process.args(["--format", "json"]);
    }
    process.output().unwrap()
}

fn fixture(name: &str) -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/s1_03")
        .join(name)
}

#[test]
fn issue_s1_03_graph_checks_twelve_component_cell_and_golden_report() {
    let path = fixture("cell-12.inv");
    let first = run("check", &path);
    let second = run("check", &path);
    assert!(
        first.status.success(),
        "{}",
        String::from_utf8_lossy(&first.stdout)
    );
    assert_eq!(first.stdout, second.stdout);
    let diagnostics: serde_json::Value = serde_json::from_slice(&first.stdout).unwrap();
    assert_eq!(diagnostics["schema"], "invar-diagnostics-v3");
    assert!(
        diagnostics["diagnostics"]
            .as_array()
            .unwrap()
            .iter()
            .all(|d| d["severity"] == "warning")
    );
    assert!(diagnostics.get("checks_pending").is_none());

    let report = run("report", &path);
    assert!(report.status.success());
    assert_eq!(report.stdout, run("report", &path).stdout);
    assert_eq!(
        report.stdout,
        include_bytes!("fixtures/s1_03/cell-12.report.md")
    );
    let ir = run("ir", &path);
    assert!(ir.status.success());
    assert_eq!(ir.stdout, run("ir", &path).stdout);
    assert_eq!(ir.stdout, include_bytes!("fixtures/s1_03/cell-12.ir.json"));
    let document: serde_json::Value = serde_json::from_slice(&ir.stdout).unwrap();
    let schema: serde_json::Value = serde_json::from_str(invar::system::SCHEMA_V2_JSON).unwrap();
    assert!(jsonschema::is_valid(&schema, &document));
    let model = invar::system::SystemIr::from_json(&ir.stdout).unwrap();
    assert_eq!(
        format!("{}\n", model.canonical_json()).as_bytes(),
        ir.stdout
    );
    assert!(
        !model.graph_ready(),
        "deserializing JSON does not establish a source check"
    );
    assert!(model.checked_document().is_err());
    let checked =
        invar::system::compile(invar::front::parse_system(&std::fs::read(&path).unwrap()).unwrap())
            .unwrap();
    assert!(checked.graph_ready());
    assert!(checked.checked_document().is_ok());
    let invalid = fixture("negative/S1250-cpu-over.inv");
    let invalid = invar::system::compile(
        invar::front::parse_system(&std::fs::read(invalid).unwrap()).unwrap(),
    )
    .unwrap();
    assert!(invalid.checked_document().is_err());
    let mut unsupported = document.clone();
    unsupported["schema"] = serde_json::json!("invar-system-v7");
    // V7 is now admitted, but relabeling a legacy document without the new
    // version/contract fields must still fail structural validation.
    assert_eq!(
        invar::system::SystemIr::from_json(unsupported.to_string().as_bytes()).unwrap_err(),
        "invalid system IR model"
    );
    unsupported["schema"] = serde_json::json!("invar-system-v9");
    assert_eq!(
        invar::system::SystemIr::from_json(unsupported.to_string().as_bytes()).unwrap_err(),
        "invalid system IR model"
    );
    unsupported["schema"] = serde_json::json!("invar-system-v13");
    assert_eq!(
        invar::system::SystemIr::from_json(unsupported.to_string().as_bytes()).unwrap_err(),
        "unsupported system IR version"
    );
    assert_eq!(
        document["system"]["instances"].as_array().unwrap().len(),
        12
    );
    let mount = &document["components"][0]["params"]["camera_mount"];
    assert_eq!(mount["kind"], "identity-transform");
    assert_eq!(mount["from"], "base");
    assert_eq!(mount["to"], "world");
    assert_eq!(mount["translation"], serde_json::json!(["0", "0", "0"]));
    assert_eq!(
        mount["rotation_wxyz"],
        serde_json::json!(["1", "0", "0", "0"])
    );
}

#[test]
fn issue_s1_03_graph_checks_rejects_unsupported_transform_constant_form() {
    let output = run("check", &fixture("unsupported-transform-constant.inv"));
    assert_eq!(output.status.code(), Some(1));
    let diagnostics: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(diagnostics["diagnostics"][0]["code"], "S1000");
    assert!(diagnostics["diagnostics"][0]["span"].is_object());
}

#[test]
fn issue_s1_03_graph_checks_v1_remains_readable_with_graph_work_pending() {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/s1_01/basic-cell.inv");
    let check = run("check", &path);
    assert!(check.status.success());
    let document: serde_json::Value = serde_json::from_slice(&check.stdout).unwrap();
    assert_eq!(
        document["checks_pending"],
        serde_json::json!(["system-v2-graph-checks"])
    );
    let ir = run("ir", &path);
    assert!(ir.status.success());
    assert_eq!(
        ir.stdout,
        include_bytes!("fixtures/s1_01/basic-cell.ir.json")
    );
    let model = invar::system::SystemIr::from_json(&ir.stdout).unwrap();
    assert!(!model.graph_ready());
    assert!(model.checked_document().is_err());
    let report = run("report", &path);
    assert_eq!(report.status.code(), Some(2));
    let error: serde_json::Value = serde_json::from_slice(&report.stdout).unwrap();
    assert_eq!(error["diagnostics"][0]["code"], "UNSUPPORTED_FEATURE");
}

#[test]
fn issue_s1_03_graph_checks_policy_and_flow_boundaries_pass() {
    for name in [
        "graph-base",
        "policy-sample-boundary",
        "policy-decimate-boundary",
    ] {
        let path = fixture(&format!("{name}.inv"));
        let result = run("check", &path);
        assert!(
            result.status.success(),
            "{name}: {}",
            String::from_utf8_lossy(&result.stdout)
        );
        assert_eq!(result.stdout, run("check", &path).stdout);
        assert!(run("report", &path).status.success(), "{name}");
    }
}

#[test]
fn issue_s1_03_graph_checks_license_lattice_positive_cases_pass() {
    for name in [
        "license-apache",
        "license-bsd2",
        "license-bsd3",
        "license-mit",
        "license-lgpl-dynamic",
        "license-gpl-process",
        "license-mpl",
        "license-internal-gpl",
        "license-proprietary-grant",
    ] {
        let path = fixture(&format!("{name}.inv"));
        let result = run("check", &path);
        assert!(
            result.status.success(),
            "{name}: {}",
            String::from_utf8_lossy(&result.stdout)
        );
    }
}

#[test]
fn issue_s1_03_graph_checks_negative_fixtures_emit_registered_source_spans() {
    let root = fixture("negative");
    let registry: serde_json::Value =
        serde_json::from_str(include_str!("../schema/diagnostic-codes-system-v2.json")).unwrap();
    let codes: std::collections::HashSet<_> = registry["codes"]
        .as_array()
        .unwrap()
        .iter()
        .map(|entry| entry["code"].as_str().unwrap())
        .collect();
    let diagnostic_schema: serde_json::Value =
        serde_json::from_str(include_str!("../schema/invar-diagnostics-v3.schema.json")).unwrap();
    let mut files: Vec<_> = std::fs::read_dir(root)
        .unwrap()
        .map(|entry| entry.unwrap().path())
        .collect();
    files.sort();
    assert!(files.len() >= 20);
    for file in files {
        let expected = file
            .file_stem()
            .unwrap()
            .to_str()
            .unwrap()
            .split('-')
            .next()
            .unwrap();
        let result = run("check", &file);
        let document: serde_json::Value = serde_json::from_slice(&result.stdout).unwrap();
        assert!(
            jsonschema::is_valid(&diagnostic_schema, &document),
            "{}: {document}",
            file.display()
        );
        assert_eq!(
            result.status.code(),
            Some(1),
            "{}: {document}",
            file.display()
        );
        assert!(codes.contains(expected), "unregistered {expected}");
        assert!(
            document["diagnostics"]
                .as_array()
                .unwrap()
                .iter()
                .any(|diagnostic| {
                    diagnostic["code"] == expected && diagnostic["span"].is_object()
                }),
            "{}: {document}",
            file.display()
        );
    }
}

#[test]
fn issue_s1_03_graph_checks_diagnostic_families_keep_stable_order() {
    let path = fixture("negative/S1100-multi-family-order.inv");
    let first = run("check", &path);
    assert_eq!(first.status.code(), Some(1));
    assert_eq!(first.stdout, run("check", &path).stdout);
    let document: serde_json::Value = serde_json::from_slice(&first.stdout).unwrap();
    let codes: Vec<_> = document["diagnostics"]
        .as_array()
        .unwrap()
        .iter()
        .map(|diagnostic| diagnostic["code"].as_str().unwrap())
        .collect();
    assert_eq!(codes, ["S1100", "S1250", "S1303"]);
}

#[test]
fn issue_s1_03_graph_checks_canonical_c_layout_sizes_cover_robotics_records() {
    let cases = [
        ("Quantity<i32,m>", 4),
        ("std::robotics::Vector3<f64,m,world>", 24),
        ("std::robotics::Pose<f64,world,base>", 56),
        ("std::robotics::JointState<f64,7>", 200),
        ("std::robotics::JointWaypoint<f64,7>", 120),
        ("std::robotics::JointTrajectory<f64,7>", 7688),
        ("std::robotics::CameraIntrinsics<f64>", 88),
        (
            "std::robotics::Stamped<std::robotics::Pose<f64,world,base>>",
            72,
        ),
    ];
    for (ty, bytes) in cases {
        assert_eq!(invar::robotics::c_layout_size(ty), Some(bytes), "{ty}");
    }
    assert_eq!(invar::robotics::c_layout_size("UserPacket"), None);
}
