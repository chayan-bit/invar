use std::process::Command;

const FIXTURE: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/s1_01/basic-cell.inv"
);

#[test]
fn issue_s1_01_system_dialect_parses_every_basic_clause() {
    let source = include_bytes!("fixtures/s1_01/basic-cell.inv");
    let parsed = invar::front::parse(source).unwrap();
    assert_eq!(parsed.system_items.len(), 8);
    assert_eq!(parsed.frames.len(), 2);
}

#[test]
fn issue_s1_01_system_dialect_emits_checked_canonical_ir() {
    let command = Command::new(env!("CARGO_BIN_EXE_invar"))
        .args(["system", "ir", FIXTURE])
        .output()
        .unwrap();
    assert!(
        command.status.success(),
        "{}",
        String::from_utf8_lossy(&command.stderr)
    );
    let ir: serde_json::Value = serde_json::from_slice(&command.stdout).unwrap();
    let schema: serde_json::Value = serde_json::from_str(invar::system::SCHEMA_JSON).unwrap();
    assert!(jsonschema::is_valid(&schema, &ir));
    let model = invar::system::SystemIr::from_json(&command.stdout).unwrap();
    assert_eq!(
        format!("{}\n", model.canonical_json()),
        String::from_utf8(command.stdout.clone()).unwrap()
    );
    assert_eq!(
        String::from_utf8(command.stdout.clone()).unwrap(),
        include_str!("fixtures/s1_01/basic-cell.ir.json")
    );
    assert_eq!(ir["schema"], "invar-system-v1");
    assert_eq!(ir["system"]["name"], "cell");
    assert_eq!(ir["system"]["product"], "SimCell");
    assert_eq!(ir["connections"].as_array().unwrap().len(), 2);
    assert_eq!(ir["components"].as_array().unwrap().len(), 2);
    assert_eq!(
        ir["components"][1]["params"]["max_velocity"]["value"],
        "1.5"
    );
    let check = Command::new(env!("CARGO_BIN_EXE_invar"))
        .args(["system", "check", FIXTURE, "--format", "json"])
        .output()
        .unwrap();
    assert!(check.status.success());
    let document: serde_json::Value = serde_json::from_slice(&check.stdout).unwrap();
    assert_eq!(document["schema"], "invar-diagnostics-v1");
    assert_eq!(document["diagnostics"], serde_json::json!([]));
    let second = Command::new(env!("CARGO_BIN_EXE_invar"))
        .args(["system", "ir", FIXTURE])
        .output()
        .unwrap();
    assert_eq!(command.stdout, second.stdout);
}

#[test]
fn issue_s1_01_system_dialect_negative_fixtures_have_registered_spans() {
    let root =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/s1_01/negative");
    let registry: serde_json::Value =
        serde_json::from_str(include_str!("../schema/diagnostic-codes-system-v1.json")).unwrap();
    let codes: std::collections::HashSet<_> = registry["codes"]
        .as_array()
        .unwrap()
        .iter()
        .map(|entry| entry["code"].as_str().unwrap())
        .collect();
    let diagnostic_schema: serde_json::Value =
        serde_json::from_str(include_str!("../schema/invar-diagnostics-v1.schema.json")).unwrap();
    let mut files: Vec<_> = std::fs::read_dir(root)
        .unwrap()
        .map(|entry| entry.unwrap().path())
        .collect();
    files.sort();
    assert!(files.len() >= 18);
    for file in files {
        let path = file.to_str().unwrap();
        let expected = file
            .file_name()
            .unwrap()
            .to_str()
            .unwrap()
            .split('-')
            .next()
            .unwrap();
        let output = Command::new(env!("CARGO_BIN_EXE_invar"))
            .args(["check", path, "--format", "json"])
            .output()
            .unwrap();
        let document: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
        assert!(
            jsonschema::is_valid(&diagnostic_schema, &document),
            "{path}: {document}"
        );
        assert!(codes.contains(expected), "{path}: unregistered {expected}");
        assert_eq!(
            document["diagnostics"][0]["code"], expected,
            "{path}: {document}"
        );
        assert!(
            document["diagnostics"][0]["span"].is_object(),
            "{path}: {document}"
        );
        if expected == "S1011" {
            assert!(output.status.success(), "{path}");
            assert_eq!(document["diagnostics"][0]["severity"], "warning");
        } else {
            assert_eq!(output.status.code(), Some(1), "{path}: {document}");
        }
    }
}

#[test]
fn issue_s1_01_system_dialect_module_permutation_is_byte_identical() {
    let fixture = |name: &str| {
        std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/s1_01")
            .join(name)
    };
    let run = |name: &str| {
        let output = Command::new(env!("CARGO_BIN_EXE_invar"))
            .args(["system", "ir", fixture(name).to_str().unwrap()])
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "{name}: {}",
            String::from_utf8_lossy(&output.stdout)
        );
        output.stdout
    };
    let original = run("module-cell.inv");
    let permuted = run("module-cell-permuted.inv");
    assert_eq!(original, permuted);
    assert_eq!(
        original,
        include_bytes!("fixtures/s1_01/module-cell.ir.json")
    );
    let ir: serde_json::Value = serde_json::from_slice(&original).unwrap();
    assert_eq!(ir["system"]["id"], "demo::cell");
    assert_eq!(ir["components"][1]["id"], "demo::Source");
    let schema: serde_json::Value = serde_json::from_str(invar::system::SCHEMA_JSON).unwrap();
    assert!(jsonschema::is_valid(&schema, &ir));
    let model = invar::system::SystemIr::from_json(&original).unwrap();
    assert_eq!(format!("{}\n", model.canonical_json()).as_bytes(), original);
}

#[test]
fn issue_s1_01_system_dialect_optional_input_and_output_warning() {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/s1_01/optional-port.inv"
    );
    let ir = Command::new(env!("CARGO_BIN_EXE_invar"))
        .args(["system", "ir", path])
        .output()
        .unwrap();
    assert!(ir.status.success());
    assert_eq!(
        ir.stdout,
        include_bytes!("fixtures/s1_01/optional-port.ir.json")
    );
    let value: serde_json::Value = serde_json::from_slice(&ir.stdout).unwrap();
    let schema: serde_json::Value = serde_json::from_str(invar::system::SCHEMA_JSON).unwrap();
    assert!(jsonschema::is_valid(&schema, &value));
    let model = invar::system::SystemIr::from_json(&ir.stdout).unwrap();
    assert_eq!(
        format!("{}\n", model.canonical_json()).as_bytes(),
        ir.stdout
    );
    let check = Command::new(env!("CARGO_BIN_EXE_invar"))
        .args(["system", "check", path, "--format", "json"])
        .output()
        .unwrap();
    assert!(check.status.success());
    let document: serde_json::Value = serde_json::from_slice(&check.stdout).unwrap();
    assert_eq!(document["diagnostics"][0]["code"], "S1011");
    assert_eq!(document["diagnostics"][0]["severity"], "warning");
}
