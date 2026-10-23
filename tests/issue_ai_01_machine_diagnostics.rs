use std::{collections::HashSet, path::PathBuf, process::Command};

use invar::{diagnostics, front::Span};
use serde_json::{Value, json};

fn schema(name: &str) -> Value {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("schema")
        .join(name);
    serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap()
}

fn code_registry() -> Value {
    let mut base = schema("diagnostic-codes-v1.json");
    for name in [
        "diagnostic-codes-external-v1.json",
        "diagnostic-codes-system-v1.json",
        "diagnostic-codes-system-v2.json",
        "diagnostic-codes-foreign-v1.json",
        "diagnostic-codes-ros2-v1.json",
        "diagnostic-codes-robot-v1.json",
        "diagnostic-codes-summary-v1.json",
        "diagnostic-codes-indexed-v1.json",
        "diagnostic-codes-tuning-v1.json",
        "diagnostic-codes-fsum-v1.json",
        "diagnostic-codes-qualification-v1.json",
    ] {
        let extension = schema(name);
        if name == "diagnostic-codes-robot-v1.json" {
            assert_eq!(extension["extends"], "invar-diagnostic-codes-ros2-v1");
        } else if name == "diagnostic-codes-ros2-v1.json" {
            assert_eq!(extension["extends"], "invar-diagnostic-codes-foreign-v1");
        } else if name == "diagnostic-codes-foreign-v1.json" {
            assert_eq!(extension["extends"], "invar-diagnostic-codes-system-v2");
        } else if name == "diagnostic-codes-system-v2.json" {
            assert_eq!(extension["extends"], "invar-diagnostic-codes-system-v1");
        } else {
            assert_eq!(extension["extends"], base["schema"]);
        }
        base["codes"]
            .as_array_mut()
            .unwrap()
            .extend(extension["codes"].as_array().unwrap().iter().cloned());
    }
    base
}

fn cli(args: &[&str]) -> (i32, Value) {
    let output = Command::new(env!("CARGO_BIN_EXE_invar"))
        .args(args)
        .output()
        .unwrap();
    assert!(
        output.stderr.is_empty(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    (
        output.status.code().unwrap(),
        serde_json::from_slice(&output.stdout).unwrap(),
    )
}

fn fixture(name: &str, source: &[u8]) -> PathBuf {
    let path = std::env::temp_dir().join(format!(
        "invar-ai-01-{name}-{}-{:?}.inv",
        std::process::id(),
        std::thread::current().id()
    ));
    std::fs::write(&path, source).unwrap();
    path
}

fn registry_codes() -> HashSet<String> {
    let registry = code_registry();
    registry["codes"]
        .as_array()
        .unwrap()
        .iter()
        .map(|entry| entry["code"].as_str().unwrap().to_owned())
        .collect()
}

fn invariant_codes() -> HashSet<String> {
    code_registry()["codes"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|entry| entry["reachability"] == "invariant")
        .map(|entry| entry["code"].as_str().unwrap().to_owned())
        .collect()
}

fn source_codes() -> HashSet<String> {
    code_registry()["codes"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|entry| entry["reachability"] == "source")
        .map(|entry| entry["code"].as_str().unwrap().to_owned())
        .collect()
}

// The tests live in the sibling files of this directory, spliced in with
// `include!` so every test keeps its flat name and one test binary.
const OWN_SOURCES: [&str; 5] = [
    include_str!("issue_ai_01_machine_diagnostics/registry.rs"),
    include_str!("issue_ai_01_machine_diagnostics/documents.rs"),
    include_str!("issue_ai_01_machine_diagnostics/fault_fixtures.rs"),
    include_str!("issue_ai_01_machine_diagnostics/contracts.rs"),
    include_str!("issue_ai_01_machine_diagnostics/indexed_and_versions.rs"),
];

include!("issue_ai_01_machine_diagnostics/registry.rs");
include!("issue_ai_01_machine_diagnostics/documents.rs");
include!("issue_ai_01_machine_diagnostics/fault_fixtures.rs");
include!("issue_ai_01_machine_diagnostics/contracts.rs");
include!("issue_ai_01_machine_diagnostics/indexed_and_versions.rs");
