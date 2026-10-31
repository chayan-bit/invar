//! `invar-qualify` refuses a bad invocation with a registered, machine-readable diagnostic.
//! The registry coverage test (`tests/diagnostics/machine_diagnostics.rs`) leaves these two codes
//! to this file because only this package can launch the binary.
use serde_json::Value;
use std::{path::Path, process::Command};

fn registered_source_code(code: &str) -> bool {
    let registry: Value = serde_json::from_slice(
        &std::fs::read(
            Path::new(env!("CARGO_MANIFEST_DIR")).join("../schema/diagnostic-codes-v1.json"),
        )
        .unwrap(),
    )
    .unwrap();
    registry["codes"]
        .as_array()
        .unwrap()
        .iter()
        .any(|entry| entry["code"] == code && entry["reachability"] == "source")
}

fn refusal(args: &[&str]) -> Value {
    let output = Command::new(env!("CARGO_BIN_EXE_invar-qualify"))
        .args(args)
        .output()
        .unwrap();
    assert_eq!(output.status.code(), Some(2));
    assert!(output.stdout.is_empty());
    serde_json::from_slice(&output.stderr).unwrap()
}

#[test]
fn qualify_cli_refuses_an_unknown_option_with_a_registered_code() {
    let diagnostic = refusal(&["--unknown"]);
    assert_eq!(diagnostic["diagnostic"], "M5_USAGE");
    assert!(registered_source_code("M5_USAGE"));
}

#[test]
fn qualify_cli_refuses_a_relative_library_directory_before_any_run() {
    let diagnostic = refusal(&["--device", "cuda", "--libraries", "not-absolute"]);
    assert_eq!(diagnostic["diagnostic"], "M5_LIBRARY_PATH");
    assert!(registered_source_code("M5_LIBRARY_PATH"));
}
