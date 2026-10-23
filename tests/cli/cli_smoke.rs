use std::process::Command;

#[test]
fn help_from_real_binary() {
    let output = Command::new(env!("CARGO_BIN_EXE_invar"))
        .arg("--help")
        .output()
        .expect("run invar binary");
    assert!(output.status.success());
    assert!(output.stderr.is_empty());
    assert!(String::from_utf8_lossy(&output.stdout).contains("Usage: invar --help"));
}

#[test]
fn foreign_extension_is_refused_with_a_typed_code() {
    let output = Command::new(env!("CARGO_BIN_EXE_invar"))
        .args(["check", "examples/sum.txt", "--format", "json"])
        .output()
        .expect("reject a foreign extension");
    assert_eq!(output.status.code(), Some(2));
    let document: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(document["diagnostics"][0]["code"], "FILE_EXTENSION");
    assert!(
        document["diagnostics"][0]["message"]
            .as_str()
            .unwrap()
            .contains(".inv")
    );
}
