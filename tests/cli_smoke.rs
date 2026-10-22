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
