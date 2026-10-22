use std::process::Command;

#[test]
fn issue_ren_01_rename_cli_and_extension() {
    let binary = env!("CARGO_BIN_EXE_invar");
    let output = Command::new(binary)
        .args([
            "run",
            "examples/sum.inv",
            "--engine",
            "reference",
            "--format",
            "json",
        ])
        .output()
        .expect("run renamed binary");
    assert!(output.status.success());
    let document: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(document["schema"], "invar-diagnostics-v1");
    assert_eq!(document["result"]["value"], 10);

    let rejected = Command::new(binary)
        .args(["check", "examples/sum.txt", "--format", "json"])
        .output()
        .expect("reject a foreign extension");
    assert_eq!(rejected.status.code(), Some(2));
    let document: serde_json::Value = serde_json::from_slice(&rejected.stdout).unwrap();
    assert_eq!(document["diagnostics"][0]["code"], "FILE_EXTENSION");
    assert!(
        document["diagnostics"][0]["message"]
            .as_str()
            .unwrap()
            .contains(".inv")
    );
}

#[test]
fn issue_ren_01_rename_packages() {
    let output = Command::new("cargo")
        .args(["metadata", "--locked", "--no-deps", "--format-version", "1"])
        .output()
        .expect("cargo metadata");
    assert!(output.status.success());
    let metadata: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    let mut packages: Vec<_> = metadata["packages"]
        .as_array()
        .unwrap()
        .iter()
        .map(|package| package["name"].as_str().unwrap())
        .collect();
    packages.sort_unstable();
    assert_eq!(
        packages,
        [
            "invar",
            "invar-cuda",
            "invar-device",
            "invar-foreign",
            "invar-math",
            "invar-metal"
        ]
    );
}
