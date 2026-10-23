use std::{path::Path, process::Command};

fn cli(args: &[&str], exit: i32, status: &str) -> serde_json::Value {
    let output = Command::new(env!("CARGO_BIN_EXE_invar"))
        .args(args)
        .output()
        .unwrap();
    assert_eq!(output.status.code(), Some(exit));
    assert!(
        output.stderr.is_empty(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let value: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(value["schema_version"], 3);
    assert_eq!(value["status"], status);
    assert!(value["diagnostics"].is_array());
    value
}

fn run(path: &Path, extra: &[&str], exit: i32, status: &str) -> serde_json::Value {
    let mut args = vec![
        "run",
        path.to_str().unwrap(),
        "--engine",
        "reference",
        "--format",
        "json",
    ];
    args.extend_from_slice(extra);
    cli(&args, exit, status)
}

#[test]
fn issue_m0_07_real_cli_examples_and_statuses() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    for (name, expected) in [("sum.inv", 10), ("ownership.inv", 41)] {
        let path = root.join("examples").join(name);
        let value = run(&path, &[], 0, "ok");
        assert_eq!(value["result"]["type"], "i32");
        assert_eq!(value["result"]["value"], expected);
    }
    let path = std::env::temp_dir().join(format!("invar m0-07 {}.inv", std::process::id()));
    std::fs::write(&path, "fn library() -> i32 { return 1; }").unwrap();
    let check = cli(
        &["check", path.to_str().unwrap(), "--format", "json"],
        0,
        "ok",
    );
    assert_eq!(check["checks_pending"], serde_json::json!([]));
    let missing = run(&path, &[], 1, "source_error");
    assert_eq!(missing["diagnostics"][0]["code"], "ENTRYPOINT_MISSING");
    assert_eq!(missing["diagnostics"][0]["path"], path.to_str().unwrap());

    std::fs::write(&path, "fn main() -> i32 { return 42; }").unwrap();
    assert_eq!(run(&path, &[], 0, "ok")["result"]["value"], 42);
    let limit = run(&path, &["--fuel", "1"], 4, "evaluation_limit");
    assert_eq!(limit["fuel_limit"], 1);
    assert_eq!(limit["steps_used"], 1);

    std::fs::write(&path, "fn main() -> i32 { return 1 / 0; }").unwrap();
    let trap = run(&path, &[], 3, "trap");
    assert_eq!(trap["trap"]["category"], "DIVISION_BY_ZERO");
    assert!(trap["trap"]["span"]["start"].is_number());

    std::fs::write(
        &path,
        "fn main() -> i32 { let xs: Array<i32> = filled_i32(16777217u32, 0); return 0; }",
    )
    .unwrap();
    let tool = run(&path, &[], 2, "tool_error");
    assert_eq!(tool["diagnostics"][0]["code"], "ARRAY_SIZE_LIMIT");

    std::fs::write(&path, b"\xff").unwrap();
    let encoding = run(&path, &[], 2, "tool_error");
    assert_eq!(encoding["diagnostics"][0]["code"], "INVALID_UTF8");
    std::fs::remove_file(path).unwrap();
}

#[test]
fn issue_m0_07_invalid_arguments_are_usage_errors() {
    for args in [
        vec!["run", "missing.inv", "--engine", "native"],
        vec!["run", "missing.inv", "--engine", "reference", "--fuel", "0"],
        vec![
            "run",
            "missing.inv",
            "--engine",
            "reference",
            "--fuel",
            "-1",
        ],
    ] {
        let output = Command::new(env!("CARGO_BIN_EXE_invar"))
            .args(args)
            .output()
            .unwrap();
        assert_eq!(output.status.code(), Some(2));
        assert!(output.stdout.is_empty());
        assert!(
            String::from_utf8(output.stderr)
                .unwrap()
                .contains("Usage: invar")
        );
    }
}
