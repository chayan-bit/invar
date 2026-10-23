use invar::{
    check::check,
    front::{Type, parse},
};
use std::process::Command;

fn checked(source: &str) -> Result<invar::check::CheckedProgram, invar::front::FrontError> {
    check(parse(source.as_bytes())?, false)
}

#[test]
fn issue_m0_03_forward_calls_recursion_and_typed_order() {
    let source = "fn main() -> i32 { return f(3); } fn f(n: i32) -> i32 { if n == 0 { return 0; } else { return f(n - 1); } } fn unused() -> Unit { let x: u32 = 1; }";
    let program = checked(source).unwrap();
    assert_eq!(program.program.functions.len(), 3);
    assert!(program.expression_types.len() >= 10);
    let simple = checked("fn f() -> i32 { return 1 + 2; }").unwrap();
    assert_eq!(simple.expression_types.len(), 3);
    assert_eq!(simple.expression_types[0].0.start, 23);
    assert_eq!(simple.expression_types[1].0.start, 27);
    assert_eq!(simple.expression_types[2].0.start, 23);
    assert_eq!(simple.expression_types[2].1, Type::I32);
    assert!(checked("fn f() -> i32 { return -2147483648; }").is_ok());
    assert!(checked("fn f() -> Unit { let x: i32 = 1; }").is_ok());
    assert!(checked("fn f() -> Unit { var x: i32 = 1; x = 2; }").is_ok());
    assert!(
        checked("fn f() -> Unit { if true { let x: i32 = 1; } else { let y: u32 = 2; } }").is_ok()
    );
    assert!(checked("fn f() -> u32 { return 1 + 2u32; }").is_ok());
    assert!(checked("fn f() -> Bool { return 1 == 2u32; }").is_ok());
}

#[test]
fn issue_m0_03_rejects_names_types_and_returns() {
    let cases = [
        ("fn f() -> i32 { return x; }", "NAME_UNKNOWN"),
        ("fn f() -> i32 { return missing(1); }", "NAME_UNKNOWN"),
        ("fn f() -> Unit { let x = 1; let x = 2; }", "NAME_DUPLICATE"),
        (
            "fn f() -> Unit { if true { let x = 1; } else { let x = 2; } }",
            "NAME_DUPLICATE",
        ),
        ("fn f() -> Unit { let len = 1; }", "NAME_DUPLICATE"),
        ("fn f() -> Unit {} fn f() -> Unit {}", "NAME_DUPLICATE"),
        ("fn len() -> Unit {}", "NAME_DUPLICATE"),
        ("fn f(x: read i32) -> Unit {}", "TYPE_MISMATCH"),
        ("fn f(x: Array<i32>) -> Unit {}", "TYPE_MISMATCH"),
        ("fn f(x: Unit) -> Unit {}", "TYPE_MISMATCH"),
        ("fn f() -> Unit { let x: Unit = f(); }", "TYPE_MISMATCH"),
        ("fn f() -> Unit { let x = 1u32 + 2i32; }", "TYPE_MISMATCH"),
        ("fn f() -> Unit { if 1 {} }", "TYPE_MISMATCH"),
        ("fn f() -> Unit { while 1 {} }", "TYPE_MISMATCH"),
        ("fn f() -> Unit { for i in 0 .. 1i32 {} }", "TYPE_MISMATCH"),
        ("fn f() -> Unit { let x = 1; x = 2; }", "TYPE_MISMATCH"),
        ("fn f() -> Unit { let x = [1, 2]; }", "TYPE_MISMATCH"),
        ("fn f() -> Unit { f(1); }", "TYPE_MISMATCH"),
        ("fn f() -> i32 { if true { return 1; } }", "RETURN_MISSING"),
        (
            "fn f() -> i32 { while true { return 1; } }",
            "RETURN_MISSING",
        ),
        ("fn f() -> i32 { return; }", "TYPE_MISMATCH"),
        ("fn f() -> Unit { return 1; }", "TYPE_MISMATCH"),
        ("fn f() -> i32 { return 1u32; }", "TYPE_MISMATCH"),
        ("fn f() -> i32 { return 2147483648i32; }", "TYPE_MISMATCH"),
        ("fn f() -> i32 { return -(2147483648); }", "TYPE_MISMATCH"),
        ("fn f() -> i32 { return -2147483649; }", "TYPE_MISMATCH"),
        ("fn f() -> u32 { return 4294967296u32; }", "TYPE_MISMATCH"),
        ("fn f() -> i32 { return -1u32; }", "TYPE_MISMATCH"),
        (
            "fn f() -> i32 { return 1; let x = 2147483648i32; }",
            "TYPE_MISMATCH",
        ),
        (
            "fn f() -> Unit {} fn unused() -> i32 { return ghost; }",
            "NAME_UNKNOWN",
        ),
        (
            "fn main(x: i32) -> i32 { return x; }",
            "ENTRYPOINT_SIGNATURE",
        ),
    ];
    for (source, code) in cases {
        assert_eq!(checked(source).unwrap_err().code, code, "{source}");
    }
    assert_eq!(
        check(parse(b"fn f() -> Unit {}").unwrap(), true)
            .unwrap_err()
            .code,
        "ENTRYPOINT_MISSING"
    );
}

#[test]
fn issue_m0_03_cli_check_json_and_status() {
    let path = std::env::temp_dir().join(format!(
        "invar-m0-03-{}-{}.inv",
        std::process::id(),
        std::thread::current().name().unwrap_or("test")
    ));
    let binary = env!("CARGO_BIN_EXE_invar");
    for (source, expected_status, expected_exit, code) in [
        (b"fn f() -> Unit {}".as_slice(), "ok", 0, None),
        (b"fn main() -> i32 { return f(1); } fn f(n: i32) -> i32 { if n == 0 { return 0; } else { return f(n - 1); } }".as_slice(), "ok", 0, None),
        (
            b"fn f() -> i32 { return missing; }".as_slice(),
            "source_error",
            1,
            Some("NAME_UNKNOWN"),
        ),
        (b"fn f() -> Unit { let x = 1; let x = 2; }".as_slice(), "source_error", 1, Some("NAME_DUPLICATE")),
        (b"fn f() -> i32 { return 1u32; }".as_slice(), "source_error", 1, Some("TYPE_MISMATCH")),
        (b"fn f() -> i32 {}".as_slice(), "source_error", 1, Some("RETURN_MISSING")),
        (b"fn f() -> i32 { return 1; let x = 2147483648i32; }".as_slice(), "source_error", 1, Some("TYPE_MISMATCH")),
        (b"\xff".as_slice(), "tool_error", 2, Some("INVALID_UTF8")),
    ] {
        std::fs::write(&path, source).unwrap();
        let output = Command::new(binary)
            .args(["check", path.to_str().unwrap(), "--format", "json"])
            .output()
            .unwrap();
        assert_eq!(output.status.code(), Some(expected_exit));
        assert!(output.stderr.is_empty());
        let value: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
        assert_eq!(value["status"], expected_status);
        assert_eq!(value["schema_version"], 3);
        assert!(value["diagnostics"].is_array());
        if let Some(code) = code {
            assert_eq!(value["diagnostics"][0]["code"], code);
        } else {
            assert!(value["diagnostics"].as_array().unwrap().is_empty());
            assert_eq!(value["checks_pending"], serde_json::json!([]));
        }
    }
    std::fs::remove_file(&path).unwrap();
}
