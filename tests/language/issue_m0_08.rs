use invar::{
    check::check,
    eval::{Value, observe},
    front::{ErrorKind, parse},
};
use std::process::Command;

fn checked(source: &str) -> invar::check::CheckedProgram {
    check(parse(source.as_bytes()).unwrap(), true).unwrap()
}

#[test]
fn issue_m0_08_bounded_deterministic_reference_corpus() {
    // Fixed corpus, no random seed: every case is below the source and runtime limits.
    let cases = [
        (
            "fn main() -> i32 { return -2147483648i32 % -1i32; }",
            Value::I32(0),
        ),
        (
            "fn main() -> i32 { if false { return 1 / 0; } else { return 7; } }",
            Value::I32(7),
        ),
        (
            "fn main() -> i32 { let xs: Array<i32> = []; if len(xs) == 0u32 { return 2; } else { return 0; } }",
            Value::I32(2),
        ),
        (
            "fn f(a: read Array<i32>, b: read Array<i32>) -> i32 { return a[0u32] + b[0u32]; } fn main() -> i32 { let xs: Array<i32> = [3]; return f(xs, xs); }",
            Value::I32(6),
        ),
        (
            "fn main() -> i32 { var xs: Array<i32> = [2]; xs[len(xs) - 1u32] = xs[0u32] + 1; return xs[0u32]; }",
            Value::I32(3),
        ),
    ];
    for (source, expected) in cases {
        let program = checked(source);
        let first = observe(&program, 10_000);
        let second = observe(&program, 10_000);
        assert_eq!(first.result.unwrap(), expected, "{source}");
        assert_eq!(second.result.unwrap(), expected, "{source}");
        assert_eq!(first.steps_used, second.steps_used, "{source}");
        assert_eq!(first.cleanup_trace, second.cleanup_trace, "{source}");
    }
}

#[test]
fn issue_m0_08_adversarial_source_corpus() {
    let cases = [
        ("fn main() -> i32 { return 1xyz; }", "LEX_INVALID_LITERAL"),
        ("fn main() -> i32 { /* unterminated", "LEX_INVALID_TOKEN"),
        ("fn main() -> i32 { return 1", "PARSE_EXPECTED_TOKEN"),
        (
            "fn main() -> i32 { let x: i32 = 1; let x: i32 = 2; return x; }",
            "NAME_DUPLICATE",
        ),
        ("fn main() -> i32 {}", "RETURN_MISSING"),
        ("fn main() -> Unit {}", "ENTRYPOINT_SIGNATURE"),
        ("fn helper() -> i32 { return 1; }", "ENTRYPOINT_MISSING"),
        ("fn main() -> i32 { return 1u32 + 2i32; }", "TYPE_MISMATCH"),
        ("fn main() -> i32 { return missing; }", "NAME_UNKNOWN"),
        (
            "fn main() -> i32 { let xs: Array<i32> = [1]; let ys: Array<i32> = move xs; return xs[0u32]; }",
            "OWN_USE_AFTER_MOVE",
        ),
        (
            "fn main() -> i32 { let xs: Array<i32> = [1]; let ys: Array<i32> = xs; return 0; }",
            "OWN_MOVE_REQUIRED",
        ),
        (
            "fn main() -> i32 { filled_i32(1u32, 0); return 0; }",
            "UNSUPPORTED_FEATURE",
        ),
        (
            "fn f(a: mut Array<i32>, b: read Array<i32>) -> Unit { return; } fn main() -> i32 { var xs: Array<i32> = [1]; f(xs, xs); return 0; }",
            "OWN_BORROW_CONFLICT",
        ),
    ];
    for (source, code) in cases {
        let error = parse(source.as_bytes())
            .and_then(|program| check(program, true))
            .unwrap_err();
        assert_eq!(error.kind, ErrorKind::Source, "{source}");
        assert_eq!(error.code, code, "{source}");
    }
    for (source, code) in [
        (vec![0xff], "INVALID_UTF8"),
        (vec![b' '; 1_048_577], "SOURCE_SIZE_LIMIT"),
        (
            format!(
                "fn main() -> i32 {{ return {}1{}; }}",
                "(".repeat(33),
                ")".repeat(33)
            )
            .into_bytes(),
            "SOURCE_NESTING_LIMIT",
        ),
    ] {
        let error = parse(&source).unwrap_err();
        assert_eq!(error.kind, ErrorKind::Tool);
        assert_eq!(error.code, code);
    }
}

#[test]
fn issue_m0_08_json_escaping_and_single_document() {
    let path = std::env::temp_dir().join(format!("invar m0-08 \"{}\".inv", std::process::id()));
    std::fs::write(&path, "fn main() -> i32 { return unknown; }").unwrap();
    let output = Command::new(env!("CARGO_BIN_EXE_invar"))
        .args(["check", path.to_str().unwrap(), "--format", "json"])
        .output()
        .unwrap();
    std::fs::remove_file(&path).unwrap();
    assert_eq!(output.status.code(), Some(1));
    assert!(output.stderr.is_empty());
    let json: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(json["status"], "source_error");
    assert_eq!(json["diagnostics"][0]["path"], path.to_str().unwrap());
    assert_eq!(json["diagnostics"][0]["code"], "NAME_UNKNOWN");
}
