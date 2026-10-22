use invar::{
    check::check,
    eval::{EvalError, TrapCategory, Value, observe, run},
    front::parse,
};
use std::process::Command;

fn checked(source: &str) -> Result<invar::check::CheckedProgram, invar::front::FrontError> {
    check(parse(source.as_bytes())?, true)
}

fn value(source: &str) -> i32 {
    let program = checked(source).unwrap();
    match run(&program, 10_000).unwrap().value {
        Value::I32(value) => value,
        other => panic!("unexpected {other:?}"),
    }
}

#[test]
fn issue_m0_05_array_values_indexing_and_filling() {
    assert_eq!(
        value(
            "fn main() -> i32 { let xs: Array<i32> = []; if len(xs) == 0u32 { return 1; } return 0; }"
        ),
        1
    );
    assert_eq!(
        value(
            "fn main() -> i32 { var xs: Array<i32> = [40, 1]; xs[1u32] = 2; return xs[0u32] + xs[1u32]; }"
        ),
        42
    );
    assert_eq!(
        value("fn main() -> i32 { let xs: Array<i32> = filled_i32(3u32, 7); return xs[2u32]; }"),
        7
    );
    assert_eq!(
        value(
            "fn main() -> i32 { var xs: Array<u32> = [0u32, 1u32]; xs[xs[0u32]] = 4u32; if xs[0u32] == 4u32 { return 1; } return 0; }"
        ),
        1
    );
    assert_eq!(
        value(
            "fn make() -> Array<i32> { let xs: Array<i32> = [4]; return xs; } fn main() -> i32 { let ys: Array<i32> = make(); return ys[0u32]; }"
        ),
        4
    );
    assert_eq!(
        value(
            "fn main() -> i32 { let a: Array<i32> = [4]; let b: Array<i32> = move a; return b[0u32]; }"
        ),
        4
    );
    assert_eq!(
        value(
            "fn main() -> i32 { for i in 0u32 .. 2u32 { let a: Array<i32> = [1]; let b: Array<i32> = move a; } return 0; }"
        ),
        0
    );
    assert_eq!(
        value(
            "fn main() -> i32 { let a: Array<i32> = [5]; if true { let b: Array<i32> = move a; return b[0u32]; } else {} return a[0u32]; }"
        ),
        5
    );
}

#[test]
fn issue_m0_05_rejects_invalid_ownership() {
    let cases = [
        (
            "fn main() -> i32 { let a: Array<i32> = [1]; let b: Array<i32> = a; return 0; }",
            "OWN_MOVE_REQUIRED",
        ),
        (
            "fn main() -> i32 { let a: Array<i32> = [1]; let b: Array<i32> = move a; return a[0u32]; }",
            "OWN_USE_AFTER_MOVE",
        ),
        (
            "fn main() -> i32 { var a: Array<i32> = [1]; a = move a; return 0; }",
            "OWN_USE_AFTER_MOVE",
        ),
        (
            "fn main() -> i32 { var a: Array<i32> = [1]; let b: Array<i32> = move a; a = [2]; return 0; }",
            "OWN_USE_AFTER_MOVE",
        ),
        (
            "fn main() -> i32 { let a: Array<i32> = [1]; a[0u32] = 2; return 0; }",
            "TYPE_MISMATCH",
        ),
        (
            "fn main() -> i32 { let a: Array<i32> = [1]; return a[0i32]; }",
            "TYPE_MISMATCH",
        ),
        (
            "fn main() -> i32 { let a: Array<i32> = [1]; if true { let b: Array<i32> = move a; } return a[0u32]; }",
            "OWN_USE_AFTER_MOVE",
        ),
        (
            "fn main() -> i32 { let a: Array<i32> = [1]; for i in 0u32 .. 1u32 { let b: Array<i32> = move a; } return 0; }",
            "OWN_MOVE_REQUIRED",
        ),
        (
            "fn main() -> i32 { let a: Array<i32> = [1]; while false { let b: Array<i32> = move a; } return 0; }",
            "OWN_MOVE_REQUIRED",
        ),
        (
            "fn f(xs: read Array<i32>) -> Array<i32> { return xs; } fn main() -> i32 { return 0; }",
            "OWN_RETURN_BORROW",
        ),
        (
            "fn f(xs: read Array<i32>) -> i32 { return xs[0u32]; } fn main() -> i32 { return f([1]); }",
            "OWN_BORROW_SOURCE",
        ),
        (
            "fn main() -> i32 { var xs: Array<u32> = [0u32]; xs[len(move xs)] = 1u32; return 0; }",
            "UNSUPPORTED_FEATURE",
        ),
    ];
    for (source, code) in cases {
        assert_eq!(checked(source).unwrap_err().code, code, "{source}");
    }
}

#[test]
fn issue_m0_05_cleanup_and_failure_order() {
    let program = checked(
        "fn main() -> i32 { let a: Array<i32> = [1]; let b: Array<u32> = [2u32, 3u32]; return 0; }",
    )
    .unwrap();
    let outcome = run(&program, 100).unwrap();
    let trace: Vec<_> = outcome
        .cleanup_trace
        .iter()
        .map(|event| (event.name.as_str(), event.length))
        .collect();
    assert_eq!(trace, [("b", 2), ("a", 1)]);

    let program = checked(
        "fn main() -> i32 { let a: Array<i32> = [1]; let b: Array<i32> = move a; return b[0u32]; }",
    )
    .unwrap();
    let outcome = run(&program, 100).unwrap();
    assert_eq!(
        outcome
            .cleanup_trace
            .iter()
            .map(|event| event.name.as_str())
            .collect::<Vec<_>>(),
        ["b"]
    );

    let program =
        checked("fn main() -> i32 { var a: Array<i32> = [1]; a = [2, 3]; return 0; }").unwrap();
    let outcome = run(&program, 100).unwrap();
    assert_eq!(
        outcome
            .cleanup_trace
            .iter()
            .map(|event| event.length)
            .collect::<Vec<_>>(),
        [1, 2]
    );

    let source = "fn main() -> i32 { let a: Array<i32> = [1]; return a[2u32]; }";
    let observation = observe(&checked(source).unwrap(), 100);
    assert!(
        observation.cleanup_trace.is_empty(),
        "trap must not perform language cleanup"
    );
    match observation.result.unwrap_err() {
        EvalError::Trap {
            category: TrapCategory::IndexOutOfBounds,
            span,
        } => assert_eq!(span.start, source.find("a[2u32]").unwrap()),
        other => panic!("unexpected {other:?}"),
    }
    let source = "fn main() -> i32 { var a: Array<i32> = [1]; a[2u32] = 1 / 0; return 0; }";
    assert!(matches!(
        observe(&checked(source).unwrap(), 100).result,
        Err(EvalError::Trap {
            category: TrapCategory::DivisionByZero,
            ..
        })
    ));
}

#[test]
fn issue_m0_05_bounded_array_allocation() {
    let program =
        checked("fn main() -> i32 { let a: Array<i32> = filled_i32(16777217u32, 0); return 0; }")
            .unwrap();
    assert!(matches!(
        run(&program, 100),
        Err(EvalError::Tool {
            code: "ARRAY_SIZE_LIMIT",
            ..
        })
    ));
}

#[test]
fn issue_m0_05_cli_local_array_and_invalid_borrow() {
    let path = std::env::temp_dir().join(format!("invar-m0-05-{}.inv", std::process::id()));
    let binary = env!("CARGO_BIN_EXE_invar");
    for (source, exit, status) in [
        (
            "fn main() -> i32 { var xs: Array<i32> = [40, 1]; xs[1u32] = 2; return xs[0u32] + xs[1u32]; }",
            0,
            "ok",
        ),
        (
            "fn f(xs: read Array<i32>) -> i32 { return xs[0u32]; } fn main() -> i32 { return f([1]); }",
            1,
            "source_error",
        ),
    ] {
        std::fs::write(&path, source).unwrap();
        let output = Command::new(binary)
            .args([
                "run",
                path.to_str().unwrap(),
                "--engine",
                "reference",
                "--format",
                "json",
            ])
            .output()
            .unwrap();
        assert_eq!(output.status.code(), Some(exit));
        assert!(output.stderr.is_empty());
        let json: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
        assert_eq!(json["status"], status);
        if exit == 0 {
            assert_eq!(json["result"]["value"], 42);
        } else {
            assert_eq!(json["diagnostics"][0]["code"], "OWN_BORROW_SOURCE");
        }
    }
    std::fs::remove_file(path).unwrap();
}
