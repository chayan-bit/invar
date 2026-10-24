use invar::{
    check::check,
    eval::{EvalError, TrapCategory, Value, run},
    front::parse,
};

fn execute(source: &str, fuel: u64) -> Result<invar::eval::Outcome, EvalError> {
    let checked = check(parse(source.as_bytes()).unwrap(), true).unwrap();
    run(&checked, fuel)
}

#[test]
fn issue_m0_04_scalar_order_demand_and_calls() {
    let source = "fn main() -> i32 { return 1 + 2; }";
    assert_eq!(execute(source, 4).unwrap().value, Value::I32(3));
    assert_eq!(execute(source, 4).unwrap().steps_used, 4);
    assert_eq!(
        execute(source, 3).unwrap_err(),
        EvalError::EvaluationLimit {
            fuel_limit: 3,
            steps_used: 3
        }
    );
    assert_eq!(
        execute("fn main() -> i32 { return (1); }", 2)
            .unwrap()
            .value,
        Value::I32(1)
    );
    assert_eq!(
        execute(
            "fn main() -> i32 { if true { return 7; } else { return 1 / 0; } }",
            100
        )
        .unwrap()
        .value,
        Value::I32(7)
    );
    assert_eq!(
        execute(
            "fn main() -> i32 { if false and 1 / 0 == 0 { return 0; } return 8; }",
            100
        )
        .unwrap()
        .value,
        Value::I32(8)
    );
    assert_eq!(
        execute(
            "fn main() -> i32 { if true or 1 / 0 == 0 { return 9; } return 0; }",
            100
        )
        .unwrap()
        .value,
        Value::I32(9)
    );
    assert_eq!(
        execute(
            "fn noop() -> Unit {} fn main() -> i32 { noop(); return 2; }",
            100
        )
        .unwrap()
        .value,
        Value::I32(2)
    );
    assert_eq!(
        execute(
            "fn main() -> i32 { var x: i32 = 0; while x < 3 { x = x + 1; } return x; }",
            100
        )
        .unwrap()
        .value,
        Value::I32(3)
    );
    assert_eq!(
        execute(
            "fn main() -> i32 { var x: i32 = 0; for i in 0u32 .. 3u32 { x = x + 1; } return x; }",
            100
        )
        .unwrap()
        .value,
        Value::I32(3)
    );
    assert_eq!(
        execute(
            "fn main() -> i32 { var x: i32 = 0; for i in 3u32 .. 2u32 { x = 1; } return x; }",
            100
        )
        .unwrap()
        .value,
        Value::I32(0)
    );
    let source = "fn f(a: i32, b: i32) -> i32 { return a; } fn main() -> i32 { return f(1 / 0, 2147483647 + 1); }";
    let error = execute(source, 100).unwrap_err();
    assert_eq!(
        error,
        EvalError::Trap {
            category: TrapCategory::DivisionByZero,
            span: invar::front::Span {
                start: source.find("/").unwrap(),
                end: source.find("/").unwrap() + 1
            }
        }
    );
}

#[test]
fn issue_m0_04_arithmetic_boundaries() {
    let completed = [
        ("fn main() -> i32 { return -2147483648 % -1; }", 0),
        ("fn main() -> i32 { return -7 / 3; }", -2),
        ("fn main() -> i32 { return -7 % 3; }", -1),
        (
            "fn main() -> i32 { if wrapping_add(4294967295u32, 1u32) == 0u32 { return 1; } return 0; }",
            1,
        ),
        (
            "fn main() -> i32 { if wrapping_mul(4294967295u32, 4294967295u32) == 1u32 { return 1; } return 0; }",
            1,
        ),
    ];
    for (source, expected) in completed {
        assert_eq!(
            execute(source, 200).unwrap().value,
            Value::I32(expected),
            "{source}"
        );
    }
    let traps = [
        (
            "fn main() -> i32 { return -2147483648 / -1; }",
            TrapCategory::IntegerOverflow,
            "/",
        ),
        (
            "fn main() -> i32 { return 1 / 0; }",
            TrapCategory::DivisionByZero,
            "/",
        ),
        (
            "fn main() -> i32 { return 1 % 0; }",
            TrapCategory::DivisionByZero,
            "%",
        ),
        (
            "fn main() -> i32 { return 2147483647 + 1; }",
            TrapCategory::IntegerOverflow,
            "+",
        ),
        (
            "fn main() -> i32 { return -2147483648 - 1; }",
            TrapCategory::IntegerOverflow,
            " - ",
        ),
        (
            "fn main() -> i32 { return 2147483647 * 2; }",
            TrapCategory::IntegerOverflow,
            "*",
        ),
        (
            "fn main() -> i32 { let x: i32 = -2147483648; return -x; }",
            TrapCategory::IntegerOverflow,
            "-x",
        ),
        (
            "fn main() -> i32 { if 0u32 - 1u32 == 0u32 { return 1; } return 0; }",
            TrapCategory::IntegerOverflow,
            " - ",
        ),
    ];
    for (source, category, marker) in traps {
        let at = source.find(marker).unwrap() + if marker == " - " { 1 } else { 0 };
        let span = match execute(source, 200).unwrap_err() {
            EvalError::Trap {
                category: found,
                span,
            } => {
                assert_eq!(found, category);
                span
            }
            other => panic!("{source}: {other:?}"),
        };
        assert_eq!(span.start, at, "{source}");
    }
}

#[test]
fn issue_m0_04_fuel_and_call_depth() {
    let source = "fn main() -> i32 { while false {} return 1; }";
    assert_eq!(execute(source, 4).unwrap().steps_used, 4);
    assert_eq!(
        execute(source, 3).unwrap_err(),
        EvalError::EvaluationLimit {
            fuel_limit: 3,
            steps_used: 3
        }
    );
    let source = "fn main() -> i32 { while true {} return 1; }";
    assert_eq!(
        execute(source, 5).unwrap_err(),
        EvalError::EvaluationLimit {
            fuel_limit: 5,
            steps_used: 5
        }
    );
    let source = "fn main() -> i32 { return f(); } fn f() -> i32 { return f(); }";
    assert!(matches!(
        execute(source, 1000).unwrap_err(),
        EvalError::Tool {
            code: "CALL_DEPTH_LIMIT",
            ..
        }
    ));
}
