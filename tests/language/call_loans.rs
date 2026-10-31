//! Issue label M0-06.
use invar::{
    check::check,
    eval::{Value, run},
    front::parse,
};

fn checked(source: &str) -> Result<invar::check::CheckedProgram, invar::front::FrontError> {
    check(parse(source.as_bytes())?, true)
}

fn result(source: &str) -> i32 {
    let program = checked(source).unwrap();
    match run(&program, 10_000).unwrap().value {
        Value::I32(value) => value,
        other => panic!("unexpected result: {other:?}"),
    }
}

#[test]
fn call_loans_read_read_and_take() {
    assert_eq!(
        result(
            "fn pair(a: read Array<i32>, b: read Array<i32>) -> i32 { return a[0u32] + b[0u32]; } fn consume(a: take Array<i32>) -> i32 { return a[0u32]; } fn main() -> i32 { let xs: Array<i32> = [7]; let n: i32 = pair(xs, xs); return n + consume(move xs); }"
        ),
        21
    );
}

#[test]
fn call_loans_mut_reborrow_restores_caller_access() {
    assert_eq!(
        result(
            "fn read_one(a: read Array<i32>) -> i32 { return a[0u32]; } fn add(a: mut Array<i32>) -> Unit { a[0u32] = a[0u32] + read_one(a); return; } fn main() -> i32 { var xs: Array<i32> = [2]; add(xs); xs[0u32] = xs[0u32] + 1; return xs[0u32]; }"
        ),
        5
    );
    assert_eq!(
        result(
            "fn bump(a: mut Array<i32>) -> Unit { a[0u32] = a[0u32] + 1; return; } fn twice(a: mut Array<i32>) -> Unit { bump(a); bump(a); return; } fn main() -> i32 { var xs: Array<i32> = [1]; twice(xs); return xs[0u32]; }"
        ),
        3
    );
}

#[test]
fn call_loans_rejects_conflicts_at_call_boundary() {
    let cases = [
        (
            "fn f(a: mut Array<i32>, b: i32) -> Unit { return; } fn g(a: read Array<i32>) -> i32 { return 0; } fn main() -> i32 { var xs: Array<i32> = [1]; f(xs, g(xs)); return 0; }",
            "OWN_BORROW_CONFLICT",
        ),
        (
            "fn f(a: read Array<i32>, b: i32) -> Unit { return; } fn g(a: mut Array<i32>) -> i32 { return 0; } fn main() -> i32 { var xs: Array<i32> = [1]; f(xs, g(xs)); return 0; }",
            "OWN_BORROW_CONFLICT",
        ),
        (
            "fn f(a: mut Array<i32>, b: mut Array<i32>) -> Unit { return; } fn main() -> i32 { var xs: Array<i32> = [1]; f(xs, xs); return 0; }",
            "OWN_BORROW_CONFLICT",
        ),
        (
            "fn f(a: read Array<i32>, b: take Array<i32>) -> Unit { return; } fn main() -> i32 { let xs: Array<i32> = [1]; f(xs, move xs); return 0; }",
            "OWN_BORROW_CONFLICT",
        ),
        (
            "fn f(a: mut Array<i32>) -> Unit { return; } fn main() -> i32 { let xs: Array<i32> = [1]; f(xs); return 0; }",
            "OWN_MUT_REQUIRED",
        ),
        (
            "fn f(a: read Array<i32>) -> Unit { return; } fn main() -> i32 { f([1]); return 0; }",
            "OWN_BORROW_SOURCE",
        ),
        (
            "fn f(a: read Array<i32>) -> Unit { return; } fn main() -> i32 { let xs: Array<i32> = [1]; f(xs[0u32]); return 0; }",
            "TYPE_MISMATCH",
        ),
        (
            "fn f(a: read Array<i32>) -> Array<i32> { return a; } fn main() -> i32 { return 0; }",
            "OWN_RETURN_BORROW",
        ),
        (
            "fn f(a: mut Array<i32>) -> Array<i32> { return move a; } fn main() -> i32 { return 0; }",
            "OWN_RETURN_BORROW",
        ),
    ];
    for (source, code) in cases {
        assert_eq!(checked(source).unwrap_err().code, code, "{source}");
    }
}
