//! Issue label M1-01.
#![forbid(unsafe_code)]

use invar::{check, eval, front, general_net, mir, mir_eval};

fn qualify(source: &str, expected: i32, pinned_steps: Option<u64>) {
    let checked = check::check(front::parse(source.as_bytes()).unwrap(), true).unwrap();
    let program = mir::lower(&checked).unwrap();
    let net = general_net::Net::lower(&checked, general_net::Limits::default()).unwrap();
    let proof = net.verify().unwrap();
    let complete = eval::observe(&checked, 1000);
    assert_eq!(complete.result, Ok(eval::Value::I32(expected)), "{source}");
    if let Some(steps) = pinned_steps {
        assert_eq!(complete.steps_used, steps, "{source}");
    }
    for fuel in 1..=complete.steps_used + 1 {
        let reference = eval::observe(&checked, fuel);
        let actual = mir_eval::observe(&program, fuel);
        assert_eq!(actual.result, reference.result, "fuel={fuel}: {source}");
        assert_eq!(actual.steps_used, reference.steps_used, "{source}");
        assert_eq!(actual.cleanup_trace, reference.cleanup_trace, "{source}");
        let lowered_again = net.compiled_mir(&proof).unwrap();
        let actual = mir_eval::observe(&lowered_again, fuel);
        assert_eq!(actual.result, reference.result, "fuel={fuel}: {source}");
        assert_eq!(actual.steps_used, reference.steps_used, "{source}");
        assert_eq!(actual.cleanup_trace, reference.cleanup_trace, "{source}");
    }
    #[cfg(feature = "native")]
    for fuel in [
        complete.steps_used - 1,
        complete.steps_used,
        complete.steps_used + 1,
    ] {
        let reference = eval::observe(&checked, fuel);
        let (status, document) = invar::native::run_program(&program, fuel).unwrap();
        assert_eq!(document["steps_used"], reference.steps_used, "{source}");
        match reference.result {
            Ok(eval::Value::I32(value)) => {
                assert_eq!(status, 0, "{document}");
                assert_eq!(document["result"]["value"], value, "{source}");
            }
            Err(eval::EvalError::EvaluationLimit { .. }) => {
                assert_eq!(status, 4, "{document}");
                assert_eq!(document["status"], "evaluation_limit");
            }
            other => panic!("unexpected fixture outcome: {other:?}"),
        }
    }
}

#[test]
fn mir_verifier_and_evaluator_explicit_move_and_borrowed_return_fuel_unchanged() {
    qualify(
        "fn make() -> Array<i32> { let xs: Array<i32> = [3,5]; return move xs; } fn main() -> i32 { let a = make(); return a[1u32]; }",
        5,
        Some(12),
    );
    qualify(
        "fn relay(xs: read Array<i32>) -> View<i32> from xs { let part = view(xs,0u32,1u32); return part; } fn main() -> i32 { let xs: Array<i32> = [7i32]; let part = relay(xs); return part[0u32]; }",
        7,
        Some(17),
    );
    qualify(
        "fn scalar() -> i32 { let x = 7; return x; } fn main() -> i32 { return scalar(); }",
        7,
        None,
    );
}
