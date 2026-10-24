#![forbid(unsafe_code)]

use general_net::{Limits, Net, oracle};
use invar::{check, eval, front, general_net};

fn lowered(source: &str) -> (check::CheckedProgram, Net) {
    let checked = check::check(front::parse(source.as_bytes()).unwrap(), true).unwrap();
    let net = Net::lower(&checked, Limits::default()).unwrap();
    (checked, net)
}

fn differential(source: &str, fuel: u64) -> oracle::Observation {
    let (checked, net) = lowered(source);
    let reference = eval::observe(&checked, fuel);
    let actual = oracle::observe(
        &net,
        &net.verify().unwrap(),
        oracle::Limits {
            fuel,
            array_elements: invar::array_limits::MAX_LIVE_ELEMENTS,
            ..oracle::Limits::default()
        },
    )
    .unwrap();
    assert_eq!(actual.semantic.result, reference.result, "{source}");
    assert_eq!(actual.semantic.steps_used, reference.steps_used, "{source}");
    assert_eq!(
        actual.semantic.cleanup_trace, reference.cleanup_trace,
        "{source}"
    );
    actual
}

#[test]
fn issue_gnet_02_owned_array_mutation_moves_returns_and_cleanup() {
    for source in [
        "fn main() -> i32 { var xs: Array<i32> = [1,2]; xs[0u32] = xs[1u32] + 3; return xs[0u32]; }",
        "fn mutate(xs: mut Array<i32>) -> Unit { xs[0u32] = 9; } fn peek(xs: read Array<i32>) -> i32 { return xs[0u32]; } fn main() -> i32 { var xs: Array<i32> = [1]; mutate(xs); return peek(xs); }",
        "fn consume(xs: take Array<i32>) -> i32 { return xs[0u32]; } fn main() -> i32 { let a: Array<i32> = [4]; let b = move a; return consume(move b); }",
        "fn make() -> Array<i32> { let xs: Array<i32> = [3,5]; return xs; } fn main() -> i32 { let a = make(); return a[1u32]; }",
        "fn main() -> i32 { var xs = filled_i32(2u32, 3); xs = filled_i32(1u32, 9); return xs[0u32]; }",
        "fn main() -> i32 { let xs: Array<u32> = []; if len(xs) == 0u32 { return 7; } return 1 / 0; }",
        "fn main() -> i32 { var acc = 0; for i in 0u32..4u32 { let xs = filled_i32(2u32, 3); acc = acc + xs[0u32]; } return acc; }",
        include_str!("../fixtures/l1_03_generic_array.inv"),
        include_str!("../fixtures/l1_03_closures.inv"),
    ] {
        assert!(differential(source, 10_000).semantic.result.is_ok());
    }
}

#[test]
fn issue_gnet_02_array_traps_preserve_span_and_completed_cleanup_prefix() {
    for source in [
        "fn main() -> i32 { let xs: Array<i32> = [1]; return xs[1u32]; }",
        "fn main() -> i32 { var xs: Array<i32> = [1]; xs[1u32] = 9; return 0; }",
        "fn main() -> i32 { if true { let done: Array<i32> = [1,2]; } let xs: Array<i32> = [1]; return xs[1u32]; }",
        "fn bad(xs: mut Array<u32>) -> Unit { xs[0u32] = 4294967295u32 + 1u32; } fn main() -> i32 { var xs: Array<u32> = [0u32]; bad(xs); return 0; }",
    ] {
        assert!(matches!(
            differential(source, 1000).semantic.result,
            Err(eval::EvalError::Trap { .. })
        ));
    }
}

#[test]
fn issue_gnet_02_array_fuel_prefixes_have_no_cleanup_on_limit() {
    let source = "fn main() -> i32 { var xs: Array<i32> = [1,2]; for i in 0u32..2u32 { xs[i] = xs[i] + 3; } return xs[0u32]; }";
    let complete = differential(source, 1000);
    for fuel in 1..=complete.semantic.steps_used {
        differential(source, fuel);
    }
}

#[test]
fn issue_gnet_02_live_array_budget_releases_normal_owners_and_bounds_allocation() {
    let source =
        "fn main() -> i32 { for i in 0u32..30u32 { let xs = filled_i32(4u32, 3); } return 7; }";
    let (_, net) = lowered(source);
    let actual = oracle::observe(
        &net,
        &net.verify().unwrap(),
        oracle::Limits {
            array_elements: 4,
            ..oracle::Limits::default()
        },
    )
    .unwrap();
    assert_eq!(actual.semantic.result, Ok(eval::Value::I32(7)));
    assert_eq!(actual.semantic.cleanup_trace.len(), 30);
    let (_, net) = lowered("fn main() -> i32 { let xs = filled_i32(5u32, 3); return 0; }");
    let actual = oracle::observe(
        &net,
        &net.verify().unwrap(),
        oracle::Limits {
            array_elements: 4,
            ..oracle::Limits::default()
        },
    )
    .unwrap();
    assert_eq!(actual.failure, Some(oracle::ToolFailure::Allocation));
    assert!(actual.semantic.cleanup_trace.is_empty());
}

#[test]
fn issue_gnet_02_source_array_caps_and_implicit_return_fuel_prefixes() {
    assert_eq!(differential("fn main() -> i32 { let a = filled_i32(1000000u32, 0); let b = filled_u32(1000000u32, 0u32); let c = filled_i32(1u32, 0); return 1; }",1000).semantic.result,Ok(eval::Value::I32(1)));
    {
        let source = "fn main() -> i32 { let a = filled_i32(16777217u32, 0); return 1; }";
        assert!(matches!(
            differential(source, 1000).semantic.result,
            Err(eval::EvalError::Tool {
                code: "ARRAY_SIZE_LIMIT",
                ..
            })
        ));
    }
    for source in [
        "fn make() -> Array<i32> { let xs: Array<i32> = [3,5]; return xs; } fn main() -> i32 { let a = make(); return a[1u32]; }",
        "fn make() -> Array<i32> { let xs: Array<i32> = [3,5]; return move xs; } fn main() -> i32 { let a = make(); return a[1u32]; }",
    ] {
        let complete = differential(source, 1000);
        for fuel in 1..=complete.semantic.steps_used {
            differential(source, fuel);
        }
    }
}

#[test]
fn issue_gnet_02_named_scalar_entry_has_runtime_inputs_and_no_synthetic_fuel() {
    let (_, net) = lowered(
        "fn fib(n: u32) -> i32 { if n < 2u32 { return 1; } return fib(n - 1u32) + fib(n - 2u32); } fn main() -> i32 { return fib(8u32); }",
    );
    let proof = net.verify().unwrap();
    for (n, value) in [1, 1, 2, 3, 5, 8, 13, 21, 34].into_iter().enumerate() {
        let result = oracle::observe_call(
            &net,
            &proof,
            "fib",
            &[eval::Value::U32(n as u32)],
            oracle::Limits::default(),
        )
        .unwrap();
        assert_eq!(result.semantic.result, Ok(eval::Value::I32(value)));
    }
    let invalid = oracle::observe_call(
        &net,
        &proof,
        "fib",
        &[eval::Value::I32(8)],
        oracle::Limits::default(),
    )
    .unwrap();
    assert!(matches!(
        invalid.semantic.result,
        Err(eval::EvalError::Tool {
            code: "ENTRYPOINT_SIGNATURE",
            ..
        })
    ));
}

#[test]
fn issue_gnet_02_frame_helper_trap_reports_the_source_call_site() {
    // `a + b` on frame vectors calls a synthetic `$vec_add_i32` helper whose own
    // spans are {0,1}; every engine reports the overflow at the `a + b` site.
    let source = include_str!("../fixtures/r1_01/frame-overflow.inv");
    let observed = differential(source, 10_000);
    let Err(eval::EvalError::Trap { category, span }) = observed.semantic.result else {
        panic!("{:?}", observed.semantic.result);
    };
    assert_eq!(category, eval::TrapCategory::IntegerOverflow);
    assert_eq!(&source[span.start..span.end], "a + b");
}

#[test]
fn issue_gnet_02_views_text_and_units_match_the_reference_with_traps() {
    for source in [
        include_str!("../fixtures/l1_02_views.inv"),
        include_str!("../fixtures/l1_04/decode-valid.inv"),
        include_str!("../fixtures/l1_04/decode-invalid.inv"),
        include_str!("../fixtures/l1_04/slice-oob.inv"),
        include_str!("../fixtures/r1_01_conversion.inv"),
        include_str!("../fixtures/r1_01/inexact-conversion.inv"),
        include_str!("../fixtures/r1_01/overflow-conversion.inv"),
        include_str!("../fixtures/r1_01_angles.inv"),
        include_str!("../fixtures/r1_01/angle-overflow.inv"),
        "fn main() -> i32 { var xs: Array<u32> = [1u32, 2u32, 3u32, 4u32, 5u32]; let all = view_mut(xs, 0u32, 5u32, 2u32); let tail = view(all, 1u32, 3u32); if len(tail) == 2u32 and tail[1u32] == 5u32 { return 7; } return 0; }",
        "fn main() -> i32 { var xs: Array<i32> = [1, 2, 3]; let v = view(xs, 0u32, 3u32, 0u32); return 0; }",
        "fn main() -> i32 { var xs: Array<i32> = [1, 2, 3]; let v = view(xs, 2u32, 1u32); return 0; }",
        "fn main() -> i32 { var xs: Array<i32> = [1, 2, 3]; let v = view(xs, 0u32, 3u32, 2u32); return v[2u32]; }",
        "fn main() -> i32 { let s: String = \"h\u{e9}\"; let r = utf8_slice(s, 0u32, 2u32); return match move r { Result::Ok(t) => 1, Result::Err(e) => 2 }; }",
        "fn main() -> i32 { let s: String = \"abc\"; let b: Bytes = utf8_encode(s); if len(b) == 3u32 and b[0u32] == 97u32 { return 7; } return 0; }",
    ] {
        let observed = differential(source, 10_000);
        assert_eq!(observed.failure, None, "{source}");
        // One step short is the exact source-fuel limit in both engines.
        if observed.semantic.steps_used > 1 {
            differential(source, observed.semantic.steps_used - 1);
        }
    }
}
