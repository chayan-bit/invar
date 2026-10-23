#![forbid(unsafe_code)]

use crate::language_limits;

use general_net::{Limits, Net, oracle};
use invar::{check, eval, front, general_net, mir};

const CASES: &[&str] = &[
    include_str!("../fixtures/l1_01/record-moved-array.inv"),
    include_str!("../fixtures/l1_01/variant-owned-pattern.inv"),
    include_str!("../fixtures/l1_01/result-propagation.inv"),
    include_str!("../fixtures/l1_04/basic.inv"),
    "struct Packet { flag: Bool, count: u32, items: Array<u32>, code: i32 } fn inspect(p: read Packet) -> i32 { if p.flag and p.count == 2u32 { return p.code; } return 0; } fn main() -> i32 { let packet = Packet { code: 9, items: [3u32,4u32], count: 2u32, flag: true }; return inspect(packet); }",
    "struct Packet { items: Array<i32>, code: i32 } fn make() -> Packet { let p = Packet { items: [1,2,3], code: 7 }; return p; } fn consume(p: take Packet) -> i32 { return p.code; } fn main() -> i32 { let p = make(); return consume(move p); }",
    "struct Packet { items: Array<i32>, code: i32 } fn main() -> i32 { var p = Packet { items: [1,2], code: 7 }; p = Packet { items: [3], code: 9 }; return p.code; }",
    "enum Choice { Empty, Number(i32) } fn inspect(c: read Choice) -> i32 { return match c { Choice::Empty => 1, Choice::Number(x) => x }; } fn main() -> i32 { let a = Choice::Empty(); let b = Choice::Number(8); return inspect(a) + inspect(b); }",
    "fn make() -> Bytes { let b: Bytes = b\"a\\xFF\"; return b; } fn inspect(b: read Bytes) -> i32 { if len(b) == 2u32 and b[1u32] == 255u32 { return 7; } return 0; } fn main() -> i32 { let b = make(); return inspect(b); }",
    "fn inspect(b: read Bytes) -> i32 { if len(b) == 2u32 { return 7; } return 0; } fn make() -> Result<Bytes,u32> { return Result::Ok(b\"ab\"); } fn wrap() -> Result<Bytes,u32> { let b: Bytes = make()?; return Result::Ok(move b); } fn main() -> i32 { let result = wrap(); return match move result { Result::Ok(b) => inspect(b), Result::Err(e) => 0 }; }",
    "fn make() -> Result<String,u32> { return Result::Err(9u32); } fn wrap() -> Result<String,u32> { let scratch: Array<i32> = [1,2]; let s: String = make()?; return Result::Ok(move s); } fn main() -> i32 { let result = wrap(); return match move result { Result::Ok(s) => 0, Result::Err(e) => 9 }; }",
    "fn main() -> i32 { let inner: Result<Bytes,u32> = Result::Ok(b\"abc\"); let outer: Result<Result<Bytes,u32>,u32> = Result::Ok(move inner); return 7; }",
    "fn inspect(s: read String) -> i32 { if len(s) == 2u32 { return 7; } return 0; } fn main() -> i32 { let inner: Result<String,u32> = Result::Ok(\"é\"); let outer: Result<Result<String,u32>,u32> = Result::Ok(move inner); return match move outer { Result::Ok(unpacked) => match move unpacked { Result::Ok(text) => inspect(text), Result::Err(e) => 0 }, Result::Err(outer_error) => 0 }; }",
    "fn main() -> i32 { let a: Bytes = b\"\"; let b: String = \"\"; return 7; }",
    "struct Pose { x: f32, items: Array<f32>, n: i32 } fn scale(p: read Pose) -> f32 { return p.x * -2.0; } fn main() -> i32 { let p = Pose { x: 1.5, items: [0.5, -2.0], n: 3 }; if scale(p) < 0.0 and p.x > 1.0 { let items: Array<f32> = move p.items; if items[1u32] < 0.0 { return p.n; } } return 0; }",
    "struct Pose { x: f32, n: i32 } fn make() -> Pose { let p = Pose { x: 0.0 / 0.0, n: 4 }; return p; } fn main() -> i32 { let p = make(); if p.x != p.x { return p.n; } return 0; }",
];

const TRAPS: &[&str] = &[
    "fn main() -> i32 { if true { let done: String = \"é\"; } let bytes: Bytes = b\"a\"; if bytes[1u32] == 0u32 { return 1; } return 0; }",
    "struct Packet { items: Array<i32>, code: i32 } fn main() -> i32 { if true { let done = Packet { items: [1,2], code: 7 }; } let live = Packet { items: [3], code: 9 }; return live.code / 0; }",
    "fn main() -> i32 { let inner: Result<Bytes,u32> = Result::Ok(b\"abc\"); let outer: Result<Result<Bytes,u32>,u32> = Result::Ok(move inner); return match move outer { Result::Ok(unpacked) => match move unpacked { Result::Ok(b) => 1 / 0, Result::Err(e) => 0 }, Result::Err(outer_error) => 0 }; }",
];

fn checked(source: &str) -> check::CheckedProgram {
    let parsed =
        front::parse(source.as_bytes()).unwrap_or_else(|e| panic!("parse: {e:?}\n{source}"));
    check::check(parsed, true).unwrap_or_else(|e| panic!("check: {e:?}\n{source}"))
}

fn compare(actual: &oracle::Observation, expected: &eval::Observation, context: &str) {
    assert_eq!(actual.failure, None, "{context}");
    assert_eq!(actual.semantic.result, expected.result, "{context}");
    assert_eq!(actual.semantic.steps_used, expected.steps_used, "{context}");
    assert_eq!(
        actual.semantic.cleanup_trace, expected.cleanup_trace,
        "{context}"
    );
}

fn differential(source: &str, fuel: u64) -> oracle::Observation {
    let checked = checked(source);
    let net = Net::lower(&checked, Limits::default()).unwrap();
    let proof = net.verify().unwrap();
    let actual = oracle::observe(
        &net,
        &proof,
        oracle::Limits {
            fuel,
            ..Default::default()
        },
    )
    .unwrap();
    compare(&actual, &eval::observe(&checked, fuel), source);
    actual
}

#[test]
fn issue_gnet_02_aggregates_source_results_moves_borrows_and_cleanup() {
    for source in CASES {
        assert!(
            differential(source, 10_000).semantic.result.is_ok(),
            "{source}"
        );
    }
}

#[test]
fn issue_gnet_02_aggregates_match_verified_mir_reference() {
    for source in CASES.iter().chain(TRAPS) {
        let program = mir::lower(&checked(source)).unwrap();
        let net = Net::encode(&program, Limits::default()).unwrap();
        let actual =
            oracle::observe(&net, &net.verify().unwrap(), oracle::Limits::default()).unwrap();
        compare(
            &actual,
            &invar::mir_eval::observe(&program, 100_000),
            source,
        );
    }
}

#[test]
fn issue_gnet_02_aggregates_every_source_fuel_prefix() {
    for source in CASES.iter().chain(TRAPS) {
        let complete = differential(source, 10_000);
        for fuel in 1..=complete.semantic.steps_used {
            differential(source, fuel);
        }
    }
}

#[test]
fn issue_gnet_02_aggregates_traps_keep_original_span_and_completed_cleanup_prefix() {
    for source in TRAPS {
        assert!(
            matches!(
                differential(source, 1000).semantic.result,
                Err(eval::EvalError::Trap { .. })
            ),
            "{source}"
        );
    }
}

#[test]
fn issue_gnet_02_aggregates_nested_cleanup_is_one_outer_event() {
    let source = "fn main() -> i32 { let inner: Result<Bytes,u32> = Result::Ok(b\"abc\"); let outer: Result<Result<Bytes,u32>,u32> = Result::Ok(move inner); return 7; }";
    let observation = differential(source, 1000);
    assert_eq!(
        observation.semantic.cleanup_trace,
        vec![eval::CleanupEvent {
            name: "outer".to_owned(),
            length: 7
        }]
    );
    let source = include_str!("../fixtures/l1_01/record-moved-array.inv");
    assert_eq!(
        differential(source, 1000).semantic.cleanup_trace,
        vec![
            eval::CleanupEvent {
                name: "moved".to_owned(),
                length: 2
            },
            eval::CleanupEvent {
                name: "bundle".to_owned(),
                length: 2
            },
        ]
    );
}

#[test]
fn issue_gnet_02_aggregates_share_and_release_live_owner_budget() {
    let source = "struct Packet { items: Array<i32>, code: i32 } fn main() -> i32 { for i in 0u32..30u32 { let p = Packet { items: [1,2], code: 7 }; let inner: Result<Bytes,u32> = Result::Ok(b\"abc\"); let outer: Result<Result<Bytes,u32>,u32> = Result::Ok(move inner); } return 7; }";
    let net = Net::lower(&checked(source), Limits::default()).unwrap();
    let proof = net.verify().unwrap();
    let actual = oracle::observe(
        &net,
        &proof,
        oracle::Limits {
            array_elements: 11,
            ..Default::default()
        },
    )
    .unwrap();
    compare(&actual, &eval::observe(&checked(source), 100_000), source);
    assert_eq!(actual.semantic.cleanup_trace.len(), 60);
    let rejected = oracle::observe(
        &net,
        &proof,
        oracle::Limits {
            array_elements: 10,
            ..Default::default()
        },
    )
    .unwrap();
    assert_eq!(rejected.failure, Some(oracle::ToolFailure::Allocation));
    assert!(rejected.semantic.cleanup_trace.is_empty());
    assert!(matches!(
        rejected.semantic.result,
        Err(eval::EvalError::Tool {
            code: "RESOURCE_LIMIT",
            ..
        })
    ));
}

#[test]
fn issue_gnet_02_aggregates_foreign_host_boundary_remains_explicitly_unsupported() {
    // Every checked intrinsic now has an oracle rule; a declared foreign call
    // stays an explicit host boundary refused before any demand.
    let source = include_str!("../fixtures/l1_04/foreign-oracle.inv");
    let net = Net::lower(&checked(source), Limits::default()).unwrap();
    let actual = oracle::observe(&net, &net.verify().unwrap(), oracle::Limits::default()).unwrap();
    assert_eq!(actual.failure, Some(oracle::ToolFailure::Unsupported));
    assert_eq!(actual.semantic.steps_used, 0);
    assert!(actual.events.is_empty());
    assert!(actual.semantic.cleanup_trace.is_empty());
}

#[test]
fn issue_gnet_02_aggregates_named_owner_results_snapshot_remaining_children() {
    for source in [
        "fn output() -> Bytes { return b\"a\\xFF\"; }",
        "fn output() -> String { return \"é\"; }",
        "struct Packet { items: Array<i32>, code: i32 } fn output() -> Packet { return Packet { code: 7, items: [1,2] }; }",
        "fn output() -> Result<Result<Bytes,u32>,u32> { let inner: Result<Bytes,u32> = Result::Ok(b\"abc\"); return Result::Ok(move inner); }",
    ] {
        // This exercises the oracle's named entry with no external owner args;
        // it does not expand the frozen compiled scalar-export ABI.
        let checked = check::check(front::parse(source.as_bytes()).unwrap(), false).unwrap();
        let net = Net::lower(&checked, Limits::default()).unwrap();
        let actual = oracle::observe_call(
            &net,
            &net.verify().unwrap(),
            "output",
            &[],
            oracle::Limits::default(),
        )
        .unwrap();
        // MIR reference exposes a main entry only. Rename the already verified
        // no-argument entry, preserving its blocks, operations and Tick stream.
        let mut reference = mir::lower(&checked).unwrap();
        reference.functions[0].name = "main".to_owned();
        mir::verify(&reference).unwrap();
        compare(
            &actual,
            &invar::mir_eval::observe(&reference, 100_000),
            source,
        );
    }
}

#[test]
fn issue_gnet_02_aggregates_blob_limit_is_independent_of_array_length_limit() {
    let bytes = "a".repeat(1_000_001);
    let source = format!(
        "fn main() -> i32 {{ let bytes: Bytes = b\"{bytes}\"; if len(bytes) == 1000001u32 {{ return 7; }} return 0; }}"
    );
    let checked = checked(&source);
    let net = Net::lower(
        &checked,
        Limits {
            literal_bytes: 1_048_576,
            ..Default::default()
        },
    )
    .unwrap();
    let actual = oracle::observe(&net, &net.verify().unwrap(), oracle::Limits::default()).unwrap();
    compare(&actual, &eval::observe(&checked, 100_000), "large blob");
    assert_eq!(actual.semantic.result, Ok(eval::Value::I32(7)));
    assert_eq!(actual.semantic.cleanup_trace[0].length, 1_000_001);
}

#[test]
fn issue_gnet_02_aggregates_source_global_live_limit_counts_record_shell() {
    // The default physical oracle budget is independent of language admission.
    let old = "struct Pair { a: Array<i32>, b: Array<i32> } fn main() -> i32 { let pair = Pair { a: filled_i32(1000000u32, 1), b: filled_i32(1000000u32, 2) }; return 0; }";
    let program = checked(old);
    assert_eq!(
        eval::observe(&program, 1000).result,
        Ok(eval::Value::I32(0))
    );
    let net = Net::lower(&program, Limits::default()).unwrap();
    let physical =
        oracle::observe(&net, &net.verify().unwrap(), oracle::Limits::default()).unwrap();
    assert_eq!(physical.failure, Some(oracle::ToolFailure::Allocation));

    // Two legal one-element arrays fill the ceiling; the record shell exceeds it.
    let source = format!(
        "struct Pair {{ a: Array<i32>, b: Array<i32> }} fn main() -> i32 {{ {} let pair = Pair {{ a: filled_i32(1u32, 1), b: filled_i32(1u32, 2) }}; return 0; }}",
        language_limits::padding(invar::array_limits::MAX_LIVE_ELEMENTS - 2)
    );
    let program = checked(&source);
    let net = Net::lower(&program, Limits::default()).unwrap();
    let actual = oracle::observe(
        &net,
        &net.verify().unwrap(),
        oracle::Limits {
            fuel: 1000,
            array_elements: invar::array_limits::MAX_LIVE_ELEMENTS,
            ..Default::default()
        },
    )
    .unwrap();
    compare(&actual, &eval::observe(&program, 1000), &source);
    assert_eq!(actual.failure, None);
    assert!(matches!(
        actual.semantic.result,
        Err(eval::EvalError::Tool {
            code: "ARRAY_SIZE_LIMIT",
            ..
        })
    ));
    assert!(actual.semantic.cleanup_trace.is_empty());
}

#[test]
fn issue_gnet_02_aggregates_source_record_shell_counts_toward_oracle_budget() {
    let source = "struct Pair { a: Array<i32>, b: Array<i32> } fn main() -> i32 { let pair = Pair { a: filled_i32(1000000u32, 1), b: filled_i32(1000000u32, 2) }; return 0; }";
    let checked = checked(source);
    let net = Net::lower(&checked, Limits::default()).unwrap();
    let proof = net.verify().unwrap();
    // The 2^28 device caps raised the language ceiling. This source is valid, but the default
    // oracle budget still admits only the two arrays, not their record shell.
    let actual = oracle::observe(&net, &proof, oracle::Limits::default()).unwrap();
    assert_eq!(actual.failure, Some(oracle::ToolFailure::Allocation));
    assert!(matches!(
        actual.semantic.result,
        Err(eval::EvalError::Tool {
            code: "RESOURCE_LIMIT",
            ..
        })
    ));
    assert!(actual.semantic.cleanup_trace.is_empty());
    let admitted = oracle::observe(
        &net,
        &proof,
        oracle::Limits {
            array_elements: 2_000_002,
            ..Default::default()
        },
    )
    .unwrap();
    compare(&admitted, &eval::observe(&checked, 1000), source);
    assert_eq!(admitted.semantic.result, Ok(eval::Value::I32(0)));
    assert_eq!(admitted.semantic.cleanup_trace[0].length, 2_000_002);
}
