//! Issue #258: `assert(condition, "message")` and its trap `ASSERTION_FAILED`,
//! identical in category, span, fuel prefix and cleanup on every engine.
use invar::general_net::{Limits, Net, oracle};
use invar::{check, engine, eval, front, mir, mir_eval};
use std::process::Command;

fn checked(source: &str) -> check::CheckedProgram {
    let parsed = front::parse(source.as_bytes()).unwrap_or_else(|e| panic!("{source}: {e:?}"));
    check::check(parsed, true).unwrap_or_else(|e| panic!("{source}: {e:?}"))
}

/// The smallest fuel at which `run` no longer ends in an evaluation limit.
fn fuel_prefix(run: impl Fn(u64) -> Result<eval::Value, eval::EvalError>) -> u64 {
    let (mut low, mut high) = (1u64, 1_000_000u64);
    while low < high {
        let mid = low + (high - low) / 2;
        match run(mid) {
            Err(eval::EvalError::EvaluationLimit { .. }) => low = mid + 1,
            _ => high = mid,
        }
    }
    low
}

/// `source` must trap `ASSERTION_FAILED` at the assert whose text starts at
/// `at`, identically on the reference, MIR, general-net oracle, cpu-parallel
/// and (with the `native` feature) native engines.
fn traps_everywhere(source: &str, at: &str) {
    let checked = checked(source);
    let start = source.find(at).unwrap_or_else(|| panic!("{at}"));
    let end = start + source[start..].find(';').unwrap();
    let expected = eval::EvalError::Trap {
        category: eval::TrapCategory::AssertionFailed,
        span: front::Span { start, end },
    };
    let reference = eval::observe(&checked, 1_000_000);
    assert_eq!(
        reference.result.as_ref().unwrap_err(),
        &expected,
        "{source}"
    );
    let lowered = mir::lower(&checked).unwrap();
    let observed = mir_eval::observe(&lowered, 1_000_000);
    assert_eq!(observed.result, reference.result, "MIR\n{source}");
    assert_eq!(observed.steps_used, reference.steps_used, "MIR\n{source}");
    assert_eq!(
        observed.cleanup_trace, reference.cleanup_trace,
        "MIR\n{source}"
    );
    let prefix = fuel_prefix(|fuel| eval::run(&checked, fuel).map(|o| o.value));
    assert_eq!(
        fuel_prefix(|fuel| mir_eval::run(&lowered, fuel).map(|o| o.value)),
        prefix,
        "MIR fuel\n{source}"
    );
    let net = Net::lower(&checked, Limits::default()).unwrap();
    let proof = net.verify().unwrap();
    let net_run = oracle::observe(&net, &proof, oracle::Limits::default()).unwrap();
    assert_eq!(net_run.failure, None, "{source}");
    assert_eq!(
        net_run.semantic.result, reference.result,
        "oracle\n{source}"
    );
    assert_eq!(
        net_run.semantic.steps_used, reference.steps_used,
        "oracle\n{source}"
    );
    for fuel in [prefix, prefix - 1] {
        let parallel = engine::run_mir(
            source.as_bytes(),
            &lowered,
            engine::Engine::CpuParallel,
            fuel,
            4,
            None,
        );
        let serial = eval::run(&checked, fuel).map(|o| o.value);
        assert_eq!(
            format!("{:?}", parallel.map(|r| r["result"].clone())),
            format!("{:?}", serial.map(|_| ())),
            "cpu-parallel at fuel {fuel}\n{source}"
        );
    }
    #[cfg(feature = "native")]
    {
        let (status, document) = invar::native::run_program(&lowered, 1_000_000).unwrap();
        assert_eq!(status, 3, "{document}");
        assert_eq!(document["trap"]["category"], "ASSERTION_FAILED");
        assert_eq!(document["trap"]["span"]["start"], start);
        assert_eq!(document["trap"]["span"]["end"], end);
        assert_eq!(document["steps_used"], reference.steps_used);
    }
}

#[test]
fn a_false_assert_traps_identically_on_every_engine() {
    traps_everywhere(
        r#"fn main() -> i32 { let x: i32 = 3; assert(x == 4, "x is four"); return x; }"#,
        "assert(x == 4",
    );
    traps_everywhere(
        r#"fn mean(xs: read Array<u32>) -> u32 {
            assert(len(xs) > 0u32, "mean of an empty array");
            return xs[0u32] / len(xs);
        }
        fn main() -> i32 {
            let full: Array<u32> = [4u32, 2u32];
            let empty: Array<u32> = [];
            if mean(full) == 2u32 and mean(empty) == 0u32 { return 1; }
            return 0;
        }"#,
        "assert(len(xs)",
    );
    // The first failing iteration in source order, with its owners live.
    traps_everywhere(
        r#"fn main() -> i32 {
            let xs: Array<u32> = [1u32, 2u32, 9u32, 3u32, 8u32];
            var ys: Array<u32> = filled_u32(5u32, 0u32);
            for i in 0u32 .. len(xs) {
                assert(xs[i] < 5u32, "small");
                ys[i] = wrapping_add(xs[i], 1u32);
            }
            return 0;
        }"#,
        "assert(xs[i]",
    );
}

#[test]
fn an_assert_runs_its_condition_then_traps_in_source_trap_order() {
    // The condition's own trap comes first; the message is never evaluated.
    let source = r#"fn main() -> i32 { let zero: u32 = 0u32; assert(1u32 / zero == 1u32, "unreachable"); return 0; }"#;
    let result = eval::run(&checked(source), 1_000).unwrap_err();
    assert!(matches!(
        result,
        eval::EvalError::Trap {
            category: eval::TrapCategory::DivisionByZero,
            ..
        }
    ));
    // A true assert costs its statement, the call and its condition.
    let with = r#"fn main() -> i32 { let x: i32 = 1; assert(x == 1, "one"); return x; }"#;
    let without = "fn main() -> i32 { let x: i32 = 1; return x; }";
    let steps = |source| crate::engines::agree(source, 1_000, 1).steps_used;
    assert_eq!(
        steps(with),
        steps(without) + 5,
        "dispatch, call, ==, x and 1"
    );
}

#[test]
fn assert_misuse_is_refused_at_check_time() {
    for (body, code) in [
        (r#"assert(true);"#, "TYPE_MISMATCH"),
        (r#"assert(true, "a", "b");"#, "TYPE_MISMATCH"),
        (r#"assert(1, "one");"#, "TYPE_MISMATCH"),
        (r#"let s: String = "x"; assert(true, s);"#, "TYPE_MISMATCH"),
        (r#"let u = assert(true, "x");"#, "TYPE_MISMATCH"),
    ] {
        let source = format!("fn main() -> i32 {{ {body} return 0; }}");
        let error = front::parse(source.as_bytes())
            .map_err(|e| e.code)
            .and_then(|p| check::check(p, true).map_err(|e| e.code));
        assert_eq!(error.err(), Some(code), "{body}");
    }
    let shadow =
        r#"fn assert(x: Bool, y: i32) -> i32 { return y; } fn main() -> i32 { return 0; }"#;
    let parsed = front::parse(shadow.as_bytes()).unwrap();
    assert!(
        check::check(parsed, true).is_err(),
        "assert is a reserved intrinsic"
    );
}

#[test]
fn the_cli_reports_the_assertion_message_and_span() {
    let source = "fn main() -> i32 {\n    assert(1 + 1 == 3, \"arithmetic \\\"works\\\"\");\n    return 0;\n}\n";
    let path = std::env::temp_dir().join(format!("invar-assert-{}.inv", std::process::id()));
    std::fs::write(&path, source).unwrap();
    let start = source.find("assert").unwrap();
    let engines = [
        "reference",
        #[cfg(feature = "native")]
        "native-serial",
    ];
    for engine in engines {
        let output = Command::new(env!("CARGO_BIN_EXE_invar"))
            .args([
                "run",
                path.to_str().unwrap(),
                "--engine",
                engine,
                "--format",
                "json",
            ])
            .output()
            .unwrap();
        assert_eq!(output.status.code(), Some(3), "{engine}");
        let document: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
        assert_eq!(
            document["trap"],
            serde_json::json!({
                "category": "ASSERTION_FAILED",
                "message": "arithmetic \"works\"",
                "span": {"start": start, "end": source.find(';').unwrap()}
            }),
            "{engine}"
        );
        assert_eq!(document["diagnostics"][0]["code"], "ASSERTION_FAILED");
        assert_eq!(
            document["diagnostics"][0]["message"], "assertion failed: arithmetic \"works\"",
            "{engine}"
        );
        assert_eq!(document["diagnostics"][0]["span"]["start_line"], 2);
    }
    let human = Command::new(env!("CARGO_BIN_EXE_invar"))
        .args(["run", path.to_str().unwrap(), "--engine", "reference"])
        .output()
        .unwrap();
    assert_eq!(human.status.code(), Some(3));
    let stderr = String::from_utf8(human.stderr).unwrap();
    assert!(
        stderr.contains("assertion failed: arithmetic \"works\""),
        "{stderr}"
    );
    std::fs::remove_file(path).unwrap();
}

#[test]
fn a_loop_with_an_assert_is_not_admitted_and_runs_serially() {
    let plain = "fn main() -> i32 {
        let xs: Array<u32> = [1u32, 2u32, 9u32, 3u32];
        var ys: Array<u32> = filled_u32(4u32, 0u32);
        for i in 0u32 .. len(xs) {
            ys[i] = wrapping_add(xs[i], 1u32);
        }
        return 0;
    }";
    let asserted = plain.replace(
        "ys[i] = wrapping_add",
        "assert(xs[i] < 5u32, \"small\");\n            ys[i] = wrapping_add",
    );
    let explain = |source: &str| {
        let path = std::env::temp_dir().join(format!(
            "invar-assert-region-{}-{}.inv",
            std::process::id(),
            source.len()
        ));
        std::fs::write(&path, source).unwrap();
        let output = Command::new(env!("CARGO_BIN_EXE_invar"))
            .args(["explain", path.to_str().unwrap(), "--format", "json"])
            .output()
            .unwrap();
        std::fs::remove_file(path).unwrap();
        let document: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
        let region = &document["regions"][0];
        (region["status"].clone(), region["chosen_plan"].clone())
    };
    assert_eq!(explain(plain).0, "admitted");
    assert_eq!(explain(&asserted), ("rejected".into(), "mir-serial".into()));
}
