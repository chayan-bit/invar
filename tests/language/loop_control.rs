//! Issue #261: `break` and `continue`, with optional loop labels, agree in
//! value, trap, fuel and cleanup on every engine; a loop that exits early
//! stays serial unless the net recovers it as a guarded body.
use crate::surface_sugar::first_code;
use invar::general_net::{Limits, Net, oracle};
use invar::{check, engine, eval, front, mir, mir_eval};

fn checked(source: &str) -> check::CheckedProgram {
    let parsed = front::parse(source.as_bytes()).unwrap_or_else(|e| panic!("{source}: {e:?}"));
    check::check(parsed, true).unwrap_or_else(|e| panic!("{source}: {e:?}"))
}

/// The cpu-parallel receipt of `source` at `fuel`.
fn parallel(source: &str, fuel: u64) -> Result<serde_json::Value, eval::EvalError> {
    let lowered = mir::lower(&checked(source)).unwrap();
    engine::run_mir(
        source.as_bytes(),
        &lowered,
        engine::Engine::CpuParallel,
        fuel,
        4,
        None,
    )
}

/// `source` ends with `expected` (a value or a trap category) identically on
/// the reference, MIR, net oracle, cpu-parallel and (with the `native`
/// feature) native engines, and at every fuel short of completion the MIR
/// and cpu-parallel engines stop where the reference does.
fn everywhere(source: &str, expected: Result<i32, eval::TrapCategory>) {
    let checked = checked(source);
    let reference = eval::observe(&checked, 1_000_000);
    match (&reference.result, expected) {
        (Ok(value), Ok(want)) => assert_eq!(value, &eval::Value::I32(want), "{source}"),
        (Err(eval::EvalError::Trap { category, .. }), Err(want)) => {
            assert_eq!(*category, want, "{source}")
        }
        (other, _) => panic!("{source}: {other:?}"),
    }
    let lowered = mir::lower(&checked).unwrap();
    let observed = mir_eval::observe(&lowered, 1_000_000);
    assert_eq!(observed.result, reference.result, "MIR\n{source}");
    assert_eq!(observed.steps_used, reference.steps_used, "MIR\n{source}");
    assert_eq!(
        observed.cleanup_trace, reference.cleanup_trace,
        "MIR\n{source}"
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
    let full = reference.steps_used;
    for fuel in [full, full - 1, full / 2, 1] {
        let serial = eval::observe(&checked, fuel);
        let mir_run = mir_eval::observe(&lowered, fuel);
        assert_eq!(
            mir_run.result, serial.result,
            "MIR at fuel {fuel}\n{source}"
        );
        assert_eq!(
            mir_run.cleanup_trace, serial.cleanup_trace,
            "MIR at fuel {fuel}\n{source}"
        );
        let receipt = engine::run_mir(
            source.as_bytes(),
            &lowered,
            engine::Engine::CpuParallel,
            fuel,
            4,
            None,
        );
        assert_eq!(
            format!("{:?}", receipt.map(|r| r["result"].clone())),
            format!(
                "{:?}",
                serial.result.map(|v| match v {
                    eval::Value::I32(v) => serde_json::json!(v),
                    other => panic!("{other:?}"),
                })
            ),
            "cpu-parallel at fuel {fuel}\n{source}"
        );
    }
    #[cfg(feature = "native")]
    {
        let (status, document) = invar::native::run_program(&lowered, 1_000_000).unwrap();
        match &reference.result {
            Ok(eval::Value::I32(value)) => {
                assert_eq!(status, 0, "{source}: {document}");
                assert_eq!(
                    document["result"],
                    serde_json::json!({"type":"i32","value":value}),
                    "{source}"
                );
            }
            Err(eval::EvalError::Trap { span, .. }) => {
                assert_eq!(status, 3, "{source}: {document}");
                assert_eq!(document["trap"]["span"]["start"], span.start, "{source}");
            }
            other => panic!("{other:?}"),
        }
        assert_eq!(
            document["steps_used"], reference.steps_used,
            "native\n{source}"
        );
        let trace: Vec<_> = reference
            .cleanup_trace
            .iter()
            .map(|event| serde_json::json!({"name":event.name,"length":event.length}))
            .collect();
        assert_eq!(
            document["cleanup_trace"],
            serde_json::json!(trace),
            "native\n{source}"
        );
    }
}

const SEARCH: &str = "fn find(xs: read Array<i32>, key: i32) -> u32 {
    var at = len(xs);
    for i in 0u32 .. len(xs) {
        if xs[i] == key { at = i; break; }
    }
    return at;
}
fn main() -> i32 {
    let xs: Array<i32> = [4, 9, 2, 7, 7, 1];
    if find(xs, 7) == 3u32 and find(xs, 5) == 6u32 { return 1; }
    return 0;
}";

#[test]
fn break_and_continue_agree_on_every_engine() {
    everywhere(SEARCH, Ok(1));
    // `continue` skips the rest of a trip; `while` and `for` alike.
    everywhere(
        "fn main() -> i32 {
            let xs: Array<i32> = [3, 8, 5, 6, 7];
            var s = 0;
            for i in 0u32 .. len(xs) {
                if xs[i] % 2 != 0 { continue; }
                s += xs[i];
            }
            var j = 0u32;
            while j < len(xs) {
                j += 1u32;
                if j == 2u32 { continue; }
                if j == 4u32 { break; }
                s += 100;
            }
            return s;
        }",
        Ok(214),
    );
    // Labels reach an outer loop from inside a nested `while`.
    everywhere(
        "fn main() -> i32 {
            var hits = 0;
            outer: for i in 0u32 .. 9u32 {
                var j = 0u32;
                while j < 9u32 {
                    j += 1u32;
                    if j == i { continue outer; }
                    if i * j > 20u32 { break outer; }
                    hits += 1;
                }
            }
            return hits;
        }",
        Ok(22),
    );
    // Owners declared in the loop are cleaned up, innermost first, on every exit.
    everywhere(
        "fn main() -> i32 {
            let keep: Array<u32> = filled_u32(2u32, 7u32);
            var n = 0;
            rows: for i in 0u32 .. 4u32 {
                let row: Array<u32> = filled_u32(3u32, i);
                for j in 0u32 .. 3u32 {
                    let cell: Array<u32> = filled_u32(1u32, j);
                    if row[j] == 1u32 and cell[0u32] == 1u32 { continue rows; }
                    if row[j] == 2u32 { break rows; }
                    n += 1;
                }
                n += 10;
            }
            if keep[1u32] == 7u32 { n += 1000; }
            return n;
        }",
        Ok(1014),
    );
    // For-each, by value and by place, and a stepped range.
    everywhere(
        "fn main() -> i32 {
            var xs: Array<u32> = [1u32, 2u32, 3u32, 4u32, 5u32];
            var s = 0u32;
            for x in xs { if x == 4u32 { break; } s += x; }
            for y in xs { if y == 2u32 { continue; } y = y * 10u32; }
            for i in 0u32 .. 20u32 step 3u32 { if i > 10u32 { break; } s += i; }
            if s == 24u32 and xs[0u32] == 10u32 and xs[1u32] == 2u32 and xs[4u32] == 50u32 { return 1; }
            return 0;
        }",
        Ok(1),
    );
    // Unconditional exits, exits in every arm, and dead code after an exit.
    everywhere(
        "fn main() -> i32 {
            var s = 0;
            for i in 0u32 .. 5u32 { break; }
            for j in 0u32 .. 5u32 { s += 1; continue; s += 1000; }
            var k = 0u32;
            while true {
                k += 1u32;
                if k < 3u32 { continue; } else if k == 3u32 { s += 10; continue; } else { break; }
            }
            return s;
        }",
        Ok(15),
    );
    // A trap after a `continue` keeps source order and its fuel prefix.
    everywhere(
        "fn main() -> i32 {
            var s = 0u32;
            for i in 0u32 .. 6u32 {
                if i == 1u32 { continue; }
                s += 12u32 / (3u32 - i);
            }
            return 0;
        }",
        Err(eval::TrapCategory::DivisionByZero),
    );
}

#[test]
fn an_early_exit_costs_and_cleans_up_like_return() {
    // Each `break` ends its Unit function exactly where `return;` does, with
    // the same owners dropped in the same order, at the same step count.
    for (exit, body) in [
        (
            "break;",
            "for i in 0u32 .. len(xs) { let t: Array<u32> = filled_u32(2u32, i); if xs[i] == 0u32 { EXIT } }",
        ),
        (
            "break outer;",
            "outer: for i in 0u32 .. len(xs) { let t: Array<u32> = filled_u32(2u32, i); \
             var j = 0u32; while j < 2u32 { let u: Array<u32> = filled_u32(3u32, j); \
             if xs[i] == j { EXIT } j += 1u32; } }",
        ),
    ] {
        let program = |exit: &str| {
            format!(
                "fn walk(xs: read Array<u32>) -> Unit {{ let w: Array<u32> = filled_u32(4u32, 1u32); {} }}\n\
                 fn main() -> i32 {{ let xs: Array<u32> = [3u32, 2u32, 0u32, 5u32]; walk(xs); return 1; }}",
                body.replace("EXIT", exit)
            )
        };
        let (early, returned) = (program(exit), program("return;"));
        for fuel in [1_000_000, 40, 20] {
            let (a, b) = (
                eval::observe(&checked(&early), fuel),
                eval::observe(&checked(&returned), fuel),
            );
            assert_eq!(a.result, b.result, "{early}");
            assert_eq!(a.steps_used, b.steps_used, "{early}");
            assert_eq!(a.cleanup_trace, b.cleanup_trace, "{early}");
        }
        everywhere(&early, Ok(1));
        everywhere(&returned, Ok(1));
    }
}

#[test]
fn a_search_with_break_plans_like_its_flag_version() {
    // Both stay serial frames: the net does not yet map `break` to the
    // first-exit law (L15) that the `return` form takes.
    let flag = SEARCH.replace(
        "var at = len(xs);\n    for i in 0u32 .. len(xs) {\n        if xs[i] == key { at = i; break; }\n    }",
        "var at = len(xs);\n    var done = false;\n    for i in 0u32 .. len(xs) {\n        if not done { if xs[i] == key { at = i; done = true; } }\n    }",
    );
    assert_ne!(flag, SEARCH);
    everywhere(&flag, Ok(1));
    let (a, b) = (
        parallel(SEARCH, 1_000_000).unwrap(),
        parallel(&flag, 1_000_000).unwrap(),
    );
    assert_eq!(a, b);
    assert_ne!(a["normal_form"]["status"], "routed", "{a}");
    assert_eq!(
        parallel(&SEARCH.replace("at = i; break;", "return i;"), 1_000_000).unwrap()["normal_form"]
            ["laws"],
        serde_json::json!(["L1", "L15"])
    );
}

#[test]
fn an_innermost_continue_is_a_guarded_body_and_break_stays_serial() {
    let map = |guard: &str| {
        format!(
            "fn main() -> i32 {{
                let xs: Array<u32> = [1u32, 0u32, 3u32, 4u32, 0u32, 6u32, 7u32, 8u32];
                var ys: Array<u32> = filled_u32(8u32, 0u32);
                for i in 0u32 .. len(xs) {{ {guard} }}
                if ys[2u32] == 4u32 and ys[7u32] == 9u32 {{ return 1; }}
                return 0;
            }}"
        )
    };
    let skip = map("if xs[i] == 0u32 { continue; } ys[i] = wrapping_add(xs[i], 1u32);");
    let guarded = map("if xs[i] != 0u32 { ys[i] = wrapping_add(xs[i], 1u32); }");
    let stopped = map("if xs[i] == 99u32 { break; } ys[i] = wrapping_add(xs[i], 1u32);");
    for source in [&skip, &guarded, &stopped] {
        everywhere(source, Ok(1));
    }
    let laws = |source: &str| parallel(source, 1_000_000).unwrap()["normal_form"].clone();
    assert_eq!(laws(&skip)["status"], "routed");
    assert_eq!(laws(&skip)["laws"], laws(&guarded)["laws"]);
    assert_eq!(laws(&skip)["laws"], serde_json::json!(["L1"]));
    assert_ne!(laws(&stopped)["status"], "routed");
}

#[test]
fn loop_control_outside_its_loop_is_refused() {
    let body = |text: &str| format!("fn main() -> i32 {{ var s = 0; {text} return s; }}");
    for (text, code) in [
        ("break;", "PARSE_LOOP_CONTROL"),
        ("continue;", "PARSE_LOOP_CONTROL"),
        ("if s == 0 { break; }", "PARSE_LOOP_CONTROL"),
        (
            "for i in 0u32 .. 2u32 { break nope; }",
            "PARSE_LOOP_CONTROL",
        ),
        (
            "a: for i in 0u32 .. 2u32 { } for j in 0u32 .. 2u32 { continue a; }",
            "PARSE_LOOP_CONTROL",
        ),
        (
            "a: for i in 0u32 .. 2u32 { a: while false { } }",
            "PARSE_LOOP_CONTROL",
        ),
        // A closure body is its own function: no loop around it is in reach.
        (
            "for i in 0u32 .. 2u32 { let f = closure[](x: i32) -> i32 { break; return x; }; s = f(1); }",
            "PARSE_LOOP_CONTROL",
        ),
        ("a: let t = 1;", "PARSE_EXPECTED_TOKEN"),
        ("a: s = 1;", "PARSE_EXPECTED_TOKEN"),
        ("for i in 0u32 .. 2u32 { break 3; }", "PARSE_EXPECTED_TOKEN"),
        ("for i in 0u32 .. 2u32 { break }", "PARSE_EXPECTED_TOKEN"),
        // Labels are reusable by sibling loops and live apart from names.
        (
            "a: for i in 0u32 .. 2u32 { break a; } a: while s < 3 { s += 1; continue a; }",
            "OK",
        ),
        (
            "let a = 1; a: for i in 0u32 .. 2u32 { s += a; break a; }",
            "OK",
        ),
        // A loop still moves no owner from outside it, exit or not.
    ] {
        assert_eq!(first_code(&body(text)), code, "{text}");
    }
    // A loop still moves no owner from outside it, exit or not.
    let moved = "let xs: Array<u32> = [1u32]; for i in 0u32 .. 2u32 { let ys = move xs; EXIT }";
    for exit in ["break;", "continue;"] {
        assert_eq!(
            first_code(&body(&moved.replace("EXIT", exit))),
            first_code(&body(&moved.replace("EXIT", "")))
        );
        assert_ne!(first_code(&body(&moved.replace("EXIT", exit))), "OK");
    }
    // An exit is not a return: a function whose loop always breaks still needs one.
    assert_eq!(
        first_code("fn main() -> i32 { while true { break; } }"),
        first_code("fn main() -> i32 { while true { } }"),
    );
    let at = "fn main() -> i32 { break; return 0; }";
    let error = front::parse(at.as_bytes()).unwrap_err();
    assert_eq!(error.span, Some(front::Span { start: 19, end: 25 }));
}

#[test]
fn a_tiered_fold_with_an_early_exit_is_refused() {
    let fold = |exit: &str| {
        format!(
            "#[numeric(fold = \"tree\")]\nfn sum_all(xs: read Array<f32>) -> f32 {{ var acc: f32 = 0.0; \
             for i in 0u32 .. len(xs) {{ {exit} acc = acc + xs[i]; }} return acc; }}\n\
             fn main() -> i32 {{ let xs: Array<f32> = [1.0, 2.0]; if sum_all(xs) == 3.0 {{ return 7; }} return 0; }}"
        )
    };
    assert_eq!(first_code(&fold("")), "OK");
    assert_eq!(
        first_code(&fold("if xs[i] > 1.5 { break; }")),
        "NUMERIC_POLICY"
    );
    assert_eq!(
        first_code(&fold("if xs[i] > 1.5 { continue; }")),
        "NUMERIC_POLICY"
    );
}
