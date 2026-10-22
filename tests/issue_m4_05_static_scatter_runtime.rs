#![forbid(unsafe_code)]
#[path = "support/language_limits.rs"]
mod language_limits;
use invar::{
    check, eval, front, general_net, mir, mir_eval,
    scatter_proof::host::{self, Budget},
};
fn array(values: &[u32]) -> String {
    format!(
        "[{}]",
        values
            .iter()
            .map(|v| format!("{v}u32"))
            .collect::<Vec<_>>()
            .join(",")
    )
}
fn leaf(lower: u32, upper: u32, n: u32, dest: &str) -> String {
    format!(
        "fn scatter(a: read Array<u32>) -> Array<u32> {{ var out = filled_u32({n}u32,0u32); for i in {lower}u32..{upper}u32 {{ out[{dest}] = a[i]; }} return out; }}"
    )
}
fn source(f: &str, input: &[u32], expected: &[u32]) -> String {
    format!(
        "{f} fn main() -> i32 {{ let a: Array<u32> = {}; let out = scatter(a); let wanted: Array<u32> = {}; if len(out) != len(wanted) {{ return 0; }} for k in 0u32..len(wanted) {{ if out[k] != wanted[k] {{ return 0; }} }} return 7; }}",
        array(input),
        array(expected)
    )
}
fn compare(a: &eval::Observation, b: &eval::Observation) {
    assert_eq!(a.result, b.result);
    assert_eq!(a.steps_used, b.steps_used);
    assert_eq!(a.cleanup_trace, b.cleanup_trace);
}
fn prefixes(s: &str, budget: Budget, admitted: bool) -> eval::Observation {
    prefixes_at(s, budget, admitted, true)
}
fn prefixes_at(s: &str, budget: Budget, admitted: bool, every_prefix: bool) -> eval::Observation {
    let checked = check::check(front::parse(s.as_bytes()).unwrap(), true).unwrap();
    let m = mir::lower(&checked).unwrap();
    let net = general_net::Net::lower(&checked, general_net::Limits::default()).unwrap();
    let proof = net.verify().unwrap();
    let complete = eval::observe(&checked, 1_000_000);
    let mut publications = 0;
    let fuels: Vec<_> = if every_prefix {
        (0..=complete.steps_used + 1).collect()
    } else {
        vec![
            complete.steps_used - 1,
            complete.steps_used,
            complete.steps_used + 1,
        ]
    };
    let prefix_count = fuels.len();
    for fuel in fuels {
        let reference = eval::observe(&checked, fuel);
        let run = host::run_source(s.as_bytes(), fuel, budget).unwrap();
        compare(&run.observation, &reference);
        compare(&mir_eval::observe(&m, fuel), &reference);
        let oracle = general_net::oracle::observe(
            &net,
            &proof,
            general_net::oracle::Limits {
                fuel,
                array_elements: invar::array_limits::MAX_LIVE_ELEMENTS,
                ..general_net::oracle::Limits::default()
            },
        )
        .unwrap();
        compare(&oracle.semantic, &reference);
        assert_eq!(run.begun, run.completed);
        assert!(run.pending.is_none());
        publications += run.completed;
    }
    let full = host::run_source(s.as_bytes(), 1_000_000, budget).unwrap();
    assert_eq!(full.completed > 0, admitted);
    assert_eq!(full.begun, full.completed);
    println!(
        "scatter runtime fuel prefixes: {}; completed loop publications: {publications}",
        prefix_count
    );
    complete
}
#[test]
fn issue_m4_05_gather_scatter_frontier_scatter_runtime_independent_outputs_holes_and_recipes() {
    for (lo, hi, n, dest, input, expected) in [
        (0, 0, 0, "i", vec![], vec![]),
        (
            0,
            0,
            3,
            "4294967295u32*i+4294967295u32",
            vec![],
            vec![0, 0, 0],
        ),
        (0, 1, 4, "0u32*i+3u32", vec![11], vec![0, 0, 0, 11]),
        (0, 1, 1, "4294967295u32*i", vec![19], vec![19]),
        (0, 3, 3, "i", vec![7, 11, 19], vec![7, 11, 19]),
        (0, 3, 6, "2u32*i", vec![7, 11, 19], vec![7, 0, 11, 0, 19, 0]),
        (
            0,
            3,
            7,
            "1u32+(i*2u32)",
            vec![7, 11, 19],
            vec![0, 7, 0, 11, 0, 19, 0],
        ),
    ] {
        let s = source(&leaf(lo, hi, n, dest), &input, &expected);
        assert_eq!(
            prefixes(&s, Budget::default(), hi > lo).result,
            Ok(eval::Value::I32(7))
        );
    }
    let f = leaf(0, 3, 6, "2u32*i")
        .replace("0u32..3u32", "(0u32)..(3u32)")
        .replace("out[2u32*i] = a[i]", "out[((2u32)*(i))] = (a[(i)])");
    for f in [
        f,
        leaf(0, 3, 6, "2u32*i").replace("return out;", "return move out;"),
        leaf(0, 3, 6, "2u32*i").replace("return out;", "return (move out);"),
    ] {
        assert_eq!(
            prefixes(
                &source(&f, &[7, 11, 19], &[7, 0, 11, 0, 19, 0]),
                Budget::default(),
                true
            )
            .result,
            Ok(eval::Value::I32(7))
        );
    }
}
#[test]
fn issue_m4_05_gather_scatter_frontier_scatter_runtime_proof_refusals_and_trap_prefixes() {
    for (f, input, category) in [
        (
            leaf(0, 3, 6, "2u32*i"),
            vec![],
            eval::TrapCategory::IndexOutOfBounds,
        ),
        (
            leaf(0, 3, 6, "2u32*i"),
            vec![7, 11],
            eval::TrapCategory::IndexOutOfBounds,
        ),
        (
            leaf(0, 2, 4, "4294967295u32*i+1u32"),
            vec![],
            eval::TrapCategory::IndexOutOfBounds,
        ),
        (
            leaf(0, 2, 4, "4294967295u32*i+1u32"),
            vec![7, 11],
            eval::TrapCategory::IntegerOverflow,
        ),
        (
            leaf(0, 3, 4, "2u32*i"),
            vec![7, 11, 19],
            eval::TrapCategory::IndexOutOfBounds,
        ),
    ] {
        let s = source(&f, &input, &[]);
        let done = prefixes(&s, Budget::default(), false);
        let Err(eval::EvalError::Trap {
            category: actual,
            span,
        }) = done.result
        else {
            panic!("source trap required")
        };
        assert_eq!(actual, category);
        assert!(done.cleanup_trace.is_empty());
        if input.len() < 3 && f.contains("0u32..3u32") || input.is_empty() {
            assert_eq!(&s[span.start..span.end], "a[i]");
        }
    }
    for (lo, hi, n, index, input, expected) in [
        (0, 3, 1, "0u32*i", vec![7, 11, 19], vec![19]),
        (1, 3, 6, "2u32*i", vec![7, 11, 19], vec![0, 0, 11, 0, 19, 0]),
        (4, 2, 2, "4294967295u32*i+4294967295u32", vec![], vec![0, 0]),
        (0, 3, 3, "i%2u32", vec![7, 11, 19], vec![19, 11, 0]),
    ] {
        assert_eq!(
            prefixes(
                &source(&leaf(lo, hi, n, index), &input, &expected),
                Budget::default(),
                false
            )
            .result,
            Ok(eval::Value::I32(7))
        );
    }
    let f = leaf(0, 3, 6, "2u32*i");
    let s = source(&f, &[7, 11, 19], &[7, 0, 11, 0, 19, 0]);
    for budget in [
        Budget {
            bytes: 0,
            visits: 65_536,
        },
        Budget {
            bytes: 163_840,
            visits: 0,
        },
    ] {
        assert_eq!(prefixes(&s, budget, false).result, Ok(eval::Value::I32(7)));
    }
}
#[test]
fn issue_m4_05_gather_scatter_frontier_scatter_runtime_source_limits_and_errors() {
    let cap = invar::array_limits::MAX_LIVE_ELEMENTS;
    for (f, body, every_prefix) in [
        (
            leaf(0, 3, 6, "2u32*i"),
            "let a = filled_u32(16777217u32,7u32); let out=scatter(a); return 7;".to_owned(),
            true,
        ),
        (
            leaf(0, 3, 6, "2u32*i"),
            format!(
                "{} let a: Array<u32> = [7u32,11u32,19u32]; let out=scatter(a); return 7;",
                language_limits::padding(cap - 2)
            ),
            false,
        ),
        (
            leaf(0, 3, 6, "2u32*i"),
            format!(
                "{} let a: Array<u32> = [7u32,11u32,19u32]; let out=scatter(a); return 7;",
                language_limits::padding(cap - 3)
            ),
            false,
        ),
        (
            leaf(0, 0, invar::array_limits::MAX_ARRAY_LEN as u32 + 1, "i"),
            "let a: Array<u32> =[]; let out=scatter(a); return 7;".to_owned(),
            true,
        ),
    ] {
        let s = format!("{f} fn main() -> i32 {{ {body} }}");
        assert!(matches!(
            prefixes_at(&s, Budget::default(), false, every_prefix).result,
            Err(eval::EvalError::Tool {
                code: "ARRAY_SIZE_LIMIT",
                ..
            })
        ));
    }
    for (f, code) in [
        (leaf(0, 3, 6, "-1i32"), "TYPE_MISMATCH"),
        (
            leaf(0, 3, 6, "2u32*i").replace("var out = filled_u32(6u32,0u32);", "var out = a;"),
            "OWN_MOVE_REQUIRED",
        ),
    ] {
        let s = source(&f, &[7, 11, 19], &[]);
        let Err(invar::scatter_proof::Error::Source(error)) =
            host::run_source(s.as_bytes(), 1000, Budget::default())
        else {
            panic!("exact source error required")
        };
        assert_eq!(error.code, code);
    }
}
#[test]
fn issue_m4_05_gather_scatter_frontier_scatter_runtime_live_calls_and_dynamic_indices_fallback() {
    let f = leaf(0, 3, 6, "2u32*i");
    let s = format!(
        "{f} fn main() -> i32 {{ var a: Array<u32> =[7u32,11u32,19u32]; var first=scatter(a); a[0u32]=41u32; let second=scatter(a); first[2u32]=99u32; if first[0u32]!=7u32 or second[0u32]!=41u32 or second[2u32]!=11u32 or a[1u32]!=11u32 {{ return 0; }} return 7; }}"
    );
    assert_eq!(
        prefixes(&s, Budget::default(), true).result,
        Ok(eval::Value::I32(7))
    );
    let full = host::run_source(s.as_bytes(), 1000, Budget::default()).unwrap();
    assert_eq!(full.completed, 2);
    for body in [
        "let a: Array<u32> =[]; if false {let unused=scatter(a);} return 7;",
        "let a: Array<u32> =[7u32]; let bad=a[1u32]; let out=scatter(a); return 7;",
    ] {
        let s = format!("{f} fn main() -> i32 {{ {body} }}");
        prefixes(&s, Budget::default(), false);
    }
    let dynamic = "fn scatter(a: read Array<u32>, ix: read Array<u32>) -> Array<u32> { var out=filled_u32(2u32,0u32); for i in 0u32..2u32 {out[ix[i]]=a[i];} return out; } fn main() -> i32 {let a: Array<u32> =[7u32,11u32]; var ix: Array<u32> =[0u32,0u32]; let first=scatter(a,ix); ix[1u32]=1u32; let second=scatter(a,ix); if first[0u32]!=11u32 or first[1u32]!=0u32 or second[0u32]!=7u32 or second[1u32]!=11u32 {return 0;} return 7;}";
    assert_eq!(
        prefixes(dynamic, Budget::default(), false).result,
        Ok(eval::Value::I32(7))
    );
}
#[test]
fn issue_m4_05_gather_scatter_frontier_scatter_runtime_maximum_and_over_limit_shapes() {
    for (h, n, index) in [(4096, 4096, "i"), (2048, 4096, "2u32*i")] {
        let input: Vec<_> = (0..4096u32).map(|k| 17 * k + 3).collect();
        let expected: Vec<_> = (0..n)
            .map(|k| {
                if h == 4096 {
                    17 * k + 3
                } else if k % 2 == 0 {
                    17 * (k / 2) + 3
                } else {
                    0
                }
            })
            .collect();
        let s = source(&leaf(0, h, n, index), &input, &expected);
        let checked = check::check(front::parse(s.as_bytes()).unwrap(), true).unwrap();
        let complete = eval::observe(&checked, 1_000_000);
        let m = mir::lower(&checked).unwrap();
        let net = general_net::Net::lower(&checked, general_net::Limits::default()).unwrap();
        let proof = net.verify().unwrap();
        for fuel in [
            complete.steps_used - 1,
            complete.steps_used,
            complete.steps_used + 1,
        ] {
            let reference = eval::observe(&checked, fuel);
            let run = host::run_source(s.as_bytes(), fuel, Budget::default()).unwrap();
            assert_eq!(run.begun, 1);
            assert_eq!(run.completed, 1);
            compare(&run.observation, &reference);
            compare(&mir_eval::observe(&m, fuel), &reference);
            let oracle = general_net::oracle::observe(
                &net,
                &proof,
                general_net::oracle::Limits {
                    fuel,
                    ..general_net::oracle::Limits::default()
                },
            )
            .unwrap();
            compare(&oracle.semantic, &reference);
        }
        assert_eq!(complete.result, Ok(eval::Value::I32(7)));
        println!("scatter runtime maximum observations per engine: 3; shape: {h}/{n}");
    }
    let f = leaf(0, 1, 1, "i");
    let s = format!(
        "{f} fn main() -> i32 {{let a=filled_u32(4097u32,7u32);let out=scatter(a);if out[0u32]!=7u32 {{return 0;}}return 7;}}"
    );
    // The 4096-element cap is gone (#57): the 4097-element input is admitted and publishes.
    assert_eq!(
        prefixes(&s, Budget::default(), true).result,
        Ok(eval::Value::I32(7))
    );
    let f = leaf(0, 0, 4097, "i");
    let s = format!(
        "{f} fn main() -> i32 {{let a: Array<u32> =[];let out=scatter(a);if len(out)!=4097u32 {{return 0;}}return 7;}}"
    );
    assert_eq!(
        prefixes(&s, Budget::default(), false).result,
        Ok(eval::Value::I32(7))
    );
}
