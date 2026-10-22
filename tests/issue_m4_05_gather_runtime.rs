#![forbid(unsafe_code)]
#[path = "support/language_limits.rs"]
mod language_limits;
use invar::{
    check, eval, front, general_net,
    indexed_source::host::{self, Budget},
    mir, mir_eval,
};
const GATHER: &str = "fn gather(a: read Array<u32>, ix: read Array<u32>) -> Array<u32> { var out = filled_u32(len(ix),0u32); for i in 0u32..len(ix) { out[i] = a[ix[i]]; } return out; }";
fn array(values: &[u32]) -> String {
    format!(
        "[{}]",
        values
            .iter()
            .map(|n| format!("{n}u32"))
            .collect::<Vec<_>>()
            .join(",")
    )
}
fn source(leaf: &str, values: &[u32], indices: &[u32], expected: &[u32], alias: bool) -> String {
    let inputs = if alias {
        format!(
            "let a: Array<u32> = {}; let result = gather(a,a);",
            array(values)
        )
    } else {
        format!(
            "let a: Array<u32> = {}; let ix: Array<u32> = {}; let result = gather(a,ix);",
            array(values),
            array(indices)
        )
    };
    format!(
        "{leaf} fn main() -> i32 {{ {inputs} let wanted: Array<u32> = {}; if len(result) != len(wanted) {{ return 0; }} for k in 0u32..len(result) {{ if result[k] != wanted[k] {{ return 0; }} }} return 7; }}",
        array(expected)
    )
}
fn compare(actual: &eval::Observation, expected: &eval::Observation) {
    assert_eq!(actual.result, expected.result);
    assert_eq!(actual.steps_used, expected.steps_used);
    assert_eq!(actual.cleanup_trace, expected.cleanup_trace);
}
fn prefixes(source: &str, budget: Budget) -> (eval::Observation, usize) {
    prefixes_at(source, budget, true)
}
fn prefixes_at(source: &str, budget: Budget, every_prefix: bool) -> (eval::Observation, usize) {
    let checked = check::check(front::parse(source.as_bytes()).unwrap(), true).unwrap();
    let compiled = mir::lower(&checked).unwrap();
    let net = general_net::Net::lower(&checked, general_net::Limits::default()).unwrap();
    let proof = net.verify().unwrap();
    let complete = eval::observe(&checked, 1_000_000);
    let mut replacements = 0;
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
        let expected = eval::observe(&checked, fuel);
        let run = host::run_source(source.as_bytes(), fuel, budget).unwrap();
        compare(&run.observation, &expected);
        compare(&mir_eval::observe(&compiled, fuel), &expected);
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
        compare(&oracle.semantic, &expected);
        assert!(run.pending.is_none());
        assert_eq!(run.begun, run.completed);
        replacements += run.completed;
    }
    println!(
        "runtime fuel prefixes: {}; completed loop publications across prefixes: {replacements}",
        prefix_count
    );
    (complete, replacements)
}

#[test]
fn issue_m4_05_gather_scatter_frontier_runtime_outputs_alias_and_exact_prefixes() {
    for (values, indices, expected, alias) in [
        (&[][..], &[][..], &[][..], false),
        (&[41][..], &[0][..], &[41][..], false),
        (
            &[7, 11, 19][..],
            &[2, 0, 2, 1][..],
            &[19, 7, 19, 11][..],
            false,
        ),
        (&[2, 0, 2][..], &[2, 0, 2][..], &[2, 2, 2][..], true),
    ] {
        let s = source(GATHER, values, indices, expected, alias);
        let (done, pubs) = prefixes(&s, Budget::default());
        assert_eq!(done.result, Ok(eval::Value::I32(7)));
        assert_eq!(pubs > 0, !indices.is_empty());
    }
    let grouped = GATHER
        .replace("0u32..len(ix)", "(0u32)..(len(ix))")
        .replace("out[i] = a[ix[i]]", "out[(i)] = (a[(ix[(i)])])");
    for leaf in [
        grouped,
        GATHER.replace("return out;", "return move out;"),
        GATHER.replace("return out;", "return (move out);"),
    ] {
        assert!(
            prefixes(
                &source(&leaf, &[7, 11, 19], &[2, 0, 2], &[19, 7, 19], false),
                Budget::default()
            )
            .1 > 0
        );
    }
}

#[test]
fn issue_m4_05_gather_scatter_frontier_runtime_refusal_trap_spans_and_limits() {
    for (values, indices) in [
        (&[][..], &[0][..]),
        (&[7][..], &[1][..]),
        (&[7][..], &[0, u32::MAX][..]),
    ] {
        let s = source(GATHER, values, indices, &[], false);
        let (done, pubs) = prefixes(&s, Budget::default());
        assert_eq!(pubs, 0);
        let Err(eval::EvalError::Trap {
            category: eval::TrapCategory::IndexOutOfBounds,
            span,
        }) = done.result
        else {
            panic!("original trap required")
        };
        assert_eq!(&s[span.start..span.end], "a[ix[i]]");
        assert!(done.cleanup_trace.is_empty());
        let run = host::run_source(s.as_bytes(), 1000, Budget::default()).unwrap();
        assert_eq!(run.begun, 0);
    }
    let s = source(GATHER, &[7], &[0], &[7], false);
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
        let (done, pubs) = prefixes(&s, budget);
        assert_eq!(pubs, 0);
        assert_eq!(done.result, Ok(eval::Value::I32(7)));
    }
    let cap = invar::array_limits::MAX_LIVE_ELEMENTS;
    for (body, every_prefix) in [
        (
            "let a = filled_u32(16777217u32,7u32); return 7;".to_owned(),
            true,
        ),
        // Input admission fails before the call; the next case fails at output allocation.
        (
            format!(
                "{} let a: Array<u32> = [7u32]; let ix: Array<u32> = [0u32]; let result = gather(a,ix); return 7;",
                language_limits::padding(cap - 1)
            ),
            false,
        ),
        (
            format!(
                "{} let a: Array<u32> = [7u32]; let ix: Array<u32> = [0u32]; let result = gather(a,ix); return 7;",
                language_limits::padding(cap - 2)
            ),
            false,
        ),
    ] {
        let s = format!("{GATHER} fn main() -> i32 {{ {body} }}");
        let (done, pubs) = prefixes_at(&s, Budget::default(), every_prefix);
        assert_eq!(pubs, 0);
        assert!(matches!(
            done.result,
            Err(eval::EvalError::Tool {
                code: "ARRAY_SIZE_LIMIT",
                ..
            })
        ));
    }
}

#[test]
fn issue_m4_05_gather_scatter_frontier_runtime_repeated_calls_reachability_and_shape() {
    let s = format!(
        "{GATHER} fn main() -> i32 {{ var a: Array<u32> = [7u32,11u32]; let ix: Array<u32> = [1u32,0u32]; let first = gather(a,ix); a[0u32] = 19u32; let second = gather(a,ix); if first[1u32] != 7u32 {{ return 0; }} if second[1u32] != 19u32 {{ return 0; }} return 7; }}"
    );
    let run = host::run_source(s.as_bytes(), 1000, Budget::default()).unwrap();
    assert_eq!(run.completed, 2);
    assert_eq!(run.begun, 2);
    assert_eq!(run.observation.result, Ok(eval::Value::I32(7)));
    assert!(prefixes(&s, Budget::default()).1 > 0);
    let unreachable = format!(
        "{GATHER} fn main() -> i32 {{ let a: Array<u32> = []; let ix: Array<u32> = [99u32]; if false {{ let result = gather(a,ix); }} return 7; }}"
    );
    assert_eq!(prefixes(&unreachable, Budget::default()).1, 0);
    let earlier = format!(
        "{GATHER} fn main() -> i32 {{ let a: Array<u32> = [7u32]; let bad = a[1u32]; let ix: Array<u32> = [0u32]; let out = gather(a,ix); return 7; }}"
    );
    let (done, pubs) = prefixes(&earlier, Budget::default());
    assert_eq!(pubs, 0);
    let Err(eval::EvalError::Trap { span, .. }) = done.result else {
        panic!()
    };
    assert_eq!(&earlier[span.start..span.end], "a[1u32]");
    let changed = GATHER.replace(
        "out[i] = a[ix[i]];",
        "out[i] = a[ix[i]]; out[i] = a[ix[i]];",
    );
    assert_eq!(
        prefixes(
            &source(&changed, &[7], &[0], &[7], false),
            Budget::default()
        )
        .1,
        0
    );
    for (invalid, code) in [
        (GATHER.replace("a[ix[i]]", "a[-1]"), "TYPE_MISMATCH"),
        (
            GATHER.replace("var out = filled_u32(len(ix),0u32);", "var out = a;"),
            "OWN_MOVE_REQUIRED",
        ),
    ] {
        let invalid = source(&invalid, &[7], &[0], &[7], false);
        let Err(invar::indexed_source::Error::Source(error)) =
            host::run_source(invalid.as_bytes(), 1000, Budget::default())
        else {
            panic!("source-invalid input must retain its checker error")
        };
        assert_eq!(error.code, code);
    }
}

#[test]
fn issue_m4_05_gather_scatter_frontier_runtime_maximum_and_over_limit_arrays() {
    let values: Vec<_> = (0..4096u32).map(|i| 17 * i + 3).collect();
    let indices: Vec<_> = (0..4096u32).rev().collect();
    let expected: Vec<_> = (0..4096u32).map(|i| 3 + 17 * (4095 - i)).collect();
    let s = source(GATHER, &values, &indices, &expected, false);
    let checked = check::check(front::parse(s.as_bytes()).unwrap(), true).unwrap();
    let run = host::run_source(s.as_bytes(), 1_000_000, Budget::default()).unwrap();
    assert_eq!(run.completed, 1);
    assert_eq!(run.begun, 1);
    compare(&run.observation, &eval::observe(&checked, 1_000_000));
    assert_eq!(run.observation.result, Ok(eval::Value::I32(7)));
    let over = format!(
        "{GATHER} fn main() -> i32 {{ let a = filled_u32(4097u32,7u32); let ix: Array<u32> = [0u32]; let out = gather(a,ix); if out[0u32] != 7u32 {{ return 0; }} return 7; }}"
    );
    // The 4096-element cap is gone (#278): 4097 values are admitted and publish; only the
    // staging byte limit refuses, here with 50_000 values (200 kB against the 160 KiB limit).
    assert!(prefixes_at(&over, Budget::default(), false).1 > 0);
    let heavy = over.replace("filled_u32(4097u32,7u32)", "filled_u32(50000u32,7u32)");
    assert_eq!(prefixes_at(&heavy, Budget::default(), false).1, 0);
}
