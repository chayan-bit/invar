#![forbid(unsafe_code)]

// Parent allocation is required before exporting this isolated source module.
use indexed_source::{Access, Contract, Error, Family};
use invar::budget::{BudgetExceeded, ByteBudget, Resource};
use invar::indexed_source;
use invar::{check, eval, front, general_net, mir, mir_eval};

const GATHER: &str = "fn gather(a: read Array<u32>, ix: read Array<u32>) -> Array<u32> { var out = filled_u32(len(ix),0u32); for i in 0u32..len(ix) { out[i] = a[ix[i]]; } return out; }";
const SCATTER: &str = "fn scatter(a: read Array<u32>, ix: read Array<u32>, n: u32) -> Array<u32> { var out = filled_u32(n,0u32); for i in 0u32..len(ix) { out[ix[i]] = a[i]; } return out; }";

fn checked(s: &str) -> check::CheckedProgram {
    check::check(front::parse(s.as_bytes()).unwrap(), false).unwrap()
}
fn contract(s: &str, entry: &str) -> Contract {
    Contract::from_checked(s.as_bytes(), &checked(s), entry).unwrap()
}
fn array(a: &[u32]) -> String {
    format!(
        "[{}]",
        a.iter()
            .map(|v| format!("{v}u32"))
            .collect::<Vec<_>>()
            .join(",")
    )
}
fn main_source(
    s: &str,
    values: &[u32],
    indices: &[u32],
    n: Option<u32>,
    expected: &[u32],
) -> String {
    let call = n.map_or_else(
        || "gather(a,ix)".to_owned(),
        |n| format!("scatter(a,ix,{n}u32)"),
    );
    format!(
        "{s} fn main() -> i32 {{ let a: Array<u32> = {}; let ix: Array<u32> = {}; let original: Array<u32> = {}; let wanted: Array<u32> = {}; let result = {call}; if len(result) != len(wanted) {{ return 0; }} for k in 0u32..len(result) {{ if result[k] != wanted[k] {{ return 0; }} }} for j in 0u32..len(a) {{ if a[j] != original[j] {{ return 0; }} }} return 7; }}",
        array(values),
        array(indices),
        array(values),
        array(expected)
    )
}

// Independent mathematical relations: gather uses positions; scatter searches
// each output position rather than replaying the candidate's write algorithm.
fn gather_reference(values: &[u32], indices: &[u32]) -> Vec<u32> {
    indices.iter().map(|&i| values[i as usize]).collect()
}
fn scatter_reference(values: &[u32], indices: &[u32], n: u32) -> Vec<u32> {
    (0..n)
        .map(|out| {
            indices
                .iter()
                .position(|&i| i == out)
                .map_or(0, |i| values[i])
        })
        .collect()
}

fn prefixes(source: &str, entry: &str, complete_ok: bool) -> u64 {
    let p = checked(source);
    let c = Contract::from_checked(source.as_bytes(), &p, entry).unwrap();
    let m = c.verified_mir().unwrap();
    mir::verify(&m).unwrap();
    assert_eq!(m.contract_digest(), c.source_contract());
    observe_prefixes(&p, &m, entry, complete_ok)
}

fn observe_prefixes(
    p: &check::CheckedProgram,
    m: &mir::Program,
    entry: &str,
    complete_ok: bool,
) -> u64 {
    let net = general_net::Net::encode(m, general_net::Limits::default()).unwrap();
    let proof = net.verify().unwrap();
    let complete = eval::observe(p, 10_000);
    if complete_ok {
        assert_eq!(complete.result, Ok(eval::Value::I32(7)));
        assert!(!complete.cleanup_trace.is_empty());
    } else {
        assert!(matches!(complete.result, Err(eval::EvalError::Trap { .. })));
        assert!(complete.cleanup_trace.is_empty());
    }
    for fuel in 0..=complete.steps_used + 1 {
        let reference = eval::observe(p, fuel);
        let actual = mir_eval::observe(m, fuel);
        assert_eq!(actual.result, reference.result, "entry={entry} fuel={fuel}");
        assert_eq!(actual.steps_used, reference.steps_used);
        assert_eq!(actual.cleanup_trace, reference.cleanup_trace);
        let interaction = general_net::oracle::observe(
            &net,
            &proof,
            general_net::oracle::Limits {
                fuel,
                ..general_net::oracle::Limits::default()
            },
        )
        .unwrap();
        assert_eq!(
            interaction.semantic.result, reference.result,
            "oracle entry={entry} fuel={fuel}"
        );
        assert_eq!(interaction.semantic.steps_used, reference.steps_used);
        assert_eq!(interaction.semantic.cleanup_trace, reference.cleanup_trace);
    }
    complete.steps_used + 2
}

#[test]
fn issue_m4_05_gather_scatter_frontier_source_bound_identity_and_mutations() {
    for (s, entry, family) in [
        (GATHER, "gather", Family::Gather),
        (SCATTER, "scatter", Family::InjectiveScatter),
    ] {
        let c = contract(s, entry);
        assert_eq!(c.family(), family);
        assert!(s[c.loop_span().start..c.loop_span().end].starts_with("for i"));
        assert_ne!(c.source_contract(), [0; 32]);
        let renamed = s
            .replace(entry, "indexed")
            .replace("out", "result")
            .replace("ix", "positions");
        let changed = contract(&renamed, "indexed");
        assert_ne!(c.source_contract(), changed.source_contract());
        let grouped = s
            .replace("0u32..len(ix)", "(0u32)..(len(ix))")
            .replace("a[ix[i]]", "(a[(ix[(i)])])")
            .replace("out[ix[i]]", "out[(ix[(i)])]")
            .replace("a[i]", "(a[(i)])")
            .replace("out[i]", "out[(i)]")
            .replace("return out;", "return move out;");
        assert_eq!(contract(&grouped, entry).family(), family);
        let mut stale = checked(s);
        stale.expression_types.clear();
        assert!(matches!(
            Contract::from_checked(s.as_bytes(), &stale, entry),
            Err(Error::CheckedMismatch)
        ));
        let mut stale = checked(s);
        stale
            .function_expression_types
            .get_mut(entry)
            .unwrap()
            .pop();
        assert!(matches!(
            Contract::from_checked(s.as_bytes(), &stale, entry),
            Err(Error::CheckedMismatch)
        ));
        assert!(matches!(
            Contract::from_checked(renamed.as_bytes(), &checked(s), "indexed"),
            Err(Error::CheckedMismatch)
        ));
        assert!(matches!(
            Contract::from_checked(s.as_bytes(), &checked(s), "missing"),
            Err(Error::Unrecognized)
        ));
    }
    for s in [
        GATHER.replace("a[ix[i]]", "a[ix[i]] + 1u32"),
        GATHER.replace("a[ix[i]]", "a[i]"),
        GATHER.replace("0u32..len(ix)", "1u32..len(ix)"),
        GATHER.replace("len(ix),0u32", "len(a),0u32"),
        GATHER.replace("return out;", "return filled_u32(0u32,0u32);"),
        GATHER.replace("out[i] = a[ix[i]];", "out[i] = a[ix[i]]; out[i] = 1u32;"),
        GATHER.replace("out[i] = a[ix[i]]", "out[i] = a[ix[len(ix) - i - 1u32]]"),
        GATHER.replace("a: read", "a: mut"),
    ] {
        assert!(
            matches!(
                Contract::from_checked(s.as_bytes(), &checked(&s), "gather"),
                Err(Error::Unrecognized)
            ),
            "{s}"
        );
    }
    let conflict = SCATTER.replace("a[i];", "wrapping_add(out[ix[i]],a[i]);");
    assert!(matches!(
        Contract::from_checked(conflict.as_bytes(), &checked(&conflict), "scatter"),
        Err(Error::Unrecognized)
    ));
}

#[test]
fn issue_m4_05_gather_scatter_frontier_cpu_candidates_independent_outputs_and_accesses() {
    let gather = contract(GATHER, "gather");
    let scatter = contract(SCATTER, "scatter");
    let mut count = 0;
    for (values, indices, n) in [
        (vec![], vec![], 0),
        (vec![], vec![], 3),
        (vec![u32::MAX], vec![0], 1),
        (vec![10, 20, 30, 40], vec![0, 1, 2, 3], 4),
        (vec![0, u32::MAX, 0x80000000, 91, 7], vec![4, 1, 3, 0, 2], 7),
    ] {
        let got = gather.candidate(&values, &indices, None).unwrap();
        let expected = gather_reference(&values, &indices);
        assert_eq!(got.output(), expected);
        assert_eq!(
            got.accesses(),
            indices
                .iter()
                .enumerate()
                .map(|(i, &j)| Access {
                    iteration: i,
                    read_index: j as usize,
                    write_index: i
                })
                .collect::<Vec<_>>()
        );
        count += prefixes(
            &main_source(GATHER, &values, &indices, None, &expected),
            "gather",
            true,
        );
        let got = scatter.candidate(&values, &indices, Some(n)).unwrap();
        let expected = scatter_reference(&values, &indices, n);
        assert_eq!(got.output(), expected);
        assert_eq!(
            got.accesses(),
            indices
                .iter()
                .enumerate()
                .map(|(i, &j)| Access {
                    iteration: i,
                    read_index: i,
                    write_index: j as usize
                })
                .collect::<Vec<_>>()
        );
        count += prefixes(
            &main_source(SCATTER, &values, &indices, Some(n), &expected),
            "scatter",
            true,
        );
    }
    // Gather can repeat reads and both immutable parameters can name one owner.
    assert_eq!(
        gather
            .candidate(&[10, 20, 30], &[2, 2, 0], None)
            .unwrap()
            .output(),
        &[30, 30, 10]
    );
    let source = format!(
        "{GATHER} fn main() -> i32 {{ let a: Array<u32> = [1u32,0u32]; let result = gather(a,a); if result[0u32] == 0u32 and result[1u32] == 1u32 {{ return 7; }} return 0; }}"
    );
    count += prefixes(&source, "gather", true);
    let shared = [1, 0];
    assert_eq!(
        gather.candidate(&shared, &shared, None).unwrap().output(),
        &[0, 1]
    );
    let mut seed = 0x4d340005u32;
    for n in [3u32, 7, 11, 15] {
        let mut indices: Vec<_> = (0..n).collect();
        for i in (1..n as usize).rev() {
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            indices.swap(i, seed as usize % (i + 1));
        }
        let values: Vec<_> = (0..n).map(|i| i.wrapping_mul(0x9e3779b9)).collect();
        let expected = gather_reference(&values, &indices);
        assert_eq!(
            gather.candidate(&values, &indices, None).unwrap().output(),
            expected
        );
        count += prefixes(
            &main_source(GATHER, &values, &indices, None, &expected),
            "gather",
            true,
        );
        let expected = scatter_reference(&values, &indices, n + 3);
        assert_eq!(
            scatter
                .candidate(&values, &indices, Some(n + 3))
                .unwrap()
                .output(),
            expected
        );
        count += prefixes(
            &main_source(SCATTER, &values, &indices, Some(n + 3), &expected),
            "scatter",
            true,
        );
    }
    println!("successful source/MIR/net fuel-prefix observations per engine: {count}");
}

#[test]
fn issue_m4_05_gather_scatter_frontier_refused_order_and_trapping_rhs_keep_source() {
    let reverse = GATHER.replace("a[ix[i]]", "a[ix[len(ix) - i - 1u32]]");
    let overflow = GATHER.replace("a[ix[i]]", "a[ix[i]] + 1u32");
    let cases = [
        (
            main_source(&reverse, &[10, 20, 30], &[0, 1, 2], None, &[30, 20, 10]),
            true,
        ),
        (main_source(&overflow, &[u32::MAX], &[0], None, &[]), false),
    ];
    let mut count = 0;
    for (source, success) in cases {
        let p = checked(&source);
        assert!(matches!(
            Contract::from_checked(source.as_bytes(), &p, "gather"),
            Err(Error::Unrecognized)
        ));
        let net = general_net::Net::lower(&p, general_net::Limits::default()).unwrap();
        let proof = net.verify().unwrap();
        let m = net.compiled_mir(&proof).unwrap();
        count += observe_prefixes(&p, &m, "refused-gather", success);
        if !success {
            assert!(matches!(
                eval::observe(&p, 10_000).result,
                Err(eval::EvalError::Trap {
                    category: eval::TrapCategory::IntegerOverflow,
                    ..
                })
            ));
        }
    }
    println!("refused-order/RHS source/MIR/net fuel-prefix observations per engine: {count}");
}

#[test]
fn issue_m4_05_gather_scatter_frontier_guards_duplicates_bounds_budget_and_fallback() {
    let gather = contract(GATHER, "gather");
    let scatter = contract(SCATTER, "scatter");
    assert_eq!(
        gather.candidate(&[], &[], Some(0)).unwrap_err(),
        Error::Shape
    );
    assert_eq!(scatter.candidate(&[], &[], None).unwrap_err(), Error::Shape);
    assert_eq!(
        gather.candidate(&[], &[0], None).unwrap_err(),
        Error::Bounds { iteration: 0 }
    );
    assert_eq!(
        gather.candidate(&[1], &[u32::MAX], None).unwrap_err(),
        Error::Bounds { iteration: 0 }
    );
    assert_eq!(
        scatter.candidate(&[1, 2], &[0, 0], Some(1)).unwrap_err(),
        Error::Duplicate {
            iteration: 1,
            index: 0
        }
    );
    assert_eq!(
        scatter.candidate(&[1], &[0, 1], Some(2)).unwrap_err(),
        Error::Bounds { iteration: 1 }
    );
    assert_eq!(
        scatter.candidate(&[1], &[1], Some(1)).unwrap_err(),
        Error::Bounds { iteration: 0 }
    );
    // The old 4096-element cap is gone: only bytes refuse. A one-MiB host budget refuses the
    // regions below with the typed refusal; the default budget admits the ones it can hold.
    let small = ByteBudget::new(1 << 20, 0, 0);
    for n in [4097usize, 1_000_000] {
        assert!(gather.candidate(&vec![0; n], &[], None).is_ok());
        assert!(
            gather
                .candidate(&[0], &vec![0; n], None)
                .is_ok_and(|c| c.output().len() == n)
        );
        assert!(scatter.candidate(&[], &[], Some(n as u32)).is_ok());
    }
    for budget in [small, ByteBudget::ZERO] {
        assert!(matches!(
            gather.candidate_within(&[0], &vec![0; 1 << 18], None, &budget),
            Err(Error::Exceeded(_))
        ));
    }
    let all: Vec<_> = (0..4096u32).rev().collect();
    assert_eq!(
        scatter
            .candidate(&all, &all, Some(4096))
            .unwrap()
            .output()
            .len(),
        4096
    );
    assert!(matches!(
        scatter.candidate(&[], &[], Some(u32::MAX)),
        Err(Error::Exceeded(BudgetExceeded {
            resource: Resource::Host,
            ..
        }))
    ));
    // Duplicate sequential overwrite is lawful source behavior, but this slice
    // refuses concurrent admission instead of inventing a conflict law.
    let duplicate = main_source(SCATTER, &[41, 99], &[0, 0], Some(2), &[99, 0]);
    let mut count = prefixes(&duplicate, "scatter", true);
    let over_candidate_budget = format!(
        "{SCATTER} fn main() -> i32 {{ let a: Array<u32> = []; let ix: Array<u32> = []; let result = scatter(a,ix,4097u32); if len(result) != 4097u32 {{ return 0; }} return 7; }}"
    );
    count += prefixes(&over_candidate_budget, "scatter", true);
    for (s, values, indices, n, entry) in [
        (GATHER, vec![7], vec![1], None, "gather"),
        (GATHER, vec![7], vec![0, u32::MAX], None, "gather"),
        (SCATTER, vec![], vec![u32::MAX], Some(0), "scatter"),
        (SCATTER, vec![7], vec![0, 1], Some(2), "scatter"),
        (SCATTER, vec![7, 8], vec![0, 2], Some(2), "scatter"),
    ] {
        let source = main_source(s, &values, &indices, n, &[]);
        count += prefixes(&source, entry, false);
        let p = checked(&source);
        let outcome = eval::observe(&p, 10_000);
        let Err(eval::EvalError::Trap {
            category: eval::TrapCategory::IndexOutOfBounds,
            span,
        }) = outcome.result
        else {
            panic!("expected original index trap")
        };
        // Scatter with both RHS and destination invalid must trap at a[i] first.
        let expected = if entry == "gather" {
            "a[ix[i]]"
        } else if values.len() < indices.len() {
            "a[i]"
        } else {
            "out[ix[i]]"
        };
        assert_eq!(&source[span.start..span.end], expected);
    }
    println!("guard/fallback source/MIR/net fuel-prefix observations per engine: {count}");
}

#[test]
fn issue_m4_05_gather_scatter_frontier_source_resource_limit_fuel_and_cleanup_prefixes() {
    let mut count = 0;
    for n in [invar::array_limits::MAX_ARRAY_LEN as u32 + 1, u32::MAX] {
        let source = format!(
            "{SCATTER} fn main() -> i32 {{ let a: Array<u32> = []; let ix: Array<u32> = []; let result = scatter(a,ix,{n}u32); return 7; }}"
        );
        let p = checked(&source);
        let c = Contract::from_checked(source.as_bytes(), &p, "scatter").unwrap();
        assert!(matches!(
            c.candidate_within(&[], &[], Some(n), &ByteBudget::new(1 << 20, 0, 0)),
            Err(Error::Exceeded(_))
        ));
        let m = c.verified_mir().unwrap();
        let net = general_net::Net::encode(&m, general_net::Limits::default()).unwrap();
        let proof = net.verify().unwrap();
        let complete = eval::observe(&p, 1000);
        assert!(matches!(
            complete.result,
            Err(eval::EvalError::Tool {
                code: "ARRAY_SIZE_LIMIT",
                ..
            })
        ));
        assert!(complete.cleanup_trace.is_empty());
        for fuel in 0..=complete.steps_used + 1 {
            let reference = eval::observe(&p, fuel);
            let actual = mir_eval::observe(&m, fuel);
            let interaction = general_net::oracle::observe(
                &net,
                &proof,
                general_net::oracle::Limits {
                    fuel,
                    ..general_net::oracle::Limits::default()
                },
            )
            .unwrap();
            for observed in [&actual, &interaction.semantic] {
                assert_eq!(
                    observed.result, reference.result,
                    "resource n={n} fuel={fuel}"
                );
                assert_eq!(observed.steps_used, reference.steps_used);
                assert_eq!(observed.cleanup_trace, reference.cleanup_trace);
            }
            count += 1;
        }
    }
    println!("source-resource-limit source/MIR/net fuel-prefix observations per engine: {count}");
}

#[test]
fn issue_m4_05_gather_scatter_frontier_source_invalid_alias_and_negative_indices() {
    for (s, code) in [
        (GATHER.replace("ix[i]", "-1"), "TYPE_MISMATCH"),
        (
            GATHER.replace("var out = filled_u32(len(ix),0u32);", "var out = a;"),
            "OWN_MOVE_REQUIRED",
        ),
        (
            "fn invalid(a: mut Array<u32>, ix: read Array<u32>) -> Unit { a[ix[0u32]] = 1u32; } fn main() -> i32 { var a: Array<u32> = [0u32]; invalid(a,a); return 7; }".to_owned(),
            "OWN_BORROW_CONFLICT",
        ),
    ] {
        let p = front::parse(s.as_bytes()).unwrap();
        assert_eq!(check::check(p, false).unwrap_err().code, code);
    }
}
