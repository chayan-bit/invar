#![forbid(unsafe_code)]

use invar::budget::ByteBudget;
use invar::histogram_source::{
    Access, Budget, Contract, Cost, Error, MAX_STAGING_BYTES, MAX_VISITS,
};
use invar::{check, eval, front, general_net, laws, mir, mir_eval};

const HIST: &str = "fn hist(a: read Array<u32>, ix: read Array<u32>, n: u32) -> Array<u32> { var out = filled_u32(n,0u32); for i in 0u32..len(ix) { out[ix[i]] = wrapping_add(out[ix[i]],a[i]); } return out; }";

fn checked(s: &str) -> check::CheckedProgram {
    check::check(front::parse(s.as_bytes()).unwrap(), false).unwrap()
}
fn contract(s: &str) -> Contract {
    Contract::from_checked(s.as_bytes(), &checked(s), "hist").unwrap()
}
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
fn source(s: &str, values: &[u32], indices: &[u32], n: u32, wanted: &[u32]) -> String {
    format!(
        "{s} fn main() -> i32 {{ let a: Array<u32> = {}; let ix: Array<u32> = {}; let retained_a: Array<u32> = {}; let retained_ix: Array<u32> = {}; let expected: Array<u32> = {}; let result = hist(a,ix,{n}u32); if len(result) != len(expected) {{ return 0; }} for j in 0u32..len(result) {{ if result[j] != expected[j] {{ return 0; }} }} for k in 0u32..len(a) {{ if a[k] != retained_a[k] {{ return 0; }} }} for q in 0u32..len(ix) {{ if ix[q] != retained_ix[q] {{ return 0; }} }} return 7; }}",
        array(values),
        array(indices),
        array(values),
        array(indices),
        array(wanted)
    )
}

// Independent mathematical bin sums in u64; the bounded domain is below 2^44.
// No candidate combine, grouping algorithm or source evaluator sets this oracle.
fn expected(values: &[u32], indices: &[u32], n: u32) -> Vec<u32> {
    (0..n)
        .map(|bin| {
            let total: u64 = indices
                .iter()
                .enumerate()
                .filter(|(_, index)| **index == bin)
                .map(|(i, _)| u64::from(values[i]))
                .sum();
            (total % (1u64 << 32)) as u32
        })
        .collect()
}

fn observations(p: &check::CheckedProgram, m: &mir::Program, is_trap: bool) -> u64 {
    let complete = eval::observe(p, 10_000);
    if is_trap {
        assert!(matches!(complete.result, Err(eval::EvalError::Trap { .. })));
        assert!(complete.cleanup_trace.is_empty());
    } else {
        assert_eq!(complete.result, Ok(eval::Value::I32(7)));
        assert!(!complete.cleanup_trace.is_empty());
    }
    let net = general_net::Net::encode(m, general_net::Limits::default()).unwrap();
    let proof = net.verify().unwrap();
    for fuel in 0..=complete.steps_used + 1 {
        let reference = eval::observe(p, fuel);
        let extracted = mir_eval::observe(m, fuel);
        let interaction = general_net::oracle::observe(
            &net,
            &proof,
            general_net::oracle::Limits {
                fuel,
                ..general_net::oracle::Limits::default()
            },
        )
        .unwrap();
        for observed in [&extracted, &interaction.semantic] {
            assert_eq!(observed.result, reference.result, "fuel={fuel}");
            assert_eq!(observed.steps_used, reference.steps_used);
            assert_eq!(observed.cleanup_trace, reference.cleanup_trace);
        }
    }
    complete.steps_used + 2
}
fn prefixes(s: &str, is_trap: bool) -> u64 {
    let p = checked(s);
    let c = Contract::from_checked(s.as_bytes(), &p, "hist").unwrap();
    let m = c.verified_mir().unwrap();
    mir::verify(&m).unwrap();
    assert_eq!(m.contract_digest(), c.source_contract());
    observations(&p, &m, is_trap)
}

/// Array size the maximum-arrays test exercises; no longer a refusal bound (#278 byte budgets).
const MAX_ARRAY: usize = 4096;

#[test]
fn issue_m4_05_gather_scatter_frontier_histogram_source_law_identity_and_mutations() {
    let c = contract(HIST);
    assert_eq!(c.law(), laws::Law::WrappingSum);
    assert_eq!(c.law().identity(), 0);
    assert_eq!(c.law_revision(), "sealed-wrapping-u32-laws-v1");
    assert!(HIST[c.loop_span().start..c.loop_span().end].starts_with("for i"));
    assert_ne!(c.source_contract(), [0; 32]);
    for changed in [
        HIST.replace("out", "buckets").replace("ix", "positions"),
        HIST.replace(
            "wrapping_add(out[ix[i]],a[i])",
            "wrapping_add(a[i],out[ix[i]])",
        ),
        HIST.replace("return out;", "return move out;"),
        HIST.replace("0u32..len(ix)", "(0u32)..(len(ix))")
            .replace("out[ix[i]]", "(out[(ix[(i)])])")
            .replace("(out[(ix[(i)])]) =", "out[(ix[(i)])] =")
            .replace("a[i]", "(a[(i)])"),
    ] {
        let changed = contract(&changed);
        assert_eq!(changed.law(), laws::Law::WrappingSum);
        assert_ne!(changed.source_contract(), c.source_contract());
        assert_eq!(
            changed
                .candidate(&[u32::MAX, 1], &[0, 0], 1)
                .unwrap()
                .output(),
            &[0]
        );
    }
    let mut p = checked(HIST);
    p.expression_types.clear();
    assert!(matches!(
        Contract::from_checked(HIST.as_bytes(), &p, "hist"),
        Err(Error::CheckedMismatch)
    ));
    let mut p = checked(HIST);
    p.function_expression_types.get_mut("hist").unwrap().pop();
    assert!(matches!(
        Contract::from_checked(HIST.as_bytes(), &p, "hist"),
        Err(Error::CheckedMismatch)
    ));
    let changed = HIST.replace("wrapping_add", "wrapping_mul");
    assert!(matches!(
        Contract::from_checked(changed.as_bytes(), &checked(HIST), "hist"),
        Err(Error::CheckedMismatch)
    ));
    let mut p = checked(HIST);
    p.program.functions[0].body.statements[1].span.end -= 1;
    assert!(matches!(
        Contract::from_checked(HIST.as_bytes(), &p, "hist"),
        Err(Error::CheckedMismatch)
    ));
    assert!(matches!(
        Contract::from_checked(HIST.as_bytes(), &checked(HIST), "missing"),
        Err(Error::Unrecognized)
    ));
}

#[test]
fn issue_m4_05_gather_scatter_frontier_histogram_duplicates_grouping_and_independent_outputs() {
    let swapped = HIST.replace(
        "wrapping_add(out[ix[i]],a[i])",
        "wrapping_add(a[i],out[ix[i]])",
    );
    let mut count = 0;
    for s in [HIST, swapped.as_str()] {
        let c = contract(s);
        for (values, indices, n) in [
            (vec![], vec![], 0),
            (vec![], vec![], 3),
            (vec![u32::MAX], vec![0], 1),
            (vec![1, 2, 3], vec![0, 1, 2], 3),
            (vec![u32::MAX, 1], vec![0, 0], 1),
            (vec![7, 11, 13, 17, 19], vec![2, 0, 2, 0, 2], 5),
            (
                vec![u32::MAX, u32::MAX, 3, 0x80000000, 0x80000000],
                vec![1, 1, 1, 0, 0],
                3,
            ),
            (vec![20, 40, 99], vec![1, 1], 2),
        ] {
            let want = expected(&values, &indices, n);
            let original_values = values.clone();
            let original_indices = indices.clone();
            let candidate = c.candidate(&values, &indices, n).unwrap();
            assert_eq!(candidate.output(), want);
            assert!(candidate.sort_visits() <= MAX_VISITS);
            let bounded = c
                .bounded_candidate(&values, &indices, n, Budget::default())
                .unwrap();
            assert_eq!(bounded.output(), want);
            assert_eq!(
                candidate.source_accesses(),
                indices
                    .iter()
                    .enumerate()
                    .map(|(i, &bin)| Access {
                        iteration: i,
                        bin: bin as usize
                    })
                    .collect::<Vec<_>>()
            );
            let grouped = candidate.grouped_accesses();
            assert!(grouped.windows(2).all(|pair| pair[0].bin <= pair[1].bin));
            let mut original_ordinals: Vec<_> = grouped.iter().map(|a| a.iteration).collect();
            original_ordinals.sort_unstable();
            assert_eq!(original_ordinals, (0..indices.len()).collect::<Vec<_>>());
            for access in grouped {
                assert_eq!(indices[access.iteration] as usize, access.bin);
            }
            assert_eq!(values, original_values);
            assert_eq!(indices, original_indices);
            count += prefixes(&source(s, &values, &indices, n, &want), false);
        }
    }
    let c = contract(HIST);
    let mut state = 0x4d350005u32;
    for n in [3usize, 7, 11, 15] {
        let mut weights = Vec::new();
        let mut indices = Vec::new();
        for _ in 0..n {
            state = state.wrapping_mul(1664525).wrapping_add(1013904223);
            indices.push(state % 3);
            state = state.wrapping_mul(1664525).wrapping_add(1013904223);
            weights.push(state);
        }
        let want = expected(&weights, &indices, 4);
        assert_eq!(c.candidate(&weights, &indices, 4).unwrap().output(), want);
        assert_eq!(
            c.bounded_candidate(&weights, &indices, 4, Budget::default())
                .unwrap()
                .output(),
            want
        );
        count += prefixes(&source(HIST, &weights, &indices, 4, &want), false);
        weights.reverse();
        indices.reverse();
        assert_eq!(c.candidate(&weights, &indices, 4).unwrap().output(), want);
        assert_eq!(
            c.bounded_candidate(&weights, &indices, 4, Budget::default())
                .unwrap()
                .output(),
            want
        );
    }
    println!("histogram successful source/MIR/net fuel-prefix observations per engine: {count}");
}

#[test]
fn issue_m4_05_gather_scatter_frontier_histogram_bounds_trap_order_and_budget_fallback() {
    let c = contract(HIST);
    for (values, indices, n, iteration) in [
        (vec![], vec![0], 1, 0),
        (vec![1], vec![1], 1, 0),
        (vec![1], vec![u32::MAX], 1, 0),
        (vec![1], vec![0, 1], 2, 1),
        (vec![1, 2], vec![0, 2], 2, 1),
    ] {
        assert_eq!(
            c.candidate(&values, &indices, n).unwrap_err(),
            Error::Bounds { iteration }
        );
    }
    let mut count = 0;
    let swapped = HIST.replace(
        "wrapping_add(out[ix[i]],a[i])",
        "wrapping_add(a[i],out[ix[i]])",
    );
    for (s, values, indices, n, trap_expression) in [
        (HIST, vec![], vec![u32::MAX], 0, "out[ix[i]]"),
        (swapped.as_str(), vec![], vec![u32::MAX], 0, "a[i]"),
        (HIST, vec![], vec![0], 1, "a[i]"),
        (HIST, vec![1], vec![1], 1, "out[ix[i]]"),
        (HIST, vec![1], vec![0, 1], 2, "a[i]"),
        (HIST, vec![1, 2], vec![0, 2], 2, "out[ix[i]]"),
    ] {
        let s = source(s, &values, &indices, n, &[]);
        count += prefixes(&s, true);
        let Err(eval::EvalError::Trap {
            category: eval::TrapCategory::IndexOutOfBounds,
            span,
        }) = eval::observe(&checked(&s), 10_000).result
        else {
            panic!("expected source index trap")
        };
        assert_eq!(&s[span.start..span.end], trap_expression);
    }
    // The old 4096-element and 128-contribution caps are gone (#278): sizes past them are
    // admitted under the default byte budget and refused only when they need more bytes.
    for n in [MAX_ARRAY + 1, 100_000] {
        assert!(c.candidate(&vec![0; n], &[], 0).is_ok());
        let all_zero = c.candidate(&vec![1; n], &vec![0; n], 1).unwrap();
        assert_eq!(all_zero.output(), [n as u32]);
        assert!(c.candidate(&[], &[], n as u32).is_ok());
    }
    assert!(matches!(
        c.candidate_within(&[0; 8], &[0; 8], 8, &ByteBudget::ZERO),
        Err(Error::Exceeded(_))
    ));
    let weights = vec![u32::MAX; MAX_ARRAY];
    let indices = vec![0; MAX_ARRAY];
    assert_eq!(
        c.candidate(&weights, &indices, MAX_ARRAY as u32)
            .unwrap()
            .output(),
        expected(&weights, &indices, MAX_ARRAY as u32)
    );
    let over_budget = format!(
        "{HIST} fn main() -> i32 {{ let a: Array<u32> = []; let ix: Array<u32> = []; let result = hist(a,ix,4097u32); if len(result) != 4097u32 {{ return 0; }} return 7; }}"
    );
    count += prefixes(&over_budget, false);
    println!(
        "histogram guard/fallback source/MIR/net fuel-prefix observations per engine: {count}"
    );
}

#[test]
fn issue_m4_05_gather_scatter_frontier_histogram_unknown_laws_and_observed_order_refuse() {
    for s in [
        HIST.replace(
            "wrapping_add(out[ix[i]],a[i])",
            "wrapping_mul(out[ix[i]],a[i])",
        ),
        HIST.replace("wrapping_add(out[ix[i]],a[i])", "out[ix[i]] + a[i]"),
        HIST.replace("wrapping_add(out[ix[i]],a[i])", "a[i]"),
        HIST.replace(
            "wrapping_add(out[ix[i]],a[i])",
            "wrapping_add(out[ix[i]],a[i] + 1u32)",
        ),
        HIST.replace("filled_u32(n,0u32)", "filled_u32(n,1u32)"),
        HIST.replace("0u32..len(ix)", "1u32..len(ix)"),
        HIST.replace("out[ix[i]],a[i]", "out[(ix[i] + 1u32) % n],a[i]"),
        HIST.replace("a: read", "a: mut"),
        HIST.replace("ix: read", "ix: mut"),
        HIST.replace("return out;", "return filled_u32(n,0u32);"),
    ] {
        assert!(
            matches!(
                Contract::from_checked(s.as_bytes(), &checked(&s), "hist"),
                Err(Error::Unrecognized)
            ),
            "{s}"
        );
    }
    let checked_add = HIST.replace("wrapping_add(out[ix[i]],a[i])", "out[ix[i]] + a[i]");
    let overwrite = HIST.replace("wrapping_add(out[ix[i]],a[i])", "a[i]");
    let cross_bin = HIST.replace("out[ix[i]],a[i]", "out[(ix[i] + 1u32) % n],a[i]");
    let observed = HIST.replace(
        "out[ix[i]] = wrapping_add(out[ix[i]],a[i]);",
        "out[ix[i]] = wrapping_add(out[ix[i]],a[i]); if out[ix[i]] > 10u32 { out[ix[i]] = 0u32; }",
    );
    let mut count = 0;
    for (s, values, indices, n, wanted, is_trap) in [
        (checked_add, vec![u32::MAX, 1], vec![0, 0], 1, vec![], true),
        (overwrite, vec![41, 99], vec![0, 0], 1, vec![99], false),
        (
            cross_bin,
            vec![1, 2, 3],
            vec![0, 1, 0],
            2,
            vec![6, 3],
            false,
        ),
        (observed, vec![5, 8], vec![0, 0], 1, vec![0], false),
    ] {
        let s = source(&s, &values, &indices, n, &wanted);
        let p = checked(&s);
        assert!(matches!(
            Contract::from_checked(s.as_bytes(), &p, "hist"),
            Err(Error::Unrecognized)
        ));
        let net = general_net::Net::lower(&p, general_net::Limits::default()).unwrap();
        let proof = net.verify().unwrap();
        let m = net.compiled_mir(&proof).unwrap();
        count += observations(&p, &m, is_trap);
        if is_trap {
            assert!(matches!(
                eval::observe(&p, 10_000).result,
                Err(eval::EvalError::Trap {
                    category: eval::TrapCategory::IntegerOverflow,
                    ..
                })
            ));
        }
    }
    println!(
        "histogram refused-law/order source/MIR/net fuel-prefix observations per engine: {count}"
    );
}

#[test]
fn issue_m4_05_gather_scatter_frontier_histogram_alias_and_source_invalid_negatives() {
    let shared = [1, 0, 1];
    assert_eq!(
        contract(HIST)
            .candidate(&shared, &shared, 2)
            .unwrap()
            .output(),
        &[0, 2]
    );
    let s = format!(
        "{HIST} fn main() -> i32 {{ let a: Array<u32> = [1u32,0u32,1u32]; let result = hist(a,a,2u32); if result[0u32] == 0u32 and result[1u32] == 2u32 {{ return 7; }} return 0; }}"
    );
    let mut count = prefixes(&s, false);
    for body in [
        "if false { let unused = hist(a,ix,0u32); } return 7;",
        "if true { return 7; } else { let unused = hist(a,ix,0u32); return 0; }",
    ] {
        let unselected = format!(
            "{HIST} fn main() -> i32 {{ let a: Array<u32> = [1u32]; let ix: Array<u32> = [4294967295u32]; {body} }}"
        );
        count += prefixes(&unselected, false);
    }
    for (s, code) in [
        (HIST.replace("ix[i]", "-1"), "TYPE_MISMATCH"),
        (HIST.replace("var out = filled_u32(n,0u32);", "var out = a;"), "OWN_MOVE_REQUIRED"),
        ("fn invalid(a: mut Array<u32>, ix: read Array<u32>) -> Unit { a[ix[0u32]] = 1u32; } fn main() -> i32 { var a: Array<u32> = [0u32]; invalid(a,a); return 7; }".to_owned(), "OWN_BORROW_CONFLICT"),
    ] {
        assert_eq!(check::check(front::parse(s.as_bytes()).unwrap(), false).unwrap_err().code, code);
    }
    println!("histogram alias source/MIR/net fuel-prefix observations per engine: {count}");
}

#[test]
fn issue_m4_05_gather_scatter_frontier_histogram_source_resource_limit_prefixes() {
    let mut count = 0;
    for n in [invar::array_limits::MAX_ARRAY_LEN as u32 + 1, u32::MAX] {
        let s = format!(
            "{HIST} fn main() -> i32 {{ let a: Array<u32> = []; let ix: Array<u32> = []; let result = hist(a,ix,{n}u32); return 7; }}"
        );
        let p = checked(&s);
        let c = Contract::from_checked(s.as_bytes(), &p, "hist").unwrap();
        assert!(matches!(
            c.candidate_within(&[], &[], n, &ByteBudget::new(1 << 20, 0, 0)),
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
            let extracted = mir_eval::observe(&m, fuel);
            let interaction = general_net::oracle::observe(
                &net,
                &proof,
                general_net::oracle::Limits {
                    fuel,
                    ..general_net::oracle::Limits::default()
                },
            )
            .unwrap();
            for observed in [&extracted, &interaction.semantic] {
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
    println!(
        "histogram source-resource source/MIR/net fuel-prefix observations per engine: {count}"
    );
}

#[test]
fn issue_m4_05_gather_scatter_frontier_histogram_bounded_independent_outputs_and_budget_edges() {
    let c = contract(HIST);
    for (values, indices, n) in [
        (vec![], vec![], 0),
        (vec![], vec![], 4096),
        (vec![u32::MAX, 1], vec![0, 0], 1),
        (vec![0, 0, 7, 11, 19, 13], vec![2, 0, 2, 0, 2, 0], 4096),
        (vec![u32::MAX; 128], (0..128).map(|i| i % 17).collect(), 128),
    ] {
        let p = c
            .bounded_candidate(&values, &indices, n, Budget::default())
            .unwrap();
        assert_eq!(p.output(), expected(&values, &indices, n));
        assert_eq!(
            p.output(),
            c.candidate(&values, &indices, n).unwrap().output()
        );
        assert_eq!(p.source_accesses().len(), indices.len());
        assert_eq!(p.grouped_accesses().len(), indices.len());
        assert!(p.visits() <= p.cost().visits);
        println!(
            "bounded histogram A={} I={} N={n}: reserved_bytes={} reserved_visits={} measured_visits={} sort_visits={} proof_visits={}",
            values.len(),
            indices.len(),
            p.cost().bytes,
            p.cost().visits,
            p.visits(),
            p.sort_visits(),
            p.proof_visits()
        );
        assert!(
            c.validate_bounded(&p, &values, &indices, n, Budget::default())
                .unwrap()
                <= p.cost().visits
        );
        let budget = Budget {
            bytes: p.cost().bytes,
            visits: p.cost().visits,
        };
        assert!(c.bounded_candidate(&values, &indices, n, budget).is_ok());
        for denied in [
            Budget {
                bytes: budget.bytes - 1,
                ..budget
            },
            Budget {
                visits: budget.visits - 1,
                ..budget
            },
        ] {
            assert_eq!(
                c.bounded_candidate(&values, &indices, n, denied)
                    .unwrap_err(),
                Error::Budget
            );
            assert_eq!(
                c.validate_bounded(&p, &values, &indices, n, denied),
                Err(Error::Budget)
            );
        }
    }
    for i in 0..=128usize {
        let h = if i <= 1 {
            0
        } else {
            (usize::BITS - (i - 1).leading_zeros()) as usize
        };
        let cost = Cost::bounded(4096, i, 4096, Budget::default()).unwrap();
        assert_eq!(
            cost.bytes,
            4 * 4096 + 12 * 4096 + (2 * size_of::<Access>() + 5) * i + 4096
        );
        assert_eq!(cost.visits, 2 * 4096 + 8 * 4096 + 28 * i + 16 * i * h + 512);
        assert!(cost.bytes <= MAX_STAGING_BYTES && cost.visits <= MAX_VISITS);
    }
    let maximum = Cost::bounded(4096, 128, 4096, Budget::default()).unwrap();
    assert_eq!(maximum.bytes, 74368);
    assert_eq!(maximum.visits, 59392); // Data-only proposal plus fixed headers.
    let weights = vec![u32::MAX; 4096];
    let indices = vec![0; 128];
    // Full proof work exceeds the frozen cap at the originally proposed
    // simultaneous maximum. Preserve cap and ordinary source fallback.
    assert_eq!(
        c.bounded_candidate(&weights, &indices, 4096, Budget::default())
            .unwrap_err(),
        Error::Budget
    );
    assert_eq!(
        c.bounded_candidate(
            &weights,
            &indices,
            4096,
            Budget {
                bytes: usize::MAX,
                visits: usize::MAX
            }
        )
        .unwrap_err(),
        Error::Budget
    );
    assert_eq!(
        c.candidate(&weights, &indices, 4096).unwrap().output(),
        expected(&weights, &indices, 4096)
    );
    // Past the old caps only bytes, visits and overflow refuse.
    for (a, i, n, admitted) in [
        (4097, 0, 0, true),
        (0, 129, 1, true),
        (0, 0, 4097, true),
        (50_000, 0, 0, false),
        (0, 2000, 1, false),
        (usize::MAX, 0, 0, false),
    ] {
        assert_eq!(
            Cost::bounded(a, i, n, Budget::default()).is_ok(),
            admitted,
            "{a} {i} {n}"
        );
    }
    assert_eq!(
        c.bounded_candidate(&vec![0; 2000], &vec![0; 2000], 1, Budget::default())
            .unwrap_err(),
        Error::Budget
    );
    for (values, indices, n, iteration) in [
        (vec![], vec![0], 0, 0),
        (vec![0], vec![1], 1, 0),
        (vec![0], vec![0, 0], 1, 1),
    ] {
        assert_eq!(
            c.bounded_candidate(&values, &indices, n, Budget::default())
                .unwrap_err(),
            Error::Bounds { iteration }
        );
    }
}

#[test]
fn issue_m4_05_gather_scatter_frontier_histogram_bounded_input_binding_and_alias() {
    let c = contract(HIST);
    let values = [0u32, 0, 0, 99];
    let indices = [0u32, 0, 0];
    let p = c
        .bounded_candidate(&values, &indices, 1, Budget::default())
        .unwrap();
    let replacement = values.to_vec();
    assert_eq!(
        c.validate_bounded(&p, &replacement, &indices, 1, Budget::default()),
        Err(Error::Receipt)
    );
    let changed_source = HIST.replace("return out;", "return move out;");
    assert_eq!(
        contract(&changed_source).validate_bounded(&p, &values, &indices, 1, Budget::default()),
        Err(Error::Receipt)
    );
    let shared = [1u32, 0, 1];
    let aliased = c
        .bounded_candidate(&shared, &shared, 2, Budget::default())
        .unwrap();
    assert_eq!(aliased.output(), [0, 2]);
    assert!(
        c.validate_bounded(&aliased, &shared, &shared, 2, Budget::default())
            .is_ok()
    );
    let strict_float = "fn hist(a: read Array<f32>, ix: read Array<u32>, n: u32) -> Array<f32> { var out=filled_f32(n,0.0); for i in 0u32..len(ix) { out[ix[i]]=out[ix[i]]+a[i]; } return out; }";
    assert!(matches!(
        Contract::from_checked(strict_float.as_bytes(), &checked(strict_float), "hist"),
        Err(Error::Unrecognized)
    ));
}
