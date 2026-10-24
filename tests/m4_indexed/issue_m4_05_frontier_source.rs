#![forbid(unsafe_code)]
use frontier_source::{Budget, Contract, Discovery, Error};
use invar::frontier_source;
use invar::{budget, check, eval, front, general_net, mir, mir_eval};

const FROZEN_A: &str = include_str!("../../bench/general-net-v01/current/frontier_graph_a.inv");
const FROZEN_B: &str = include_str!("../../bench/general-net-v01/current/frontier_graph_b.inv");

fn checked(s: &str) -> check::CheckedProgram {
    check::check(front::parse(s.as_bytes()).unwrap(), false).unwrap()
}
fn contract(s: &str) -> Contract {
    let contract = Contract::from_checked(s.as_bytes(), &checked(s), "reach").unwrap();
    let spans = contract.branch_spans();
    assert_eq!(
        spans.iter().collect::<std::collections::HashSet<_>>().len(),
        7
    );
    let outer = contract.loop_span();
    assert!(
        spans
            .iter()
            .all(|s| outer.start <= s.start && s.end <= outer.end)
    );
    contract
}
fn function(n: u32, rounds: u32) -> String {
    FROZEN_A
        .split("\nfn main()")
        .next()
        .unwrap()
        .replace("5u32", &format!("{n}u32"))
        .replace(
            &format!("for step in 0u32 .. {n}u32"),
            &format!("for step in 0u32 .. {rounds}u32"),
        )
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
fn source(f: &str, edges: &[u32], start: u32, count: i32) -> String {
    format!(
        "{f} fn main() -> i32 {{ let edges: Array<u32> = {}; let retained: Array<u32> = {}; let result = reach(edges,{start}u32); if result != {count} {{ return 0; }} for keep in 0u32..len(edges) {{ if edges[keep] != retained[keep] {{ return 0; }} }} return 7; }}",
        array(edges),
        array(edges)
    )
}
fn matrix(n: usize, pairs: &[(usize, usize)]) -> Vec<u32> {
    let mut m = vec![0; n * n];
    for &(from, to) in pairs {
        m[from * n + to] = 1;
    }
    m
}

// Independent shortest-distance relaxation, not the source's frontier/seen
// update algorithm. After R rounds the source has visited distances below R.
fn distance_reference(edges: &[u32], n: usize, start: usize, rounds: u32) -> Vec<u32> {
    let mut distances = vec![u32::MAX; n];
    distances[start] = 0;
    for _ in 0..n {
        let old = distances.clone();
        for destination in 0..n {
            for predecessor in 0..n {
                if edges[predecessor * n + destination] == 1 && old[predecessor] != u32::MAX {
                    distances[destination] = distances[destination].min(old[predecessor] + 1);
                }
            }
        }
    }
    distances
        .into_iter()
        .map(|d| u32::from(d < rounds))
        .collect()
}

fn compare_prefixes(p: &check::CheckedProgram, m: &mir::Program, expected: Option<i32>) -> u64 {
    let complete = eval::observe(p, 20_000);
    if let Some(value) = expected {
        assert_eq!(complete.result, Ok(eval::Value::I32(value)));
    } else {
        assert!(matches!(complete.result, Err(eval::EvalError::Trap { .. })));
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
fn prefixes(s: &str, expected: Option<i32>) -> u64 {
    let p = checked(s);
    let c = Contract::from_checked(s.as_bytes(), &p, "reach").unwrap();
    let m = c.verified_mir().unwrap();
    mir::verify(&m).unwrap();
    assert_eq!(c.source_contract(), m.contract_digest());
    compare_prefixes(&p, &m, expected)
}

#[test]
fn issue_m4_05_gather_scatter_frontier_frozen_source_identity_and_shape_negatives() {
    for s in [FROZEN_A, FROZEN_B] {
        let c = contract(s);
        assert_eq!(c.vertices(), 5);
        assert_eq!(c.round_count(), 5);
        assert!(s[c.loop_span().start..c.loop_span().end].starts_with("for step"));
        assert_ne!(c.source_contract(), [0; 32]);
        let mut p = checked(s);
        p.expression_types.clear();
        assert!(matches!(
            Contract::from_checked(s.as_bytes(), &p, "reach"),
            Err(Error::CheckedMismatch)
        ));
        let mut p = checked(s);
        p.function_expression_types.get_mut("reach").unwrap().pop();
        assert!(matches!(
            Contract::from_checked(s.as_bytes(), &p, "reach"),
            Err(Error::CheckedMismatch)
        ));
        let mut p = checked(s);
        p.program.functions[0].body.statements[3].span.end -= 1;
        assert!(matches!(
            Contract::from_checked(s.as_bytes(), &p, "reach"),
            Err(Error::CheckedMismatch)
        ));
    }
    let s = function(3, 3);
    let renamed = s
        .replace("reach", "explore")
        .replace("frontier", "current")
        .replace("seen", "visited")
        .replace("next", "following")
        .replace("edges", "adjacency");
    let c = Contract::from_checked(renamed.as_bytes(), &checked(&renamed), "explore").unwrap();
    assert_eq!(
        c.candidate(&matrix(3, &[(0, 1), (1, 2)]), 0, Budget::default())
            .unwrap()
            .count(),
        3
    );
    let grouped = s
        .replace("u * 3u32 + neighbor", "((u) * (3u32)) + (neighbor)")
        .replace("frontier[u] == 1u32", "(frontier[(u)]) == (1u32)");
    assert_eq!(contract(&grouped).vertices(), 3);
    assert!(matches!(
        Contract::from_checked(renamed.as_bytes(), &checked(&s), "explore"),
        Err(Error::CheckedMismatch)
    ));
    for changed in [
        s.replace("seen[u] = 1u32;", "seen[u] = 0u32;"),
        s.replace(
            "next[neighbor] = 1u32;",
            "next[neighbor] = wrapping_add(next[neighbor],1u32);",
        ),
        s.replace("edges[u * 3u32 + neighbor]", "edges[edges[u]]"),
        s.replace(
            "frontier[copy_index] = next[copy_index];",
            "frontier[copy_index] = frontier[copy_index];",
        ),
        s.replace(
            "for copy_index in 0u32 .. 3u32 { frontier[copy_index] = next[copy_index]; }",
            "",
        ),
        s.replace(
            "for copy_index in 0u32 .. 3u32",
            "for copy_index in 0u32 .. 2u32",
        ),
        s.replace("edges: read", "edges: mut"),
        s.replace("for step in 0u32 .. 3u32", "while true"),
        s.replace("seen[neighbor] == 0u32", "seen[u] == 0u32"),
        s.replace(
            "edges[u * 3u32 + neighbor] == 1u32",
            "edges[u * 3u32 + neighbor] != 0u32",
        ),
        s.replace(
            "frontier[u] == 1u32 and seen[u] == 0u32",
            "seen[u] == 0u32 and frontier[u] == 1u32",
        ),
    ] {
        assert!(
            matches!(
                Contract::from_checked(changed.as_bytes(), &checked(&changed), "reach"),
                Err(Error::Unrecognized)
            ),
            "{changed}"
        );
    }
}

#[test]
fn issue_m4_05_gather_scatter_frontier_frozen_graphs_and_distance_oracle() {
    let mut count = 0;
    for (n, rounds, pairs) in [
        (1, 0, vec![]),
        (1, 3, vec![]),
        (1, 2, vec![(0, 0)]),
        (3, 1, vec![(0, 1), (1, 2)]),
        (3, 3, vec![(0, 1), (1, 2)]),
        (4, 2, vec![(0, 1), (0, 2), (0, 3)]),
        (4, 3, vec![(0, 1), (0, 2), (1, 3), (2, 3)]),
        (4, 3, vec![(0, 1), (1, 2), (2, 3), (3, 0)]),
        (4, 3, vec![(0, 1), (0, 2), (1, 2), (2, 3), (3, 1)]),
    ] {
        let f = function(n as u32, rounds);
        let edges = matrix(n, &pairs);
        let original = edges.clone();
        let c = contract(&f);
        let got = c.candidate(&edges, 0, Budget::default()).unwrap();
        let seen = distance_reference(&edges, n, 0, rounds);
        assert_eq!(got.seen(), seen);
        assert_eq!(got.count(), seen.iter().sum::<u32>() as i32);
        assert_eq!(got.rounds().len(), rounds as usize);
        assert_eq!(edges, original);
        for (index, round) in got.rounds().iter().enumerate() {
            assert_eq!(round.number(), index as u32);
            let scheduled: Vec<_> = round
                .frontier_before()
                .iter()
                .enumerate()
                .filter(|(_, v)| **v == 1)
                .map(|(i, _)| i as u32)
                .collect();
            assert_eq!(round.scheduled(), scheduled);
            assert!(round.processed().windows(2).all(|p| p[0] < p[1]));
            let reads: Vec<_> = round
                .processed()
                .iter()
                .flat_map(|u| (0..n).map(move |v| *u as usize * n + v))
                .collect();
            assert_eq!(round.edge_reads(), reads);
            if index + 1 < got.rounds().len() {
                assert_eq!(round.next(), got.rounds()[index + 1].frontier_before());
            }
            assert!(round.seen_after().iter().all(|b| *b <= 1));
        }
        count += prefixes(&source(&f, &edges, 0, got.count()), Some(7));
    }
    let frozen_edges = [
        matrix(5, &[(0, 1), (0, 2), (1, 3), (2, 3), (3, 4), (4, 1)]),
        matrix(5, &[(0, 1), (2, 3), (3, 2), (3, 4)]),
    ];
    for (s, edges, start, want) in [
        (FROZEN_A, &frozen_edges[0], 0, 5),
        (FROZEN_B, &frozen_edges[1], 2, 3),
    ] {
        assert_eq!(
            contract(s)
                .candidate(edges, start, Budget::default())
                .unwrap()
                .count(),
            want
        );
        count += prefixes(s, Some(want));
    }
    // Source tests edges == 1, rather than interpreting every nonzero as an edge.
    let odd = [0, u32::MAX, 2, 0];
    let f = function(2, 2);
    assert_eq!(
        contract(&f)
            .candidate(&odd, 0, Budget::default())
            .unwrap()
            .count(),
        1
    );
    count += prefixes(&source(&f, &odd, 0, 1), Some(7));
    println!("frontier positive source/MIR/net fuel-prefix observations per engine: {count}");
}

#[test]
fn issue_m4_05_gather_scatter_frontier_frozen_duplicate_and_same_round_order_records() {
    let f = function(4, 3);
    let edges = matrix(4, &[(0, 1), (0, 2), (1, 3), (2, 3)]);
    let c = contract(&f);
    let got = c.candidate(&edges, 0, Budget::default()).unwrap();
    assert_eq!(got.rounds()[0].processed(), &[0]);
    assert_eq!(got.rounds()[0].next(), &[0, 1, 1, 0]);
    assert_eq!(got.rounds()[1].processed(), &[1, 2]);
    assert_eq!(got.rounds()[1].next(), &[0, 0, 0, 1]);
    assert_eq!(
        got.rounds()[1].discoveries(),
        &[
            Discovery {
                from: 1,
                to: 3,
                edge_index: 7
            },
            Discovery {
                from: 2,
                to: 3,
                edge_index: 11
            }
        ]
    );
    assert_eq!(got.rounds()[2].scheduled(), &[3]);
    assert_eq!(got.count(), 4);
    // Later-in-round marking must not retroactively clear next's earlier bit.
    let f = function(3, 3);
    let edges = matrix(3, &[(0, 1), (0, 2), (1, 2)]);
    let got = contract(&f)
        .candidate(&edges, 0, Budget::default())
        .unwrap();
    assert_eq!(got.rounds()[1].processed(), &[1, 2]);
    assert_eq!(got.rounds()[1].next(), &[0, 0, 1]);
    assert_eq!(got.rounds()[1].seen_after(), &[1, 1, 1]);
    assert_eq!(got.rounds()[2].scheduled(), &[2]);
    assert!(got.rounds()[2].processed().is_empty());
    assert_eq!(got.rounds()[2].next(), &[0, 0, 0]);
    let count = prefixes(&source(&f, &edges, 0, 3), Some(7));
    println!("frontier exact-round source/MIR/net fuel-prefix observations per engine: {count}");
}

#[test]
fn issue_m4_05_gather_scatter_frontier_frozen_capacity_work_bounds_and_fallback() {
    let f = function(4, 2);
    let edges = matrix(4, &[(0, 1), (0, 2), (0, 3)]);
    let c = contract(&f);
    assert_eq!(
        c.candidate(
            &edges,
            0,
            Budget {
                capacity: 2,
                work: 8192
            }
        )
        .unwrap_err(),
        Error::Capacity {
            round: 0,
            required: 3,
            capacity: 2
        }
    );
    assert_eq!(
        c.candidate(
            &edges,
            0,
            Budget {
                capacity: 0,
                work: 8192
            }
        )
        .unwrap_err(),
        Error::Capacity {
            round: 0,
            required: 1,
            capacity: 0
        }
    );
    let complete = c.candidate(&edges, 0, Budget::default()).unwrap();
    for work in 0..complete.work() {
        assert_eq!(
            c.candidate(&edges, 0, Budget { capacity: 4, work })
                .unwrap_err(),
            Error::Work
        );
    }
    assert_eq!(
        c.candidate(
            &edges,
            0,
            Budget {
                capacity: 4,
                work: complete.work()
            }
        )
        .unwrap()
        .count(),
        4
    );
    let mut count = prefixes(&source(&f, &edges, 0, 4), Some(7));
    assert_eq!(
        c.candidate(&edges, u32::MAX, Budget::default())
            .unwrap_err(),
        Error::Bounds
    );
    assert_eq!(
        c.candidate(&edges[..3], 0, Budget::default()).unwrap_err(),
        Error::Bounds
    );
    for (f, edges, start) in [
        (function(0, 0), vec![], 0),
        (function(2, 1), vec![0], 0),
        (function(2, 1), vec![0; 4], 2),
    ] {
        let s = source(&f, &edges, start, 0);
        count += prefixes(&s, None);
        let Err(eval::EvalError::Trap {
            category: eval::TrapCategory::IndexOutOfBounds,
            span,
        }) = eval::observe(&checked(&s), 20_000).result
        else {
            panic!("expected original bounds trap")
        };
        let expected = if start == 2 || edges.is_empty() {
            "frontier[start]"
        } else {
            "edges[u * 2u32 + neighbor]"
        };
        assert_eq!(&s[span.start..span.end], expected);
    }
    // Conservative full-matrix guard can refuse a lawful short source payload.
    let f = function(3, 2);
    assert_eq!(
        contract(&f)
            .candidate(&[0; 3], 0, Budget::default())
            .unwrap_err(),
        Error::Bounds
    );
    count += prefixes(&source(&f, &[0; 3], 0, 1), Some(7));
    // The old 16-vertex and 16-round caps are gone (#57): 17 of either is admitted under the
    // default byte budget and refused only by a budget too small for its footprint.
    for (n, r) in [(17u32, 0u32), (1, 17)] {
        let f = function(n, r);
        let edges = matrix(n as usize, &[]);
        let c = contract(&f);
        assert_eq!(
            c.candidate(&edges, 0, Budget::default()).unwrap().count(),
            i32::from(r != 0)
        );
        assert!(matches!(
            c.candidate_within(&edges, 0, Budget::default(), &budget::ByteBudget::ZERO),
            Err(Error::Exceeded(_))
        ));
        count += prefixes(
            &source(
                &f,
                if r == 0 { &[] } else { &edges },
                0,
                if r == 0 { 0 } else { 1 },
            ),
            Some(7),
        );
    }
    let f = function(1, 0);
    assert_eq!(
        contract(&f)
            .candidate(
                &[0],
                0,
                Budget {
                    capacity: 0,
                    work: 1
                }
            )
            .unwrap()
            .count(),
        0
    );
    println!("frontier guard/fallback source/MIR/net fuel-prefix observations per engine: {count}");
}

#[test]
fn issue_m4_05_gather_scatter_frontier_frozen_refused_pointer_chasing_and_unbounded_rounds() {
    let f = function(2, 2);
    let pointer = f.replace("edges[u * 2u32 + neighbor]", "edges[edges[u]]");
    let s = source(&pointer, &[0; 4], 0, 1);
    let p = checked(&s);
    assert!(matches!(
        Contract::from_checked(s.as_bytes(), &p, "reach"),
        Err(Error::Unrecognized)
    ));
    let net = general_net::Net::lower(&p, general_net::Limits::default()).unwrap();
    let proof = net.verify().unwrap();
    let m = net.compiled_mir(&proof).unwrap();
    let mut count = compare_prefixes(&p, &m, Some(7));
    for (changed, edges, wanted) in [
        (
            function(4, 3).replace(
                "next[neighbor] = 1u32;",
                "next[neighbor] = wrapping_add(next[neighbor],1u32);",
            ),
            matrix(4, &[(0, 1), (0, 2), (1, 3), (2, 3)]),
            3,
        ),
        (
            function(3, 3).replace(
                "for copy_index in 0u32 .. 3u32 { frontier[copy_index] = next[copy_index]; }",
                "",
            ),
            matrix(3, &[(0, 1), (1, 2)]),
            1,
        ),
    ] {
        let changed = source(&changed, &edges, 0, wanted);
        let p = checked(&changed);
        assert!(matches!(
            Contract::from_checked(changed.as_bytes(), &p, "reach"),
            Err(Error::Unrecognized)
        ));
        let net = general_net::Net::lower(&p, general_net::Limits::default()).unwrap();
        let proof = net.verify().unwrap();
        let m = net.compiled_mir(&proof).unwrap();
        count += compare_prefixes(&p, &m, Some(7));
    }
    let unbounded = f.replace("for step in 0u32 .. 2u32", "while true");
    let s = source(&unbounded, &[0; 4], 0, 0);
    let p = checked(&s);
    assert!(matches!(
        Contract::from_checked(s.as_bytes(), &p, "reach"),
        Err(Error::Unrecognized)
    ));
    let net = general_net::Net::lower(&p, general_net::Limits::default()).unwrap();
    let proof = net.verify().unwrap();
    let m = net.compiled_mir(&proof).unwrap();
    for fuel in 0..=128 {
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
        if fuel == 0 {
            assert!(matches!(
                reference.result,
                Err(eval::EvalError::Tool {
                    code: "INVALID_FUEL",
                    ..
                })
            ));
        } else {
            assert!(
                matches!(reference.result,Err(eval::EvalError::EvaluationLimit {steps_used,..}) if steps_used==fuel)
            );
        }
        for actual in [&extracted, &interaction.semantic] {
            assert_eq!(actual.result, reference.result);
            assert_eq!(actual.steps_used, reference.steps_used);
            assert_eq!(actual.cleanup_trace, reference.cleanup_trace);
        }
    }
    println!(
        "frontier refused pointer/unbounded source/MIR/net fuel-prefix observations per engine: {}",
        count + 129
    );
}

#[test]
fn issue_m4_05_gather_scatter_frontier_frozen_alias_negative_and_resource_prefixes() {
    let f = function(2, 2);
    for (s,code) in [
        (f.replace("frontier[start]","frontier[-1]"),"TYPE_MISMATCH"),
        (f.replace("var seen: Array<u32> = filled_u32(2u32, 0u32);","var seen: Array<u32> = frontier;"),"OWN_MOVE_REQUIRED"),
        ("fn invalid(a: mut Array<u32>, b: read Array<u32>) -> Unit { a[0u32] = b[0u32]; } fn main() -> i32 { var a: Array<u32> = [0u32]; invalid(a,a); return 7; }".to_owned(),"OWN_BORROW_CONFLICT"),
    ] { assert_eq!(check::check(front::parse(s.as_bytes()).unwrap(),false).unwrap_err().code,code); }
    let mut count = 0;
    let s = format!(
        "{f} fn main() -> i32 {{ let edges: Array<u32> = []; if false {{ let unused = reach(edges,4294967295u32); }} return 7; }}"
    );
    count += prefixes(&s, Some(7));
    for n in [invar::array_limits::MAX_ARRAY_LEN as u32 + 1, u32::MAX] {
        let f = function(n, 0);
        let s = source(&f, &[], 0, 0);
        let p = checked(&s);
        let c = Contract::from_checked(s.as_bytes(), &p, "reach").unwrap();
        assert!(matches!(
            c.candidate_within(
                &[],
                0,
                Budget::default(),
                &budget::ByteBudget::new(1 << 20, 0, 0)
            ),
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
            for actual in [&extracted, &interaction.semantic] {
                assert_eq!(actual.result, reference.result);
                assert_eq!(actual.steps_used, reference.steps_used);
                assert_eq!(actual.cleanup_trace, reference.cleanup_trace);
            }
            count += 1;
        }
    }
    println!(
        "frontier demand/resource source/MIR/net fuel-prefix observations per engine: {count}"
    );
}
