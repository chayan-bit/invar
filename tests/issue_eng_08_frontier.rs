//! #57 ENG-08 part A: the dense-mask frontier observer admits by bytes, not by 16 vertices and
//! 16 rounds. Every test asserts a nonzero case count. Device rows above the old caps are pending
//! until run on hardware.
use invar::budget::{BudgetExceeded, ByteBudget, Resource};
use invar::frontier_source::{self, Budget, Contract, Error};

#[path = "support/eng_08.rs"]
mod eng_08;
use eng_08::{agree_on_every_cpu_engine, allocated_during, array, checked};

const FROZEN_A: &str = include_str!("../bench/general-net-v01/current/frontier_graph_a.inv");

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
fn contract(n: u32, rounds: u32) -> Contract {
    let f = function(n, rounds);
    Contract::from_checked(f.as_bytes(), &checked(&f), "reach").unwrap()
}
fn matrix(n: usize, pairs: &[(usize, usize)]) -> Vec<u32> {
    let mut m = vec![0; n * n];
    for &(from, to) in pairs {
        m[from * n + to] = 1;
    }
    m
}
fn chain(n: usize) -> Vec<u32> {
    let pairs: Vec<_> = (0..n - 1).map(|i| (i, i + 1)).collect();
    matrix(n, &pairs)
}
fn star(n: usize) -> Vec<u32> {
    let pairs: Vec<_> = (1..n).map(|i| (0, i)).collect();
    matrix(n, &pairs)
}
/// Independent breadth-first distances; after `rounds` rounds the source has seen distance < rounds.
fn reachable_within(edges: &[u32], n: usize, start: usize, rounds: usize) -> Vec<u32> {
    let mut dist = vec![usize::MAX; n];
    dist[start] = 0;
    let mut queue = std::collections::VecDeque::from([start]);
    while let Some(u) = queue.pop_front() {
        for v in 0..n {
            if edges[u * n + v] == 1 && dist[v] == usize::MAX {
                dist[v] = dist[u] + 1;
                queue.push_back(v);
            }
        }
    }
    dist.iter().map(|&d| u32::from(d < rounds)).collect()
}
fn program(f: &str, edges: &[u32], start: u32, count: i32) -> String {
    format!(
        "{f} fn main() -> i32 {{ let edges: Array<u32> = {}; let result = reach(edges,{start}u32); if result != {count} {{ return 0; }} return 7; }}",
        array(edges)
    )
}

#[test]
fn issue_eng_08_frontier_past_the_old_caps_is_admitted_and_exact() {
    let mut cases = 0;
    for (n, rounds) in [(17usize, 17u32), (20, 20), (40, 40), (64, 5)] {
        let c = contract(n as u32, rounds);
        for edges in [chain(n), star(n)] {
            let expected = reachable_within(&edges, n, 0, rounds as usize);
            let got = c.candidate(&edges, 0, Budget::default()).unwrap();
            assert_eq!(got.seen(), expected, "n={n} rounds={rounds}");
            assert_eq!(got.count(), expected.iter().sum::<u32>() as i32);
            assert_eq!(got.rounds().len(), rounds as usize);
            cases += 1;
        }
    }
    assert_eq!(cases, 8);
}

#[test]
fn issue_eng_08_frontier_matches_every_cpu_engine_past_the_old_caps() {
    let mut engines = 0;
    for (n, rounds) in [(17usize, 17u32), (20, 6)] {
        let f = function(n as u32, rounds);
        let edges = chain(n);
        let count = reachable_within(&edges, n, 0, rounds as usize)
            .iter()
            .sum::<u32>() as i32;
        engines += agree_on_every_cpu_engine(&program(&f, &edges, 0, count));
    }
    assert_eq!(engines, 10);
}

#[test]
fn issue_eng_08_default_capacity_no_longer_caps_the_frontier_at_16() {
    let n = 20;
    let c = contract(n as u32, 3);
    let edges = star(n);
    // Round two's frontier holds 19 vertices; the old default capacity was 16.
    let got = c.candidate(&edges, 0, Budget::default()).unwrap();
    assert_eq!(got.count(), 20);
    assert_eq!(got.rounds()[0].scheduled(), [0]);
    // An explicit capacity still bounds the observer.
    let refused = c
        .candidate(
            &edges,
            0,
            Budget {
                capacity: 4,
                work: u64::MAX,
            },
        )
        .unwrap_err();
    assert!(
        matches!(
            refused,
            Error::Capacity {
                required: 5,
                capacity: 4,
                ..
            }
        ),
        "{refused:?}"
    );
    // And an explicit work bound still does.
    let complete = got.work();
    assert_eq!(
        c.candidate(
            &edges,
            0,
            Budget {
                capacity: usize::MAX,
                work: complete - 1
            }
        )
        .unwrap_err(),
        Error::Work
    );
}

#[test]
fn issue_eng_08_one_byte_below_the_footprint_is_refused_and_at_it_is_admitted() {
    let mut cases = 0;
    for (n, rounds) in [(1usize, 1u32), (17, 17), (30, 3)] {
        let c = contract(n as u32, rounds);
        let edges = chain(n);
        let need = frontier_source::footprint(n, rounds as usize, edges.len())
            .bytes(Resource::Host)
            .unwrap();
        assert!(
            c.candidate_within(&edges, 0, Budget::default(), &ByteBudget::new(need, 0, 0))
                .is_ok()
        );
        assert_eq!(
            c.candidate_within(
                &edges,
                0,
                Budget::default(),
                &ByteBudget::new(need - 1, 0, 0)
            )
            .unwrap_err(),
            Error::Exceeded(BudgetExceeded {
                family: "frontier",
                resource: Resource::Host,
                needed: Some(need),
                available: need - 1,
            })
        );
        cases += 1;
    }
    assert_eq!(cases, 3);
}

#[test]
fn issue_eng_08_footprint_matches_measured_allocation() {
    // The sizes are part of the source, so each is its own contract. Re-verifying a contract costs
    // the same for literals of equal width, so what remains after subtracting the footprint's
    // owned bytes is one fixed number per width group.
    let mut cases = 0;
    for group in [
        [(17usize, 17u32), (18, 17), (19, 17)],
        [(10, 3), (11, 5), (12, 9)],
    ] {
        let mut fixed = Vec::new();
        for (n, rounds) in group {
            let c = contract(n as u32, rounds);
            let edges = chain(n);
            let (total, _kept) =
                allocated_during(|| c.candidate(&edges, 0, Budget::default()).unwrap());
            let owned = frontier_source::footprint(n, rounds as usize, edges.len())
                .bytes(Resource::Host)
                .unwrap()
                - 4 * edges.len();
            fixed.push(total as i64 - owned as i64);
            cases += 1;
        }
        assert!(
            fixed[0] > 0 && fixed.windows(2).all(|w| w[0] == w[1]),
            "{fixed:?}"
        );
    }
    assert_eq!(cases, 6);
}

#[test]
fn issue_eng_08_overflow_and_oversized_sources_refuse_without_allocating() {
    assert_eq!(
        frontier_source::footprint(usize::MAX, 1, 0).bytes(Resource::Host),
        None
    );
    assert_eq!(
        frontier_source::footprint(2, usize::MAX, 0).bytes(Resource::Host),
        None
    );
    assert_eq!(
        frontier_source::footprint(0, 0, usize::MAX).bytes(Resource::Host),
        None
    );
    // A source declaring the largest u32 vertex count: the footprint alone refuses it.
    let c = contract(u32::MAX, 0);
    let err = c.candidate(&[], 0, Budget::default()).unwrap_err();
    assert!(
        matches!(err, Error::Exceeded(e) if e.resource == Resource::Host),
        "{err:?}"
    );
    // Zero vertices need no bytes: a zero budget admits the region, and the start vertex is then out of bounds.
    let empty = contract(0, 0);
    assert!(
        empty
            .candidate_within(&[], 0, Budget::default(), &ByteBudget::ZERO)
            .is_err_and(|e| e == Error::Bounds)
    );
}

#[test]
fn issue_eng_08_deleted_caps_no_longer_exist_in_frontier_source() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
    let text = std::fs::read_to_string(root.join("frontier_source.rs")).unwrap();
    let mut checked = 0;
    for deleted in ["MAX_VERTICES", "MAX_ROUNDS", "1_000_000", "8192"] {
        assert!(!text.contains(deleted), "{deleted}");
        checked += 1;
    }
    assert_eq!(checked, 4);
}

#[test]
fn issue_eng_08_language_array_limit_still_refuses_below_the_byte_budget() {
    let c = contract(3, 2);
    let big = vec![0u32; invar::array_limits::MAX_ARRAY_LEN + 1];
    // 64 MiB of edges passes the default byte budget; only the language limit refuses it.
    let footprint = frontier_source::footprint(3, 2, big.len());
    assert!(ByteBudget::default().admit("frontier", &footprint).is_ok());
    assert!(matches!(
        c.candidate(&big, 0, Budget::default()),
        Err(Error::Budget)
    ));
    assert!(
        c.candidate(
            &big[..invar::array_limits::MAX_ARRAY_LEN],
            0,
            Budget::default()
        )
        .is_ok()
    );
}
