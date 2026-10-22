use invar::{check, front, frontier_source, histogram_source, indexed_device, indexed_source};
use invar_device::{
    Capabilities,
    indexed::{Bound, Output},
};
pub const GATHER: &str = "fn work(a: read Array<u32>, ix: read Array<u32>) -> Array<u32> { var out = filled_u32(len(ix),0u32); for i in 0u32..len(ix) { out[i] = a[ix[i]]; } return out; }";
pub const SCATTER: &str = "fn work(a: read Array<u32>, ix: read Array<u32>, n: u32) -> Array<u32> { var out = filled_u32(n,0u32); for i in 0u32..len(ix) { out[ix[i]] = a[i]; } return out; }";
pub const HIST: &str = "fn work(a: read Array<u32>, ix: read Array<u32>, n: u32) -> Array<u32> { var out = filled_u32(n,0u32); for i in 0u32..len(ix) { out[ix[i]] = wrapping_add(out[ix[i]],a[i]); } return out; }";
pub struct Case {
    pub name: &'static str,
    pub bound: Bound,
    pub expected: Output,
}
fn checked(s: &str) -> check::CheckedProgram {
    check::check(front::parse(s.as_bytes()).unwrap(), false).unwrap()
}
fn indexed(
    name: &'static str,
    s: &str,
    a: &[u32],
    ix: &[u32],
    n: Option<u32>,
    caps: Capabilities,
) -> Case {
    let bound = indexed_device::Plan::from_source(s.as_bytes(), "work")
        .unwrap()
        .bind(a, ix, n, caps)
        .unwrap();
    let expected = if s == HIST {
        histogram_source::Contract::from_checked(s.as_bytes(), &checked(s), "work")
            .unwrap()
            .candidate(a, ix, n.unwrap())
            .unwrap()
            .output()
            .to_vec()
    } else {
        indexed_source::Contract::from_checked(s.as_bytes(), &checked(s), "work")
            .unwrap()
            .candidate(a, ix, n)
            .unwrap()
            .output()
            .to_vec()
    };
    Case {
        name,
        bound,
        expected: Output {
            values: expected,
            frontier_masks: vec![],
            compaction_maps: vec![],
        },
    }
}
pub fn frontier_source(n: u32, rounds: u32) -> String {
    include_str!("../../bench/general-net-v01/current/frontier_graph_a.inv")
        .split("\nfn main()")
        .next()
        .unwrap()
        .replace("reach", "work")
        .replace("5u32", &format!("{n}u32"))
        .replace(
            &format!("for step in 0u32 .. {n}u32"),
            &format!("for step in 0u32 .. {rounds}u32"),
        )
}
fn frontier(
    name: &'static str,
    n: u32,
    rounds: u32,
    pairs: &[(usize, usize)],
    caps: Capabilities,
) -> Case {
    let s = frontier_source(n, rounds);
    let mut edges = vec![0; n as usize * n as usize];
    for &(u, v) in pairs {
        edges[u * n as usize + v] = 1;
    }
    let bound = indexed_device::Plan::from_source(s.as_bytes(), "work")
        .unwrap()
        .bind(&edges, &[], Some(0), caps)
        .unwrap();
    let oracle = frontier_source::Contract::from_checked(s.as_bytes(), &checked(&s), "work")
        .unwrap()
        .candidate(&edges, 0, Default::default())
        .unwrap();
    Case {
        name,
        bound,
        expected: Output {
            values: oracle.seen().to_vec(),
            frontier_masks: oracle.rounds().iter().map(|r| r.next().to_vec()).collect(),
            compaction_maps: oracle
                .rounds()
                .iter()
                .map(|r| {
                    r.next()
                        .iter()
                        .enumerate()
                        .filter_map(|(i, &v)| (v == 1).then_some(i as u32))
                        .collect()
                })
                .collect(),
        },
    }
}
pub fn cases(caps: Capabilities) -> Vec<Case> {
    let values: Vec<_> = (0..257).map(|i| u32::MAX.wrapping_sub(i * 17)).collect();
    let sorted: Vec<_> = (0..257).collect();
    let mut random: Vec<_> = (0..257).collect();
    let mut seed = 0x0004_d405u32;
    for i in (1..random.len()).rev() {
        seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        random.swap(i, seed as usize % (i + 1));
    }
    let mut out = vec![
        indexed("gather-sorted", GATHER, &values, &sorted, None, caps),
        indexed(
            "gather-permuted-ragged",
            GATHER,
            &values,
            &random,
            None,
            caps,
        ),
        indexed(
            "gather-duplicates",
            GATHER,
            &[7, 11],
            &[1, 0, 1],
            None,
            caps,
        ),
        indexed("gather-empty", GATHER, &[], &[], None, caps),
        indexed("gather-singleton", GATHER, &[u32::MAX], &[0], None, caps),
        indexed(
            "scatter-ragged-permuted",
            SCATTER,
            &values,
            &random,
            Some(259),
            caps,
        ),
        indexed("scatter-empty", SCATTER, &[], &[], Some(3), caps),
        indexed(
            "histogram-modular-overflow",
            HIST,
            &[u32::MAX, 1, 7, u32::MAX, 2],
            &[0, 0, 1, 1, 1],
            Some(3),
            caps,
        ),
        indexed("histogram-empty", HIST, &[], &[], Some(0), caps),
        indexed(
            "histogram-all-duplicate",
            HIST,
            &values,
            &vec![0; 257],
            Some(1),
            caps,
        ),
        frontier("frontier-empty-rounds", 1, 0, &[], caps),
        frontier("frontier-singleton", 1, 1, &[], caps),
        frontier(
            "frontier-chain",
            5,
            5,
            &[(0, 1), (1, 2), (2, 3), (3, 4)],
            caps,
        ),
        frontier(
            "frontier-round-exhaustion",
            5,
            2,
            &[(0, 1), (1, 2), (2, 3), (3, 4)],
            caps,
        ),
        frontier(
            "frontier-star",
            5,
            5,
            &[(0, 1), (0, 2), (0, 3), (0, 4)],
            caps,
        ),
        frontier(
            "frontier-balanced",
            7,
            7,
            &[(0, 1), (0, 2), (1, 3), (1, 4), (2, 5), (2, 6)],
            caps,
        ),
        frontier(
            "frontier-skewed",
            7,
            7,
            &[(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6)],
            caps,
        ),
        frontier(
            "frontier-ragged-cyclic-duplicates",
            5,
            5,
            &[(0, 1), (0, 2), (1, 3), (2, 3), (3, 1), (3, 4), (4, 0)],
            caps,
        ),
    ];
    let s = "fn work(a: read Array<u32>) -> Array<u32> {var out=filled_u32(7u32,0u32);for i in 0u32..3u32 {out[i*2u32+1u32]=a[i];}return out;}";
    out.push(Case {
        name: "scatter-source-affine-proof",
        bound: indexed_device::Plan::from_source(s.as_bytes(), "work")
            .unwrap()
            .bind(&[7, 11, 13], &[], None, caps)
            .unwrap(),
        expected: Output {
            values: vec![0, 7, 0, 11, 0, 13, 0],
            frontier_masks: vec![],
            compaction_maps: vec![],
        },
    });
    let empty = s.replace("0u32..3u32", "2u32..0u32");
    let oracle =
        invar::scatter_proof::Contract::from_checked(empty.as_bytes(), &checked(&empty), "work")
            .unwrap()
            .candidate(&[])
            .unwrap();
    out.push(Case {
        name: "scatter-affine-empty-undemanded-input",
        bound: indexed_device::Plan::from_source(empty.as_bytes(), "work")
            .unwrap()
            .bind(&[], &[], None, caps)
            .unwrap(),
        expected: Output {
            values: oracle,
            frontier_masks: vec![],
            compaction_maps: vec![],
        },
    });
    out
}

/// Every rejection happens before a device submission and returns no output.
pub fn refusals(caps: Capabilities) -> usize {
    let tests = [
        ("gather-oob", GATHER, vec![7], vec![1], None),
        ("gather-negative-u32", GATHER, vec![7], vec![u32::MAX], None),
        (
            "scatter-duplicate",
            SCATTER,
            vec![7, 11],
            vec![0, 0],
            Some(1),
        ),
        (
            "scatter-rhs-oob-first",
            SCATTER,
            vec![],
            vec![u32::MAX],
            Some(1),
        ),
        ("histogram-oob", HIST, vec![1], vec![1], Some(1)),
        (
            "histogram-bin-oob-first",
            HIST,
            vec![],
            vec![u32::MAX],
            Some(1),
        ),
        (
            "resource-limit",
            SCATTER,
            vec![],
            vec![],
            Some(invar_device::indexed::MAX_ELEMENTS + 1),
        ),
    ];
    for (name, source, values, indices, n) in tests {
        let error = indexed_device::Plan::from_source(source.as_bytes(), "work")
            .unwrap()
            .bind(&values, &indices, n, caps)
            .expect_err("guard must refuse before submit");
        println!(
            "{name} requested_target={caps:?} actual_target=none refused={error:?} publication=none"
        );
    }
    7
}

/// Large inputs bypass the bounded evaluator, retaining the exact source law.
pub fn large_bound(family: &str, elements: u32, caps: Capabilities) -> Bound {
    let n = elements as usize;
    assert!(elements.is_power_of_two());
    assert!(elements <= invar_device::indexed::MAX_ELEMENTS);
    if family == "frontier" {
        let vertices = elements.isqrt();
        assert_eq!(vertices * vertices, elements);
        let mut edges = vec![0; n];
        // Nontrivial chain plus a last-row read and convergent discoveries.
        // Only reached rows are read; the allocation still covers n*n cells.
        for u in 0..16usize.min(vertices as usize - 1) {
            edges[u * vertices as usize + u + 1] = 1;
            edges[u * vertices as usize + vertices as usize - 1] = 1;
        }
        edges[n - vertices as usize] = 1;
        return indexed_device::Plan::from_source(frontier_source(vertices, 16).as_bytes(), "work")
            .unwrap()
            .bind(&edges, &[], Some(0), caps)
            .unwrap();
    }
    let values: Vec<u32> = (0..elements)
        .map(|i| u32::MAX.wrapping_sub(i.wrapping_mul(17)))
        .collect();
    let indices: Vec<u32> = (0..elements)
        .map(|i| match family {
            "gather" => i.wrapping_mul(17) & (elements - 1),
            "scatter" => elements - 1 - i,
            "histogram" if i == elements - 1 => elements - 1,
            "histogram" => i % elements.min(65_537),
            _ => panic!("unknown family"),
        })
        .collect();
    let (source, scalar) = match family {
        "gather" => (GATHER, None),
        "scatter" => (SCATTER, Some(elements)),
        "histogram" => (HIST, Some(elements)),
        _ => unreachable!(),
    };
    indexed_device::Plan::from_source(source.as_bytes(), "work")
        .unwrap()
        .bind(&values, &indices, scalar, caps)
        .unwrap()
}

/// Plain sequential Rust family laws over the very same guarded inputs.
/// This deliberately calls neither emitter nor source/reference evaluator.
pub fn sequential_reference(bound: &Bound) -> Output {
    use invar_device::indexed::Operation;
    let mut output = Output {
        values: vec![0; bound.plan().ir().output.elements as usize],
        frontier_masks: vec![],
        compaction_maps: vec![],
    };
    match bound.plan().ir().operation {
        Operation::Gather => {
            for (i, &index) in bound.indices().iter().enumerate() {
                output.values[i] = bound.values()[index as usize];
            }
        }
        Operation::InjectiveScatter => {
            for (i, &index) in bound.indices().iter().enumerate() {
                output.values[index as usize] = bound.values()[i];
            }
        }
        Operation::Histogram => {
            for (i, &index) in bound.indices().iter().enumerate() {
                let slot = &mut output.values[index as usize];
                *slot = slot.wrapping_add(bound.values()[i]);
            }
        }
        Operation::Frontier {
            vertices, rounds, ..
        } => {
            let n = vertices as usize;
            output.values.resize(n, 0);
            let mut frontier = vec![0; n];
            frontier[bound.start() as usize] = 1;
            for _ in 0..rounds {
                let mut next = vec![0; n];
                for (u, &active) in frontier.iter().enumerate() {
                    if active == 1 && output.values[u] == 0 {
                        output.values[u] = 1;
                        for (v, bit) in next.iter_mut().enumerate() {
                            if bound.values()[u * n + v] == 1 && output.values[v] == 0 {
                                *bit = 1;
                            }
                        }
                    }
                }
                output.compaction_maps.push(
                    next.iter()
                        .enumerate()
                        .filter_map(|(v, &bit)| (bit == 1).then_some(v as u32))
                        .collect(),
                );
                output.frontier_masks.push(next.clone());
                frontier = next;
            }
        }
    }
    output
}
