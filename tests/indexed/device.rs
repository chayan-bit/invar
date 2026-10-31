//! Issue #5 (M4-05).
#![forbid(unsafe_code)]
#[path = "../support/m4_05_device.rs"]
mod corpus;
use invar::indexed_device::{self, Plan};
use invar_device::{
    Capabilities,
    indexed::{self as ir, Error, Operation},
};
fn caps() -> Capabilities {
    Capabilities::Cuda {
        major: 12,
        minor: 9,
        sm: 120,
    }
}
#[test]
fn gather_scatter_frontier_device_ir_emitters_same_corpus() {
    for target in [
        caps(),
        Capabilities::Metal {
            apple_family: 7,
            msl_major: 3,
            msl_minor: 1,
        },
    ] {
        let cases = corpus::cases(target);
        assert_eq!(cases.len(), 20);
        for case in cases {
            let p = case.bound.plan();
            let source = match target {
                Capabilities::Cuda { .. } => ir::emit_cuda(p),
                _ => ir::emit_msl(p),
            }
            .unwrap();
            assert!(source.contains(&format!("caps={target:?}")));
            assert!(source.contains("invar-indexed-u32-v1"));
            assert_eq!(
                ir::decode(
                    p,
                    case.expected
                        .values
                        .iter()
                        .chain(case.expected.frontier_masks.iter().flatten())
                        .copied()
                        .collect(),
                    &[0, 0]
                )
                .unwrap(),
                case.expected
            );
            if let Operation::Frontier { rounds, .. } = p.ir().operation {
                assert_eq!(ir::schedule(p).len(), rounds as usize * 2);
                assert!(source.contains("control[1]"));
            }
            println!("{} {target:?} verified", case.name);
        }
    }
}
#[test]
fn gather_scatter_frontier_device_mutations_refuse() {
    let cases = corpus::cases(caps());
    let original = cases[0].bound.plan().ir();
    for mutation in 0..13 {
        let mut changed = original.clone();
        match mutation {
            0 => changed.version += 1,
            1 => changed.source_identity ^= 1,
            2 => changed.graph_revision += 1,
            3 => changed.source_contract[0] ^= 1,
            4 => changed.span.end += 1,
            5 => changed.values.elements += 1,
            6 => changed.indices.elements += 1,
            7 => changed.output.elements += 1,
            8 => changed.output.id = changed.values.id,
            9 => changed.law = Some(ir::HISTOGRAM_LAW.into()),
            10 => changed.operation = Operation::Histogram,
            11 => changed.values.layout = invar_device::Layout::Unknown,
            _ => changed.output.address = invar_device::AddressSpace::Host,
        };
        assert!(
            ir::verify(changed, original, caps()).is_err(),
            "mutation {mutation}"
        );
    }
    for case in &cases {
        let p = case.bound.plan();
        assert!(ir::decode(p, vec![], &[0, 1]).is_err());
        assert!(ir::decode(p, vec![0; p.ir().output.elements as usize], &[0]).is_err());
    }
    assert!(
        ir::verify(
            original.clone(),
            original,
            Capabilities::Cuda {
                major: 12,
                minor: 4,
                sm: 120
            }
        )
        .is_err()
    );
    assert!(ir::emit_msl(corpus::cases(caps())[0].bound.plan()).is_err());
}
#[test]
fn gather_scatter_frontier_device_adversarial_guards() {
    let gather = Plan::from_source(corpus::GATHER.as_bytes(), "work").unwrap();
    for ix in [&[1u32][..], &[u32::MAX][..]] {
        assert!(matches!(
            gather.bind(&[7], ix, None, caps()),
            Err(indexed_device::Error::Device(Error::Index {
                iteration: 0,
                ..
            }))
        ));
    }
    assert!(gather.bind(&vec![0; 4097], &[], None, caps()).is_ok());
    let scatter = Plan::from_source(corpus::SCATTER.as_bytes(), "work").unwrap();
    assert!(matches!(
        scatter.bind(&[7, 11], &[0, 0], Some(1), caps()),
        Err(indexed_device::Error::Device(Error::Duplicate {
            iteration: 1,
            index: 0
        }))
    ));
    assert!(matches!(
        scatter.bind(&[], &[u32::MAX], Some(1), caps()),
        Err(indexed_device::Error::Device(Error::Index {
            operand: "values",
            ..
        }))
    ));
    let hist = Plan::from_source(corpus::HIST.as_bytes(), "work").unwrap();
    assert!(hist.bind(&[1], &[1], Some(1), caps()).is_err());
    assert!(matches!(
        hist.bind(&[], &[u32::MAX], Some(1), caps()),
        Err(indexed_device::Error::Device(Error::Index {
            operand: "indices",
            ..
        }))
    ));
    assert!(matches!(
        hist.bind(&[], &[0], Some(1), caps()),
        Err(indexed_device::Error::Device(Error::Index {
            operand: "values",
            ..
        }))
    ));
    assert!(
        Plan::from_source(
            corpus::HIST
                .replace("wrapping_add(out[ix[i]],a[i])", "out[ix[i]]+a[i]")
                .as_bytes(),
            "work"
        )
        .is_err()
    );
    let empty = corpus::frontier_source(0, 0);
    assert!(
        Plan::from_source(empty.as_bytes(), "work")
            .unwrap()
            .bind(&[], &[], Some(0), caps())
            .is_err()
    );
    let s = corpus::frontier_source(5, 65);
    assert!(
        Plan::from_source(s.as_bytes(), "work")
            .unwrap()
            .bind(&[0; 25], &[], Some(0), caps())
            .is_err()
    );
    let s = corpus::frontier_source(5, 5);
    let p = Plan::from_source(s.as_bytes(), "work").unwrap();
    assert!(p.bind(&[0; 24], &[], Some(0), caps()).is_err());
    assert!(p.bind(&[0; 25], &[], Some(5), caps()).is_err());
}

#[test]
fn gather_scatter_frontier_physical_verifier_and_capacity_mutants() {
    let cases = corpus::cases(caps());
    for (slot, mutation) in [
        (0, 0),
        (0, 1),
        (0, 2),
        (0, 3),
        (0, 4),
        (7, 5),
        (12, 6),
        (12, 7),
        (12, 8),
    ] {
        let mut changed = cases[slot].bound.plan().ir().clone();
        match mutation {
            0 => changed.output.id = changed.values.id,
            1 => changed.values.layout = invar_device::Layout::Unknown,
            2 => changed.output.address = invar_device::AddressSpace::Host,
            3 => changed.output.elements += 1,
            4 => changed.values.elements = ir::MAX_ELEMENTS + 1,
            5 => changed.law = Some("checked-add".into()),
            6 => {
                changed.operation = Operation::Frontier {
                    vertices: 5,
                    rounds: 65,
                    capacity: 5,
                }
            }
            7 => {
                changed.operation = Operation::Frontier {
                    vertices: 5,
                    rounds: 5,
                    capacity: 4,
                }
            }
            _ => changed.values.elements = 24,
        };
        assert!(
            ir::verify(changed.clone(), &changed, caps()).is_err(),
            "intrinsic physical mutation {mutation}"
        );
    }
    let p = cases[12].bound.plan();
    assert!(ir::decode(p, vec![0; p.ir().output.elements as usize], &[6, 0]).is_err());
    let mut words = vec![0; p.ir().output.elements as usize];
    words[0] = 2;
    assert!(ir::decode(p, words, &[0, 0]).is_err());
    assert!(
        ir::verify(
            cases[0].bound.plan().ir().clone(),
            cases[0].bound.plan().ir(),
            Capabilities::Cuda {
                major: 12,
                minor: 9,
                sm: 65
            }
        )
        .is_err()
    );
    assert_eq!(corpus::refusals(caps()), 7);
}

#[test]
fn gather_scatter_frontier_transfer_accounting() {
    // Five-vertex/five-round chain: 25 adjacency words, one index dummy,
    // 30 output words initialized and read back, 15 worklist/mask words,
    // two control words initialized and read back: 105 words on CUDA.
    let cuda = corpus::cases(caps());
    assert_eq!(cuda[12].bound.transfer_bytes(), 420);
    // Metal additionally copies ten round scalars, one per completed phase.
    let metal = corpus::cases(Capabilities::Metal {
        apple_family: 9,
        msl_major: 3,
        msl_minor: 1,
    });
    assert_eq!(metal[12].bound.transfer_bytes(), 460);
}

#[test]
fn gather_scatter_frontier_caps_2_28_verifier_and_plan() {
    for target in [
        caps(),
        Capabilities::Metal {
            apple_family: 9,
            msl_major: 3,
            msl_minor: 1,
        },
    ] {
        let cases = corpus::cases(target);
        for (slot, op) in [
            (0, Operation::Gather),
            (5, Operation::InjectiveScatter),
            (7, Operation::Histogram),
        ] {
            let mut large = cases[slot].bound.plan().ir().clone();
            large.operation = op;
            large.values.elements = 1 << 28;
            large.indices.elements = 1 << 28;
            large.output.elements = 1 << 28;
            let p = ir::verify(large.clone(), &large, target).unwrap();
            assert_eq!(ir::launch_dimensions(&p), (65_535, 256));
            let emitted = match target {
                Capabilities::Cuda { .. } => ir::emit_cuda(&p),
                _ => ir::emit_msl(&p),
            }
            .unwrap();
            assert!(emitted.contains("i<268435456u;i+=16776960u"));
            for buffer in 0..3 {
                let mut above = large.clone();
                match buffer {
                    0 => above.values.elements += 1,
                    1 => above.indices.elements += 1,
                    _ => above.output.elements += 1,
                }
                assert_eq!(
                    ir::verify(above.clone(), &above, target).unwrap_err(),
                    Error::Limit
                );
            }
        }
        let n = 16_384u32;
        let mut frontier = cases[12].bound.plan().ir().clone();
        frontier.operation = Operation::Frontier {
            vertices: n,
            rounds: 16,
            capacity: n,
        };
        frontier.values.elements = n.checked_mul(n).unwrap();
        frontier.output.elements = n.checked_mul(17).unwrap();
        let p = ir::verify(frontier.clone(), &frontier, target).unwrap();
        assert_eq!(p.ir().values.elements, 1 << 28);
        assert_eq!(p.ir().output.elements, 278_528);
        assert_eq!(ir::launch_dimensions(&p), (16_384, 64));
        assert_eq!(ir::phase_launch_dimensions(&p, 1), (256, 64));
        assert_eq!(ir::schedule(&p).len(), 32);
        for (vertices, rounds) in [(n + 1, 16), (n, 17), (u32::MAX, u32::MAX)] {
            let mut above = frontier.clone();
            above.operation = Operation::Frontier {
                vertices,
                rounds,
                capacity: vertices,
            };
            assert_eq!(
                ir::verify(above.clone(), &above, target).unwrap_err(),
                Error::Limit
            );
            let source = corpus::frontier_source(vertices, rounds);
            let plan = Plan::from_source(source.as_bytes(), "work").unwrap();
            assert!(matches!(
                plan.bind(&[], &[], Some(0), target),
                Err(indexed_device::Error::Device(Error::Limit))
            ));
        }
        let plan = Plan::from_source(corpus::frontier_source(n, 16).as_bytes(), "work").unwrap();
        // Reaching shape validation, rather than a cap refusal, proves this
        // source-derived plan admits the maximum without a one-GiB CPU fixture.
        assert!(matches!(
            plan.bind(&[], &[], Some(0), target),
            Err(indexed_device::Error::Device(Error::Shape))
        ));
    }
    let n = u64::from(ir::MAX_VERTICES);
    assert_eq!((n - 1) * n + (n - 1), (1 << 28) - 1);
    assert!(n * n + (2 * 16 + 1) * n <= u64::from(u32::MAX));
    assert!(u64::from(ir::MAX_ELEMENTS) + 65_535 * 256 <= u64::from(u32::MAX));
}

#[test]
fn gather_scatter_frontier_independent_rust_reference() {
    for case in corpus::cases(caps()) {
        assert_eq!(
            corpus::sequential_reference(&case.bound),
            case.expected,
            "{}",
            case.name
        );
    }
    // Above the source observer cap, use the same device API and independent
    // Rust oracle as the large hardware tests, with bounded CPU allocations.
    for family in ["gather", "scatter", "histogram", "frontier"] {
        let bound = corpus::large_bound(family, 1 << 16, caps());
        let expected = corpus::sequential_reference(&bound);
        let bytes = bound.buffer_bytes();
        assert!(bytes.iter().all(|&n| n <= 1 << 30));
        let words = expected
            .values
            .iter()
            .chain(expected.frontier_masks.iter().flatten())
            .copied()
            .collect();
        assert_eq!(ir::decode(bound.plan(), words, &[0, 0]).unwrap(), expected);
        if family == "frontier" {
            assert_eq!(expected.values.iter().sum::<u32>(), 17);
            assert_eq!(expected.frontier_masks.len(), 16);
        }
    }
}
