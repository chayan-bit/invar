#![forbid(unsafe_code)]
use invar::{
    bounded_summary::{Budget, Error as BudgetError},
    typed_net::{
        BoundedFamily, BoundedScanError, Effect, Error, Limits, Net, Node, Op, Plan, Prefix, Type,
    },
};

fn limits() -> Limits {
    Limits {
        nodes: 16,
        ports: 64,
        depth: 8,
        array_len: 1_000_000,
        map_ops: 8,
    }
}
fn budget() -> Budget {
    Budget {
        max_steps: 1024,
        max_bytes: 1_000_000,
        max_composition_ops: 1_000_000,
    }
}
fn graph(family: BoundedFamily, prefix: Prefix) -> Net {
    let mut n = Net::empty(3);
    let (ty, op) = match family {
        BoundedFamily::Table(_) => (Type::Nat, Op::InputScalar(0)),
        BoundedFamily::Matrix(_) => (Type::ArrayU32, Op::InputU32(1)),
    };
    n.push(Node {
        source_id: 0,
        ty,
        effect: Effect::Total,
        inputs: vec![],
        op,
    });
    n.push(Node {
        source_id: 1,
        ty: Type::ArrayU32,
        effect: Effect::Total,
        inputs: vec![],
        op: Op::InputU32(0),
    });
    n.push(Node {
        source_id: 2,
        ty: Type::Nat,
        effect: Effect::Total,
        inputs: vec![],
        op: Op::InputScalar(1),
    });
    n.push(Node {
        source_id: 3,
        ty: if matches!(family, BoundedFamily::Table(_)) && prefix == Prefix::None {
            Type::Nat
        } else {
            Type::ArrayU32
        },
        effect: Effect::MayFail,
        inputs: vec![0, 1, 2],
        op: Op::BoundedScan {
            family,
            prefix,
            rule: family.rule(),
        },
    });
    n.seal_source();
    n
}
fn data(family: BoundedFamily, steps: u32) -> (Vec<u32>, Vec<u32>) {
    match family {
        BoundedFamily::Table(k) => (
            (0..steps)
                .flat_map(|i| (0..u32::from(k)).map(move |s| (s * (i + 1) + i + 1) % u32::from(k)))
                .collect(),
            vec![u32::from(k) - 1],
        ),
        BoundedFamily::Matrix(d) => (
            (0..steps * u32::from(d) * u32::from(d))
                .map(|i| u32::MAX.wrapping_sub(i.wrapping_mul(747796405)))
                .collect(),
            [u32::MAX, 0x80000000, 3, 11][..usize::from(d)].to_vec(),
        ),
    }
}
fn sequential(family: BoundedFamily, coefficients: &[u32], seed: &[u32]) -> Vec<Vec<u32>> {
    let width = match family {
        BoundedFamily::Table(k) => usize::from(k),
        BoundedFamily::Matrix(d) => usize::from(d) * usize::from(d),
    };
    let mut state = seed.to_vec();
    let mut states = vec![];
    for row in coefficients.chunks_exact(width) {
        state = match family {
            BoundedFamily::Table(_) => vec![row[state[0] as usize]],
            BoundedFamily::Matrix(d) => (0..usize::from(d))
                .map(|r| {
                    let sum: u128 = (0..usize::from(d))
                        .map(|c| u128::from(row[r * usize::from(d) + c]) * u128::from(state[c]))
                        .sum();
                    (sum % (1u128 << 32)) as u32
                })
                .collect(),
        };
        states.push(state.clone());
    }
    states
}
fn families() -> [BoundedFamily; 6] {
    [
        BoundedFamily::Table(1),
        BoundedFamily::Table(2),
        BoundedFamily::Table(16),
        BoundedFamily::Matrix(1),
        BoundedFamily::Matrix(2),
        BoundedFamily::Matrix(4),
    ]
}

#[test]
fn issue_m4_07_typed_bounded_scan_graph_runtime_matches_independent_ordered_actions() {
    let mut fixtures = 0;
    for family in families() {
        for prefix in [Prefix::None, Prefix::Inclusive] {
            for steps in [0, 1, 2, 3, 5, 7] {
                let mut n = graph(family, prefix);
                let proof = n.verify(limits()).unwrap();
                let plan = n.extract(&proof).unwrap();
                assert!(
                    matches!(plan,Plan::BoundedScan { family:f,prefix:p,seed,input:0,steps:1,source_identity,graph_revision:0 } if f==family && p==prefix && seed==u32::from(matches!(family,BoundedFamily::Matrix(_))) && source_identity==n.source_identity)
                );
                assert_eq!(n.rewrite_beta(limits()).unwrap(), None);
                assert_eq!(n.rewrite_map_fusion(limits()).unwrap(), None);
                assert_eq!(n.extract(&proof).unwrap(), plan);
                assert!(plan.run_array().is_err());
                assert!(n.decode(&proof).is_err());
                let term_observation = n
                    .observe(
                        &proof,
                        invar::term::Limits {
                            nodes: 16,
                            depth: 8,
                            steps: 16,
                        },
                    )
                    .unwrap();
                assert_eq!(term_observation.result, Err(invar::term::Error::Type(3)));
                let (coeff, seed) = data(family, steps);
                let expected = sequential(family, &coeff, &seed);
                let observation = plan
                    .run_bounded_scan(&[&coeff, &seed], &[seed[0], steps], budget())
                    .unwrap();
                assert_eq!(observation.family(), family);
                assert_eq!(observation.prefix(), prefix);
                assert_eq!(observation.seed(), seed);
                assert_eq!(observation.coefficients(), coeff);
                assert_eq!(observation.prefixes(), expected);
                assert_eq!(observation.final_state(), expected.last().unwrap_or(&seed));
                assert_eq!(observation.transitions().len(), steps as usize);
                fixtures += 1;
            }
        }
    }
    println!("typed_graph_runtime_independent_fixtures={fixtures}");
    assert_eq!(fixtures, 72);
}

#[test]
fn issue_m4_07_typed_bounded_scan_rejects_forged_graph_facts_and_stale_proofs() {
    let mut mutations = 0;
    for family in families() {
        for prefix in [Prefix::None, Prefix::Inclusive] {
            let n = graph(family, prefix);
            let proof = n.verify(limits()).unwrap();
            let mut cases = vec![];
            let mut mismatch = n.clone();
            mismatch.nodes[3].as_mut().unwrap().op = Op::BoundedScan {
                family,
                prefix,
                rule: match family {
                    BoundedFamily::Table(_) => invar::typed_net::MATRIX_SCAN_RULE,
                    BoundedFamily::Matrix(_) => invar::typed_net::TABLE_SCAN_RULE,
                },
            };
            cases.push(mismatch);
            let mut m = n.clone();
            m.nodes[3].as_mut().unwrap().op = Op::BoundedScan {
                family,
                prefix,
                rule: "forged",
            };
            cases.push(m);
            for bad in [
                BoundedFamily::Table(0),
                BoundedFamily::Table(17),
                BoundedFamily::Matrix(0),
                BoundedFamily::Matrix(5),
            ] {
                let mut m = n.clone();
                m.nodes[3].as_mut().unwrap().op = Op::BoundedScan {
                    family: bad,
                    prefix,
                    rule: bad.rule(),
                };
                cases.push(m);
            }
            let mut m = n.clone();
            m.nodes[3].as_mut().unwrap().op = Op::BoundedScan {
                family,
                prefix: Prefix::Exclusive,
                rule: family.rule(),
            };
            cases.push(m);
            for id in 0..4 {
                for effect in [Effect::Total, Effect::MayFail, Effect::Unknown] {
                    if effect == n.nodes[id].as_ref().unwrap().effect {
                        continue;
                    }
                    let mut m = n.clone();
                    m.nodes[id].as_mut().unwrap().effect = effect;
                    cases.push(m);
                }
            }
            for id in 0..4 {
                let mut m = n.clone();
                m.nodes[id].as_mut().unwrap().ty = Type::Bool;
                cases.push(m);
            }
            for ports in [
                vec![],
                vec![0, 1],
                vec![0, 1, 2, 0],
                vec![0, 0, 2],
                vec![0, 4, 2],
                vec![0, 1, 3],
                vec![0, 2, 1],
            ] {
                let mut m = n.clone();
                m.nodes[3].as_mut().unwrap().inputs = ports;
                cases.push(m);
            }
            if matches!(family, BoundedFamily::Table(_)) {
                let mut m = n.clone();
                m.nodes[3].as_mut().unwrap().inputs = vec![1, 0, 2];
                cases.push(m);
            }
            let mut m = n.clone();
            m.nodes[0].as_mut().unwrap().op = match family {
                BoundedFamily::Table(_) => Op::Nat(0),
                BoundedFamily::Matrix(_) => Op::ArrayU32(vec![0]),
            };
            cases.push(m);
            let mut m = n.clone();
            m.nodes[1].as_mut().unwrap().op = Op::ArrayU32(vec![]);
            cases.push(m);
            let mut m = n.clone();
            m.nodes[2].as_mut().unwrap().op = Op::Nat(3);
            cases.push(m);
            let mut m = n.clone();
            m.nodes[0].as_mut().unwrap().op = Op::Var(77);
            cases.push(m);
            let mut m = n.clone();
            m.nodes[0].as_mut().unwrap().source_id = 1;
            cases.push(m);
            let mut m = n.clone();
            m.nodes[1] = None;
            cases.push(m);
            let mut m = n.clone();
            m.revision = 1;
            cases.push(m);
            let mut m = n.clone();
            m.rule_history.push(family.rule());
            m.revision = 1;
            cases.push(m);
            let mut m = n.clone();
            m.schema_version += 1;
            cases.push(m);
            let mut m = n.clone();
            m.source_identity = 0;
            cases.push(m);
            if matches!(family, BoundedFamily::Matrix(_)) {
                let mut m = n.clone();
                m.nodes[0].as_mut().unwrap().op = Op::InputU32(0);
                cases.push(m);
            }
            for m in cases {
                assert_eq!(m.extract(&proof), Err(Error::StaleProof));
                assert!(m.verify(limits()).is_err(), "forged graph accepted: {m:?}");
                mutations += 1;
            }
            // Legal rewiring changes extracted slots; old proof cannot authorize it.
            for (id, op) in [
                (1, Op::InputU32(8)),
                (2, Op::InputScalar(9)),
                (
                    0,
                    match family {
                        BoundedFamily::Table(_) => Op::InputScalar(8),
                        BoundedFamily::Matrix(_) => Op::InputU32(8),
                    },
                ),
            ] {
                let mut m = n.clone();
                m.nodes[id].as_mut().unwrap().op = op;
                assert_eq!(m.extract(&proof), Err(Error::StaleProof));
                let changed = m.extract(&m.verify(limits()).unwrap()).unwrap();
                assert_ne!(changed, n.extract(&proof).unwrap());
                mutations += 1;
            }
            let mut m = n.clone();
            m.source_identity ^= 1;
            assert_eq!(m.extract(&proof), Err(Error::StaleProof));
            assert!(m.verify(limits()).is_ok());
            mutations += 1;
        }
    }
    println!("typed_graph_mutations_checked={mutations}");
    assert!(mutations > 400);
}

#[test]
fn issue_m4_07_typed_bounded_scan_runtime_shape_alias_overflow_and_budgets() {
    for family in families() {
        for prefix in [Prefix::None, Prefix::Inclusive] {
            let n = graph(family, prefix);
            let plan = n.extract(&n.verify(limits()).unwrap()).unwrap();
            let (coeff, seed) = data(family, 3);
            let arrays = [&coeff[..], &seed[..]];
            let scalars = [seed[0], 3];
            let actual = plan.run_bounded_scan(&arrays, &scalars, budget()).unwrap();
            let exact = Budget {
                max_steps: 3,
                max_bytes: actual.required_bytes(),
                max_composition_ops: actual.required_work(),
            };
            plan.run_bounded_scan(&arrays, &scalars, exact).unwrap();
            for (bad, error) in [
                (
                    Budget {
                        max_steps: 2,
                        ..exact
                    },
                    BudgetError::StepLimit,
                ),
                (
                    Budget {
                        max_bytes: exact.max_bytes - 1,
                        ..exact
                    },
                    BudgetError::ByteLimit,
                ),
                (
                    Budget {
                        max_composition_ops: exact.max_composition_ops - 1,
                        ..exact
                    },
                    BudgetError::CompositionLimit,
                ),
            ] {
                assert_eq!(
                    plan.run_bounded_scan(&arrays, &scalars, bad).unwrap_err(),
                    BoundedScanError::Budget(error)
                );
            }
            assert_eq!(
                plan.run_bounded_scan(&[], &scalars, budget()).unwrap_err(),
                BoundedScanError::Slot
            );
            assert_eq!(
                plan.run_bounded_scan(&arrays, &[], budget()).unwrap_err(),
                BoundedScanError::Slot
            );
            assert!(matches!(
                plan.run_bounded_scan(&[&coeff[..coeff.len() - 1], &seed], &scalars, budget()),
                Err(BoundedScanError::Shape)
            ));
            let huge = plan
                .run_bounded_scan(&[&[], &seed], &[seed[0], u32::MAX], budget())
                .unwrap_err();
            assert!(matches!(
                huge,
                BoundedScanError::Bounds | BoundedScanError::Shape
            ));
            let coefficient_width = match family {
                BoundedFamily::Table(k) => u32::from(k),
                BoundedFamily::Matrix(d) => u32::from(d) * u32::from(d),
            };
            let over_steps = invar::array_limits::MAX_ARRAY_LEN as u32 / coefficient_width + 1;
            let too_long = vec![0; (over_steps * coefficient_width) as usize];
            assert!(matches!(
                plan.run_bounded_scan(&[&too_long, &seed], &[seed[0], over_steps], budget()),
                Err(BoundedScanError::Shape)
            ));
            let zero = Budget {
                max_steps: 0,
                max_bytes: 0,
                max_composition_ops: 0,
            };
            let empty = plan
                .run_bounded_scan(&[&[], &seed], &[seed[0], 0], zero)
                .unwrap();
            assert_eq!(empty.final_state(), seed);
            assert!(empty.prefixes().is_empty());
            assert_eq!(empty.required_bytes(), 0);
            assert_eq!(empty.required_work(), 0);
            match family {
                BoundedFamily::Table(k) => {
                    assert_eq!(
                        plan.run_bounded_scan(&[&[]], &[u32::from(k), 0], zero)
                            .unwrap_err(),
                        BoundedScanError::Seed
                    );
                    let mut bad = coeff.clone();
                    bad[0] = u32::from(k);
                    assert_eq!(
                        plan.run_bounded_scan(&[&bad], &scalars, budget())
                            .unwrap_err(),
                        BoundedScanError::Shape
                    );
                }
                BoundedFamily::Matrix(d) => {
                    assert_eq!(
                        plan.run_bounded_scan(&[&[], &[]], &[0, 0], zero)
                            .unwrap_err(),
                        BoundedScanError::Seed
                    );
                    assert_eq!(
                        plan.run_bounded_scan(
                            &[&coeff, &coeff[..usize::from(d)]],
                            &scalars,
                            budget()
                        )
                        .unwrap_err(),
                        BoundedScanError::Alias
                    );
                }
            }
            let mut bad = plan.clone();
            if let Plan::BoundedScan {
                source_identity, ..
            } = &mut bad
            {
                *source_identity = 0;
            }
            assert_eq!(
                bad.run_bounded_scan(&arrays, &scalars, budget())
                    .unwrap_err(),
                BoundedScanError::Plan
            );
            let mut bad = plan.clone();
            if let Plan::BoundedScan { family, .. } = &mut bad {
                *family = BoundedFamily::Matrix(5);
            }
            assert_eq!(
                bad.run_bounded_scan(&arrays, &scalars, budget())
                    .unwrap_err(),
                BoundedScanError::Plan
            );
            let mut bad = plan.clone();
            if let Plan::BoundedScan { prefix, .. } = &mut bad {
                *prefix = Prefix::Exclusive;
            }
            assert_eq!(
                bad.run_bounded_scan(&arrays, &scalars, budget())
                    .unwrap_err(),
                BoundedScanError::Plan
            );
        }
    }
}
