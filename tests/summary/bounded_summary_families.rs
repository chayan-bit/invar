//! Issue #6 (M4-07).
#![forbid(unsafe_code)]
use invar::bounded_summary::{Budget, Error, SummaryFamily, prefixes};
use std::mem::size_of;

fn budget() -> Budget {
    Budget {
        max_steps: 1024,
        max_bytes: 1024 * 1024,
        max_composition_ops: 1_000_000,
    }
}
fn table(entries: &[u8]) -> SummaryFamily {
    SummaryFamily::table(entries).unwrap()
}
fn matrix(d: u8, entries: &[u32]) -> SummaryFamily {
    SummaryFamily::matrix(d, entries).unwrap()
}

fn table_functions(k: usize) -> Vec<Vec<u8>> {
    (0..k.pow(k as u32))
        .map(|mut code| {
            (0..k)
                .map(|_| {
                    let value = (code % k) as u8;
                    code /= k;
                    value
                })
                .collect()
        })
        .collect()
}

/// Independent ring action: bounded u128 dot product, then reduce once modulo
/// 2^32. It shares no wrapping helper or summary composition with the module.
fn ring_action(d: usize, coefficients: &[u32], state: &[u32]) -> Vec<u32> {
    (0..d)
        .map(|row| {
            let exact: u128 = (0..d)
                .map(|col| u128::from(coefficients[row * d + col]) * u128::from(state[col]))
                .sum();
            u32::try_from(exact % (1u128 << 32)).unwrap()
        })
        .collect()
}

#[test]
fn bounded_summary_families_constructor_bounds_immutability_and_ids() {
    assert_eq!(SummaryFamily::table(&[]), Err(Error::TableSize));
    assert_eq!(SummaryFamily::table(&[0; 17]), Err(Error::TableSize));
    assert_eq!(
        SummaryFamily::table(&[0, 2]),
        Err(Error::TableEntry { index: 1, value: 2 })
    );
    for d in [0, 5, u8::MAX] {
        assert_eq!(SummaryFamily::matrix(d, &[]), Err(Error::MatrixDimension));
    }
    for d in 1..=4 {
        for len in [
            0,
            usize::from(d) * usize::from(d) - 1,
            usize::from(d) * usize::from(d) + 1,
        ] {
            assert_eq!(
                SummaryFamily::matrix(d, &vec![0; len]),
                Err(Error::MatrixCount)
            );
        }
    }
    let mut entries = vec![1, 0];
    let frozen = table(&entries);
    entries[0] = 0;
    assert_eq!(frozen.apply(&[0]).unwrap(), vec![1]);
    let mut coefficients = vec![3];
    let frozen = matrix(1, &coefficients);
    coefficients[0] = 7;
    assert_eq!(frozen.apply(&[2]).unwrap(), vec![6]);
    assert_eq!(table(&[0]).family_id(), "finite-state-table-v1");
    assert_eq!(matrix(1, &[1]).family_id(), "u32-matrix-mod2p32-v1");
    assert_eq!(frozen.version(), 1);
    assert_eq!(frozen.dimension(), 1);
}

#[test]
fn bounded_summary_families_tables_exhaustive_pairs_triples_and_action() {
    let mut pairs = 0;
    let mut triples = 0;
    for k in 1..=3 {
        let functions = table_functions(k);
        for a in &functions {
            let first = table(a);
            let identity = first.identity();
            assert_eq!(identity.then(&first).unwrap(), first);
            assert_eq!(first.then(&identity).unwrap(), first);
            for b in &functions {
                let second = table(b);
                let composed = first.then(&second).unwrap();
                for state in 0..k {
                    assert_eq!(
                        composed.apply(&[state as u32]).unwrap(),
                        vec![u32::from(b[usize::from(a[state])])]
                    );
                    pairs += 1;
                }
                for c in &functions {
                    let third = table(c);
                    let left = composed.then(&third).unwrap();
                    let right = first.then(&second.then(&third).unwrap()).unwrap();
                    assert_eq!(left, right);
                    for state in 0..k {
                        assert_eq!(
                            left.apply(&[state as u32]).unwrap(),
                            vec![u32::from(c[usize::from(b[usize::from(a[state])])])]
                        );
                        triples += 1;
                    }
                }
            }
        }
    }
    assert_eq!(pairs, 2220);
    assert_eq!(triples, 59178);
    println!("exhaustive_table_pair_actions={pairs} triple_actions={triples}");
}

#[test]
fn bounded_summary_families_source_order_is_not_permutation() {
    let flip = table(&[1, 0]);
    let zero = table(&[0, 0]);
    assert_eq!(flip.then(&zero).unwrap().apply(&[0]).unwrap(), vec![0]);
    assert_eq!(zero.then(&flip).unwrap().apply(&[0]).unwrap(), vec![1]);
    let shear = matrix(2, &[1, 1, 0, 1]);
    let scale = matrix(2, &[2, 0, 0, 1]);
    assert_eq!(
        shear.then(&scale).unwrap().apply(&[2, 3]).unwrap(),
        vec![10, 3]
    );
    assert_eq!(
        scale.then(&shear).unwrap().apply(&[2, 3]).unwrap(),
        vec![7, 3]
    );
    assert_eq!(
        prefixes(&[shear, scale], &[2, 3], budget()).unwrap(),
        vec![vec![5, 3], vec![10, 3]]
    );
}

#[test]
fn bounded_summary_families_k1_k2_k16_every_ordered_prefix() {
    for k in [1, 2, 16] {
        for len in [0, 1, 3, 7, 16, 17, 31, 33, 257] {
            let raw: Vec<Vec<u8>> = (0..len)
                .map(|i| (0..k).map(|s| ((s * 3 + i + 1) % k) as u8).collect())
                .collect();
            let transitions: Vec<_> = raw.iter().map(|r| table(r)).collect();
            let seed = (k - 1) as u32;
            let actual = prefixes(&transitions, &[seed], budget()).unwrap();
            let mut state = seed;
            let expected: Vec<_> = raw
                .iter()
                .map(|r| {
                    state = u32::from(r[state as usize]);
                    vec![state]
                })
                .collect();
            assert_eq!(actual, expected, "k={k},n={len}");
            if len > 0 {
                assert_eq!(
                    transitions[0].identity().apply(&[seed]).unwrap(),
                    vec![seed]
                );
            }
        }
    }
    assert_eq!(
        prefixes(
            &[],
            &[u32::MAX, 4],
            Budget {
                max_steps: 0,
                max_bytes: 0,
                max_composition_ops: 0
            }
        )
        .unwrap(),
        Vec::<Vec<u32>>::new()
    );
}

fn next(seed: &mut u64) -> u32 {
    *seed ^= *seed << 13;
    *seed ^= *seed >> 7;
    *seed ^= *seed << 17;
    (*seed >> 16) as u32
}

#[test]
fn bounded_summary_families_matrices_ring_identity_associativity_and_action() {
    let mut rng = 0x9e3779b97f4a7c15u64;
    for d in 1..=4 {
        let width = usize::from(d);
        for case in 0..128 {
            let a: Vec<_> = (0..width * width).map(|_| next(&mut rng)).collect();
            let b: Vec<_> = (0..width * width).map(|_| next(&mut rng)).collect();
            let c: Vec<_> = (0..width * width).map(|_| next(&mut rng)).collect();
            let state: Vec<_> = (0..width).map(|_| next(&mut rng)).collect();
            let (first, second, third) = (matrix(d, &a), matrix(d, &b), matrix(d, &c));
            let identity = first.identity();
            assert_eq!(identity.apply(&state).unwrap(), state);
            assert_eq!(first.then(&identity).unwrap(), first);
            assert_eq!(identity.then(&first).unwrap(), first);
            let composed = first.then(&second).unwrap();
            assert_eq!(
                composed.then(&third).unwrap(),
                first.then(&second.then(&third).unwrap()).unwrap(),
                "d={d},case={case}"
            );
            let expected = ring_action(
                width,
                &c,
                &ring_action(width, &b, &ring_action(width, &a, &state)),
            );
            assert_eq!(
                composed.then(&third).unwrap().apply(&state).unwrap(),
                expected
            );
        }
    }
    // Arithmetic crosses both multiplication and addition overflow boundaries.
    assert_eq!(
        matrix(2, &[u32::MAX, u32::MAX, u32::MAX, u32::MAX])
            .apply(&[u32::MAX, u32::MAX])
            .unwrap(),
        vec![2, 2]
    );
}

#[test]
fn bounded_summary_families_d1_d2_d4_prefixes_match_independent_ring() {
    let mut rng = 0x6a09e667f3bcc909u64;
    for d in [1, 2, 4] {
        for len in [0, 1, 3, 7, 16, 17, 31, 33, 257] {
            let width = usize::from(d);
            let raw: Vec<Vec<u32>> = (0..len)
                .map(|_| (0..width * width).map(|_| next(&mut rng)).collect())
                .collect();
            let transitions: Vec<_> = raw.iter().map(|r| matrix(d, r)).collect();
            let seed: Vec<_> = (0..width).map(|_| next(&mut rng)).collect();
            let actual = prefixes(&transitions, &seed, budget()).unwrap();
            let mut state = seed;
            let expected: Vec<_> = raw
                .iter()
                .map(|r| {
                    state = ring_action(width, r, &state);
                    state.clone()
                })
                .collect();
            assert_eq!(actual, expected, "d={d},n={len}");
        }
    }
}

#[test]
fn bounded_summary_families_reject_shape_family_and_state_errors() {
    let small = table(&[0]);
    let table2 = table(&[0, 1]);
    let mat1 = matrix(1, &[1]);
    let mat2 = matrix(2, &[1, 0, 0, 1]);
    assert_eq!(small.then(&mat1), Err(Error::FamilyMismatch));
    assert_eq!(small.then(&table2), Err(Error::ShapeMismatch));
    assert_eq!(mat1.then(&mat2), Err(Error::ShapeMismatch));
    assert_eq!(table2.apply(&[2]), Err(Error::StateValue));
    assert_eq!(table2.apply(&[u32::MAX]), Err(Error::StateValue));
    assert_eq!(table2.apply(&[]), Err(Error::StateShape));
    assert_eq!(table2.apply(&[0, 1]), Err(Error::StateShape));
    assert_eq!(mat2.apply(&[0]), Err(Error::StateShape));
    assert_eq!(mat2.apply(&[0, 1, 2]), Err(Error::StateShape));
    assert_eq!(
        prefixes(&[small, mat1], &[0], budget()),
        Err(Error::FamilyMismatch)
    );
    assert_eq!(
        prefixes(&[small, table2], &[0], budget()),
        Err(Error::ShapeMismatch)
    );
    assert_eq!(prefixes(&[table2], &[2], budget()), Err(Error::StateValue));
}

/// Independently calculate documented logical work/storage requirements.
fn requirements(n: usize, width: usize, composition: u64, action: u64) -> (usize, u64) {
    let p = n.next_power_of_two();
    let combines = 2 * (p - 1) + n;
    (
        (p + 2) * size_of::<SummaryFamily>() + n * (size_of::<Vec<u32>>() + width * 4),
        combines as u64 * composition + n as u64 * action,
    )
}

#[test]
fn bounded_summary_families_every_budget_is_admitted_before_work() {
    let mut fixtures = Vec::new();
    for k in [1, 2, 16] {
        let entries: Vec<u8> = (0..k).map(|i| ((i + 1) % k) as u8).collect();
        fixtures.push((table(&entries), vec![(k - 1) as u32], 3 * k as u64, 2));
    }
    for d in [1, 2, 4] {
        let width = usize::from(d);
        fixtures.push((
            matrix(d, &vec![u32::MAX; width * width]),
            vec![3; width],
            4 * u64::from(d).pow(3) + u64::from(d).pow(2),
            4 * u64::from(d).pow(2) + u64::from(d),
        ));
    }
    for (family, seed, composition, action) in fixtures {
        for n in [1, 7, 17] {
            let transitions = vec![family; n];
            let (bytes, ops) = requirements(n, seed.len(), composition, action);
            let exact = Budget {
                max_steps: n,
                max_bytes: bytes,
                max_composition_ops: ops,
            };
            let expected = prefixes(&transitions, &seed, exact).unwrap();
            assert_eq!(expected.len(), n);
            assert_eq!(
                prefixes(
                    &transitions,
                    &seed,
                    Budget {
                        max_steps: n - 1,
                        ..exact
                    }
                ),
                Err(Error::StepLimit)
            );
            assert_eq!(
                prefixes(
                    &transitions,
                    &seed,
                    Budget {
                        max_bytes: bytes - 1,
                        ..exact
                    }
                ),
                Err(Error::ByteLimit)
            );
            assert_eq!(
                prefixes(
                    &transitions,
                    &seed,
                    Budget {
                        max_composition_ops: ops - 1,
                        ..exact
                    }
                ),
                Err(Error::CompositionLimit)
            );
            println!(
                "family={} dimension={} n={n} admitted_bytes={bytes} admitted_work={ops}",
                family.family_id(),
                family.dimension()
            );
        }
    }
}
