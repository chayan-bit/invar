use super::spatial_oracle_support as oracle;

use invar::numeric_contract::F32Bits as F;
use oracle::{
    Cloud, Error, cell_index, grid_excluded, knn, output_slots, radius_neighbors, squared_distance,
    validate_grid,
};

fn f(value: f32) -> F {
    F::from_bits(value.to_bits())
}

fn cloud(points: &[[F; 3]]) -> [Vec<F>; 3] {
    std::array::from_fn(|axis| points.iter().map(|point| point[axis]).collect())
}

fn view(axes: &[Vec<F>; 3]) -> Cloud<'_> {
    Cloud::new(&axes[0], &axes[1], &axes[2]).unwrap()
}

fn origin() -> [Vec<F>; 3] {
    cloud(&[[F::POS_ZERO; 3]])
}

#[test]
fn strict_rounding_rejects_fma_reassociation_and_flush_to_zero_mutations() {
    let zero = [F::POS_ZERO; 3];
    let x = F::from_bits(0x3f80_0001);
    let y = F::from_bits(0x3980_0000); // 2^-12, square is half an ulp at one.
    let strict = squared_distance([x, y, f(0.0)], zero);
    assert_eq!(strict.to_bits(), 0x3f80_0002);
    assert_eq!(x.fma(x, y.mul(y)).to_bits(), 0x3f80_0003);

    let strict = squared_distance([f(4096.0), f(1.0), f(1.0)], zero);
    assert_eq!(strict.to_bits(), 0x4b80_0000);
    assert_ne!(strict, f(4096.0).mul(f(4096.0)).add(f(1.0).add(f(1.0))));

    let tiny = F::from_bits(0x1a80_0000); // 2^-74, square is a subnormal.
    assert_eq!(squared_distance([tiny, f(0.0), f(0.0)], zero).to_bits(), 2);
    assert_ne!(squared_distance([tiny, f(0.0), f(0.0)], zero), F::POS_ZERO);
}

#[test]
fn radius_true_counts_survive_truncation_and_zero_stride() {
    let points = cloud(&[
        [f(2.0), f(0.0), f(0.0)],
        [f(-1.0), f(0.0), f(0.0)],
        [f(1.0), f(0.0), f(0.0)],
        [f(0.0), f(0.0), f(0.0)],
    ]);
    let queries = origin();
    let result = radius_neighbors(view(&points), view(&queries), f(2.0), 2).unwrap();
    assert_eq!(result, (vec![4], vec![3, 1]));
    assert_ne!(result.0, vec![2]); // Mutation: count clipped to stride.
    assert_ne!(result.1, vec![3, 2]); // Mutation: unstable equal-distance tie.
    let zero = radius_neighbors(view(&points), view(&queries), f(2.0), 0).unwrap();
    assert_eq!(zero, (vec![4], vec![]));
}

#[test]
fn exact_knn_orders_by_distance_then_original_index_and_pads() {
    let points = cloud(&[
        [f(10.0), f(0.0), f(0.0)],
        [f(-1.0), f(0.0), f(0.0)],
        [f(1.0), f(0.0), f(0.0)],
    ]);
    let queries = cloud(&[[f(0.0); 3], [f(10.0), f(0.0), f(0.0)]]);
    let (idx, d2) = knn(view(&points), view(&queries), 4).unwrap();
    assert_eq!(idx, [1, 2, 0, u32::MAX, 0, 2, 1, u32::MAX]);
    assert_eq!(
        d2,
        [
            f(1.0),
            f(1.0),
            f(100.0),
            F::INFINITY,
            f(0.0),
            f(81.0),
            f(121.0),
            F::INFINITY
        ]
    );
    assert_eq!(
        knn(view(&points), view(&queries), 0).unwrap(),
        (vec![], vec![])
    );
}

#[test]
fn duplicate_identical_empty_and_signed_zero_clouds() {
    let points = cloud(&[[f(1.0); 3]; 5]);
    let queries = cloud(&[[f(1.0); 3]; 2]);
    assert_eq!(
        radius_neighbors(view(&points), view(&queries), F::NEG_ZERO, 3).unwrap(),
        (vec![5, 5], vec![0, 1, 2, 0, 1, 2])
    );
    let empty = cloud(&[]);
    assert_eq!(
        knn(view(&empty), view(&queries), 2).unwrap(),
        (vec![u32::MAX; 4], vec![F::INFINITY; 4])
    );
    assert_eq!(
        radius_neighbors(view(&empty), view(&queries), f(0.0), 2).unwrap(),
        (vec![0, 0], vec![u32::MAX; 4])
    );
    assert_eq!(
        knn(view(&points), view(&empty), 8).unwrap(),
        (vec![], vec![])
    );
    assert_eq!(
        radius_neighbors(view(&points), view(&empty), f(1.0), 8).unwrap(),
        (vec![], vec![])
    );
    let points = cloud(&[[F::NEG_ZERO; 3], [F::POS_ZERO; 3]]);
    assert_eq!(
        knn(view(&points), view(&origin()), 2).unwrap(),
        (vec![0, 1], vec![F::POS_ZERO; 2])
    );
}

#[test]
fn squared_underflow_and_overflow_are_observable_query_semantics() {
    let tiny = F::from_bits(0x1780_0000); // 2^-80, square rounds to zero.
    let points = cloud(&[[tiny, f(0.0), f(0.0)]]);
    assert_eq!(
        radius_neighbors(view(&points), view(&origin()), f(0.0), 1).unwrap(),
        (vec![1], vec![0])
    );
    let largest = F::from_bits(0x7f7f_ffff);
    let points = cloud(&[[largest; 3], [largest.negate(); 3]]);
    let queries = origin();
    assert_eq!(
        knn(view(&points), view(&queries), 3).unwrap(),
        (vec![0, 1, u32::MAX], vec![F::INFINITY; 3])
    );
    assert_eq!(
        radius_neighbors(view(&points), view(&queries), largest, 1).unwrap(),
        (vec![2], vec![0])
    );
}

#[test]
fn rulings_filter_nonfinite_points_queries_and_preserve_original_indices() {
    assert_eq!(
        Cloud::new(&[f(0.0)], &[], &[]).err(),
        Some(Error::IndexOutOfBounds)
    );
    let points = cloud(&[
        [F::NAN, f(0.0), f(0.0)],
        [f(1.0); 3],
        [f(0.0), F::INFINITY, f(0.0)],
        [f(1.0); 3],
    ]);
    let queries = cloud(&[[f(1.0); 3], [F::NEG_INFINITY; 3], [F::NAN; 3]]);
    assert_eq!(grid_excluded(view(&points)), 2);
    assert_eq!(
        radius_neighbors(view(&points), view(&queries), F::INFINITY, 3).unwrap(),
        (
            vec![2, 0, 0],
            vec![
                1,
                3,
                u32::MAX,
                u32::MAX,
                u32::MAX,
                u32::MAX,
                u32::MAX,
                u32::MAX,
                u32::MAX
            ]
        )
    );
    assert_eq!(
        knn(view(&points), view(&queries), 2).unwrap(),
        (
            vec![1, 3, u32::MAX, u32::MAX, u32::MAX, u32::MAX],
            vec![
                f(0.0),
                f(0.0),
                F::INFINITY,
                F::INFINITY,
                F::INFINITY,
                F::INFINITY
            ]
        )
    );
    assert_eq!(
        knn(view(&points), view(&queries), 0).unwrap(),
        (vec![], vec![])
    );
}

#[test]
fn rulings_radius_is_inclusive_negative_or_nan_matches_nothing() {
    let point = cloud(&[[f(1.0), f(0.0), f(0.0)]]);
    assert_eq!(
        radius_neighbors(view(&point), view(&origin()), f(1.0), 2).unwrap(),
        (vec![1], vec![0, u32::MAX])
    );
    for radius in [f(-1.0), F::NEG_INFINITY, F::NAN] {
        assert_eq!(
            radius_neighbors(view(&point), view(&origin()), radius, 2).unwrap(),
            (vec![0], vec![u32::MAX; 2])
        );
    }
    assert_eq!(
        radius_neighbors(view(&point), view(&origin()), F::INFINITY, 1).unwrap(),
        (vec![1], vec![0])
    );
}

#[test]
fn rulings_exact_cell_floor_and_integer_overflow() {
    // Rounded f32 division yields 10.0, but the exact rational quotient is below 10.
    let coord = f(1.0);
    let cell = f(0.1);
    assert_eq!(coord.div(cell).trunc_i32(), Ok(10));
    assert_eq!(cell_index(coord, cell), Ok(9));
    assert_eq!(cell_index(coord.negate(), cell), Ok(-10));
    assert_eq!(cell_index(f(-3.0), f(2.0)), Ok(-2));
    assert_eq!(cell_index(F::NEG_ZERO, f(1.0)), Ok(0));
    assert_eq!(cell_index(f(-2147483648.0), f(1.0)), Ok(i32::MIN));
    assert_eq!(cell_index(f(2147483520.0), f(1.0)), Ok(2147483520));
    for (coord, cell) in [
        (f(2147483648.0), f(1.0)),
        (f(-2147483904.0), f(1.0)),
        (f(1.0), F::from_bits(1)),
    ] {
        assert_eq!(cell_index(coord, cell), Err(Error::IntegerOverflow));
    }
    for cell in [
        f(0.0),
        F::NEG_ZERO,
        f(-1.0),
        F::NAN,
        F::INFINITY,
        F::NEG_INFINITY,
    ] {
        assert_eq!(
            validate_grid(view(&cloud(&[])), cell),
            Err(Error::IntegerOverflow)
        );
    }
    let excluded = cloud(&[[F::INFINITY; 3], [F::NAN; 3]]);
    assert_eq!(validate_grid(view(&excluded), F::from_bits(1)), Ok(()));
    assert_eq!(output_slots(2, u32::MAX), Err(Error::IntegerOverflow));
    assert_eq!(output_slots(0, u32::MAX), Ok(0));
    assert_eq!(output_slots(3, 2), Ok(6));
    assert_eq!(cell_index(F::from_bits(1), F::from_bits(1)), Ok(1));
    let overflowing = cloud(&[[f(2147483648.0), f(0.0), f(0.0)]]);
    assert_eq!(
        validate_grid(view(&overflowing), f(1.0)),
        Err(Error::IntegerOverflow)
    );
}

#[test]
fn oracle_resource_limits_are_checked_before_work() {
    let queries = cloud(&[[f(0.0); 3]; 2]);
    assert_eq!(
        knn(view(&origin()), view(&queries), u32::MAX),
        Err(Error::IntegerOverflow)
    );
    assert_eq!(
        radius_neighbors(view(&origin()), view(&queries), f(1.0), u32::MAX),
        Err(Error::IntegerOverflow)
    );
    assert_eq!(
        knn(view(&origin()), view(&origin()), u32::MAX),
        Err(Error::Limit)
    );
    assert_eq!(
        radius_neighbors(view(&origin()), view(&origin()), f(1.0), u32::MAX),
        Err(Error::Limit)
    );
    let too_large = vec![f(0.0); oracle::MAX_POINTS + 1];
    assert_eq!(
        Cloud::new(&too_large, &too_large, &too_large).err(),
        Some(Error::Limit)
    );
    let many = cloud(&[[f(0.0); 3]; 257]);
    assert_eq!(knn(view(&many), view(&many), 0), Err(Error::Limit));
    assert_eq!(
        radius_neighbors(view(&many), view(&many), f(0.0), 0),
        Err(Error::Limit)
    );
    assert_eq!(knn(view(&origin()), view(&many), 300), Err(Error::Limit));
}

fn generated(seed: u32, count: usize, clustered: bool) -> [Vec<F>; 3] {
    let mut state = seed;
    let points: Vec<_> = (0..count)
        .map(|index| {
            std::array::from_fn(|axis| {
                state = state.wrapping_mul(1664525).wrapping_add(1013904223);
                let value = ((state >> 24) as i32 - 128) as f32 / 8.0;
                f(if clustered {
                    (index % 3) as f32 + value / 64.0
                } else {
                    value + axis as f32
                })
            })
        })
        .collect();
    cloud(&points)
}

#[test]
fn deterministic_random_and_clustered_corpus_is_ready_for_engine_differentials() {
    for clustered in [false, true] {
        for seed in [0, 1, 0x5eed_1234] {
            let points = generated(seed, 32, clustered);
            let queries = generated(seed ^ 0xcafe, 9, clustered);
            let full = knn(view(&points), view(&queries), 32).unwrap();
            for k in [0, 1, 5, 32, 35] {
                let actual = knn(view(&points), view(&queries), k).unwrap();
                for q in 0..9 {
                    for slot in 0..k as usize {
                        let position = q * k as usize + slot;
                        let expected = if slot < 32 {
                            (full.0[q * 32 + slot], full.1[q * 32 + slot])
                        } else {
                            (u32::MAX, F::INFINITY)
                        };
                        assert_eq!((actual.0[position], actual.1[position]), expected);
                    }
                }
            }
            let radius = radius_neighbors(view(&points), view(&queries), f(2.0), 5).unwrap();
            for q in 0..9 {
                let count = full.1[q * 32..q * 32 + 32]
                    .iter()
                    .filter(|d2| d2.to_bits() <= f(4.0).to_bits())
                    .count();
                assert_eq!(radius.0[q] as usize, count);
                for slot in 0..5 {
                    let expected = if slot < count {
                        full.0[q * 32 + slot]
                    } else {
                        u32::MAX
                    };
                    assert_eq!(radius.1[q * 5 + slot], expected);
                }
            }
        }
    }
}
