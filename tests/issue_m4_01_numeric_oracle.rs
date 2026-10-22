use invar::numeric_contract::{
    Comparison as C, F32Bits as F, NumericPolicy, NumericTrap, PolicyError,
};

fn bits(value: u32) -> F {
    F::from_bits(value)
}

fn exact(actual: F, expected: u32) {
    assert_eq!(actual.to_bits(), expected);
}

#[test]
fn issue_m4_01_numeric_oracle_policy_version_fixture_and_mutations() {
    let policy = NumericPolicy::default();
    assert_eq!(policy.id(), "strict-f32-v1");
    assert_eq!(policy.version(), 1);
    assert_eq!(policy.kind(), "strict-f32");
    let fixture = serde_json::json!({"version":1,"kind":"strict-f32"});
    assert_eq!(policy.to_json(), fixture);
    assert_eq!(
        policy.to_json().to_string(),
        "{\"kind\":\"strict-f32\",\"version\":1}"
    );
    assert_eq!(NumericPolicy::from_parts(1, "strict-f32"), Ok(policy));
    for version in [0, 3, u32::MAX] {
        assert_eq!(
            NumericPolicy::from_parts(version, "strict-f32"),
            Err(PolicyError::UnsupportedVersion(version))
        );
    }
    for kind in ["", "fast", "reassoc_add", "tc-f16-f32", "strict-f32-v1"] {
        assert_eq!(
            NumericPolicy::from_parts(1, kind),
            Err(PolicyError::UnsupportedKind)
        );
    }
}

#[test]
fn issue_m4_01_numeric_oracle_nan_class_is_not_source_equality() {
    let nans = [
        0x7f80_0001,
        0x7fc0_0000,
        0x7fff_ffff,
        0xff80_0001,
        0xffc0_1234,
    ];
    for a in nans {
        for b in nans {
            assert!(bits(a).strict_eq(bits(b)));
            assert!(!bits(a).compare(C::Eq, bits(b)));
            assert!(bits(a).compare(C::Ne, bits(b)));
        }
        assert!(bits(a).is_nan());
        assert!(!bits(a).strict_eq(F::INFINITY));
        for other in [
            F::NEG_INFINITY,
            F::NEG_ZERO,
            F::POS_ZERO,
            bits(1),
            F::INFINITY,
        ] {
            for predicate in [C::Eq, C::Lt, C::Le, C::Gt, C::Ge] {
                assert!(!bits(a).compare(predicate, other));
                assert!(!other.compare(predicate, bits(a)));
            }
            assert!(bits(a).compare(C::Ne, other));
            assert!(bits(a).add(other).is_nan());
            assert!(other.mul(bits(a)).is_nan());
            assert!(other.div(bits(a)).is_nan());
            assert!(other.fma(bits(0x3f80_0000), bits(a)).is_nan());
        }
    }
    assert!(!F::INFINITY.is_nan());
    assert!(!F::POS_ZERO.strict_eq(F::NEG_ZERO));
    assert!(F::POS_ZERO.compare(C::Eq, F::NEG_ZERO));
}

#[test]
fn issue_m4_01_numeric_oracle_signed_zeros_and_cancellation() {
    let one = bits(0x3f80_0000);
    exact(F::NEG_ZERO.add(F::NEG_ZERO), 0x8000_0000);
    exact(F::NEG_ZERO.add(F::POS_ZERO), 0);
    exact(F::POS_ZERO.sub(F::POS_ZERO), 0);
    exact(F::NEG_ZERO.sub(F::POS_ZERO), 0x8000_0000);
    exact(one.add(one.negate()), 0);
    exact(F::NEG_ZERO.mul(one), 0x8000_0000);
    exact(F::NEG_ZERO.mul(one.negate()), 0);
    exact(F::NEG_ZERO.div(one), 0x8000_0000);
    exact(one.div(F::NEG_INFINITY), 0x8000_0000);
    exact(F::NEG_ZERO.sqrt(), 0x8000_0000);
    exact(F::POS_ZERO.negate(), 0x8000_0000);
    exact(F::NEG_ZERO.fma(one, F::NEG_ZERO), 0x8000_0000);
    exact(F::NEG_ZERO.fma(one, F::POS_ZERO), 0);
    exact(F::NEG_ZERO.fma(one.negate(), F::NEG_ZERO), 0);
}

#[test]
fn issue_m4_01_numeric_oracle_nearest_even_midpoints() {
    // 2^-24 is half of the binary32 spacing at 1; even low bits win ties.
    let half_ulp = bits(0x3380_0000);
    exact(bits(0x3f80_0000).add(half_ulp), 0x3f80_0000);
    exact(bits(0x3f80_0001).add(half_ulp), 0x3f80_0002);
    exact(bits(0xbf80_0000).sub(half_ulp), 0xbf80_0000);
    exact(bits(0xbf80_0001).sub(half_ulp), 0xbf80_0002);
    exact(bits(0x3f80_0000).add(bits(0x3380_0001)), 0x3f80_0001);
    exact(bits(0x3f80_0000).add(bits(0x337f_ffff)), 0x3f80_0000);
}

#[test]
fn issue_m4_01_numeric_oracle_gradual_underflow_and_normal_boundary() {
    let half = bits(0x3f00_0000);
    exact(bits(1).add(bits(1)), 2);
    exact(bits(0x007f_ffff).add(bits(1)), 0x0080_0000);
    exact(bits(0x0080_0000).sub(bits(1)), 0x007f_ffff);
    exact(bits(0x0080_0000).mul(half), 0x0040_0000);
    exact(bits(1).mul(half), 0);
    exact(bits(0x8000_0001).mul(half), 0x8000_0000);
    exact(bits(3).mul(half), 2);
    exact(bits(0x0080_0001).mul(half), 0x0040_0000);
    exact(bits(0x0080_0003).mul(half), 0x0040_0002);
    exact(bits(0x0080_0000).mul(bits(0x3f7f_ffff)), 0x0080_0000);
}

#[test]
fn issue_m4_01_numeric_oracle_explicit_fma_differs_from_two_roundings() {
    // (1+2^-23)*(1-2^-23)-1 = -2^-46 exactly.
    let a = bits(0x3f80_0001);
    let b = bits(0x3f7f_fffe);
    let minus_one = bits(0xbf80_0000);
    exact(a.mul(b).add(minus_one), 0);
    exact(a.fma(b, minus_one), 0xa880_0000);
    // An exact fused product may exceed f32 range before cancellation.
    let max = bits(0x7f7f_ffff);
    let two = bits(0x4000_0000);
    exact(max.mul(two).add(max.negate()), 0x7f80_0000);
    exact(max.fma(two, max.negate()), 0x7f7f_ffff);
    // Half a subnormal is retained until the single fused rounding.
    let half = bits(0x3f00_0000);
    exact(bits(1).mul(half).add(bits(1)), 1);
    exact(bits(1).fma(half, bits(1)), 2);
}

#[test]
fn issue_m4_01_numeric_oracle_no_reassociation_and_empty_fold_bits() {
    let large = bits(0x4b80_0000); // 2^24
    let one = bits(0x3f80_0000);
    let negative = large.negate();
    exact(large.add(one).add(negative), 0);
    exact(large.add(one.add(negative)), 0x3f80_0000);
    exact(F::ordered_sum(F::POS_ZERO, &[large, one, negative]), 0);
    exact(
        F::ordered_sum(F::POS_ZERO, &[large, negative, one]),
        0x3f80_0000,
    );
    exact(F::ordered_sum(F::NEG_ZERO, &[]), 0x8000_0000);
    exact(F::ordered_sum(F::NEG_ZERO, &[F::NEG_ZERO]), 0x8000_0000);
}

#[test]
fn issue_m4_01_numeric_oracle_infinities_invalid_operations_and_overflow() {
    let one = bits(0x3f80_0000);
    let max = bits(0x7f7f_ffff);
    exact(max.add(max), 0x7f80_0000);
    exact(max.negate().mul(bits(0x4000_0000)), 0xff80_0000);
    // Max + 2^103 is precisely the overflow midpoint; infinity wins the tie.
    exact(max.add(bits(0x7300_0000)), 0x7f80_0000);
    exact(max.add(bits(0x72ff_ffff)), 0x7f7f_ffff);
    assert!(F::INFINITY.add(F::NEG_INFINITY).is_nan());
    assert!(F::INFINITY.mul(F::POS_ZERO).is_nan());
    assert!(F::NEG_ZERO.div(F::POS_ZERO).is_nan());
    assert!(F::INFINITY.div(F::INFINITY).is_nan());
    assert!(F::INFINITY.fma(F::POS_ZERO, one).is_nan());
    assert!(F::INFINITY.fma(one, F::NEG_INFINITY).is_nan());
    assert!(F::NEG_INFINITY.sqrt().is_nan());
    assert!(one.negate().sqrt().is_nan());
    exact(one.div(F::POS_ZERO), 0x7f80_0000);
    exact(one.div(F::NEG_ZERO), 0xff80_0000);
    exact(F::NEG_INFINITY.div(one), 0xff80_0000);
    exact(F::INFINITY.sqrt(), 0x7f80_0000);
}

#[test]
fn issue_m4_01_numeric_oracle_cpu_division_and_sqrt_rounding() {
    exact(bits(0x3f80_0000).div(bits(0x4040_0000)), 0x3eaa_aaab);
    exact(bits(0x0080_0000).div(bits(0x4000_0000)), 0x0040_0000);
    exact(bits(1).div(bits(0x4000_0000)), 0);
    exact(bits(3).div(bits(0x4000_0000)), 2);
    exact(bits(0x3f80_0000).div(bits(1)), 0x7f80_0000);
    exact(bits(0x4080_0000).sqrt(), 0x4000_0000);
    exact(bits(0x4000_0000).sqrt(), 0x3fb5_04f3);
    exact(bits(1).sqrt(), 0x1a35_04f3);
    exact(bits(0x7f7f_ffff).sqrt(), 0x5f7f_ffff);
}

#[test]
fn issue_m4_01_numeric_oracle_integer_conversion_rounding_and_traps() {
    exact(F::from_i32(0), 0);
    exact(F::from_i32(i32::MIN), 0xcf00_0000);
    exact(F::from_i32(i32::MAX), 0x4f00_0000);
    exact(F::from_u32(u32::MAX), 0x4f80_0000);
    exact(F::from_u32(16_777_217), 0x4b80_0000);
    exact(F::from_u32(16_777_219), 0x4b80_0002);
    exact(F::from_i32(-16_777_217), 0xcb80_0000);
    for (input, expected) in [
        (0, 0),
        (0x8000_0000, 0),
        (1, 0),
        (0x8000_0001, 0),
        (0x3fff_ffff, 1),
        (0xbfff_ffff, -1),
        (0x4eff_ffff, 2_147_483_520),
        (0xcf00_0000, i32::MIN),
        (0xceff_ffff, -2_147_483_520),
    ] {
        assert_eq!(bits(input).trunc_i32(), Ok(expected));
    }
    for input in [
        0x4f00_0000,
        0xcf00_0001,
        0x7f7f_ffff,
        0xff7f_ffff,
        0x7f80_0000,
        0xff80_0000,
        0x7f80_0001,
        0xffc0_1234,
    ] {
        assert_eq!(bits(input).trunc_i32(), Err(NumericTrap::IntegerOverflow));
    }
    assert_eq!(NumericTrap::IntegerOverflow.category(), "INTEGER_OVERFLOW");
}

#[test]
fn issue_m4_01_numeric_oracle_ordered_comparisons_all_sign_classes() {
    let ordered = [
        F::NEG_INFINITY,
        bits(0xff7f_ffff),
        bits(0xbf80_0000),
        bits(0x8000_0001),
        F::NEG_ZERO,
        F::POS_ZERO,
        bits(1),
        bits(0x3f80_0000),
        bits(0x7f7f_ffff),
        F::INFINITY,
    ];
    for (i, &a) in ordered.iter().enumerate() {
        for (j, &b) in ordered.iter().enumerate() {
            let same = i == j || (a.is_zero() && b.is_zero());
            assert_eq!(a.compare(C::Eq, b), same);
            assert_eq!(a.compare(C::Ne, b), !same);
            assert_eq!(a.compare(C::Lt, b), i < j && !same);
            assert_eq!(a.compare(C::Le, b), i < j || same);
            assert_eq!(a.compare(C::Gt, b), i > j && !same);
            assert_eq!(a.compare(C::Ge, b), i > j || same);
        }
    }
}

// Supplemental host differential checks do not replace hand-derived fixtures.
// Fixed xorshift seed, bounded cases, default CPU host environment only.
fn next(state: &mut u32) -> u32 {
    *state ^= *state << 13;
    *state ^= *state >> 17;
    *state ^= *state << 5;
    *state
}

fn agrees(actual: F, expected: f32, operands: &[u32], operation: &str) {
    assert!(
        actual.strict_eq(bits(expected.to_bits())),
        "{operation}: inputs={operands:x?} actual={:08x} expected={:08x}",
        actual.to_bits(),
        expected.to_bits()
    );
}

#[test]
fn issue_m4_01_numeric_oracle_bounded_host_differential() {
    let mut state = 0x4d34_0101;
    for _ in 0..512 {
        let inputs = [next(&mut state), next(&mut state), next(&mut state)];
        let [a, b, c] = inputs.map(bits);
        let [ha, hb, hc] = inputs.map(f32::from_bits);
        agrees(a.add(b), ha + hb, &inputs, "add");
        agrees(a.sub(b), ha - hb, &inputs, "sub");
        agrees(a.mul(b), ha * hb, &inputs, "mul");
        agrees(a.div(b), ha / hb, &inputs, "div");
        agrees(a.fma(b, c), ha.mul_add(hb, hc), &inputs, "fma");
        agrees(a.sqrt(), ha.sqrt(), &inputs, "sqrt");
        agrees(
            F::from_u32(inputs[0]),
            inputs[0] as f32,
            &inputs,
            "u32_to_f32",
        );
        agrees(
            F::from_i32(inputs[0] as i32),
            inputs[0] as i32 as f32,
            &inputs,
            "i32_to_f32",
        );
    }
}

#[test]
fn issue_m4_01_numeric_oracle_carry_midpoints_across_binades() {
    // For a boundary 2^(e-127), its predecessor is one lower-binade ULP
    // below it. Adding half that spacing ties to the boundary's even mantissa.
    // e=255 exercises overflow to infinity; earlier e cover normal carries.
    for exponent in 26_u32..=255 {
        let boundary = exponent << 23;
        let predecessor = bits(boundary - 1);
        let half_spacing = bits((exponent - 25) << 23);
        exact(predecessor.add(half_spacing), boundary);
        exact(
            predecessor.negate().sub(half_spacing),
            boundary | 0x8000_0000,
        );
        exact(
            predecessor.add(bits(half_spacing.to_bits() - 1)),
            boundary - 1,
        );
        exact(predecessor.add(bits(half_spacing.to_bits() + 1)), boundary);
    }
}

#[test]
fn issue_m4_01_numeric_oracle_exhaustive_small_subnormal_integer_grid() {
    // These values are integer multiples of 2^-149. Expected results are
    // integer coordinates, independent of both oracle and host FP arithmetic.
    for a in 0_u32..=32 {
        for b in 0_u32..=32 {
            exact(bits(a).add(bits(b)), a + b);
            exact(
                bits(a | 0x8000_0000).add(bits(b | 0x8000_0000)),
                (a + b) | 0x8000_0000,
            );
            let difference = if a >= b { a - b } else { (b - a) | 0x8000_0000 };
            exact(bits(a).sub(bits(b)), difference);
        }
    }
    for coordinate in 0_u32..=256 {
        // Integer division by 2 plus a tie increment when the low integer is odd.
        let low = coordinate / 2;
        let nearest_even = low + u32::from(coordinate % 2 == 1 && low % 2 == 1);
        exact(bits(coordinate).mul(bits(0x3f00_0000)), nearest_even);
        exact(
            bits(coordinate | 0x8000_0000).div(bits(0x4000_0000)),
            nearest_even | 0x8000_0000,
        );
    }
}

#[test]
fn issue_m4_01_numeric_oracle_negative_fused_underflow_and_sqrt_neighbors() {
    let half = bits(0x3f00_0000);
    let minus_subnormal = bits(0x8000_0001);
    // +0.5 quantum - 1 quantum = -0.5 quantum, which ties to negative zero.
    exact(bits(1).fma(half, minus_subnormal), 0x8000_0000);
    exact(bits(1).mul(half).add(minus_subnormal), 0x8000_0001);
    // The midpoint between the largest subnormal and the smallest normal
    // rounds to the even normal significand. Perturb the unrounded product
    // to either side; this cannot be represented by a separate f32 product.
    let largest_subnormal = bits(0x007f_ffff);
    exact(bits(1).fma(half, largest_subnormal), 0x0080_0000);
    exact(
        bits(1).fma(bits(0x3eff_ffff), largest_subnormal),
        0x007f_ffff,
    );
    exact(
        bits(1).fma(bits(0x3f00_0001), largest_subnormal),
        0x0080_0000,
    );
    exact(
        bits(0x8000_0001).fma(half, largest_subnormal.negate()),
        0x8080_0000,
    );
    // The square root of the predecessor of 4 is just below the midpoint
    // between the predecessor of 2 and 2; the successor of 4 is just below
    // the midpoint between 2 and its successor.
    exact(bits(0x407f_ffff).sqrt(), 0x3fff_ffff);
    exact(bits(0x4080_0001).sqrt(), 0x4000_0000);
    exact(bits(0x4080_0002).sqrt(), 0x4000_0001);
}

#[test]
fn issue_m4_01_numeric_oracle_initial_accumulator_is_observed() {
    let seed = bits(0x4b80_0000); // 2^24
    let one = bits(0x3f80_0000);
    // Keeping the initial accumulator in the source-order fold matters:
    // ((2^24 + 1) - 1) = 2^24 - 1, whereas 2^24 + (1 - 1) = 2^24.
    exact(F::ordered_sum(seed, &[one, one.negate()]), 0x4b7f_ffff);
    exact(seed.add(one.add(one.negate())), 0x4b80_0000);
    exact(F::ordered_sum(F::POS_ZERO, &[F::NEG_ZERO]), 0);
    exact(F::ordered_sum(F::NEG_ZERO, &[F::NEG_ZERO]), 0x8000_0000);
    // With no demanded addition, even NaN payload bits remain the seed bits.
    exact(F::ordered_sum(bits(0xff80_1234), &[]), 0xff80_1234);
}

#[test]
fn issue_m4_01_numeric_oracle_conversion_strata_use_integer_expectations() {
    for exponent in 0_u32..=255 {
        for fraction in [0, 1, 0x003f_ffff, 0x0040_0000, 0x007f_fffe, 0x007f_ffff] {
            for negative in [false, true] {
                let input =
                    bits((exponent << 23) | fraction | if negative { 0x8000_0000 } else { 0 });
                // Compute the truncated integer's magnitude directly from the
                // encoding, using only bounded u64 shifts. For binary32, values
                // outside i32 bounds are already integers, so this also checks
                // the oracle's pre-truncation range rule at both endpoints.
                let expected = if exponent == 255 || exponent > 158 {
                    Err(NumericTrap::IntegerOverflow)
                } else {
                    let magnitude = if exponent < 127 {
                        0
                    } else {
                        let significand = u64::from(fraction | 0x0080_0000);
                        if exponent >= 150 {
                            significand << (exponent - 150)
                        } else {
                            significand >> (150 - exponent)
                        }
                    };
                    let signed = if negative {
                        -(magnitude as i64)
                    } else {
                        magnitude as i64
                    };
                    i32::try_from(signed).map_err(|_| NumericTrap::IntegerOverflow)
                };
                assert_eq!(input.trunc_i32(), expected, "input={:08x}", input.to_bits());
            }
        }
    }
}

#[test]
fn issue_m4_01_numeric_oracle_special_class_matrix_matches_host() {
    let classes = [
        0,
        0x8000_0000,
        1,
        0x8000_0001,
        0x0080_0000,
        0x8080_0000,
        0x3f80_0000,
        0xbf80_0000,
        0x7f7f_ffff,
        0xff7f_ffff,
        0x7f80_0000,
        0xff80_0000,
        0x7f80_0001,
        0xffc0_1234,
    ];
    for a in classes {
        for b in classes {
            let (ha, hb) = (f32::from_bits(a), f32::from_bits(b));
            let inputs = [a, b];
            agrees(bits(a).add(bits(b)), ha + hb, &inputs, "class_add");
            agrees(bits(a).sub(bits(b)), ha - hb, &inputs, "class_sub");
            agrees(bits(a).mul(bits(b)), ha * hb, &inputs, "class_mul");
            agrees(bits(a).div(bits(b)), ha / hb, &inputs, "class_div");
            for c in classes {
                agrees(
                    bits(a).fma(bits(b), bits(c)),
                    ha.mul_add(hb, f32::from_bits(c)),
                    &[a, b, c],
                    "class_fma",
                );
            }
        }
    }
}
