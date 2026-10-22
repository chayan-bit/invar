//! strict-f64-v1 oracle: exact rational binary64 with one rounding per operation.
use invar::numeric_contract::{
    Comparison as C, F32Bits, F64Bits as F, NumericPolicy, NumericTrap, PolicyError,
};

fn d(value: f64) -> F {
    F::from_bits(value.to_bits())
}

fn same(actual: F, expected: f64, context: &str) {
    assert!(
        actual.strict_eq(d(expected)),
        "{context}: {:#018x} != {:#018x}",
        actual.to_bits(),
        expected.to_bits()
    );
}

#[test]
fn v02_f64_policy_is_versioned_beside_strict_f32() {
    let policy = NumericPolicy::STRICT_F64;
    assert_eq!(policy.id(), "strict-f64-v1");
    assert_eq!(policy.kind(), "strict-f64");
    assert_eq!(policy.version(), 1);
    assert_eq!(NumericPolicy::from_parts(1, "strict-f64"), Ok(policy));
    assert_eq!(
        NumericPolicy::from_parts(1, "strict-f32"),
        Ok(NumericPolicy::STRICT_F32)
    );
    assert_eq!(NumericPolicy::default(), NumericPolicy::STRICT_F32);
    // Compiled-net receipts digest this spelling; strict-f32-v1's must not move.
    assert_eq!(
        format!("{:?}", NumericPolicy::STRICT_F32),
        "NumericPolicy { version: 1 }"
    );
    assert_eq!(
        format!("{policy:?}"),
        "NumericPolicy { version: 1, binary64: true }"
    );
    assert_eq!(
        NumericPolicy::from_parts(3, "strict-f64"),
        Err(PolicyError::UnsupportedVersion(3))
    );
    for kind in ["strict-f64-v1", "f64", "fast-f64"] {
        assert_eq!(
            NumericPolicy::from_parts(1, kind),
            Err(PolicyError::UnsupportedKind)
        );
    }
    assert_eq!(
        policy.to_json(),
        serde_json::json!({"version":1,"kind":"strict-f64"})
    );
}

#[test]
fn v02_f64_decimal_literals_round_once_to_nearest_even() {
    for (raw, bits) in [
        ("0.1", 0x3fb9_9999_9999_999a_u64),
        ("0.3", 0x3fd3_3333_3333_3333),
        ("1.0", 0x3ff0_0000_0000_0000),
        ("9007199254740993.0", 0x4340_0000_0000_0000),
        ("9007199254740995.0", 0x4340_0000_0000_0002),
    ] {
        assert_eq!(F::from_decimal(raw).unwrap().to_bits(), bits, "{raw}");
    }
    // The shared 256-digit literal cap (strict-f32-v1) also bounds binary64
    // spellings; extreme magnitudes are computed, not written.
    let tiny = format!("0.{}5", "0".repeat(323));
    assert_eq!(
        F::from_decimal(&tiny),
        Err(invar::numeric_contract::DecimalError::DigitLimit)
    );
    let smallest = format!("0.{}1", "0".repeat(253));
    assert_eq!(
        F::from_decimal(&smallest).unwrap().to_bits(),
        1e-254f64.to_bits()
    );
}

#[test]
fn v02_f64_arithmetic_signed_zero_nan_and_infinity_rules() {
    same(d(1.0).sub(d(1.0)), 0.0, "exact cancellation is +0");
    same(d(-0.0).add(d(-0.0)), -0.0, "-0 + -0");
    same(d(-0.0).mul(d(5.0)), -0.0, "sign of zero product");
    assert!(d(f64::INFINITY).sub(d(f64::INFINITY)).is_nan());
    assert!(d(0.0).div(d(0.0)).is_nan());
    same(d(1.0).div(d(-0.0)), f64::NEG_INFINITY, "1/-0");
    same(d(f64::MAX).add(d(f64::MAX)), f64::INFINITY, "overflow");
    let nan = F::from_bits(0xfff8_0000_0000_1234);
    assert!(nan.strict_eq(F::NAN) && !nan.compare(C::Eq, nan) && nan.compare(C::Ne, nan));
    assert!(d(-0.0).compare(C::Eq, d(0.0)) && !d(-0.0).strict_eq(d(0.0)));
    assert!(d(-1.0).compare(C::Lt, d(-0.0)) && d(f64::MIN_POSITIVE).compare(C::Gt, d(0.0)));
}

#[test]
fn v02_f64_explicit_fma_rounds_once_and_never_contracts() {
    let a = d(1.0 + f64::EPSILON);
    let b = d(1.0 - f64::EPSILON);
    same(a.mul(b).sub(d(1.0)), 0.0, "two roundings");
    same(
        a.fma(b, d(-1.0)),
        -f64::EPSILON * f64::EPSILON,
        "one rounding",
    );
    same(d(0.0).fma(d(-1.0), d(-0.0)), -0.0, "fma zero sign");
}

#[test]
fn v02_f64_sqrt_and_gradual_underflow() {
    same(d(2.0).sqrt(), std::f64::consts::SQRT_2, "sqrt 2");
    same(d(-0.0).sqrt(), -0.0, "sqrt -0");
    assert!(d(-1.0).sqrt().is_nan());
    same(
        F::from_bits(1).sqrt(),
        f64::from_bits(1).sqrt(),
        "subnormal sqrt",
    );
    same(
        d(f64::MIN_POSITIVE).div(d(4.0)),
        f64::MIN_POSITIVE / 4.0,
        "subnormal result",
    );
    same(
        F::from_bits(1).div(d(2.0)),
        0.0,
        "half of min subnormal ties to even",
    );
    same(
        F::from_bits(3).div(d(2.0)),
        f64::from_bits(2),
        "1.5 ulp ties to even",
    );
}

#[test]
fn v02_f64_conversions_round_exactly_and_trap_out_of_range() {
    same(F::from_i32(i32::MIN), -2147483648.0, "i32 min");
    same(F::from_u32(u32::MAX), 4294967295.0, "u32 max");
    assert_eq!(d(-2147483648.9).trunc_i32(), Ok(i32::MIN));
    assert_eq!(d(2147483647.9).trunc_i32(), Ok(i32::MAX));
    assert_eq!(d(-0.9).trunc_u32(), Ok(0));
    assert_eq!(d(4294967295.5).trunc_u32(), Ok(u32::MAX));
    for bad in [
        2147483648.0,
        -2147483649.0,
        f64::NAN,
        f64::INFINITY,
        f64::NEG_INFINITY,
    ] {
        assert_eq!(
            d(bad).trunc_i32(),
            Err(NumericTrap::IntegerOverflow),
            "{bad}"
        );
    }
    for bad in [-1.0, 4294967296.0, f64::NAN] {
        assert_eq!(
            d(bad).trunc_u32(),
            Err(NumericTrap::IntegerOverflow),
            "{bad}"
        );
    }
    let third = d(1.0).div(d(3.0));
    assert_eq!(third.to_f32().to_bits(), ((1.0f64 / 3.0) as f32).to_bits());
    assert_eq!(d(f64::MAX).to_f32().to_bits(), f32::INFINITY.to_bits());
    assert_eq!(F::from_bits(1).to_f32().to_bits(), 0);
    assert_eq!(d(-1e-300).to_f32().to_bits(), (-0.0f32).to_bits());
    assert!(F::NAN.to_f32().is_nan());
    let sub = F32Bits::from_bits(1);
    same(
        F::from_f32(sub),
        f64::from(f32::from_bits(1)),
        "exact widening",
    );
    assert_eq!(F::from_f32(sub).to_f32(), sub);
    assert_eq!(
        F32Bits::from_bits(0xff80_0001).to_f64().to_bits(),
        F::NAN.to_bits()
    );
}

#[test]
fn v02_f64_exact_math_rounding_sign_and_nan_rules() {
    for (x, floor, ceil, trunc, round) in [
        (2.5, 2.0, 3.0, 2.0, 3.0),
        (-2.5, -3.0, -2.0, -2.0, -3.0),
        (-0.4, -1.0, -0.0, -0.0, -0.0),
        (0.5, 0.0, 1.0, 0.0, 1.0),
        (
            4503599627370497.0,
            4503599627370497.0,
            4503599627370497.0,
            4503599627370497.0,
            4503599627370497.0,
        ),
        (f64::from_bits(1), 0.0, 1.0, 0.0, 0.0),
    ] {
        same(d(x).floor(), floor, "floor");
        same(d(x).ceil(), ceil, "ceil");
        same(d(x).trunc(), trunc, "trunc");
        same(d(x).round(), round, "round ties away");
    }
    same(d(-3.0).abs(), 3.0, "abs");
    same(d(3.0).copysign(d(-0.0)), -3.0, "copysign");
    same(
        d(-3.0).copysign(F::from_bits(0xfff8_0000_0000_0000)),
        3.0,
        "NaN sign is positive",
    );
    same(d(-0.0).min(d(0.0)), -0.0, "min zero");
    same(d(0.0).max(d(-0.0)), 0.0, "max zero");
    same(F::NAN.min(d(2.0)), 2.0, "minimumNumber ignores NaN");
    same(d(2.0).max(F::NAN), 2.0, "maximumNumber ignores NaN");
    assert!(F::NAN.max(F::NAN).is_nan());
    assert_eq!(
        F32Bits::from_bits(0x3fc0_0000).round().to_bits(),
        2.0f32.to_bits()
    );
    assert_eq!(
        F32Bits::from_bits(0xbfc0_0000).floor().to_bits(),
        (-2.0f32).to_bits()
    );
    assert_eq!(
        F32Bits::from_bits(0xff80_0001)
            .max(F32Bits::from_bits(0x3f80_0000))
            .to_bits(),
        1.0f32.to_bits()
    );
}

fn next(state: &mut u64) -> u64 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    *state
}

/// Rust's f64 operations are IEEE correctly rounded on the CI hosts; the oracle
/// must agree with them on every class, including subnormals.
#[test]
fn v02_f64_oracle_bounded_host_differential() {
    let mut state = 0x9e37_79b9_7f4a_7c15_u64;
    let specials = [
        0.0,
        -0.0,
        1.0,
        -1.0,
        f64::MIN_POSITIVE,
        f64::MAX,
        f64::INFINITY,
        f64::NAN,
    ];
    for round in 0..600 {
        let pick = |state: &mut u64| {
            let raw = next(state);
            match raw % 4 {
                0 => specials[(raw >> 8) as usize % specials.len()],
                1 => f64::from_bits(raw >> 12),
                _ => f64::from_bits(
                    (raw & 0x800f_ffff_ffff_ffff) | ((0x3c0 + (raw >> 52) % 128) << 52),
                ),
            }
        };
        let (a, b, c) = (pick(&mut state), pick(&mut state), pick(&mut state));
        let context = format!("round={round} a={a:e} b={b:e} c={c:e}");
        same(d(a).add(d(b)), a + b, &format!("add {context}"));
        same(d(a).sub(d(b)), a - b, &format!("sub {context}"));
        same(d(a).mul(d(b)), a * b, &format!("mul {context}"));
        same(d(a).div(d(b)), a / b, &format!("div {context}"));
        same(
            d(a).fma(d(b), d(c)),
            a.mul_add(b, c),
            &format!("fma {context}"),
        );
        same(
            d(a.abs()).sqrt(),
            a.abs().sqrt(),
            &format!("sqrt {context}"),
        );
        same(d(a).floor(), a.floor(), &format!("floor {context}"));
        same(d(a).round(), a.round(), &format!("round {context}"));
        assert_eq!(
            d(a).to_f32().to_bits(),
            if a.is_nan() {
                F32Bits::NAN.to_bits()
            } else {
                (a as f32).to_bits()
            },
            "narrow {context}"
        );
        for predicate in [C::Eq, C::Ne, C::Lt, C::Le, C::Gt, C::Ge] {
            let host = match predicate {
                C::Eq => a == b,
                C::Ne => a != b,
                C::Lt => a < b,
                C::Le => a <= b,
                C::Gt => a > b,
                C::Ge => a >= b,
            };
            assert_eq!(
                d(a).compare(predicate, d(b)),
                host,
                "{predicate:?} {context}"
            );
        }
    }
}
