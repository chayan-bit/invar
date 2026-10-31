//! Issue #1 (M4-01).
#![forbid(unsafe_code)]

use invar::numeric_contract::{DecimalError, F32Bits};
use num_bigint::BigInt;

#[test]
fn decimal_literals_exact_bits_and_ties() {
    for (raw, bits) in [
        ("0.0", 0x0000_0000),
        ("00.000", 0x0000_0000),
        ("1.0", 0x3f80_0000),
        ("0.1", 0x3dcc_cccd),
        ("0.2", 0x3e4c_cccd),
        ("0.3", 0x3e99_999a),
        ("1.000000059604644775390625", 0x3f80_0000),
        ("1.000000178813934326171875", 0x3f80_0002),
        ("340282346638528859811704183484516925440.0", 0x7f7f_ffff),
        ("340282356779733661637539395458142568448.0", 0x7f80_0000),
    ] {
        assert_eq!(F32Bits::from_decimal(raw).unwrap().to_bits(), bits, "{raw}");
    }
    assert_eq!(
        F32Bits::from_decimal("0.0").unwrap().negate(),
        F32Bits::NEG_ZERO
    );
}

#[test]
fn decimal_literals_exact_subnormal_boundaries() {
    // 2^-n = 5^n / 10^n gives a decimal spelling independently of FP parsing.
    for (power, bits) in [(149, 1), (150, 0)] {
        let digits = BigInt::from(5).pow(power).to_string();
        let raw = format!("0.{digits:0>width$}", width = power as usize);
        assert_eq!(F32Bits::from_decimal(&raw).unwrap().to_bits(), bits);
    }
    let digits = (BigInt::from(3) * BigInt::from(5).pow(150)).to_string();
    let raw = format!("0.{digits:0>150}");
    assert_eq!(F32Bits::from_decimal(&raw).unwrap().to_bits(), 2);
}

#[test]
fn decimal_literals_reject_malformed_and_bound_work() {
    for raw in [
        "", "1", ".1", "1.", "1.2.3", "-0.0", "+1.0", "1.0f64", "1.0f", "1.0e1", "NaN", "∞.0",
    ] {
        assert_eq!(
            F32Bits::from_decimal(raw),
            Err(DecimalError::Malformed),
            "{raw}"
        );
    }
    // strict-f64-v1 lets a literal carry its own type's suffix (v0.2).
    assert_eq!(
        F32Bits::from_decimal("1.5f32"),
        F32Bits::from_decimal("1.5")
    );
    let accepted = format!("{}.0", "9".repeat(255));
    assert_eq!(F32Bits::from_decimal(&accepted), Ok(F32Bits::INFINITY));
    let rejected = format!("{}.0", "9".repeat(256));
    assert_eq!(
        F32Bits::from_decimal(&rejected),
        Err(DecimalError::DigitLimit)
    );
}
