//! The `strict-f64-v1` oracle: the strict-f32-v1 rules applied to IEEE 754
//! binary64. Every finite operation rounds once to nearest-even, there is no
//! implicit contraction, and NaNs form one unobservable class whose canonical
//! encoding is `0x7ff8000000000000`.

use super::{
    Comparison, DecimalError, F32Bits, NumericTrap,
    ieee::{self, Integral},
};

const F64: ieee::Format = ieee::BINARY64;
const F32: ieee::Format = ieee::BINARY32;

/// Binary64 oracle encoding. Derived equality compares encodings;
/// `strict_eq` compares engine outcomes and `compare` implements predicates.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct F64Bits(u64);

impl F64Bits {
    pub const POS_ZERO: Self = Self(0);
    pub const NEG_ZERO: Self = Self(1 << 63);
    pub const INFINITY: Self = Self(0x7ff0_0000_0000_0000);
    pub const NEG_INFINITY: Self = Self(0xfff0_0000_0000_0000);
    /// The canonical representative of the unobservable NaN class.
    pub const NAN: Self = Self(0x7ff8_0000_0000_0000);

    pub const fn from_bits(bits: u64) -> Self {
        Self(bits)
    }

    pub const fn to_bits(self) -> u64 {
        self.0
    }

    /// Round a positive `digits.digits` spelling once, exactly, to binary64.
    /// An `f64` suffix is accepted; any other suffix is malformed.
    pub fn from_decimal(raw: &str) -> Result<Self, DecimalError> {
        match super::decimal_digits(raw) {
            (digits, None | Some("f64")) => F64.parse_decimal(digits).map(Self),
            _ => Err(DecimalError::Malformed),
        }
    }

    pub const fn is_nan(self) -> bool {
        F64.is_nan(self.0)
    }

    pub const fn is_infinite(self) -> bool {
        F64.is_infinite(self.0)
    }

    pub const fn is_zero(self) -> bool {
        F64.is_zero(self.0)
    }

    pub const fn is_negative(self) -> bool {
        F64.is_negative(self.0)
    }

    /// Strict engine agreement: identical non-NaN encodings, all NaNs equal.
    pub const fn strict_eq(self, other: Self) -> bool {
        self.0 == other.0 || (self.is_nan() && other.is_nan())
    }

    /// IEEE predicates: signed zeros equal and NaNs unordered.
    pub fn compare(self, predicate: Comparison, other: Self) -> bool {
        F64.compare(self.0, predicate, other.0)
    }

    pub const fn negate(self) -> Self {
        Self(self.0 ^ F64.sign())
    }

    #[allow(
        clippy::should_implement_trait,
        reason = "Named oracle calls make each strict rounding site explicit"
    )]
    pub fn add(self, other: Self) -> Self {
        Self(F64.add(self.0, other.0))
    }

    #[allow(
        clippy::should_implement_trait,
        reason = "Named oracle calls make each strict rounding site explicit"
    )]
    pub fn sub(self, other: Self) -> Self {
        self.add(other.negate())
    }

    #[allow(
        clippy::should_implement_trait,
        reason = "Named oracle calls make each strict rounding site explicit"
    )]
    pub fn mul(self, other: Self) -> Self {
        Self(F64.mul(self.0, other.0))
    }

    #[allow(
        clippy::should_implement_trait,
        reason = "Named oracle calls make each strict rounding site explicit"
    )]
    pub fn div(self, other: Self) -> Self {
        Self(F64.div(self.0, other.0))
    }

    /// Explicit fused multiply-add, rounded once.
    pub fn fma(self, multiplier: Self, addend: Self) -> Self {
        Self(F64.fma(self.0, multiplier.0, addend.0))
    }

    /// Correctly rounded square root; -0 is preserved and negatives are NaN.
    pub fn sqrt(self) -> Self {
        Self(F64.sqrt(self.0))
    }

    /// Exact: every i32 is a binary64 value.
    pub fn from_i32(value: i32) -> Self {
        Self(F64.encode_integer(i64::from(value)))
    }

    /// Exact: every u32 is a binary64 value.
    pub fn from_u32(value: u32) -> Self {
        Self(F64.encode_integer(i64::from(value)))
    }

    /// Exact widening; a NaN becomes the canonical binary64 NaN.
    pub fn from_f32(value: F32Bits) -> Self {
        let bits = u64::from(value.to_bits());
        if F32.is_nan(bits) {
            return Self::NAN;
        }
        if F32.is_infinite(bits) {
            return Self(F64.infinity(F32.is_negative(bits)));
        }
        Self(F64.round(&F32.exact(bits), F32.is_negative(bits)))
    }

    /// Narrowing rounds once to nearest-even; overflow gives an infinity and
    /// a NaN becomes the canonical binary32 NaN.
    pub fn to_f32(self) -> F32Bits {
        if self.is_nan() {
            return F32Bits::NAN;
        }
        let bits = if self.is_infinite() {
            F32.infinity(self.is_negative())
        } else {
            F32.round(&F64.exact(self.0), self.is_negative())
        };
        F32Bits::from_bits(bits as u32)
    }

    /// Truncate toward zero into `i32`; NaN, infinities and out-of-range
    /// truncations trap `INTEGER_OVERFLOW`.
    pub fn trunc_i32(self) -> Result<i32, NumericTrap> {
        F64.trunc_integer(self.0, i64::from(i32::MIN), i64::from(i32::MAX))
            .map(|value| value as i32)
    }

    /// Truncate toward zero into `u32`; NaN, infinities, negative and
    /// oversized truncations trap `INTEGER_OVERFLOW`.
    pub fn trunc_u32(self) -> Result<u32, NumericTrap> {
        F64.trunc_integer(self.0, 0, i64::from(u32::MAX))
            .map(|value| value as u32)
    }

    pub fn floor(self) -> Self {
        Self(F64.integral(self.0, Integral::Floor))
    }

    pub fn ceil(self) -> Self {
        Self(F64.integral(self.0, Integral::Ceil))
    }

    pub fn trunc(self) -> Self {
        Self(F64.integral(self.0, Integral::Trunc))
    }

    /// Nearest integer with ties away from zero.
    pub fn round(self) -> Self {
        Self(F64.integral(self.0, Integral::NearestAway))
    }

    pub fn abs(self) -> Self {
        Self(F64.abs(self.0))
    }

    pub fn copysign(self, sign: Self) -> Self {
        Self(F64.copysign(self.0, sign.0))
    }

    /// IEEE 754-2019 minimumNumber.
    pub fn min(self, other: Self) -> Self {
        Self(F64.min_max(self.0, other.0, false))
    }

    /// IEEE 754-2019 maximumNumber.
    pub fn max(self, other: Self) -> Self {
        Self(F64.min_max(self.0, other.0, true))
    }

    /// A strict source-order left fold; empty input returns the initial bits.
    pub fn ordered_sum(initial: Self, inputs: &[Self]) -> Self {
        inputs.iter().fold(initial, |acc, &value| acc.add(value))
    }
}

impl F32Bits {
    pub fn floor(self) -> Self {
        Self::from_bits(F32.integral(u64::from(self.to_bits()), Integral::Floor) as u32)
    }

    pub fn ceil(self) -> Self {
        Self::from_bits(F32.integral(u64::from(self.to_bits()), Integral::Ceil) as u32)
    }

    pub fn trunc(self) -> Self {
        Self::from_bits(F32.integral(u64::from(self.to_bits()), Integral::Trunc) as u32)
    }

    /// Nearest integer with ties away from zero.
    pub fn round(self) -> Self {
        Self::from_bits(F32.integral(u64::from(self.to_bits()), Integral::NearestAway) as u32)
    }

    pub fn abs(self) -> Self {
        Self::from_bits(F32.abs(u64::from(self.to_bits())) as u32)
    }

    pub fn copysign(self, sign: Self) -> Self {
        Self::from_bits(F32.copysign(u64::from(self.to_bits()), u64::from(sign.to_bits())) as u32)
    }

    /// IEEE 754-2019 minimumNumber.
    pub fn min(self, other: Self) -> Self {
        Self::from_bits(
            F32.min_max(u64::from(self.to_bits()), u64::from(other.to_bits()), false) as u32,
        )
    }

    /// IEEE 754-2019 maximumNumber.
    pub fn max(self, other: Self) -> Self {
        Self::from_bits(
            F32.min_max(u64::from(self.to_bits()), u64::from(other.to_bits()), true) as u32,
        )
    }

    /// Exact widening to binary64.
    pub fn to_f64(self) -> F64Bits {
        F64Bits::from_f32(self)
    }
}
