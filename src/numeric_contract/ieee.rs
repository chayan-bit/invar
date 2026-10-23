//! Format-generic exact IEEE 754 binary operations on raw encodings.
//!
//! Every finite operation is computed exactly with rationals and rounded once
//! to nearest-even. No host floating-point operation, flag or environment is
//! consulted. `strict-f32-v1` and `strict-f64-v1` are two instances.

use num_bigint::{BigInt, Sign};
use num_rational::BigRational;
use std::cmp::Ordering;

use super::{Comparison, DecimalError, MAX_DECIMAL_DIGITS, NumericTrap};

/// One binary interchange format: `precision` significand bits including the
/// hidden bit and the normal exponent range `[min_exponent, max_exponent]`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Format {
    pub precision: u32,
    pub max_exponent: i32,
}

pub(crate) const BINARY32: Format = Format {
    precision: 24,
    max_exponent: 127,
};
pub(crate) const BINARY64: Format = Format {
    precision: 53,
    max_exponent: 1023,
};
/// IEEE binary16, a storage-only format (`docs/numeric.md`).
pub(crate) const BINARY16: Format = Format {
    precision: 11,
    max_exponent: 15,
};
/// bfloat16: binary32's exponent range with an 8-bit significand; storage only.
pub(crate) const BFLOAT16: Format = Format {
    precision: 8,
    max_exponent: 127,
};

impl Format {
    /// Sign, exponent field (`max_exponent + 1` is a power of two) and fraction.
    const fn width(self) -> u32 {
        2 + (self.max_exponent as u32 + 1).trailing_zeros() + self.fraction_bits()
    }
    const fn fraction_bits(self) -> u32 {
        self.precision - 1
    }
    pub(crate) const fn sign(self) -> u64 {
        1 << (self.width() - 1)
    }
    pub(crate) const fn exponent_mask(self) -> u64 {
        (self.sign() - 1) & !self.fraction_mask()
    }
    pub(crate) const fn fraction_mask(self) -> u64 {
        (1 << self.fraction_bits()) - 1
    }
    const fn min_exponent(self) -> i32 {
        1 - self.max_exponent
    }
    const fn bias(self) -> i32 {
        self.max_exponent
    }
    /// The canonical quiet NaN of the strict contracts.
    pub(crate) const fn nan(self) -> u64 {
        self.exponent_mask() | (1 << (self.fraction_bits() - 1))
    }
    pub(crate) const fn infinity(self, negative: bool) -> u64 {
        self.exponent_mask() | if negative { self.sign() } else { 0 }
    }
    pub(crate) const fn zero(self, negative: bool) -> u64 {
        if negative { self.sign() } else { 0 }
    }
    pub(crate) const fn is_nan(self, a: u64) -> bool {
        a & self.exponent_mask() == self.exponent_mask() && a & self.fraction_mask() != 0
    }
    pub(crate) const fn is_infinite(self, a: u64) -> bool {
        a & !self.sign() == self.exponent_mask()
    }
    pub(crate) const fn is_zero(self, a: u64) -> bool {
        a & !self.sign() == 0
    }
    pub(crate) const fn is_negative(self, a: u64) -> bool {
        a & self.sign() != 0
    }

    /// Exact value of a finite encoding.
    pub(crate) fn exact(self, a: u64) -> BigRational {
        let biased = ((a & self.exponent_mask()) >> self.fraction_bits()) as i32;
        let significand = (a & self.fraction_mask())
            | if biased == 0 {
                0
            } else {
                1 << self.fraction_bits()
            };
        let power = if biased == 0 {
            self.min_exponent() - self.fraction_bits() as i32
        } else {
            biased - self.bias() - self.fraction_bits() as i32
        };
        let mut numerator = BigInt::from(significand);
        if self.is_negative(a) {
            numerator = -numerator;
        }
        if power >= 0 {
            BigRational::from_integer(numerator << power as usize)
        } else {
            BigRational::new(numerator, BigInt::from(1) << (-power) as usize)
        }
    }

    /// Round an exact finite rational once to nearest-even. The zero sign is
    /// supplied because rationals have no signed zero.
    pub(crate) fn round(self, value: &BigRational, negative_zero: bool) -> u64 {
        let numerator = value.numer().magnitude();
        if numerator.bits() == 0 {
            return self.zero(negative_zero);
        }
        let negative = value.numer().sign() == Sign::Minus;
        let denominator = value.denom().magnitude();
        let mut exponent = numerator.bits() as i32 - denominator.bits() as i32;
        let below_power = if exponent >= 0 {
            numerator < &(denominator << exponent as usize)
        } else {
            &(numerator << (-exponent) as usize) < denominator
        };
        if below_power {
            exponent -= 1;
        }
        if exponent > self.max_exponent {
            return self.infinity(negative);
        }
        let fraction_bits = self.fraction_bits() as i32;
        let subnormal = exponent < self.min_exponent();
        let shift = if subnormal {
            fraction_bits - self.min_exponent()
        } else {
            fraction_bits - exponent
        };
        let (scaled_num, scaled_den) = if shift >= 0 {
            (numerator << shift as usize, denominator.clone())
        } else {
            (numerator.clone(), denominator << (-shift) as usize)
        };
        let mut quotient = &scaled_num / &scaled_den;
        let remainder = &scaled_num % &scaled_den;
        match (remainder << 1_usize).cmp(&scaled_den) {
            Ordering::Greater => quotient += 1_u32,
            Ordering::Equal if quotient.bit(0) => quotient += 1_u32,
            _ => {}
        }
        // The scaling proves quotient <= 2^precision; zero has no digits.
        let significand = quotient.to_u64_digits().first().copied().unwrap_or(0);
        let magnitude = if subnormal {
            // Rounding may produce the smallest normal encoding.
            significand
        } else if significand == 1 << self.precision {
            ((exponent + 1 + self.bias()) as u64) << fraction_bits
        } else {
            (((exponent + self.bias()) as u64) << fraction_bits)
                | (significand & self.fraction_mask())
        };
        magnitude | if negative { self.sign() } else { 0 }
    }

    pub(crate) fn parse_decimal(self, raw: &str) -> Result<u64, DecimalError> {
        if raw.len() > MAX_DECIMAL_DIGITS + 1 {
            return Err(DecimalError::DigitLimit);
        }
        let (whole, fraction) = raw.split_once('.').ok_or(DecimalError::Malformed)?;
        if whole.is_empty()
            || fraction.is_empty()
            || !whole
                .bytes()
                .chain(fraction.bytes())
                .all(|b| b.is_ascii_digit())
        {
            return Err(DecimalError::Malformed);
        }
        if whole.len() + fraction.len() > MAX_DECIMAL_DIGITS {
            return Err(DecimalError::DigitLimit);
        }
        let digits = format!("{whole}{fraction}");
        let numerator =
            BigInt::parse_bytes(digits.as_bytes(), 10).ok_or(DecimalError::Malformed)?;
        // The preceding digit cap proves this exponent is at most 256.
        let denominator = BigInt::from(10).pow(fraction.len() as u32);
        Ok(self.round(&BigRational::new(numerator, denominator), false))
    }

    pub(crate) fn compare(self, a: u64, predicate: Comparison, b: u64) -> bool {
        if self.is_nan(a) || self.is_nan(b) {
            return predicate == Comparison::Ne;
        }
        let order = if self.is_zero(a) && self.is_zero(b) {
            Ordering::Equal
        } else {
            self.ordered_key(a).cmp(&self.ordered_key(b))
        };
        match predicate {
            Comparison::Eq => order == Ordering::Equal,
            Comparison::Ne => order != Ordering::Equal,
            Comparison::Lt => order == Ordering::Less,
            Comparison::Le => order != Ordering::Greater,
            Comparison::Gt => order == Ordering::Greater,
            Comparison::Ge => order != Ordering::Less,
        }
    }

    fn ordered_key(self, a: u64) -> u64 {
        if self.is_negative(a) {
            !a & (self.sign() | (self.sign() - 1))
        } else {
            a | self.sign()
        }
    }

    pub(crate) fn add(self, a: u64, b: u64) -> u64 {
        if self.is_nan(a) || self.is_nan(b) {
            return self.nan();
        }
        if self.is_infinite(a) {
            return if self.is_infinite(b) && self.is_negative(a) != self.is_negative(b) {
                self.nan()
            } else {
                a
            };
        }
        if self.is_infinite(b) {
            return b;
        }
        let both_negative = self.is_negative(a) && self.is_negative(b);
        self.round(&(self.exact(a) + self.exact(b)), both_negative)
    }

    pub(crate) fn mul(self, a: u64, b: u64) -> u64 {
        let negative = self.is_negative(a) ^ self.is_negative(b);
        if self.is_nan(a)
            || self.is_nan(b)
            || (self.is_infinite(a) && self.is_zero(b))
            || (self.is_infinite(b) && self.is_zero(a))
        {
            return self.nan();
        }
        if self.is_infinite(a) || self.is_infinite(b) {
            return self.infinity(negative);
        }
        self.round(&(self.exact(a) * self.exact(b)), negative)
    }

    pub(crate) fn div(self, a: u64, b: u64) -> u64 {
        let negative = self.is_negative(a) ^ self.is_negative(b);
        if self.is_nan(a)
            || self.is_nan(b)
            || (self.is_infinite(a) && self.is_infinite(b))
            || (self.is_zero(a) && self.is_zero(b))
        {
            return self.nan();
        }
        if self.is_infinite(a) || self.is_zero(b) {
            return self.infinity(negative);
        }
        if self.is_infinite(b) || self.is_zero(a) {
            return self.zero(negative);
        }
        self.round(&(self.exact(a) / self.exact(b)), negative)
    }

    /// Exact a*b+c rounded once; no intermediate rounding.
    pub(crate) fn fma(self, a: u64, b: u64, c: u64) -> u64 {
        let product_negative = self.is_negative(a) ^ self.is_negative(b);
        if self.is_nan(a)
            || self.is_nan(b)
            || self.is_nan(c)
            || (self.is_infinite(a) && self.is_zero(b))
            || (self.is_infinite(b) && self.is_zero(a))
        {
            return self.nan();
        }
        if self.is_infinite(a) || self.is_infinite(b) {
            return if self.is_infinite(c) && product_negative != self.is_negative(c) {
                self.nan()
            } else {
                self.infinity(product_negative)
            };
        }
        if self.is_infinite(c) {
            return c;
        }
        let negative_zero = (self.is_zero(a) || self.is_zero(b))
            && product_negative
            && self.is_zero(c)
            && self.is_negative(c);
        self.round(
            &(self.exact(a) * self.exact(b) + self.exact(c)),
            negative_zero,
        )
    }

    /// Correctly rounded square root by bisection over positive encodings;
    /// comparing the midpoint's square resolves nearest-even.
    pub(crate) fn sqrt(self, a: u64) -> u64 {
        if self.is_nan(a) || (self.is_negative(a) && !self.is_zero(a)) {
            return self.nan();
        }
        if self.is_zero(a) || self.is_infinite(a) {
            return a;
        }
        let input = self.exact(a);
        let mut lower = 0_u64;
        let mut upper = self.exponent_mask();
        while upper - lower > 1 {
            let middle = lower + (upper - lower) / 2;
            let candidate = self.exact(middle);
            if &candidate * &candidate <= input {
                lower = middle;
            } else {
                upper = middle;
            }
        }
        let midpoint = (self.exact(lower) + self.exact(upper)) / BigInt::from(2);
        match input.cmp(&(&midpoint * &midpoint)) {
            Ordering::Less => lower,
            Ordering::Greater => upper,
            Ordering::Equal => {
                if lower & 1 == 0 {
                    lower
                } else {
                    upper
                }
            }
        }
    }

    pub(crate) fn encode_integer(self, value: i64) -> u64 {
        self.round(&BigRational::from_integer(BigInt::from(value)), false)
    }

    /// Truncate toward zero after checking representability in `[min, max]`.
    pub(crate) fn trunc_integer(self, a: u64, min: i64, max: i64) -> Result<i64, NumericTrap> {
        if self.is_nan(a) || self.is_infinite(a) {
            return Err(NumericTrap::IntegerOverflow);
        }
        let value = self.exact(a);
        let truncated = value.trunc();
        if truncated < BigRational::from_integer(BigInt::from(min))
            || truncated > BigRational::from_integer(BigInt::from(max))
        {
            return Err(NumericTrap::IntegerOverflow);
        }
        i64::try_from(truncated.to_integer()).map_err(|_| NumericTrap::IntegerOverflow)
    }
}

/// Rounding direction of the exact round-to-integral operations.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Integral {
    Floor,
    Ceil,
    Trunc,
    /// Nearest integer, ties away from zero (C `round`).
    NearestAway,
}

impl Format {
    /// Exact round-to-integral; the result is always representable, NaN is
    /// canonical and the sign of a zero result is the sign of the input.
    pub(crate) fn integral(self, a: u64, mode: Integral) -> u64 {
        if self.is_nan(a) {
            return self.nan();
        }
        if self.is_infinite(a) || self.is_zero(a) {
            return a;
        }
        let value = self.exact(a);
        let integral = match mode {
            Integral::Floor => value.floor(),
            Integral::Ceil => value.ceil(),
            Integral::Trunc => value.trunc(),
            Integral::NearestAway => value.round(),
        };
        self.round(&integral, self.is_negative(a))
    }

    /// IEEE 754-2019 minimumNumber (`maximum` false) or maximumNumber: a NaN
    /// operand is ignored, two NaNs give the canonical NaN, and -0 < +0.
    pub(crate) fn min_max(self, a: u64, b: u64, maximum: bool) -> u64 {
        match (self.is_nan(a), self.is_nan(b)) {
            (true, true) => return self.nan(),
            (true, false) => return b,
            (false, true) => return a,
            (false, false) => {}
        }
        let a_first = if self.is_zero(a) && self.is_zero(b) {
            self.is_negative(a) != maximum
        } else {
            self.compare(a, Comparison::Lt, b) != maximum
        };
        if a_first { a } else { b }
    }

    /// Magnitude of `a` with the sign of `b`. A NaN sign source counts as
    /// positive because NaN signs are not observable; a NaN magnitude stays
    /// the canonical NaN.
    pub(crate) fn copysign(self, a: u64, b: u64) -> u64 {
        if self.is_nan(a) {
            return self.nan();
        }
        let negative = !self.is_nan(b) && self.is_negative(b);
        (a & !self.sign()) | if negative { self.sign() } else { 0 }
    }

    pub(crate) fn abs(self, a: u64) -> u64 {
        if self.is_nan(a) {
            return self.nan();
        }
        a & !self.sign()
    }
}
