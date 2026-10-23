//! Storage-only narrow types of numeric policy v2 (`docs/numeric.md`).
//!
//! `f16` (IEEE binary16) and `bf16` widen exactly to binary32 and narrow by
//! one round-to-nearest-even; `i8` and `u8` widen exactly and narrow with an
//! `INTEGER_OVERFLOW` trap. No arithmetic is defined on the narrow types.
//! Every rounding is the rational oracle's, independent of host floats.

use super::ieee::{BFLOAT16, BINARY16, Format};
use super::{F32, F32Bits, NumericTrap};

/// Exact binary32 value of a narrow float encoding; NaN becomes the class.
fn widen(format: Format, bits: u16) -> F32Bits {
    let a = u64::from(bits);
    if format.is_nan(a) {
        F32Bits::NAN
    } else if format.is_infinite(a) {
        if format.is_negative(a) {
            F32Bits::NEG_INFINITY
        } else {
            F32Bits::INFINITY
        }
    } else {
        F32Bits::narrow(F32.round(&format.exact(a), format.is_negative(a)))
    }
}

/// One round-to-nearest-even of a binary32 value into a narrow format.
fn narrow(format: Format, value: F32Bits) -> u16 {
    let a = value.wide();
    let bits = if value.is_nan() {
        format.nan()
    } else if value.is_infinite() {
        format.infinity(value.is_negative())
    } else {
        format.round(&F32.exact(a), value.is_negative())
    };
    // Every result of a 16-bit format fits the low 16 bits.
    bits as u16
}

macro_rules! narrow_float {
    ($name:ident, $format:expr, $doc:literal) => {
        #[doc = $doc]
        #[derive(Clone, Copy, Debug, PartialEq, Eq)]
        pub struct $name(u16);

        impl $name {
            pub const fn from_bits(bits: u16) -> Self {
                Self(bits)
            }

            pub const fn to_bits(self) -> u16 {
                self.0
            }

            /// Load: the exact binary32 value (every encoding is representable).
            pub fn widen(self) -> F32Bits {
                widen($format, self.0)
            }

            /// Store: round once to nearest even; overflow is the signed
            /// infinity and NaN stays NaN.
            pub fn round(value: F32Bits) -> Self {
                Self(narrow($format, value))
            }

            pub fn is_nan(self) -> bool {
                $format.is_nan(u64::from(self.0))
            }
        }
    };
}

narrow_float!(F16Bits, BINARY16, "An IEEE binary16 storage encoding.");
narrow_float!(Bf16Bits, BFLOAT16, "A bfloat16 storage encoding.");

/// Store into `i8`: values outside `[-128, 127]` trap.
pub fn narrow_i8(value: i32) -> Result<i8, NumericTrap> {
    i8::try_from(value).map_err(|_| NumericTrap::IntegerOverflow)
}

/// Store into `u8`: values above 255 trap.
pub fn narrow_u8(value: u32) -> Result<u8, NumericTrap> {
    u8::try_from(value).map_err(|_| NumericTrap::IntegerOverflow)
}
