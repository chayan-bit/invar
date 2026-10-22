//! Reference semantics of the deterministic float reductions, used by the
//! source evaluator. It is written independently of the fast engine code in
//! `fsum.rs`: the tree is reduced level by level with the rational binary32
//! oracle, and exact sums use one arbitrary-precision integer and the oracle's
//! single rounding. Engines must equal these results bit for bit.

use crate::fsum::LEAF;
use crate::numeric_contract::F32Bits;
use num_bigint::BigInt;
use num_rational::BigRational;

fn canonical(value: F32Bits) -> F32Bits {
    if value.is_nan() { F32Bits::NAN } else { value }
}

/// Fixed-shape pairwise sum of `terms` (`sum_tree` on the terms).
pub fn tree(terms: &[F32Bits]) -> F32Bits {
    let mut level = terms
        .chunks(LEAF)
        .map(|block| block[1..].iter().fold(block[0], |acc, &term| acc.add(term)))
        .collect::<Vec<_>>();
    if level.is_empty() {
        return F32Bits::POS_ZERO;
    }
    while level.len() > 1 {
        level = level
            .chunks(2)
            .map(|pair| match pair {
                [left, right] => left.add(*right),
                [single] => *single,
                _ => unreachable!("chunks(2) yields one or two items"),
            })
            .collect();
    }
    canonical(level[0])
}

pub fn sum_tree(xs: &[F32Bits]) -> F32Bits {
    tree(xs)
}

/// Rounded elementwise products, then the same tree. Callers check lengths.
pub fn dot_tree(a: &[F32Bits], b: &[F32Bits]) -> F32Bits {
    let products = a.iter().zip(b).map(|(x, y)| x.mul(*y)).collect::<Vec<_>>();
    tree(&products)
}

/// A finite binary32 word as an exact integer multiple of `2^-149`.
fn scaled(value: F32Bits) -> BigInt {
    let bits = value.to_bits();
    let exponent = (bits >> 23) & 0xff;
    let fraction = bits & 0x007f_ffff;
    let magnitude = if exponent == 0 {
        BigInt::from(fraction)
    } else {
        BigInt::from(fraction | 1 << 23) << (exponent - 1)
    };
    if value.is_negative() {
        -magnitude
    } else {
        magnitude
    }
}

/// Exactly rounded sum of finite terms given as integers times `2^-shift`.
fn exact(terms: impl Iterator<Item = BigInt>, shift: usize, all_negative_zero: bool) -> F32Bits {
    let total: BigInt = terms.sum();
    crate::numeric_contract::round_rational(
        BigRational::new(total, BigInt::from(1) << shift),
        all_negative_zero,
    )
}

/// The correctly rounded exact sum, with the documented special values.
pub fn sum_exact(xs: &[F32Bits]) -> F32Bits {
    if xs.is_empty() {
        return F32Bits::POS_ZERO;
    }
    if let Some(special) = special(xs.iter().copied()) {
        return special;
    }
    let negative_zero = xs
        .iter()
        .all(|x| x.to_bits() == F32Bits::NEG_ZERO.to_bits());
    exact(xs.iter().map(|&x| scaled(x)), 149, negative_zero)
}

/// The correctly rounded exact sum of exact products.
pub fn dot_exact(a: &[F32Bits], b: &[F32Bits]) -> F32Bits {
    if a.is_empty() {
        return F32Bits::POS_ZERO;
    }
    if a.iter().chain(b).any(|x| x.is_nan())
        || a.iter()
            .zip(b)
            .any(|(x, y)| (x.is_infinite() && y.is_zero()) || (x.is_zero() && y.is_infinite()))
    {
        return F32Bits::NAN;
    }
    // A product with an infinite factor is that signed infinity.
    let infinite = a
        .iter()
        .zip(b)
        .filter(|(x, y)| x.is_infinite() || y.is_infinite());
    if let Some(special) = special(infinite.map(|(x, y)| x.mul(*y))) {
        return special;
    }
    let negative_zero = a
        .iter()
        .zip(b)
        .all(|(x, y)| (x.is_zero() || y.is_zero()) && x.is_negative() != y.is_negative());
    exact(
        a.iter().zip(b).map(|(&x, &y)| scaled(x) * scaled(y)),
        298,
        negative_zero,
    )
}

/// NaN, or `+inf` with `-inf`, is NaN; otherwise any infinity wins.
fn special(terms: impl Iterator<Item = F32Bits>) -> Option<F32Bits> {
    let (mut positive, mut negative) = (false, false);
    for term in terms {
        if term.is_nan() {
            return Some(F32Bits::NAN);
        }
        if term.is_infinite() {
            if term.is_negative() {
                negative = true;
            } else {
                positive = true;
            }
        }
    }
    match (positive, negative) {
        (true, true) => Some(F32Bits::NAN),
        (true, false) => Some(F32Bits::INFINITY),
        (false, true) => Some(F32Bits::NEG_INFINITY),
        (false, false) => None,
    }
}
