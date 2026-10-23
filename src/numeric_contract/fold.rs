//! Fold tiers of numeric policy v2 (`docs/numeric.md`) and their oracle.
//!
//! A tiered fold has `init` and then the reached addends in source order as
//! its summands. Strict, Tree and Exact each name one result; Relaxed admits
//! any result inside the `reassoc_add` bound, checked here in exact rationals.

use super::{F32, F32Bits};
use num_bigint::BigInt;
use num_rational::BigRational;

/// The folds the numeric checker lowered (ENG-03), as (declared tier, algorithm
/// that ran). A Strict function's `reassoc_add` folds record as Relaxed.
pub type LoweredFolds = std::collections::BTreeSet<(FoldTier, FoldTier)>;

/// How a binary32 fold may associate. Strict is the v1 source-order fold.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum FoldTier {
    Strict,
    Tree,
    Exact,
    Relaxed,
}

impl FoldTier {
    pub const ALL: [Self; 4] = [Self::Strict, Self::Tree, Self::Exact, Self::Relaxed];

    /// The source and policy spelling of the tier.
    pub const fn name(self) -> &'static str {
        match self {
            Self::Strict => "strict",
            Self::Tree => "tree",
            Self::Exact => "exact",
            Self::Relaxed => "relaxed",
        }
    }

    /// Exact spellings only; `fast` and every other string are refused.
    pub fn from_name(name: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|tier| tier.name() == name)
    }

    /// The one result of a Strict, Tree or Exact fold. Relaxed has no single
    /// result; check a candidate with [`relaxed_admits`].
    pub fn reference(self, init: F32Bits, terms: &[F32Bits]) -> Option<F32Bits> {
        match self {
            Self::Strict => Some(F32Bits::ordered_sum(init, terms)),
            Self::Tree if terms.is_empty() => Some(init),
            Self::Tree => Some(init.add(crate::fsum_reference::sum_tree(terms))),
            Self::Exact => {
                let summands = std::iter::once(init).chain(terms.iter().copied());
                Some(crate::fsum_reference::sum_exact(
                    &summands.collect::<Vec<_>>(),
                ))
            }
            Self::Relaxed => None,
        }
    }
}

/// Whether `result` is an admissible Relaxed fold of `init` and `terms`, with
/// `N = terms.len() + 1` summands and `B = (N - 1) * 2^-24 * sum(|x_i|)` over
/// the finite summands (the Jeannerod-Rump bound, which every summation tree
/// meets, so Strict, Tree and Exact are all admitted):
/// - NaN, or both infinities, among the summands gives NaN;
/// - outside the finite domain, `sum(|x_i|) + B > 2^127`, where a partial sum
///   of some order may overflow, any IEEE value is admitted;
/// - otherwise infinities of one sign give that infinity, and a finite fold is
///   within `B` of the exact sum and is `-0` exactly when every summand is `-0`.
pub fn relaxed_admits(init: F32Bits, terms: &[F32Bits], result: F32Bits) -> bool {
    let summands = || std::iter::once(init).chain(terms.iter().copied());
    let positive = summands().any(|x| x == F32Bits::INFINITY);
    let negative = summands().any(|x| x == F32Bits::NEG_INFINITY);
    if summands().any(F32Bits::is_nan) || (positive && negative) {
        return result.is_nan();
    }
    let finite = || summands().filter(|x| !x.is_infinite());
    let magnitude: BigRational = finite().map(|x| F32.exact(F32.abs(x.wide()))).sum();
    let count = BigInt::from(terms.len());
    let bound = &magnitude * BigRational::new(count, BigInt::from(1) << 24);
    if magnitude + &bound > BigRational::from_integer(BigInt::from(1) << 127) {
        return true;
    }
    if positive || negative {
        return result
            == if positive {
                F32Bits::INFINITY
            } else {
                F32Bits::NEG_INFINITY
            };
    }
    if result.is_nan() || result.is_infinite() {
        return false;
    }
    if result.is_zero() && result.is_negative() != summands().all(|x| x == F32Bits::NEG_ZERO) {
        return false;
    }
    let exact: BigRational = finite().map(|x| F32.exact(x.wide())).sum();
    let error = F32.exact(result.wide()) - exact;
    error <= bound && -error <= bound
}
