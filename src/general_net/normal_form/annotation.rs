//! Lowering annotations a certificate can carry (#140, P4a iteration 2).
//!
//! P4a showed that a law-directed flat form matches handwritten device code
//! once the normal form carries: (1) bounds and signedness of intermediates
//! plus the guards that license narrow unchecked arithmetic, (2) exact
//! operator spans, demand order and checked-fallback state, (3) combine laws,
//! coefficients and modulus boundaries, and (6) provenance that keeps
//! legality apart from profitability, planning cost and device acceptance.
//! Combine laws and coefficients are already certificate facts
//! ([`super::Fact::Affine`], [`super::Fact::Combine`]); this module adds the
//! rest. Enumeration costs and target thresholds (items 4 and 5) are cost
//! model inputs of a region, never certificate facts, and are not typed here.

use crate::{front::Span, mir::BlockId};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Signedness {
    Signed,
    Unsigned,
}

/// What an [`Annotation`] constrains.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Target {
    /// A scalar local of the region.
    Var(String),
    /// Every element of an array of the region.
    Array(String),
    /// A value flowing along a CFG edge `(from, to)` of the region.
    Edge(BlockId, BlockId),
    /// A path-sum weight or prologue value of the region (L9).
    PathWeight,
    /// The result of the operator, index or store site with this exact span.
    Op(Span),
}

/// A runtime check `low <= var <= high` on a region live-in at region entry.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct RangeGuard {
    pub var: String,
    pub low: i64,
    pub high: i64,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Annotation {
    /// Item 1: `low <= value <= high` for every value of `target`.
    Bounds {
        target: Target,
        low: i64,
        high: i64,
        signedness: Signedness,
    },
    /// Item 1: `target` may run as unchecked `bits`-bit arithmetic when every
    /// `license` guard holds at region entry. Otherwise the exact checked
    /// fallback runs and reports the first source-order trap.
    Narrow {
        target: Target,
        bits: u8,
        signedness: Signedness,
        license: Vec<RangeGuard>,
    },
    /// Item 2: every operator, index and store site of the region in demand
    /// (source evaluation) order; the first failing site in this order is the
    /// trap the checked fallback reports.
    Demand(Vec<Span>),
    /// Item 2: the state the checked fallback restores before it re-runs the
    /// region serially: the outer scalars and arrays the region writes.
    Fallback {
        vars: Vec<String>,
        arrays: Vec<String>,
    },
    /// Item 3: `target`, a `% P` site with a literal `P >= 2`, reduces modulo
    /// `modulus`; a regrouped combine must not move work across this
    /// boundary. `% 1` is an ordinary site, as in every ring and path sum.
    Modulus { target: Target, modulus: u32 },
}

/// Item 6: the kinds of evidence, never mixed. A certificate carries only
/// legality; the others belong to the planner and the device runs.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Evidence {
    Legality,
    Profitability,
    PlanningCost,
    DeviceAcceptance,
}

/// Where an annotation comes from.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Source {
    /// The checker re-derives it from the net.
    Rederived,
    /// A named, versioned analysis (range inference). The checker verifies it
    /// only against the matching [`super::certificate::Analysis`] the caller
    /// supplies, and reports it as unverified otherwise.
    Analysis { name: String, version: u32 },
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Provenance {
    pub evidence: Evidence,
    pub source: Source,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Annotated {
    pub annotation: Annotation,
    pub provenance: Provenance,
}

impl Annotated {
    /// A legality annotation the checker re-derives from the net.
    pub fn rederived(annotation: Annotation) -> Self {
        Self {
            annotation,
            provenance: Provenance {
                evidence: Evidence::Legality,
                source: Source::Rederived,
            },
        }
    }
}
