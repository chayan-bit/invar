//! P1b: law-directed parallel normal form (PNF) with checked law certificates
//! (#133 sections 2, 3 and 6; v0.1 law set L1-L7, L9-L15, L19, L20), plus the
//! cost-gated L8 and large-k L5 of #281 (see [`COST_UNCALIBRATED`]).
//!
//! [`extract`] derives the PNF from the region partition of a verified net. Every
//! applied law yields a [`Certificate`]; [`certificate::check`] re-verifies it
//! against the current net with an independent analysis, never by name, hash
//! or benchmark id. [`exec`] is the CPU reference semantics of the PNF (law
//! representations, runtime guards and serial fallback) that the differential
//! tests compare with the CPU net oracle. Lowering is out of scope here; see
//! `docs/net-normal-form-v01.md` for the API that P2, P3, P4a/b and P8 consume.

pub mod annotation;
pub mod certificate;
pub mod exec;
mod fsm;
mod gate;
mod linear;
mod loops;
mod poly;
mod recursion;

use super::regions::{self, Counting, Partition, RegionId};
use crate::{front::Span, laws::Law};
use regions::tree::Expr;

pub use annotation::{
    Annotated, Annotation, Evidence, Provenance, RangeGuard, Signedness, Source, Target,
};

pub const VERSION: u32 = 1;

/// The v0.1 law set of #133 section 10 (decision L4).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum LawId {
    L1Map,
    L2SealedFold,
    L3TrapLiftedFold,
    L4Affine,
    L5Matrix,
    L6Tropical,
    L7Select,
    /// L8 (#281): outside the v0.1 set [`LawId::ALL`]; cost-gated.
    L8FiniteState,
    L9PathSum,
    L10RangeSum,
    L11ScatterCombine,
    L12OrderedWrite,
    L13Compaction,
    L14MonotoneFrontier,
    L15FirstExit,
    L19MapFusion,
    L20ForkJoin,
}

impl LawId {
    pub const ALL: [Self; 16] = [
        Self::L1Map,
        Self::L2SealedFold,
        Self::L3TrapLiftedFold,
        Self::L4Affine,
        Self::L5Matrix,
        Self::L6Tropical,
        Self::L7Select,
        Self::L9PathSum,
        Self::L10RangeSum,
        Self::L11ScatterCombine,
        Self::L12OrderedWrite,
        Self::L13Compaction,
        Self::L14MonotoneFrontier,
        Self::L15FirstExit,
        Self::L19MapFusion,
        Self::L20ForkJoin,
    ];

    /// The law's number in the v0.1 law table (`L<number>`).
    pub const fn number(self) -> u8 {
        match self {
            Self::L1Map => 1,
            Self::L2SealedFold => 2,
            Self::L3TrapLiftedFold => 3,
            Self::L4Affine => 4,
            Self::L5Matrix => 5,
            Self::L6Tropical => 6,
            Self::L7Select => 7,
            Self::L8FiniteState => 8,
            Self::L9PathSum => 9,
            Self::L10RangeSum => 10,
            Self::L11ScatterCombine => 11,
            Self::L12OrderedWrite => 12,
            Self::L13Compaction => 13,
            Self::L14MonotoneFrontier => 14,
            Self::L15FirstExit => 15,
            Self::L19MapFusion => 19,
            Self::L20ForkJoin => 20,
        }
    }
}

impl std::fmt::Display for LawId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.pad(&format!("L{}", self.number()))
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum IntTy {
    I32,
    U32,
}

/// The ring a closed-composition representation is evaluated in.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Ring {
    /// `Z/2^32` for the sealed `wrapping_add`/`wrapping_mul` intrinsics.
    Wrapping32,
    /// Checked integers regrouped in `Z/2^64` (Lemma A): exact for every
    /// prefix before the first violation, which per-lane replay locates.
    Checked64(IntTy),
    /// `Z/P` for an outer literal `% P` (Lemma B). Exact only on non-negative
    /// seeds and coefficients, which runtime guards establish.
    Residue { modulus: u32, ty: IntTy },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Order {
    Min,
    Max,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ScalarTy {
    I32,
    U32,
    F32,
}

/// A law representation `R` of the PNF grammar: identity, combine and apply.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Rep {
    /// L2: one u32 word, `(Z/2^32, +, 0)` or `(Z/2^32, *, 1)`.
    Sealed(Law),
    /// L3 sum: `(total, min prefix, max prefix)` in exact 64-bit words.
    LiftedSum(IntTy),
    /// L3 product: a signed magnitude saturated above the integer range.
    SaturatedProduct(IntTy),
    /// L4: `s -> a s + b` over `ring` (two words).
    Affine(Ring),
    /// L5: `k <= 16` jointly linear carried variables, `(k+1)^2` words;
    /// groups above 4 are cost-gated.
    Matrix { ring: Ring, vars: Vec<String> },
    /// L6: `s -> max(s + a, b)` (or `min`), exact in 64-bit (Lemma A).
    Tropical { order: Order, ty: IntTy },
    /// L7: leftmost (strict compare) or rightmost (non-strict) extremum with
    /// companions; f32 NaN elements are the identity.
    Select {
        order: Order,
        strict: bool,
        elem: ScalarTy,
        companions: Vec<String>,
    },
    /// L12 on a scalar: the last value written in source order.
    Last,
    /// L13: an exclusive counter scan whose prefixes are store positions.
    Counter(IntTy),
    /// A representation given as data (#281 decision 2).
    Lifted(RepExpr),
}

/// Law representations as data (#281 decision 2): laws beyond the sealed
/// v0.1 catalogue are described here, so a new law adds no executor variant.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum RepExpr {
    /// L8: the maps `[0, S) -> [0, S)` under composition. Every update is
    /// `s := table[stride * s + c]` with `c` free of `s`, and
    /// `S = len(table) / stride`; [`Guard::StateTable`] keeps states in range.
    FunctionTable { table: String, stride: u32 },
}

/// What a runtime guard checks before the region publishes anything. A false
/// guard selects the serial fallback for the whole region.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Guard {
    /// Lemma B: the carried seed is a reduced residue, `0 <= seed < P`, so its
    /// ring word is its source value and lane replay sees `a s + b` (#266).
    ReducedSeed(String),
    /// Lemma B: every per-iteration coefficient of the update is non-negative.
    NonNegativeCoefficients(String),
    /// L1 affine footprint: a loop-invariant store stride is not zero.
    StrideNonZero(String),
    /// L9: every value combined under a modulus lies in the table `[0, P)`.
    TableRange { modulus: u32 },
    /// L9: every node constant is non-negative.
    NonNegativeConstants,
    /// L9: `sum_j c_j (P - 1) + k` cannot overflow the source combine.
    NoOverflowCombine,
    /// L8: with `S = len(table) / stride`, `1 <= S <= fsm::MAX_STATES`, the
    /// seed and every table entry lie in `[0, S)`.
    StateTable {
        table: String,
        stride: u32,
        seed: String,
    },
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Subject {
    /// L1 region legality: iterations are independent apart from the state
    /// the loop's other certificates cover (completeness of the loop's PNF).
    Iterations,
    /// A carried scalar (or the lead variable of a matrix group).
    Var(String),
    Array(String),
    /// The loop's early `return` (L15).
    Exit,
    /// The function's own recursion (L9, L10).
    Recursion,
    /// The next adjacent loop, fused into this one (L19).
    Fused(RegionId),
    /// Independent sibling calls starting at the call with this span (L20).
    Fork(Span),
}

/// Law-specific facts the certificate relies on, re-derived by the checker.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Fact {
    /// One update of the subject, in source order: `s := (a s + b) mod P`.
    Affine { coefficient: Expr, offset: Expr },
    /// `s := value` with `value` free of carried state.
    Reset(Expr),
    /// `if value <cmp> s { s = value; companions... }`.
    Select(Expr),
    /// A store `array[index] = value` the law covers.
    Store { index: Expr, value: Expr },
    /// L9/L10 combine `(sum_j c_j r_j + k) mod P` at one return.
    Combine {
        coefficients: Vec<(String, i64)>,
        constant: Expr,
    },
    /// Number of sibling calls in a fork site.
    Siblings(usize),
    /// L8: one update of the subject, in source order: `s := table[index]`.
    Transition(Expr),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Certificate {
    pub version: u32,
    /// Compiled-MIR contract digest of the net the certificate is bound to.
    pub fingerprint: [u8; 32],
    pub region: RegionId,
    pub law: LawId,
    pub subject: Subject,
    pub rep: Option<Rep>,
    pub guards: Vec<Guard>,
    pub facts: Vec<Fact>,
    /// Lowering annotations (see [`annotation`]); the extractor emits none
    /// in v1 and the checker validates any it is given.
    pub annotations: Vec<Annotated>,
}

/// How a written array is combined across iterations.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ArrayLaw {
    /// L1: iteration-injective (or block-owned) stores, reads at store indices.
    Own,
    /// L11: `a[k] = a[k] op e` with an associative, commutative `op`.
    Combine { wrapping: bool },
    /// L12: write-only; the last writer in source order wins.
    Ordered,
    /// L13: stores at a certified counter.
    Compaction { counter: String },
    /// L14: constant flag stores; reads reconstructed from round-start state.
    Monotone { own_writer: bool },
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Carried {
    pub var: String,
    pub law: LawId,
    pub rep: Rep,
    /// The value is read outside its own updates, so lowering needs a scan
    /// (prefixes), not only a reduction.
    pub scan: bool,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LoopNode {
    pub counting: Counting,
    /// Carried state in evaluation order: a variable's representation may read
    /// variables earlier in this list (successive scans).
    pub carried: Vec<Carried>,
    pub arrays: Vec<(String, ArrayLaw)>,
    pub first_exit: bool,
    /// L19: the adjacent loop executed in the same lanes after this one.
    pub fused_with: Option<RegionId>,
}

impl LoopNode {
    /// The PNF node kind of #133 section 3 (`map`, `reduce`, `scan`, ...).
    pub fn shape(&self) -> &'static str {
        if self.first_exit {
            return "first";
        }
        if self
            .arrays
            .iter()
            .any(|(_, law)| matches!(law, ArrayLaw::Monotone { .. }))
        {
            return "frontier";
        }
        if self
            .arrays
            .iter()
            .any(|(_, law)| matches!(law, ArrayLaw::Combine { .. } | ArrayLaw::Ordered))
        {
            return "scatter";
        }
        if self.carried.iter().any(|carried| carried.scan) {
            return "scan";
        }
        if !self.carried.is_empty() {
            return "reduce";
        }
        "map"
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Node {
    /// The frame keeps this loop serial; `reason` names the missing law.
    Serial {
        reason: String,
    },
    Parallel(LoopNode),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LoopForm {
    pub region: RegionId,
    pub node: Node,
    pub certificates: Vec<Certificate>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Recursion {
    /// L9: `paths(prologue) ; map(leaf * weight) ; reduce(+ mod P)`.
    PathSum { modulus: u32 },
    /// L10: range-split checked or wrapping sum over the call tree.
    RangeSum,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ForkSite {
    /// Span of the first sibling call.
    pub start: Span,
    pub siblings: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FunctionForm {
    pub index: usize,
    pub name: String,
    /// Why the whole function stays an opaque serial frame, if it does.
    pub opaque: Option<&'static str>,
    pub recursion: Option<Recursion>,
    pub loops: Vec<LoopForm>,
    pub forks: Vec<ForkSite>,
    /// Function-level certificates (L9, L10, L20).
    pub certificates: Vec<Certificate>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NormalForm {
    pub version: u32,
    pub fingerprint: [u8; 32],
    pub functions: Vec<FunctionForm>,
}

impl NormalForm {
    pub fn certificates(&self) -> impl Iterator<Item = &Certificate> {
        self.functions.iter().flat_map(|function| {
            function
                .certificates
                .iter()
                .chain(function.loops.iter().flat_map(|form| &form.certificates))
        })
    }

    pub fn loop_form(&self, region: RegionId) -> Option<&LoopForm> {
        self.functions
            .get(region.function)?
            .loops
            .iter()
            .find(|form| form.region == region)
    }

    /// Whether the certified program is flat: no residual fork-join (D5 tier 1).
    pub fn is_flat(&self) -> bool {
        self.functions
            .iter()
            .all(|function| function.forks.is_empty())
    }
}

#[derive(Debug, PartialEq, Eq)]
pub enum Error {
    Regions(regions::Error),
}

/// Laws whose work grows with a state count (L8) or with `(k + 1)^2` (L5 over
/// more than four variables) are chosen only when the cost model prefers them
/// (#133 section 6.4), which needs the M5-01 calibration (#228). Until then
/// [`extract`] keeps such a loop serial with this reason, and
/// [`extract_uncalibrated`] admits it for the reference executor.
pub const COST_UNCALIBRATED: &str = "law_cost_uncalibrated";

/// Derive the parallel normal form of a verified net.
pub fn extract(net: &super::Net, proof: &super::Proof) -> Result<NormalForm, Error> {
    let partition = regions::partition(net, proof).map_err(Error::Regions)?;
    Ok(from_partition(&partition))
}

/// [`extract`], admitting the cost-gated laws as if the cost model chose them.
pub fn extract_uncalibrated(net: &super::Net, proof: &super::Proof) -> Result<NormalForm, Error> {
    let partition = regions::partition(net, proof).map_err(Error::Regions)?;
    Ok(derive(&partition, true))
}

/// Derive the PNF from an already re-derived partition.
pub fn from_partition(partition: &Partition) -> NormalForm {
    derive(partition, false)
}

fn derive(partition: &Partition, uncalibrated: bool) -> NormalForm {
    let functions = partition
        .functions
        .iter()
        .map(|function| {
            let tree = match &function.tree {
                Ok(tree) => tree,
                Err(reason) => {
                    return FunctionForm {
                        index: function.index,
                        name: function.name.clone(),
                        opaque: Some(reason.reason()),
                        recursion: None,
                        loops: Vec::new(),
                        forks: Vec::new(),
                        certificates: Vec::new(),
                    };
                }
            };
            let context = loops::Context {
                partition,
                fingerprint: partition.fingerprint,
                tree,
            };
            let loops = loops::extract_loops(&context)
                .into_iter()
                .map(|form| gate::admit(partition, form, uncalibrated))
                .collect();
            let (recursion, forks, certificates) = recursion::extract(&context);
            FunctionForm {
                index: function.index,
                name: function.name.clone(),
                opaque: None,
                recursion,
                loops,
                forks,
                certificates,
            }
        })
        .collect();
    NormalForm {
        version: VERSION,
        fingerprint: partition.fingerprint,
        functions,
    }
}
