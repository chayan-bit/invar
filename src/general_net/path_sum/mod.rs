//! P4a: path-sum and chain-recursion lowering (#133 L9 and L10, #140 A').
//!
//! [`lower`] turns a checked L9 (`(sum c_j r_j + k) % P`) or L10 (checked
//! sum) recursion certificate into one program-specific device source with a
//! proven-narrow 32-bit fast path:
//!
//! * the **path-replay kernel**: one lane per point of the root interval.
//!   Interval-owned recursions (children partition `[lo, hi)` in order) are
//!   replayed root to leaf, lane `k` following the unique child that contains
//!   `lo + k` and owning the leaf (and node constants) whose interval starts at
//!   its key. The split geometry is validated per lane, never assumed;
//! * the **weighted reduction**: edge coefficients multiply a lane weight mod
//!   P; owned leaves and node constants add `weight * value mod P`, reduced in
//!   the group and then across groups (P8 `ResidueSum`);
//! * the **raw-root case**: a root that reaches a leaf through tail calls only
//!   (a chain) returns the leaf's raw value, not its residue.
//!
//! The kernel runs unchecked 32-bit arithmetic only under a **narrow licence**:
//! [`range`] proves, for a box of root parameter ranges, that no demanded
//! operator can trap and that every L9 guard holds. The host checks the box
//! (the guard) before launch; any input outside it, any lane flag (geometry,
//! depth) or any trap while evaluating the entry takes the exact checked
//! fallback, which owns source-order traps. The shape [`CostModel`] then picks
//! one lane, one fused group or parallel groups.

pub mod device;
mod emit;
mod entry;
mod geometry;
pub mod host;
pub mod range;

use super::{
    Net, Proof,
    law_collective::{self, Emission, LawRepr, Layout, Representation},
    normal_form::{
        self, Annotated, Annotation, Certificate, Evidence, Fact, LawId, Provenance, RangeGuard,
        Recursion, Signedness, Source, Subject, Target, certificate::Rejection,
    },
    regions::{
        self, Partition,
        tree::{self, Expr, FunctionTree, Stmt},
    },
};
use crate::front::Type;
pub use entry::Entry;
pub use host::{CostModel, Launch, Launcher, Outcome, Plan, Scope, Shape};
use range::{Combines, Iv};

pub const VERSION: u32 = 1;
/// Threads per group in every path-sum kernel.
pub const GROUP: u32 = 256;
/// Oracle call-frame limit the lanes must respect (`oracle::MAX_CALL_DEPTH`).
const MAX_FRAMES: usize = 64;
/// Largest lane count a launch accepts; larger roots take the fallback.
pub const MAX_LANES: u32 = 1 << 26;

/// Lane flags (`out[0]`): any of them discards the fast result.
pub const FLAG_GEOMETRY: u32 = 1;
pub const FLAG_DEPTH: u32 = 2;
pub const FLAG_BUDGET: u32 = 4;
pub const FLAG_RAW: u32 = 8;
/// An L9 guard failed at run time (leaf table range or combine constant).
pub const FLAG_GUARD: u32 = 16;

/// Named lowering mutations; each must be caught by the differential tests.
#[doc(hidden)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Mutation {
    #[default]
    None,
    /// Lanes do not validate that children partition the parent interval.
    DropGeometry,
    /// The partition is trusted as statically proven without a proof.
    AssumeGeometry,
    /// A leaf reached through tail calls only is reduced as a residue.
    SkipRaw,
    /// Leaves under a combine are not checked against `[0, P)`.
    DropTableGuard,
    /// Edge coefficients do not scale the path weight.
    WrongWeight,
    /// Every lane through a combine node adds its constant, not only the owner.
    ConstantEveryLane,
    /// The host guard admits every argument (no narrow licence).
    WidenLicence,
    /// Device reductions drop the lane flag and raw words (device kernels only).
    DropLaneFlags,
}

/// How lanes are mapped to call-tree paths.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Enumerator {
    /// Lane `k` replays the path to the leaf owning point `lo + k`.
    Interval { lo: usize, hi: usize },
    /// One lane evaluates the depth-bounded recursion in source form.
    Serial,
}

/// The reduction of the certified recursion.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Combine {
    /// L9: weighted residues modulo P.
    Residue(u32),
    /// L10: exact 64-bit sum with an absolute-sum overflow witness.
    Sum,
}

#[derive(Debug, PartialEq, Eq)]
pub enum Refusal {
    Net(String),
    Certificate(Rejection),
    /// The entry does not tail-call a certified L9/L10 recursion.
    Shape(&'static str),
    /// No narrow licence could be proved (the reason names the obstacle).
    Range(&'static str),
}

/// One certified recursion lowered for every target.
#[derive(Clone, Debug)]
pub struct Lowering {
    pub function: usize,
    pub name: String,
    pub combine: Combine,
    pub enumerator: Enumerator,
    pub entry: Entry,
    /// Host guard: the root call's arguments must lie in these ranges.
    pub licence: Vec<RangeGuard>,
    pub facts: range::Facts,
    /// The L9/L10 certificate with this lowering's annotations attached.
    pub certificate: Certificate,
    /// P8's checked representation of the reduction (L9 only: P8 has no L10
    /// representation, so L10 always uses the kernel's own group reduction).
    pub collective: Option<LawRepr>,
    /// Deepest node (root depth 0) a lane may visit before the frame limit.
    pub max_depth: u32,
    /// Every child interval is statically proven nonempty (with the chain
    /// proven by [`interval_pair`], a partition): device lanes skip the
    /// geometry checks, which the CPU model still re-checks.
    pub geometry: bool,
    pub mutation: Mutation,
    source: String,
}

impl Lowering {
    /// One source for Metal, CUDA (NVRTC) and the CPU model.
    pub fn source(&self) -> &str {
        &self.source
    }
    pub fn param_count(&self) -> usize {
        self.licence.len()
    }
    /// Words of `out`: flags, raw value, then the reduced value.
    pub fn out_words(&self) -> usize {
        2 + self.value_words()
    }
    /// Words of one group partial.
    pub fn value_words(&self) -> usize {
        match self.combine {
            Combine::Residue(_) => 1,
            Combine::Sum => 4,
        }
    }
}

/// Lower the L9/L10 recursion tail-called by `entry`.
pub fn lower(net: &Net, proof: &Proof, entry: &str) -> Result<Lowering, Refusal> {
    lower_with(net, proof, entry, Mutation::None)
}

#[doc(hidden)]
pub fn lower_with(
    net: &Net,
    proof: &Proof,
    entry: &str,
    mutation: Mutation,
) -> Result<Lowering, Refusal> {
    let partition = regions::partition(net, proof).map_err(|e| Refusal::Net(format!("{e:?}")))?;
    let form = normal_form::from_partition(&partition);
    normal_form::certificate::check_all(net, proof, &form).map_err(Refusal::Certificate)?;
    let entry = Entry::find(&partition, entry, |i| {
        form.functions.get(i).is_some_and(|f| f.recursion.is_some())
    })?;
    let function = form
        .functions
        .get(entry.root)
        .ok_or(Refusal::Shape("root function"))?;
    let combine = match &function.recursion {
        Some(Recursion::PathSum { modulus }) => Combine::Residue(*modulus),
        Some(Recursion::RangeSum) => Combine::Sum,
        None => return Err(Refusal::Shape("the root call has no L9/L10 law")),
    };
    let certificate = function
        .certificates
        .iter()
        .find(|c| c.subject == Subject::Recursion)
        .ok_or(Refusal::Shape("recursion certificate"))?
        .clone();
    let collective = collective(net, proof, &certificate, combine)?;
    let trees = trees(&partition);
    let tree = trees[entry.root].ok_or(Refusal::Shape("unstructured root"))?;
    let combines = combines(tree, &certificate)?;
    let enumerator = match interval_pair(tree) {
        Some((lo, hi)) => Enumerator::Interval { lo, hi },
        None => Enumerator::Serial,
    };
    let (licence, facts) = search(&trees, &entry, enumerator, &combines)?;
    let geometry = match enumerator {
        Enumerator::Interval { .. } if mutation == Mutation::AssumeGeometry => true,
        Enumerator::Interval { lo, hi } => geometry::proven(tree, (lo, hi), &facts.params),
        Enumerator::Serial => false,
    };
    let base = if entry.direct { 1 } else { 2 };
    let max_depth = MAX_FRAMES
        .checked_sub(base + facts.frames)
        .ok_or(Refusal::Range("callee frames exceed the frame limit"))?;
    let guards = guards(tree, &licence);
    let mut certificate = certificate;
    certificate.annotations = annotations(&partition, &certificate, combine, &facts, &guards)
        .map_err(Refusal::Certificate)?;
    // P1 verifies the attached range annotations through this analysis.
    normal_form::certificate::check_with_analyses(
        &partition,
        &certificate,
        &[&RangeAnalysis(&certificate)],
    )
    .map_err(Refusal::Certificate)?;
    let host = if mutation == Mutation::WidenLicence {
        tree.params
            .iter()
            .map(|p| {
                let r = range::type_range(&p.ty).unwrap_or(Iv::point(0));
                RangeGuard {
                    var: p.name.clone(),
                    low: r.lo,
                    high: r.hi,
                }
            })
            .collect()
    } else {
        guards
    };
    let mut lowering = Lowering {
        function: entry.root,
        name: tree.name.clone(),
        combine,
        enumerator,
        entry,
        licence: host,
        facts,
        certificate,
        collective,
        max_depth: u32::try_from(max_depth).map_err(|_| Refusal::Range("depth"))?,
        geometry,
        mutation,
        source: String::new(),
    };
    lowering.source = emit::source(&trees, &lowering, &combines).map_err(Refusal::Shape)?;
    Ok(lowering)
}

/// P8's representation of an L9 certificate, re-checked by P8; its modulus
/// must be the one the kernel reduces by.
fn collective(
    net: &Net,
    proof: &Proof,
    certificate: &Certificate,
    combine: Combine,
) -> Result<Option<LawRepr>, Refusal> {
    let Combine::Residue(modulus) = combine else {
        return Ok(None);
    };
    let repr = LawRepr::from_certificate(net, proof, certificate)
        .map_err(|_| Refusal::Shape("P8 rejects the L9 certificate"))?;
    match repr.representation() {
        Representation::ResidueSum { modulus: m } if *m == modulus => Ok(Some(repr)),
        _ => Err(Refusal::Shape(
            "P8 residue modulus differs from the certificate",
        )),
    }
}

impl Lowering {
    /// P8's group reduction for the residue column, when P8 emits one; the
    /// kernel's own reduction (which also carries the flag and raw words) is
    /// used otherwise.
    pub fn p8_reduce(&self, cuda: bool) -> Option<Result<Emission, law_collective::Error>> {
        let repr = self.collective.as_ref()?;
        let layout = Layout {
            lanes: GROUP,
            items_per_lane: 1,
        };
        Some(if cuda {
            law_collective::cuda::emit_reduce(repr, layout)
        } else {
            law_collective::msl::emit_reduce(repr, layout)
        })
    }
}

pub(super) fn trees(partition: &Partition) -> Vec<Option<&FunctionTree>> {
    partition
        .functions
        .iter()
        .map(|f| f.tree.as_ref().ok())
        .collect()
}

fn combines(tree: &FunctionTree, certificate: &Certificate) -> Result<Combines, Refusal> {
    let holes = holes(tree);
    let mut spans = Vec::new();
    tree::walk(&tree.body, &mut Vec::new(), 0, &mut |stmt, _, _| {
        if let Stmt::Return {
            value: Some(value),
            span,
        } = stmt
            && holes.iter().any(|h| value.mentions(h))
        {
            spans.push((span.start, span.end));
        }
    });
    let facts: Vec<_> = certificate
        .facts
        .iter()
        .filter_map(|fact| match fact {
            Fact::Combine {
                coefficients,
                constant,
            } => Some((coefficients.clone(), constant.clone())),
            _ => None,
        })
        .collect();
    if facts.len() != spans.len() {
        return Err(Refusal::Shape("combine facts differ from the returns"));
    }
    Ok(spans.into_iter().zip(facts).collect())
}

/// Names bound by `let x = f(..)` self calls of `tree`.
pub(super) fn holes(tree: &FunctionTree) -> Vec<String> {
    let mut out = Vec::new();
    tree::walk(&tree.body, &mut Vec::new(), 0, &mut |stmt, _, _| {
        if let Stmt::Let {
            name,
            value: Expr::Call { name: callee, .. },
        } = stmt
            && *callee == tree.name
        {
            out.push(name.clone());
        }
    });
    out
}

/// Child lists in source order: the self-call argument lists of each
/// statement list (`let` holes, or one tail call).
pub(super) fn child_lists(tree: &FunctionTree) -> Vec<Vec<&[Expr]>> {
    fn visit<'a>(name: &str, list: &'a [Stmt], out: &mut Vec<Vec<&'a [Expr]>>) {
        let mut children = Vec::new();
        for stmt in list {
            match stmt {
                Stmt::Let {
                    value: Expr::Call { name: n, args, .. },
                    ..
                }
                | Stmt::Return {
                    value: Some(Expr::Call { name: n, args, .. }),
                    ..
                } if n == name => children.push(&args[..]),
                Stmt::If { yes, no, .. } => {
                    visit(name, yes, out);
                    visit(name, no, out);
                }
                Stmt::Loop(inner) => visit(name, &inner.body, out),
                _ => {}
            }
        }
        if !children.is_empty() {
            out.push(children);
        }
    }
    let mut out = Vec::new();
    visit(&tree.name, &tree.body, &mut out);
    out
}

/// The `(lo, hi)` parameters whose self calls chain `lo .. m .. hi` in every
/// child list. Nonempty children are proven by [`geometry`] or re-checked by
/// every lane.
pub(super) fn interval_pair(tree: &FunctionTree) -> Option<(usize, usize)> {
    let lists = child_lists(tree);
    let local = |e: &Expr, name: &str| matches!(e, Expr::Local(n) if n == name);
    let n = tree.params.len();
    (0..n)
        .flat_map(|lo| (0..n).map(move |hi| (lo, hi)))
        .filter(|(lo, hi)| lo != hi)
        .filter(|(lo, hi)| [*lo, *hi].iter().all(|p| tree.params[*p].ty == Type::I32))
        .find(|(lo, hi)| {
            let (lo_name, hi_name) = (&tree.params[*lo].name, &tree.params[*hi].name);
            !lists.is_empty()
                && lists.iter().all(|children| {
                    let first = children.first().and_then(|c| c.get(*lo));
                    let last = children.last().and_then(|c| c.get(*hi));
                    first.is_some_and(|e| local(e, lo_name))
                        && last.is_some_and(|e| local(e, hi_name))
                        && children.windows(2).all(|pair| {
                            match (pair[0].get(*hi), pair[1].get(*lo)) {
                                (Some(a), Some(b)) => tree::same(a, b),
                                _ => false,
                            }
                        })
                })
        })
}

fn guards(tree: &FunctionTree, licence: &[Iv]) -> Vec<RangeGuard> {
    tree.params
        .iter()
        .zip(licence)
        .map(|(param, iv)| RangeGuard {
            var: param.name.clone(),
            low: iv.lo,
            high: iv.hi,
        })
        .collect()
}

/// Grow a licence box around zero, one coordinate at a time (the interval-owned
/// pair together), to a maximal box under which the analysis proves the lane:
/// highs first, then lows, each by doubling and then bisection.
fn search(
    trees: &[Option<&FunctionTree>],
    entry: &Entry,
    enumerator: Enumerator,
    combines: &Combines,
) -> Result<(Vec<Iv>, range::Facts), Refusal> {
    let tree = trees[entry.root].ok_or(Refusal::Shape("root"))?;
    let owned = match enumerator {
        Enumerator::Interval { lo, hi } => Some((lo, hi)),
        Enumerator::Serial => None,
    };
    let prove = |licence: &[Iv]| {
        range::analyse(&range::Spec {
            trees,
            root: entry.root,
            owned,
            licence,
            combines,
        })
    };
    let n = tree.params.len();
    let mut licence = vec![Iv::point(0); n];
    let mut facts = prove(&licence).map_err(Refusal::Range)?;
    let mut groups: Vec<Vec<usize>> = Vec::new();
    if let Some((lo, hi)) = owned {
        groups.push(vec![lo, hi]);
    }
    groups.extend(
        (0..n)
            .filter(|i| !owned.is_some_and(|(lo, hi)| *i == lo || *i == hi))
            .map(|i| vec![i]),
    );
    for group in &groups {
        let types: Vec<Iv> = group
            .iter()
            .map(|i| range::type_range(&tree.params[*i].ty).unwrap_or(Iv::point(0)))
            .collect();
        let max = types.iter().map(|t| t.hi).min().unwrap_or(0);
        let min = types.iter().map(|t| t.lo).max().unwrap_or(0);
        for high in [true, false] {
            let limit = if high { max } else { -min };
            let set = |licence: &[Iv], v: i64| {
                let mut out = licence.to_vec();
                for i in group {
                    out[*i] = if high {
                        Iv::new(out[*i].lo, v)
                    } else {
                        Iv::new(-v, out[*i].hi)
                    };
                }
                out
            };
            let (mut good, mut bad) = (0i64, None);
            while good < limit {
                let next = (good * 2 + 1).min(limit);
                match prove(&set(&licence, next)) {
                    Ok(f) => (good, facts) = (next, f),
                    Err(_) => {
                        bad = Some(next);
                        break;
                    }
                }
            }
            if let Some(mut bad) = bad {
                while bad - good > 1 {
                    let mid = good + (bad - good) / 2;
                    match prove(&set(&licence, mid)) {
                        Ok(f) => (good, facts) = (mid, f),
                        Err(_) => bad = mid,
                    }
                }
            }
            licence = set(&licence, good);
        }
    }
    let facts = prove(&licence).map(|_| facts).map_err(Refusal::Range)?;
    Ok((licence, facts))
}

fn provenance() -> Provenance {
    Provenance {
        evidence: Evidence::Legality,
        source: Source::Analysis {
            name: range::NAME.to_owned(),
            version: range::VERSION,
        },
    }
}

/// Bounds and narrowing of every operator site of the recursion's own region
/// (callees are proved by the same analysis and re-derived with it).
/// P1's re-derived annotations (demand, fallback, moduli), then this
/// analysis' bounds and narrowing under `licence`.
pub(super) fn annotations(
    partition: &Partition,
    certificate: &Certificate,
    combine: Combine,
    facts: &range::Facts,
    licence: &[RangeGuard],
) -> Result<Vec<Annotated>, Rejection> {
    let base = normal_form::certificate::rederive(partition, certificate.region)?;
    let function = certificate.region.function;
    let mut out = Vec::new();
    for ((f, (start, end)), iv) in &facts.ops {
        if *f != function || iv.is_bottom() {
            continue;
        }
        let target = Target::Op(crate::front::Span {
            start: *start,
            end: *end,
        });
        let signedness = if iv.lo < 0 {
            Signedness::Signed
        } else {
            Signedness::Unsigned
        };
        out.push(Annotation::Bounds {
            target: target.clone(),
            low: iv.lo,
            high: iv.hi,
            signedness,
        });
        out.push(Annotation::Narrow {
            target,
            bits: 32,
            signedness,
            license: licence.to_vec(),
        });
    }
    if let Combine::Residue(p) = combine {
        out.push(Annotation::Bounds {
            target: Target::PathWeight,
            low: 0,
            high: i64::from(p) - 1,
            signedness: Signedness::Unsigned,
        });
    }
    Ok(base
        .into_iter()
        .chain(out.into_iter().map(|annotation| Annotated {
            annotation,
            provenance: provenance(),
        }))
        .collect())
}

/// This range analysis as P1's named, versioned [`normal_form::certificate::Analysis`]
/// for one certificate: it re-derives the bounds and narrowing under the
/// licence the certificate's `Narrow` annotations state, and derives nothing
/// for another region or a licence that does not keep every narrow site in
/// 32 bits.
pub struct RangeAnalysis<'a>(pub &'a Certificate);

impl normal_form::certificate::Analysis for RangeAnalysis<'_> {
    fn name(&self) -> &str {
        range::NAME
    }

    fn version(&self) -> u32 {
        range::VERSION
    }

    fn rederive(&self, partition: &Partition, region: regions::RegionId) -> Vec<Annotation> {
        if region != self.0.region {
            return Vec::new();
        }
        rederive(partition, self.0).unwrap_or_default()
    }
}

/// Re-derive this analysis' annotations of an L9/L10 certificate under the
/// licence its `Narrow` annotations state (the certificate checker's hook).
pub fn rederive(
    partition: &Partition,
    certificate: &Certificate,
) -> Result<Vec<Annotation>, &'static str> {
    if !matches!(certificate.law, LawId::L9PathSum | LawId::L10RangeSum)
        || certificate.subject != Subject::Recursion
    {
        return Err("range annotations on a non-recursion certificate");
    }
    let licence = certificate
        .annotations
        .iter()
        .find_map(|a| match &a.annotation {
            Annotation::Narrow { license, .. } => Some(license.clone()),
            _ => None,
        })
        .ok_or("no narrowing licence")?;
    let trees = trees(partition);
    let index = certificate.region.function;
    let tree = trees.get(index).copied().flatten().ok_or("region")?;
    if licence.len() != tree.params.len()
        || licence
            .iter()
            .zip(&tree.params)
            .any(|(g, p)| g.var != p.name)
    {
        return Err("licence does not name the parameters");
    }
    let combine = match certificate.guards.iter().find_map(|g| match g {
        normal_form::Guard::TableRange { modulus } => Some(*modulus),
        _ => None,
    }) {
        Some(p) => Combine::Residue(p),
        None => Combine::Sum,
    };
    let combines = combines(tree, certificate).map_err(|_| "combine facts")?;
    let owned = interval_pair(tree);
    let boxes: Vec<Iv> = licence.iter().map(|g| Iv::new(g.low, g.high)).collect();
    let spec = range::Spec {
        trees: &trees,
        root: index,
        owned,
        licence: &boxes,
        combines: &combines,
    };
    let facts = range::analyse(&spec)?;
    Ok(
        annotations(partition, certificate, combine, &facts, &licence)
            .map_err(|_| "P1 re-derivation")?
            .into_iter()
            .map(|a| a.annotation)
            .collect(),
    )
}
