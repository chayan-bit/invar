//! Independent certificate checker (normal-form design section 6, decision L1).
//!
//! A certificate is accepted only if every fact it states is re-derived from
//! the current net: the compiled-MIR fingerprint, the region, the subject's
//! update or store sites, the representation (ring, modulus, order,
//! companions), the exact runtime-guard set and the stated facts. The checker
//! shares only the net-derived statement tree with the extractor; its
//! algebra is the polynomial normal form of `poly`, not the extractor's
//! syntactic linear forms, and it never looks at names of files, hashes or
//! benchmarks. [`check_all`] additionally checks that a normal form's
//! certificates are complete for every parallel region.

mod annotations;
mod fsm;
mod function;
mod loops;
mod view;

pub use annotations::{Analysis, rederive};

use std::collections::BTreeSet;

use super::{ArrayLaw, Certificate, LawId, Node, NormalForm, Recursion, Rep, Subject, VERSION};
use crate::general_net::{
    Net, Proof,
    regions::{self, Partition, tree::FunctionTree},
};

/// Why a certificate (or a normal form) is rejected.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Rejection {
    Net(String),
    Version,
    Fingerprint,
    /// A lowering annotation is malformed, not legality evidence, or differs
    /// from the one re-derived from the region.
    Annotation(&'static str),
    /// Well-formed bounds or narrowing that need range inference, which v1
    /// does not have; never accepted.
    Unverified,
    /// The region does not exist, or its function is not structured.
    Region,
    /// The subject does not exist in the region, or is the wrong kind.
    Subject,
    /// The law's conditions do not hold for the subject.
    Law(&'static str),
    /// The representation does not match the re-derived updates.
    Rep(&'static str),
    /// The guard set differs from the one the law requires.
    Guards,
    /// A stated fact differs from the re-derived one.
    Facts,
    /// The normal form's certificates do not cover a parallel region.
    Incomplete(String),
}

/// Check one certificate against the current net.
pub fn check(net: &Net, proof: &Proof, certificate: &Certificate) -> Result<(), Rejection> {
    let partition = regions::partition(net, proof).map_err(|e| Rejection::Net(format!("{e:?}")))?;
    check_with(&partition, certificate)
}

/// Check one certificate against an already re-derived partition.
pub fn check_with(partition: &Partition, certificate: &Certificate) -> Result<(), Rejection> {
    check_with_analyses(partition, certificate, &[])
}

/// [`check_with`], verifying [`super::Source::Analysis`] annotations with the
/// matching named, versioned analysis in `analyses`.
pub fn check_with_analyses(
    partition: &Partition,
    certificate: &Certificate,
    analyses: &[&dyn Analysis],
) -> Result<(), Rejection> {
    if certificate.version != VERSION {
        return Err(Rejection::Version);
    }
    if certificate.fingerprint != partition.fingerprint {
        return Err(Rejection::Fingerprint);
    }
    let tree = tree_of(partition, certificate.region.function)?;
    annotations::check(partition, tree, certificate, analyses)?;
    match certificate.region.head {
        None => function::check(partition, tree, certificate),
        Some(head) => {
            let inner = regions::tree::find_loop(&tree.body, head).ok_or(Rejection::Region)?;
            partition
                .region(certificate.region)
                .ok_or(Rejection::Region)?;
            loops::check(partition, tree, inner, certificate)
        }
    }
}

pub(super) fn tree_of(partition: &Partition, function: usize) -> Result<&FunctionTree, Rejection> {
    partition
        .function(function)
        .and_then(|f| f.tree.as_ref().ok())
        .ok_or(Rejection::Region)
}

/// Check every certificate of `form` and its completeness: each parallel loop
/// is covered exactly by its certificates (every carried scalar and written
/// array has one, in a dependency-respecting order), and every function law
/// has its certificate. Returns the number of certificates checked.
pub fn check_all(net: &Net, proof: &Proof, form: &NormalForm) -> Result<usize, Rejection> {
    check_all_with_analyses(net, proof, form, &[])
}

/// [`check_all`] with named analyses for [`super::Source::Analysis`]
/// annotations (see [`check_with_analyses`]).
pub fn check_all_with_analyses(
    net: &Net,
    proof: &Proof,
    form: &NormalForm,
    analyses: &[&dyn Analysis],
) -> Result<usize, Rejection> {
    let partition = regions::partition(net, proof).map_err(|e| Rejection::Net(format!("{e:?}")))?;
    if form.version != VERSION {
        return Err(Rejection::Version);
    }
    if form.fingerprint != partition.fingerprint
        || form.functions.len() != partition.functions.len()
    {
        return Err(Rejection::Fingerprint);
    }
    let mut count = 0;
    for certificate in form.certificates() {
        check_with_analyses(&partition, certificate, analyses)?;
        count += 1;
    }
    for function in &form.functions {
        let tree = partition
            .function(function.index)
            .and_then(|f| f.tree.as_ref().ok());
        let Some(tree) = tree else {
            if function.loops.is_empty() && function.certificates.is_empty() {
                continue;
            }
            return Err(Rejection::Incomplete(format!(
                "{}: laws on an opaque function",
                function.name
            )));
        };
        function_complete(function, tree)?;
        for loop_form in &function.loops {
            if loop_form.region.function != function.index {
                return Err(Rejection::Incomplete(
                    "loop form in the wrong function".to_owned(),
                ));
            }
            match &loop_form.node {
                Node::Serial { .. } if loop_form.certificates.is_empty() => {}
                Node::Serial { .. } => {
                    return Err(Rejection::Incomplete(
                        "certificates on a serial loop".to_owned(),
                    ));
                }
                Node::Parallel(node) => {
                    loops::complete(&partition, tree, function, loop_form, node)?
                }
            }
        }
    }
    Ok(count)
}

fn function_complete(function: &super::FunctionForm, tree: &FunctionTree) -> Result<(), Rejection> {
    let laws: Vec<(LawId, &Subject)> = function
        .certificates
        .iter()
        .map(|c| (c.law, &c.subject))
        .collect();
    let recursion_law = laws
        .iter()
        .find(|(_, s)| **s == Subject::Recursion)
        .map(|(l, _)| *l);
    let expected = match &function.recursion {
        Some(Recursion::PathSum { .. }) => Some(LawId::L9PathSum),
        Some(Recursion::RangeSum) => Some(LawId::L10RangeSum),
        None => None,
    };
    if recursion_law != expected {
        return Err(Rejection::Incomplete(format!(
            "{}: recursion law",
            function.name
        )));
    }
    if let Some(Recursion::PathSum { modulus }) = &function.recursion {
        let stated = function.certificates.iter().any(|c| {
            c.guards
                .contains(&super::Guard::TableRange { modulus: *modulus })
        });
        if !stated {
            return Err(Rejection::Incomplete(format!(
                "{}: path-sum modulus",
                function.name
            )));
        }
    }
    let forks: BTreeSet<(usize, usize, usize)> = function
        .forks
        .iter()
        .map(|f| (f.start.start, f.start.end, f.siblings))
        .collect();
    let certified: BTreeSet<(usize, usize, usize)> = function
        .certificates
        .iter()
        .filter_map(|c| match (&c.subject, c.facts.as_slice()) {
            (Subject::Fork(span), [super::Fact::Siblings(n)]) => Some((span.start, span.end, *n)),
            _ => None,
        })
        .collect();
    if forks != certified {
        return Err(Rejection::Incomplete(format!(
            "{}: fork sites",
            function.name
        )));
    }
    if function
        .certificates
        .iter()
        .any(|c| c.region.head.is_some() || c.region.function != tree.index)
    {
        return Err(Rejection::Incomplete(format!(
            "{}: function certificate region",
            function.name
        )));
    }
    Ok(())
}

/// The certified law of an array in a loop node.
pub(super) fn array_law_id(law: &ArrayLaw) -> LawId {
    match law {
        ArrayLaw::Own => LawId::L1Map,
        ArrayLaw::Combine { .. } => LawId::L11ScatterCombine,
        ArrayLaw::Ordered => LawId::L12OrderedWrite,
        ArrayLaw::Compaction { .. } => LawId::L13Compaction,
        ArrayLaw::Monotone { .. } => LawId::L14MonotoneFrontier,
    }
}

/// The law a carried representation is certified under.
pub(super) fn rep_law(rep: &Rep) -> LawId {
    match rep {
        Rep::Sealed(_) => LawId::L2SealedFold,
        Rep::LiftedSum(_) | Rep::SaturatedProduct(_) => LawId::L3TrapLiftedFold,
        Rep::Affine(_) => LawId::L4Affine,
        Rep::Matrix { .. } => LawId::L5Matrix,
        Rep::Tropical { .. } => LawId::L6Tropical,
        Rep::Select { .. } => LawId::L7Select,
        Rep::Last => LawId::L12OrderedWrite,
        Rep::Counter(_) => LawId::L13Compaction,
        Rep::Lifted(super::RepExpr::FunctionTable { .. }) => LawId::L8FiniteState,
    }
}
