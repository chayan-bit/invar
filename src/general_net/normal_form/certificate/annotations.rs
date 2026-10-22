//! Validation of a certificate's lowering annotations (#140 items 1-3, 6).
//!
//! Every annotation must be legality evidence and well formed against the
//! region: its target exists, bounds are ordered and agree with the target's
//! signedness, narrowing widths and licences are sound shapes. Demand order,
//! fallback state and modulus boundaries are then re-derived from the region
//! and must match exactly. P1 itself derives no bounds or narrowing: those,
//! and anything with [`Source::Analysis`], are verified only by the named,
//! versioned [`Analysis`] the caller supplies, which must re-derive the exact
//! annotation for the region. Without one the answer is
//! [`Rejection::Unverified`], never acceptance.

use std::collections::BTreeSet;

use super::Rejection;
use crate::{
    front::{BinaryOp, Span, Type},
    general_net::{
        normal_form::{
            Annotated, Annotation, Certificate, Evidence, LawId, RangeGuard, Signedness, Source,
            Target,
        },
        regions::{
            Partition, Region, RegionId,
            tree::{self, Expr, FunctionTree, LoopKind, Stmt},
        },
    },
};

/// An operator, index or store site; `modulus` is the literal `P >= 2` of a
/// `% P` (the rule of every ring and path sum: `% 1` is an ordinary site).
struct Site {
    span: Span,
    modulus: Option<u32>,
}

/// A range or narrowing analysis outside P1 (for example P4a's path-sum
/// analysis) that the checker consults for [`Source::Analysis`] annotations.
/// It is trusted compiler code: it must derive its annotations from the
/// partition alone, never from names of files, hashes or benchmarks.
pub trait Analysis {
    fn name(&self) -> &str;
    fn version(&self) -> u32;
    /// Every annotation the analysis derives for `region`.
    fn rederive(&self, partition: &Partition, region: RegionId) -> Vec<Annotation>;
}

pub(super) fn check(
    partition: &Partition,
    tree: &FunctionTree,
    certificate: &Certificate,
    analyses: &[&dyn Analysis],
) -> Result<(), Rejection> {
    if certificate.annotations.is_empty() {
        return Ok(());
    }
    let region = partition
        .region(certificate.region)
        .ok_or(Rejection::Region)?;
    let sites = sites(tree, certificate.region.head)?;
    let derived = rederive(partition, certificate.region)?;
    let mut unverified = false;
    for Annotated {
        annotation,
        provenance,
    } in &certificate.annotations
    {
        if provenance.evidence != Evidence::Legality {
            return Err(Rejection::Annotation("only legality evidence"));
        }
        let scope = Scope {
            region,
            tree,
            sites: &sites,
            law: certificate.law,
        };
        scope.well_formed(annotation)?;
        match &provenance.source {
            Source::Analysis { name, version } => {
                let Some(analysis) = analyses
                    .iter()
                    .find(|a| a.name() == name && a.version() == *version)
                else {
                    unverified = true;
                    continue;
                };
                if !analysis
                    .rederive(partition, certificate.region)
                    .contains(annotation)
                {
                    return Err(Rejection::Annotation("differs from the named analysis"));
                }
            }
            Source::Rederived => match annotation {
                Annotation::Bounds { .. } | Annotation::Narrow { .. } => unverified = true,
                _ => {
                    if !derived.iter().any(|d| d.annotation == *annotation) {
                        return Err(Rejection::Annotation("differs from the re-derived region"));
                    }
                }
            },
        }
    }
    if unverified {
        return Err(Rejection::Unverified);
    }
    Ok(())
}

/// The annotations of region `id` that the checker re-derives: its demand
/// order, its fallback state and one modulus boundary per literal `% P`
/// site. A lowering may attach these to any certificate of the region.
pub fn rederive(partition: &Partition, id: RegionId) -> Result<Vec<Annotated>, Rejection> {
    let region = partition.region(id).ok_or(Rejection::Region)?;
    let tree = super::tree_of(partition, id.function)?;
    let sites = sites(tree, id.head)?;
    let written: BTreeSet<&String> = region.array_writes.iter().collect();
    let vars: BTreeSet<String> = region
        .live_out
        .iter()
        .filter(|name| !written.contains(name))
        .cloned()
        .collect();
    let mut out = vec![
        Annotation::Demand(sites.iter().map(|site| site.span).collect()),
        Annotation::Fallback {
            vars: vars.into_iter().collect(),
            arrays: written.into_iter().cloned().collect(),
        },
    ];
    for site in &sites {
        if let Some(modulus) = site.modulus {
            out.push(Annotation::Modulus {
                target: Target::Op(site.span),
                modulus,
            });
        }
    }
    Ok(out.into_iter().map(Annotated::rederived).collect())
}

struct Scope<'a> {
    region: &'a Region,
    tree: &'a FunctionTree,
    sites: &'a [Site],
    law: LawId,
}

impl Scope<'_> {
    fn well_formed(&self, annotation: &Annotation) -> Result<(), Rejection> {
        let bad = |what| Err(Rejection::Annotation(what));
        match annotation {
            Annotation::Bounds {
                target,
                low,
                high,
                signedness,
            } => {
                self.target(target)?;
                if low > high {
                    return bad("bounds out of order");
                }
                if *signedness == Signedness::Unsigned && *low < 0 {
                    return bad("unsigned bounds below zero");
                }
                self.signedness(target, *signedness)
            }
            Annotation::Narrow {
                target,
                bits,
                signedness,
                license,
            } => {
                if !matches!(target, Target::Op(_) | Target::Var(_)) {
                    return bad("narrowing applies to an operator or a scalar");
                }
                self.target(target)?;
                if !matches!(bits, 8 | 16 | 32) {
                    return bad("narrow width");
                }
                self.signedness(target, *signedness)?;
                self.license(license)
            }
            Annotation::Demand(order) => {
                let unique: BTreeSet<_> = order.iter().map(|s| (s.start, s.end)).collect();
                if unique.len() != order.len() {
                    return bad("demand site repeated");
                }
                order
                    .iter()
                    .try_for_each(|span| self.target(&Target::Op(*span)))
            }
            Annotation::Fallback { vars, arrays } => {
                if !strictly_sorted(vars) || !strictly_sorted(arrays) {
                    return bad("fallback state not sorted and unique");
                }
                Ok(())
            }
            Annotation::Modulus { target, modulus } => {
                if !matches!(target, Target::Op(_)) || *modulus < 2 {
                    return bad("modulus boundary");
                }
                self.target(target)
            }
        }
    }

    fn target(&self, target: &Target) -> Result<(), Rejection> {
        let region = self.region;
        let exists = match target {
            Target::Var(name) => self.tree.locals.get(name).is_some_and(scalar),
            Target::Array(name) => {
                region.array_reads.contains(name) || region.array_writes.contains(name)
            }
            Target::Edge(from, to) => region.blocks.contains(from) && region.blocks.contains(to),
            Target::PathWeight => self.law == LawId::L9PathSum,
            Target::Op(span) => self.sites.iter().any(|site| site.span == *span),
        };
        if exists {
            Ok(())
        } else {
            Err(Rejection::Annotation("target not in the region"))
        }
    }

    fn signedness(&self, target: &Target, signedness: Signedness) -> Result<(), Rejection> {
        let declared = match target {
            Target::Var(name) => self.tree.locals.get(name),
            _ => None,
        };
        let agrees = match declared {
            Some(Type::I32) => signedness == Signedness::Signed,
            Some(Type::U32) => signedness == Signedness::Unsigned,
            Some(_) => false,
            None => true,
        };
        if agrees {
            Ok(())
        } else {
            Err(Rejection::Annotation(
                "signedness differs from the declared type",
            ))
        }
    }

    fn license(&self, license: &[RangeGuard]) -> Result<(), Rejection> {
        let mut seen = BTreeSet::new();
        for guard in license {
            if !self.region.live_in.contains(&guard.var)
                || guard.low > guard.high
                || !seen.insert(&guard.var)
            {
                return Err(Rejection::Annotation("narrowing licence"));
            }
        }
        Ok(())
    }
}

fn scalar(ty: &Type) -> bool {
    matches!(ty, Type::I32 | Type::U32 | Type::F32 | Type::Bool)
}

fn strictly_sorted(names: &[String]) -> bool {
    names.windows(2).all(|pair| pair[0] < pair[1])
}

/// The region's sites in demand order: statements in source pre-order, each
/// statement's expressions in evaluation order, operands before operators.
fn sites(tree: &FunctionTree, head: Option<crate::mir::BlockId>) -> Result<Vec<Site>, Rejection> {
    let mut out = Vec::new();
    let body = match head {
        None => &tree.body[..],
        Some(head) => {
            let inner = tree::find_loop(&tree.body, head).ok_or(Rejection::Region)?;
            if let LoopKind::While { cond } = &inner.kind {
                expr_sites(cond, &mut out);
            }
            &inner.body[..]
        }
    };
    tree::walk(body, &mut Vec::new(), 0, &mut |stmt, _, _| {
        for expr in tree::stmt_exprs(stmt) {
            expr_sites(expr, &mut out);
        }
        if let Stmt::Store { span, .. } = stmt {
            out.push(Site {
                span: *span,
                modulus: None,
            });
        }
    });
    Ok(out)
}

fn expr_sites(expr: &Expr, out: &mut Vec<Site>) {
    match expr {
        Expr::Index { index, span, .. } => {
            expr_sites(index, out);
            out.push(Site {
                span: *span,
                modulus: None,
            });
        }
        Expr::Unary { value, span, .. } => {
            expr_sites(value, out);
            out.push(Site {
                span: *span,
                modulus: None,
            });
        }
        Expr::Binary {
            op,
            left,
            right,
            span,
        } => {
            expr_sites(left, out);
            expr_sites(right, out);
            let modulus = (*op == BinaryOp::Rem)
                .then(|| right.literal())
                .flatten()
                .and_then(|p| u32::try_from(p).ok())
                .filter(|p| *p > 1);
            out.push(Site {
                span: *span,
                modulus,
            });
        }
        Expr::Logic { left, right, .. } => {
            expr_sites(left, out);
            expr_sites(right, out);
        }
        Expr::Call { args, .. } => args.iter().for_each(|arg| expr_sites(arg, out)),
        Expr::ArrayLit { items, .. } => items.iter().for_each(|item| expr_sites(item, out)),
        _ => {}
    }
}
