//! P3: static family admission over checked PNF nodes and their net-derived
//! bodies. This is analysis, not a live guard, device plan or publication permit.
//! No source recognizer or A1 adapter is called here. Registry precedence and
//! live bounds/alias/fuel/numeric checks remain the dispatcher's responsibility.

mod core;
mod grid;
mod indexed;
mod segmented;
mod summary;

use super::{
    RegionId,
    tree::{self, Expr, FunctionTree, Loop, LoopKind, Stmt},
};
use crate::{
    front::{BinaryOp, Mode, Span, Type},
    general_net::{
        Net, Proof,
        normal_form::{self, LoopNode, Node, NormalForm, certificate},
    },
    mir,
};

/// Static subfamilies, in A0 registry order. More than one can match; admission
/// never chooses a different family after a live refusal.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Family {
    Map,
    FusedMap,
    Dag,
    Reduce,
    Scan,
    Transpose,
    Stencil,
    Tensor,
    SegmentedMap,
    SegmentedReduce,
    Compaction,
    Gather,
    Scatter,
    AffineScatter,
    Histogram,
    Frontier,
    Summary,
}

impl Family {
    pub const ALL: [Self; 17] = [
        Self::Map,
        Self::FusedMap,
        Self::Dag,
        Self::Scan,
        Self::Reduce,
        Self::Transpose,
        Self::Stencil,
        Self::Tensor,
        Self::SegmentedMap,
        Self::SegmentedReduce,
        Self::Compaction,
        Self::Gather,
        Self::Scatter,
        Self::AffineScatter,
        Self::Histogram,
        Self::Frontier,
        Self::Summary,
    ];
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Admission {
    pub region: RegionId,
    /// Original MIR loop span, usable with the A0 reached-loop selector.
    pub span: Span,
    pub family: Family,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Report {
    pub fingerprint: [u8; 32],
    pub admissions: Vec<Admission>,
    /// Recognized shapes deliberately left on the native path because the PNF
    /// cannot certify them. A refusal grants no parallel execution authority.
    pub refusals: Vec<Refusal>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RefusalReason {
    MissingFiniteStateLaw,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Refusal {
    pub candidate: Admission,
    pub reason: RefusalReason,
}

#[derive(Debug, PartialEq, Eq)]
pub enum Error {
    Certificate(certificate::Rejection),
    Regions(super::Error),
    Net(crate::general_net::Error),
    /// A checked certificate alone is not permission to change the loop node,
    /// omit a loop or invent a new PNF classification.
    FormMismatch,
}

/// Check the entire presented form before any predicate runs. Predicates inspect
/// `(node, representation, body op set)` and the function's type/ownership facts.
/// Serial/opaque nodes and unsupported representations yield no admission,
/// except a serial round wrapper whose frontier and copy children are checked.
/// Cost, live guard results and backend availability are deliberately absent.
pub fn analyze(net: &Net, proof: &Proof, form: &NormalForm) -> Result<Report, Error> {
    certificate::check_all(net, proof, form).map_err(Error::Certificate)?;
    let partition = super::partition(net, proof).map_err(Error::Regions)?;
    // P1's checker verifies certificates, not equality with the extractor's
    // classification. Bind the node and loop inventory too, without rejecting
    // additional independently checked lowering annotations.
    let mut classification = form.clone();
    for function in &mut classification.functions {
        for cert in &mut function.certificates {
            cert.annotations.clear();
        }
        for loop_form in &mut function.loops {
            for cert in &mut loop_form.certificates {
                cert.annotations.clear();
            }
        }
    }
    if classification != normal_form::from_partition(&partition) {
        return Err(Error::FormMismatch);
    }
    let program = net.compiled_mir(proof).map_err(Error::Net)?;
    let mut admissions = Vec::new();
    let mut refusals = Vec::new();
    for function in &form.functions {
        let Some(tree) = partition
            .function(function.index)
            .and_then(|f| f.tree.as_ref().ok())
        else {
            continue;
        };
        for loop_form in &function.loops {
            let Some(head) = loop_form.region.head else {
                continue;
            };
            let Some(inner) = tree::find_loop(&tree.body, head) else {
                continue;
            };
            let Some(lowered) = program.functions.get(function.index) else {
                continue;
            };
            let Some(mir::Terminator::Branch { span, .. }) =
                lowered.blocks.get(head).and_then(|b| b.end.as_ref())
            else {
                continue;
            };
            let Node::Parallel(node) = &loop_form.node else {
                if summary::table(tree, lowered, inner) {
                    refusals.push(Refusal {
                        candidate: Admission {
                            region: loop_form.region,
                            span: *span,
                            family: Family::Summary,
                        },
                        reason: RefusalReason::MissingFiniteStateLaw,
                    });
                }
                if indexed::frontier(tree, inner) && indexed::frontier_laws(tree, inner, form) {
                    admissions.push(Admission {
                        region: loop_form.region,
                        span: *span,
                        family: Family::Frontier,
                    });
                }
                continue;
            };
            let context = Context {
                tree,
                lowered,
                inner,
                node,
                form,
            };
            for family in Family::ALL {
                if context.admits(family) {
                    admissions.push(Admission {
                        region: loop_form.region,
                        span: *span,
                        family,
                    });
                }
            }
        }
    }
    Ok(Report {
        fingerprint: form.fingerprint,
        admissions,
        refusals,
    })
}

struct Context<'a> {
    tree: &'a FunctionTree,
    lowered: &'a mir::Function,
    inner: &'a Loop,
    node: &'a LoopNode,
    form: &'a NormalForm,
}

impl Context<'_> {
    fn admits(&self, family: Family) -> bool {
        if self.node.first_exit {
            return false;
        }
        match family {
            Family::Map => core::map(self),
            Family::FusedMap => core::fusion(self, false),
            Family::Dag => core::fusion(self, true),
            Family::Reduce => core::reduce(self),
            Family::Scan => core::scan(self),
            Family::Transpose => grid::transpose(self),
            Family::Stencil => grid::stencil(self),
            Family::Tensor => grid::tensor(self),
            Family::SegmentedMap => segmented::map(self),
            Family::SegmentedReduce => segmented::reduce(self),
            Family::Compaction => segmented::compaction(self),
            Family::Gather => indexed::gather_scatter(self, false),
            Family::Scatter => indexed::gather_scatter(self, true),
            Family::AffineScatter => indexed::affine(self),
            Family::Histogram => indexed::histogram(self),
            Family::Frontier => {
                indexed::frontier(self.tree, self.inner)
                    && indexed::frontier_laws(self.tree, self.inner, self.form)
            }
            Family::Summary => summary::admit(self),
        }
    }

    fn mutable(&self, name: &str, ty: Type) -> bool {
        self.lowered
            .locals
            .get(name)
            .is_some_and(|l| l.ty == ty && l.mutable)
    }

    fn ty(&self, name: &str, ty: Type) -> bool {
        self.tree.locals.get(name) == Some(&ty)
    }

    fn params(&self, types: &[(Type, Option<Mode>)], result: Type) -> bool {
        params(self.tree, types, result)
    }

    fn parallel(&self, inner: &Loop) -> bool {
        self.form
            .loop_form(RegionId {
                function: self.tree.index,
                head: Some(inner.head),
            })
            .is_some_and(|l| matches!(l.node, Node::Parallel(_)))
    }
}

fn params(tree: &FunctionTree, types: &[(Type, Option<Mode>)], result: Type) -> bool {
    tree.result == result
        && tree.params.len() == types.len()
        && tree
            .params
            .iter()
            .zip(types)
            .all(|(p, (ty, mode))| p.ty == *ty && p.mode == *mode)
}

fn named(e: &Expr, n: &str) -> bool {
    matches!(e, Expr::Local(x) if x == n)
}
fn uint(e: &Expr, n: u32) -> bool {
    matches!(e, Expr::U32(x) if *x == n)
}
fn binary(e: &Expr, op: BinaryOp) -> Option<(&Expr, &Expr)> {
    match e {
        Expr::Binary {
            op: actual,
            left,
            right,
            ..
        } if *actual == op => Some((left, right)),
        _ => None,
    }
}
fn call<'a>(e: &'a Expr, n: &str) -> Option<&'a [Expr]> {
    match e {
        Expr::Call { name, args, .. } if name == n => Some(args),
        _ => None,
    }
}
fn index<'a>(e: &'a Expr, n: &str) -> Option<&'a Expr> {
    match e {
        Expr::Index { array, index, .. } if array == n => Some(index),
        _ => None,
    }
}
fn element(e: &Expr, array: &str, i: &str) -> bool {
    index(e, array).is_some_and(|x| named(x, i))
}
fn len(e: &Expr, array: &str) -> bool {
    matches!(e, Expr::Len(n) if n == array)
}
fn product(e: &Expr, a: &str, b: &str) -> bool {
    binary(e, BinaryOp::Mul).is_some_and(|(l, r)| named(l, a) && named(r, b))
}
fn flat(e: &Expr, i: &str, stride: &str, j: &str) -> bool {
    binary(e, BinaryOp::Add).is_some_and(|(l, r)| product(l, i, stride) && named(r, j))
}
fn counted(inner: &Loop) -> Option<(&str, &Expr)> {
    match &inner.kind {
        LoopKind::For {
            var,
            lo: Expr::U32(0),
            hi,
        } => Some((var, hi)),
        _ => None,
    }
}
fn bound<'a>(inner: &'a Loop, upper: &str) -> Option<&'a str> {
    let (i, hi) = counted(inner)?;
    named(hi, upper).then_some(i)
}
fn as_loop(s: &Stmt) -> Option<&Loop> {
    if let Stmt::Loop(l) = s { Some(l) } else { None }
}
fn binding(s: &Stmt) -> Option<(&str, &Expr)> {
    if let Stmt::Let { name, value } = s {
        Some((name, value))
    } else {
        None
    }
}
fn assign<'a>(s: &'a Stmt, n: &str) -> Option<&'a Expr> {
    if let Stmt::Assign { name, value } = s
        && name == n
    {
        Some(value)
    } else {
        None
    }
}
fn returned(s: &Stmt, n: &str) -> bool {
    matches!(s, Stmt::Return { value: Some(Expr::Move(x) | Expr::Local(x) | Expr::Owner { name: x, .. }), .. } if x == n)
}
fn allocated<'a>(s: &'a Stmt, allocator: &str) -> Option<(&'a str, &'a Expr)> {
    let (name, value) = binding(s)?;
    let [length, fill] = call(value, allocator)? else {
        return None;
    };
    let zero = if allocator == "filled_f32" {
        matches!(fill, Expr::F32(x) if *x == crate::numeric_contract::F32Bits::POS_ZERO)
    } else {
        uint(fill, 0)
    };
    zero.then_some((name, length))
}
/// Exactly one fresh owner, this outer loop, then the returned owner.
fn fresh<'a>(cx: &'a Context<'_>, allocator: &str) -> Option<(&'a str, &'a Expr)> {
    let [alloc, stmt, ret] = cx.tree.body.as_slice() else {
        return None;
    };
    let inner = as_loop(stmt)?;
    let (out, length) = allocated(alloc, allocator)?;
    (inner.head == cx.inner.head
        && returned(ret, out)
        && cx.lowered.locals.get(out).is_some_and(|l| l.mutable))
    .then_some((out, length))
}
