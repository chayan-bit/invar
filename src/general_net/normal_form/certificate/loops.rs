//! Loop certificates: L1 (iterations and owned arrays), L2-L7 carried
//! representations, L11-L14 array laws, L15 first exit and L19 fusion, each
//! re-derived on the checker's [`View`] with the polynomial normal form.

use std::collections::{BTreeMap, BTreeSet};

use super::{
    Rejection, array_law_id, rep_law,
    view::{self, Update, View, names},
};
use crate::{
    front::{BinaryOp, Type},
    general_net::{
        normal_form::{
            ArrayLaw, Certificate, Fact, FunctionForm, Guard, IntTy, LawId, LoopForm, LoopNode,
            Node, Order, Rep, Ring, ScalarTy, Subject,
            poly::{Arith, Poly, local_key, normalize},
        },
        regions::{
            self, Partition, RegionId,
            tree::{self, Expr, FunctionTree, Loop, LoopKind, Stmt, same},
        },
    },
    laws::Law,
};

/// Matrix groups larger than this are outside the law set (#133 L5, #281).
const MAX_MATRIX: usize = 16;

pub(super) fn check(
    partition: &Partition,
    tree: &FunctionTree,
    inner: &Loop,
    cert: &Certificate,
) -> Result<(), Rejection> {
    let view = View::new(tree, inner)?;
    match (&cert.law, &cert.subject) {
        (LawId::L1Map, Subject::Iterations) => iterations(partition, &view, cert),
        (LawId::L8FiniteState, Subject::Var(var)) => super::fsm::check(&view, var, cert),
        (_, Subject::Var(var)) => carried(&view, var, cert),
        (_, Subject::Array(name)) => array(&view, name, cert),
        (LawId::L15FirstExit, Subject::Exit) => {
            bare(cert)?;
            view.returns
                .then_some(())
                .ok_or(Rejection::Law("no early return"))
        }
        (LawId::L19MapFusion, Subject::Fused(second)) => {
            fused(partition, tree, &view, inner, *second, cert)
        }
        _ => Err(Rejection::Subject),
    }
}

/// No representation, guards or facts.
fn bare(cert: &Certificate) -> Result<(), Rejection> {
    if cert.rep.is_some() {
        return Err(Rejection::Rep("unexpected representation"));
    }
    if !cert.guards.is_empty() {
        return Err(Rejection::Guards);
    }
    if !cert.facts.is_empty() {
        return Err(Rejection::Facts);
    }
    Ok(())
}

/// The exact guard set, order-insensitive and without duplicates.
pub(super) fn guards_are(cert: &Certificate, expected: &[Guard]) -> Result<(), Rejection> {
    let unique = cert
        .guards
        .iter()
        .enumerate()
        .all(|(i, g)| !cert.guards[..i].contains(g));
    let same_set =
        cert.guards.len() == expected.len() && expected.iter().all(|g| cert.guards.contains(g));
    if unique && same_set {
        Ok(())
    } else {
        Err(Rejection::Guards)
    }
}

/// L1 legality: no host effect, no `mut` loan, no outer owner moved, only pure
/// callees, and an induction the body never assigns.
fn iterations(partition: &Partition, view: &View<'_>, cert: &Certificate) -> Result<(), Rejection> {
    bare(cert)?;
    let region = partition.region(view.id).ok_or(Rejection::Region)?;
    let effects = &region.effects;
    if effects.host || !effects.mut_loans.is_empty() {
        return Err(Rejection::Law("host effect or mut loan in the loop"));
    }
    if region
        .ownership
        .moved
        .iter()
        .any(|m| !region.ownership.declared.contains(m))
    {
        return Err(Rejection::Law("outer owner moved in the loop"));
    }
    let impure = effects
        .calls
        .iter()
        .any(|&callee| !partition.function(callee).is_some_and(|f| f.pure));
    if impure {
        return Err(Rejection::Law("impure callee"));
    }
    if view.assigned.contains(&view.induction) {
        return Err(Rejection::Law("induction assigned in the body"));
    }
    Ok(())
}

// ----- carried scalars -------------------------------------------------------

fn int_ty(ty: Option<&Type>) -> Option<IntTy> {
    match ty? {
        Type::I32 => Some(IntTy::I32),
        Type::U32 => Some(IntTy::U32),
        _ => None,
    }
}

pub(super) fn mentions(poly: &Poly, vars: &BTreeSet<String>) -> bool {
    let keys: Vec<String> = vars.iter().map(|v| local_key(v)).collect();
    poly.mentions(&|atom| keys.iter().any(|k| atom.contains(k.as_str())))
}

pub(super) fn expr_mentions(expr: &Expr, vars: &BTreeSet<String>) -> bool {
    names(expr).iter().any(|n| vars.contains(n))
}

pub(super) fn one(var: &str) -> BTreeSet<String> {
    BTreeSet::from([var.to_owned()])
}

/// `value = c * var + rest` in `arith`, with `c` and `rest` free of `vars`.
fn affine(arith: Arith, value: &Expr, var: &str, vars: &BTreeSet<String>) -> Option<(Poly, Poly)> {
    let (c, rest) = normalize(arith, value)?.split_linear(&local_key(var))?;
    (!mentions(&c, vars) && !mentions(&rest, vars)).then_some((c, rest))
}

/// `if value <cmp> var { var = value }`: the innermost guard compares the
/// assigned value with the variable. Returns `(order, strict)`.
fn select_of(var: &str, update: &Update) -> Option<(Order, bool)> {
    if !update.exact {
        return None;
    }
    let value = update.value.as_ref()?;
    let (
        Some(Expr::Binary {
            op, left, right, ..
        }),
        true,
    ) = update.guards.last()?
    else {
        return None;
    };
    let strict = matches!(op, BinaryOp::Lt | BinaryOp::Gt);
    let less = match op {
        BinaryOp::Lt | BinaryOp::Le => true,
        BinaryOp::Gt | BinaryOp::Ge => false,
        _ => return None,
    };
    let local = Expr::Local(var.to_owned());
    let order = |min: bool| if min { Order::Min } else { Order::Max };
    if same(left, value) && **right == local {
        return Some((order(less), strict));
    }
    if same(right, value) && **left == local {
        return Some((order(!less), strict));
    }
    None
}

/// Guards of `update` (all, or all but the select comparison) avoid `vars`.
pub(super) fn guards_free(update: &Update, vars: &BTreeSet<String>, skip_last: bool) -> bool {
    let n = update.guards.len() - usize::from(skip_last && !update.guards.is_empty());
    update.guards[..n]
        .iter()
        .all(|(g, _)| g.as_ref().is_some_and(|g| !expr_mentions(g, vars)))
}

/// What one update of a single-variable representation is.
enum Step {
    Affine(Poly, Poly),
    Reset,
    Select(Order, bool),
}

fn steps(view: &View<'_>, var: &str, rep: &Rep) -> Result<Vec<Step>, Rejection> {
    let updates = view.updates.get(var).ok_or(Rejection::Subject)?;
    let own = one(var);
    let mut out = Vec::with_capacity(updates.len());
    for update in updates {
        let value = update
            .value
            .as_ref()
            .ok_or(Rejection::Rep("inexpressible update"))?;
        if let Some((order, strict)) = select_of(var, update)
            && matches!(rep, Rep::Select { .. } | Rep::Tropical { .. })
            && !expr_mentions(value, &own)
            && guards_free(update, &own, true)
        {
            out.push(Step::Select(order, strict));
            continue;
        }
        if !guards_free(update, &own, false) {
            return Err(bad_rep("update guarded by its own state"));
        }
        if !expr_mentions(value, &own) {
            out.push(Step::Reset);
            continue;
        }
        let (arith, inner) = arith_of(rep, value)?;
        let (c, rest) = affine(arith, inner, var, &own).ok_or(bad_rep("update not affine"))?;
        out.push(Step::Affine(c, rest));
    }
    Ok(out)
}

fn bad_rep(reason: &'static str) -> Rejection {
    Rejection::Rep(reason)
}

/// The arithmetic of a representation and the expression it applies to (the
/// operand of the outer `% P` for residues).
fn arith_of<'e>(rep: &Rep, value: &'e Expr) -> Result<(Arith, &'e Expr), Rejection> {
    let ring = match rep {
        Rep::Sealed(_) => Ring::Wrapping32,
        Rep::Affine(ring) | Rep::Matrix { ring, .. } => *ring,
        _ => return Ok((Arith::Exact, value)),
    };
    match ring {
        Ring::Wrapping32 => Ok((Arith::Wrap32, value)),
        Ring::Checked64(_) => Ok((Arith::Exact, value)),
        Ring::Residue { modulus, .. } => match value {
            Expr::Binary {
                op: BinaryOp::Rem,
                left,
                right,
                ..
            } if right.literal() == Some(i64::from(modulus)) && modulus > 1 => {
                Ok((Arith::Exact, left))
            }
            _ => Err(bad_rep("residue update without its modulus")),
        },
    }
}

fn ring_ty(ring: &Ring) -> Option<IntTy> {
    match ring {
        Ring::Wrapping32 => Some(IntTy::U32),
        Ring::Checked64(ty) | Ring::Residue { ty, .. } => Some(*ty),
    }
}

fn residue_guards(vars: &[String]) -> Vec<Guard> {
    vars.iter()
        .flat_map(|v| {
            [
                Guard::ReducedSeed(v.clone()),
                Guard::NonNegativeCoefficients(v.clone()),
            ]
        })
        .collect()
}

fn carried(view: &View<'_>, var: &str, cert: &Certificate) -> Result<(), Rejection> {
    if !view.carried().iter().any(|c| c == var) {
        return Err(Rejection::Subject);
    }
    let rep = cert.rep.as_ref().ok_or(bad_rep("missing representation"))?;
    if cert.law != rep_law(rep) {
        return Err(Rejection::Law("law and representation differ"));
    }
    let ty = view.tree.locals.get(var);
    if let Rep::Matrix { ring, vars } = rep {
        return matrix(view, var, *ring, vars, cert);
    }
    let steps = steps(view, var, rep)?;
    let all = |f: &dyn Fn(&Step) -> bool| steps.iter().all(f);
    let constant = |p: &Poly, k: i128| p.as_constant() == Some(k);
    let ok = match rep {
        Rep::Sealed(law) => {
            int_ty(ty) == Some(IntTy::U32)
                && match law {
                    Law::WrappingSum => all(&|s| matches!(s, Step::Affine(c, _) if constant(c, 1))),
                    Law::WrappingProduct => {
                        all(&|s| matches!(s, Step::Affine(_, r) if r.is_zero()))
                    }
                }
        }
        Rep::LiftedSum(t) | Rep::Counter(t) => {
            int_ty(ty) == Some(*t) && all(&|s| matches!(s, Step::Affine(c, _) if constant(c, 1)))
        }
        Rep::SaturatedProduct(t) => {
            int_ty(ty) == Some(*t) && all(&|s| matches!(s, Step::Affine(_, r) if r.is_zero()))
        }
        Rep::Affine(ring) => {
            int_ty(ty) == ring_ty(ring) && all(&|s| matches!(s, Step::Affine(..) | Step::Reset))
        }
        Rep::Tropical { order, ty: t } => {
            int_ty(ty) == Some(*t)
                && steps.iter().any(|s| matches!(s, Step::Select(..)))
                && all(&|s| match s {
                    Step::Select(o, _) => o == order,
                    Step::Affine(c, _) => constant(c, 1),
                    Step::Reset => true,
                })
        }
        Rep::Select {
            order,
            strict,
            elem,
            companions,
        } => {
            scalar_ty(ty) == Some(*elem)
                && all(&|s| matches!(s, Step::Select(o, st) if o == order && st == strict))
                && companions_ok(view, var, companions)
        }
        Rep::Last => all(&|s| matches!(s, Step::Reset)),
        Rep::Matrix { .. } | Rep::Lifted(_) => false,
    };
    if !ok {
        return Err(bad_rep("representation does not match the updates"));
    }
    if let Rep::Counter(_) = rep {
        let indexed = view
            .stores
            .values()
            .flatten()
            .any(|site| *site.index == Expr::Local(var.to_owned()));
        if !indexed {
            return Err(Rejection::Law("counter is not a store position"));
        }
    }
    let expected = match rep {
        Rep::Affine(Ring::Residue { .. }) => residue_guards(&[var.to_owned()]),
        _ => Vec::new(),
    };
    guards_are(cert, &expected)?;
    facts_match(view, var, rep, &cert.facts)
}

fn scalar_ty(ty: Option<&Type>) -> Option<ScalarTy> {
    match ty? {
        Type::I32 => Some(ScalarTy::I32),
        Type::U32 => Some(ScalarTy::U32),
        Type::F32 => Some(ScalarTy::F32),
        _ => None,
    }
}

/// Every companion is carried and assigned only inside the select's own arms,
/// from values that read neither the select variable nor any companion.
fn companions_ok(view: &View<'_>, var: &str, companions: &[String]) -> bool {
    let sorted = companions.windows(2).all(|w| w[0] < w[1]);
    let mut group: BTreeSet<String> = companions.iter().cloned().collect();
    group.insert(var.to_owned());
    let Some(selects) = view.updates.get(var) else {
        return false;
    };
    let carried = view.carried();
    sorted
        && companions.iter().all(|c| {
            c != var
                && carried.contains(c)
                && view.updates.get(c).is_some_and(|updates| {
                    updates.iter().all(|u| {
                        u.value.as_ref().is_some_and(|v| !expr_mentions(v, &group))
                            && selects.iter().any(|s| s.scope == u.scope)
                    })
                })
        })
}

/// L5: `2 <= k <= MAX_MATRIX` variables, every update jointly linear over the ring.
fn matrix(
    view: &View<'_>,
    lead: &str,
    ring: Ring,
    vars: &[String],
    cert: &Certificate,
) -> Result<(), Rejection> {
    let group: BTreeSet<String> = vars.iter().cloned().collect();
    let carried = view.carried();
    let well_formed = vars.windows(2).all(|w| w[0] < w[1])
        && (2..=MAX_MATRIX).contains(&vars.len())
        && vars.first().is_some_and(|v| v == lead)
        && vars
            .iter()
            .all(|v| carried.contains(v) && int_ty(view.tree.locals.get(v)) == ring_ty(&ring));
    if !well_formed {
        return Err(bad_rep("matrix group malformed"));
    }
    let rep = Rep::Matrix {
        ring,
        vars: vars.to_vec(),
    };
    for var in vars {
        for update in view.updates.get(var).into_iter().flatten() {
            let value = update
                .value
                .as_ref()
                .ok_or(bad_rep("inexpressible update"))?;
            if !guards_free(update, &group, false) {
                return Err(bad_rep("matrix update guarded by group state"));
            }
            let (arith, inner) = arith_of(&rep, value)?;
            let mut rest = normalize(arith, inner).ok_or(bad_rep("update not linear"))?;
            for member in vars {
                let (c, r) = rest
                    .split_linear(&local_key(member))
                    .ok_or(bad_rep("update not linear"))?;
                if mentions(&c, &group) {
                    return Err(bad_rep("coefficient reads group state"));
                }
                rest = r;
            }
            if mentions(&rest, &group) {
                return Err(bad_rep("update not linear"));
            }
        }
    }
    let expected = match ring {
        Ring::Residue { .. } => residue_guards(vars),
        _ => Vec::new(),
    };
    guards_are(cert, &expected)?;
    if cert.facts.is_empty() {
        Ok(())
    } else {
        Err(Rejection::Facts)
    }
}

/// The stated per-update facts equal the re-derived updates, in source order.
fn facts_match(view: &View<'_>, var: &str, rep: &Rep, facts: &[Fact]) -> Result<(), Rejection> {
    let updates = view.updates.get(var).ok_or(Rejection::Subject)?;
    if facts.len() != updates.len() {
        return Err(Rejection::Facts);
    }
    let own = one(var);
    for (fact, update) in facts.iter().zip(updates) {
        let value = update.value.as_ref().ok_or(Rejection::Facts)?;
        let ok = match fact {
            Fact::Select(e) => select_of(var, update).is_some() && same(e, value),
            Fact::Reset(e) => !expr_mentions(value, &own) && same(e, value),
            Fact::Affine {
                coefficient,
                offset,
            } => {
                let (arith, inner) = arith_of(rep, value)?;
                let stated = (|| {
                    let c = normalize(arith, coefficient)?;
                    let x = Poly::atom(arith, local_key(var));
                    c.mul(&x)?.add(&normalize(arith, offset)?)
                })();
                stated.is_some() && stated == normalize(arith, inner)
            }
            _ => false,
        };
        if !ok {
            return Err(Rejection::Facts);
        }
    }
    Ok(())
}

// ----- arrays ----------------------------------------------------------------

fn array(view: &View<'_>, name: &str, cert: &Certificate) -> Result<(), Rejection> {
    let sites = view.stores.get(name).ok_or(Rejection::Subject)?;
    if cert.rep.is_some() {
        return Err(bad_rep("unexpected representation"));
    }
    if view.owners.contains(name) {
        return Err(Rejection::Law("stored array used as an owner"));
    }
    let facts_ok = cert.facts.len() == sites.len()
        && cert.facts.iter().zip(sites).all(|(fact, site)| {
            matches!(fact, Fact::Store { index, value } if same(index, site.index) && same(value, site.value))
        });
    if !facts_ok {
        return Err(Rejection::Facts);
    }
    let reads = view.reads_of(name);
    let expected = match cert.law {
        LawId::L1Map => {
            owned(view, name).ok_or(Rejection::Law("stores are not iteration-owned"))?
        }
        LawId::L11ScatterCombine => {
            let elem_int = matches!(
                view.tree.locals.get(name),
                Some(Type::ArrayI32 | Type::ArrayU32)
            );
            let at_sites = reads.len() == sites.len()
                && reads.iter().all(|r| sites.iter().any(|s| same(s.index, r)));
            if !(elem_int && at_sites && combine_kind(view, name).is_some()) {
                return Err(Rejection::Law("not a commutative scatter-combine"));
            }
            Vec::new()
        }
        LawId::L12OrderedWrite if reads.is_empty() => Vec::new(),
        LawId::L13Compaction if reads.is_empty() && counter_of(view, name).is_some() => Vec::new(),
        LawId::L14MonotoneFrontier if monotone(view, name) => Vec::new(),
        _ => return Err(Rejection::Law("array law does not hold")),
    };
    guards_are(cert, &expected)
}

/// `Some(wrapping)` when every store is `a[k] = a[k] + e` (checked) or
/// `wrapping_add(a[k], e)`, all of one kind, with `e` not reading `a`.
fn combine_kind(view: &View<'_>, name: &str) -> Option<bool> {
    let mut kinds = BTreeSet::new();
    for site in view.stores.get(name)? {
        let target = |e: &Expr| matches!(e, Expr::Index { array, index, .. } if array == name && same(index, site.index));
        let (wrapping, a, b) = match site.value {
            Expr::Binary {
                op: BinaryOp::Add,
                left,
                right,
                ..
            } => (false, left.as_ref(), right.as_ref()),
            Expr::Call { name: f, args, .. } if f == "wrapping_add" => match args.as_slice() {
                [a, b] => (true, a, b),
                _ => return None,
            },
            _ => return None,
        };
        let other = if target(a) {
            b
        } else if target(b) {
            a
        } else {
            return None;
        };
        if other.mentions(name) {
            return None;
        }
        kinds.insert(wrapping);
    }
    match kinds.into_iter().collect::<Vec<_>>()[..] {
        [wrapping] => Some(wrapping),
        _ => None,
    }
}

/// The carried counter every store of `name` is indexed by.
fn counter_of(view: &View<'_>, name: &str) -> Option<String> {
    let carried = view.carried();
    let counters: BTreeSet<&String> = view
        .stores
        .get(name)?
        .iter()
        .map(|site| match site.index {
            Expr::Local(c) if carried.contains(c) => Some(c),
            _ => None,
        })
        .collect::<Option<_>>()?;
    match counters.into_iter().collect::<Vec<_>>()[..] {
        [counter] => Some(counter.clone()),
        _ => None,
    }
}

fn is_literal(expr: &Expr) -> bool {
    matches!(
        expr,
        Expr::I32(_) | Expr::U32(_) | Expr::F32(_) | Expr::Bool(_)
    )
}

/// L14: one constant value stored; if the loop also reads the array, a single
/// top-level own-index writer whose guards read only round-start state.
fn monotone(view: &View<'_>, name: &str) -> bool {
    let Some(sites) = view.stores.get(name) else {
        return false;
    };
    let Some(first) = sites.first() else {
        return false;
    };
    if !sites
        .iter()
        .all(|s| is_literal(s.value) && *s.value == *first.value)
    {
        return false;
    }
    view.reads_of(name).is_empty() || own_writer(view, name)
}

fn own_writer(view: &View<'_>, name: &str) -> bool {
    let Some([site]) = view.stores.get(name).map(Vec::as_slice) else {
        return false;
    };
    let induction = Expr::Local(view.induction.clone());
    if site.depth != 0 || *site.index != induction {
        return false;
    }
    let varying = view.varying();
    site.guards.iter().all(|guard| {
        let scalars_ok = names(guard).iter().all(|n| {
            *n == view.induction || n == name || view.stores.contains_key(n) || !varying.contains(n)
        });
        let mut reads = Vec::new();
        let mut owners = BTreeSet::new();
        collect_reads(guard, &mut reads, &mut owners);
        let reads_ok = reads.iter().all(|(array, index)| {
            if *array == name {
                **index == induction
            } else {
                !view.stores.contains_key(*array)
            }
        });
        scalars_ok && reads_ok && owners.is_empty()
    })
}

fn collect_reads<'e>(
    expr: &'e Expr,
    out: &mut Vec<(&'e str, &'e Expr)>,
    owners: &mut BTreeSet<String>,
) {
    match expr {
        Expr::Index { array, index, .. } => {
            out.push((array, index));
            collect_reads(index, out, owners);
        }
        Expr::Owner { name, .. } | Expr::Move(name) => {
            owners.insert(name.clone());
        }
        Expr::ArrayLit { items, .. } | Expr::Call { args: items, .. } => {
            items
                .iter()
                .for_each(|item| collect_reads(item, out, owners));
        }
        Expr::Unary { value, .. } => collect_reads(value, out, owners),
        Expr::Binary { left, right, .. } | Expr::Logic { left, right, .. } => {
            collect_reads(left, out, owners);
            collect_reads(right, out, owners);
        }
        _ => {}
    }
}

/// L1 footprint: the required stride guards when every store is
/// iteration-injective and every read is at a store index, or when every
/// access lies in the iteration's own block `[K i, K i + K)`.
fn owned(view: &View<'_>, name: &str) -> Option<Vec<Guard>> {
    let sites = view.stores.get(name)?;
    let reads = view.reads_of(name);
    let mut strides = BTreeSet::new();
    let injective = sites.iter().all(|site| match injective(view, site.index) {
        Some(Some(stride)) => {
            strides.insert(stride);
            true
        }
        Some(None) => true,
        None => false,
    });
    if injective && reads.iter().all(|r| sites.iter().any(|s| same(s.index, r))) {
        return Some(strides.into_iter().map(Guard::StrideNonZero).collect());
    }
    let k = block(view, sites.first()?.index)?;
    let mut all = sites.iter().map(|s| s.index).chain(reads);
    all.all(|ix| block(view, ix) == Some(k)).then(Vec::new)
}

/// Whether an atom may differ between iterations.
fn variant<'v>(view: &'v View<'_>) -> impl Fn(&str) -> bool + 'v {
    let varying: Vec<String> = view.varying().iter().map(|v| local_key(v)).collect();
    let written: Vec<String> = view
        .stores
        .keys()
        .map(|a| format!("array: {a:?}"))
        .collect();
    move |atom: &str| {
        varying
            .iter()
            .chain(&written)
            .any(|k| atom.contains(k.as_str()))
    }
}

/// `Some(None)` injective, `Some(Some(s))` injective when the invariant
/// stride `s` is not zero, `None` not shown injective.
fn injective(view: &View<'_>, index: &Expr) -> Option<Option<String>> {
    let (c, rest) = normalize(Arith::Exact, index)?.split_linear(&local_key(&view.induction))?;
    let variant = variant(view);
    if !c.mentions(&variant) && !rest.mentions(&variant) {
        match c.as_constant() {
            Some(0) => {}
            Some(_) => return Some(None),
            None => {
                if let Some(atom) = c.single_atom()
                    && let Some(name) = atom
                        .strip_prefix("Local(\"")
                        .and_then(|a| a.strip_suffix("\")"))
                {
                    return Some(Some(name.to_owned()));
                }
            }
        }
    }
    // Mixed radix `i + Q * hi` for `i in lo .. hi` (u32): `i` is the residue.
    if let Some((_, hi)) = view.bounds
        && c.as_constant() == Some(1)
        && let Some(h) = normalize(Arith::Exact, hi)
        && !h.mentions(&variant)
        && multiple_of(&rest, &h)
    {
        return Some(None);
    }
    // Mixed radix `i * K + w` for a nested `for w in 0 .. K`.
    if !c.mentions(&variant)
        && let Some(atom) = rest.single_atom()
        && let Some(w) = atom
            .strip_prefix("Local(\"")
            .and_then(|a| a.strip_suffix("\")"))
        && !view.assigned.contains(w)
        && let Some(extents) = view.nested.get(w)
        && extents.iter().all(|(lo, hi)| {
            lo.literal() == Some(0) && normalize(Arith::Exact, hi).as_ref() == Some(&c)
        })
    {
        return Some(None);
    }
    None
}

/// `rest` is `h * q` for some polynomial `q`.
fn multiple_of(rest: &Poly, h: &Poly) -> bool {
    if let Some(atom) = h.single_atom() {
        return rest.split_linear(atom).is_some_and(|(_, r)| r.is_zero());
    }
    match h.as_constant() {
        Some(k) if k > 0 => rest.terms().all(|(_, c)| c % k == 0),
        _ => false,
    }
}

/// `K` when `index = K * i + r` with literal `K > 0` and `0 <= r < K` for
/// every value of the nested `for` variables in `r`.
fn block(view: &View<'_>, index: &Expr) -> Option<i128> {
    let (c, rest) = normalize(Arith::Exact, index)?.split_linear(&local_key(&view.induction))?;
    let k = c.as_constant().filter(|k| *k > 0)?;
    let mut upper: i128 = 0;
    for (monomial, coefficient) in rest.terms() {
        if coefficient < 0 {
            return None;
        }
        let mut term = coefficient;
        for atom in monomial {
            let w = atom
                .strip_prefix("Local(\"")
                .and_then(|a| a.strip_suffix("\")"))?;
            if view.assigned.contains(w) {
                return None;
            }
            let extents = view.nested.get(w)?;
            let mut max = 0_i128;
            for (lo, hi) in extents {
                if lo.literal() != Some(0) {
                    return None;
                }
                max = max.max(i128::from(hi.literal()?) - 1);
            }
            term = term.checked_mul(max.max(0))?;
        }
        upper = upper.checked_add(term)?;
    }
    (upper < k).then_some(k)
}

// ----- L19 -------------------------------------------------------------------

/// The statement list directly containing the loop headed by `head`.
fn list_of(body: &[Stmt], head: crate::mir::BlockId) -> Option<&[Stmt]> {
    if body
        .iter()
        .any(|s| matches!(s, Stmt::Loop(l) if l.head == head))
    {
        return Some(body);
    }
    body.iter().find_map(|stmt| match stmt {
        Stmt::If { yes, no, .. } => list_of(yes, head).or_else(|| list_of(no, head)),
        Stmt::Loop(inner) => list_of(&inner.body, head),
        _ => None,
    })
}

fn fused(
    partition: &Partition,
    tree: &FunctionTree,
    first_view: &View<'_>,
    first: &Loop,
    second: RegionId,
    cert: &Certificate,
) -> Result<(), Rejection> {
    bare(cert)?;
    if second.function != tree.index {
        return Err(Rejection::Law("fusion across functions"));
    }
    let second_loop = view::loop_of(partition, tree, second)?;
    let list = list_of(&tree.body, first.head).ok_or(Rejection::Region)?;
    let at = list
        .iter()
        .position(|s| matches!(s, Stmt::Loop(l) if l.head == first.head))
        .ok_or(Rejection::Region)?;
    let rest = list.get(at + 1..).unwrap_or(&[]);
    let next = rest
        .iter()
        .position(|s| matches!(s, Stmt::Loop(_)))
        .ok_or(Rejection::Law("no adjacent loop"))?;
    if !matches!(rest.get(next), Some(Stmt::Loop(l)) if l.head == second_loop.head) {
        return Err(Rejection::Law("loops not adjacent"));
    }
    let between = rest.get(..next).unwrap_or(&[]);
    let second_view = View::new(tree, second_loop)?;
    let (
        LoopKind::For {
            var: va,
            lo: la,
            hi: ha,
        },
        LoopKind::For {
            var: vb,
            lo: lb,
            hi: hb,
        },
    ) = (&first.kind, &second_loop.kind)
    else {
        return Err(Rejection::Law("fusion needs for loops"));
    };
    let maps = [first_view, &second_view]
        .iter()
        .all(|v| v.carried().is_empty() && !v.returns);
    let equal_upper = same(ha, hb)
        || immutable_extent(tree, ha)
            .zip(immutable_extent(tree, hb))
            .is_some_and(|(a, b)| same(&a, &b));
    if !(same(la, lb) && equal_upper && maps) {
        return Err(Rejection::Law("loops are not equal-range maps"));
    }
    let aliases = between_ok(between).ok_or(Rejection::Law("statements between fused loops"))?;
    if footprints_ok(first_view, &second_view, va, vb, &aliases) {
        Ok(())
    } else {
        Err(Rejection::Law("fused footprints overlap"))
    }
}

/// Independently follow immutable bindings, with a bounded, cycle-free chain.
fn immutable_extent(tree: &FunctionTree, upper: &Expr) -> Option<Expr> {
    let mut extent = upper.clone();
    let mut seen = BTreeSet::new();
    for _ in 0..=16 {
        let owner = match &extent {
            Expr::U32(_) => return Some(extent),
            Expr::Len(owner) if seen.insert(owner.clone()) => owner.clone(),
            _ => return None,
        };
        let mut value = None;
        let mut invalid = false;
        tree::walk(
            &tree.body,
            &mut Vec::new(),
            0,
            &mut |stmt, _, _| match stmt {
                Stmt::Assign { name, .. } if *name == owner => invalid = true,
                Stmt::Let { name, value: e } if *name == owner => {
                    invalid |= value.is_some();
                    value = Some(e.clone());
                }
                _ => {}
            },
        );
        if invalid {
            return None;
        }
        extent = match value {
            None if tree.params.iter().any(|p| p.name == owner) => return Some(extent),
            Some(Expr::Move(from)) => Expr::Len(from),
            Some(Expr::ArrayLit { items, .. }) => Expr::U32(u32::try_from(items.len()).ok()?),
            Some(Expr::Call { name, args, .. })
                if matches!(name.as_str(), "filled_u32" | "filled_i32" | "filled_f32") =>
            {
                args.first()?.clone()
            }
            _ => return None,
        };
    }
    None
}

/// Allocations, literals and moves only, none reading the first loop's
/// outputs. Returns `alias -> owner` for the moves.
fn between_ok(between: &[Stmt]) -> Option<BTreeMap<String, String>> {
    let mut aliases = BTreeMap::new();
    for stmt in between {
        let Stmt::Let { name, value } = stmt else {
            return None;
        };
        match value {
            Expr::Move(owner) => {
                aliases.insert(name.clone(), owner.clone());
            }
            Expr::Call {
                name: callee, args, ..
            } if callee.starts_with("filled_") => {
                if !args.iter().all(clean) {
                    return None;
                }
            }
            e if is_literal(e) => {}
            _ => return None,
        }
    }
    Some(aliases)
}

/// Scalars, lengths and arithmetic only: no array element is read.
fn clean(expr: &Expr) -> bool {
    match expr {
        Expr::Local(_)
        | Expr::Len(_)
        | Expr::I32(_)
        | Expr::U32(_)
        | Expr::F32(_)
        | Expr::Bool(_) => true,
        Expr::Binary { left, right, .. } => clean(left) && clean(right),
        _ => false,
    }
}

fn footprints_ok(
    a: &View<'_>,
    b: &View<'_>,
    va: &str,
    vb: &str,
    aliases: &BTreeMap<String, String>,
) -> bool {
    let resolve = |n: &str| aliases.get(n).cloned().unwrap_or_else(|| n.to_owned());
    let writes_a: BTreeSet<String> = a.stores.keys().cloned().collect();
    let reads_a: BTreeSet<String> = a.reads.iter().map(|(n, _)| (*n).to_owned()).collect();
    let writes_b: BTreeSet<String> = b.stores.keys().map(|n| resolve(n)).collect();
    if writes_b
        .iter()
        .any(|n| writes_a.contains(n) || reads_a.contains(n))
    {
        return false;
    }
    let own_a = Expr::Local(va.to_owned());
    let a_own = a
        .stores
        .values()
        .flatten()
        .all(|s| s.depth == 0 && *s.index == own_a);
    let own_b = Expr::Local(vb.to_owned());
    let b_reads = b
        .reads
        .iter()
        .filter(|(n, _)| writes_a.contains(&resolve(n)))
        .all(|(_, ix)| **ix == own_b);
    let b_owners = b.owners.iter().all(|o| !writes_a.contains(&resolve(o)));
    a_own && b_reads && b_owners
}

// ----- completeness ----------------------------------------------------------

fn incomplete(what: &str) -> Rejection {
    Rejection::Incomplete(what.to_owned())
}

/// The loop's certificates cover exactly its carried state, written arrays,
/// exit and fusion, in a dependency-respecting order.
pub(super) fn complete(
    partition: &Partition,
    tree: &FunctionTree,
    function: &FunctionForm,
    form: &LoopForm,
    node: &LoopNode,
) -> Result<(), Rejection> {
    let inner = view::loop_of(partition, tree, form.region)?;
    let view = View::new(tree, inner)?;
    let certs = &form.certificates;
    if node.counting != regions::counting(inner) || certs.iter().any(|c| c.region != form.region) {
        return Err(incomplete("loop region"));
    }
    let count = |pred: &dyn Fn(&Certificate) -> bool| certs.iter().filter(|c| pred(c)).count();
    if count(&|c| c.law == LawId::L1Map && c.subject == Subject::Iterations) != 1 {
        return Err(incomplete("L1 iterations"));
    }
    carried_complete(&view, node, certs)?;
    arrays_complete(&view, node, certs)?;
    if node.first_exit != view.returns
        || count(&|c| c.subject == Subject::Exit) != usize::from(node.first_exit)
    {
        return Err(incomplete("first exit"));
    }
    let fusions = count(&|c| matches!(c.subject, Subject::Fused(_)));
    match node.fused_with {
        Some(second) => {
            let certified = count(&|c| c.subject == Subject::Fused(second)) == 1;
            let map = function.loops.iter().any(|f| {
                f.region == second
                    && matches!(&f.node, Node::Parallel(n) if n.carried.is_empty() && !n.first_exit
                        && n.arrays.iter().all(|(_, law)| *law == ArrayLaw::Own))
            });
            if !(certified && fusions == 1 && map) {
                return Err(incomplete("fusion"));
            }
        }
        None if fusions != 0 => return Err(incomplete("fusion")),
        None => {}
    }
    let expected =
        1 + node.carried.len() + node.arrays.len() + usize::from(node.first_exit) + fusions;
    if certs.len() != expected {
        return Err(incomplete("extra certificates"));
    }
    Ok(())
}

fn update_names(view: &View<'_>, var: &str) -> BTreeSet<String> {
    let mut out = BTreeSet::new();
    for update in view.updates.get(var).into_iter().flatten() {
        out.extend(update.value.iter().flat_map(names));
        for (guard, _) in &update.guards {
            out.extend(guard.iter().flat_map(names));
        }
    }
    out
}

fn carried_complete(
    view: &View<'_>,
    node: &LoopNode,
    certs: &[Certificate],
) -> Result<(), Rejection> {
    let carried: BTreeSet<String> = view.carried().into_iter().collect();
    let mut covered = BTreeSet::new();
    for c in &node.carried {
        let matching: Vec<&Certificate> = certs
            .iter()
            .filter(|k| k.subject == Subject::Var(c.var.clone()))
            .collect();
        let [cert] = matching[..] else {
            return Err(incomplete("carried certificate"));
        };
        if cert.law != c.law || cert.rep.as_ref() != Some(&c.rep) {
            return Err(incomplete("carried law"));
        }
        let mut group: Vec<String> = match &c.rep {
            Rep::Matrix { vars, .. } => vars.clone(),
            _ => vec![c.var.clone()],
        };
        if let Rep::Select { companions, .. } = &c.rep {
            group.extend(companions.iter().cloned());
        }
        let members: BTreeSet<String> = group.iter().cloned().collect();
        let mut deps = BTreeSet::new();
        for name in &group {
            if !covered.insert(name.clone()) {
                return Err(incomplete("carried variable covered twice"));
            }
            deps.extend(update_names(view, name));
        }
        if deps
            .iter()
            .any(|d| carried.contains(d) && !members.contains(d) && !covered.contains(d))
        {
            return Err(incomplete("carried dependency order"));
        }
        if !c.scan && read_elsewhere(view, &members) {
            return Err(incomplete("carried prefix read without a scan"));
        }
    }
    if covered != carried {
        return Err(incomplete("carried state"));
    }
    let var_certs = certs
        .iter()
        .filter(|k| matches!(k.subject, Subject::Var(_)))
        .count();
    if var_certs != node.carried.len() {
        return Err(incomplete("carried certificate"));
    }
    Ok(())
}

/// A group variable is read outside its own updates (its prefix is used):
/// anywhere but an assignment to a member or the condition of an `if` whose
/// arms only assign members.
fn read_elsewhere(view: &View<'_>, members: &BTreeSet<String>) -> bool {
    let mut found = false;
    tree::walk(view.body, &mut Vec::new(), 0, &mut |stmt, _, _| {
        let own = match stmt {
            Stmt::Assign { name, .. } => members.contains(name),
            Stmt::If { yes, no, .. } => only_assigns(yes, members) && only_assigns(no, members),
            _ => false,
        };
        if !own
            && tree::stmt_exprs(stmt)
                .iter()
                .any(|e| expr_mentions(e, members))
        {
            found = true;
        }
    });
    found || other_updates_read(view, members)
}

fn only_assigns(list: &[Stmt], members: &BTreeSet<String>) -> bool {
    list.iter().all(|stmt| match stmt {
        Stmt::Assign { name, .. } => members.contains(name),
        Stmt::If { yes, no, .. } => only_assigns(yes, members) && only_assigns(no, members),
        _ => false,
    })
}

fn other_updates_read(view: &View<'_>, members: &BTreeSet<String>) -> bool {
    view.carried()
        .iter()
        .filter(|v| !members.contains(*v))
        .any(|v| update_names(view, v).iter().any(|n| members.contains(n)))
}

fn arrays_complete(
    view: &View<'_>,
    node: &LoopNode,
    certs: &[Certificate],
) -> Result<(), Rejection> {
    let written: BTreeSet<&String> = view.stores.keys().collect();
    let named: BTreeSet<&String> = node.arrays.iter().map(|(n, _)| n).collect();
    if written != named || named.len() != node.arrays.len() {
        return Err(incomplete("written arrays"));
    }
    for (name, law) in &node.arrays {
        let matching = certs
            .iter()
            .filter(|k| k.subject == Subject::Array(name.clone()))
            .collect::<Vec<_>>();
        let [cert] = matching[..] else {
            return Err(incomplete("array certificate"));
        };
        let variant_ok = match law {
            ArrayLaw::Combine { wrapping } => combine_kind(view, name) == Some(*wrapping),
            ArrayLaw::Compaction { counter } => {
                counter_of(view, name).as_ref() == Some(counter)
                    && node
                        .carried
                        .iter()
                        .any(|c| c.var == *counter && matches!(c.rep, Rep::Counter(_)))
            }
            ArrayLaw::Monotone { own_writer } => *own_writer != view.reads_of(name).is_empty(),
            ArrayLaw::Own | ArrayLaw::Ordered => true,
        };
        if cert.law != array_law_id(law) || !variant_ok {
            return Err(incomplete("array law"));
        }
    }
    let counters_ok = node.carried.iter().all(|c| {
        !matches!(c.rep, Rep::Counter(_))
            || node.arrays.iter().any(
                |(_, law)| matches!(law, ArrayLaw::Compaction { counter } if *counter == c.var),
            )
    });
    let array_certs = certs
        .iter()
        .filter(|k| matches!(k.subject, Subject::Array(_)))
        .count();
    if !counters_ok || array_certs != node.arrays.len() {
        return Err(incomplete("array certificate"));
    }
    Ok(())
}
