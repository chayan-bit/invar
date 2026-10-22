//! Function certificates: L9 linear mod-P path sums, L10 range sums and L20
//! independent sibling calls, re-derived from the function's statement tree.

use std::collections::{BTreeMap, BTreeSet};

use super::{Rejection, view::names};
use crate::{
    front::{BinaryOp, Span, Type},
    general_net::{
        normal_form::{
            Certificate, Fact, Guard, LawId, Subject,
            poly::{Arith, Poly, local_key, normalize},
            recursion::COMBINE_BOUND,
        },
        regions::{
            Partition,
            tree::{self, Expr, FunctionTree, Stmt},
        },
    },
};

pub(super) fn check(
    partition: &Partition,
    tree: &FunctionTree,
    cert: &Certificate,
) -> Result<(), Rejection> {
    if cert.rep.is_some() {
        return Err(Rejection::Rep("unexpected representation"));
    }
    match (&cert.law, &cert.subject) {
        (LawId::L9PathSum | LawId::L10RangeSum, Subject::Recursion) => {
            recursion(partition, tree, cert)
        }
        (LawId::L20ForkJoin, Subject::Fork(span)) => fork(partition, tree, *span, cert),
        _ => Err(Rejection::Subject),
    }
}

/// One return that combines self-call results: `(sum_j c_j r_j + k) [% P]`.
struct Combine {
    modulus: Option<u32>,
    coefficients: BTreeMap<String, i128>,
    constant: Poly,
}

fn recursion(
    partition: &Partition,
    tree: &FunctionTree,
    cert: &Certificate,
) -> Result<(), Rejection> {
    let pure = partition.function(tree.index).is_some_and(|f| f.pure);
    let scalar = tree
        .params
        .iter()
        .all(|p| matches!(p.ty, Type::I32 | Type::U32) && p.mode.is_none());
    if !pure || tree.result != Type::I32 || !scalar {
        return Err(Rejection::Law("recursion signature"));
    }
    let lets = self_lets(tree).ok_or(Rejection::Law("self calls not in let or tail position"))?;
    if !confined(&tree.body, &lets) {
        return Err(Rejection::Law("self-call results escape the returns"));
    }
    let combines =
        combines(&tree.body, &lets).ok_or(Rejection::Law("return is not a linear combine"))?;
    let moduli: BTreeSet<Option<u32>> = combines.iter().map(|c| c.modulus).collect();
    let modulus = match moduli.into_iter().collect::<Vec<_>>()[..] {
        [modulus] => modulus,
        _ => return Err(Rejection::Law("mixed moduli")),
    };
    match (cert.law, modulus) {
        (LawId::L9PathSum, Some(p)) => {
            let bound = i128::from(COMBINE_BOUND);
            let fits = combines.iter().all(|c| {
                c.coefficients.values().all(|k| *k >= 0)
                    && c.coefficients.values().sum::<i128>() * (i128::from(p) - 1) < bound
            });
            if !fits {
                return Err(Rejection::Law("combine may overflow"));
            }
            let expected = [
                Guard::TableRange { modulus: p },
                Guard::NonNegativeConstants,
                Guard::NoOverflowCombine,
            ];
            let unique = cert
                .guards
                .iter()
                .enumerate()
                .all(|(i, g)| !cert.guards[..i].contains(g));
            if !(unique
                && cert.guards.len() == expected.len()
                && expected.iter().all(|g| cert.guards.contains(g)))
            {
                return Err(Rejection::Guards);
            }
        }
        (LawId::L10RangeSum, None) => {
            let sealed = combines
                .iter()
                .all(|c| c.coefficients.values().all(|k| *k == 1) && c.constant.is_zero());
            if !sealed {
                return Err(Rejection::Law("range sum is not a plain sum of children"));
            }
            if !cert.guards.is_empty() {
                return Err(Rejection::Guards);
            }
        }
        _ => return Err(Rejection::Law("law does not match the modulus")),
    }
    facts_match(&combines, &cert.facts)
}

/// Names bound by `let x = f(..)` self calls, if every self call is such a
/// top-level let or a tail `return f(..)` and no argument reads a bound name.
fn self_lets(tree: &FunctionTree) -> Option<BTreeSet<String>> {
    let mut lets = BTreeSet::new();
    let mut any = false;
    let mut shaped = true;
    tree::walk(&tree.body, &mut Vec::new(), 0, &mut |stmt, _, depth| {
        for expr in tree::stmt_exprs(stmt) {
            let mut calls = Vec::new();
            expr.calls(&mut calls);
            for call in calls {
                let Expr::Call { name, args, .. } = call else {
                    continue;
                };
                if *name != tree.name {
                    continue;
                }
                any = true;
                let direct = std::ptr::eq(expr, call) && depth == 0;
                match stmt {
                    Stmt::Let { name, .. } if direct => {
                        lets.insert(name.clone());
                    }
                    Stmt::Return { .. } if direct => {}
                    _ => shaped = false,
                }
                if args
                    .iter()
                    .any(|a| names(a).iter().any(|n| lets.contains(n)))
                {
                    shaped = false;
                }
            }
        }
    });
    (any && shaped).then_some(lets)
}

/// Bound names are read only by returns, never by conditions, and each list
/// that binds one ends in a return.
fn confined(body: &[Stmt], lets: &BTreeSet<String>) -> bool {
    let reads = |e: &Expr| names(e).iter().any(|n| lets.contains(n));
    let mut ok = true;
    tree::walk(body, &mut Vec::new(), 0, &mut |stmt, guards, _| {
        if guards.iter().any(|(g, _)| reads(g)) {
            ok = false;
        }
        if !matches!(stmt, Stmt::Return { .. }) && tree::stmt_exprs(stmt).iter().any(|e| reads(e)) {
            ok = false;
        }
    });
    ok && lists_end_in_return(body, lets)
}

fn lists_end_in_return(list: &[Stmt], lets: &BTreeSet<String>) -> bool {
    let binds = list
        .iter()
        .any(|s| matches!(s, Stmt::Let { name, .. } if lets.contains(name)));
    if binds && !matches!(list.last(), Some(Stmt::Return { .. })) {
        return false;
    }
    list.iter().all(|stmt| match stmt {
        Stmt::If { yes, no, .. } => lists_end_in_return(yes, lets) && lists_end_in_return(no, lets),
        Stmt::Loop(inner) => lists_end_in_return(&inner.body, lets),
        _ => true,
    })
}

/// The combine of every return that reads a bound name, in source order.
fn combines(body: &[Stmt], lets: &BTreeSet<String>) -> Option<Vec<Combine>> {
    let mut out = Vec::new();
    let mut ok = true;
    tree::walk(body, &mut Vec::new(), 0, &mut |stmt, _, _| {
        let Stmt::Return {
            value: Some(value), ..
        } = stmt
        else {
            return;
        };
        if !names(value).iter().any(|n| lets.contains(n)) {
            return;
        }
        match combine(value, lets) {
            Some(c) => out.push(c),
            None => ok = false,
        }
    });
    ok.then_some(out)
}

fn combine(value: &Expr, lets: &BTreeSet<String>) -> Option<Combine> {
    let (inner, modulus) = match value {
        Expr::Binary {
            op: BinaryOp::Rem,
            left,
            right,
            ..
        } => (
            left.as_ref(),
            Some(u32::try_from(right.literal()?).ok().filter(|p| *p > 1)?),
        ),
        _ => (value, None),
    };
    let mut rest = normalize(Arith::Exact, inner)?;
    let mut coefficients = BTreeMap::new();
    for name in lets {
        let (c, r) = rest.split_linear(&local_key(name))?;
        let c = c.as_constant()?;
        if c != 0 {
            coefficients.insert(name.clone(), c);
        }
        rest = r;
    }
    let keys: Vec<String> = lets.iter().map(|n| local_key(n)).collect();
    if rest.mentions(&|atom| keys.iter().any(|k| atom.contains(k.as_str()))) {
        return None;
    }
    Some(Combine {
        modulus,
        coefficients,
        constant: rest,
    })
}

fn facts_match(combines: &[Combine], facts: &[Fact]) -> Result<(), Rejection> {
    if combines.len() != facts.len() {
        return Err(Rejection::Facts);
    }
    for (combine, fact) in combines.iter().zip(facts) {
        let Fact::Combine {
            coefficients,
            constant,
        } = fact
        else {
            return Err(Rejection::Facts);
        };
        let stated: BTreeMap<String, i128> = coefficients
            .iter()
            .filter(|(_, c)| *c != 0)
            .map(|(n, c)| (n.clone(), i128::from(*c)))
            .collect();
        let distinct = coefficients
            .iter()
            .map(|(n, _)| n)
            .collect::<BTreeSet<_>>()
            .len()
            == coefficients.len();
        if !distinct
            || stated != combine.coefficients
            || normalize(Arith::Exact, constant).as_ref() != Some(&combine.constant)
        {
            return Err(Rejection::Facts);
        }
    }
    Ok(())
}

/// L20: `n >= 2` consecutive `let x = g(..)` calls to pure functions with
/// scalar by-value parameters, none reading a name bound by an earlier one.
fn fork(
    partition: &Partition,
    tree: &FunctionTree,
    span: Span,
    cert: &Certificate,
) -> Result<(), Rejection> {
    if !cert.guards.is_empty() {
        return Err(Rejection::Guards);
    }
    let [Fact::Siblings(n)] = cert.facts[..] else {
        return Err(Rejection::Facts);
    };
    if n < 2 {
        return Err(Rejection::Facts);
    }
    let run = find_run(&tree.body, span, n).ok_or(Rejection::Subject)?;
    let mut bound = BTreeSet::new();
    for stmt in run {
        let Stmt::Let { name, value } = stmt else {
            return Err(Rejection::Facts);
        };
        let Expr::Call { name: callee, .. } = value else {
            return Err(Rejection::Facts);
        };
        let function = partition
            .functions
            .iter()
            .find(|f| &f.name == callee)
            .ok_or(Rejection::Law("sibling is not a user function"))?;
        let scalar = function.tree.as_ref().is_ok_and(|t| {
            t.params.iter().all(|p| {
                matches!(p.ty, Type::I32 | Type::U32 | Type::F32 | Type::Bool) && p.mode.is_none()
            })
        });
        if !function.pure || !scalar {
            return Err(Rejection::Law("sibling is impure or takes arrays"));
        }
        if names(value).iter().any(|n| bound.contains(n)) {
            return Err(Rejection::Law("sibling reads an earlier sibling"));
        }
        bound.insert(name.clone());
    }
    Ok(())
}

/// The `n` statements starting at the let whose call has `span`.
fn find_run(list: &[Stmt], span: Span, n: usize) -> Option<&[Stmt]> {
    let start = list.iter().position(
        |stmt| matches!(stmt, Stmt::Let { value: Expr::Call { span: s, .. }, .. } if *s == span),
    );
    if let Some(start) = start {
        return list.get(start..start.checked_add(n)?);
    }
    list.iter().find_map(|stmt| match stmt {
        Stmt::If { yes, no, .. } => find_run(yes, span, n).or_else(|| find_run(no, span, n)),
        Stmt::Loop(inner) => find_run(&inner.body, span, n),
        _ => None,
    })
}
