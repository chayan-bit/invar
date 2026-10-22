//! Function-level laws (#133 sections 6.3 and 6.4): L9 linear mod-P path sums,
//! L10 range sums and L20 independent sibling calls.

use std::collections::BTreeSet;

use super::{
    Certificate, Fact, ForkSite, Guard, LawId, Recursion, Subject, VERSION,
    linear::{const_val, linform, mentions_any, outer_modulus},
    loops::Context,
};
use crate::{
    front::Type,
    general_net::regions::{
        RegionId,
        tree::{self, Expr, Stmt},
    },
};

/// Static bound on `sum_j c_j (P - 1)`: the combine stays below `i32::MAX` with
/// `2^20` of headroom for the runtime constant term (#133 L9).
pub(super) const COMBINE_BOUND: i64 = (1 << 31) - (1 << 20);

/// Self-call shape of a recursive function: the names bound by `let x = f(..)`
/// and whether every self call is such a let or a tail `return f(..)`.
pub(super) struct SelfCalls {
    pub lets: BTreeSet<String>,
    pub any: bool,
    pub shaped: bool,
}

pub(super) fn self_calls(name: &str, body: &[Stmt]) -> SelfCalls {
    let mut out = SelfCalls {
        lets: BTreeSet::new(),
        any: false,
        shaped: true,
    };
    tree::walk(body, &mut Vec::new(), 0, &mut |stmt, _, depth| {
        for expr in tree::stmt_exprs(stmt) {
            let mut calls = Vec::new();
            expr.calls(&mut calls);
            for call in calls {
                let Expr::Call {
                    name: callee, args, ..
                } = call
                else {
                    continue;
                };
                if callee != name {
                    continue;
                }
                out.any = true;
                let direct = std::ptr::eq(expr, call);
                match stmt {
                    Stmt::Let { name: bound, .. } if direct && depth == 0 => {
                        out.lets.insert(bound.clone());
                    }
                    Stmt::Return { .. } if direct && depth == 0 => {}
                    _ => out.shaped = false,
                }
                if args.iter().any(|arg| mentions_any(arg, &out.lets)) {
                    out.shaped = false;
                }
            }
        }
    });
    out
}

/// Self-let names are read only by return expressions, and each list that
/// binds one ends in a `return` (no self-let escapes into control flow).
fn lets_confined(body: &[Stmt], lets: &BTreeSet<String>) -> bool {
    let mut ok = true;
    tree::walk(body, &mut Vec::new(), 0, &mut |stmt, guards, _| {
        if guards.iter().any(|(g, _)| mentions_any(g, lets)) {
            ok = false;
        }
        if matches!(stmt, Stmt::Return { .. }) {
            return;
        }
        if tree::stmt_exprs(stmt).iter().any(|e| mentions_any(e, lets)) {
            ok = false;
        }
    });
    let mut lists_end_in_return = true;
    visit_lists(body, &mut |list| {
        let binds = list
            .iter()
            .any(|stmt| matches!(stmt, Stmt::Let { name, .. } if lets.contains(name)));
        if binds && !matches!(list.last(), Some(Stmt::Return { .. })) {
            lists_end_in_return = false;
        }
    });
    ok && lists_end_in_return
}

fn visit_lists(body: &[Stmt], visit: &mut dyn FnMut(&[Stmt])) {
    visit(body);
    for stmt in body {
        match stmt {
            Stmt::If { yes, no, .. } => {
                visit_lists(yes, visit);
                visit_lists(no, visit);
            }
            Stmt::Loop(inner) => visit_lists(&inner.body, visit),
            _ => {}
        }
    }
}

/// One combine return `(sum_j c_j r_j + k) [% P]`.
pub(super) struct CombineReturn {
    pub modulus: Option<u32>,
    pub coefficients: Vec<(String, i64)>,
    pub constant: Expr,
}

pub(super) fn combine_returns(
    body: &[Stmt],
    lets: &BTreeSet<String>,
) -> Option<Vec<CombineReturn>> {
    let mut out = Vec::new();
    let mut ok = true;
    tree::walk(body, &mut Vec::new(), 0, &mut |stmt, _, _| {
        let Stmt::Return {
            value: Some(value), ..
        } = stmt
        else {
            return;
        };
        if !mentions_any(value, lets) {
            return;
        }
        let (inner, modulus) = match outer_modulus(value) {
            Some((inner, modulus)) => (inner, Some(modulus)),
            None => (value, None),
        };
        let Some(form) = linform(inner, lets) else {
            ok = false;
            return;
        };
        let mut coefficients = Vec::new();
        for (key, coefficient) in &form {
            let Some(name) = key else { continue };
            match const_val(coefficient) {
                Some(c) if c >= 0 => coefficients.push((name.clone(), c)),
                _ => ok = false,
            }
        }
        let constant = form.get(&None).cloned().unwrap_or(Expr::I32(0));
        if mentions_any(&constant, lets) {
            ok = false;
        }
        out.push(CombineReturn {
            modulus,
            coefficients,
            constant,
        });
    });
    ok.then_some(out)
}

/// L9 or L10 for a self-recursive function, if its returns qualify.
fn recursion_law(cx: &Context<'_>) -> Option<(Recursion, Certificate)> {
    let tree = cx.tree;
    if tree.result != Type::I32
        || tree
            .params
            .iter()
            .any(|p| p.ty != Type::I32 && p.ty != Type::U32)
    {
        return None;
    }
    let calls = self_calls(&tree.name, &tree.body);
    if !calls.any || !calls.shaped || !lets_confined(&tree.body, &calls.lets) {
        return None;
    }
    let returns = combine_returns(&tree.body, &calls.lets)?;
    let moduli: BTreeSet<Option<u32>> = returns.iter().map(|r| r.modulus).collect();
    let modulus = match moduli.into_iter().collect::<Vec<_>>()[..] {
        [modulus] => modulus,
        [] => None,
        _ => return None,
    };
    let facts = returns
        .iter()
        .map(|r| Fact::Combine {
            coefficients: r.coefficients.clone(),
            constant: r.constant.clone(),
        })
        .collect();
    let region = RegionId {
        function: tree.index,
        head: None,
    };
    let certificate = |law, guards| Certificate {
        version: VERSION,
        fingerprint: cx.fingerprint,
        region,
        law,
        subject: Subject::Recursion,
        rep: None,
        guards,
        facts,
        annotations: Vec::new(),
    };
    match modulus {
        Some(p) => {
            let bound = returns.iter().try_fold(0_i64, |worst, r| {
                let sum = r.coefficients.iter().try_fold(0_i64, |acc, (_, c)| {
                    acc.checked_add(c.checked_mul(i64::from(p) - 1)?)
                })?;
                Some(worst.max(sum))
            })?;
            if bound >= COMBINE_BOUND {
                return None;
            }
            Some((
                Recursion::PathSum { modulus: p },
                certificate(
                    LawId::L9PathSum,
                    vec![
                        Guard::TableRange { modulus: p },
                        Guard::NonNegativeConstants,
                        Guard::NoOverflowCombine,
                    ],
                ),
            ))
        }
        None => {
            let sealed = returns.iter().all(|r| {
                r.coefficients.iter().all(|(_, c)| *c == 1) && const_val(&r.constant) == Some(0)
            });
            sealed.then(|| {
                (
                    Recursion::RangeSum,
                    certificate(LawId::L10RangeSum, Vec::new()),
                )
            })
        }
    }
}

fn sibling<'s>(cx: &Context<'_>, stmt: &'s Stmt) -> Option<(String, &'s Expr)> {
    let Stmt::Let {
        name,
        value: call @ Expr::Call { name: callee, .. },
    } = stmt
    else {
        return None;
    };
    let function = cx.partition.functions.iter().find(|f| &f.name == callee)?;
    let scalar = function.tree.as_ref().ok()?.params.iter().all(|p| {
        matches!(p.ty, Type::I32 | Type::U32 | Type::F32 | Type::Bool) && p.mode.is_none()
    });
    (function.pure && scalar).then(|| (name.clone(), call))
}

/// Runs of at least two consecutive `let x = g(..)` calls to pure functions
/// with scalar parameters, none reading an earlier sibling.
pub(super) fn fork_runs<'a>(cx: &Context<'_>, list: &'a [Stmt]) -> Vec<&'a [Stmt]> {
    let mut runs = Vec::new();
    let mut i = 0;
    while i < list.len() {
        let mut bound = BTreeSet::new();
        let mut j = i;
        while let Some(Some((name, call))) = list.get(j).map(|stmt| sibling(cx, stmt)) {
            if mentions_any(call, &bound) {
                break;
            }
            bound.insert(name);
            j += 1;
        }
        if j - i >= 2 {
            runs.push(&list[i..j]);
            i = j;
        } else {
            i += 1;
        }
    }
    runs
}

pub(super) fn extract(
    cx: &Context<'_>,
) -> (Option<Recursion>, Vec<super::ForkSite>, Vec<Certificate>) {
    if let Some((recursion, certificate)) = recursion_law(cx) {
        return (Some(recursion), Vec::new(), vec![certificate]);
    }
    let mut forks = Vec::new();
    let mut certificates = Vec::new();
    visit_lists(&cx.tree.body, &mut |list| {
        for run in fork_runs(cx, list) {
            let Some(Stmt::Let {
                value: Expr::Call { span, .. },
                ..
            }) = run.first()
            else {
                continue;
            };
            forks.push(ForkSite {
                start: *span,
                siblings: run.len(),
            });
            certificates.push(Certificate {
                version: VERSION,
                fingerprint: cx.fingerprint,
                region: RegionId {
                    function: cx.tree.index,
                    head: None,
                },
                law: LawId::L20ForkJoin,
                subject: Subject::Fork(*span),
                rep: None,
                guards: Vec::new(),
                facts: vec![Fact::Siblings(run.len())],
                annotations: Vec::new(),
            });
        }
    });
    (None, forks, certificates)
}
