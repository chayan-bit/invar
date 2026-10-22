//! Syntactic linear-form extraction (normal-form design section 6.2 step 2): an expression
//! as `sum_v c_v * v + k` over a set of carried variables, with coefficient and
//! constant expressions free of carried state. This is the extractor's view;
//! the certificate checker re-derives the same facts with `poly`.

use std::collections::{BTreeMap, BTreeSet};

use crate::{
    front::{BinaryOp, Span, UnaryOp},
    general_net::regions::tree::Expr,
};

pub(super) const SYNTHETIC: Span = Span { start: 0, end: 0 };

/// Key `None` is the constant term.
pub(super) type LinForm = BTreeMap<Option<String>, Expr>;

pub(super) fn mentions_any(expr: &Expr, names: &BTreeSet<String>) -> bool {
    let mut found = Vec::new();
    expr.names(&mut found);
    found.iter().any(|name| names.contains(name))
}

pub(super) fn names_of(expr: &Expr) -> BTreeSet<String> {
    let mut found = Vec::new();
    expr.names(&mut found);
    found.into_iter().collect()
}

fn binary(op: BinaryOp, left: Expr, right: Expr) -> Expr {
    Expr::Binary {
        op,
        left: Box::new(left),
        right: Box::new(right),
        span: SYNTHETIC,
    }
}

/// `c * f`, without a literal unit factor.
fn product(c: Expr, f: Expr) -> Expr {
    if is_one(&c) {
        f
    } else if is_one(&f) {
        c
    } else {
        binary(BinaryOp::Mul, c, f)
    }
}

pub(super) fn one() -> Expr {
    Expr::I32(1)
}

pub(super) fn zero() -> Expr {
    Expr::I32(0)
}

/// `expr` as a linear form in `carried`, or `None` when it is not linear.
pub(super) fn linform(expr: &Expr, carried: &BTreeSet<String>) -> Option<LinForm> {
    if !mentions_any(expr, carried) {
        return Some(BTreeMap::from([(None, expr.clone())]));
    }
    match expr {
        Expr::Local(name) => Some(BTreeMap::from([(Some(name.clone()), one())])),
        Expr::Binary {
            op: op @ (BinaryOp::Add | BinaryOp::Sub),
            left,
            right,
            ..
        } => {
            let (a, b) = (linform(left, carried)?, linform(right, carried)?);
            let mut out = a;
            for (key, coefficient) in b {
                let coefficient = if *op == BinaryOp::Add {
                    coefficient
                } else {
                    Expr::Unary {
                        op: UnaryOp::Neg,
                        value: Box::new(coefficient),
                        span: SYNTHETIC,
                    }
                };
                let merged = match out.remove(&key) {
                    Some(previous) => binary(BinaryOp::Add, previous, coefficient),
                    None => coefficient,
                };
                out.insert(key, merged);
            }
            Some(out)
        }
        Expr::Binary {
            op: BinaryOp::Mul,
            left,
            right,
            ..
        } => {
            for (variable, factor) in [(left, right), (right, left)] {
                if !mentions_any(factor, carried) {
                    let form = linform(variable, carried)?;
                    return Some(
                        form.into_iter()
                            .map(|(key, c)| (key, product(c, (**factor).clone())))
                            .collect(),
                    );
                }
            }
            None
        }
        Expr::Call { name, args, span } if args.len() == 2 => {
            let op = match name.as_str() {
                "wrapping_add" => BinaryOp::Add,
                "wrapping_mul" => BinaryOp::Mul,
                _ => return None,
            };
            linform(
                &Expr::Binary {
                    op,
                    left: Box::new(args[0].clone()),
                    right: Box::new(args[1].clone()),
                    span: *span,
                },
                carried,
            )
        }
        _ => None,
    }
}

/// Value of a literal-only integer expression.
pub(super) fn const_val(expr: &Expr) -> Option<i64> {
    match expr {
        Expr::I32(_) | Expr::U32(_) => expr.literal(),
        Expr::Unary {
            op: UnaryOp::Neg,
            value,
            ..
        } => const_val(value)?.checked_neg(),
        Expr::Binary {
            op, left, right, ..
        } => {
            let (a, b) = (const_val(left)?, const_val(right)?);
            match op {
                BinaryOp::Add => a.checked_add(b),
                BinaryOp::Sub => a.checked_sub(b),
                BinaryOp::Mul => a.checked_mul(b),
                _ => None,
            }
        }
        _ => None,
    }
}

pub(super) fn is_one(expr: &Expr) -> bool {
    const_val(expr) == Some(1)
}

/// Whether `expr` uses the sealed wrapping intrinsics anywhere.
pub(super) fn uses_wrapping(expr: &Expr) -> bool {
    let mut calls = Vec::new();
    expr.calls(&mut calls);
    calls.iter().any(|call| {
        matches!(call, Expr::Call { name, .. } if name == "wrapping_add" || name == "wrapping_mul")
    })
}

/// Wrapping composition cannot erase a checked operation on carried state.
/// State-free coefficients may contain traps; lane replay preserves them.
pub(super) fn wrapping_state(expr: &Expr, carried: &BTreeSet<String>) -> bool {
    if !mentions_any(expr, carried) {
        return true;
    }
    match expr {
        Expr::Local(_) => true,
        Expr::Call { name, args, .. }
            if matches!(name.as_str(), "wrapping_add" | "wrapping_mul") && args.len() == 2 =>
        {
            args.iter().all(|arg| wrapping_state(arg, carried))
        }
        _ => false,
    }
}

/// `expr % P` with a positive literal `P`: `(inner, P)`.
pub(super) fn outer_modulus(expr: &Expr) -> Option<(&Expr, u32)> {
    match expr {
        Expr::Binary {
            op: BinaryOp::Rem,
            left,
            right,
            ..
        } => {
            let modulus = u32::try_from(right.literal()?).ok().filter(|p| *p > 1)?;
            Some((left, modulus))
        }
        _ => None,
    }
}
