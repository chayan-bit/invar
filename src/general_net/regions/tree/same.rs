//! Span-blind structural equality of tree expressions.

use super::Expr;

/// Structural equality that ignores source spans: two occurrences of the same
/// source expression at different positions compare equal.
pub fn same(a: &Expr, b: &Expr) -> bool {
    match (a, b) {
        (
            Expr::Index { array, index, .. },
            Expr::Index {
                array: x, index: y, ..
            },
        ) => array == x && same(index, y),
        (
            Expr::Unary { op, value, .. },
            Expr::Unary {
                op: o, value: v, ..
            },
        ) => op == o && same(value, v),
        (
            Expr::Binary {
                op, left, right, ..
            },
            Expr::Binary {
                op: o,
                left: l,
                right: r,
                ..
            },
        ) => op == o && same(left, l) && same(right, r),
        (
            Expr::Logic { and, left, right },
            Expr::Logic {
                and: a2,
                left: l,
                right: r,
            },
        ) => and == a2 && same(left, l) && same(right, r),
        (
            Expr::Call { name, args, .. },
            Expr::Call {
                name: n, args: a2, ..
            },
        ) => name == n && args.len() == a2.len() && args.iter().zip(a2).all(|(x, y)| same(x, y)),
        (Expr::ArrayLit { elem, items }, Expr::ArrayLit { elem: e, items: i }) => {
            elem == e && items.len() == i.len() && items.iter().zip(i).all(|(x, y)| same(x, y))
        }
        _ => a == b,
    }
}
