//! Small queries on expressions.

use super::*;

/// Strips any number of enclosing `Group` nodes.
pub fn ungroup(mut e: &Expr) -> &Expr {
    while let ExprKind::Group(inner) = &e.kind {
        e = inner;
    }
    e
}

/// True when `e`, ignoring grouping, is the plain name `name`.
pub fn named(e: &Expr, name: &str) -> bool {
    matches!(&ungroup(e).kind, ExprKind::Name(n) if n == name)
}
