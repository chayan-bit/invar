//! A walk over the span-carrying nodes of a function body, shared by the
//! numeric checks and the storage lowering (`docs/numeric-v2.md`).

use super::*;

/// Span-carrying nodes of a statement tree, visited outermost first.
pub(in crate::check) trait Visit {
    /// A statement, before its parts.
    fn stmt(&mut self, _: &mut Stmt) {}
    fn expr(&mut self, _: &mut Expr) {}
    fn span(&mut self, _: &mut Span) {}
    fn place(&mut self, _: &str) {}
    /// The name a `let` or `var` declares.
    fn bind(&mut self, _: &str) {}
}

pub(in crate::check) fn walk_block(block: &mut Block, v: &mut impl Visit) {
    v.span(&mut block.span);
    block.statements.iter_mut().for_each(|s| walk_stmt(s, v));
}

pub(super) fn walk_stmt(statement: &mut Stmt, v: &mut impl Visit) {
    v.stmt(statement);
    v.span(&mut statement.span);
    match &mut statement.kind {
        StmtKind::Binding { name, value, .. } => {
            v.bind(name);
            walk_expr(value, v)
        }
        StmtKind::Expr(value) | StmtKind::Return(Some(value)) => walk_expr(value, v),
        StmtKind::Assign { place, value } => {
            let (Place::Name(name, span) | Place::Index { name, span, .. }) = place;
            v.place(name);
            v.span(span);
            if let Place::Index { index, .. } = place {
                walk_expr(index, v);
            }
            walk_expr(value, v);
        }
        StmtKind::Return(None) => {}
        StmtKind::If {
            condition,
            then_block,
            else_arm,
        } => {
            walk_expr(condition, v);
            walk_block(then_block, v);
            match else_arm {
                Some(ElseArm::Block(arm)) => walk_block(arm, v),
                Some(ElseArm::If(arm)) => walk_stmt(arm, v),
                None => {}
            }
        }
        StmtKind::While { condition, body } => {
            walk_expr(condition, v);
            walk_block(body, v);
        }
        StmtKind::For {
            name,
            lower,
            upper,
            body,
        } => {
            v.place(name);
            walk_expr(lower, v);
            walk_expr(upper, v);
            walk_block(body, v);
        }
    }
}

pub(super) fn walk_expr(e: &mut Expr, v: &mut impl Visit) {
    v.expr(e);
    match &mut e.kind {
        ExprKind::Call { args, .. } | ExprKind::Array(args) => {
            args.iter_mut().for_each(|a| walk_expr(a, v))
        }
        ExprKind::Record { fields, .. } => {
            fields.iter_mut().for_each(|f| walk_expr(&mut f.value, v))
        }
        ExprKind::Match { value, arms } => {
            walk_expr(value, v);
            arms.iter_mut().for_each(|arm| walk_expr(&mut arm.value, v));
        }
        ExprKind::MoveField { base, .. }
        | ExprKind::Field { base, .. }
        | ExprKind::Group(base)
        | ExprKind::Try(base) => walk_expr(base, v),
        ExprKind::Unary { op_span, value, .. } => {
            v.span(op_span);
            walk_expr(value, v);
        }
        ExprKind::Binary {
            left,
            op_span,
            right,
            ..
        } => {
            v.span(op_span);
            walk_expr(left, v);
            walk_expr(right, v);
        }
        ExprKind::Index { base, index } => {
            walk_expr(base, v);
            walk_expr(index, v);
        }
        _ => {}
    }
}
