//! Hoisting a call argument into a preceding binding, for the loan
//! diagnostics: a borrowed argument must be a bare name, and an inner call
//! cannot run while an outer call holds a loan.

use super::scope::{self, Operand, children, contains};
use super::{Candidate, Edit, Generated, Recheck};
use crate::front::{BinaryOp, Expr, ExprKind, Mode, Param, Program, Span, Stmt, StmtKind};

/// What the diagnostic points at, which decides which arguments are hoisted.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum Site {
    /// `OWN_BORROW_SOURCE`: the span is the argument that is not a name.
    Argument,
    /// `OWN_BORROW_CONFLICT`: the span is inside an inner call; hoist the
    /// call argument that holds it, outermost first.
    Inside,
}

pub(super) fn generate(source: &[u8], span: Span, site: Site) -> Generated {
    let Ok(program) = crate::front::parse(source) else {
        return Generated::none();
    };
    let Some(function) = scope::enclosing(&program, span) else {
        return Generated::none();
    };
    let Some(stmt) = scope::statement_at(&function.body, span) else {
        return Generated::none();
    };
    let Some(root) = root(stmt) else {
        return Generated::none();
    };
    let mut targets = Vec::new();
    arguments(root, span, site, root.span, None, &mut targets);
    let candidates = targets
        .into_iter()
        .filter(|(arg, _)| in_order(root, arg.span) == Some(true))
        .filter_map(|(arg, callee)| candidate(source, &program, stmt, arg, callee))
        .collect();
    Generated::of(candidates, Recheck::OtherFunctions)
}

/// The expression a hoistable statement evaluates.
fn root(stmt: &Stmt) -> Option<&Expr> {
    match &stmt.kind {
        StmtKind::Binding { value, .. }
        | StmtKind::Assign { value, .. }
        | StmtKind::Expr(value) => Some(value),
        StmtKind::Return(value) => value.as_ref(),
        _ => None,
    }
}

/// Calls and arguments that are not bare names and meet `site`, with the
/// callee and position of each when it is a direct call argument, outermost
/// first. The statement's own expression (`root`) is never a target.
fn arguments<'a>(
    expr: &'a Expr,
    span: Span,
    site: Site,
    root: Span,
    link: Option<(&'a str, usize)>,
    out: &mut Vec<Operand<'a>>,
) {
    if !contains(expr.span, span) {
        return;
    }
    // `OWN_BORROW_SOURCE` for `view` and `view_mut` points at the whole call:
    // the owner to hoist is the first argument.
    let view_owner = match &expr.kind {
        ExprKind::Call { name, args } if matches!(name.as_str(), "view" | "view_mut") => args
            .first()
            .filter(|owner| !matches!(owner.kind, ExprKind::Name(_))),
        _ => None,
    };
    if let (Site::Argument, true, Some(owner)) = (site, expr.span == span, view_owner) {
        out.push((owner, None));
    } else {
        let hit = match site {
            Site::Argument => expr.span == span,
            Site::Inside => expr.span != span && matches!(expr.kind, ExprKind::Call { .. }),
        };
        if expr.span != root && hit && !matches!(expr.kind, ExprKind::Name(_)) {
            out.push((expr, link));
        }
    }
    for (child, link) in children(expr) {
        arguments(child, span, site, root, link, out);
    }
}

/// A name or a scalar literal: evaluating it has no effect and cannot trap.
/// Text and bytes literals allocate and can hit a resource limit, so they are
/// not trivial.
fn trivial(expr: &Expr) -> bool {
    matches!(
        expr.kind,
        ExprKind::Name(_) | ExprKind::Integer(_) | ExprKind::Float(_) | ExprKind::Bool(_)
    )
}

/// `Some(true)` when the expression at `target` can be evaluated first
/// without reordering any effect or trap: everything evaluated before it on
/// the way down is trivial, and it is not under a short-circuit or a match.
/// `None` when `expr` does not hold `target`.
fn in_order(expr: &Expr, target: Span) -> Option<bool> {
    if expr.span == target {
        return Some(true);
    }
    if !contains(expr.span, target) {
        return None;
    }
    if matches!(expr.kind, ExprKind::Match { .. }) {
        return Some(false);
    }
    let mut before = true;
    for (position, (child, _)) in children(expr).into_iter().enumerate() {
        if let Some(inside) = in_order(child, target) {
            let short_circuit = position == 1
                && matches!(
                    expr.kind,
                    ExprKind::Binary {
                        op: BinaryOp::And | BinaryOp::Or,
                        ..
                    }
                );
            return Some(inside && before && !short_circuit);
        }
        before &= trivial(child);
    }
    Some(false)
}

/// The hoisting edits for one argument, or `None` when the source layout does
/// not allow them.
fn candidate(
    source: &[u8],
    program: &Program,
    stmt: &Stmt,
    arg: &Expr,
    callee: Option<(&str, usize)>,
) -> Option<Candidate> {
    let text = std::str::from_utf8(source.get(arg.span.start..arg.span.end)?).ok()?;
    let name = fresh_name(source)?;
    let param = callee.and_then(|(callee, at)| parameter(program, callee, at));
    let keyword = if param.is_some_and(|param| param.mode == Some(Mode::Mut)) {
        "var"
    } else {
        "let"
    };
    // An array literal has no type of its own: give it the parameter's.
    let annotation = match (&arg.kind, param) {
        (ExprKind::Array(_), Some(param)) => format!(": {}", parameter_type(source, param)?),
        _ => String::new(),
    };
    let line_start = source[..stmt.span.start]
        .iter()
        .rposition(|byte| *byte == b'\n')
        .map_or(0, |at| at + 1);
    let indent = &source[line_start..stmt.span.start];
    let separator = if indent.iter().all(|byte| matches!(byte, b' ' | b'\t')) {
        format!("\n{}", String::from_utf8_lossy(indent))
    } else {
        " ".to_owned()
    };
    Some(Candidate {
        description: format!("Hoist the argument into '{name}'"),
        edits: vec![
            Edit {
                span: Span {
                    start: stmt.span.start,
                    end: stmt.span.start,
                },
                replacement: format!("{keyword} {name}{annotation} = {text};{separator}"),
            },
            Edit {
                span: arg.span,
                replacement: name,
            },
        ],
    })
}

/// The type of `param` as written in the source, without its mode.
fn parameter_type<'a>(source: &'a [u8], param: &Param) -> Option<&'a str> {
    let text = std::str::from_utf8(source.get(param.span.start..param.span.end)?).ok()?;
    let rest = text.split_once(':')?.1.trim_start();
    let keyword = match param.mode? {
        Mode::Read => "read",
        Mode::Mut => "mut",
        Mode::Take => "take",
    };
    Some(rest.strip_prefix(keyword)?.trim()).filter(|ty| !ty.is_empty())
}

/// Parameter `at` of the function `callee`; none for a generic function,
/// whose parameter types are not written out. Foreign functions take scalars
/// and `read` or `take` Bytes and String only (src/check.rs), so they never
/// need a `var` or an annotation and are not looked up.
fn parameter<'a>(program: &'a Program, callee: &str, at: usize) -> Option<&'a Param> {
    let function = program.functions.iter().find(|f| f.name == callee)?;
    function
        .generic_params
        .is_empty()
        .then(|| function.params.get(at))
        .flatten()
}

/// `tmp_1`, `tmp_2`, ...: the first name that appears nowhere in the source.
fn fresh_name(source: &[u8]) -> Option<String> {
    let text = std::str::from_utf8(source).ok()?;
    (1..=64)
        .map(|n| format!("tmp_{n}"))
        .find(|name| !text.contains(name.as_str()))
}
