//! Source lookups the generators share: names, the enclosing function and the
//! bindings visible at a span.

use crate::front::{
    Block, ElseArm, Expr, ExprKind, Function, Mode, Param, Place, Program, Span, Stmt, StmtKind,
};

/// The text of `span` when it is exactly one plain name.
pub(super) fn bare_name(source: &[u8], span: Span) -> Option<&str> {
    let text = std::str::from_utf8(source.get(span.start..span.end)?).ok()?;
    let mut chars = text.chars();
    let first = chars.next()?;
    ((first.is_ascii_alphabetic() || first == '_')
        && chars.all(|c| c.is_ascii_alphanumeric() || c == '_'))
    .then_some(text)
}

/// Bytes of leading whitespace and comments in `text`: `//` through newline and
/// nested `/* ... */`, as the lexer reads them. An unterminated comment runs to
/// the end of `text`.
pub(super) fn trivia_len(text: &str) -> usize {
    let bytes = text.as_bytes();
    let mut i = 0;
    loop {
        while bytes.get(i).is_some_and(u8::is_ascii_whitespace) {
            i += 1;
        }
        if bytes[i..].starts_with(b"//") {
            while bytes.get(i).is_some_and(|byte| *byte != b'\n') {
                i += 1;
            }
        } else if bytes[i..].starts_with(b"/*") {
            let mut depth = 0;
            while i < bytes.len() {
                if bytes[i..].starts_with(b"/*") {
                    depth += 1;
                    i += 2;
                } else if bytes[i..].starts_with(b"*/") {
                    depth -= 1;
                    i += 2;
                    if depth == 0 {
                        break;
                    }
                } else {
                    i += 1;
                }
            }
        } else {
            return i;
        }
    }
}

/// The function whose source range holds `span`.
pub(super) fn enclosing(program: &Program, span: Span) -> Option<&Function> {
    program
        .functions
        .iter()
        .find(|function| function.span.start <= span.start && span.end <= function.span.end)
}

/// Statements in scope at `at` that precede it, outermost first: the earlier
/// siblings of every block that contains `at`, and each `for` whose body does.
pub(super) fn preceding<'a>(block: &'a Block, at: Span, out: &mut Vec<&'a Stmt>) {
    for stmt in &block.statements {
        if stmt.span.end <= at.start {
            out.push(stmt);
        } else if stmt.span.start <= at.start && at.end <= stmt.span.end {
            descend(stmt, at, out);
            return;
        }
    }
}

fn descend<'a>(stmt: &'a Stmt, at: Span, out: &mut Vec<&'a Stmt>) {
    match &stmt.kind {
        StmtKind::If {
            then_block,
            else_arm,
            ..
        } => {
            if contains(then_block.span, at) {
                preceding(then_block, at, out);
            } else {
                match else_arm {
                    Some(ElseArm::Block(block)) if contains(block.span, at) => {
                        preceding(block, at, out);
                    }
                    Some(ElseArm::If(inner)) if contains(inner.span, at) => descend(inner, at, out),
                    _ => {}
                }
            }
        }
        StmtKind::While { body, .. } if contains(body.span, at) => preceding(body, at, out),
        StmtKind::For { body, .. } if contains(body.span, at) => {
            // The loop variable is in scope in the body.
            out.push(stmt);
            preceding(body, at, out);
        }
        _ => {}
    }
}

pub(super) fn contains(outer: Span, inner: Span) -> bool {
    outer.start <= inner.start && inner.end <= outer.end
}

/// The innermost statement that holds `at` in one of its own expressions.
/// `None` when `at` sits in a loop or branch condition, which cannot be
/// hoisted out without changing how often it runs.
pub(super) fn statement_at(block: &Block, at: Span) -> Option<&Stmt> {
    let stmt = block
        .statements
        .iter()
        .find(|stmt| contains(stmt.span, at))?;
    innermost(stmt, at)
}

fn innermost(stmt: &Stmt, at: Span) -> Option<&Stmt> {
    match &stmt.kind {
        StmtKind::If {
            then_block,
            else_arm,
            ..
        } => {
            if contains(then_block.span, at) {
                return statement_at(then_block, at);
            }
            match else_arm {
                Some(ElseArm::Block(block)) if contains(block.span, at) => statement_at(block, at),
                Some(ElseArm::If(inner)) if contains(inner.span, at) => innermost(inner, at),
                _ => None,
            }
        }
        StmtKind::While { body, .. } | StmtKind::For { body, .. } => {
            statement_at(body, at).filter(|_| contains(body.span, at))
        }
        _ => Some(stmt),
    }
}

/// An operand and, for a call argument, the callee name and its position.
pub(super) type Operand<'a> = (&'a Expr, Option<(&'a str, usize)>);

fn plain(items: Vec<&Expr>) -> Vec<Operand<'_>> {
    items.into_iter().map(|item| (item, None)).collect()
}

/// Direct operands of `expr` in evaluation order; a call argument carries the
/// callee name and its position.
pub(super) fn children(expr: &Expr) -> Vec<Operand<'_>> {
    match &expr.kind {
        ExprKind::Call { name, args } => args
            .iter()
            .enumerate()
            .map(|(at, arg)| (arg, Some((name.as_str(), at))))
            .collect(),
        ExprKind::Array(items) => plain(items.iter().collect()),
        ExprKind::Record { fields, .. } => plain(fields.iter().map(|field| &field.value).collect()),
        ExprKind::Group(value)
        | ExprKind::Try(value)
        | ExprKind::Unary { value, .. }
        | ExprKind::Field { base: value, .. }
        | ExprKind::MoveField { base: value, .. } => plain(vec![value]),
        ExprKind::Index { base, index } => plain(vec![base, index]),
        ExprKind::Binary { left, right, .. } => plain(vec![left, right]),
        ExprKind::Match { value, arms } => plain(
            std::iter::once(&**value)
                .chain(arms.iter().map(|arm| &arm.value))
                .collect(),
        ),
        _ => Vec::new(),
    }
}

/// Visit every expression of a block in source order, nested ones included.
pub(super) fn walk_block<'a>(block: &'a Block, visit: &mut dyn FnMut(&'a Expr)) {
    for stmt in &block.statements {
        walk_stmt(stmt, visit);
    }
}

fn walk_stmt<'a>(stmt: &'a Stmt, visit: &mut dyn FnMut(&'a Expr)) {
    match &stmt.kind {
        StmtKind::Binding { value, .. } | StmtKind::Expr(value) => walk_expr(value, visit),
        StmtKind::Assign { place, value } => {
            if let Place::Index { index, .. } = place {
                walk_expr(index, visit);
            }
            walk_expr(value, visit);
        }
        StmtKind::Return(value) => {
            if let Some(value) = value {
                walk_expr(value, visit);
            }
        }
        StmtKind::If {
            condition,
            then_block,
            else_arm,
        } => {
            walk_expr(condition, visit);
            walk_block(then_block, visit);
            match else_arm {
                Some(ElseArm::Block(block)) => walk_block(block, visit),
                Some(ElseArm::If(inner)) => walk_stmt(inner, visit),
                None => {}
            }
        }
        StmtKind::While { condition, body } => {
            walk_expr(condition, visit);
            walk_block(body, visit);
        }
        StmtKind::For {
            lower, upper, body, ..
        } => {
            walk_expr(lower, visit);
            walk_expr(upper, visit);
            walk_block(body, visit);
        }
    }
}

fn walk_expr<'a>(expr: &'a Expr, visit: &mut dyn FnMut(&'a Expr)) {
    visit(expr);
    for (child, _) in children(expr) {
        walk_expr(child, visit);
    }
}

/// The span of the mode keyword of `param` (`read`, `mut` or `take`), found
/// after the colon, when it is written as `want`.
pub(super) fn mode_keyword(source: &[u8], param: &Param, want: Mode) -> Option<Span> {
    let keyword = match want {
        Mode::Read => "read",
        Mode::Mut => "mut",
        Mode::Take => "take",
    };
    if param.mode != Some(want) {
        return None;
    }
    let text = std::str::from_utf8(source.get(param.span.start..param.span.end)?).ok()?;
    let colon = text.find(':')?;
    let rest = &text[colon + 1..];
    let start = param.span.start + colon + 1 + trivia_len(rest);
    let span = Span {
        start,
        end: start + keyword.len(),
    };
    (source.get(span.start..span.end) == Some(keyword.as_bytes())).then_some(span)
}
