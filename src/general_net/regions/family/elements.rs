//! #344: a for-each loop is its counted form with `let x = xs[i];` first
//! (NB.7.6). For family admission only, each such element is read as `xs[i]`
//! where the body reads `x`, so the families see the shape of the counted
//! form that reads `xs[i]` in place. It holds because nothing in the body
//! writes, moves or mutably loans `xs` or assigns `x`, so `xs[i]` keeps its
//! value for the whole iteration; the certificates were checked on the real net.

use super::*;

/// `inner` with its leading element bindings forwarded, or `None` when it
/// starts with none.
pub(super) fn forward(inner: &Loop) -> Option<Loop> {
    let (i, _) = counted(inner)?;
    let mut elements = Vec::new();
    for stmt in &inner.body {
        match stmt {
            Stmt::Let {
                name,
                value: value @ Expr::Index { index, .. },
            } if named(index, i) => elements.push((name.as_str(), value)),
            _ => break,
        }
    }
    let leading = elements.len();
    let rest = &inner.body[leading..];
    elements.retain(|(name, value)| {
        let Expr::Index { array, .. } = value else {
            return false;
        };
        rest.iter()
            .all(|stmt| !writes(stmt, array) && !writes(stmt, name))
    });
    if elements.is_empty() {
        return None;
    }
    // A binding the body changes stays as written.
    let kept = inner.body[..leading].iter().filter(
        |stmt| !matches!(stmt, Stmt::Let { name, .. } if elements.iter().any(|(x, _)| x == name)),
    );
    let body = kept
        .cloned()
        .chain(rest.iter().map(|stmt| statement(stmt, &elements)));
    Some(Loop {
        head: inner.head,
        kind: inner.kind.clone(),
        body: body.collect(),
    })
}

type Elements<'a> = [(&'a str, &'a Expr)];

/// Whether `stmt` assigns, stores into, moves or loans for writing `name`.
fn writes(stmt: &Stmt, name: &str) -> bool {
    let exprs = |exprs: &[&Expr]| exprs.iter().any(|e| changes(e, name));
    let block = |stmts: &[Stmt]| stmts.iter().any(|s| writes(s, name));
    match stmt {
        Stmt::Let { name: bound, value } => bound == name || changes(value, name),
        Stmt::Assign {
            name: target,
            value,
        } => target == name || changes(value, name),
        Stmt::Store {
            array,
            index,
            value,
            ..
        } => array == name || exprs(&[index, value]),
        Stmt::Eval(value) => changes(value, name),
        Stmt::If { cond, yes, no } => changes(cond, name) || block(yes) || block(no),
        Stmt::Loop(inner) => {
            let kind = match &inner.kind {
                LoopKind::For { var, lo, hi } => var == name || exprs(&[lo, hi]),
                LoopKind::While { cond } => changes(cond, name),
            };
            kind || block(&inner.body)
        }
        Stmt::Return { value, .. } => value.as_ref().is_some_and(|v| changes(v, name)),
    }
}

/// Whether evaluating `e` moves `name` or loans it for writing.
fn changes(e: &Expr, name: &str) -> bool {
    match e {
        Expr::Move(moved) => moved == name,
        Expr::Owner { name: owner, loan } => {
            owner == name && !matches!(loan, None | Some(Mode::Read))
        }
        Expr::ArrayLit { items, .. } | Expr::Call { args: items, .. } => {
            items.iter().any(|item| changes(item, name))
        }
        Expr::Index { index, .. } => changes(index, name),
        Expr::Unary { value, .. } => changes(value, name),
        Expr::Binary { left, right, .. } | Expr::Logic { left, right, .. } => {
            changes(left, name) || changes(right, name)
        }
        _ => false,
    }
}

fn statement(stmt: &Stmt, elements: &Elements<'_>) -> Stmt {
    let e = |value: &Expr| expr(value, elements);
    let block = |stmts: &[Stmt]| stmts.iter().map(|s| statement(s, elements)).collect();
    match stmt {
        Stmt::Let { name, value } => Stmt::Let {
            name: name.clone(),
            value: e(value),
        },
        Stmt::Assign { name, value } => Stmt::Assign {
            name: name.clone(),
            value: e(value),
        },
        Stmt::Store {
            array,
            index,
            value,
            span,
        } => Stmt::Store {
            array: array.clone(),
            index: e(index),
            value: e(value),
            span: *span,
        },
        Stmt::Eval(value) => Stmt::Eval(e(value)),
        Stmt::If { cond, yes, no } => Stmt::If {
            cond: e(cond),
            yes: block(yes),
            no: block(no),
        },
        Stmt::Loop(inner) => Stmt::Loop(Box::new(Loop {
            head: inner.head,
            kind: match &inner.kind {
                LoopKind::For { var, lo, hi } => LoopKind::For {
                    var: var.clone(),
                    lo: e(lo),
                    hi: e(hi),
                },
                LoopKind::While { cond } => LoopKind::While { cond: e(cond) },
            },
            body: block(&inner.body),
        })),
        Stmt::Return { value, span } => Stmt::Return {
            value: value.as_ref().map(e),
            span: *span,
        },
    }
}

fn expr(value: &Expr, elements: &Elements<'_>) -> Expr {
    let e = |value: &Expr| Box::new(expr(value, elements));
    match value {
        Expr::Local(name) => elements
            .iter()
            .find(|(element, _)| element == name)
            .map_or_else(|| value.clone(), |(_, read)| (*read).clone()),
        Expr::ArrayLit { elem, items } => Expr::ArrayLit {
            elem: elem.clone(),
            items: items.iter().map(|item| expr(item, elements)).collect(),
        },
        Expr::Index { array, index, span } => Expr::Index {
            array: array.clone(),
            index: e(index),
            span: *span,
        },
        Expr::Unary { op, value, span } => Expr::Unary {
            op: *op,
            value: e(value),
            span: *span,
        },
        Expr::Binary {
            op,
            left,
            right,
            span,
        } => Expr::Binary {
            op: *op,
            left: e(left),
            right: e(right),
            span: *span,
        },
        Expr::Logic { and, left, right } => Expr::Logic {
            and: *and,
            left: e(left),
            right: e(right),
        },
        Expr::Call { name, args, span } => Expr::Call {
            name: name.clone(),
            args: args.iter().map(|arg| expr(arg, elements)).collect(),
            span: *span,
        },
        other => other.clone(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn local(name: &str) -> Expr {
        Expr::Local(name.to_owned())
    }

    fn element(array: &str) -> Expr {
        Expr::Index {
            array: array.to_owned(),
            index: Box::new(local("i")),
            span: Span { start: 1, end: 2 },
        }
    }

    fn fold(update: Expr, extra: Option<Stmt>) -> Loop {
        let bind = Stmt::Let {
            name: "x".to_owned(),
            value: element("xs"),
        };
        let assign = Stmt::Assign {
            name: "s".to_owned(),
            value: update,
        };
        Loop {
            head: 1,
            kind: LoopKind::For {
                var: "i".to_owned(),
                lo: Expr::U32(0),
                hi: Expr::Len("xs".to_owned()),
            },
            body: std::iter::once(bind).chain(extra).chain([assign]).collect(),
        }
    }

    fn sum(left: Expr, right: Expr) -> Expr {
        Expr::Call {
            name: "wrapping_add".to_owned(),
            args: vec![left, right],
            span: Span { start: 3, end: 4 },
        }
    }

    #[test]
    fn an_element_read_becomes_the_counted_read_unless_the_body_changes_it() {
        let forwarded = forward(&fold(sum(local("s"), local("x")), None)).unwrap();
        let counted = Stmt::Assign {
            name: "s".to_owned(),
            value: sum(local("s"), element("xs")),
        };
        assert_eq!(forwarded.body, [counted]);
        let store = Stmt::Store {
            array: "xs".to_owned(),
            index: local("i"),
            value: Expr::U32(9),
            span: Span { start: 5, end: 6 },
        };
        assert_eq!(
            forward(&fold(sum(local("s"), local("x")), Some(store))),
            None
        );
        let moved = Stmt::Eval(Expr::Call {
            name: "consume".to_owned(),
            args: vec![Expr::Move("xs".to_owned())],
            span: Span { start: 7, end: 8 },
        });
        assert_eq!(
            forward(&fold(sum(local("s"), local("x")), Some(moved))),
            None
        );
        let reassigned = Stmt::Assign {
            name: "x".to_owned(),
            value: Expr::U32(1),
        };
        assert_eq!(forward(&fold(local("x"), Some(reassigned))), None);
    }
}
