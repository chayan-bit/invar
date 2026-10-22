//! `OWN_USE_AFTER_MOVE`: relax the `take` parameter that consumed the value,
//! or say where it was moved. `OWN_RETURN_BORROW` has no fix, only the rule.

use super::scope::{self, walk_block};
use super::{Candidate, Edit, Generated, Recheck};
use crate::front::{Expr, ExprKind, Mode, Place, Program, Span, StmtKind, Type};

/// The nearest `move name` that is evaluated before `used`, and the call
/// argument it is, if any.
struct Move<'a> {
    at: Span,
    callee: Option<(&'a str, usize)>,
}

fn earlier_move<'a>(program: &'a Program, source: &[u8], used: Span) -> Option<Move<'a>> {
    let name = scope::bare_name(source, used)?;
    let function = scope::enclosing(program, used)?;
    // A name is declared once per function (`NAME_DUPLICATE`, src/check.rs
    // `declare`), so the spelling identifies the binding.
    let limit = evaluated_at(function, used);
    let mut found: Option<Move<'a>> = None;
    walk_block(&function.body, &mut |expr: &'a Expr| {
        if let ExprKind::Call { name: callee, args } = &expr.kind {
            for (at, arg) in args.iter().enumerate() {
                if matches!(&arg.kind, ExprKind::Move(moved) if moved == name)
                    && arg.span.end <= limit
                    && found
                        .as_ref()
                        .is_none_or(|best| best.at.start < arg.span.start)
                {
                    found = Some(Move {
                        at: arg.span,
                        callee: Some((callee.as_str(), at)),
                    });
                }
            }
        }
        if matches!(&expr.kind, ExprKind::Move(moved) if moved == name)
            && expr.span.end <= limit
            && found
                .as_ref()
                .is_none_or(|best| best.at.start < expr.span.start)
        {
            found = Some(Move {
                at: expr.span,
                callee: None,
            });
        }
    });
    found
}

/// Where `used` is evaluated: the end of the statement for the destination of
/// an assignment, whose right-hand side runs before the store, else its start.
fn evaluated_at(function: &crate::front::Function, used: Span) -> usize {
    scope::statement_at(&function.body, used)
        .filter(|stmt| match &stmt.kind {
            StmtKind::Assign { place, .. } => match place {
                Place::Name(_, span) | Place::Index { span, .. } => *span == used,
            },
            _ => false,
        })
        .map_or(used.start, |stmt| stmt.span.end)
}

/// Make the consuming parameter `read` and drop the `move` at every call of
/// that function. The edits span several functions and a signature, so the
/// recheck has to come back clean for every caller.
pub(super) fn relax_take(source: &[u8], used: Span) -> Generated {
    let Ok(program) = crate::front::parse(source) else {
        return Generated::none();
    };
    let Some(Move {
        callee: Some((callee, position)),
        ..
    }) = earlier_move(&program, source, used)
    else {
        return Generated::none();
    };
    let Some(function) = program
        .functions
        .iter()
        .find(|function| function.name == callee && function.generic_params.is_empty())
    else {
        return Generated::none();
    };
    let Some(keyword) = function
        .params
        .get(position)
        .and_then(|param| scope::mode_keyword(source, param, Mode::Take))
    else {
        return Generated::none();
    };
    let mut edits = vec![Edit {
        span: keyword,
        replacement: "read".to_owned(),
    }];
    for caller in &program.functions {
        walk_block(&caller.body, &mut |expr: &Expr| {
            if let ExprKind::Call { name, args } = &expr.kind
                && name == callee
                && let Some(arg) = args.get(position)
                && let ExprKind::Move(moved) = &arg.kind
            {
                edits.push(Edit {
                    span: Span {
                        start: arg.span.start,
                        end: arg.span.end - moved.len(),
                    },
                    replacement: String::new(),
                });
            }
        });
    }
    edits.sort_by_key(|edit| edit.span.start);
    edits.dedup_by_key(|edit| edit.span);
    let description = format!("Make '{callee}' borrow its argument");
    Generated::of(vec![Candidate { description, edits }], Recheck::Clean)
}

/// Where the used value was moved.
pub(super) fn moved_here(source: &[u8], used: Span) -> Vec<String> {
    let Ok(program) = crate::front::parse(source) else {
        return Vec::new();
    };
    let Some(earlier) = earlier_move(&program, source, used) else {
        return Vec::new();
    };
    let Some(location) = crate::diagnostics::location("", source, earlier.at) else {
        return Vec::new();
    };
    vec![format!(
        "moved here, line {} column {} (bytes {}..{}); a moved array is gone, there is no implicit clone",
        location["start_line"], location["start_column"], earlier.at.start, earlier.at.end
    )]
}

/// The rule and the two valid forms, for a borrowed array parameter returned
/// as an owned array. Other `OWN_RETURN_BORROW` causes (a view from the wrong
/// parameter, a borrowed record) get no note rather than a misleading one.
pub(super) fn return_borrow(source: &[u8], span: Span) -> Vec<String> {
    let Ok(program) = crate::front::parse(source) else {
        return Vec::new();
    };
    let borrowed_array = scope::bare_name(source, span)
        .zip(scope::enclosing(&program, span))
        .and_then(|(name, function)| function.params.iter().find(|param| param.name == name))
        .is_some_and(|param| {
            matches!(param.mode, Some(Mode::Read | Mode::Mut))
                && matches!(
                    param.ty,
                    Type::ArrayI32
                        | Type::ArrayU32
                        | Type::ArrayF32
                        | Type::ArrayF64
                        | Type::ArrayBool
                )
        });
    if !borrowed_array {
        return Vec::new();
    }
    vec![
        "a borrowed parameter cannot be returned as an owned array: return a copy of its elements, or declare the result `-> View<T> from <parameter>` to return a view of it"
            .to_owned(),
    ]
}
