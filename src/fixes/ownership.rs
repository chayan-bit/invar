//! Generators for the ownership-class diagnostics with a mechanical remedy.

use super::{Candidate, Edit, Generated, Recheck, scope};
use crate::front::{Mode, Span, StmtKind};

fn edit(at: Span, replacement: &str) -> Edit {
    Edit {
        span: at,
        replacement: replacement.to_owned(),
    }
}

fn candidate(description: String, edit: Edit) -> Candidate {
    Candidate {
        description,
        edits: vec![edit],
    }
}

/// `OWN_MOVE_REQUIRED` on a bare owner name: insert `move `. A `move` that is
/// already written, a borrowed owner and a move out of a loop are left to the
/// recheck, which rejects them.
pub(super) fn move_required(source: &[u8], span: Span) -> Generated {
    let Some(name) = scope::bare_name(source, span) else {
        return Generated::none();
    };
    let at = Span {
        start: span.start,
        end: span.start,
    };
    Generated::of(
        vec![candidate(format!("Move '{name}'"), edit(at, "move "))],
        Recheck::OtherFunctions,
    )
}

/// `OWN_MUT_REQUIRED` on a bare owner name: make the nearest preceding `let`
/// of that name a `var`, or else make a `read` parameter of that name `mut`.
pub(super) fn mut_required(source: &[u8], span: Span) -> Generated {
    let Some(name) = scope::bare_name(source, span) else {
        return Generated::none();
    };
    let Ok(program) = crate::front::parse(source) else {
        return Generated::none();
    };
    let Some(function) = scope::enclosing(&program, span) else {
        return Generated::none();
    };
    let mut visible = Vec::new();
    scope::preceding(&function.body, span, &mut visible);
    let local = visible.into_iter().rev().find_map(|stmt| match &stmt.kind {
        StmtKind::Binding {
            name: bound,
            mutable,
            ..
        } if bound == name => Some((stmt.span, *mutable)),
        _ => None,
    });
    if let Some((stmt, mutable)) = local {
        let keyword = Span {
            start: stmt.start,
            end: stmt.start + 3,
        };
        if mutable || source.get(keyword.start..keyword.end) != Some(b"let".as_slice()) {
            return Generated::none();
        }
        return Generated::of(
            vec![candidate(
                format!("Declare '{name}' with var"),
                edit(keyword, "var"),
            )],
            Recheck::OtherFunctions,
        );
    }
    let Some(param) = function
        .params
        .iter()
        .find(|param| param.name == name && param.mode == Some(Mode::Read))
    else {
        return Generated::none();
    };
    let Some(keyword) = scope::mode_keyword(source, param, Mode::Read) else {
        return Generated::none();
    };
    // A parameter mode is part of the signature, so every caller is rechecked.
    Generated::of(
        vec![candidate(
            format!("Make parameter '{name}' mut"),
            edit(keyword, "mut"),
        )],
        Recheck::Clean,
    )
}
