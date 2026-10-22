//! Generators for `TYPE_MISMATCH` on an integer literal and `NAME_UNKNOWN`.

use std::collections::BTreeSet;

use super::{Candidate, Edit, Generated, Recheck, scope};
use crate::front::{Span, StmtKind};

/// Names this far (in single-character edits) or nearer are candidates.
const MAX_DISTANCE: usize = 2;

/// Identifiers longer than this (in characters) are never compared: the
/// parser sets no length bound, and the distance work grows with the product
/// of two lengths.
const MAX_NAME_LEN: usize = 64;

/// The most characters of a misspelled call for which the intrinsics are
/// probed at distance 2: the probe tries every string two edits away.
const MAX_PROBE_LEN: usize = 12;

fn replace(span: Span, description: String, replacement: &str) -> Candidate {
    Candidate {
        description,
        edits: vec![Edit {
            span,
            replacement: replacement.to_owned(),
        }],
    }
}

/// `TYPE_MISMATCH` on a literal written as digits and an `i32` or `u32`
/// suffix: write the other suffix. The recheck rejects it when the other type
/// does not fit, for instance a literal that is out of range.
pub(super) fn literal_suffix(source: &[u8], span: Span) -> Generated {
    let Some(text) = source
        .get(span.start..span.end)
        .and_then(|text| std::str::from_utf8(text).ok())
    else {
        return Generated::none();
    };
    let (digits, other) = if let Some(digits) = text.strip_suffix("i32") {
        (digits, "u32")
    } else if let Some(digits) = text.strip_suffix("u32") {
        (digits, "i32")
    } else {
        return Generated::none();
    };
    if digits.is_empty() || !digits.bytes().all(|byte| byte.is_ascii_digit()) {
        return Generated::none();
    }
    Generated::of(
        vec![replace(
            span,
            format!("Write the literal as {other}"),
            &format!("{digits}{other}"),
        )],
        Recheck::OtherFunctions,
    )
}

/// `NAME_UNKNOWN` on a binding or a called function: the one nearest name in
/// scope within [`MAX_DISTANCE`] edits.
pub(super) fn nearest_name(source: &[u8], span: Span) -> Generated {
    let Some(text) = source
        .get(span.start..span.end)
        .and_then(|text| std::str::from_utf8(text).ok())
    else {
        return Generated::none();
    };
    let Ok(program) = crate::front::parse(source) else {
        return Generated::none();
    };
    let mut call = false;
    let (unknown, names): (&str, BTreeSet<String>) =
        if let Some(name) = scope::bare_name(source, span) {
            let Some(function) = scope::enclosing(&program, span) else {
                return Generated::none();
            };
            let mut visible = Vec::new();
            scope::preceding(&function.body, span, &mut visible);
            let mut names: BTreeSet<String> = function
                .params
                .iter()
                .map(|param| param.name.clone())
                .collect();
            for stmt in visible {
                match &stmt.kind {
                    StmtKind::Binding { name, .. } | StmtKind::For { name, .. } => {
                        names.insert(name.clone());
                    }
                    _ => {}
                }
            }
            (name, names)
        } else if let Some((name, _)) = text.split_once('(') {
            let name = name.trim_end();
            if scope::bare_name(
                name.as_bytes(),
                Span {
                    start: 0,
                    end: name.len(),
                },
            )
            .is_none()
            {
                return Generated::none();
            }
            call = true;
            let names = program
                .functions
                .iter()
                .map(|function| function.name.clone())
                .chain(
                    program
                        .foreign_functions
                        .iter()
                        .map(|function| function.name.clone()),
                )
                .collect();
            (name, names)
        } else {
            return Generated::none();
        };
    let Some((nearest, distance)) = unique_nearest(unknown, &names) else {
        return Generated::none();
    };
    // A call may have meant an intrinsic (`dos` is one edit from `cos`), and
    // the recheck cannot tell which was meant: no fix when one is as near.
    if call && intrinsic_within(unknown, distance) {
        return Generated::none();
    }
    let at = Span {
        start: span.start,
        end: span.start + unknown.len(),
    };
    Generated::of(
        vec![replace(at, format!("Use '{nearest}'"), nearest)],
        Recheck::OtherFunctions,
    )
}

/// The only name within [`MAX_DISTANCE`] edits of `unknown` that is nearest,
/// with its distance; never for a name so short that one edit rewrites it,
/// and never when `unknown` is longer than [`MAX_NAME_LEN`].
fn unique_nearest<'a>(unknown: &str, names: &'a BTreeSet<String>) -> Option<(&'a str, usize)> {
    let length = unknown.chars().count();
    if length > MAX_NAME_LEN {
        return None;
    }
    let mut best: Option<(usize, &str)> = None;
    let mut tied = false;
    for name in names {
        let other = name.chars().count();
        if other > MAX_NAME_LEN || length.abs_diff(other) > MAX_DISTANCE {
            continue;
        }
        let distance = edit_distance(unknown, name);
        if distance == 0 || distance > MAX_DISTANCE || distance >= length {
            continue;
        }
        match best {
            Some((nearest, _)) if distance > nearest => {}
            Some((nearest, _)) if distance == nearest => tied = true,
            _ => {
                best = Some((distance, name));
                tied = false;
            }
        }
    }
    best.filter(|_| !tied)
        .map(|(distance, name)| (name, distance))
}

const ALPHABET: &[u8] = b"abcdefghijklmnopqrstuvwxyz0123456789_";

/// Call `visit` on every string one edit away from `word` (an ASCII
/// identifier) until it returns true.
fn one_edit(word: &[u8], visit: &mut dyn FnMut(&[u8]) -> bool) -> bool {
    let mut near = Vec::with_capacity(word.len() + 1);
    for at in 0..=word.len() {
        for &letter in ALPHABET {
            near.clear();
            near.extend_from_slice(&word[..at]);
            near.push(letter);
            near.extend_from_slice(&word[at..]);
            if visit(&near) {
                return true;
            }
        }
        if at < word.len() {
            near.clear();
            near.extend_from_slice(&word[..at]);
            near.extend_from_slice(&word[at + 1..]);
            if visit(&near) {
                return true;
            }
            for &letter in ALPHABET.iter().filter(|letter| **letter != word[at]) {
                near.clear();
                near.extend_from_slice(word);
                near[at] = letter;
                if visit(&near) {
                    return true;
                }
            }
        }
    }
    false
}

/// Whether an intrinsic is within `distance` edits of `unknown`. The
/// intrinsics have no list to scan, so every string that near is tried; a
/// longer name than [`MAX_PROBE_LEN`] at distance 2 is assumed to be near one.
fn intrinsic_within(unknown: &str, distance: usize) -> bool {
    if !unknown.is_ascii() || (distance >= MAX_DISTANCE && unknown.len() > MAX_PROBE_LEN) {
        return true;
    }
    let is_intrinsic = |word: &[u8]| std::str::from_utf8(word).is_ok_and(crate::check::intrinsic);
    one_edit(unknown.as_bytes(), &mut |near| {
        is_intrinsic(near)
            || (distance >= MAX_DISTANCE && one_edit(near, &mut |far| is_intrinsic(far)))
    })
}

/// Levenshtein distance over characters.
pub(super) fn edit_distance(a: &str, b: &str) -> usize {
    let b: Vec<char> = b.chars().collect();
    let mut row: Vec<usize> = (0..=b.len()).collect();
    for (i, ca) in a.chars().enumerate() {
        let mut diagonal = row[0];
        row[0] = i + 1;
        for (j, cb) in b.iter().enumerate() {
            let above = row[j + 1];
            row[j + 1] = (row[j] + 1)
                .min(above + 1)
                .min(diagonal + usize::from(ca != *cb));
            diagonal = above;
        }
    }
    row[b.len()]
}
