//! Validated source fixes for diagnostics.
//!
//! A generator proposes ordered candidate edits for one diagnostic. A candidate
//! is emitted only after the edited source is reparsed and rechecked, so a fix
//! is never speculative: an unvalidated candidate is dropped. Bounds and order
//! are fixed, so the same source and diagnostic give the same fixes.

use serde_json::{Value, json};

use crate::diagnostics::{location, valid_edit_spans};
use crate::front::Span;

mod builtin;
mod hoist;
mod moves;
mod names;
mod ownership;
mod rounds;
mod scope;

pub use builtin::unit_conversion;
pub use rounds::{MAX_ROUNDS, Outcome, fix, unified_diff, write_if_unchanged};

/// Candidates considered per diagnostic; later ones are ignored.
pub const MAX_CANDIDATES: usize = 8;
/// Largest edited source the engine rechecks (the parser's own bound).
const MAX_SOURCE_BYTES: usize = 1_048_576;

/// One replacement of a half-open byte range of the original source.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Edit {
    pub span: Span,
    pub replacement: String,
}

/// An ordered set of edits that together address one diagnostic.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Candidate {
    pub description: String,
    pub edits: Vec<Edit>,
}

/// What a recheck may still report after the edit.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Recheck {
    /// The edited source must check clean. Required for edits outside one
    /// function body, such as a parameter mode, which can break any caller.
    Clean,
    /// Another error is tolerated when it lies outside the one function body
    /// that holds every edit. `check` reports the first error only, so this is
    /// how a file with several errors still gets a fix for the first one. An
    /// edit inside a body cannot create an error in another function, and one
    /// that creates an error in its own function is still dropped.
    OtherFunctions,
}

/// Apply sorted, non-overlapping edits. `None` if the edits are invalid or the
/// result would exceed the source bound.
pub fn apply(source: &[u8], edits: &[Edit]) -> Option<Vec<u8>> {
    let spans: Vec<Span> = edits.iter().map(|edit| edit.span).collect();
    if !valid_edit_spans(&spans, source.len()) {
        return None;
    }
    let mut out = Vec::new();
    let mut at = 0;
    for edit in edits {
        out.extend_from_slice(&source[at..edit.span.start]);
        out.extend_from_slice(edit.replacement.as_bytes());
        at = edit.span.end;
        if out.len() > MAX_SOURCE_BYTES {
            return None;
        }
    }
    out.extend_from_slice(&source[at..]);
    (out.len() <= MAX_SOURCE_BYTES).then_some(out)
}

/// Where, in the edited source, the one function whose body holds every edit
/// lies.
fn function_range(source: &[u8], edits: &[Edit]) -> Option<Span> {
    let program = crate::front::parse(source).ok()?;
    let function = program.functions.iter().find(|function| {
        edits.iter().all(|edit| {
            function.body.span.start <= edit.span.start && edit.span.end <= function.body.span.end
        })
    })?;
    let delta: isize = edits
        .iter()
        .map(|edit| edit.replacement.len() as isize - (edit.span.end - edit.span.start) as isize)
        .sum();
    Some(Span {
        start: function.span.start,
        end: usize::try_from(function.span.end as isize + delta).ok()?,
    })
}

/// Reparse and recheck the edited source.
pub fn validates(
    source: &[u8],
    candidate: &Candidate,
    recheck: Recheck,
    require_main: bool,
) -> bool {
    if candidate.edits.is_empty() {
        return false;
    }
    let Some(edited) = apply(source, &candidate.edits) else {
        return false;
    };
    match crate::front::parse(&edited)
        .and_then(|program| crate::check::check(program, require_main))
    {
        Ok(_) => true,
        Err(error) => {
            let (Recheck::OtherFunctions, Some(span)) = (recheck, error.span) else {
                return false;
            };
            error.kind == crate::front::ErrorKind::Source
                && function_range(source, &candidate.edits)
                    .is_some_and(|own| span.end <= own.start || own.end <= span.start)
        }
    }
}

/// Candidates that pass validation, at most [`MAX_CANDIDATES`], in order.
pub fn validated(
    source: &[u8],
    candidates: Vec<Candidate>,
    recheck: Recheck,
    require_main: bool,
) -> Vec<Candidate> {
    candidates
        .into_iter()
        .take(MAX_CANDIDATES)
        .filter(|candidate| validates(source, candidate, recheck, require_main))
        .collect()
}

/// The `suggested_fixes` entries for validated candidates. A candidate whose
/// edit has no valid location is dropped.
pub fn to_json(path: &str, source: &[u8], candidates: &[Candidate]) -> Vec<Value> {
    candidates
        .iter()
        .filter_map(|candidate| {
            let edits = candidate
                .edits
                .iter()
                .map(|edit| {
                    location(path, source, edit.span)
                        .map(|span| json!({"span": span, "replacement": edit.replacement}))
                })
                .collect::<Option<Vec<_>>>()?;
            Some(json!({"description": candidate.description, "edits": edits}))
        })
        .collect()
}

/// What a generator proposes and how its candidates are validated.
pub struct Generated {
    pub candidates: Vec<Candidate>,
    pub recheck: Recheck,
    /// Whether the recheck requires `main`, as `invar run` does.
    pub require_main: bool,
}

impl Generated {
    pub(crate) fn none() -> Self {
        Self::of(Vec::new(), Recheck::Clean)
    }

    pub(crate) fn of(candidates: Vec<Candidate>, recheck: Recheck) -> Self {
        Self {
            candidates,
            recheck,
            require_main: false,
        }
    }
}

/// Generator candidates for a diagnostic of the given code and span. Sources
/// that do not parse give none.
pub fn generate(code: &str, source: &[u8], span: Option<Span>) -> Generated {
    match (code, span) {
        ("ENTRYPOINT_MISSING", _) => Generated {
            require_main: true,
            ..Generated::of(builtin::entrypoint(source), Recheck::Clean)
        },
        ("OWN_MOVE_REQUIRED", Some(span)) => ownership::move_required(source, span),
        ("OWN_MUT_REQUIRED", Some(span)) => ownership::mut_required(source, span),
        ("TYPE_MISMATCH", Some(span)) => names::literal_suffix(source, span),
        ("NAME_UNKNOWN", Some(span)) => names::nearest_name(source, span),
        ("OWN_USE_AFTER_MOVE", Some(span)) => moves::relax_take(source, span),
        ("OWN_BORROW_SOURCE", Some(span)) => hoist::generate(source, span, hoist::Site::Argument),
        ("OWN_BORROW_CONFLICT", Some(span)) => hoist::generate(source, span, hoist::Site::Inside),
        _ => Generated::none(),
    }
}

/// Validated `suggested_fixes` for a diagnostic.
pub fn suggested_fixes(code: &str, path: &str, source: &[u8], span: Option<Span>) -> Vec<Value> {
    let generated = generate(code, source, span);
    to_json(
        path,
        source,
        &validated(
            source,
            generated.candidates,
            generated.recheck,
            generated.require_main,
        ),
    )
}

/// Notes that explain a diagnostic without changing the source.
pub fn notes(code: &str, source: &[u8], span: Option<Span>) -> Vec<String> {
    match (code, span) {
        ("OWN_USE_AFTER_MOVE", Some(span)) => moves::moved_here(source, span),
        ("OWN_RETURN_BORROW", Some(span)) => moves::return_borrow(source, span),
        _ => Vec::new(),
    }
}
