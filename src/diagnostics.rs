//! Versioned, bounded CLI diagnostics. Source locations refer to UTF-8 bytes.

use serde_json::{Value, json};

use crate::front::Span;

pub const SCHEMA: &str = "invar-diagnostics-v1";

/// Convert a half-open byte range to a source location. Invalid ranges are
/// omitted rather than being replaced with a plausible but false location.
pub fn location(path: &str, source: &[u8], span: Span) -> Option<Value> {
    let text = std::str::from_utf8(source).ok()?;
    if span.start > span.end
        || span.end > text.len()
        || !text.is_char_boundary(span.start)
        || !text.is_char_boundary(span.end)
    {
        return None;
    }
    let position = |offset: usize| {
        let prefix = &text[..offset];
        let line = prefix.bytes().filter(|byte| *byte == b'\n').count() + 1;
        let column = prefix.rsplit('\n').next().unwrap_or("").chars().count() + 1;
        (line, column)
    };
    let (start_line, start_column) = position(span.start);
    let (end_line, end_column) = position(span.end);
    Some(json!({
        "path": path,
        "start": span.start,
        "end": span.end,
        "start_line": start_line,
        "start_column": start_column,
        "end_line": end_line,
        "end_column": end_column,
    }))
}

pub fn valid_edit_spans(spans: &[Span], source_len: usize) -> bool {
    spans
        .iter()
        .all(|span| span.start <= span.end && span.end <= source_len)
        && spans.windows(2).all(|pair| pair[0].end <= pair[1].start)
}

pub fn diagnostic(
    code: &str,
    message: &str,
    path: &str,
    source: &[u8],
    span: Option<Span>,
) -> Value {
    let bounded = |value: &str, max: usize| value.chars().take(max).collect::<String>();
    let primary = span.and_then(|span| location(path, source, span));
    let fixes = crate::fixes::suggested_fixes(code, path, source, span);
    let notes: Vec<String> = crate::fixes::notes(code, source, span)
        .iter()
        .map(|note| bounded(note, 1024))
        .collect();
    json!({
        "code": code,
        "severity": "error",
        "message": bounded(message, 4096),
        "path": bounded(path, 4096),
        "span": primary,
        "secondary_spans": [],
        "suggested_fixes": fixes,
        "notes": notes,
        "ir_path": null,
    })
}

pub fn document(status: &str, mut diagnostics: Vec<Value>) -> Value {
    diagnostics.sort_by_key(|entry| {
        (
            entry["span"]["start"].as_u64().unwrap_or(u64::MAX),
            entry["code"].as_str().unwrap_or("").to_owned(),
        )
    });
    document_in_order(status, diagnostics)
}

pub fn document_in_order(status: &str, diagnostics: Vec<Value>) -> Value {
    if diagnostics.len() > 64 {
        return json!({
            "schema": SCHEMA,
            "schema_version": 1,
            "status": status,
            "diagnostics": [diagnostic(
                "DIAGNOSTIC_COUNT_LIMIT",
                "too many diagnostics to emit",
                "",
                &[],
                None,
            )],
        });
    }
    json!({
        "schema": SCHEMA,
        "schema_version": 1,
        "status": status,
        "diagnostics": diagnostics,
    })
}
