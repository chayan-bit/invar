//! Shared GNET-05 corpus guards: consistent identifier renaming, comment and
//! whitespace disguises, span-free emitted code, and source-tree walks.
#![allow(dead_code)]
use std::fs;
use std::path::{Path, PathBuf};

pub const RENAME_PREFIX: &str = "zz_";

/// Consistent renaming of every user identifier except the entry `run`.
pub fn renamed(source: &str) -> String {
    let mut out = String::new();
    let mut word = String::new();
    let flush = |word: &mut String, out: &mut String| {
        if !word.is_empty() {
            let user = word.starts_with(|c: char| c.is_ascii_alphabetic() || c == '_')
                && !KEYWORDS.contains(&word.as_str())
                && word != "run";
            if user {
                out.push_str(RENAME_PREFIX);
            }
            out.push_str(word);
            word.clear();
        }
    };
    let mut digits = false;
    for c in source.chars() {
        if c.is_ascii_alphanumeric() || c == '_' {
            if word.is_empty() {
                digits = c.is_ascii_digit();
            }
            word.push(c);
            continue;
        }
        if digits {
            out.push_str(&word);
            word.clear();
        } else {
            flush(&mut word, &mut out);
        }
        out.push(c);
    }
    if digits {
        out.push_str(&word);
    } else {
        flush(&mut word, &mut out);
    }
    out
}

/// Comments naming benchmark IDs and fixture paths, and changed whitespace:
/// a different source hash, the same program.
pub fn recommented(source: &str) -> String {
    let body = source.replace("\n", "\n\n").replace("    ", "\t");
    format!(
        "// gnet05-divide-00 tests/fixtures/gnet_parallel/divide_conquer.inv\n{body}// manifest id gnet05-frontier-11\n"
    )
}

pub const KEYWORDS: &[&str] = &[
    "fn",
    "let",
    "var",
    "if",
    "else",
    "while",
    "for",
    "in",
    "return",
    "true",
    "false",
    "and",
    "or",
    "not",
    "i32",
    "u32",
    "bool",
    "Array",
    "read",
    "filled_i32",
    "filled_u32",
    "wrapping_add",
    "wrapping_mul",
    "f32",
    "Bool",
    "Unit",
    "mut",
    "take",
    "move",
    "len",
    "fma",
    "filled_f32",
    "f32_from_i32",
    "f32_from_u32",
];

/// Replaces the source byte offsets in `start=N` / `end=N` trap spans with `#`.
pub fn without_spans(code: &str) -> String {
    let mut out = String::with_capacity(code.len());
    let mut rest = code;
    while let Some(at) = rest.find("=") {
        let (head, tail) = rest.split_at(at + 1);
        out.push_str(head);
        rest = tail;
        if head.ends_with("start=") || head.ends_with("end=") {
            let digits = rest.len() - rest.trim_start_matches(|c: char| c.is_ascii_digit()).len();
            if digits > 0 {
                out.push('#');
                rest = &rest[digits..];
            }
        }
    }
    out.push_str(rest);
    out
}

pub fn walk(dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(entries) = fs::read_dir(dir) else {
        return;
    };
    for entry in entries {
        let path = entry.unwrap().path();
        if path.is_dir() {
            walk(&path, out);
        } else if path.extension().is_some_and(|e| e == "rs") {
            out.push(path);
        }
    }
}
