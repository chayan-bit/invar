//! Applying validated fixes in rounds, and showing the result as a diff.

use std::collections::HashSet;
use std::path::Path;

use sha2::{Digest, Sha256};

use super::{apply, generate, validated};
use crate::front::{ErrorKind, FrontError};

/// Rounds of fix-and-recheck before giving up.
pub const MAX_ROUNDS: usize = 8;
/// Most cells the line diff fills in before it falls back to replacing the
/// differing middle as one block.
const MAX_DIFF_CELLS: usize = 4_000_000;
const CONTEXT: usize = 3;

/// The result of fixing a source.
pub struct Outcome {
    /// The source after every applied fix; equal to the input when none applied.
    pub fixed: Vec<u8>,
    /// The description of each applied fix, in order.
    pub applied: Vec<String>,
    /// The error that is left, `None` when the fixed source checks clean.
    pub remaining: Option<FrontError>,
}

/// Apply the first validated fix of the first error, recheck, and repeat, at
/// most [`MAX_ROUNDS`] times. A round that would return to a source already
/// seen stops the run, so the loop cannot oscillate.
pub fn fix(source: &[u8]) -> Outcome {
    let mut current = source.to_vec();
    let mut seen: HashSet<Vec<u8>> = HashSet::from([current.clone()]);
    let mut applied = Vec::new();
    for _ in 0..=MAX_ROUNDS {
        let error = match crate::front::parse(&current)
            .and_then(|program| crate::check::check(program, false))
        {
            Ok(_) => return outcome(current, applied, None),
            Err(error) => error,
        };
        if applied.len() == MAX_ROUNDS || error.kind != ErrorKind::Source {
            return outcome(current, applied, Some(error));
        }
        let generated = generate(error.code, &current, error.span);
        let first = validated(
            &current,
            generated.candidates,
            generated.recheck,
            generated.require_main,
        )
        .into_iter()
        .next();
        let Some(next) = first
            .and_then(|fix| apply(&current, &fix.edits).map(|edited| (fix.description, edited)))
        else {
            return outcome(current, applied, Some(error));
        };
        if !seen.insert(next.1.clone()) {
            return outcome(current, applied, Some(error));
        }
        applied.push(next.0);
        current = next.1;
    }
    unreachable!("the loop returns once the round limit is reached")
}

fn outcome(fixed: Vec<u8>, applied: Vec<String>, remaining: Option<FrontError>) -> Outcome {
    Outcome {
        fixed,
        applied,
        remaining,
    }
}

/// Replace the file with `fixed` through a temporary file in the same
/// directory, after checking that it still hashes to `checked`, the digest of
/// the bytes the fixes were computed from.
pub fn write_if_unchanged(
    path: impl AsRef<Path>,
    checked: &[u8],
    fixed: &[u8],
) -> Result<(), String> {
    let path = path.as_ref();
    let current = std::fs::read(path).map_err(|error| format!("cannot read source: {error}"))?;
    if Sha256::digest(&current)[..] != *checked {
        return Err("refusing to write: the file changed after it was checked".to_owned());
    }
    let mut temporary = path.as_os_str().to_owned();
    temporary.push(".invar-fix");
    let temporary = std::path::PathBuf::from(temporary);
    let result = std::fs::metadata(path)
        .and_then(|metadata| {
            std::fs::write(&temporary, fixed)?;
            std::fs::set_permissions(&temporary, metadata.permissions())
        })
        .and_then(|()| std::fs::rename(&temporary, path));
    if result.is_err() {
        let _ = std::fs::remove_file(&temporary);
    }
    result.map_err(|error| format!("cannot write source: {error}"))
}

/// A unified diff of two texts with three lines of context, empty when equal.
pub fn unified_diff(path: &str, old: &str, new: &str) -> String {
    let old: Vec<&str> = old.split_inclusive('\n').collect();
    let new: Vec<&str> = new.split_inclusive('\n').collect();
    let script = edit_script(&old, &new);
    if script.iter().all(|line| matches!(line, Line::Same(..))) {
        return String::new();
    }
    let mut out = format!("--- a/{path}\n+++ b/{path}\n");
    let changed: Vec<usize> = script
        .iter()
        .enumerate()
        .filter(|(_, line)| !matches!(line, Line::Same(..)))
        .map(|(at, _)| at)
        .collect();
    let mut start = 0;
    while start < changed.len() {
        let mut end = start;
        while end + 1 < changed.len() && changed[end + 1] - changed[end] <= 2 * CONTEXT + 1 {
            end += 1;
        }
        let from = changed[start].saturating_sub(CONTEXT);
        let to = (changed[end] + CONTEXT + 1).min(script.len());
        hunk(&mut out, &script[..from], &script[from..to]);
        start = end + 1;
    }
    out
}

enum Line<'a> {
    Same(&'a str),
    Removed(&'a str),
    Added(&'a str),
}

fn hunk(out: &mut String, before: &[Line<'_>], lines: &[Line<'_>]) {
    let old_start = before
        .iter()
        .filter(|line| !matches!(line, Line::Added(..)))
        .count();
    let new_start = before
        .iter()
        .filter(|line| !matches!(line, Line::Removed(..)))
        .count();
    let old_len = lines
        .iter()
        .filter(|line| !matches!(line, Line::Added(..)))
        .count();
    let new_len = lines
        .iter()
        .filter(|line| !matches!(line, Line::Removed(..)))
        .count();
    let begin = |start: usize, len: usize| if len == 0 { start } else { start + 1 };
    out.push_str(&format!(
        "@@ -{},{} +{},{} @@\n",
        begin(old_start, old_len),
        old_len,
        begin(new_start, new_len),
        new_len
    ));
    for line in lines {
        let (mark, text) = match line {
            Line::Same(text) => (' ', text),
            Line::Removed(text) => ('-', text),
            Line::Added(text) => ('+', text),
        };
        out.push(mark);
        out.push_str(text);
        if !text.ends_with('\n') {
            out.push_str("\n\\ No newline at end of file\n");
        }
    }
}

/// A line-level edit script: the common prefix and suffix are kept, and the
/// middle is aligned by a longest common subsequence when it is small enough.
fn edit_script<'a>(old: &[&'a str], new: &[&'a str]) -> Vec<Line<'a>> {
    let prefix = old.iter().zip(new).take_while(|(a, b)| a == b).count();
    let suffix = old[prefix..]
        .iter()
        .rev()
        .zip(new[prefix..].iter().rev())
        .take_while(|(a, b)| a == b)
        .count();
    let (old_mid, new_mid) = (
        &old[prefix..old.len() - suffix],
        &new[prefix..new.len() - suffix],
    );
    let mut script: Vec<Line<'a>> = old[..prefix].iter().map(|text| Line::Same(text)).collect();
    if (old_mid.len() + 1).saturating_mul(new_mid.len() + 1) > MAX_DIFF_CELLS {
        script.extend(old_mid.iter().map(|text| Line::Removed(text)));
        script.extend(new_mid.iter().map(|text| Line::Added(text)));
    } else {
        let width = new_mid.len() + 1;
        let mut common = vec![0u32; (old_mid.len() + 1) * width];
        for i in (0..old_mid.len()).rev() {
            for j in (0..new_mid.len()).rev() {
                common[i * width + j] = if old_mid[i] == new_mid[j] {
                    common[(i + 1) * width + j + 1] + 1
                } else {
                    common[(i + 1) * width + j].max(common[i * width + j + 1])
                };
            }
        }
        let (mut i, mut j) = (0, 0);
        while i < old_mid.len() || j < new_mid.len() {
            if i < old_mid.len() && j < new_mid.len() && old_mid[i] == new_mid[j] {
                script.push(Line::Same(old_mid[i]));
                i += 1;
                j += 1;
            } else if j == new_mid.len()
                || (i < old_mid.len() && common[(i + 1) * width + j] >= common[i * width + j + 1])
            {
                script.push(Line::Removed(old_mid[i]));
                i += 1;
            } else {
                script.push(Line::Added(new_mid[j]));
                j += 1;
            }
        }
    }
    script.extend(
        old[old.len() - suffix..]
            .iter()
            .map(|text| Line::Same(text)),
    );
    script
}
