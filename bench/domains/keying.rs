//! The no-keying rule (#56, extending `tests/pnf_family_admission.rs`): no
//! file under `src/` or `crates/` may contain a suite program id, a kernel
//! name as a string, a suite fixture path, a held-out hash, or a domain word
//! as a string literal (the only way to use one as a dispatch key). The compiler must treat these programs like any
//! others.
use super::{SUITES, manifest, programs};
use serde_json::Value;
use std::{fs, io, path::Path};

const FIXTURE_DIR: &str = "bench/domains";
const HARNESS: &str = "src/bin/invar-bench";

pub struct Needles {
    exact: Vec<String>,
    domain_words: Vec<String>,
}

pub fn needles() -> Result<Needles, String> {
    let mut exact = vec![FIXTURE_DIR.to_owned(), "manifest-v3-proposal".to_owned()];
    for suite in SUITES {
        for program in programs(suite)? {
            exact.push(program.id.clone());
            exact.push(format!("\"{}\"", program.kernel));
        }
    }
    let proposal = manifest::build()?;
    for suite in SUITES {
        let section = &proposal["suites"][suite];
        exact.push(section["held_out_sha256"].as_str().unwrap_or("").to_owned());
        for e in section["programs"].as_array().into_iter().flatten() {
            if e["split"] == "held_out" {
                for key in ["source_sha256", "canonical_sha256"] {
                    exact.push(Value::as_str(&e[key]).unwrap_or("").to_owned());
                }
            }
        }
    }
    exact.sort();
    exact.dedup();
    if exact.iter().any(String::is_empty) {
        return Err("empty needle".into());
    }
    Ok(Needles {
        exact,
        domain_words: SUITES.iter().map(|s| (*s).to_owned()).collect(),
    })
}

fn walk(dir: &Path, out: &mut Vec<std::path::PathBuf>) -> io::Result<()> {
    for item in fs::read_dir(dir)? {
        let path = item?.path();
        let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
        if path.is_dir() {
            if name != "target" && !name.starts_with('.') {
                walk(&path, out)?;
            }
        } else {
            out.push(path);
        }
    }
    Ok(())
}

/// A string literal that is exactly a domain word, anywhere in the file: in a
/// comparison, a `matches!`, a lookup table, split across lines or inside any
/// macro. No compiler file has a reason to spell a suite name, so the scan does
/// not try to guess the surrounding expression. Case is ignored.
fn quoted_domain_word(lowered: &str, word: &str) -> bool {
    lowered.contains(&format!("\"{word}\""))
}

/// Every violation under `root/src` and `root/crates`, as `path: what`.
pub fn scan(root: &Path, needles: &Needles) -> io::Result<Vec<String>> {
    let mut files = Vec::new();
    for top in ["src", "crates"] {
        if root.join(top).is_dir() {
            walk(&root.join(top), &mut files)?;
        }
    }
    files.sort();
    let mut found = Vec::new();
    for path in files {
        let Ok(text) = fs::read_to_string(&path) else {
            continue; // not UTF-8: a binary, never source
        };
        let shown = path
            .strip_prefix(root)
            .unwrap_or(&path)
            .display()
            .to_string();
        for needle in &needles.exact {
            // The bench harness includes the suites by path; that is the one
            // reference a harness binary may make. Every other needle applies.
            if needle == FIXTURE_DIR && shown.starts_with(HARNESS) {
                continue;
            }
            if text.contains(needle.as_str()) {
                found.push(format!("{shown}: {needle}"));
            }
        }
        let lowered = text.to_lowercase();
        for word in &needles.domain_words {
            if quoted_domain_word(&lowered, word) {
                found.push(format!("{shown}: quoted domain word \"{word}\""));
            }
        }
    }
    Ok(found)
}
