//! Runnable source corpus shared by the GNET-02 corpus and native validations.

use crate::corpus_sample::walk;
use invar::{check, front};
use std::path::Path;

/// Every source under tests/fixtures and examples that parses and checks as an
/// executable program (system declarations, negative fixtures and libraries do not).
pub fn runnable_corpus() -> Vec<(String, check::CheckedProgram)> {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let mut files = Vec::new();
    walk(&root.join("tests/fixtures"), &mut files);
    walk(&root.join("examples"), &mut files);
    files
        .into_iter()
        .filter_map(|path| {
            let bytes = std::fs::read(&path).ok()?;
            let checked = check::check(front::parse(&bytes).ok()?, true).ok()?;
            // Fixture names use `/` on every host so name filters match on Windows.
            let name = path
                .strip_prefix(root)
                .unwrap()
                .display()
                .to_string()
                .replace('\\', "/");
            Some((name, checked))
        })
        .collect()
}
