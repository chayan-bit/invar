//! The coverage gate: every example in `examples/index.json` that is expected
//! to fail with an ownership-class code (`OWN_*`) either has a validated fix
//! or is on the no-fix list below with the reason. The measured rate is
//! printed, and neither the baseline nor the list may drift: a new fix moves
//! its example from the list to the baseline, and a lost fix fails the test.

use std::{collections::BTreeMap, path::PathBuf, process::Command};

use serde_json::Value;

/// The recorded rate: this many of this many ownership-class negatives have a
/// validated fix. Raise `BASELINE_FIXED` when a fix lands, and
/// `BASELINE_TOTAL` only together with it; the rate must never fall below
/// `BASELINE_FIXED / BASELINE_TOTAL`, so a negative added without a fix needs a
/// fix elsewhere to keep the ratio.
const BASELINE_FIXED: usize = 1;
const BASELINE_TOTAL: usize = 8;

/// Whether `fixed` of `total` is at least the recorded rate.
fn rate_held(fixed: usize, total: usize) -> bool {
    fixed * BASELINE_TOTAL >= BASELINE_FIXED * total
}

/// Examples whose diagnostic has a validated fix today.
const FIXED: &[&str] = &["examples/language/use-after-move.inv"];

/// Examples that have none, and why. Fixes are never speculative: each reason
/// is a case where no edit is safe or where the remedy changes meaning.
const NO_FIX: &[(&str, &str)] = &[
    (
        "examples/language/loan-conflict.inv",
        "one name is lent mutably and read-only in the same call: the remedy is a copy, which changes meaning",
    ),
    (
        "examples/types/record-use-after-move.inv",
        "the earlier move is a `let q = move p`, not a call argument: no parameter to relax, and `move` is required for an owned record",
    ),
    (
        "examples/bytes/foreign-use-after-take.inv",
        "the consumer is an `extern c` declaration: its `take` is part of the foreign contract, not a source parameter to relax",
    ),
    (
        "examples/generics/closure-double-move.inv",
        "the value is moved by a closure capture (`closure [take ys: ..]`), not by a call argument: there is no parameter mode to relax",
    ),
    (
        "examples/views/invalid-return.inv",
        "OWN_RETURN_BORROW has no automatic fix (a view of a local cannot be returned): the note states the rule and the two valid forms",
    ),
    (
        "examples/views/overlap.inv",
        "two mutable views of one array overlap on element 1: any edit changes which elements each view sees",
    ),
    (
        "examples/views/owner-move.inv",
        "the owner is moved while a view of it is read later: ending the view or the move changes the program",
    ),
];

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

/// `(path, expected code)` of every ownership-class negative in the index.
fn ownership_negatives() -> Vec<(String, String)> {
    let index: Value =
        serde_json::from_slice(&std::fs::read(root().join("examples/index.json")).unwrap())
            .unwrap();
    index["examples"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|entry| entry["expect"]["status"] == "source_error")
        .filter_map(|entry| {
            let code = entry["expect"]["code"].as_str()?;
            code.starts_with("OWN_")
                .then(|| (entry["path"].as_str().unwrap().to_owned(), code.to_owned()))
        })
        .collect()
}

fn fix_count(path: &str) -> (String, usize) {
    let output = Command::new(env!("CARGO_BIN_EXE_invar"))
        .args(["check", path, "--format", "json"])
        .current_dir(root())
        .output()
        .unwrap();
    let document: Value = serde_json::from_slice(&output.stdout).unwrap();
    let diagnostic = &document["diagnostics"][0];
    (
        diagnostic["code"].as_str().unwrap().to_owned(),
        diagnostic["suggested_fixes"].as_array().unwrap().len(),
    )
}

#[test]
fn issue_dx_22_ownership_negatives_have_a_fix_or_a_recorded_reason() {
    let negatives = ownership_negatives();
    assert!(
        negatives.len() >= BASELINE_TOTAL,
        "the corpus has {} ownership negatives",
        negatives.len()
    );
    let reasons: BTreeMap<&str, &str> = NO_FIX.iter().copied().collect();
    assert_eq!(reasons.len(), NO_FIX.len(), "a path is listed twice");
    let mut fixed = Vec::new();
    for (path, code) in &negatives {
        let (reported, count) = fix_count(path);
        assert_eq!(&reported, code, "{path}");
        let listed = reasons.contains_key(path.as_str());
        let baseline = FIXED.contains(&path.as_str());
        assert!(
            listed != baseline,
            "{path} must be in exactly one of FIXED and NO_FIX"
        );
        if count > 0 {
            assert!(
                baseline,
                "{path} gained a fix: move it from NO_FIX to FIXED"
            );
            fixed.push(path.as_str());
        } else {
            assert!(listed, "{path} lost its fix: FIXED may not shrink");
        }
    }
    for path in FIXED.iter().copied().chain(reasons.keys().copied()) {
        assert!(
            negatives.iter().any(|(listed, _)| listed == path),
            "{path} is not an ownership negative of the index"
        );
    }
    println!(
        "issue_dx_22 coverage: {} of {} ownership-class negatives have a validated fix ({:.1}%)",
        fixed.len(),
        negatives.len(),
        100.0 * fixed.len() as f64 / negatives.len() as f64
    );
    for (path, reason) in NO_FIX {
        println!("issue_dx_22 no fix: {path}: {reason}");
    }
    assert!(FIXED.len() >= BASELINE_FIXED, "FIXED may not shrink");
    assert!(
        rate_held(fixed.len(), negatives.len()),
        "the fix rate fell below {BASELINE_FIXED} of {BASELINE_TOTAL}: {} of {}",
        fixed.len(),
        negatives.len()
    );
}

#[test]
fn issue_dx_22_coverage_rate_cannot_fall_by_adding_negatives() {
    assert!(rate_held(1, 8));
    // A ninth negative without a fix lowers the rate from 12.5% to 11.1%.
    assert!(!rate_held(1, 9));
    assert!(rate_held(2, 9));
    assert!(!rate_held(2, 17));
    assert!(!rate_held(0, 8));
}
