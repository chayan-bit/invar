//! Worker-count invariance checked at run time (#209): the evidence behind the
//! N0 oracle (`docs/spec/invar-v0.2-axis-n.md` NA7.4 and NA7.13), recorded in
//! the run receipt as what was actually compared, never as a claim.

use serde_json::{Value, json};

use crate::{check::CheckedProgram, eval::EvalError, mir, schedule};

/// Most worker counts one check runs.
pub const MAX_COUNTS: usize = 16;

/// What every run is compared on, against the reference evaluator.
pub const COMPARED: [&str; 3] = ["result", "trap", "fuel"];

/// Parse `--check-workers`: comma-separated, strictly increasing counts in
/// `1..=schedule::MAX_WORKERS`, at most [`MAX_COUNTS`] of them.
pub fn parse_counts(text: &str) -> Option<Vec<usize>> {
    let counts = text
        .split(',')
        .map(|count| {
            (!count.is_empty() && count.bytes().all(|byte| byte.is_ascii_digit()))
                .then(|| count.parse::<usize>().ok())
                .flatten()
                .filter(|count| (1..=schedule::MAX_WORKERS).contains(count))
        })
        .collect::<Option<Vec<_>>>()?;
    let increasing = counts.windows(2).all(|pair| pair[0] < pair[1]);
    (increasing && counts.len() <= MAX_COUNTS).then_some(counts)
}

/// The compared part of one run: its result or error, and the fuel it used.
fn outcome(observation: &crate::eval::Observation) -> String {
    format!("{:?}; fuel {}", observation.result, observation.steps_used)
}

/// The index of the first run whose outcome differs from `expected`.
fn first_difference(expected: &str, runs: &[String]) -> Option<usize> {
    runs.iter().position(|run| run != expected)
}

/// Run `program` on the reference evaluator and on the CPU workers at each of
/// `counts`, compare result, trap (category and position) and fuel used, and
/// return the receipt's `invariance` object, or `INVARIANCE_MISMATCH` naming
/// the first worker count that differs from the reference.
pub fn check_workers(
    checked: &CheckedProgram,
    program: &mir::Program,
    source: &[u8],
    fuel: u64,
    counts: &[usize],
) -> Result<Value, EvalError> {
    let reference = crate::eval::observe(checked, fuel);
    let expected = outcome(&reference);
    let mut outcomes = Vec::with_capacity(counts.len());
    let mut runs = Vec::with_capacity(counts.len());
    for &workers in counts {
        let run = schedule::run_source(source, program, fuel, workers).map_err(|error| {
            EvalError::Tool {
                code: error.code(),
                message: format!("CPU execution failed: {error:?}"),
            }
        })?;
        outcomes.push(outcome(&run.observation));
        runs.push(json!({
            "workers": workers,
            "published_regions": run.executed.len(),
            "worker_joins": run.joined,
            "last_split": run.chunks.iter().map(|chunk| chunk.start).collect::<Vec<_>>(),
        }));
    }
    if let Some(at) = first_difference(&expected, &outcomes) {
        return Err(mismatch(counts[at], &outcomes[at], &expected));
    }
    Ok(json!({
        "engines": ["reference", "cpu-parallel"],
        "workers": counts,
        "compared": COMPARED,
        "fuel_used": reference.steps_used,
        "runs": runs,
        "levels": crate::axes::levels_used(crate::axes::declarations(program)),
    }))
}

fn mismatch(workers: usize, got: &str, expected: &str) -> EvalError {
    EvalError::Tool {
        code: "INVARIANCE_MISMATCH",
        message: format!(
            "cpu-parallel at {workers} workers gave {got}, the reference evaluator gave {expected}"
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn worker_lists_are_increasing_bounded_counts() {
        assert_eq!(parse_counts("1,2,7,64"), Some(vec![1, 2, 7, 64]));
        assert_eq!(parse_counts("3"), Some(vec![3]));
        for refused in [
            "",
            "0",
            "65",
            "2,1",
            "1,1",
            "1,,2",
            "1, 2",
            "+1",
            "1,2,",
            "x",
            "18446744073709551617",
        ] {
            assert_eq!(parse_counts(refused), None, "{refused:?}");
        }
        let seventeen: Vec<_> = (1..=17).map(|n| n.to_string()).collect();
        assert_eq!(parse_counts(&seventeen.join(",")), None);
        assert!(parse_counts(&seventeen[..16].join(",")).is_some());
    }

    /// The comparison names the first differing run; no source program
    /// reaches it while the CPU engine meets its oracle, so the fault is
    /// injected here (`INVARIANCE_MISMATCH` is an invariant code).
    #[test]
    fn machine_diagnostics_invariance_mismatch_names_the_first_differing_run() {
        let runs = [
            "Ok(I32(6)); fuel 9",
            "Ok(I32(7)); fuel 9",
            "Ok(I32(8)); fuel 9",
        ]
        .map(String::from);
        assert_eq!(first_difference("Ok(I32(6)); fuel 9", &runs), Some(1));
        assert_eq!(first_difference("Ok(I32(6)); fuel 9", &runs[..1]), None);
        let fuel_only = ["Ok(I32(6)); fuel 10".to_owned()];
        assert_eq!(first_difference("Ok(I32(6)); fuel 9", &fuel_only), Some(0));
        let error = mismatch(7, &runs[1], &runs[0]);
        let document = error.to_diagnostics_json("fixture.inv", b"");
        assert_eq!(document["diagnostics"][0]["code"], "INVARIANCE_MISMATCH");
        let schema: Value =
            serde_json::from_str(include_str!("../schema/invar-diagnostics-v3.schema.json"))
                .unwrap();
        assert!(jsonschema::is_valid(&schema, &document));
    }
}
