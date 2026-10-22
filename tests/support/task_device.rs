//! P4b device differential: the C1 family and generated fork-joins run on a
//! device runner and are compared with the CPU net oracle.
#![allow(dead_code)]

use std::collections::BTreeMap;

use invar::{
    eval::{EvalError, Value},
    general_net::task::{Plan, Schedule, host::Outcome},
};

use super::model::{Admitted, budget_exhausted, observe, same};

/// Device tally.
#[derive(Default, Debug)]
pub struct Tally {
    pub cases: u64,
    pub values: u64,
    pub traps: u64,
    pub depth_limits: u64,
    pub skipped: u64,
    pub budget: u64,
    pub multi_submission: u64,
    /// Cases whose call ran at least one subtree serially (cascade).
    pub serial: u64,
    /// Cases per runner path.
    pub paths: BTreeMap<&'static str, u64>,
    pub mismatches: Vec<String>,
}

/// Run every case on a device call and compare with the oracle.
pub fn device_differential(
    name: &str,
    admitted: &Admitted,
    cases: &[Vec<i32>],
    call: &mut dyn FnMut(&[Value]) -> Result<Outcome, String>,
    tally: &mut Tally,
) {
    for case in cases {
        let Some(expected) = observe(admitted, case) else {
            tally.skipped += 1;
            continue;
        };
        let args: Vec<Value> = case.iter().map(|x| Value::I32(*x)).collect();
        let outcome = call(&args).unwrap_or_else(|e| panic!("{name} {case:?}: device {e}"));
        if budget_exhausted(&outcome.result) {
            tally.budget += 1;
            continue;
        }
        tally.cases += 1;
        match &expected {
            Ok(_) => tally.values += 1,
            Err(EvalError::Trap { .. }) => tally.traps += 1,
            Err(_) => tally.depth_limits += 1,
        }
        if outcome.submissions > 1 {
            tally.multi_submission += 1;
        }
        if outcome.path.ends_with("cascade") && outcome.tasks > 0 {
            tally.serial += 1;
        }
        *tally.paths.entry(outcome.path).or_default() += 1;
        if !same(&outcome.result, &expected) {
            tally.mismatches.push(format!(
                "{name} {case:?}: device {:?}, oracle {expected:?}",
                outcome.result
            ));
        }
    }
}

/// Plans exercised on a device: the profile's plan and a narrow one whose
/// small arena forces many joins and arena fallbacks. Staged: a small width
/// and one epoch per submission, so most calls span several submissions.
/// Cascade: the width past the arena, so concurrent forks fill the arena and
/// fall back to serial runs while every lane contends for a few nodes.
pub fn device_plans(wide: Plan) -> [(&'static str, Plan); 2] {
    let wide = Plan {
        budget: DEVICE_BUDGET,
        ..wide
    };
    let narrow = Plan {
        width: match wide.schedule {
            Schedule::Staged => 32,
            Schedule::Cascade => 1 << 16,
        },
        arena: 256,
        epochs: 1,
        ..wide.clone()
    };
    [("wide", wide), ("narrow", narrow)]
}

/// Lane step budget of the device tests (bounded kernel time).
pub const DEVICE_BUDGET: u32 = 1 << 22;

pub fn print(label: &str, tally: &Tally) {
    eprintln!(
        "{label}: {} cases ({} values, {} traps, {} depth limits, {} multi-submission, {} cascade serial), {} skipped (oracle limit), {} skipped (lane budget); paths {:?}",
        tally.cases,
        tally.values,
        tally.traps,
        tally.depth_limits,
        tally.multi_submission,
        tally.serial,
        tally.skipped,
        tally.budget,
        tally.paths
    );
}
