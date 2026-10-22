//! Reached ordinary-source bounded scans on the local CPU.
//! Private completion tickets preserve original allocation, fuel and cleanup.
use super::{Contract, Error, Family, Prepared};
use crate::{
    bounded_summary::Budget,
    eval::Observation,
    mir_eval::{self, LiveLoop, LoopOverride},
    typed_net::{BoundedObservation, Plan},
};
use std::rc::Rc;

fn plan_failure(message: &'static str) -> crate::eval::EvalError {
    crate::eval::EvalError::Tool {
        code: "SUMMARY_PLAN",
        message: message.into(),
    }
}

pub(crate) struct Completion {
    pub(super) contract: Rc<Contract>,
    pub(super) prepared: Prepared,
    pub(super) plan: Plan,
    pub(super) candidate: BoundedObservation,
    pub(super) budget: Budget,
    pub(super) steps: u32,
}
impl Completion {
    pub(crate) fn parts(
        &self,
    ) -> (
        &Contract,
        &Prepared,
        &Plan,
        &BoundedObservation,
        Budget,
        u32,
    ) {
        (
            &self.contract,
            &self.prepared,
            &self.plan,
            &self.candidate,
            self.budget,
            self.steps,
        )
    }
}
#[cfg(test)]
impl Completion {
    /// Test-only summary receipt for cross-family ambiguity tests. It is built
    /// by the real preparation, typed extraction, binding and bounded scan of a
    /// two-state table source, and is never published.
    pub(crate) fn for_ambiguity_test() -> Self {
        const TABLE: &str = "fn run(coeff: read Array<u32>, seed: u32, count: u32) -> u32 { var state = seed; for i in 0u32..count { state = coeff[i * 2u32 + state]; } return state; }";
        let checked =
            crate::check::check(crate::front::parse(TABLE.as_bytes()).unwrap(), false).unwrap();
        let contract = Rc::new(Contract::from_checked(&checked, "run").unwrap());
        let budget = Budget {
            max_steps: 1024,
            max_bytes: 1_000_000,
            max_composition_ops: 1_000_000,
        };
        let (coefficients, seed, steps) = ([1, 0, 0, 1], [1], 2);
        let prepared = contract
            .prepare(&coefficients, &seed, steps, budget)
            .unwrap();
        let plan = prepared.typed_plan().unwrap();
        contract
            .verify_plan(&prepared, &plan, &coefficients, &seed, steps, budget)
            .unwrap();
        let candidate = plan
            .run_bounded_scan(&[&coefficients, &seed], &[seed[0], steps], budget)
            .unwrap();
        Self {
            contract,
            prepared,
            plan,
            candidate,
            budget,
            steps,
        }
    }
}
pub struct Run {
    pub observation: Observation,
    pub completed: usize,
    pub refusals: Vec<&'static str>,
}

/// Execute checked source, replacing only actually reached verified scan loops.
/// Budget bytes/work include both immutable preparation and typed execution;
/// each receives half the private logical budget. No GPU/performance claim.
pub fn run_source(source: &[u8], fuel: u64, budget: Budget) -> Result<Run, Error> {
    let checked = crate::front::parse(source)
        .and_then(|p| crate::check::check(p, true))
        .map_err(Error::Source)?;
    let program =
        crate::mir::lower(&checked).map_err(|e| Error::Net(crate::general_net::Error::Mir(e)))?;
    let mut contracts = Vec::new();
    contracts
        .try_reserve_exact(checked.program.functions.len())
        .map_err(|_| Error::Allocation)?;
    let mut refusals = Vec::new();
    for function in &checked.program.functions {
        match Contract::from_checked(&checked, &function.name) {
            Ok(contract) => contracts.push(Rc::new(contract)),
            Err(Error::Unrecognized) => {}
            // A near-miss scan form stays serial and records why.
            Err(error) => refusals.push(error.reason().ok_or(error)?),
        }
    }
    let staging = Budget {
        max_bytes: budget.max_bytes / 2,
        max_composition_ops: budget.max_composition_ops / 2,
        ..budget
    };
    let mut provider = |live: &LiveLoop<'_>| {
        let Some(contract) = contracts
            .iter()
            .find(|c| c.function() == live.function && c.loop_span() == live.span)
        else {
            return Ok(None);
        };
        let Some(&steps) = live.scalars.get(contract.count()) else {
            return Ok(None);
        };
        if steps as usize != live.iterations
            || contract
                .loop_fuel(steps)
                .is_none_or(|ticks| ticks > live.fuel_remaining)
        {
            refusals.push("summary_source_fuel");
            return Ok(None);
        }
        let Some(coefficients) = live.array(contract.coefficients()) else {
            return Ok(None);
        };
        let seed_array = live.array(contract.seed());
        let mut scalar_seed = [0; 1];
        let seed = match contract.family() {
            Family::Table(_) => {
                let Some(&seed) = live.scalars.get(contract.seed()) else {
                    return Ok(None);
                };
                scalar_seed[0] = seed;
                scalar_seed.as_slice()
            }
            Family::Matrix(_) => {
                if live.same_object(contract.coefficients(), contract.seed()) {
                    refusals.push("summary_alias");
                    return Ok(None);
                }
                let Some(seed) = seed_array.as_ref() else {
                    return Ok(None);
                };
                seed.as_slice()
            }
        };
        if contract.state_names().len() != seed.len()
            || contract
                .state_names()
                .iter()
                .zip(seed)
                .any(|(name, n)| live.scalars.get(name) != Some(n))
        {
            refusals.push("summary_seed");
            return Ok(None);
        }
        // Input, seed and source-allocated output loans remain held throughout
        // preparation and execution. No external callback receives these owners.
        let output = contract.output().and_then(|name| live.array(name));
        if let Some(name) = contract.output()
            && (live.same_object(name, contract.coefficients())
                || live.same_object(name, contract.seed())
                || output.as_ref().is_none_or(|v| {
                    (steps as usize).checked_mul(contract.family().state_width()) != Some(v.len())
                }))
        {
            refusals.push("summary_output");
            return Ok(None);
        }
        let prepared = match contract.prepare(&coefficients, seed, steps, staging) {
            Ok(p) => p,
            Err(_) => {
                refusals.push("summary_admission");
                return Ok(None);
            }
        };
        let plan = prepared
            .typed_plan()
            .map_err(|_| plan_failure("verified source scan extraction failed"))?;
        contract
            .verify_plan(&prepared, &plan, &coefficients, seed, steps, staging)
            .map_err(|_| plan_failure("private source scan binding failed"))?;
        let scalar_inputs = [scalar_seed[0], steps];
        let arrays: [&[u32]; 2] = [&coefficients, seed];
        let candidate = match plan.run_bounded_scan(&arrays, &scalar_inputs, staging) {
            Ok(c) => c,
            Err(_) => {
                refusals.push("summary_candidate");
                return Ok(None);
            }
        };
        Ok(Some(LoopOverride {
            histogram: None,
            scatter: None,
            gather: None,
            iterations: live.iterations,
            input_lengths: Vec::new(),
            scalar: None,
            array: None,
            tensor: None,
            transpose: None,
            stencil: None,
            summary: Some(Completion {
                contract: contract.clone(),
                prepared,
                plan,
                candidate,
                budget: staging,
                steps,
            }),
            region: None,
        }))
    };
    let (observation, completed) =
        mir_eval::observe_with_scheduler(&program, fuel, Some(&mut provider));
    Ok(Run {
        observation,
        completed,
        refusals,
    })
}

#[cfg(test)]
mod diagnostic_tests {
    #[test]
    fn issue_ai_01_machine_diagnostics_summary_plan_fault() {
        for message in [
            "verified source scan extraction failed",
            "private source scan binding failed",
        ] {
            // Exercise the same error conversion used after an internal plan
            // verifier failure, without granting an invalid plan admission.
            let error = super::plan_failure(message);
            let document = error.to_diagnostics_json("summary.inv", b"");
            assert_eq!(document["diagnostics"][0]["code"], "SUMMARY_PLAN");
            assert_eq!(
                document["diagnostics"][0]["suggested_fixes"],
                serde_json::json!([])
            );
        }
    }
}
