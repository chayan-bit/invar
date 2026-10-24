//! GNET-02 whole-corpus agreement: every checked runnable source fixture and
//! example runs through the reference evaluator, the MIR evaluator and the
//! bounded CPU net oracle, at full fuel and one step short of its own use.
//! The only admitted non-agreements are classified by program property, never
//! by file name: declared foreign host boundaries, programs that compute
//! binary64, and the documented structural net budget.
#![forbid(unsafe_code)]

use crate::agree;
use crate::corpus;

use agree::same;
use corpus::runnable_corpus;
use general_net::{Error, Limits, Net, oracle};
use invar::{eval, general_net, mir, mir_eval};
use std::collections::BTreeMap;

const FUEL: u64 = 100_000;

fn observe(net: &Net, fuel: u64) -> oracle::Observation {
    oracle::observe(
        net,
        &net.verify().unwrap(),
        oracle::Limits {
            fuel,
            ..oracle::Limits::default()
        },
    )
    .unwrap()
}

#[test]
fn issue_gnet_02_corpus_every_runnable_source_agrees_with_both_reference_engines() {
    let corpus = runnable_corpus();
    let mut tally: BTreeMap<&str, usize> = BTreeMap::new();
    for (name, checked) in &corpus {
        let program = mir::lower(checked).unwrap();
        let reference = eval::observe(checked, FUEL);
        same(&mir_eval::observe(&program, FUEL), &reference, name);
        let net = match Net::lower(checked, Limits::default()) {
            Ok(net) => net,
            Err(error) => {
                // Structural budgets are explicit refusals, never a semantic result.
                assert_eq!(error, Error::Limit, "{name}");
                *tally.entry("structural_net_limit").or_default() += 1;
                continue;
            }
        };
        let observed = observe(&net, FUEL);
        if program.functions.iter().any(|f| f.foreign_oracle.is_some()) {
            // Declared foreign calls are host boundaries: refused before any demand.
            assert_eq!(observed.failure, Some(oracle::ToolFailure::Unsupported));
            assert_eq!(observed.semantic.steps_used, 0, "{name}");
            assert!(observed.events.is_empty(), "{name}");
            *tally.entry("foreign_host_boundary").or_default() += 1;
            continue;
        }
        if !invar::math::f64_sites(&program).is_empty() {
            // The net contract has no binary64 words (strict-f64-v1, v0.2):
            // refused before any demand, like a foreign host boundary.
            assert_eq!(observed.failure, Some(oracle::ToolFailure::Unsupported));
            assert_eq!(observed.semantic.steps_used, 0, "{name}");
            assert!(observed.events.is_empty(), "{name}");
            *tally.entry("binary64_outside_net").or_default() += 1;
            continue;
        }
        assert_eq!(observed.failure, None, "{name}: {:?}", observed.semantic);
        same(&observed.semantic, &reference, name);
        // Exhaustion one step short is the exact source-fuel limit, with only
        // completed cleanup observable, in all three engines.
        if reference.steps_used > 1
            && !matches!(
                reference.result,
                Err(eval::EvalError::EvaluationLimit { .. })
            )
        {
            let short = reference.steps_used - 1;
            let expected = eval::observe(checked, short);
            assert!(
                matches!(
                    expected.result,
                    Err(eval::EvalError::EvaluationLimit { .. })
                ),
                "{name}"
            );
            same(&mir_eval::observe(&program, short), &expected, name);
            same(&observe(&net, short).semantic, &expected, name);
        }
        let outcome = match &reference.result {
            Ok(_) => "agree_value",
            Err(eval::EvalError::Trap { .. }) => "agree_trap",
            Err(eval::EvalError::EvaluationLimit { .. }) => "agree_fuel_limit",
            Err(_) => "agree_tool_result",
        };
        *tally.entry(outcome).or_default() += 1;
    }
    println!("gnet_02_corpus runnable={} {tally:?}", corpus.len());
    let agreed: usize = tally
        .iter()
        .filter(|(key, _)| key.starts_with("agree_"))
        .map(|(_, count)| count)
        .sum();
    assert!(agreed >= 300, "{tally:?}");
    assert!(tally["agree_trap"] > 0 && tally["agree_value"] > 0);
}
