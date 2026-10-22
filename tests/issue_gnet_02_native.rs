//! GNET-02 compiled native CPU lowering against the CPU net oracle and the
//! reference evaluator. The native program is the verified net's
//! `compiled_mir` boundary output, compiled through LLVM 21 and the native host.
//! Programs the native backend refuses are counted as explicit refusals; an
//! admitted program must agree exactly (status, value, trap category and span,
//! tool code, source steps and cleanup trace).
#![cfg(feature = "native")]
#![forbid(unsafe_code)]

#[path = "support/gnet_corpus.rs"]
mod corpus;
#[path = "support/gnet_generator.rs"]
mod generator;

use general_net::{Limits, Net, oracle};
use invar::{
    check,
    eval::{self, EvalError, TrapCategory, Value},
    front, general_net, native,
};
use serde_json::{Value as Json, json};
use std::{collections::BTreeMap, sync::Mutex};

const FUEL: u64 = 100_000;
const WORKERS: usize = 6;

/// The native host document fields implied by an oracle observation.
fn expected(observation: &eval::Observation) -> (i32, Json) {
    let (status, fields) = match &observation.result {
        Ok(Value::I32(value)) => (
            0,
            json!({"status":"ok","result":{"type":"i32","value":value}}),
        ),
        Err(EvalError::Trap { category, span }) => {
            let category = match category {
                TrapCategory::IntegerOverflow => "INTEGER_OVERFLOW",
                TrapCategory::DivisionByZero => "DIVISION_BY_ZERO",
                TrapCategory::IndexOutOfBounds => "INDEX_OUT_OF_BOUNDS",
                TrapCategory::UnitConversionInexact => "UNIT_CONVERSION_INEXACT",
            };
            (
                3,
                json!({"status":"trap","trap":{"category":category,"span":{"start":span.start,"end":span.end}}}),
            )
        }
        Err(EvalError::EvaluationLimit { fuel_limit, .. }) => (
            4,
            json!({"status":"evaluation_limit","fuel_limit":fuel_limit}),
        ),
        Err(EvalError::Tool { code, .. }) => (2, json!({"status":"tool_error","tool_code":code})),
        other => panic!("unexpected main result: {other:?}"),
    };
    (status, fields)
}

enum Outcome {
    Agree,
    Refused(String),
}

fn check_one(source_name: &str, checked: &check::CheckedProgram) -> Option<Outcome> {
    let net = Net::lower(checked, Limits::default()).ok()?;
    let proof = net.verify().unwrap();
    let observed = oracle::observe(
        &net,
        &proof,
        oracle::Limits {
            fuel: FUEL,
            ..oracle::Limits::default()
        },
    )
    .unwrap();
    // Host boundaries and oracle refusals are not native validation inputs.
    if observed.failure.is_some() {
        return None;
    }
    let reference = eval::observe(checked, FUEL);
    match (&observed.semantic.result, &reference.result) {
        // Engines word tool messages differently; the diagnostic code is shared.
        (Err(EvalError::Tool { code: a, .. }), Err(EvalError::Tool { code: b, .. })) => {
            assert_eq!(a, b, "{source_name}")
        }
        (a, b) => assert_eq!(a, b, "{source_name}"),
    }
    let program = net.compiled_mir(&proof).unwrap();
    let (status, document) = match native::run_program(&program, FUEL) {
        Ok(run) => run,
        Err(reason) => return Some(Outcome::Refused(reason)),
    };
    // A native-only resource bound (for example the native cleanup-trace cap)
    // is an explicit refusal of that program, never a semantic disagreement.
    if let Some(code) = document["tool_code"].as_str()
        && code.starts_with("NATIVE_")
    {
        return Some(Outcome::Refused(format!("native tool limit {code}")));
    }
    let (expected_status, fields) = expected(&observed.semantic);
    assert_eq!(status, expected_status, "{source_name}: {document}");
    for key in ["status", "result", "trap", "fuel_limit", "tool_code"] {
        if !fields[key].is_null() {
            assert_eq!(document[key], fields[key], "{key}: {source_name}");
        }
    }
    assert_eq!(
        document["steps_used"], observed.semantic.steps_used,
        "{source_name}"
    );
    let cleanup: Vec<Json> = observed
        .semantic
        .cleanup_trace
        .iter()
        .map(|event| json!({"name":event.name,"length":event.length}))
        .collect();
    assert_eq!(document["cleanup_trace"], json!(cleanup), "{source_name}");
    Some(Outcome::Agree)
}

fn run_all(cases: Vec<(String, check::CheckedProgram)>) -> (usize, BTreeMap<String, usize>) {
    let next = Mutex::new(cases.into_iter());
    let agreed = Mutex::new(0usize);
    let refused = Mutex::new(BTreeMap::new());
    std::thread::scope(|scope| {
        for _ in 0..WORKERS {
            scope.spawn(|| {
                loop {
                    let Some((name, checked)) = next.lock().unwrap().next() else {
                        break;
                    };
                    match check_one(&name, &checked) {
                        Some(Outcome::Agree) => *agreed.lock().unwrap() += 1,
                        Some(Outcome::Refused(reason)) => {
                            *refused.lock().unwrap().entry(reason).or_insert(0) += 1
                        }
                        None => {}
                    }
                }
            });
        }
    });
    (agreed.into_inner().unwrap(), refused.into_inner().unwrap())
}

#[test]
fn issue_gnet_02_native_cpu_matches_oracle_on_the_runnable_corpus() {
    let (agreed, refused) = run_all(corpus::runnable_corpus());
    println!("gnet_02_native_corpus agreed={agreed} refused={refused:?}");
    assert!(agreed >= 320, "agreed={agreed} refused={refused:?}");
}

#[test]
fn issue_gnet_02_native_cpu_matches_oracle_on_generated_compositions() {
    let mut rng = generator::Rng(generator::SEED);
    let cases = (0..generator::PROGRAMS)
        .map(|index| {
            let (source, _) = generator::program(&mut rng);
            let checked = check::check(front::parse(source.as_bytes()).unwrap(), true).unwrap();
            (format!("generated-{index}: {source}"), checked)
        })
        .collect();
    let (agreed, refused) = run_all(cases);
    println!("gnet_02_native_generated agreed={agreed} refused={refused:?}");
    assert_eq!(agreed, generator::PROGRAMS, "refused={refused:?}");
}
