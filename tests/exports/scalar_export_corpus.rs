//! Scalar source corpus preparation. Named-call/native-export gates require the
//! core owner's frozen API; appended `main` runs do not establish those gates.
//! Issue #14 (L1-06).
use invar::{check, eval, front, mir, mir_eval, ownership};
use serde_json::{Value, json};

#[path = "../../test-support/scalar-exports/inventory.rs"]
mod inventory;

#[path = "../../test-support/scalar-exports/facade_corpus.rs"]
mod facade_corpus;

fn vectors() -> Value {
    serde_json::from_str(include_str!(
        "../../test-support/scalar-exports/scalar-vectors-v1.json"
    ))
    .unwrap()
}

fn observed(observation: &eval::Observation) -> Value {
    match &observation.result {
        Ok(eval::Value::I32(value)) => json!({"status":"ok","value":value}),
        Err(error) => {
            let document = error.to_json("scalar_exports.inv");
            match error {
                eval::EvalError::Trap { .. } => {
                    json!({"status":"trap","trap":document["trap"]})
                }
                eval::EvalError::EvaluationLimit {
                    fuel_limit,
                    steps_used,
                } => {
                    json!({"status":"evaluation_limit","fuel_limit":fuel_limit,"steps_used":steps_used})
                }
                other => panic!("unexpected preparation failure: {other:?}"),
            }
        }
        other => panic!("unexpected reference main result: {other:?}"),
    }
}

fn expected_main(case: &Value) -> Value {
    if case["expected"]["status"] == "ok" {
        json!({"status":"ok","value":case["reference_main_expected_value"]})
    } else {
        case["expected"].clone()
    }
}

#[test]
fn scalar_export_corpus_reference_vectors_and_mir_preparation() {
    let document = vectors();
    let cases = document["cases"].as_array().unwrap();
    assert_eq!(cases.len(), 59);
    let mut counts = std::collections::BTreeMap::new();
    for case in cases {
        let source = case["reference_main_source"].as_str().unwrap();
        let checked = check::check(front::parse(source.as_bytes()).unwrap(), true).unwrap();
        ownership::check(&checked).unwrap();
        let program = mir::lower(&checked).unwrap();
        let fuel = case["fuel"].as_u64().unwrap();
        let reference = eval::observe(&checked, fuel);
        assert_eq!(observed(&reference), expected_main(case), "{}", case["id"]);
        let serial = mir_eval::observe(&program, fuel);
        assert_eq!(serial.result, reference.result, "{}", case["id"]);
        assert_eq!(serial.steps_used, reference.steps_used, "{}", case["id"]);
        assert_eq!(serial.cleanup_trace, reference.cleanup_trace);
        if case["expected"]["status"] == "trap" {
            let span = &case["expected"]["trap"]["span"];
            let start = span["start"].as_u64().unwrap() as usize;
            let end = span["end"].as_u64().unwrap() as usize;
            assert!(matches!(&source[start..end], "+" | "-" | "*" | "/" | "%"));
            assert!(start > source.find('é').unwrap());
            assert!(reference.cleanup_trace.is_empty());
        }
        *counts
            .entry(case["expected"]["status"].as_str().unwrap())
            .or_insert(0) += 1;
    }
    eprintln!("source-main preparation: 59 vectors, {counts:?}; compiled exports not executed");
}

#[test]
fn scalar_export_corpus_pinned_source_fuel_preparation() {
    // Independent source accounting: return + literal = two steps. This enters
    // main directly and pins the convention the future named-call API must use.
    let source = b"fn main() -> i32 { return 42; }";
    let checked = check::check(front::parse(source).unwrap(), true).unwrap();
    for fuel in [1, 2, 3] {
        let run = eval::observe(&checked, fuel);
        if fuel == 1 {
            assert_eq!(
                run.result,
                Err(eval::EvalError::EvaluationLimit {
                    fuel_limit: 1,
                    steps_used: 1,
                })
            );
        } else {
            assert_eq!(run.result, Ok(eval::Value::I32(42)));
            assert_eq!(run.steps_used, 2);
        }
    }
    // main: return, call, two literals; callee: return, binary, two names = 8.
    let source =
        b"fn add(x: i32, y: i32) -> i32 { return x + y; } fn main() -> i32 { return add(20, 22); }";
    let checked = check::check(front::parse(source).unwrap(), true).unwrap();
    for fuel in [7, 8, 9] {
        let run = eval::observe(&checked, fuel);
        if fuel == 7 {
            assert_eq!(
                run.result,
                Err(eval::EvalError::EvaluationLimit {
                    fuel_limit: 7,
                    steps_used: 7,
                })
            );
        } else {
            assert_eq!(run.result, Ok(eval::Value::I32(42)));
            assert_eq!(run.steps_used, 8);
        }
    }
}

#[test]
fn scalar_export_corpus_actual_facade_admission_and_reference_replay() {
    let corpus = inventory::enumerate();
    assert!(corpus["summary"]["input_entries"].as_u64().unwrap() > 751);
    assert!(corpus["summary"]["checked_functions"].as_u64().unwrap() > 0);
    assert!(corpus["summary"]["scalar_mir_candidates"].as_u64().unwrap() > 0);
    assert_eq!(
        corpus["compiled_export_admission"],
        "separate_actual_facade_qualification"
    );
    eprintln!("preparation inventory: {}", corpus["summary"]);
    if let Some(path) = std::env::var_os("INVAR_SCALAR_INVENTORY_OUTPUT") {
        std::fs::write(path, serde_json::to_vec_pretty(&corpus).unwrap()).unwrap();
    }
    let report = facade_corpus::qualify(corpus);
    assert!(report["summary"]["accepted_functions"].as_u64().unwrap() > 0);
    assert!(report["summary"]["excluded_functions"].as_u64().unwrap() > 0);
    eprintln!("actual export admission: {}", report["summary"]);
    if let Some(path) = std::env::var_os("INVAR_SCALAR_EXPORT_ADMISSION_OUTPUT") {
        std::fs::write(path, serde_json::to_vec_pretty(&report).unwrap()).unwrap();
    }
    let replay = facade_corpus::replay(&report, false);
    eprintln!("actual reference exports: {}", replay["summary"]);
    if let Some(path) = std::env::var_os("INVAR_SCALAR_EXPORT_REPLAY_OUTPUT") {
        std::fs::write(path, serde_json::to_vec_pretty(&replay).unwrap()).unwrap();
    }
    facade_corpus::run_vectors(vectors(), false);
}

#[test]
fn scalar_export_corpus_facade_exact_source_fuel() {
    use invar::exports::{ExportEngine, ExportError, ScalarValue, compile_exports};
    let source =
        b"fn constant() -> i32 { return 42; } fn add(x: i32, y: i32) -> i32 { return x + y; }";
    #[cfg(not(feature = "native"))]
    let engines = [ExportEngine::Reference];
    #[cfg(feature = "native")]
    let engines = [ExportEngine::Reference, ExportEngine::NativeSerial];
    for engine in engines {
        let mut artifact =
            compile_exports(source, "exact-fuel.inv", &["constant", "add"], engine).unwrap();
        for (name, args, steps) in [
            ("constant", vec![], 2),
            ("add", vec![ScalarValue::I32(20), ScalarValue::I32(22)], 4),
        ] {
            let error = artifact.invoke(name, &args, steps - 1).unwrap_err();
            assert!(
                matches!(error,ExportError::Evaluation(eval::EvalError::EvaluationLimit {fuel_limit,steps_used}) if fuel_limit == steps - 1 && steps_used == steps - 1)
            );
            assert_eq!(
                artifact.invoke(name, &args, steps).unwrap(),
                ScalarValue::I32(42)
            );
            assert_eq!(
                artifact.invoke(name, &args, steps + 1).unwrap(),
                ScalarValue::I32(42)
            );
        }
    }
}

#[cfg(feature = "native")]
#[test]
fn scalar_export_corpus_actual_native_facade_full_subset_replay() {
    let report = facade_corpus::qualify(inventory::enumerate());
    let replay = facade_corpus::replay(&report, true);
    eprintln!(
        "actual reference/native export equivalence: {}",
        replay["summary"]
    );
    if let Some(path) = std::env::var_os("INVAR_SCALAR_EXPORT_NATIVE_REPLAY_OUTPUT") {
        std::fs::write(path, serde_json::to_vec_pretty(&replay).unwrap()).unwrap();
    }
    facade_corpus::run_vectors(vectors(), true);
}

#[test]
fn scalar_export_corpus_screen_checks_transitive_bodies() {
    let report = inventory::classify("transitive".to_string(),
        "fn scalar() -> i32 { return helper(); } fn helper() -> i32 { let xs: Array<i32> = [1]; return xs[0u32]; } fn plain(x: u32) -> u32 { return wrapping_add(x, 1u32); }".to_string());
    assert_eq!(report["status"], "checked");
    let functions = report["functions"].as_array().unwrap();
    let scalar = functions.iter().find(|f| f["name"] == "scalar").unwrap();
    assert_eq!(scalar["status"], "excluded_by_preparation_screen");
    assert!(
        scalar["reasons"]
            .as_array()
            .unwrap()
            .iter()
            .any(|r| r == "reachable_nonscalar_value:helper")
    );
    let plain = functions.iter().find(|f| f["name"] == "plain").unwrap();
    assert_eq!(plain["status"], "scalar_mir_candidate");
    let library = inventory::classify(
        "no-main".to_string(),
        "fn f(x: i32) -> i32 { return x; }".to_string(),
    );
    assert_eq!(library["functions"][0]["status"], "scalar_mir_candidate");
}

#[cfg(feature = "native")]
#[test]
fn scalar_export_corpus_native_source_main_preparation() {
    for case in vectors()["cases"].as_array().unwrap() {
        let source = case["reference_main_source"].as_str().unwrap();
        let checked = check::check(front::parse(source.as_bytes()).unwrap(), true).unwrap();
        ownership::check(&checked).unwrap();
        let program = mir::lower(&checked).unwrap();
        let fuel = case["fuel"].as_u64().unwrap();
        let reference = eval::observe(&checked, fuel);
        let (_, native) = invar::native::run_program(&program, fuel).unwrap();
        let actual = match native["status"].as_str().unwrap() {
            "ok" => json!({"status":"ok","value":native["result"]["value"]}),
            "trap" => json!({"status":"trap","trap":native["trap"]}),
            "evaluation_limit" => {
                json!({"status":"evaluation_limit","fuel_limit":native["fuel_limit"],"steps_used":native["steps_used"]})
            }
            other => panic!("{}: unexpected native status {other}", case["id"]),
        };
        assert_eq!(actual, observed(&reference), "{}", case["id"]);
        assert_eq!(native["steps_used"], reference.steps_used, "{}", case["id"]);
    }
    eprintln!(
        "59 real native source-main preparations; compiled named exports remain a separate gate"
    );
}
