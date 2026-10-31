#![forbid(unsafe_code)]
//! P7 (#54): the GNET-05 acceptance corpus (`bench/general-net-v01/gnet05`)
//! through `invar run`. Every program runs on every recorded input (the deep
//! inputs in Release only, as `tests/gnet05_corpus/corpus.rs` does) through
//! the CPU engine, which now reaches the checked normal form, and must equal
//! the reference evaluator (`eval::run`) in value, first trap and limit at the
//! input's declared step budget. Route counts are reported, not tuned: no
//! program here is routed by name, and none is held to a speed.

use invar::{
    engine::{self, Engine},
    eval::{self, EvalError, Value},
};
use serde_json::Value as Json;
use std::{collections::BTreeMap, fs, path::Path};

fn root() -> &'static Path {
    Path::new(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/bench/general-net-v01/gnet05"
    ))
}

fn literal(param: &str, value: i64) -> String {
    match param {
        "u32" => format!("{}u32", value as u32),
        "i32" if value < 0 => format!("(0 - {})", -value),
        "i32" => value.to_string(),
        other => panic!("parameter type {other}"),
    }
}

/// The program with its recorded `main` replaced by one calling `run(a, b)`.
fn with_input(source: &str, param: &str, (a, b): (i64, i64)) -> String {
    let start = source.rfind("fn main()").expect("recorded main");
    format!(
        "{}fn main() -> i32 {{ return run({}, {}); }}\n",
        &source[..start],
        literal(param, a),
        literal(param, b)
    )
}

fn checked(source: &str) -> invar::check::CheckedProgram {
    invar::front::parse(source.as_bytes())
        .and_then(|program| invar::check::check(program, true))
        .unwrap_or_else(|e| panic!("{source}: {e:?}"))
}

fn same(left: &Result<Value, EvalError>, right: &Result<Value, EvalError>) -> bool {
    match (left, right) {
        (Ok(a), Ok(b)) => a == b,
        (Err(EvalError::Tool { code: a, .. }), Err(EvalError::Tool { code: b, .. })) => a == b,
        (Err(a), Err(b)) => a == b,
        _ => false,
    }
}

#[test]
fn gnet05_corpus_through_run_equals_the_reference_evaluator() {
    // The reference evaluator recurses up to its 64-frame limit; Debug frames
    // need more than a test thread's default stack (as in
    // `tests/support/gnet_device.rs`). `invar run` itself runs on the main
    // thread, where every one of these inputs completes.
    std::thread::Builder::new()
        .stack_size(64 * 1024 * 1024)
        .spawn(corpus_through_run)
        .unwrap()
        .join()
        .unwrap();
}

fn corpus_through_run() {
    let manifest: Json =
        serde_json::from_str(&fs::read_to_string(root().join("manifest.json")).unwrap()).unwrap();
    let programs = manifest["programs"].as_array().unwrap();
    assert_eq!(programs.len(), 72);
    let mut routes = BTreeMap::<String, usize>::new();
    let mut cases = 0;
    for p in programs {
        let id = p["id"].as_str().unwrap();
        let param = p["param"].as_str().unwrap();
        let source = fs::read_to_string(root().join(p["file"].as_str().unwrap())).unwrap();
        let inputs = p["inputs"].as_array().unwrap();
        let budgets = p["budgets"].as_array().unwrap();
        assert_eq!(inputs.len(), budgets.len(), "{id}");
        let deep = p["deep_input"].as_u64().map(|i| i as usize);
        for (index, (input, budget)) in inputs.iter().zip(budgets).enumerate() {
            let fuel = budget.as_u64().unwrap();
            // The manifest's deep input runs in Release only, as in
            // `corpus.rs` (its budget may be any size above the
            // 100,000-step default, including 1,000,000).
            if deep == Some(index) && cfg!(debug_assertions) {
                continue;
            }
            let pair = (input[0].as_i64().unwrap(), input[1].as_i64().unwrap());
            let source = with_input(&source, param, pair);
            let program = checked(&source);
            let expected = eval::run(&program, fuel).map(|outcome| outcome.value);
            let mir = invar::mir::lower(&program).unwrap();
            let (got, route) = match engine::run_mir(
                source.as_bytes(),
                &mir,
                Engine::CpuParallel,
                fuel,
                2,
                None,
            ) {
                Ok(receipt) => (
                    Ok(Value::I32(receipt["result"].as_i64().unwrap() as i32)),
                    receipt.get("normal_form").cloned(),
                ),
                Err(error) => (Err(error), None),
            };
            assert!(
                same(&got, &expected),
                "{id} {pair:?} fuel {fuel}: run {got:?} reference {expected:?}"
            );
            let route = match &route {
                Some(document) if document["status"] == "routed" => "routed".to_owned(),
                Some(document) => document["reason"].as_str().unwrap().to_owned(),
                None if got.is_err() => "error-before-receipt".to_owned(),
                None => "no-certified-law".to_owned(),
            };
            *routes.entry(route).or_default() += 1;
            cases += 1;
        }
    }
    eprintln!("gnet05 through run: {cases} cases, routes {routes:?}");
    // 334 recorded inputs plus 30 deep inputs (`bench/general-net-v01/gnet05`).
    assert_eq!(cases, if cfg!(debug_assertions) { 334 } else { 364 });
    // The route is exercised, not only its fallbacks.
    assert!(routes.get("routed").is_some_and(|&n| n > 0), "{routes:?}");
}
