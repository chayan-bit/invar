#![cfg(feature = "native")]

use invar::{check, eval, front, mir, native};
use serde_json::json;

#[test]
#[ignore = "requires INVAR_M3_05_CORPUS_DIR with the exported 252-program M2 corpus"]
fn issue_m3_05_native_full_m2_probe_corpus() {
    let directory = std::env::var_os("INVAR_M3_05_CORPUS_DIR")
        .expect("INVAR_M3_05_CORPUS_DIR must name the exported M2 probe corpus");
    let mut files = std::fs::read_dir(directory)
        .unwrap()
        .map(|entry| entry.unwrap().path())
        .filter(|path| {
            path.extension()
                .is_some_and(|extension| extension == "invar")
        })
        .collect::<Vec<_>>();
    files.sort();
    assert_eq!(files.len(), 252, "expected every M2 generated probe");
    for file in files {
        let source = std::fs::read(&file).unwrap();
        let checked = check::check(front::parse(&source).unwrap(), true).unwrap();
        let program = mir::lower(&checked).unwrap();
        let reference = eval::observe(&checked, 1_000_000);
        let (status, document) = native::run_program(&program, 1_000_000).unwrap();
        let (expected_status, expected) = match &reference.result {
            Ok(eval::Value::I32(value)) => (
                0,
                json!({"status":"ok","result":{"type":"i32","value":value}}),
            ),
            Ok(_) => panic!("checked main returned non-i32 for {file:?}"),
            Err(error) => (
                match error {
                    eval::EvalError::Trap { .. } => 3,
                    eval::EvalError::EvaluationLimit { .. } => 4,
                    eval::EvalError::Tool { .. } => 2,
                },
                error.to_json(""),
            ),
        };
        assert_eq!(status, expected_status, "native status {file:?}");
        for key in ["status", "result", "trap", "fuel_limit", "tool_code"] {
            if !expected[key].is_null() {
                assert_eq!(document[key], expected[key], "native {key} {file:?}");
            }
        }
        assert_eq!(
            document["steps_used"], reference.steps_used,
            "native fuel {file:?}"
        );
        assert_eq!(
            document["cleanup_trace"],
            json!(
                reference
                    .cleanup_trace
                    .iter()
                    .map(|event| json!({"name":event.name,"length":event.length}))
                    .collect::<Vec<_>>()
            ),
            "native cleanup {file:?}"
        );
    }
    eprintln!("native-serial agrees with reference for all 252 M2 probes");
}
