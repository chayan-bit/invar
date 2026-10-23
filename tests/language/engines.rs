//! One "every engine agrees" check for the language-core suites: the reference evaluator, the MIR
//! evaluator and (with `--features native`) the native backend run the same program and must report
//! the same value, step count and cleanup trace.
use invar::{check, eval, front, mir, mir_eval};

/// Run `source` (which must complete with `I32(expected)`) on every engine and return the reference outcome.
pub fn agree(source: &str, fuel: u64, expected: i32) -> eval::Outcome {
    let checked = check::check(front::parse(source.as_bytes()).unwrap(), true).unwrap();
    let reference = eval::run(&checked, fuel).unwrap();
    assert_eq!(reference.value, eval::Value::I32(expected), "{source}");
    let lowered = mir::lower(&checked).unwrap();
    assert_eq!(
        mir_eval::run(&lowered, fuel).unwrap(),
        reference,
        "{source}"
    );
    #[cfg(feature = "native")]
    {
        let (status, document) = invar::native::run_program(&lowered, fuel).unwrap();
        assert_eq!(status, 0, "{source}: {document}");
        assert_eq!(
            document["result"],
            serde_json::json!({"type":"i32","value":expected}),
            "{source}"
        );
        assert_eq!(document["steps_used"], reference.steps_used, "{source}");
        assert_eq!(
            document["cleanup_trace"],
            serde_json::json!(
                reference
                    .cleanup_trace
                    .iter()
                    .map(|event| serde_json::json!({"name":event.name,"length":event.length}))
                    .collect::<Vec<_>>()
            ),
            "{source}"
        );
    }
    reference
}
