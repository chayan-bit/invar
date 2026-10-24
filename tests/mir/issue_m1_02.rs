#![cfg(feature = "native")]

use invar::{
    check::check,
    eval::{self, EvalError, TrapCategory, Value},
    front::parse,
    mir, mir_eval, native,
};
use serde_json::{Value as Json, json};

use crate::{corpus::CORPUS, scratch};

fn lower(source: &str) -> (invar::check::CheckedProgram, mir::Program) {
    let checked = check(parse(source.as_bytes()).unwrap(), true).unwrap();
    let program = mir::lower(&checked).unwrap();
    (checked, program)
}

fn observed(result: Result<Value, EvalError>) -> (i32, Json) {
    match result {
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
        Err(EvalError::EvaluationLimit {
            fuel_limit,
            steps_used,
        }) => (
            4,
            json!({"status":"evaluation_limit","fuel_limit":fuel_limit,"steps_used":steps_used}),
        ),
        Err(EvalError::Tool { code, .. }) => (2, json!({"status":"tool_error","tool_code":code})),
        other => panic!("unexpected main result: {other:?}"),
    }
}

/// Native-only additions to the shared MIR corpus: signed/unsigned comparison, overflow and resource traps.
const EXTRA: &[(&str, u64)] = &[
    (
        "fn main() -> i32 { if -2 < -1 { return 1; } else { return 0; } }",
        10_000,
    ),
    (
        "fn main() -> i32 { if 4294967295u32 > 1u32 { return 1; } else { return 0; } }",
        10_000,
    ),
    ("fn main() -> i32 { return 2147483647i32 + 1i32; }", 10_000),
    ("fn main() -> i32 { return -2147483648i32 - 1i32; }", 10_000),
    (
        "fn main() -> i32 { if wrapping_mul(4294967295u32, 4294967295u32) == 1u32 { return 1; } else { return 0; } }",
        10_000,
    ),
    (
        "fn main() -> i32 { let xs: Array<i32> = filled_i32(16777217u32, 0); return 0; }",
        10_000,
    ),
    (
        "fn recurse() -> i32 { return recurse(); } fn main() -> i32 { return recurse(); }",
        10_000,
    ),
];

#[test]
fn issue_m1_02_native_cpu_trap_abi_differential() {
    for &(source, fuel) in CORPUS.iter().chain(EXTRA) {
        let (checked, program) = lower(source);
        let reference = eval::observe(&checked, fuel);
        let mir = mir_eval::observe(&program, fuel);
        let mir_result = observed(mir.result);
        let expected_result = observed(reference.result);
        assert_eq!(mir_result, expected_result, "MIR disagreement: {source}");
        assert_eq!(mir.steps_used, reference.steps_used, "MIR steps: {source}");
        assert_eq!(
            mir.cleanup_trace, reference.cleanup_trace,
            "MIR cleanup: {source}"
        );
        let (status, document) = native::run_program(&program, fuel).unwrap();
        let (expected_status, expected) = expected_result;
        assert_eq!(status, expected_status, "{source}");
        for key in ["status", "result", "trap", "fuel_limit", "tool_code"] {
            if !expected[key].is_null() {
                assert_eq!(document[key], expected[key], "{key}: {source}");
            }
        }
        assert_eq!(document["steps_used"], reference.steps_used, "{source}");
        assert_eq!(
            document["cleanup_trace"],
            json!(
                reference
                    .cleanup_trace
                    .iter()
                    .map(|event| json!({"name":event.name,"length":event.length}))
                    .collect::<Vec<_>>()
            ),
            "{source}"
        );
    }
}

#[test]
fn issue_m1_02_native_cpu_trap_abi_cli_build_and_run() {
    let directory = std::env::temp_dir().join(format!("invar-native-cli-{}", scratch::unique()));
    std::fs::create_dir(&directory).unwrap();
    let source = directory.join("program.inv");
    let output = directory.join("program");
    std::fs::write(&source, "fn main() -> i32 { return 6 * 7; }").unwrap();
    let binary = env!("CARGO_BIN_EXE_invar");
    let build = std::process::Command::new(binary)
        .args([
            "build",
            source.to_str().unwrap(),
            "--target",
            "native",
            "--output",
            output.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert_eq!(
        build.status.code(),
        Some(0),
        "{}",
        String::from_utf8_lossy(&build.stderr)
    );
    let direct = std::process::Command::new(&output).output().unwrap();
    assert_eq!(direct.status.code(), Some(0));
    assert_eq!(
        serde_json::from_slice::<Json>(&direct.stdout).unwrap()["result"]["value"],
        42
    );
    let run = std::process::Command::new(binary)
        .args([
            "run",
            source.to_str().unwrap(),
            "--engine",
            "native-serial",
            "--format",
            "json",
        ])
        .output()
        .unwrap();
    assert_eq!(
        run.status.code(),
        Some(0),
        "{}",
        String::from_utf8_lossy(&run.stderr)
    );
    assert_eq!(
        serde_json::from_slice::<Json>(&run.stdout).unwrap()["result"]["value"],
        42
    );
    let overwrite = std::process::Command::new(binary)
        .args([
            "build",
            source.to_str().unwrap(),
            "--target",
            "native",
            "--output",
            output.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert_eq!(overwrite.status.code(), Some(2));
    std::fs::remove_dir_all(directory).unwrap();
}

#[test]
fn issue_m1_02_native_cpu_trap_abi_rejects_invalid_mir_and_ir_promises() {
    let (_, program) = lower("fn main() -> i32 { return 2 + 3; }");
    let object = native::emit_mir(&program).unwrap();
    for forbidden in [
        " nsw ",
        " nuw ",
        " noalias ",
        " mustprogress ",
        " inbounds ",
    ] {
        assert!(!object.ir.contains(forbidden));
    }
    assert!(object.ir.contains("icmp slt i64"));
    assert!(object.ir.contains("@invar_trap_v1"));
    let mut invalid = program;
    invalid.schema_version += 1;
    assert!(
        native::emit_mir(&invalid)
            .unwrap_err()
            .contains("unknown MIR schema")
    );
}
