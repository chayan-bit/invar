use std::process::Command;

use invar::{check, eval, front, mir, mir_eval};

use crate::engines::agree;

const MODULES: &str = include_str!("../fixtures/l1_01/modules-forward.inv");
const RECORD: &str = include_str!("../fixtures/l1_01/record-fields.inv");
const VARIANT: &str = include_str!("../fixtures/l1_01/variant-match.inv");
const MOVED_ARRAY: &str = include_str!("../fixtures/l1_01/record-moved-array.inv");
const RESULT: &str = include_str!("../fixtures/l1_01/result-propagation.inv");

/// `result` of `invar run --engine reference` on a l1_01 fixture.
fn cli_result(fixture: &str) -> serde_json::Value {
    let path = format!(
        "{}/tests/fixtures/l1_01/{fixture}",
        env!("CARGO_MANIFEST_DIR")
    );
    let output = Command::new(env!("CARGO_BIN_EXE_invar"))
        .args(["run", &path, "--engine", "reference", "--format", "json"])
        .output()
        .unwrap();
    assert!(output.status.success());
    let document: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    document["result"].clone()
}

#[test]
fn issue_l1_01_language_core_nested_result_and_propagation() {
    let reference = agree(RESULT, 10_000, 28);
    assert_eq!(
        reference
            .cleanup_trace
            .iter()
            .filter(|event| event.name == "scratch" && event.length == 2)
            .count(),
        2
    );
    assert_eq!(
        cli_result("result-propagation.inv"),
        serde_json::json!({"type":"i32","value":28})
    );
}

#[test]
fn issue_l1_01_language_core_rejects_result_payload_theft_in_mir() {
    let source = "fn peek(value: read Result<i32, u32>) -> i32 { return match value { Result::Ok(number) => number, Result::Err(code) => 0i32 }; } fn main() -> i32 { let result: Result<i32, u32> = Result::Ok(3i32); return peek(result); }";
    let checked = check::check(front::parse(source.as_bytes()).unwrap(), true).unwrap();
    let mut lowered = mir::lower(&checked).unwrap();
    let mut changed = false;
    for function in &mut lowered.functions {
        for block in &mut function.blocks {
            for instruction in &mut block.instructions {
                if let mir::Operation::VariantPayload { value, tag } = &instruction.kind {
                    instruction.kind = mir::Operation::VariantTakePayload {
                        value: *value,
                        tag: *tag,
                    };
                    changed = true;
                    break;
                }
            }
        }
    }
    assert!(changed);
    assert_eq!(mir::verify(&lowered).unwrap_err().code, "MIR_INVALID");
}

#[test]
fn issue_l1_01_language_core_result_rejects_wrong_ownership_and_conversion() {
    for (source, code) in [
        (
            include_str!("../fixtures/l1_01/result-wrong-error.inv"),
            "TYPE_MISMATCH",
        ),
        (
            include_str!("../fixtures/l1_01/result-borrow-owned-pattern.inv"),
            "OWN_MOVE_REQUIRED",
        ),
        (
            "fn main() -> i32 { let value: Result<i32, u32> = Result::Err(2i32); return 0i32; }",
            "TYPE_MISMATCH",
        ),
        (
            "fn main() -> i32 { let value = Result::Ok(2i32); return 0i32; }",
            "TYPE_MISMATCH",
        ),
        (
            "fn main() -> i32 { let value: i32 = Result::Ok(2i32)?; return value; }",
            "TYPE_MISMATCH",
        ),
    ] {
        let error = check::check(front::parse(source.as_bytes()).unwrap(), true).unwrap_err();
        assert_eq!(error.code, code, "{source}");
        assert!(error.span.is_some());
    }
}

#[test]
fn issue_l1_01_language_core_owned_and_borrowed_patterns() {
    agree(
        include_str!("../fixtures/l1_01/variant-owned-pattern.inv"),
        10_000,
        8,
    );
    let borrowed = include_str!("../fixtures/l1_01/variant-borrowed-pattern.inv");
    assert_eq!(
        check::check(front::parse(borrowed.as_bytes()).unwrap(), true)
            .unwrap()
            .program
            .functions
            .len(),
        1
    );
    for (source, code) in [
        (
            include_str!("../fixtures/l1_01/variant-reuse-after-match.inv"),
            "OWN_USE_AFTER_MOVE",
        ),
        (
            include_str!("../fixtures/l1_01/variant-read-param-move.inv"),
            "OWN_MOVE_REQUIRED",
        ),
        (
            include_str!("../fixtures/l1_01/variant-borrow-conflict.inv"),
            "OWN_BORROW_CONFLICT",
        ),
    ] {
        assert_eq!(
            check::check(front::parse(source.as_bytes()).unwrap(), true)
                .unwrap_err()
                .code,
            code
        );
    }
}

#[test]
fn issue_l1_01_language_core_moves_record_array_field_once() {
    agree(MOVED_ARRAY, 10_000, 9);

    for (source, code) in [
        (
            include_str!("../fixtures/l1_01/record-field-needs-move.inv").to_owned(),
            "OWN_MOVE_REQUIRED",
        ),
        (
            include_str!("../fixtures/l1_01/record-field-use-after-move.inv").to_owned(),
            "OWN_USE_AFTER_MOVE",
        ),
        (
            MOVED_ARRAY.replace(
                "return moved[1u32]",
                "let whole: Bundle = move bundle; return moved[1u32]",
            ),
            "OWN_USE_AFTER_MOVE",
        ),
    ] {
        let error = check::check(front::parse(source.as_bytes()).unwrap(), true).unwrap_err();
        assert_eq!(error.code, code, "{source}");
        assert!(error.span.is_some());
    }
}

#[test]
fn issue_l1_01_language_core_exhaustive_variant_match_agrees_across_engines() {
    agree(VARIANT, 10_000, 11);
    let low = VARIANT.replace("Choice::High(11i32)", "Choice::Low()");
    agree(&low, 10_000, 1);
    let checked = check::check(front::parse(low.as_bytes()).unwrap(), true).unwrap();
    let mut lowered = mir::lower(&checked).unwrap();
    for function in &mut lowered.functions {
        for block in &mut function.blocks {
            for instruction in &mut block.instructions {
                if let mir::Operation::Variant { tag, .. } = &mut instruction.kind {
                    *tag = usize::MAX;
                }
            }
        }
    }
    assert_eq!(mir::verify(&lowered).unwrap_err().code, "MIR_INVALID");
}

#[test]
fn issue_l1_01_language_core_rejects_nonexhaustive_variant_match() {
    let source = "enum Choice { Low, High(i32) } fn main() -> i32 { let x: Choice = Choice::Low(); return match x { Choice::Low => 1i32 }; }";
    let error = check::check(front::parse(source.as_bytes()).unwrap(), true).unwrap_err();
    assert_eq!(error.code, "MATCH_NONEXHAUSTIVE");
    assert!(error.span.is_some());
}

#[test]
fn issue_l1_01_language_core_imported_variant_and_visibility() {
    let source = "module m { pub enum Choice { Low, High(i32) } } import m::Choice; fn main() -> i32 { let choice: Choice = Choice::High(9i32); return match choice { Choice::Low => 0i32, Choice::High(v) => v }; }";
    agree(source, 10_000, 9);

    let private = source.replace("pub enum", "enum");
    let error = check::check(front::parse(private.as_bytes()).unwrap(), true).unwrap_err();
    assert_eq!(error.code, "NAME_UNKNOWN");
}

#[test]
fn issue_l1_01_language_core_record_fields_agree_across_engines() {
    agree(RECORD, 10_000, -7);
    assert_eq!(
        cli_result("record-fields.inv"),
        serde_json::json!({"type":"i32","value":-7})
    );
}

#[test]
fn issue_l1_01_language_core_rejects_missing_record_field() {
    let source = "struct Pair { left: i32, right: u32 } fn main() -> i32 { let pair: Pair = Pair { left: 1i32 }; return pair.left; }";
    let error = check::check(front::parse(source.as_bytes()).unwrap(), true).unwrap_err();
    assert_eq!(error.code, "TYPE_MISMATCH");
    assert!(error.span.is_some());
}

#[test]
fn issue_l1_01_language_core_named_condition_remains_a_block() {
    let source = "fn main() -> i32 { let ok: Bool = true; if ok { return 1; } else { return 2; } }";
    let checked = check::check(front::parse(source.as_bytes()).unwrap(), true).unwrap();
    assert_eq!(eval::run(&checked, 100).unwrap().value, eval::Value::I32(1));
}

#[test]
fn issue_l1_01_language_core_record_borrow_and_take() {
    agree(
        include_str!("../fixtures/l1_01/record-borrow-take.inv"),
        10_000,
        6,
    );
}

#[test]
fn issue_l1_01_language_core_rejects_record_use_after_move() {
    let source = include_str!("../fixtures/l1_01/record-use-after-move.inv");
    let error = check::check(front::parse(source.as_bytes()).unwrap(), true).unwrap_err();
    assert_eq!(error.code, "OWN_USE_AFTER_MOVE");
    assert!(error.span.is_some());
}

#[test]
fn issue_l1_01_language_core_forward_module_resolution_agrees_across_engines() {
    let checked = check::check(front::parse(MODULES.as_bytes()).unwrap(), true).unwrap();
    let reference = eval::run(&checked, 10_000).unwrap();
    assert_eq!(reference.value, eval::Value::I32(21));
    let lowered = mir::lower(&checked).unwrap();
    assert_eq!(
        mir_eval::run(&lowered, 10_000).unwrap().value,
        reference.value
    );
    assert_eq!(
        cli_result("modules-forward.inv"),
        serde_json::json!({"type":"i32","value":21})
    );
    #[cfg(feature = "native")]
    {
        let (status, document) = invar::native::run_program(&lowered, 10_000).unwrap();
        assert_eq!(status, 0);
        assert_eq!(
            document["result"],
            serde_json::json!({"type":"i32","value":21})
        );
    }
}

#[test]
fn issue_l1_01_language_core_rejects_visibility_duplicates_cycles_and_bad_imports() {
    for (source, code) in [
        (
            "module m { fn hidden() -> i32 { return 1; } } import m::hidden; fn main() -> i32 { return hidden(); }",
            "NAME_UNKNOWN",
        ),
        (
            "module m { pub fn f() -> i32 { return 1; } pub fn f() -> i32 { return 2; } } fn main() -> i32 { return 0; }",
            "NAME_DUPLICATE",
        ),
        (
            include_str!("../fixtures/l1_01/module-import-cycle.inv"),
            "MODULE_IMPORT_CYCLE",
        ),
        (
            "import missing::f; fn main() -> i32 { return 0; }",
            "NAME_UNKNOWN",
        ),
        (
            include_str!("../fixtures/l1_01/module-count-limit.inv"),
            "MODULE_LIMIT",
        ),
    ] {
        let program = front::parse(source.as_bytes()).unwrap();
        let error = check::check(program, true).unwrap_err();
        assert_eq!(error.code, code, "{source}");
        assert!(error.span.is_some());
    }
    let error = front::parse(b"import m::; fn main() -> i32 { return 0; }").unwrap_err();
    assert_eq!(error.code, "PARSE_EXPECTED_TOKEN");
    assert!(error.span.is_some());
}
