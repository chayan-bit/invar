//! Issue label L1-03.
use std::process::Command;

use invar::{check, eval, front};

use crate::engines::agree;

#[test]
fn generics_and_closures_monomorphized_functions_agree() {
    let source = include_str!("../fixtures/l1_03_generics.inv");
    let checked = check::check(front::parse(source.as_bytes()).unwrap(), true).unwrap();
    assert_eq!(checked.program.functions.len(), 4);
    agree(source, 1000, 12);
}

#[test]
fn generics_and_closures_array_type_substitution_agrees() {
    agree(
        include_str!("../fixtures/l1_03_generic_array.inv"),
        1000,
        17,
    );
}

#[test]
fn generics_and_closures_affine_capture_engines_agree() {
    let reference = agree(include_str!("../fixtures/l1_03_closures.inv"), 1000, 10);
    assert_eq!(
        reference.cleanup_trace,
        vec![
            eval::CleanupEvent {
                name: "ys".into(),
                length: 1
            },
            eval::CleanupEvent {
                name: "xs".into(),
                length: 2
            },
        ]
    );
}

#[test]
fn generics_and_closures_module_identity_and_unused_take_cleanup() {
    agree(
        include_str!("../fixtures/l1_03_module_closure.inv"),
        1000,
        6,
    );

    let source = "fn main() -> i32 { let xs: Array<i32> = [9i32]; let c = closure [take xs: Array<i32>]() -> i32 { return xs[0u32]; }; return 3i32; }";
    let reference = agree(source, 1000, 3);
    assert_eq!(reference.cleanup_trace.len(), 1);
    assert_eq!(reference.cleanup_trace[0].length, 1);
    assert!(reference.cleanup_trace[0].name.starts_with("$cap@"));
}

#[test]
fn generics_and_closures_public_check_rejects_invalid_sources() {
    for fixture in [
        "l1_03_generics.inv",
        "l1_03_closures.inv",
        "l1_03_generic_array.inv",
    ] {
        let path = format!("{}/tests/fixtures/{fixture}", env!("CARGO_MANIFEST_DIR"));
        let output = Command::new(env!("CARGO_BIN_EXE_invar"))
            .args(["check", &path, "--format", "json"])
            .output()
            .unwrap();
        assert!(output.status.success(), "{fixture}");
        let document: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
        assert_eq!(document["status"], "ok", "{fixture}");
    }
    for (fixture, code) in [
        ("unconstrained.inv", "TYPE_MISMATCH"),
        ("conflict.inv", "TYPE_MISMATCH"),
        ("explicit-mismatch.inv", "TYPE_MISMATCH"),
        ("recursive.inv", "UNSUPPORTED_FEATURE"),
        ("borrow-conflict.inv", "OWN_BORROW_CONFLICT"),
        ("double-move.inv", "OWN_USE_AFTER_MOVE"),
        ("escape.inv", "UNSUPPORTED_FEATURE"),
        ("double-call.inv", "OWN_USE_AFTER_MOVE"),
        ("unlisted-capture.inv", "NAME_UNKNOWN"),
    ] {
        let path = format!(
            "{}/tests/fixtures/l1_03/{fixture}",
            env!("CARGO_MANIFEST_DIR")
        );
        let output = Command::new(env!("CARGO_BIN_EXE_invar"))
            .args(["check", &path, "--format", "json"])
            .output()
            .unwrap();
        assert_eq!(output.status.code(), Some(1), "{fixture}");
        let document: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
        assert_eq!(document["status"], "source_error", "{fixture}");
        assert_eq!(document["diagnostics"][0]["code"], code, "{fixture}");
    }
}

#[test]
fn generics_and_closures_resource_limits() {
    let locals = (0..9)
        .map(|i| format!("let x{i}: i32 = 0i32;"))
        .collect::<String>();
    let captures = (0..9)
        .map(|i| format!("copy x{i}: i32"))
        .collect::<Vec<_>>()
        .join(",");
    let source = format!(
        "fn main() -> i32 {{ {locals} let c = closure [{captures}]() -> i32 {{ return 0i32; }}; return c(); }}"
    );
    assert_eq!(
        check::check(front::parse(source.as_bytes()).unwrap(), true)
            .unwrap_err()
            .code,
        "RESOURCE_LIMIT"
    );

    let mut source = "fn main() -> i32 { return 0i32; }".to_owned();
    for i in 0..128 {
        source.push_str(&format!("fn ordinary{i}() -> i32 {{ return 0i32; }}"));
    }
    assert_eq!(
        check::check(front::parse(source.as_bytes()).unwrap(), true)
            .unwrap_err()
            .code,
        "RESOURCE_LIMIT"
    );

    let source = format!(
        "fn main() -> i32 {{ let c = closure []() -> i32 {{{}return 0i32; }}; return c(); }}",
        " ".repeat(65_536)
    );
    assert_eq!(
        check::check(front::parse(source.as_bytes()).unwrap(), true)
            .unwrap_err()
            .code,
        "RESOURCE_LIMIT"
    );

    let mut source = String::new();
    for i in 0..17 {
        source.push_str(&format!(
            "fn g{i}<T: Value>(x: T) -> T {{ return {}; }}\n",
            if i == 16 {
                "x".to_owned()
            } else {
                format!("g{}(x)", i + 1)
            }
        ));
    }
    source.push_str("fn main() -> i32 { return g0(1i32); }");
    assert_eq!(
        check::check(front::parse(source.as_bytes()).unwrap(), true)
            .unwrap_err()
            .code,
        "RESOURCE_LIMIT"
    );

    let mut source = String::new();
    for i in 0..128 {
        source.push_str(&format!("fn f{i}<T: Value>(x: T) -> T {{ return x; }}\n"));
    }
    source.push_str("fn main() -> i32 {");
    for i in 0..128 {
        source.push_str(&format!("let x{i}: i32 = f{i}(0i32);"));
    }
    source.push_str("return 0i32; }");
    assert_eq!(
        check::check(front::parse(source.as_bytes()).unwrap(), true)
            .unwrap_err()
            .code,
        "RESOURCE_LIMIT"
    );
}

#[test]
fn generics_and_closures_rejects_trait_and_higher_kind_syntax() {
    for (source, code) in [
        (
            "fn f<T: Add>(x: T) -> T { return x; } fn main() -> i32 { return 0i32; }",
            "UNSUPPORTED_FEATURE",
        ),
        (
            "fn f<T<U>: Value>(x: T) -> T { return x; } fn main() -> i32 { return 0i32; }",
            "PARSE_EXPECTED_TOKEN",
        ),
    ] {
        assert_eq!(front::parse(source.as_bytes()).unwrap_err().code, code);
    }
}

#[test]
fn generics_and_closures_rejects_shadowed_closure_names() {
    for source in [
        "fn main() -> i32 { let x = 1i32; let x = closure []() -> i32 { return 2i32; }; return x(); }",
        "fn f() -> i32 { return 1i32; } fn main() -> i32 { let f = closure []() -> i32 { return 2i32; }; return f(); }",
        "fn main() -> i32 { let c = closure []() -> i32 { return 1i32; }; let c: i32 = 2i32; return c(); }",
    ] {
        assert_eq!(
            check::check(front::parse(source.as_bytes()).unwrap(), true)
                .unwrap_err()
                .code,
            "NAME_DUPLICATE"
        );
    }
}
