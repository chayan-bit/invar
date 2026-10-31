//! Parent-authored end-to-end failure checks for the frozen scalar export API.
//! Native cases require the real native feature and are not replaced by mocks.
//! Issue #14 (L1-06).

use invar::exports::{ExportEngine, ExportError, ScalarValue, compile_exports};

const PATH: &str = "export-failure-original.inv";
const DIVIDE: &str =
    "// π preserves UTF-8 byte provenance\nfn target(x: i32) -> i32 { return 8 / x; }";

fn assert_recovery(engine: ExportEngine) {
    let mut artifact = compile_exports(DIVIDE.as_bytes(), PATH, &["target"], engine)
        .expect("admitted no-main scalar source");
    assert_eq!(
        artifact
            .invoke("target", &[ScalarValue::I32(2)], 100)
            .expect("successful scalar call"),
        ScalarValue::I32(4)
    );
    let error = artifact
        .invoke("target", &[ScalarValue::I32(0)], 100)
        .expect_err("source division trap");
    assert!(
        matches!(error, ExportError::Evaluation(_)),
        "must preserve a source outcome"
    );
    let first = artifact.diagnostics(&error);
    assert_eq!(first, error.to_diagnostics_json(PATH, DIVIDE.as_bytes()));
    assert_eq!(
        artifact
            .invoke("target", &[ScalarValue::I32(4)], 100)
            .expect("successful scalar call"),
        ScalarValue::I32(2)
    );
    let repeated = artifact
        .invoke("target", &[ScalarValue::I32(0)], 100)
        .expect_err("repeat trap");
    assert_eq!(
        artifact.diagnostics(&repeated),
        first,
        "no stale state or span drift"
    );
}

fn assert_boundary_reset(engine: ExportEngine) {
    let mut artifact =
        compile_exports(DIVIDE.as_bytes(), PATH, &["target"], engine).expect("scalar artifact");
    assert!(
        artifact
            .invoke("target", &[ScalarValue::I32(2)], 0)
            .is_err()
    );
    assert!(artifact.invoke("target", &[], 100).is_err());
    assert!(
        artifact
            .invoke("target", &[ScalarValue::U32(2)], 100)
            .is_err()
    );
    assert!(
        artifact
            .invoke("missing", &[ScalarValue::I32(2)], 100)
            .is_err()
    );
    assert_eq!(
        artifact
            .invoke("target", &[ScalarValue::I32(2)], 100)
            .expect("successful scalar call"),
        ScalarValue::I32(4)
    );
}

#[test]
fn export_failure_reference_recovery() {
    assert_recovery(ExportEngine::Reference);
}
#[test]
fn export_failure_reference_boundary_reset() {
    assert_boundary_reset(ExportEngine::Reference);
}
#[test]
fn export_failure_reference_fuel_reset() {
    let source = b"fn target(x: i32) -> i32 { var y: i32 = x; while y == 1 { y = x; } return y; }";
    let mut artifact = compile_exports(source, PATH, &["target"], ExportEngine::Reference)
        .expect("bounded-by-fuel loop");
    let error = artifact
        .invoke("target", &[ScalarValue::I32(1)], 32)
        .expect_err("fuel exhaustion");
    assert!(matches!(error, ExportError::Evaluation(_)));
    assert_eq!(
        artifact
            .invoke("target", &[ScalarValue::I32(0)], 100)
            .expect("successful scalar call"),
        ScalarValue::I32(0)
    );
}
#[test]
fn export_failure_selected_branch_only() {
    let source =
        b"fn target(x: i32) -> i32 { if x == 0 { return 7; } else { return 1 / (x - x); } }";
    let mut artifact = compile_exports(source, PATH, &["target"], ExportEngine::Reference)
        .expect("valid branch source");
    assert_eq!(
        artifact
            .invoke("target", &[ScalarValue::I32(0)], 100)
            .expect("successful scalar call"),
        ScalarValue::I32(7)
    );
}
#[cfg(feature = "native")]
#[test]
fn export_failure_native_recovery() {
    assert_recovery(ExportEngine::NativeSerial);
}
#[cfg(feature = "native")]
#[test]
fn export_failure_native_boundary_reset() {
    assert_boundary_reset(ExportEngine::NativeSerial);
}
#[cfg(feature = "native")]
#[test]
fn export_failure_native_reference_diagnostics() {
    let cases = [
        (DIVIDE, 0, 100),
        ("fn target(x: i32) -> i32 { return x + 1; }", i32::MAX, 100),
        ("fn target(x: i32) -> i32 { return x / -1; }", i32::MIN, 100),
        (
            "fn target(x: i32) -> i32 { return (1 / x) + (2147483647 + 1); }",
            0,
            100,
        ),
        (
            "fn inner(x: i32) -> i32 { return 8 / x; } fn target(x: i32) -> i32 { return inner(x) + 1; }",
            0,
            100,
        ),
        (
            "fn target(x: i32) -> i32 { var y: i32 = x; while y == x { y = x; } return y; }",
            1,
            32,
        ),
    ];
    for (source, arg, fuel) in cases {
        let mut reference = compile_exports(
            source.as_bytes(),
            PATH,
            &["target"],
            ExportEngine::Reference,
        )
        .expect("reference admission");
        let mut native = compile_exports(
            source.as_bytes(),
            PATH,
            &["target"],
            ExportEngine::NativeSerial,
        )
        .expect("real native admission");
        let expected = reference
            .invoke("target", &[ScalarValue::I32(arg)], fuel)
            .expect_err("reference failure");
        let actual = native
            .invoke("target", &[ScalarValue::I32(arg)], fuel)
            .expect_err("native failure returns instead of terminating");
        assert!(matches!(expected, ExportError::Evaluation(_)));
        assert!(matches!(actual, ExportError::Evaluation(_)));
        assert_eq!(
            native.diagnostics(&actual),
            reference.diagnostics(&expected),
            "{source}"
        );
    }
}
#[cfg(feature = "native")]
#[test]
fn export_failure_native_remainder_minimum() {
    let source = b"fn target(x: i32) -> i32 { return x % -1; }";
    let mut artifact = compile_exports(source, PATH, &["target"], ExportEngine::NativeSerial)
        .expect("remainder source");
    assert_eq!(
        artifact
            .invoke("target", &[ScalarValue::I32(i32::MIN)], 100)
            .expect("successful scalar call"),
        ScalarValue::I32(0)
    );
}
