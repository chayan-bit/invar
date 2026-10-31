//! Issue #14 (L1-06).
use invar::exports::{ExportEngine, ExportError, ScalarValue, compile_exports};

#[test]
fn scalar_exports_admission_checks_original_and_reachable_types() {
    for source in [
        "fn target() -> Bool { return true; }",
        "fn target() -> Unit { return; }",
        "fn target() -> i32 { let xs: Array<i32> = [7]; return xs[0]; }",
        "fn helper() -> i32 { let xs: Array<i32> = [7]; return xs[0]; } fn target() -> i32 { return helper(); }",
    ] {
        assert!(
            matches!(
                compile_exports(
                    source.as_bytes(),
                    "admission.inv",
                    &["target"],
                    ExportEngine::Reference
                ),
                Err(ExportError::Boundary { .. })
            ),
            "{source}"
        );
    }
    let source = b"fn target() -> i32 { return 7; } fn unused() -> i32 { let xs: Array<i32> = [4]; return xs[0]; }";
    let mut artifact = compile_exports(
        source,
        "admission.inv",
        &["target"],
        ExportEngine::Reference,
    )
    .unwrap();
    assert_eq!(
        artifact.invoke("target", &[], 2).unwrap(),
        ScalarValue::I32(7)
    );
    for requested in [&[][..], &["target", "target"][..], &["missing"][..]] {
        assert!(
            compile_exports(source, "admission.inv", requested, ExportEngine::Reference).is_err()
        );
    }
    let invalid = b"fn target() -> i32 { return 7; } fn unused() -> i32 { return missing; }";
    assert!(matches!(
        compile_exports(
            invalid,
            "admission.inv",
            &["target"],
            ExportEngine::Reference
        ),
        Err(ExportError::Source(_))
    ));
}

#[test]
fn scalar_exports_identity_and_full_descriptor_binding() {
    let source = b"fn first() -> i32 { return 1; } fn second() -> u32 { return 2; }";
    let mut a = compile_exports(
        source,
        "identity.inv",
        &["first", "second"],
        ExportEngine::Reference,
    )
    .unwrap();
    let b = compile_exports(
        source,
        "identity.inv",
        &["second", "first"],
        ExportEngine::Reference,
    )
    .unwrap();
    let c = compile_exports(
        source,
        "other.inv",
        &["first", "second"],
        ExportEngine::Reference,
    )
    .unwrap();
    assert_ne!(a.artifact_id(), b.artifact_id());
    assert_ne!(a.artifact_id(), c.artifact_id());
    let descriptor = b.signatures()[1].clone();
    assert!(a.invoke_descriptor(&descriptor, &[], 100).is_err());
    let mut descriptor = a.signatures()[0].clone();
    descriptor.symbol = "inv_export_1".into();
    assert!(a.invoke_descriptor(&descriptor, &[], 100).is_err());
    assert_eq!(a.invoke("first", &[], 2).unwrap(), ScalarValue::I32(1));
}

#[test]
fn scalar_exports_qualified_visibility_and_frame_limits() {
    let source = b"module math { pub fn target(x: i32) -> i32 { return helper(x); } fn helper(x: i32) -> i32 { return x + 2; } }";
    let mut artifact = compile_exports(
        source,
        "module.inv",
        &["math::target"],
        ExportEngine::Reference,
    )
    .unwrap();
    assert_eq!(
        artifact
            .invoke("math::target", &[ScalarValue::I32(5)], 100)
            .unwrap(),
        ScalarValue::I32(7)
    );
    assert!(
        compile_exports(
            source,
            "module.inv",
            &["math::helper"],
            ExportEngine::Reference
        )
        .is_err()
    );
    let mut source = String::from("fn target(x: i32) -> i32 {");
    for index in 0..100 {
        source.push_str(&format!(" let v{index}: i32 = x + {index};"));
    }
    source.push_str(" return x; }");
    for engine in [ExportEngine::Reference, ExportEngine::NativeSerial] {
        assert!(matches!(
            compile_exports(source.as_bytes(), "large.inv", &["target"], engine),
            Err(ExportError::Boundary {
                code: "NATIVE_ABI",
                ..
            })
        ));
    }
}

#[cfg(feature = "native")]
#[test]
fn scalar_exports_two_artifacts_and_bounded_heap_recursion() {
    let source =
        b"fn target(x: u32) -> u32 { if x == 0 { return 0; } else { return target(x - 1) + 1; } }";
    let other = b"fn target(x: u32) -> u32 { return wrapping_add(x, 17); }";
    let mut a = compile_exports(
        source,
        "native-a.inv",
        &["target"],
        ExportEngine::NativeSerial,
    )
    .unwrap();
    let mut b = compile_exports(
        other,
        "native-b.inv",
        &["target"],
        ExportEngine::NativeSerial,
    )
    .unwrap();
    let mut reference =
        compile_exports(source, "native-a.inv", &["target"], ExportEngine::Reference).unwrap();
    assert_ne!(a.artifact_id(), reference.artifact_id());
    assert_ne!(a.artifact_id(), b.artifact_id());
    assert_eq!(
        a.invoke("target", &[ScalarValue::U32(63)], 5000).unwrap(),
        ScalarValue::U32(63)
    );
    assert_eq!(
        b.invoke("target", &[ScalarValue::U32(u32::MAX)], 100)
            .unwrap(),
        ScalarValue::U32(16)
    );
    let actual = a
        .invoke("target", &[ScalarValue::U32(64)], 5000)
        .unwrap_err();
    let expected = reference
        .invoke("target", &[ScalarValue::U32(64)], 5000)
        .unwrap_err();
    assert_eq!(a.diagnostics(&actual), reference.diagnostics(&expected));
    assert_eq!(
        a.invoke("target", &[ScalarValue::U32(3)], 5000).unwrap(),
        ScalarValue::U32(3)
    );
    assert_eq!(
        b.invoke("target", &[ScalarValue::U32(4)], 100).unwrap(),
        ScalarValue::U32(21)
    );
}
