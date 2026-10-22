#![forbid(unsafe_code)]

use invar::exports::{
    CompiledExports, ExportEngine, ExportError, ExportSignature, ScalarType, ScalarValue,
    compile_exports,
};

const A: &[u8] = include_bytes!("../test-support/export-lineage/source-a.inv");
const B: &[u8] = include_bytes!("../test-support/export-lineage/source-b.inv");
const UNSELECTED: &[u8] =
    include_bytes!("../test-support/export-lineage/source-unselected-change.inv");
const SIGNATURE_CHANGE: &[u8] =
    include_bytes!("../test-support/export-lineage/source-signature-change.inv");
const SOURCE_PATH: &str = "/virtual/scalar-lineage.inv";
const BOTH: &[&str] = &["shared_symbol", "peer_symbol"];

fn engines() -> Vec<ExportEngine> {
    #[cfg(feature = "native")]
    {
        vec![ExportEngine::Reference, ExportEngine::NativeSerial]
    }
    #[cfg(not(feature = "native"))]
    {
        vec![ExportEngine::Reference]
    }
}

fn compile(source: &[u8], path: &str, names: &[&str], engine: ExportEngine) -> CompiledExports {
    match compile_exports(source, path, names, engine) {
        Ok(artifact) => artifact,
        Err(error) => panic!(
            "compile failed: {}",
            error.to_diagnostics_json(path, source)
        ),
    }
}

fn call(artifact: &mut CompiledExports, name: &str, args: &[ScalarValue]) -> ScalarValue {
    match artifact.invoke(name, args, 10_000) {
        Ok(value) => value,
        Err(error) => panic!("call failed: {}", artifact.diagnostics(&error)),
    }
}

fn descriptor_call(
    artifact: &mut CompiledExports,
    descriptor: &ExportSignature,
    args: &[ScalarValue],
) -> ScalarValue {
    match artifact.invoke_descriptor(descriptor, args, 10_000) {
        Ok(value) => value,
        Err(error) => panic!("descriptor call failed: {}", artifact.diagnostics(&error)),
    }
}

fn values(artifact: &mut CompiledExports, signed: i32, unsigned: u32) {
    assert_eq!(
        call(artifact, "shared_symbol", &[ScalarValue::I32(5)]),
        ScalarValue::I32(signed)
    );
    assert_eq!(
        call(artifact, "peer_symbol", &[ScalarValue::U32(7)]),
        ScalarValue::U32(unsigned)
    );
}

fn reject_stale(artifact: &mut CompiledExports, descriptor: &ExportSignature) {
    // MAX would overflow the original shared_symbol body. A stale descriptor
    // must fail at the boundary before that body can produce an evaluation error.
    for value in [5, i32::MAX] {
        assert!(
            matches!(
                artifact.invoke_descriptor(descriptor, &[ScalarValue::I32(value)], 10_000),
                Err(ExportError::Boundary { .. })
            ),
            "stale descriptor must reject before execution"
        );
    }
}

#[test]
fn issue_l1_06_export_lineage_simultaneous_artifacts_matching_symbols() {
    for engine in engines() {
        let mut a = compile(A, SOURCE_PATH, BOTH, engine);
        let mut b = compile(B, SOURCE_PATH, BOTH, engine);
        assert_ne!(a.artifact_id(), b.artifact_id());
        assert_eq!(a.signatures().len(), 2);
        assert_eq!(b.signatures().len(), 2);
        for (left, right) in a.signatures().iter().zip(b.signatures()) {
            assert_eq!(left.symbol, right.symbol);
            assert_eq!(left.params, right.params);
            assert_eq!(left.result, right.result);
            assert_eq!(left.artifact_id, a.artifact_id());
            assert_eq!(right.artifact_id, b.artifact_id());
        }
        let descriptor_a = a.signatures()[0].clone();
        let descriptor_b = b.signatures()[0].clone();
        for _ in 0..3 {
            values(&mut a, 16, 21);
            values(&mut b, 28, 36);
            reject_stale(&mut b, &descriptor_a);
            reject_stale(&mut a, &descriptor_b);
        }
        assert_eq!(
            descriptor_call(&mut a, &descriptor_a, &[ScalarValue::I32(5)]),
            ScalarValue::I32(16)
        );
        assert_eq!(
            descriptor_call(&mut b, &descriptor_b, &[ScalarValue::I32(5)]),
            ScalarValue::I32(28)
        );
    }
}

#[test]
fn issue_l1_06_export_lineage_source_bytes_path_and_unselected_changes() {
    for engine in engines() {
        let mut a = compile(A, SOURCE_PATH, &["shared_symbol"], engine);
        let mut unselected = compile(UNSELECTED, SOURCE_PATH, &["shared_symbol"], engine);
        let mut other_path = compile(A, "/virtual/changed-path.inv", &["shared_symbol"], engine);
        assert_ne!(a.artifact_id(), unselected.artifact_id());
        assert_ne!(a.artifact_id(), other_path.artifact_id());
        let descriptor = a.signatures()[0].clone();
        reject_stale(&mut unselected, &descriptor);
        reject_stale(&mut other_path, &descriptor);
        for artifact in [&mut a, &mut unselected, &mut other_path] {
            assert_eq!(
                call(artifact, "shared_symbol", &[ScalarValue::I32(5)]),
                ScalarValue::I32(16)
            );
        }
    }
}

#[test]
fn issue_l1_06_export_lineage_selected_membership_and_order() {
    for engine in engines() {
        let mut single = compile(A, SOURCE_PATH, &["shared_symbol"], engine);
        let mut both = compile(A, SOURCE_PATH, BOTH, engine);
        let mut reversed = compile(A, SOURCE_PATH, &["peer_symbol", "shared_symbol"], engine);
        assert_ne!(single.artifact_id(), both.artifact_id());
        assert_ne!(both.artifact_id(), reversed.artifact_id());
        let names = |artifact: &CompiledExports| {
            artifact
                .signatures()
                .iter()
                .map(|signature| signature.source_name.clone())
                .collect::<Vec<_>>()
        };
        assert_eq!(names(&single), ["shared_symbol"]);
        assert_eq!(names(&both), ["shared_symbol", "peer_symbol"]);
        assert_eq!(names(&reversed), ["peer_symbol", "shared_symbol"]);
        let descriptor = single.signatures()[0].clone();
        reject_stale(&mut both, &descriptor);
        reject_stale(&mut reversed, &descriptor);
        assert!(matches!(
            single.invoke("peer_symbol", &[ScalarValue::U32(7)], 10_000),
            Err(ExportError::Boundary { .. })
        ));
        assert_eq!(
            descriptor_call(&mut single, &descriptor, &[ScalarValue::I32(5)]),
            ScalarValue::I32(16)
        );
        values(&mut both, 16, 21);
        values(&mut reversed, 16, 21);
    }
}

#[test]
fn issue_l1_06_export_lineage_full_descriptor_validation_and_signature_change() {
    for engine in engines() {
        let mut a = compile(A, SOURCE_PATH, BOTH, engine);
        let mut changed = compile(SIGNATURE_CHANGE, SOURCE_PATH, BOTH, engine);
        let original = a.signatures()[0].clone();
        reject_stale(&mut changed, &original);
        assert_ne!(a.artifact_id(), changed.artifact_id());
        assert_eq!(changed.signatures()[0].params[0].ty, ScalarType::U32);
        assert_eq!(changed.signatures()[0].result, ScalarType::U32);
        assert_eq!(
            call(&mut changed, "shared_symbol", &[ScalarValue::U32(5)]),
            ScalarValue::U32(46)
        );
        assert_eq!(
            call(&mut changed, "peer_symbol", &[ScalarValue::I32(5)]),
            ScalarValue::I32(25)
        );

        let mut variants = Vec::new();
        let mut descriptor = original.clone();
        descriptor.source_name = "peer_symbol".into();
        variants.push(descriptor);
        let mut descriptor = original.clone();
        descriptor.symbol = "inv_export_999".into();
        variants.push(descriptor);
        let mut descriptor = original.clone();
        descriptor.span.start ^= 1;
        variants.push(descriptor);
        let mut descriptor = original.clone();
        descriptor.params[0].name = "changed".into();
        variants.push(descriptor);
        let mut descriptor = original.clone();
        descriptor.params[0].ty = ScalarType::U32;
        variants.push(descriptor);
        let mut descriptor = original.clone();
        descriptor.result = ScalarType::U32;
        variants.push(descriptor);
        let mut descriptor = original.clone();
        descriptor.artifact_id = "0".repeat(64);
        variants.push(descriptor);
        for variant in variants {
            reject_stale(&mut a, &variant);
        }
        assert_eq!(
            descriptor_call(&mut a, &original, &[ScalarValue::I32(5)]),
            ScalarValue::I32(16)
        );
    }
}

#[test]
fn issue_l1_06_export_lineage_retained_metadata_after_artifact_drop() {
    for engine in engines() {
        let a = compile(A, SOURCE_PATH, BOTH, engine);
        let retained = a.signatures()[0].clone();
        let identity = a.artifact_id().to_owned();
        assert_eq!(identity.len(), 64);
        assert!(
            identity
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
        );
        drop(a);
        let mut b = compile(B, SOURCE_PATH, BOTH, engine);
        reject_stale(&mut b, &retained);
        assert_eq!(retained.artifact_id, identity);
        values(&mut b, 28, 36);
    }
    // The frozen reference identity is a content identity, not a generation
    // counter. A descriptor from equal checked inputs remains interchangeable.
    let first = compile(A, SOURCE_PATH, BOTH, ExportEngine::Reference);
    let retained = first.signatures()[0].clone();
    let identity = first.artifact_id().to_owned();
    drop(first);
    let mut same_inputs = compile(A, SOURCE_PATH, BOTH, ExportEngine::Reference);
    assert_eq!(same_inputs.artifact_id(), identity);
    assert_eq!(
        descriptor_call(&mut same_inputs, &retained, &[ScalarValue::I32(5)]),
        ScalarValue::I32(16)
    );
}

#[cfg(feature = "native")]
#[test]
fn issue_l1_06_export_lineage_native_and_reference_engine_identity() {
    let mut reference = compile(A, SOURCE_PATH, BOTH, ExportEngine::Reference);
    let mut native = compile(A, SOURCE_PATH, BOTH, ExportEngine::NativeSerial);
    assert_ne!(reference.artifact_id(), native.artifact_id());
    println!(
        "{}",
        serde_json::json!({
            "kind": "scalar_export_build_identity_observation",
            "compiler_build_digest": env!("INVAR_BUILD_DIGEST"),
            "debug_assertions": cfg!(debug_assertions),
            "target_arch": std::env::consts::ARCH,
            "target_os": std::env::consts::OS,
            "reference_artifact_id": reference.artifact_id(),
            "native_artifact_id": native.artifact_id(),
        })
    );
    let reference_descriptor = reference.signatures()[0].clone();
    let native_descriptor = native.signatures()[0].clone();
    reject_stale(&mut native, &reference_descriptor);
    reject_stale(&mut reference, &native_descriptor);
    values(&mut reference, 16, 21);
    values(&mut native, 16, 21);
}
