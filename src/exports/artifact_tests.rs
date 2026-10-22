//! Generated-library lineage checks through a private, safe build seam.
use super::*;
use invar_foreign::exports::{ExportLoadError, LoadedExports};
use invar_foreign::host_array::{HostArray, HostArrayCall};
use invar_foreign::{Arg, Error, Output};

#[test]
fn issue_l1_06_native_arrays_reordered_same_signature_inventory() {
    let artifact = compile_exports(
        b"fn first(xs: read Array<i32>) -> i32 { return xs[0u32] + 100; } fn second(xs: read Array<i32>) -> i32 { return xs[0u32] + 101; }",
        "reordered.inv",
        &["first", "second"],
        ExportEngine::NativeSerial,
    )
    .unwrap();
    let library = native(&artifact)
        .directory
        .join(format!("scalar.{}", std::env::consts::DLL_EXTENSION));
    let spec = native(&artifact).spec.clone();
    let mut reordered = spec.clone();
    reordered.descriptors.swap(0, 1);
    assert!(matches!(
        LoadedExports::load(&library, &reordered),
        Err(ExportLoadError::SymbolOrdinal {
            position: 0,
            ordinal: 1
        })
    ));
    let mut loaded = LoadedExports::load(&library, &spec).unwrap();
    let input = [0];
    let output = HostArrayCall::prepare(vec![HostArray::ReadI32(&input)])
        .unwrap()
        .invoke(
            &mut loaded,
            &spec.descriptors[1],
            &[],
            100,
            |invocation, _| Ok::<_, ()>(invocation.output),
        )
        .unwrap();
    assert_eq!(output, Some(Output::I32(101)));
}

#[test]
fn issue_l1_06_export_core_descriptor_mapping_uses_symbols() {
    use crate::eval::host_read::{HostArgument, HostReadArray};
    let mut artifact = compile_exports(
        b"fn first(xs: read Array<i32>) -> i32 { return xs[0u32] + 100; } fn second(xs: read Array<i32>) -> i32 { return xs[0u32] + 101; } fn plain() -> i32 { return 102; } fn peer() -> i32 { return 103; }",
        "mapping.inv",
        &["first", "second", "plain", "peer"],
        ExportEngine::NativeSerial,
    ).unwrap();
    // Fault seam: these private vectors normally come from the same compiler
    // order. Calls must bind by symbol even if their storage order differs.
    let Implementation::Native(native) = &mut artifact.implementation else {
        unreachable!()
    };
    native.descriptors.swap(0, 1);
    native.descriptors.swap(2, 3);
    artifact.signatures.reverse();
    let input = [0];
    for (name, expected) in [("first", 100), ("second", 101)] {
        let mut args = [HostArgument::Read(HostReadArray::I32(&input))];
        assert_eq!(
            artifact.invoke_arrays(name, &mut args, 100).unwrap().value,
            ScalarValue::I32(expected)
        );
    }
    for (name, expected) in [("plain", 102), ("peer", 103)] {
        assert_eq!(
            artifact.invoke(name, &[], 100).unwrap(),
            ScalarValue::I32(expected)
        );
        assert_eq!(
            artifact.invoke_arrays(name, &mut [], 100).unwrap().value,
            ScalarValue::I32(expected)
        );
    }
}

fn native(artifact: &CompiledExports) -> &NativeArtifact {
    match &artifact.implementation {
        Implementation::Native(native) => native,
        Implementation::Reference => panic!("test requires the actual native artifact"),
    }
}

fn value(artifact: &mut CompiledExports, expected: i32) {
    assert_eq!(
        artifact
            .invoke("same", &[ScalarValue::I32(5)], 100)
            .unwrap(),
        ScalarValue::I32(expected)
    );
}

#[test]
fn issue_l1_06_export_lineage_generated_library_removal_replacement_and_verified_reload() {
    let mut a = compile_exports(
        b"fn same(x: i32) -> i32 { return x + 11; }",
        "generated-a.inv",
        &["same"],
        ExportEngine::NativeSerial,
    )
    .unwrap();
    let mut b = compile_exports(
        b"fn same(x: i32) -> i32 { return x + 23; }",
        "generated-b.inv",
        &["same"],
        ExportEngine::NativeSerial,
    )
    .unwrap();
    let library_name = format!("scalar.{}", std::env::consts::DLL_EXTENSION);
    let path_a = native(&a).directory.join(&library_name);
    let path_b = native(&b).directory.join(library_name);
    let spec_a = native(&a).spec.clone();
    let spec_b = native(&b).spec.clone();
    let mut retained = LoadedExports::load(&path_a, &spec_a).unwrap();
    value(&mut a, 16);
    value(&mut b, 28);

    std::fs::remove_file(&path_a).unwrap();
    assert!(matches!(
        LoadedExports::load(&path_a, &spec_a),
        Err(ExportLoadError::Boundary(Error::Library))
    ));
    value(&mut a, 16);
    assert_eq!(
        retained
            .invoke(&spec_a.descriptors[0], &[Arg::I32(5)], 100)
            .unwrap()
            .output,
        Some(Output::I32(16))
    );

    // Atomic pathname replacement does not replace either retained Library.
    std::fs::rename(&path_b, &path_a).unwrap();
    assert!(matches!(
        LoadedExports::load(&path_a, &spec_a),
        Err(ExportLoadError::Identity)
    ));
    value(&mut a, 16);
    value(&mut b, 28);
    assert_eq!(
        retained
            .invoke(&spec_a.descriptors[0], &[Arg::I32(5)], 100)
            .unwrap()
            .output,
        Some(Output::I32(16))
    );

    // Reload uses a fresh private path: dynamic loaders may cache an old path.
    let reload_path = native(&a).directory.join(format!(
        "verified-reload.{}",
        std::env::consts::DLL_EXTENSION
    ));
    std::fs::copy(&path_a, &reload_path).unwrap();
    let mut reloaded = LoadedExports::load(&reload_path, &spec_b).unwrap();
    assert!(matches!(
        reloaded.invoke(&spec_a.descriptors[0], &[Arg::I32(5)], 100),
        Err(ExportLoadError::Identity)
    ));
    assert_eq!(
        reloaded
            .invoke(&spec_b.descriptors[0], &[Arg::I32(5)], 100)
            .unwrap()
            .output,
        Some(Output::I32(28))
    );
    drop(b);
    drop(a);
    assert_eq!(
        retained
            .invoke(&spec_a.descriptors[0], &[Arg::I32(5)], 100)
            .unwrap()
            .output,
        Some(Output::I32(16))
    );
    assert_eq!(
        reloaded
            .invoke(&spec_b.descriptors[0], &[Arg::I32(5)], 100)
            .unwrap()
            .output,
        Some(Output::I32(28))
    );
}

#[test]
fn issue_l1_06_native_arrays_have_no_public_export_symbol() {
    let artifact = compile_exports(
        b"fn first(xs: read Array<i32>) -> i32 { return xs[0u32]; } fn plain(x: i32) -> i32 { return x; }",
        "symbols.inv",
        &["first", "plain"],
        ExportEngine::NativeSerial,
    )
    .unwrap();
    let library = native(&artifact)
        .directory
        .join(format!("scalar.{}", std::env::consts::DLL_EXTENSION));
    let spec = native(&artifact).spec.clone();
    // The real descriptor loads through the private array entry.
    assert!(LoadedExports::load(&library, &spec).is_ok());
    // Claiming the array function is scalar forces a public-symbol lookup,
    // which fails: no inv_export_0 exists for a function with arrays.
    let mut scalar_claim = spec.clone();
    scalar_claim.descriptors[0].parameter_tags = vec![invar_foreign::I32];
    assert!(matches!(
        LoadedExports::load(&library, &scalar_claim),
        Err(ExportLoadError::Boundary(Error::Symbol))
    ));
    // The scalar function keeps its public symbol, and the public scalar
    // entry refuses the array descriptor before any begin.
    let mut loaded = LoadedExports::load(&library, &spec).unwrap();
    assert_eq!(
        loaded
            .invoke(&spec.descriptors[1], &[Arg::I32(4)], 100)
            .unwrap()
            .output,
        Some(Output::I32(4))
    );
    assert!(matches!(
        loaded.invoke(&spec.descriptors[0], &[Arg::I32(4)], 100),
        Err(ExportLoadError::Boundary(Error::Layout))
    ));
}

#[test]
fn issue_l1_06_f32_native_decode_failure_and_panic_restore_bits() {
    use invar_foreign::host_array::{HostArray, HostArrayCall, HostCallError};
    let source = b"fn write(xs: mut Array<f32>) -> u32 { xs[0u32] = 1.0; return len(xs); }";
    let mut artifact = compile_exports(
        source,
        "decode-f32.inv",
        &["write"],
        ExportEngine::NativeSerial,
    )
    .unwrap();
    let Implementation::Native(native) = &mut artifact.implementation else {
        unreachable!()
    };
    for failure in ["decode", "panic", "loader"] {
        let mut xs = [f32::from_bits(0x7f800001), -0.0, f32::from_bits(1)];
        let original = xs.map(f32::to_bits);
        let address = xs.as_ptr().addr();
        let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let call = HostArrayCall::prepare(vec![HostArray::MutF32(&mut xs)]).unwrap();
            let result = call.invoke(
                &mut native.loaded,
                &native.descriptors[0],
                &[],
                if failure == "loader" { 0 } else { 100 },
                |invocation, addresses| {
                    assert_eq!(addresses, &[address]);
                    assert_eq!(invocation.failure.kind, 0);
                    assert_eq!(invocation.output, Some(Output::U32(3)));
                    if failure == "panic" {
                        panic!("injected result decoding panic");
                    }
                    Err::<(), _>("injected result decoding failure")
                },
            );
            match failure {
                "loader" => assert!(matches!(result, Err(HostCallError::Loader(_)))),
                _ => assert_eq!(
                    result,
                    Err(HostCallError::Decode("injected result decoding failure"))
                ),
            }
        }));
        assert_eq!(outcome.is_err(), failure == "panic");
        assert_eq!(xs.map(f32::to_bits), original);
    }
}

/// #320: any thread-local in the generated image gives it a TLV on macOS, which
/// dyld never unloads and which costs one of 512 pthread keys, so about 500
/// libraries in one process abort the interpreter. The image must hold none.
#[test]
fn issue_320_generated_library_has_no_thread_local_storage() {
    let artifact = compile_exports(
        b"fn f(x: u32) -> u32 { return x + 1u32; }",
        "tls.inv",
        &["f"],
        ExportEngine::NativeSerial,
    )
    .unwrap();
    let library = native(&artifact)
        .directory
        .join(format!("scalar.{}", std::env::consts::DLL_EXTENSION));
    let bytes = std::fs::read(library).unwrap();
    let contains = |needle: &[u8]| bytes.windows(needle.len()).any(|window| window == needle);
    for section in [
        &b"__thread_vars"[..],
        b"__thread_data",
        b"__thread_bss",
        b".tdata",
        b".tbss",
    ] {
        assert!(
            !contains(section),
            "generated library has thread-local section {}",
            String::from_utf8_lossy(section)
        );
    }
}
