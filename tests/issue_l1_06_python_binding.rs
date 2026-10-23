//! Core-independent boundary policy gates; actual Python/C execution is in pytest.
#[path = "../crates/invar-python/src/validation.rs"]
mod validation;

use validation::{Dtype, Failure, Metadata, validate};

fn owned() -> Metadata {
    Metadata {
        dtype: Dtype::U32,
        rank: 1,
        length: 4,
        contiguous: true,
        aligned: true,
        writable: true,
        owns_data: true,
        address: 0x1000,
    }
}

#[test]
fn issue_l1_06_python_binding_validation_and_extents() {
    assert_eq!(validate(owned(), Dtype::U32, Some(4), true), Ok(()));
    let mut cases = Vec::new();
    let mut wrong = owned();
    wrong.dtype = Dtype::I32;
    cases.push((wrong, Failure::Dtype));
    let mut wrong = owned();
    wrong.rank = 2;
    cases.push((wrong, Failure::Rank));
    let mut wrong = owned();
    wrong.length = 3;
    cases.push((wrong, Failure::Shape));
    let mut wrong = owned();
    wrong.contiguous = false;
    cases.push((wrong, Failure::Contiguity));
    let mut wrong = owned();
    wrong.aligned = false;
    cases.push((wrong, Failure::Alignment));
    let mut wrong = owned();
    wrong.address += 1;
    cases.push((wrong, Failure::Alignment));
    let mut wrong = owned();
    wrong.writable = false;
    cases.push((wrong, Failure::Writability));
    let mut wrong = owned();
    wrong.owns_data = false;
    cases.push((wrong, Failure::Ownership));
    for (metadata, failure) in cases {
        assert_eq!(validate(metadata, Dtype::U32, Some(4), true), Err(failure));
        assert!(failure.code().starts_with("PY_"));
    }
    let mut oversized = owned();
    oversized.length = validation::MAX_ELEMENTS + 1;
    assert_eq!(
        validate(oversized, Dtype::U32, None, false),
        Err(Failure::Limit)
    );
    let mut overflowed = owned();
    overflowed.address = usize::MAX - 3;
    assert_eq!(
        validate(overflowed, Dtype::U32, None, false),
        Err(Failure::Limit)
    );
    assert_eq!(validation::overlaps(owned(), owned()), Ok(true));
    let mut adjacent = owned();
    adjacent.address += 16;
    assert_eq!(validation::overlaps(owned(), adjacent), Ok(false));
    let mut empty = owned();
    empty.length = 0;
    assert_eq!(validation::overlaps(empty, owned()), Ok(false));
    assert_eq!(Failure::Alias.code(), "PY_BUFFER_ALIAS");
}

#[test]
fn issue_l1_06_python_binding_empty_large_and_readonly_inputs() {
    for (name, dtype) in [
        ("uint8", Dtype::U8),
        ("int32", Dtype::I32),
        ("uint32", Dtype::U32),
        ("float32", Dtype::F32),
    ] {
        assert_eq!(Dtype::parse(name), Some(dtype));
        for length in [0, 1, 100_000] {
            let metadata = Metadata {
                dtype,
                length,
                writable: false,
                ..owned()
            };
            assert_eq!(validate(metadata, dtype, Some(length), false), Ok(()));
            assert_eq!(
                validate(metadata, dtype, Some(length), true),
                Err(Failure::Writability)
            );
        }
    }
    assert_eq!(Dtype::parse("float64"), Some(Dtype::F64));
    assert_eq!(Dtype::parse("bool"), Some(Dtype::Bool));
    assert_eq!(Dtype::parse("float16"), None);
    assert_eq!((Dtype::F64.size(), Dtype::Bool.size()), (8, 1));
    // HA-1.4: an f64 buffer must be 8-byte aligned, a bool buffer has no alignment, and a bool byte is 0 or 1.
    let f64s = Metadata {
        dtype: Dtype::F64,
        address: 0x1004,
        ..owned()
    };
    assert_eq!(
        validate(f64s, Dtype::F64, None, false),
        Err(Failure::Alignment)
    );
    let bools = Metadata {
        dtype: Dtype::Bool,
        address: 0x1001,
        ..owned()
    };
    assert_eq!(validate(bools, Dtype::Bool, None, false), Ok(()));
    assert!(validation::bool_bytes_valid(&[0, 1, 1, 0]));
    assert!(validation::bool_bytes_valid(&[]));
    assert!(!validation::bool_bytes_valid(&[0, 2]));
    assert!(!validation::bool_bytes_valid(&[255]));
    assert_eq!(Failure::BoolValue.code(), "PY_ARRAY_BOOL_VALUE");
    let metadata = Metadata {
        dtype: Dtype::U8,
        length: validation::MAX_BYTES + 1,
        ..owned()
    };
    assert_eq!(
        validate(metadata, Dtype::U8, None, false),
        Err(Failure::Limit)
    );
}

#[test]
fn issue_l1_06_python_binding_real_source_diagnostics_and_reference_smoke() {
    let source = b"fn main() -> i32 { return 6 * 7; }";
    let checked = invar::check::check(invar::front::parse(source).unwrap(), true).unwrap();
    invar::ownership::check(&checked).unwrap();
    assert_eq!(
        invar::eval::run(&checked, 100).unwrap().value,
        invar::eval::Value::I32(42)
    );
    let source = b"fn main() -> i32 { return 1 / 0; }";
    let checked = invar::check::check(invar::front::parse(source).unwrap(), true).unwrap();
    let document = invar::eval::run(&checked, 100)
        .unwrap_err()
        .to_diagnostics_json("<python>", source);
    let schema: serde_json::Value =
        serde_json::from_str(include_str!("../schema/invar-diagnostics-v3.schema.json")).unwrap();
    assert!(jsonschema::is_valid(&schema, &document));
    assert_eq!(document["diagnostics"][0]["code"], "DIVISION_BY_ZERO");
}

#[test]
fn issue_l1_06_python_binding_frozen_abi_has_no_numeric_array_or_export_tags() {
    let header = include_str!("../include/invar_foreign_v1.h");
    assert!(header.contains("INV_TAG_BYTES_V1 3u"));
    assert!(!header.contains("INV_TAG_F32"));
    assert!(!header.contains("INV_TAG_ARRAY"));
    assert!(include_str!("../src/lib.rs").contains("#![forbid(unsafe_code)]"));
}

#[test]
fn issue_l1_06_python_binding_scalar_artifact_values_trap_and_recovery() {
    use invar::exports::{ExportEngine, ScalarValue, compile_exports};
    let source = "// UTF-8 span witness: é.\nfn negate(x: i32) -> i32 { return -x; }\nfn same(x: u32) -> u32 { return x; }".as_bytes();
    let mut artifact = compile_exports(
        source,
        "binding-snapshot.inv",
        &["same", "negate"],
        ExportEngine::Reference,
    )
    .unwrap_or_else(|error| {
        panic!(
            "{}",
            error.to_diagnostics_json("binding-snapshot.inv", source)
        )
    });
    assert_eq!(
        artifact
            .signatures()
            .iter()
            .map(|signature| signature.source_name.as_str())
            .collect::<Vec<_>>(),
        vec!["same", "negate"]
    );
    assert_eq!(
        artifact
            .invoke("same", &[ScalarValue::U32(u32::MAX)], 100)
            .ok(),
        Some(ScalarValue::U32(u32::MAX))
    );
    assert_eq!(
        artifact.invoke("negate", &[ScalarValue::I32(1)], 100).ok(),
        Some(ScalarValue::I32(-1))
    );
    let error = artifact
        .invoke("negate", &[ScalarValue::I32(i32::MIN)], 100)
        .expect_err("minimum negation must trap");
    let document = artifact.diagnostics(&error);
    let span_start = source.windows(2).position(|bytes| bytes == b"-x").unwrap();
    assert_eq!(
        document["trap"],
        serde_json::json!({"category":"INTEGER_OVERFLOW", "span":{"start":span_start,"end":span_start+1}})
    );
    assert_eq!(document["diagnostics"][0]["path"], "binding-snapshot.inv");
    let schema: serde_json::Value =
        serde_json::from_str(include_str!("../schema/invar-diagnostics-v3.schema.json")).unwrap();
    assert!(jsonschema::is_valid(&schema, &document));
    assert_eq!(
        artifact
            .invoke("same", &[ScalarValue::U32(u32::MAX)], 100)
            .ok(),
        Some(ScalarValue::U32(u32::MAX))
    );
    assert_eq!(
        artifact.invoke("negate", &[ScalarValue::I32(-1)], 100).ok(),
        Some(ScalarValue::I32(1))
    );
}

#[test]
fn issue_l1_06_python_binding_engine_names_and_guarded_receipts() {
    use invar::engine::{Engine, run_mir};
    for name in [
        "reference",
        "native-serial",
        "cpu-parallel",
        "auto",
        "cuda",
        "metal",
    ] {
        assert_eq!(Engine::parse(name).unwrap().name(), name);
    }
    for name in ["native", "", "unknown", "CPU"] {
        assert!(Engine::parse(name).is_none());
    }
    let source = b"fn main() -> i32 { return 42; }";
    let checked = invar::check::check(invar::front::parse(source).unwrap(), true).unwrap();
    let program = invar::mir::lower(&checked).unwrap();
    let receipt = run_mir(source, &program, Engine::Auto, 1000, 1, None).unwrap();
    assert_eq!(receipt["result"], 42);
    assert_eq!(receipt["backends"], serde_json::json!(["mir-serial"]));
    let schema = serde_json::from_str(include_str!(
        "../schema/invar-explain-device-v5.schema.json"
    ))
    .unwrap();
    assert!(jsonschema::is_valid(
        &schema,
        &invar::engine::device_document(&receipt)
    ));
    let receipt = run_mir(source, &program, Engine::CpuParallel, 1000, 1, None).unwrap();
    assert_eq!(receipt["fallback_reason"], "NO_ADMITTED_CANDIDATE");
    assert_eq!(receipt["backends"], serde_json::json!(["mir-serial"]));
    for engine in [Engine::Reference, Engine::NativeSerial] {
        assert!(matches!(
            run_mir(source, &program, engine, 1000, 1, None),
            Err(invar::eval::EvalError::Tool {
                code: "NATIVE_ABI",
                ..
            })
        ));
    }
}

#[test]
fn issue_l1_06_python_binding_cpu_regions_traps_and_limits() {
    use invar::engine::{Engine, run_mir};
    let source = b"fn main() -> i32 { let xs: Array<u32> = [1u32,2u32,3u32]; var out: Array<u32> = filled_u32(len(xs),0u32); for i in 0u32..len(xs) { out[i] = wrapping_add(xs[i],3u32); } if out[2u32] == 6u32 { return 42; } return 1; }";
    let checked = invar::check::check(invar::front::parse(source).unwrap(), true).unwrap();
    let program = invar::mir::lower(&checked).unwrap();
    for workers in [1, 2, 4] {
        let receipt = run_mir(source, &program, Engine::CpuParallel, 1000, workers, None).unwrap();
        assert_eq!(receipt["result"], 42);
        assert_eq!(receipt["backends"], serde_json::json!(["cpu-parallel"]));
        assert!(receipt["fallback_reason"].is_null());
        assert!(!receipt["choices"].as_array().unwrap().is_empty());
    }
    for workers in [0, 65] {
        assert!(matches!(
            run_mir(source, &program, Engine::CpuParallel, 1000, workers, None),
            Err(invar::eval::EvalError::Tool {
                code: "CPU_WORKERS",
                ..
            })
        ));
    }
    assert!(matches!(
        run_mir(source, &program, Engine::CpuParallel, 1, 1, None),
        Err(invar::eval::EvalError::EvaluationLimit { .. })
    ));
    let source = b"fn main() -> i32 { return 1 / 0; }";
    let checked = invar::check::check(invar::front::parse(source).unwrap(), true).unwrap();
    let program = invar::mir::lower(&checked).unwrap();
    assert!(matches!(
        run_mir(source, &program, Engine::CpuParallel, 100, 1, None),
        Err(invar::eval::EvalError::Trap { .. })
    ));
}

#[test]
fn issue_l1_06_python_binding_missing_device_features_never_replay() {
    use invar::engine::{Engine, run_mir};
    let source = b"fn main() -> i32 { return 42; }";
    let checked = invar::check::check(invar::front::parse(source).unwrap(), true).unwrap();
    let program = invar::mir::lower(&checked).unwrap();
    if !cfg!(feature = "cuda") {
        assert!(matches!(
            run_mir(source, &program, Engine::Cuda, 100, 1, None),
            Err(invar::eval::EvalError::Tool {
                code: "CUDA_UNAVAILABLE",
                ..
            })
        ));
    }
    if !cfg!(all(feature = "metal", target_os = "macos")) {
        assert!(matches!(
            run_mir(source, &program, Engine::Metal, 100, 1, None),
            Err(invar::eval::EvalError::Tool {
                code: "METAL_UNAVAILABLE",
                ..
            })
        ));
    }
}
