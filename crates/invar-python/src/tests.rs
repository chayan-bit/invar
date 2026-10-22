//! Binding unit tests: reference entry, artifact calls and error mapping.
use super::*;

const REFERENCE_SOURCE: &str = "fn first(xs: read Array<i32>) -> i32 { return xs[0u32]; }\nfn bump(xs: mut Array<i32>, d: i32) -> i32 { xs[0u32] = 9; return xs[1u32] / d; }";

fn reference_module(py: Python<'_>) -> CheckedModule {
    compile_source(
        py,
        PyString::new(py, REFERENCE_SOURCE).as_any(),
        "reference.inv",
        "reference",
        None,
    )
    .unwrap()
}

/// A NumPy-owned int32 array (`zeros` allocates and owns its data).
fn owned<'py>(py: Python<'py>, values: &[i32]) -> Bound<'py, numpy::PyArray1<i32>> {
    let array = numpy::PyArray1::<i32>::zeros(py, values.len(), false);
    array
        .readwrite()
        .as_slice_mut()
        .unwrap()
        .copy_from_slice(values);
    array
}

fn code_and_steps(py: Python<'_>, error: &PyErr) -> (String, u64) {
    let document: Value = serde_json::from_str(
        &error
            .value(py)
            .getattr("diagnostic_json")
            .unwrap()
            .extract::<String>()
            .unwrap(),
    )
    .unwrap();
    let steps = error
        .value(py)
        .getattr("steps_used")
        .unwrap()
        .extract::<u64>()
        .unwrap();
    (
        document["diagnostics"][0]["code"]
            .as_str()
            .unwrap()
            .to_owned(),
        steps,
    )
}

#[test]
fn ha1_reference_entry_refuses_conflicting_borrows_and_releases_guards() {
    Python::attach(|py| {
        let module = reference_module(py);
        let a = owned(py, &[5, 6]);
        // An existing writer: a read borrow by the call conflicts.
        {
            let held = a.readwrite();
            let args = PyTuple::new(py, [a.as_any()]).unwrap();
            let error = module
                .call_reference(py, "first", &args, Fuel(100))
                .unwrap_err();
            assert!(error.is_instance_of::<ValidationError>(py));
            assert_eq!(code_and_steps(py, &error), ("PY_BUFFER_ALIAS".into(), 0));
            assert_eq!(
                held.as_slice().unwrap(),
                &[5, 6],
                "the holder's guard stays valid"
            );
        }
        // An existing reader: a mut borrow by the call conflicts.
        {
            let held = a.readonly();
            let args = PyTuple::new(
                py,
                [
                    a.as_any().clone(),
                    1i32.into_pyobject(py).unwrap().into_any(),
                ],
            )
            .unwrap();
            let error = module
                .call_reference(py, "bump", &args, Fuel(100))
                .unwrap_err();
            assert_eq!(code_and_steps(py, &error), ("PY_BUFFER_ALIAS".into(), 0));
            assert_eq!(held.as_slice().unwrap(), &[5, 6], "unchanged");
        }
        // After a trap the guards are released and the bytes unchanged.
        let args = PyTuple::new(
            py,
            [
                a.as_any().clone(),
                0i32.into_pyobject(py).unwrap().into_any(),
            ],
        )
        .unwrap();
        let error = module
            .call_reference(py, "bump", &args, Fuel(100))
            .unwrap_err();
        assert!(error.is_instance_of::<TrapError>(py));
        assert!(code_and_steps(py, &error).1 > 0);
        assert!(a.try_readwrite().is_ok(), "guards released after a trap");
        assert_eq!(a.readonly().as_slice().unwrap(), &[5, 6]);
        // After a normal return the write is committed and guards released.
        let args = PyTuple::new(
            py,
            [
                a.as_any().clone(),
                2i32.into_pyobject(py).unwrap().into_any(),
            ],
        )
        .unwrap();
        let document = module.call_reference(py, "bump", &args, Fuel(100)).unwrap();
        let document = document.bind(py);
        assert_eq!(
            document
                .get_item("result")
                .unwrap()
                .extract::<i32>()
                .unwrap(),
            3
        );
        assert_eq!(a.readonly().as_slice().unwrap(), &[9, 6]);
        assert!(a.try_readwrite().is_ok(), "guards released after a return");
    });
}

#[test]
fn issue_l1_06_python_binding_artifact_call_original_trap_and_recovery() {
    Python::attach(|py| {
        let source = PyString::new(
            py,
            "// é\nfn divide(x: i32, y: i32) -> i32 { return x / y; }",
        );
        let selection = PyList::new(py, ["divide"]).unwrap();
        let mut module = compile_source(
            py,
            source.as_any(),
            "original.inv",
            "reference",
            Some(selection.as_any()),
        )
        .unwrap();
        let args = PyTuple::new(py, [-7, 3]).unwrap();
        assert_eq!(
            module
                .call(py, "divide", &args, Fuel(100), false)
                .unwrap()
                .extract::<i32>(py)
                .unwrap(),
            -2
        );
        let args = PyTuple::new(py, [1, 0]).unwrap();
        let error = module
            .call(py, "divide", &args, Fuel(100), false)
            .unwrap_err();
        assert!(error.is_instance_of::<TrapError>(py));
        let document: Value = serde_json::from_str(
            &error
                .value(py)
                .getattr("diagnostic_json")
                .unwrap()
                .extract::<String>()
                .unwrap(),
        )
        .unwrap();
        assert_eq!(document["diagnostics"][0]["code"], "DIVISION_BY_ZERO");
        assert_eq!(document["diagnostics"][0]["path"], "original.inv");
        let expected = module
            .source
            .windows(5)
            .position(|bytes| bytes == b"x / y")
            .unwrap()
            + 2;
        assert_eq!(
            document["trap"]["span"],
            json!({"start":expected,"end":expected+1})
        );
        let args = PyTuple::new(py, [6, 3]).unwrap();
        assert_eq!(
            module
                .call(py, "divide", &args, Fuel(100), false)
                .unwrap()
                .extract::<i32>(py)
                .unwrap(),
            2
        );
    });
}

#[test]
fn issue_l1_06_python_binding_exact_scalar_validation_without_callbacks() {
    Python::attach(|py| {
        for value in [i32::MIN, -1, 0, i32::MAX] {
            let argument = value.into_pyobject(py).unwrap();
            assert_eq!(scalar_i32(py, argument.as_any()).unwrap(), value);
        }
        for value in [0, 1, u32::MAX] {
            let argument = value.into_pyobject(py).unwrap();
            assert_eq!(scalar_u32(py, argument.as_any()).unwrap(), value);
        }

        let scope = PyDict::new(py);
        py.run(
            c"called = False\nclass Index:\n def __index__(self):\n  global called\n  called = True\n  return 1\nclass Integer(int): pass\nvalues = [True, 1.0, '1', None, Index(), Integer(1), 1 << 100]\n",
            Some(&scope),
            Some(&scope),
        )
        .unwrap();
        let values = scope.get_item("values").unwrap().unwrap();
        let schema: Value = serde_json::from_str(include_str!(
            "../../../schema/invar-diagnostics-v1.schema.json"
        ))
        .unwrap();
        for value in values.try_iter().unwrap() {
            let value = value.unwrap();
            for error in [
                scalar_i32(py, &value).unwrap_err(),
                scalar_u32(py, &value).unwrap_err(),
            ] {
                assert!(error.is_instance_of::<ValidationError>(py));
                let document: Value = serde_json::from_str(
                    &error
                        .value(py)
                        .getattr("diagnostic_json")
                        .unwrap()
                        .extract::<String>()
                        .unwrap(),
                )
                .unwrap();
                assert_eq!(document["diagnostics"][0]["code"], "TYPE_MISMATCH");
                assert!(jsonschema::is_valid(&schema, &document));
            }
        }
        assert!(
            !scope
                .get_item("called")
                .unwrap()
                .unwrap()
                .extract::<bool>()
                .unwrap()
        );
        for value in [i64::from(i32::MIN) - 1, i64::from(i32::MAX) + 1] {
            assert!(scalar_i32(py, value.into_pyobject(py).unwrap().as_any()).is_err());
        }
        for value in [-1_i64, i64::from(u32::MAX) + 1] {
            assert!(scalar_u32(py, value.into_pyobject(py).unwrap().as_any()).is_err());
        }
    });
}

// Inject dependency-boundary failures, including allocation/version failures
// whose real occurrence is host-dependent. This tests the actual error bridge,
// not C allocation-failure qualification or compiler-source reachability.
#[test]
fn issue_l1_06_python_binding_foreign_error_boundary_documents() {
    Python::attach(|py| {
        let schema: Value = serde_json::from_str(include_str!(
            "../../../schema/invar-diagnostics-v1.schema.json"
        ))
        .unwrap();
        for (external, expected) in [
            (invar_foreign::Error::Library, "PY_FOREIGN_LIBRARY"),
            (invar_foreign::Error::Version, "PY_FOREIGN_VERSION"),
            (invar_foreign::Error::Symbol, "PY_FOREIGN_SYMBOL"),
            (invar_foreign::Error::Status(7), "PY_FOREIGN_STATUS"),
            (invar_foreign::Error::Layout, "PY_FOREIGN_LAYOUT"),
            (invar_foreign::Error::Utf8, "INVALID_UTF8"),
            (invar_foreign::Error::Limit, "PY_BUFFER_LIMIT"),
            (invar_foreign::Error::Allocation, "PY_FOREIGN_ALLOCATION"),
        ] {
            let error = boundary_error(
                py,
                Class::Foreign,
                foreign_code(&external),
                &format!("C import failed: {external:?}"),
            );
            assert!(error.is_instance_of::<ForeignError>(py));
            let document: Value = serde_json::from_str(
                &error
                    .value(py)
                    .getattr("diagnostic_json")
                    .unwrap()
                    .extract::<String>()
                    .unwrap(),
            )
            .unwrap();
            assert_eq!(document["diagnostics"][0]["code"], expected);
            assert!(jsonschema::is_valid(&schema, &document));
            assert_eq!(document["diagnostics"][0]["suggested_fixes"], json!([]));
        }
    });
}

/// Boundary test for PY_STAGING_LIMIT (AI-01 registry): both the core's
/// staging refusal and the binding's f32 staging reservation failure map
/// to a schema-valid ResourceError. Neither is reachable with admissible
/// NumPy inputs (per-array limits keep staging under its bound, and
/// allocation failure cannot be forced from Python), so the mapping is
/// exercised directly.
#[test]
fn issue_l1_06_python_binding_reference_staging_limit_is_a_typed_resource_error() {
    Python::attach(|py| {
        let schema: Value = serde_json::from_str(include_str!(
            "../../../schema/invar-diagnostics-v1.schema.json"
        ))
        .unwrap();
        let core = invar::eval::EvalError::Tool {
            code: "HOST_STAGING_LIMIT",
            message: "host staging exceeds its byte bound".into(),
        };
        for error in [
            host_error(py, &core, "reference.inv", b""),
            staging_limit(py),
        ] {
            assert!(error.is_instance_of::<ResourceError>(py));
            let document: Value = serde_json::from_str(
                &error
                    .value(py)
                    .getattr("diagnostic_json")
                    .unwrap()
                    .extract::<String>()
                    .unwrap(),
            )
            .unwrap();
            assert_eq!(document["diagnostics"][0]["code"], "PY_STAGING_LIMIT");
            assert!(jsonschema::is_valid(&schema, &document));
            assert_eq!(document["diagnostics"][0]["suggested_fixes"], json!([]));
        }
    });
}

/// Map one core host code through `host_error` and check the typed,
/// schema-valid Python exception it becomes.
fn assert_host_mapping<E: pyo3::PyTypeInfo>(
    py: Python<'_>,
    core: &invar::eval::EvalError,
    expected: &str,
) {
    let schema: Value = serde_json::from_str(include_str!(
        "../../../schema/invar-diagnostics-v1.schema.json"
    ))
    .unwrap();
    let error = host_error(py, core, "reference.inv", b"");
    assert!(error.is_instance_of::<E>(py), "{core:?}");
    let document: Value = serde_json::from_str(
        &error
            .value(py)
            .getattr("diagnostic_json")
            .unwrap()
            .extract::<String>()
            .unwrap(),
    )
    .unwrap();
    assert_eq!(document["diagnostics"][0]["code"], expected, "{core:?}");
    assert!(jsonschema::is_valid(&schema, &document));
    assert_eq!(document["diagnostics"][0]["suggested_fixes"], json!([]));
}

fn core_tool(code: &'static str) -> invar::eval::EvalError {
    invar::eval::EvalError::Tool {
        code,
        message: "core host refusal".into(),
    }
}

/// AI-01 boundary test for core `HOST_ARRAY_ALIAS` (overlap involving a
/// mut buffer or of different element types; not constructible from
/// safe Rust slices or NumPy-owned arrays, so the mapping is tested).
#[test]
fn issue_l1_06_python_binding_reference_host_array_alias_maps_to_buffer_alias() {
    Python::attach(|py| {
        assert_host_mapping::<ValidationError>(
            py,
            &core_tool("HOST_ARRAY_ALIAS"),
            "PY_BUFFER_ALIAS",
        )
    });
}

/// AI-01 boundary test for core `HOST_ARRAY_EXTENT` (extent overflow or
/// misalignment; impossible for real slices, so the mapping is tested).
#[test]
fn issue_l1_06_python_binding_reference_host_array_extent_maps_to_buffer_limit() {
    Python::attach(|py| {
        assert_host_mapping::<ValidationError>(
            py,
            &core_tool("HOST_ARRAY_EXTENT"),
            "PY_BUFFER_LIMIT",
        )
    });
}

/// AI-01 boundary test for core `HOST_STAGING_LIMIT`.
#[test]
fn issue_l1_06_python_binding_reference_host_staging_limit_maps_to_staging_limit() {
    Python::attach(|py| {
        assert_host_mapping::<ResourceError>(
            py,
            &core_tool("HOST_STAGING_LIMIT"),
            "PY_STAGING_LIMIT",
        )
    });
}

/// AI-01 boundary test for core `HOST_TAKE_UNAVAILABLE`: emitted for real
/// by `observe_host_call` on a take signature, then mapped.
#[test]
fn issue_l1_06_python_binding_reference_host_take_unavailable_maps_to_take_unavailable() {
    Python::attach(|py| {
        let module = compile_source(
            py,
            PyString::new(py, "fn eat(xs: take Array<i32>) -> u32 { return len(xs); }").as_any(),
            "take.inv",
            "reference",
            None,
        )
        .unwrap();
        let mut xs = [1, 2];
        let mut args = [HostArgument::Mut(HostMutArray::I32(&mut xs))];
        let observation = observe_host_call(&module.checked, "eat", &mut args, 10);
        let Err(core) = &observation.result else {
            panic!("{:?}", observation.result)
        };
        assert_eq!((observation.steps_used, xs), (0, [1, 2]));
        assert!(matches!(
            core,
            invar::eval::EvalError::Tool {
                code: "HOST_TAKE_UNAVAILABLE",
                ..
            }
        ));
        assert_host_mapping::<ValidationError>(py, core, "PY_TAKE_UNAVAILABLE");
    });
}

mod bridge_faults;

#[test]
fn issue_l1_06_python_binding_device_failures_preserve_codes_without_replay() {
    Python::attach(|py| {
        let schema: Value = serde_json::from_str(include_str!(
            "../../../schema/invar-diagnostics-v1.schema.json"
        ))
        .unwrap();
        for code in [
            "CUDA_UNAVAILABLE",
            "METAL_UNAVAILABLE",
            "CUDA_PENDING",
            "METAL_PENDING",
            "DEVICE_TOOLCHAIN_MISMATCH",
            "CPU_WORKERS",
            "CPU_WORKER_PANIC",
        ] {
            let error = engines::execution_error(py, core_tool(code), "engines.inv", b"");
            assert!(error.is_instance_of::<EngineError>(py));
            let document: Value = serde_json::from_str(
                &error
                    .value(py)
                    .getattr("diagnostic_json")
                    .unwrap()
                    .extract::<String>()
                    .unwrap(),
            )
            .unwrap();
            assert_eq!(document["diagnostics"][0]["code"], code);
            assert!(jsonschema::is_valid(&schema, &document));
        }
    });
}
