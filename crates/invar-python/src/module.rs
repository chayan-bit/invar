//! Python-visible methods of `CheckedModule`.
use super::engines::{Workers, execution_error};
use super::*;
use invar::engine::Engine;
use std::path::PathBuf;

/// A compiled-ahead Metal library, as the CLI's `--artifact` takes it: a file
/// path (`str` or `os.PathLike`) or the container itself (`bytes` or
/// `bytearray`). Other types, including bytes paths and `memoryview`, are
/// refused, never guessed at.
pub(crate) enum MetalArtifact {
    Path(PathBuf),
    Bytes(Vec<u8>),
}

impl MetalArtifact {
    /// The decoded, validated library, with the CLI's refusal on failure.
    fn decode(&self) -> Result<invar::metal_aot::Image, invar::eval::EvalError> {
        match self {
            Self::Path(path) => invar::engine::metal_image_from_path(path),
            Self::Bytes(bytes) => invar::engine::metal_image_from_bytes(bytes),
        }
    }
}

impl<'a, 'py> FromPyObject<'a, 'py> for MetalArtifact {
    type Error = PyErr;

    fn extract(value: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
        let py = value.py();
        if let Ok(bytes) = value.cast::<PyBytes>() {
            return Ok(Self::Bytes(bytes.as_bytes().to_vec()));
        }
        if let Ok(bytes) = value.cast::<PyByteArray>() {
            return Ok(Self::Bytes(bytes.to_vec()));
        }
        let invalid = || {
            boundary_error(
                py,
                Class::Validation,
                "TYPE_MISMATCH",
                "artifact requires a str or os.PathLike file path, or bytes or bytearray holding a compiled-ahead Metal library",
            )
        };
        let path = py
            .import("os")?
            .call_method1("fspath", (&*value,))
            .map_err(|_| invalid())?;
        let path = path.cast::<PyString>().map_err(|_| invalid())?;
        Ok(Self::Path(PathBuf::from(
            path.to_str().map_err(|_| invalid())?,
        )))
    }
}

#[pymethods]
impl CheckedModule {
    /// Run main with the selected engine. The receipt describes this completed
    /// invocation, never a predicted engine or a replay during explain.
    /// `artifact` names a compiled-ahead Metal library for `metal` and `auto`
    /// (a path or the container's bytes), decoded with the CLI's validation.
    #[pyo3(signature = (*, fuel=Fuel(1_000_000), workers=None, observe=false, artifact=None))]
    pub(crate) fn run_main(
        &mut self,
        py: Python<'_>,
        fuel: Fuel,
        workers: Option<Workers>,
        observe: bool,
        artifact: Option<MetalArtifact>,
    ) -> PyResult<Py<PyAny>> {
        // Argument refusals run no invocation, so the previous receipt stays
        // the receipt of the last completed run; execution failures clear it.
        let engine = Engine::parse(self.engine).ok_or_else(|| {
            boundary_error(py, Class::Engine, "PY_ENGINE_UNAVAILABLE", "unknown engine")
        })?;
        if workers.is_some() && engine != Engine::CpuParallel {
            return Err(boundary_error(
                py,
                Class::Engine,
                "CPU_WORKERS",
                "workers is only accepted for cpu-parallel",
            ));
        }
        if artifact.is_some() && !matches!(engine, Engine::Metal | Engine::Auto) {
            return Err(boundary_error(
                py,
                Class::Engine,
                "PY_ENGINE_UNAVAILABLE",
                "artifact is only accepted for metal and auto",
            ));
        }
        // As the CLI does, a build without Metal refuses the library before any backend is chosen.
        if artifact.is_some() && !cfg!(all(feature = "metal", target_os = "macos")) {
            return Err(boundary_error(
                py,
                Class::Engine,
                "METAL_UNAVAILABLE",
                "this invar build has no Metal support, so artifact cannot be honored",
            ));
        }
        self.last_execution = None;
        let checked = invar::front::parse(&self.source)
            .and_then(|program| invar::check::check(program, true))
            .map_err(|error| {
                exception(
                    py,
                    Class::Check,
                    error.to_diagnostics_json(&self.path, &self.source),
                )
            })?;
        invar::ownership::check(&checked).map_err(|error| {
            exception(
                py,
                Class::Check,
                error.to_diagnostics_json(&self.path, &self.source),
            )
        })?;
        let receipt = if engine == Engine::Reference {
            let outcome = invar::eval::run(&checked, fuel.0)
                .map_err(|error| execution_error(py, error, &self.path, &self.source))?;
            let invar::eval::Value::I32(value) = outcome.value else {
                return Err(boundary_error(
                    py,
                    Class::Check,
                    "ENTRYPOINT_SIGNATURE",
                    "entrypoint result must be i32",
                ));
            };
            json!({"requested_engine": self.engine, "backends": [self.engine], "result": value})
        } else {
            let program = invar::mir::lower(&checked)
                .map_err(|error| boundary_error(py, Class::Engine, error.code, &error.message))?;
            if engine == Engine::NativeSerial {
                let (status, document) =
                    invar::engine::run_native(&program, fuel.0, &self.path, &self.source, None)
                        .map_err(|error| execution_error(py, error, &self.path, &self.source))?;
                if status != 0 {
                    let class = match document["status"].as_str() {
                        Some("trap") => Class::Trap,
                        Some("evaluation_limit") => Class::Resource,
                        _ => Class::Engine,
                    };
                    return Err(exception(py, class, document));
                }
                json!({"requested_engine": self.engine, "backends": [self.engine], "result": document["result"]["value"]})
            } else {
                let image = artifact
                    .as_ref()
                    .map(MetalArtifact::decode)
                    .transpose()
                    .map_err(|error| execution_error(py, error, &self.path, &self.source))?;
                let receipt = invar::engine::run_mir(
                    &self.source,
                    &program,
                    engine,
                    fuel.0,
                    workers.map_or(1, |count| count.0),
                    image.as_ref(),
                )
                .map_err(|error| execution_error(py, error, &self.path, &self.source))?;
                // The CLI CPU scheduler reports a serial guard path explicitly.
                // Python's forced engine contract refuses a wholly serial result.
                if engine == Engine::CpuParallel
                    && let Some(reason) = receipt["fallback_reason"].as_str()
                {
                    return Err(boundary_error(
                        py,
                        Class::Engine,
                        "PY_ENGINE_UNAVAILABLE",
                        &format!(
                            "cpu-parallel published no parallel region: {reason}; use auto for serial fallback"
                        ),
                    ));
                }
                receipt
            }
        };
        if receipt.to_string().len() > 1_000_000 {
            return Err(boundary_error(
                py,
                Class::Resource,
                "DIAGNOSTIC_DOCUMENT_LIMIT",
                "execution receipt exceeds one million bytes",
            ));
        }
        let value = if observe {
            object(py, &receipt)?
        } else {
            object(py, &receipt["result"])?
        };
        self.last_execution = Some(receipt);
        Ok(value)
    }

    /// Receipt from the last successful run_main; cleared when execution fails.
    pub(crate) fn receipt(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        self.last_execution
            .as_ref()
            .map_or_else(|| Ok(py.None()), |receipt| object(py, receipt))
    }
    /// Source signatures, with explicit export/callability metadata.
    pub(crate) fn functions(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        object(py, &self.signatures)
    }

    /// Selected, artifact-bound source descriptors in the requested order.
    pub(crate) fn exports(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let descriptors = self.artifact.as_ref().map_or_else(Vec::new, |artifact| {
            artifact.signatures().iter().map(|signature| json!({
                "name": signature.source_name,
                "symbol": signature.symbol,
                "span": {"start": signature.span.start, "end": signature.span.end},
                "parameters": signature.params.iter().map(|parameter| match parameter.array {
                    None => json!({
                        "name": parameter.name, "type": scalar_type_name(parameter.ty), "mode": "value",
                    }),
                    Some(mode) => json!({
                        "name": parameter.name,
                        "type": format!("Array<{}>", scalar_type_name(parameter.ty)),
                        "mode": if mode == invar::exports::ArrayMode::Mut { "mut" } else { "read" },
                        "zero_copy": self.engine == "native-serial",
                    }),
                }).collect::<Vec<_>>(),
                "result": scalar_type_name(signature.result),
                "artifact_id": signature.artifact_id,
                "engine": self.engine, "exported": true, "callable": true,
            })).collect()
        });
        object(py, &json!(descriptors))
    }

    #[getter]
    pub(crate) fn artifact_id(&self) -> Option<&str> {
        self.artifact.as_ref().map(CompiledExports::artifact_id)
    }

    #[getter]
    pub(crate) fn engine(&self) -> &'static str {
        self.engine
    }

    /// Call a selected export. Functions with `read`/`mut` `Array<i32|u32>`
    /// parameters take NumPy arrays (HA-1.2); `observe=True` returns
    /// `{"engine", "copying", "result", "record_addresses"}` for them.
    #[pyo3(signature = (name, *args, fuel=Fuel(1_000_000), observe=false))]
    pub(crate) fn call(
        &mut self,
        py: Python<'_>,
        name: &str,
        args: &Bound<'_, PyTuple>,
        fuel: Fuel,
        observe: bool,
    ) -> PyResult<Py<PyAny>> {
        let function = self
            .checked
            .program
            .functions
            .iter()
            .find(|function| function.name == name)
            .ok_or_else(|| {
                boundary_error(
                    py,
                    Class::Validation,
                    "NAME_UNKNOWN",
                    "unknown function name",
                )
            })?;
        if args.len() != function.params.len() {
            return Err(boundary_error(
                py,
                Class::Validation,
                "PY_ARGUMENT_COUNT",
                "wrong argument count",
            ));
        }
        if function
            .params
            .iter()
            .any(|param| array_dtype(&param.ty).is_some())
        {
            return self.array_call(py, name, args, fuel, observe);
        }
        for (param, value) in function.params.iter().zip(args.iter()) {
            match param.ty {
                Type::I32 => {
                    scalar_i32(py, &value)?;
                }
                Type::U32 => {
                    scalar_u32(py, &value)?;
                }
                _ => {}
            }
        }
        let unavailable = || {
            boundary_error(
                py,
                Class::Engine,
                "PY_EXPORT_UNAVAILABLE",
                "function was not selected for this module's scalar export artifact",
            )
        };
        let artifact = self.artifact.as_mut().ok_or_else(unavailable)?;
        let signature = artifact
            .signatures()
            .iter()
            .find(|signature| signature.source_name == name)
            .ok_or_else(unavailable)?;
        // Conversion uses the artifact's exact typed inventory. All arguments
        // become owned scalars before invoke, with no Python callbacks or borrows.
        let values = signature
            .params
            .iter()
            .zip(args.iter())
            .map(|(parameter, value)| match parameter.ty {
                ScalarType::I32 => scalar_i32(py, &value).map(ScalarValue::I32),
                ScalarType::U32 => scalar_u32(py, &value).map(ScalarValue::U32),
                ScalarType::F32 | ScalarType::F64 | ScalarType::Bool => {
                    Err(invalid(py, Failure::Dtype))
                }
            })
            .collect::<PyResult<Vec<_>>>()?;
        let value = artifact
            .invoke(name, &values, fuel.0)
            .map_err(|error| export_exception(py, &error, artifact.diagnostics(&error)))?;
        match value {
            ScalarValue::I32(value) => Ok(value.into_pyobject(py)?.into_any().unbind()),
            ScalarValue::U32(value) => Ok(value.into_pyobject(py)?.into_any().unbind()),
        }
    }

    /// Use the existing AI-01 analysis; do not synthesize a device execution report.
    pub(crate) fn explain(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let lowered = invar::mir::lower(&self.checked)
            .map_err(|error| boundary_error(py, Class::Check, error.code, &error.message))?;
        let report = invar::admission::analyze(&self.source, &lowered)
            .map_err(|error| boundary_error(py, Class::Check, error.code, &error.message))?;
        let mut document = report.to_json();
        document["schema"] = json!(invar::admission::SCHEMA);
        if let Some(receipt) = &self.last_execution
            && matches!(self.engine, "auto" | "cuda" | "metal")
        {
            document["device_execution"] = invar::engine::device_document(receipt);
            document["device_schema"] = json!(invar::admission::DEVICE_SCHEMA);
        }
        if document.to_string().len() > 1_000_000 {
            return Err(boundary_error(
                py,
                Class::Resource,
                "DIAGNOSTIC_DOCUMENT_LIMIT",
                "explain document exceeds one million bytes",
            ));
        }
        object(py, &document)
    }

    /// Call a checked extern C import, not a compiled source function. The GIL
    /// stays held; ndarray references and read guards remain live through decoding.
    /// The selected library must obey L1-04 (no callbacks, retention or input writes).
    #[pyo3(signature = (name, library, *args))]
    pub(crate) fn call_foreign(
        &self,
        py: Python<'_>,
        name: &str,
        library: &str,
        args: &Bound<'_, PyTuple>,
    ) -> PyResult<Py<PyAny>> {
        let function = self
            .checked
            .program
            .foreign_functions
            .iter()
            .find(|function| function.name == name)
            .ok_or_else(|| {
                boundary_error(
                    py,
                    Class::Validation,
                    "NAME_UNKNOWN",
                    "unknown extern C import",
                )
            })?;
        if args.len() != function.params.len() {
            return Err(boundary_error(
                py,
                Class::Validation,
                "PY_ARGUMENT_COUNT",
                "wrong argument count",
            ));
        }
        if !Path::new(library).is_absolute() || !Path::new(library).is_file() {
            return Err(boundary_error(
                py,
                Class::Validation,
                "PY_LIBRARY_PATH",
                "library must be an explicit absolute file path",
            ));
        }
        // Complete validation for every argument before borrowing any element slice.
        let values = args.iter().collect::<Vec<_>>();
        for (param, value) in function.params.iter().zip(&values) {
            if param.mode == Some(Mode::Take) {
                return Err(boundary_error(
                    py,
                    Class::Validation,
                    "PY_TAKE_UNAVAILABLE",
                    "Python buffers cannot be consumed by the current binding; take requires an explicit transferable owner",
                ));
            }
            match &param.ty {
                Type::I32 => {
                    scalar_i32(py, value)?;
                }
                Type::U32 => {
                    scalar_u32(py, value)?;
                }
                Type::Named(name) if name == "Bytes" => {
                    if let Ok(bytes) = value.cast::<PyBytes>() {
                        if bytes.as_bytes().len() > invar_foreign::MAX_BYTES {
                            return Err(invalid(py, Failure::Limit));
                        }
                    } else {
                        validation::validate(metadata(py, value)?, Dtype::U8, None, false)
                            .map_err(|failure| invalid(py, failure))?;
                    }
                }
                Type::Named(name) if name == "String" => {
                    let text = value.cast::<PyString>().map_err(|_| {
                        boundary_error(
                            py,
                            Class::Validation,
                            "TYPE_MISMATCH",
                            "String requires str",
                        )
                    })?;
                    let text = text.to_str().map_err(|_| {
                        boundary_error(
                            py,
                            Class::Validation,
                            "INVALID_UTF8",
                            "String must encode as UTF-8",
                        )
                    })?;
                    if text.len() > invar_foreign::MAX_BYTES {
                        return Err(invalid(py, Failure::Limit));
                    }
                }
                _ => {
                    return Err(boundary_error(
                        py,
                        Class::Engine,
                        "PY_ABI_TYPE_UNAVAILABLE",
                        "type is outside the current C ABI",
                    ));
                }
            }
        }
        let mut guards = Vec::new();
        for (param, value) in function.params.iter().zip(&values) {
            if matches!(&param.ty, Type::Named(name) if name == "Bytes")
                && value.cast::<PyBytes>().is_err()
            {
                let array = value
                    .cast::<PyArrayDyn<u8>>()
                    .map_err(|_| invalid(py, Failure::Dtype))?;
                guards.push(
                    array
                        .try_readonly()
                        .map_err(|_| invalid(py, Failure::Alias))?,
                );
            }
        }
        // Preserve the allocation address even for zero-length NumPy arrays.
        // rust-numpy 0.29 normalizes its empty slice to &[], whereas the
        // guarded ndarray view retains the original pointer without copying.
        let views: Vec<_> = guards.iter().map(|guard| guard.as_array()).collect();
        let mut views = views.iter();
        let mut wire = Vec::new();
        for (param, value) in function.params.iter().zip(&values) {
            let arg = match &param.ty {
                Type::I32 => invar_foreign::Arg::I32(scalar_i32(py, value)?),
                Type::U32 => invar_foreign::Arg::U32(scalar_u32(py, value)?),
                Type::Named(name) if name == "Bytes" => {
                    let bytes = if let Ok(bytes) = value.cast::<PyBytes>() {
                        bytes.as_bytes()
                    } else {
                        views
                            .next()
                            .ok_or_else(|| invalid(py, Failure::Ownership))?
                            .as_slice()
                            .ok_or_else(|| invalid(py, Failure::Contiguity))?
                    };
                    invar_foreign::Arg::Bytes(bytes, invar_foreign::Mode::Read)
                }
                Type::Named(name) if name == "String" => invar_foreign::Arg::String(
                    value.cast::<PyString>()?.to_str()?,
                    invar_foreign::Mode::Read,
                ),
                _ => {
                    return Err(boundary_error(
                        py,
                        Class::Engine,
                        "PY_ABI_TYPE_UNAVAILABLE",
                        "type is outside the current C ABI",
                    ));
                }
            };
            wire.push(arg);
        }
        let expected = match type_name(&function.result) {
            "Result<i32,ForeignError>" => invar_foreign::I32,
            "Result<u32,ForeignError>" => invar_foreign::U32,
            "Result<Bytes,ForeignError>" => invar_foreign::BYTES,
            "Result<String,ForeignError>" => invar_foreign::STRING,
            _ => {
                return Err(boundary_error(
                    py,
                    Class::Engine,
                    "PY_ABI_TYPE_UNAVAILABLE",
                    "return type is outside the current C ABI",
                ));
            }
        };
        let output =
            invar_foreign::call(Path::new(library), name, &wire, expected).map_err(|error| {
                boundary_error(
                    py,
                    Class::Foreign,
                    foreign_code(&error),
                    &format!("C import failed: {error:?}"),
                )
            })?;
        match output {
            invar_foreign::Output::I32(value) => Ok(value.into_pyobject(py)?.into_any().unbind()),
            invar_foreign::Output::U32(value) => Ok(value.into_pyobject(py)?.into_any().unbind()),
            invar_foreign::Output::Bytes(value) => Ok(PyBytes::new(py, &value).into_any().unbind()),
            invar_foreign::Output::String(value) => {
                Ok(PyString::new(py, &value).into_any().unbind())
            }
        }
    }

    /// Explicit reference-engine call with NumPy array arguments (HA-1.1).
    /// It never uses or falls back to a native path. Every argument is
    /// validated, overlaps and staging are checked on the original metadata,
    /// and borrow guards are taken before any source step; the GIL and the
    /// guards' strong references are held for the whole call and no Python
    /// code runs during evaluation. Caller arrays are written only after the
    /// Python result was built and every writeback was prechecked.
    #[pyo3(signature = (name, *args, fuel=Fuel(1_000_000)))]
    pub(crate) fn call_reference(
        &self,
        py: Python<'_>,
        name: &str,
        args: &Bound<'_, PyTuple>,
        fuel: Fuel,
    ) -> PyResult<Py<PyAny>> {
        self.reference_call(py, name, args, fuel, BindingInject::None)
    }
}
