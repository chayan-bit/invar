//! Python boundary for selected scalar artifacts and the existing import ABI.
#![forbid(unsafe_code)]

use std::path::Path;

use invar::{
    check::CheckedProgram,
    eval::host_read::{HostArgument, HostMutArray, HostReadArray, observe_host_call},
    exports::{CompiledExports, ExportEngine, ExportError, ScalarType, ScalarValue},
    front::{ErrorKind, Mode, Type},
    numeric_contract::{F32Bits, F64Bits},
};
use numpy::{
    BorrowError, PyArray1, PyArrayDyn, PyArrayMethods, PyReadonlyArrayDyn, PyReadwriteArrayDyn,
    PyUntypedArrayMethods,
};
use pyo3::{
    create_exception,
    exceptions::PyException,
    prelude::*,
    types::{PyByteArray, PyBytes, PyDict, PyFloat, PyInt, PyList, PyString, PyTuple},
};
use serde_json::{Value, json};

pub mod validation;
use validation::{Dtype, Failure, Metadata};

mod array_call;
mod boolview;
mod bridge;
mod calls;
mod convert;
mod engines;
mod module;
use boolview::check_bool_bytes;
use bridge::*;
use calls::*;
use convert::*;

// Parse integer controls at the Python boundary rather than letting PyO3's
// primitive conversions emit unstructured OverflowError/TypeError exceptions.
struct Fuel(u64);

impl<'a, 'py> FromPyObject<'a, 'py> for Fuel {
    type Error = PyErr;

    fn extract(value: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
        let py = value.py();
        let invalid = || {
            boundary_error(
                py,
                Class::Resource,
                "INVALID_FUEL",
                "fuel must be an exact built-in integer in 1..=10000000",
            )
        };
        if !value.get_type().is(py.get_type::<PyInt>()) {
            return Err(invalid());
        }
        let fuel = value.extract::<u64>().map_err(|_| invalid())?;
        if fuel == 0 || fuel > 10_000_000 {
            return Err(invalid());
        }
        Ok(Self(fuel))
    }
}

struct ArrayLength(usize);

impl<'a, 'py> FromPyObject<'a, 'py> for ArrayLength {
    type Error = PyErr;

    fn extract(value: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
        let py = value.py();
        if !value.get_type().is(py.get_type::<PyInt>()) {
            return Err(invalid(py, Failure::Shape));
        }
        value
            .extract::<usize>()
            .map(Self)
            .map_err(|_| invalid(py, Failure::Shape))
    }
}

create_exception!(invar_python, InvarError, PyException);
create_exception!(invar_python, CompileError, InvarError);
create_exception!(invar_python, CheckError, InvarError);
create_exception!(invar_python, TrapError, InvarError);
create_exception!(invar_python, ValidationError, InvarError);
create_exception!(invar_python, ResourceError, InvarError);
create_exception!(invar_python, EngineError, InvarError);
create_exception!(invar_python, ForeignError, InvarError);

enum Class {
    Compile,
    Check,
    Trap,
    Validation,
    Resource,
    Engine,
    Foreign,
}

fn object(py: Python<'_>, value: &Value) -> PyResult<Py<PyAny>> {
    Ok(match value {
        Value::Null => py.None(),
        Value::Bool(value) => value.into_pyobject(py)?.to_owned().into_any().unbind(),
        Value::Number(value) => {
            if let Some(value) = value.as_i64() {
                value.into_pyobject(py)?.into_any().unbind()
            } else if let Some(value) = value.as_u64() {
                value.into_pyobject(py)?.into_any().unbind()
            } else {
                value
                    .as_f64()
                    .unwrap_or(0.0)
                    .into_pyobject(py)?
                    .into_any()
                    .unbind()
            }
        }
        Value::String(value) => PyString::new(py, value).into_any().unbind(),
        Value::Array(values) => {
            let list = PyList::empty(py);
            for value in values {
                list.append(object(py, value)?)?;
            }
            list.into_any().unbind()
        }
        Value::Object(values) => {
            let dict = PyDict::new(py);
            for (key, value) in values {
                dict.set_item(key, object(py, value)?)?;
            }
            dict.into_any().unbind()
        }
    })
}

fn exception(py: Python<'_>, class: Class, document: Value) -> PyErr {
    let message = document["diagnostics"][0]["message"]
        .as_str()
        .unwrap_or("Invar error")
        .to_owned();
    let error = match class {
        Class::Compile => CompileError::new_err(message),
        Class::Check => CheckError::new_err(message),
        Class::Trap => TrapError::new_err(message),
        Class::Validation => ValidationError::new_err(message),
        Class::Resource => ResourceError::new_err(message),
        Class::Engine => EngineError::new_err(message),
        Class::Foreign => ForeignError::new_err(message),
    };
    let attributes = || -> PyResult<()> {
        error
            .value(py)
            .setattr("diagnostics", object(py, &document)?)?;
        error
            .value(py)
            .setattr("diagnostic_json", document.to_string())
    };
    if let Err(error) = attributes() {
        return error;
    }
    error
}

fn boundary_error(py: Python<'_>, class: Class, code: &str, message: &str) -> PyErr {
    exception(
        py,
        class,
        invar::diagnostics::document(
            "tool_error",
            vec![invar::diagnostics::diagnostic(
                code,
                message,
                "<python>",
                &[],
                None,
            )],
        ),
    )
}

fn invalid(py: Python<'_>, failure: Failure) -> PyErr {
    boundary_error(
        py,
        Class::Validation,
        failure.code(),
        &format!("array boundary rejected: {failure:?}"),
    )
}

/// Whether `call_foreign` can invoke a C import with these parameters: its argument validation refuses `take`
/// (PY_TAKE_UNAVAILABLE) and every type but exact i32, u32, Bytes and String (PY_ABI_TYPE_UNAVAILABLE).
fn foreign_callable(params: &[invar::front::Param]) -> bool {
    params.iter().all(|param| {
        param.mode != Some(Mode::Take)
            && match &param.ty {
                Type::I32 | Type::U32 => true,
                Type::Named(name) => name == "Bytes" || name == "String",
                _ => false,
            }
    })
}

fn parameters(params: &[invar::front::Param]) -> Value {
    json!(params.iter().map(|param| json!({
        "name": param.name, "type": param.ty.source_name(),
        "mode": match param.mode { None => "value", Some(Mode::Read) => "read", Some(Mode::Mut) => "mut", Some(Mode::Take) => "take" }
    })).collect::<Vec<_>>())
}

fn metadata_typed<T: numpy::Element>(
    array: &Bound<'_, PyArrayDyn<T>>,
    dtype: Dtype,
) -> PyResult<Metadata> {
    let flags = array.getattr("flags")?;
    Ok(Metadata {
        dtype,
        rank: array.ndim(),
        length: array.len(),
        contiguous: array.is_c_contiguous(),
        aligned: flags.getattr("aligned")?.extract()?,
        writable: flags.getattr("writeable")?.extract()?,
        owns_data: flags.getattr("owndata")?.extract()?,
        address: array.data() as usize,
    })
}

fn metadata(py: Python<'_>, value: &Bound<'_, PyAny>) -> PyResult<Metadata> {
    // Exact ndarray type excludes Python subclasses with overriding properties.
    if !value.get_type().is(py.get_type::<PyArrayDyn<u8>>()) {
        return Err(boundary_error(
            py,
            Class::Validation,
            "PY_ARRAY_DTYPE",
            "expected an exact NumPy ndarray",
        ));
    }
    if let Ok(array) = value.cast::<PyArrayDyn<u8>>() {
        metadata_typed(array, Dtype::U8)
    } else if let Ok(array) = value.cast::<PyArrayDyn<i32>>() {
        metadata_typed(array, Dtype::I32)
    } else if let Ok(array) = value.cast::<PyArrayDyn<u32>>() {
        metadata_typed(array, Dtype::U32)
    } else if let Ok(array) = value.cast::<PyArrayDyn<f32>>() {
        metadata_typed(array, Dtype::F32)
    } else if let Ok(array) = value.cast::<PyArrayDyn<f64>>() {
        metadata_typed(array, Dtype::F64)
    } else if let Ok(array) = value.cast::<PyArrayDyn<bool>>() {
        metadata_typed(array, Dtype::Bool)
    } else {
        Err(invalid(py, Failure::Dtype))
    }
}

/// Validate numeric buffers without claiming that the current C ABI accepts them.
/// The returned address is observational metadata, not a retained pin or a handle.
#[pyfunction]
#[pyo3(signature = (array, dtype, *, length=None, output=false))]
fn validate_array(
    py: Python<'_>,
    array: &Bound<'_, PyAny>,
    dtype: &str,
    length: Option<ArrayLength>,
    output: bool,
) -> PyResult<Py<PyAny>> {
    let expected = Dtype::parse(dtype).ok_or_else(|| invalid(py, Failure::Dtype))?;
    let meta = metadata(py, array)?;
    validation::validate(meta, expected, length.map(|length| length.0), output)
        .map_err(|failure| invalid(py, failure))?;
    if expected == Dtype::Bool {
        check_bool_bytes(py, array)?;
    }
    object(
        py,
        &json!({"address":meta.address,"length":meta.length,"itemsize":meta.dtype.size(),"dtype":dtype,"writable":meta.writable,"owns_data":meta.owns_data}),
    )
}

/// Validate an input/output pair before any element references or foreign call.
#[pyfunction]
#[pyo3(signature = (input, output, dtype))]
fn validate_buffers(
    py: Python<'_>,
    input: &Bound<'_, PyAny>,
    output: &Bound<'_, PyAny>,
    dtype: &str,
) -> PyResult<()> {
    let expected = Dtype::parse(dtype).ok_or_else(|| invalid(py, Failure::Dtype))?;
    let (input_value, output_value) = (input, output);
    let input = metadata(py, input_value)?;
    let output = metadata(py, output_value)?;
    validation::validate(input, expected, None, false).map_err(|failure| invalid(py, failure))?;
    validation::validate(output, expected, Some(input.length), true)
        .map_err(|failure| invalid(py, failure))?;
    if expected == Dtype::Bool {
        check_bool_bytes(py, input_value)?;
        check_bool_bytes(py, output_value)?;
    }
    if validation::overlaps(input, output).map_err(|failure| invalid(py, failure))? {
        return Err(invalid(py, Failure::Alias));
    }
    Ok(())
}

// The artifact's native loader is deliberately !Send/!Sync. PyO3 enforces
// same-thread use and its mutable borrow serializes invoke; the GIL stays held.
#[pyclass(unsendable, module = "invar_python")]
struct CheckedModule {
    source: Vec<u8>,
    path: String,
    checked: CheckedProgram,
    signatures: Value,
    artifact: Option<CompiledExports>,
    engine: &'static str,
    last_execution: Option<Value>,
}

#[pyfunction]
#[pyo3(signature = (source, *, path="<python>", engine="reference", exports=None))]
fn compile_source(
    py: Python<'_>,
    source: &Bound<'_, PyAny>,
    path: &str,
    engine: &str,
    exports: Option<&Bound<'_, PyAny>>,
) -> PyResult<CheckedModule> {
    let path = path.chars().take(4096).collect::<String>();
    let selected = invar::engine::Engine::parse(engine).ok_or_else(|| {
        boundary_error(
            py,
            Class::Engine,
            "PY_ENGINE_UNAVAILABLE",
            "unknown engine; use reference, native-serial, cpu-parallel, auto, cuda or metal",
        )
    })?;
    let engine = selected.name();
    let export_engine = match selected {
        invar::engine::Engine::NativeSerial => ExportEngine::NativeSerial,
        invar::engine::Engine::Reference => ExportEngine::Reference,
        _ if exports.is_some() => {
            return Err(boundary_error(
                py,
                Class::Engine,
                "NATIVE_ABI",
                "selected host exports support reference and native-serial only (HA-1.2); scheduled engines require run_main",
            ));
        }
        _ => ExportEngine::Reference, // No export artifact is constructed.
    };
    let requested = exports
        .map(|selection| export_names(py, selection))
        .transpose()?;
    let limited = || {
        boundary_error(
            py,
            Class::Resource,
            "SOURCE_SIZE_LIMIT",
            "source exceeds one MiB",
        )
    };
    let source = if let Ok(source) = source.cast::<PyBytes>() {
        if source.as_bytes().len() > 1_048_576 {
            return Err(limited());
        }
        source.as_bytes().to_vec()
    } else if let Ok(source) = source.cast::<PyString>() {
        if source.len()? > 1_048_576 {
            return Err(limited());
        }
        let source = source.to_str().map_err(|_| {
            boundary_error(
                py,
                Class::Compile,
                "INVALID_UTF8",
                "source must encode as UTF-8",
            )
        })?;
        if source.len() > 1_048_576 {
            return Err(limited());
        }
        source.as_bytes().to_vec()
    } else {
        return Err(boundary_error(
            py,
            Class::Validation,
            "TYPE_MISMATCH",
            "source requires str or bytes",
        ));
    };
    let program = invar::front::parse(&source).map_err(|error| {
        let class = if error.kind == ErrorKind::Tool {
            Class::Resource
        } else {
            Class::Compile
        };
        exception(py, class, error.to_diagnostics_json(&path, &source))
    })?;
    let mut signatures = json!({
        "source_functions": program.functions.iter().map(|function| json!({"name":function.name,"parameters":parameters(&function.params),"result":function.result.source_name(),"exported":false,"callable":false})).collect::<Vec<_>>(),
        "c_imports": program.foreign_functions.iter().map(|function| json!({"name":function.name,"parameters":parameters(&function.params),"result":function.result.source_name(),"exported":false,"callable":foreign_callable(&function.params)})).collect::<Vec<_>>()
    });
    let checked = invar::check::check(program, false)
        .map_err(|error| exception(py, Class::Check, error.to_diagnostics_json(&path, &source)))?;
    invar::ownership::check(&checked)
        .map_err(|error| exception(py, Class::Check, error.to_diagnostics_json(&path, &source)))?;
    let artifact = if let Some(requested) = requested {
        let names = requested.iter().map(String::as_str).collect::<Vec<_>>();
        Some(
            invar::exports::compile_exports(&source, &path, &names, export_engine).map_err(
                |error| export_exception(py, &error, error.to_diagnostics_json(&path, &source)),
            )?,
        )
    } else {
        None
    };
    if let (Some(artifact), Some(functions)) = (
        artifact.as_ref(),
        signatures["source_functions"].as_array_mut(),
    ) {
        for function in functions {
            let selected = artifact
                .signatures()
                .iter()
                .any(|signature| function["name"].as_str() == Some(signature.source_name.as_str()));
            function["exported"] = json!(selected);
            function["callable"] = json!(selected);
        }
    }
    Ok(CheckedModule {
        source,
        path,
        checked,
        signatures,
        artifact,
        engine,
        last_execution: None,
    })
}

#[pyfunction]
fn capabilities(py: Python<'_>) -> PyResult<Py<PyAny>> {
    object(
        py,
        &json!({
            "scalar_artifact_exports":["i32","u32"],
            "compiled_c_exports":"native-serial requires invar/native at build time",
            "export_engines":["reference","native-serial"],
            "export_selection":"explicit exports list or tuple",
            "numeric_array_abi":false, "f32_source_arithmetic":true,
            "c_import_types":["i32","u32","read Bytes","read String"],
            "c_output_buffers":"validated copy followed by C release (L1-04)",
            "numpy_validation_types":["uint8","int32","uint32","float32","float64","bool"],
            "entrypoint_engines":["reference","native-serial","cpu-parallel","auto","cuda","metal"],
            "engine_features":{"native":cfg!(feature="native"),"cuda":cfg!(feature="cuda"),"metal":cfg!(all(feature="metal", target_os="macos"))},
            "metal_artifact":"run_main(artifact=path) for metal and auto; needs engine_features.metal",
            "scheduled_host_arrays":false,"gil_released":false,"external_array_owners":false,
            "reference_array_entry":"copying",
            "take_buffers":false,
            "caller_output_buffers":"mut Array<i32|u32|f32|f64|Bool> export parameters",
            "array_exports":["read Array<i32>","read Array<u32>","mut Array<i32>","mut Array<u32>","read Array<f32>","mut Array<f32>","read Array<f64>","mut Array<f64>","read Array<Bool>","mut Array<Bool>"],
            "array_export_paths":{"native-serial":"zero-copy at the NumPy address, rollback on failure","reference":"copying"},
            "f32_arrays":"native-serial zero-copy read/mut (HA-1.3); reference paths copy",
            "f64_bool_arrays":"native-serial zero-copy read/mut (HA-1.4); scalar f64 and returned arrays stay reference-only",
            "reference_v02_types":["f64","read Array<f64>","mut Array<f64>","read Array<Bool>","mut Array<Bool>"],
            "bool_arrays":"NumPy bool_ with every byte 0 or 1 (PY_ARRAY_BOOL_VALUE otherwise)",
            "returned_arrays":"copying reference path only (call_reference)",
            "acceptance":"HA-1.4 f64 and Bool arrays"
        }),
    )
}

#[pymodule]
fn invar_python(module: &Bound<'_, PyModule>) -> PyResult<()> {
    let py = module.py();
    module.add_class::<CheckedModule>()?;
    module.add_function(wrap_pyfunction!(compile_source, module)?)?;
    module.add_function(wrap_pyfunction!(validate_array, module)?)?;
    module.add_function(wrap_pyfunction!(validate_buffers, module)?)?;
    module.add_function(wrap_pyfunction!(capabilities, module)?)?;
    module.add("InvarError", py.get_type::<InvarError>())?;
    module.add("CompileError", py.get_type::<CompileError>())?;
    module.add("CheckError", py.get_type::<CheckError>())?;
    module.add("TrapError", py.get_type::<TrapError>())?;
    module.add("ValidationError", py.get_type::<ValidationError>())?;
    module.add("ResourceError", py.get_type::<ResourceError>())?;
    module.add("EngineError", py.get_type::<EngineError>())?;
    module.add("ForeignError", py.get_type::<ForeignError>())?;
    Ok(())
}

#[cfg(test)]
mod tests;
