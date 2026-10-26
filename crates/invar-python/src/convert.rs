//! Result, error and scalar conversions at the Python boundary.
use super::*;

/// The normal-return document; every allocation happens here, before any
/// caller array is written.
pub(crate) fn reference_document(
    py: Python<'_>,
    ty: &Type,
    value: &invar::eval::Value,
    observation: &invar::eval::Observation,
    committed: Vec<usize>,
) -> PyResult<Py<PyAny>> {
    let result = reference_result(py, ty, value)?;
    let cleanup = PyList::empty(py);
    for event in &observation.cleanup_trace {
        let entry = PyDict::new(py);
        entry.set_item("name", &event.name)?;
        entry.set_item("length", event.length)?;
        cleanup.append(entry)?;
    }
    let document = PyDict::new(py);
    document.set_item("engine", "reference")?;
    document.set_item("copying", true)?;
    document.set_item("result", result)?;
    document.set_item("steps", observation.steps_used)?;
    document.set_item("cleanup", cleanup)?;
    document.set_item("committed", committed)?;
    Ok(document.into_any().unbind())
}

/// Map a normal reference result into Python. `Result` scalars become
/// `{"variant": "Ok"|"Err", "value": ...}`; `Err` is a returned value.
pub(crate) fn reference_result(
    py: Python<'_>,
    ty: &Type,
    value: &invar::eval::Value,
) -> PyResult<Py<PyAny>> {
    use invar::eval::Value as V;
    let scalar = |value: &V| -> PyResult<Py<PyAny>> {
        Ok(match value {
            V::I32(x) => x.into_pyobject(py)?.into_any().unbind(),
            V::U32(x) => x.into_pyobject(py)?.into_any().unbind(),
            V::Bool(x) => x.into_pyobject(py)?.to_owned().into_any().unbind(),
            V::Unit => py.None(),
            // A Python float is exactly binary64, so the result carries every bit (HA-1.4).
            V::F64(x) => PyFloat::new(py, f64::from_bits(x.to_bits()))
                .into_any()
                .unbind(),
            V::F32(x) => {
                let entry = PyDict::new(py);
                entry.set_item("f32", f64::from(f32::from_bits(x.to_bits())))?;
                entry.set_item("bits", x.to_bits())?;
                entry.into_any().unbind()
            }
            _ => {
                return Err(boundary_error(
                    py,
                    Class::Engine,
                    "PY_ABI_TYPE_UNAVAILABLE",
                    "result value is outside the reference array entry point",
                ));
            }
        })
    };
    Ok(match (ty, value) {
        (Type::Named(_), V::Variant(storage)) => {
            let (tag, payload) = &*storage.borrow();
            let entry = PyDict::new(py);
            entry.set_item("variant", if *tag == 0 { "Ok" } else { "Err" })?;
            entry.set_item(
                "value",
                match payload {
                    Some(payload) => scalar(payload)?,
                    None => py.None(),
                },
            )?;
            entry.into_any().unbind()
        }
        (_, V::ArrayI32(v)) => PyArray1::from_slice(py, &v.borrow()).into_any().unbind(),
        (_, V::ArrayU32(v)) => PyArray1::from_slice(py, &v.borrow()).into_any().unbind(),
        (_, V::ArrayBool(v)) => PyArray1::from_slice(py, &v.borrow()).into_any().unbind(),
        (_, V::ArrayF64(v)) => {
            let source = v.borrow();
            let array = PyArray1::<f64>::zeros(py, source.len(), false);
            {
                let unavailable = || {
                    boundary_error(
                        py,
                        Class::Engine,
                        "PY_ABI_TYPE_UNAVAILABLE",
                        "result array is unavailable",
                    )
                };
                let mut target = array.try_readwrite().map_err(|_| unavailable())?;
                let target = target.as_slice_mut().map_err(|_| unavailable())?;
                for (slot, bits) in target.iter_mut().zip(source.iter()) {
                    *slot = f64::from_bits(bits.to_bits());
                }
            }
            array.into_any().unbind()
        }
        (_, V::ArrayF32(v)) => {
            let source = v.borrow();
            let array = PyArray1::<f32>::zeros(py, source.len(), false);
            {
                let mut target = array.try_readwrite().map_err(|_| {
                    boundary_error(
                        py,
                        Class::Engine,
                        "PY_ABI_TYPE_UNAVAILABLE",
                        "result array is unavailable",
                    )
                })?;
                let target = target.as_slice_mut().map_err(|_| {
                    boundary_error(
                        py,
                        Class::Engine,
                        "PY_ABI_TYPE_UNAVAILABLE",
                        "result array is unavailable",
                    )
                })?;
                for (slot, bits) in target.iter_mut().zip(source.iter()) {
                    *slot = f32::from_bits(bits.to_bits());
                }
            }
            array.into_any().unbind()
        }
        (_, other) => scalar(other)?,
    })
}

/// Typed exception for a non-normal reference outcome, keeping the AI-01
/// document (and its source span) of the core error.
pub(crate) fn host_error(
    py: Python<'_>,
    error: &invar::eval::EvalError,
    path: &str,
    source: &[u8],
) -> PyErr {
    use invar::eval::EvalError as E;
    let remapped = |class, code: &str, message: &str| boundary_error(py, class, code, message);
    match error {
        E::Trap { .. } => exception(py, Class::Trap, error.to_diagnostics_json(path, source)),
        E::Tool { code, message } => match *code {
            "HOST_ARRAY_EXTENT" => remapped(Class::Validation, "PY_BUFFER_LIMIT", message),
            "HOST_ARRAY_ALIAS" => remapped(Class::Validation, "PY_BUFFER_ALIAS", message),
            "HOST_STAGING_LIMIT" => staging_limit(py),
            "HOST_TAKE_UNAVAILABLE" => remapped(Class::Validation, "PY_TAKE_UNAVAILABLE", message),
            "NATIVE_ABI" => exception(
                py,
                Class::Validation,
                error.to_diagnostics_json(path, source),
            ),
            "INTERNAL_HOST_READ" | "INTERNAL_HOST_COMMIT" => {
                exception(py, Class::Engine, error.to_diagnostics_json(path, source))
            }
            _ => exception(py, Class::Resource, error.to_diagnostics_json(path, source)),
        },
        _ => exception(py, Class::Resource, error.to_diagnostics_json(path, source)),
    }
}

pub(crate) fn scalar_type_name(ty: ScalarType) -> &'static str {
    match ty {
        ScalarType::I32 => "i32",
        ScalarType::U32 => "u32",
        ScalarType::F32 => "f32",
        ScalarType::F64 => "f64",
        ScalarType::Bool => "Bool",
    }
}

pub(crate) fn export_exception(py: Python<'_>, error: &ExportError, document: Value) -> PyErr {
    let class = match error {
        ExportError::Source(error) if error.kind == ErrorKind::Tool => Class::Resource,
        ExportError::Source(_) => Class::Check,
        ExportError::Evaluation(invar::eval::EvalError::Trap { .. }) => Class::Trap,
        ExportError::Evaluation(_) => Class::Resource,
        ExportError::Boundary { code, .. }
            if matches!(
                *code,
                "INVALID_FUEL" | "RESOURCE_LIMIT" | "CALL_DEPTH_LIMIT"
            ) =>
        {
            Class::Resource
        }
        ExportError::Boundary { .. } => Class::Engine,
    };
    exception(py, class, document)
}

pub(crate) fn export_names(py: Python<'_>, selection: &Bound<'_, PyAny>) -> PyResult<Vec<String>> {
    let invalid = || {
        boundary_error(
            py,
            Class::Validation,
            "TYPE_MISMATCH",
            "exports requires an exact built-in list or tuple of exact built-in strings",
        )
    };
    let limited = || {
        boundary_error(
            py,
            Class::Resource,
            "RESOURCE_LIMIT",
            "export selection exceeds 4096 names or one MiB of name text",
        )
    };
    let exact_list = selection.get_type().is(py.get_type::<PyList>());
    let exact_tuple = selection.get_type().is(py.get_type::<PyTuple>());
    if !exact_list && !exact_tuple {
        return Err(invalid());
    }
    if selection.len()? > 4096 {
        return Err(limited());
    }
    let values = if selection.get_type().is(py.get_type::<PyList>()) {
        selection.cast::<PyList>()?.iter().collect::<Vec<_>>()
    } else if selection.get_type().is(py.get_type::<PyTuple>()) {
        selection.cast::<PyTuple>()?.iter().collect::<Vec<_>>()
    } else {
        return Err(invalid());
    };
    let mut names = Vec::new();
    names
        .try_reserve_exact(values.len())
        .map_err(|_| limited())?;
    let mut bytes = 0_usize;
    for value in &values {
        if !value.get_type().is(py.get_type::<PyString>()) {
            return Err(invalid());
        }
        let name = value.cast::<PyString>()?.to_str().map_err(|_| invalid())?;
        bytes = bytes
            .checked_add(name.len())
            .filter(|length| *length <= 1_048_576)
            .ok_or_else(limited)?;
        names.push(name.to_owned());
    }
    Ok(names)
}

pub(crate) fn scalar_i32(py: Python<'_>, value: &Bound<'_, PyAny>) -> PyResult<i32> {
    // Exact built-in integers cannot execute __index__ callbacks while slices are live.
    if !value.get_type().is(py.get_type::<PyInt>()) {
        return Err(boundary_error(
            py,
            Class::Validation,
            "TYPE_MISMATCH",
            "i32 requires an exact built-in integer",
        ));
    }
    value.extract::<i32>().map_err(|_| {
        boundary_error(
            py,
            Class::Validation,
            "TYPE_MISMATCH",
            "i32 requires an exactly representable integer",
        )
    })
}

pub(crate) fn scalar_u32(py: Python<'_>, value: &Bound<'_, PyAny>) -> PyResult<u32> {
    if !value.get_type().is(py.get_type::<PyInt>()) {
        return Err(boundary_error(
            py,
            Class::Validation,
            "TYPE_MISMATCH",
            "u32 requires an exact built-in integer",
        ));
    }
    value.extract::<u32>().map_err(|_| {
        boundary_error(
            py,
            Class::Validation,
            "TYPE_MISMATCH",
            "u32 requires an exactly representable integer",
        )
    })
}

/// An exact built-in `float`: Python's float is binary64, so every bit survives. An `int`, a `float` subclass
/// (including `numpy.float64`) and anything else is refused rather than converted silently.
pub(crate) fn scalar_f64(py: Python<'_>, value: &Bound<'_, PyAny>) -> PyResult<f64> {
    if !value.get_type().is(py.get_type::<PyFloat>()) {
        return Err(boundary_error(
            py,
            Class::Validation,
            "TYPE_MISMATCH",
            "f64 requires an exact built-in float",
        ));
    }
    value.extract::<f64>()
}

pub(crate) fn foreign_code(error: &invar_foreign::Error) -> &'static str {
    match error {
        invar_foreign::Error::Library => "PY_FOREIGN_LIBRARY",
        invar_foreign::Error::Version => "PY_FOREIGN_VERSION",
        invar_foreign::Error::Symbol => "PY_FOREIGN_SYMBOL",
        invar_foreign::Error::Status(_) => "PY_FOREIGN_STATUS",
        invar_foreign::Error::Layout => "PY_FOREIGN_LAYOUT",
        invar_foreign::Error::Utf8 => "INVALID_UTF8",
        invar_foreign::Error::Limit => "PY_BUFFER_LIMIT",
        invar_foreign::Error::Allocation => "PY_FOREIGN_ALLOCATION",
    }
}
