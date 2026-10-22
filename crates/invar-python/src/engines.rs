//! Whole-program execution through the shared CLI engine selector.
use super::*;

pub(crate) struct Workers(pub(crate) usize);

impl<'a, 'py> FromPyObject<'a, 'py> for Workers {
    type Error = PyErr;

    fn extract(value: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
        let py = value.py();
        let invalid = || {
            boundary_error(
                py,
                Class::Engine,
                "CPU_WORKERS",
                "workers must be an exact built-in integer in 1..=64 for cpu-parallel",
            )
        };
        if !value.get_type().is(py.get_type::<PyInt>()) {
            return Err(invalid());
        }
        let count = value.extract::<usize>().map_err(|_| invalid())?;
        if !(1..=invar::schedule::MAX_WORKERS).contains(&count) {
            return Err(invalid());
        }
        Ok(Self(count))
    }
}

pub(crate) fn execution_error(
    py: Python<'_>,
    error: invar::eval::EvalError,
    path: &str,
    source: &[u8],
) -> PyErr {
    let class = match &error {
        invar::eval::EvalError::Trap { .. } => Class::Trap,
        invar::eval::EvalError::EvaluationLimit { .. } => Class::Resource,
        invar::eval::EvalError::Tool { code, .. }
            if matches!(*code, "RESOURCE_LIMIT" | "ARRAY_SIZE_LIMIT") =>
        {
            Class::Resource
        }
        _ => Class::Engine,
    };
    exception(py, class, error.to_diagnostics_json(path, source))
}
