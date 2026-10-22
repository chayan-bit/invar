//! NumPy `bool_` arrays (HA-1.4): one byte per element, and a byte other than 0 or 1 is not a valid `Bool`.
use super::*;

/// Prove every byte of a validated `bool_` array is 0 or 1 before any `&[bool]` is formed over it. An invalid
/// `bool` is undefined behaviour in Rust, and a `uint8` view lets NumPy code write one. The scan reads the same
/// memory through a `uint8` view that NumPy itself creates, so it needs no unsafe code and takes no borrow that
/// outlives the call.
pub(crate) fn check_bool_bytes(py: Python<'_>, value: &Bound<'_, PyAny>) -> PyResult<()> {
    let view = value.call_method1("view", ("uint8",))?;
    let view = view
        .cast::<PyArrayDyn<u8>>()
        .map_err(|_| invalid(py, Failure::Dtype))?;
    let guard = view
        .try_readonly()
        .map_err(|_| invalid(py, Failure::Alias))?;
    let bytes = guard
        .as_slice()
        .map_err(|_| invalid(py, Failure::Contiguity))?;
    if validation::bool_bytes_valid(bytes) {
        Ok(())
    } else {
        Err(invalid(py, Failure::BoolValue))
    }
}
