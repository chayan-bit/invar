//! HA-1.2 artifact calls with `read`/`mut` `Array<i32|u32>` parameters.
//!
//! Every argument is validated, every overlap involving a `mut` array is
//! refused, NumPy borrow guards (strong references) are taken and metadata is
//! rechecked before the core is entered. The GIL is held for the whole call
//! and no Python code runs while slices are live. Native-serial artifacts run
//! at the NumPy addresses (zero-copy) and restore every `mut` array's exact
//! bits on any non-normal outcome; reference artifacts copy.
use super::*;
use invar::{eval::host_read::HostMutArray, exports::ArrayMode};

impl CheckedModule {
    pub(crate) fn array_call(
        &mut self,
        py: Python<'_>,
        name: &str,
        args: &Bound<'_, PyTuple>,
        fuel: Fuel,
        observe: bool,
    ) -> PyResult<Py<PyAny>> {
        let unavailable = || {
            boundary_error(
                py,
                Class::Engine,
                "PY_EXPORT_UNAVAILABLE",
                "function was not selected for this module's export artifact",
            )
        };
        let artifact = self.artifact.as_mut().ok_or_else(unavailable)?;
        let signature = artifact
            .signatures()
            .iter()
            .find(|signature| signature.source_name == name)
            .ok_or_else(unavailable)?
            .clone();
        let values = args.iter().collect::<Vec<_>>();
        if values.len() != signature.params.len() {
            return Err(boundary_error(
                py,
                Class::Validation,
                "PY_ARGUMENT_COUNT",
                "wrong argument count",
            ));
        }
        // 1. Validate every argument before any borrow.
        let mut arrays: Vec<(usize, Metadata, bool)> = Vec::new();
        for (index, (param, value)) in signature.params.iter().zip(&values).enumerate() {
            let dtype = match param.ty {
                ScalarType::I32 => Dtype::I32,
                ScalarType::U32 => Dtype::U32,
                ScalarType::F32 => Dtype::F32,
            };
            match param.array {
                None if dtype == Dtype::I32 => {
                    scalar_i32(py, value)?;
                }
                None => {
                    scalar_u32(py, value)?;
                }
                Some(mode) => {
                    let mutable = mode == ArrayMode::Mut;
                    let meta = metadata(py, value)?;
                    validation::validate(meta, dtype, None, mutable)
                        .map_err(|failure| invalid(py, failure))?;
                    arrays.push((index, meta, mutable));
                }
            }
        }
        // 2. Any overlap involving a mut array is refused before any borrow.
        for (position, (_, left, left_mut)) in arrays.iter().enumerate() {
            for (_, right, right_mut) in &arrays[position + 1..] {
                if (*left_mut || *right_mut)
                    && validation::overlaps(*left, *right)
                        .map_err(|failure| invalid(py, failure))?
                {
                    return Err(invalid(py, Failure::Alias));
                }
            }
        }
        // 3. Borrow guards (strong references), then recheck metadata.
        let mut readers: Vec<ReadGuard<'_>> = Vec::new();
        let mut writers: Vec<WriteGuard<'_>> = Vec::new();
        for (index, meta, mutable) in &arrays {
            borrow_guard(
                &values[*index],
                meta.dtype,
                *mutable,
                &mut readers,
                &mut writers,
            )
            .map_err(|failure| invalid(py, failure))?;
            let again = metadata(py, &values[*index])?;
            if !same_metadata(&again, meta) {
                return Err(invalid(py, Failure::Alias));
            }
        }
        // 4. Safe slices; each must be exactly its validated extent.
        let read_slices = readers
            .iter()
            .map(ReadGuard::slice)
            .collect::<Result<Vec<_>, _>>()
            .map_err(|failure| invalid(py, failure))?;
        let mut write_slices = writers
            .iter_mut()
            .map(WriteGuard::slice)
            .collect::<Result<Vec<_>, _>>()
            .map_err(|failure| invalid(py, failure))?;
        {
            let (mut r, mut w) = (read_slices.iter(), write_slices.iter());
            for (_, meta, mutable) in &arrays {
                let (address, length) = if *mutable {
                    w.next().map(WriteSlice::extent)
                } else {
                    r.next().map(ReadSlice::extent)
                }
                .unwrap_or((usize::MAX, usize::MAX));
                if length != meta.length || (length != 0 && address != meta.address) {
                    return Err(invalid(py, Failure::Alias));
                }
            }
        }
        // 5. Arguments in parameter order: the NumPy slices themselves.
        let mut host_args: Vec<HostArgument<'_>> = Vec::with_capacity(values.len());
        {
            let mut r = read_slices.iter();
            let mut w = write_slices.iter_mut();
            for (param, value) in signature.params.iter().zip(&values) {
                let argument = match (param.array, param.ty) {
                    (None, ScalarType::I32) => HostArgument::I32(scalar_i32(py, value)?),
                    (None, ScalarType::U32) => HostArgument::U32(scalar_u32(py, value)?),
                    (None, ScalarType::F32) => return Err(invalid(py, Failure::Dtype)),
                    (Some(ArrayMode::Read), _) => match r.next() {
                        Some(ReadSlice::I32(v)) => HostArgument::Read(HostReadArray::I32(v)),
                        Some(ReadSlice::U32(v)) => HostArgument::Read(HostReadArray::U32(v)),
                        Some(ReadSlice::F32(v)) => HostArgument::Read(HostReadArray::Float32(v)),
                        _ => return Err(invalid(py, Failure::Ownership)),
                    },
                    (Some(ArrayMode::Mut), _) => match w.next() {
                        Some(WriteSlice::I32(v)) => HostArgument::Mut(HostMutArray::I32(v)),
                        Some(WriteSlice::U32(v)) => HostArgument::Mut(HostMutArray::U32(v)),
                        Some(WriteSlice::F32(v)) => HostArgument::Mut(HostMutArray::Float32(v)),
                        _ => return Err(invalid(py, Failure::Ownership)),
                    },
                };
                host_args.push(argument);
            }
        }
        // 6. Execute; no Python runs here. The core commits or restores.
        let outcome = artifact.invoke_arrays(name, &mut host_args, fuel.0);
        drop(host_args);
        let outcome = outcome.map_err(|error| match &error {
            ExportError::Evaluation(core) => host_error(py, core, &self.path, &self.source),
            _ => export_exception(py, &error, artifact.diagnostics(&error)),
        })?;
        let result = match outcome.value {
            ScalarValue::I32(value) => value.into_pyobject(py)?.into_any().unbind(),
            ScalarValue::U32(value) => value.into_pyobject(py)?.into_any().unbind(),
        };
        if !observe {
            return Ok(result);
        }
        let document = PyDict::new(py);
        document.set_item("engine", self.engine)?;
        document.set_item("copying", !outcome.zero_copy)?;
        document.set_item("result", result)?;
        document.set_item("record_addresses", outcome.record_addresses)?;
        Ok(document.into_any().unbind())
    }
}
