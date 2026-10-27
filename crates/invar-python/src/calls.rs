//! Reference array calls: admission, guards, bridges and commit.
use super::*;

impl CheckedModule {
    pub(crate) fn reference_call(
        &self,
        py: Python<'_>,
        name: &str,
        args: &Bound<'_, PyTuple>,
        fuel: Fuel,
        inject: BindingInject,
    ) -> PyResult<Py<PyAny>> {
        let function = self
            .checked
            .program
            .function(name)
            .ok_or_else(|| {
                steps(
                    py,
                    boundary_error(
                        py,
                        Class::Validation,
                        "NAME_UNKNOWN",
                        "unknown function name",
                    ),
                    0,
                )
            })?;
        if args.len() != function.params.len() {
            return Err(steps(
                py,
                boundary_error(
                    py,
                    Class::Validation,
                    "PY_ARGUMENT_COUNT",
                    "wrong argument count",
                ),
                0,
            ));
        }
        let values = args.iter().collect::<Vec<_>>();
        // 1. Validate every argument before any borrow.
        let mut arrays: Vec<(usize, Metadata, bool)> = Vec::new();
        for (index, (param, value)) in function.params.iter().zip(&values).enumerate() {
            match (param.mode, &param.ty) {
                (Some(Mode::Take), _) => {
                    return Err(steps(
                        py,
                        boundary_error(
                            py,
                            Class::Validation,
                            "PY_TAKE_UNAVAILABLE",
                            "Python buffers cannot be consumed by the current binding; take requires an explicit transferable owner",
                        ),
                        0,
                    ));
                }
                (None, Type::I32) => {
                    scalar_i32(py, value).map_err(|error| steps(py, error, 0))?;
                }
                (None, Type::U32) => {
                    scalar_u32(py, value).map_err(|error| steps(py, error, 0))?;
                }
                (None, Type::Bool) => {
                    scalar_bool(py, value).map_err(|error| steps(py, error, 0))?;
                }
                (None, Type::F64) => {
                    scalar_f64(py, value).map_err(|error| steps(py, error, 0))?;
                }
                (Some(mode @ (Mode::Read | Mode::Mut)), ty) if array_dtype(ty).is_some() => {
                    let dtype = array_dtype(ty).unwrap_or(Dtype::I32);
                    let meta = metadata(py, value).map_err(|error| steps(py, error, 0))?;
                    validation::validate(meta, dtype, None, mode == Mode::Mut)
                        .map_err(|failure| steps(py, invalid(py, failure), 0))?;
                    if dtype == Dtype::Bool {
                        check_bool_bytes(py, value).map_err(|error| steps(py, error, 0))?;
                    }
                    arrays.push((index, meta, mode == Mode::Mut));
                }
                _ => {
                    return Err(steps(
                        py,
                        boundary_error(
                            py,
                            Class::Engine,
                            "PY_ABI_TYPE_UNAVAILABLE",
                            "parameter type is outside the reference array entry point (exact i32, u32, f64, Bool, read/mut Array<i32|u32|f32|f64|Bool>)",
                        ),
                        0,
                    ));
                }
            }
        }
        // 2. Any overlap involving a mut array is refused; read/read overlap
        // is left to the core (exact aliases share one owner).
        for (position, (_, left, left_mut)) in arrays.iter().enumerate() {
            for (_, right, right_mut) in &arrays[position + 1..] {
                if (*left_mut || *right_mut)
                    && validation::overlaps(*left, *right)
                        .map_err(|failure| steps(py, invalid(py, failure), 0))?
                {
                    return Err(steps(py, invalid(py, Failure::Alias), 0));
                }
            }
        }
        // 3. Plan bridges and account every simultaneous host temporary on
        // the original metadata, before any borrow or reservation.
        let planned: Vec<PlanArray> = arrays
            .iter()
            .map(|(_, meta, mutable)| PlanArray {
                dtype: meta.dtype,
                address: meta.address,
                length: meta.length,
                mutable: *mutable,
            })
            .collect();
        let plan = match plan_bridges(&planned) {
            Ok(plan) => plan,
            Err(PlanError::Live) => {
                let core = invar::eval::EvalError::Tool {
                    code: "ARRAY_SIZE_LIMIT",
                    message: "host arrays exceed the live element limit".into(),
                };
                return Err(steps(
                    py,
                    host_error(py, &core, &self.path, &self.source),
                    0,
                ));
            }
            Err(PlanError::Alias) => return Err(steps(py, invalid(py, Failure::Alias), 0)),
            Err(PlanError::Staging) => return Err(steps(py, staging_limit(py), 0)),
        };
        // 4. Borrow guards (strong references), then recheck metadata.
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
            .map_err(|failure| steps(py, invalid(py, failure), 0))?;
            let again = metadata(py, &values[*index]).map_err(|error| steps(py, error, 0))?;
            if !same_metadata(&again, meta) {
                return Err(steps(py, invalid(py, Failure::Alias), 0));
            }
        }
        // 5. Safe source slices; each must be exactly its validated extent.
        let read_slices = readers
            .iter()
            .map(ReadGuard::slice)
            .collect::<Result<Vec<_>, _>>()
            .map_err(|failure| steps(py, invalid(py, failure), 0))?;
        let mut write_slices = writers
            .iter_mut()
            .map(WriteGuard::slice)
            .collect::<Result<Vec<_>, _>>()
            .map_err(|failure| steps(py, invalid(py, failure), 0))?;
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
                    return Err(steps(py, invalid(py, Failure::Alias), 0));
                }
            }
        }
        // 6. Reserve and fill every bridge (fallible, within the plan).
        let mut reservations = 0usize;
        let mut unions: Vec<Vec<F32Bits>> = Vec::new();
        let mut bridges: Vec<Bridge> = Vec::new();
        let mut pairs: Vec<Pair<'_, '_>> = Vec::new();
        if unions.try_reserve_exact(plan.unions.len()).is_err()
            || bridges.try_reserve_exact(write_slices.len()).is_err()
            || pairs.try_reserve_exact(write_slices.len()).is_err()
        {
            return Err(steps(py, staging_limit(py), 0));
        }
        for length in &plan.unions {
            let mut buffer = Vec::new();
            if inject == BindingInject::BridgeReserveFailAt(reservations)
                || buffer.try_reserve_exact(*length).is_err()
            {
                return Err(steps(py, staging_limit(py), 0));
            }
            reservations += 1;
            buffer.resize(*length, F32Bits::from_bits(0));
            unions.push(buffer);
        }
        {
            let mut r = read_slices.iter();
            for ((_, meta, mutable), route) in arrays.iter().zip(&plan.routes) {
                if *mutable {
                    continue;
                }
                let slice = r.next();
                if let (Route::Union { union, offset }, Some(ReadSlice::F32(source))) =
                    (route, slice)
                {
                    for (slot, value) in unions[*union][*offset..*offset + meta.length]
                        .iter_mut()
                        .zip(source.iter())
                    {
                        *slot = F32Bits::from_bits(value.to_bits());
                    }
                }
            }
        }
        for slice in &write_slices {
            let failed = inject == BindingInject::BridgeReserveFailAt(reservations);
            reservations += 1;
            let bridge = Bridge::copy(slice).filter(|_| !failed);
            let Some(bridge) = bridge else {
                return Err(steps(py, staging_limit(py), 0));
            };
            bridges.push(bridge);
        }
        // 7. Arguments in parameter order: direct i32/u32 reads, f32 reads as
        // slices of their union bridge at the original offsets (exact and
        // partial aliases preserved), and every mut array as its bridge.
        let mut host_args: Vec<HostArgument<'_>> = Vec::with_capacity(values.len());
        {
            let mut routes = plan.routes.iter().zip(arrays.iter());
            let mut r = read_slices.iter();
            let mut w = bridges.iter_mut();
            for (param, value) in function.params.iter().zip(&values) {
                let argument = match (param.mode, &param.ty) {
                    (None, Type::I32) => HostArgument::I32(scalar_i32(py, value)?),
                    (None, Type::U32) => HostArgument::U32(scalar_u32(py, value)?),
                    (None, Type::Bool) => HostArgument::Bool(scalar_bool(py, value)?),
                    (None, Type::F64) => {
                        HostArgument::F64(F64Bits::from_bits(scalar_f64(py, value)?.to_bits()))
                    }
                    _ => {
                        let Some((route, (_, meta, _))) = routes.next() else {
                            return Err(steps(py, invalid(py, Failure::Ownership), 0));
                        };
                        match route {
                            Route::Mut => match w.next() {
                                Some(bridge) => bridge.argument(),
                                None => return Err(steps(py, invalid(py, Failure::Ownership), 0)),
                            },
                            Route::Union { union, offset } => {
                                r.next();
                                HostArgument::Read(HostReadArray::F32(
                                    &unions[*union][*offset..*offset + meta.length],
                                ))
                            }
                            Route::EmptyF32 => {
                                r.next();
                                HostArgument::Read(HostReadArray::F32(&[]))
                            }
                            Route::Direct => match r.next() {
                                Some(ReadSlice::I32(v)) => {
                                    HostArgument::Read(HostReadArray::I32(v))
                                }
                                Some(ReadSlice::U32(v)) => {
                                    HostArgument::Read(HostReadArray::U32(v))
                                }
                                Some(ReadSlice::F64(v)) => {
                                    HostArgument::Read(HostReadArray::Float64(v))
                                }
                                Some(ReadSlice::Bool(v)) => {
                                    HostArgument::Read(HostReadArray::Bool(v))
                                }
                                _ => return Err(steps(py, invalid(py, Failure::Ownership), 0)),
                            },
                        }
                    }
                };
                host_args.push(argument);
            }
        }
        // 8. Evaluate on the reference engine; no Python runs here. The core
        // commits mut writes into the bridges, never into NumPy memory.
        let observation = observe_host_call(&self.checked, name, &mut host_args, fuel.0);
        drop(host_args);
        let value = match &observation.result {
            Ok(value) => value,
            Err(error) => {
                return Err(steps(
                    py,
                    host_error(py, error, &self.path, &self.source),
                    observation.steps_used,
                ));
            }
        };
        // 9. Build the whole Python result (all fallible allocations) before
        // any caller write. A failure here leaves every caller array unchanged.
        if inject == BindingInject::ResultBuildFail {
            return Err(steps(
                py,
                pyo3::exceptions::PyMemoryError::new_err("injected result construction failure"),
                observation.steps_used,
            ));
        }
        let committed: Vec<usize> = arrays
            .iter()
            .filter(|(_, _, mutable)| *mutable)
            .map(|(index, _, _)| *index)
            .collect();
        let document = reference_document(py, &function.result, value, &observation, committed)?;
        // 10. Precheck every writeback pair (element type and length) into
        // capacity reserved above; nothing is written yet.
        for (slice, bridge) in write_slices.iter_mut().zip(bridges.iter()) {
            let Some(pair) = Pair::new(slice, bridge) else {
                return Err(steps(
                    py,
                    boundary_error(
                        py,
                        Class::Engine,
                        "INTERNAL_HOST_COMMIT",
                        "a NumPy writeback failed its precheck",
                    ),
                    observation.steps_used,
                ));
            };
            pairs.push(pair);
        }
        if inject == BindingInject::PanicBeforeWriteback {
            panic!("injected panic before NumPy writeback");
        }
        // 11. Writeback: equal-length element copies only (exact f32 bits);
        // no allocation, no borrow acquisition, no fallible call.
        for pair in pairs.drain(..) {
            pair.write();
        }
        Ok(document)
    }
}

/// Metadata recheck under the borrow guards (field-by-field).
pub(crate) fn same_metadata(a: &Metadata, b: &Metadata) -> bool {
    (
        a.dtype,
        a.rank,
        a.length,
        a.contiguous,
        a.aligned,
        a.writable,
        a.owns_data,
        a.address,
    ) == (
        b.dtype,
        b.rank,
        b.length,
        b.contiguous,
        b.aligned,
        b.writable,
        b.owns_data,
        b.address,
    )
}

/// Set the additive `steps_used` attribute on a reference-entry exception.
pub(crate) fn steps(py: Python<'_>, error: PyErr, steps_used: u64) -> PyErr {
    if let Err(failure) = error.value(py).setattr("steps_used", steps_used) {
        return failure;
    }
    error
}

pub(crate) fn scalar_bool(py: Python<'_>, value: &Bound<'_, PyAny>) -> PyResult<bool> {
    if !value.get_type().is(py.get_type::<pyo3::types::PyBool>()) {
        return Err(boundary_error(
            py,
            Class::Validation,
            "TYPE_MISMATCH",
            "Bool requires an exact built-in bool",
        ));
    }
    value.extract::<bool>()
}

pub(crate) fn array_dtype(ty: &Type) -> Option<Dtype> {
    match ty {
        Type::ArrayI32 => Some(Dtype::I32),
        Type::ArrayU32 => Some(Dtype::U32),
        Type::ArrayF32 => Some(Dtype::F32),
        Type::ArrayF64 => Some(Dtype::F64),
        Type::ArrayBool => Some(Dtype::Bool),
        _ => None,
    }
}
