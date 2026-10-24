//! HA-1.2 array parameters of selected exports: admission of the bounded
//! subset and the array call (`docs/host-array-arguments-ha1.md`). Native
//! execution happens at the caller's addresses through the private host-array
//! protocol; the reference engine is the copying semantic oracle.
use super::{
    ArrayMode, CompiledExports, ExportError, ExportParameter, Implementation, ScalarType,
    ScalarValue, boundary,
};
use crate::{
    eval::{
        self,
        host_read::{HostArgument, observe_host_call},
    },
    front::{Mode, Type},
    mir::{self, ValueId},
};
use std::collections::{HashMap, HashSet};

/// The result of one array call.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ArrayCallOutcome {
    pub value: ScalarValue,
    /// True only for native execution at the caller's addresses.
    pub zero_copy: bool,
    /// Native only: the address in each private record handed to native code,
    /// in array-argument order (0 for an empty array). Empty for reference.
    pub record_addresses: Vec<usize>,
}

/// Signature admission of one parameter of a selected export.
pub(super) fn export_parameter(p: &crate::front::Param) -> Result<ExportParameter, ExportError> {
    let (ty, array) = match (p.mode, &p.ty) {
        (None, ty) => (ScalarType::from_type(ty), None),
        (Some(Mode::Read), Type::ArrayI32) => (Some(ScalarType::I32), Some(ArrayMode::Read)),
        (Some(Mode::Read), Type::ArrayU32) => (Some(ScalarType::U32), Some(ArrayMode::Read)),
        (Some(Mode::Mut), Type::ArrayI32) => (Some(ScalarType::I32), Some(ArrayMode::Mut)),
        (Some(Mode::Mut), Type::ArrayU32) => (Some(ScalarType::U32), Some(ArrayMode::Mut)),
        (Some(Mode::Read), Type::ArrayF32) => (Some(ScalarType::F32), Some(ArrayMode::Read)),
        (Some(Mode::Mut), Type::ArrayF32) => (Some(ScalarType::F32), Some(ArrayMode::Mut)),
        (Some(Mode::Read), Type::ArrayF64) => (Some(ScalarType::F64), Some(ArrayMode::Read)),
        (Some(Mode::Mut), Type::ArrayF64) => (Some(ScalarType::F64), Some(ArrayMode::Mut)),
        (Some(Mode::Read), Type::ArrayBool) => (Some(ScalarType::Bool), Some(ArrayMode::Read)),
        (Some(Mode::Mut), Type::ArrayBool) => (Some(ScalarType::Bool), Some(ArrayMode::Mut)),
        _ => {
            return Err(boundary(
                "NATIVE_ABI",
                "export parameters must copy i32/u32 values or borrow read/mut Array<i32|u32|f32|f64|bool>",
            ));
        }
    };
    Ok(ExportParameter {
        name: p.name.clone(),
        ty: ty.ok_or_else(|| boundary("NATIVE_ABI", "export parameter type is unsupported"))?,
        array,
    })
}

#[cfg(feature = "native")]
pub(super) fn parameter_tag(p: &ExportParameter) -> u32 {
    use invar_foreign::host_array as host;
    let element = match p.ty {
        ScalarType::I32 => host::ELEMENT_I32,
        ScalarType::U32 => host::ELEMENT_U32,
        ScalarType::F32 => host::ELEMENT_F32,
        ScalarType::F64 => host::ELEMENT_F64,
        ScalarType::Bool => host::ELEMENT_BOOL,
    };
    match p.array {
        None => p.ty.tag(),
        Some(ArrayMode::Read) => host::descriptor_tag(element, host::MODE_READ),
        Some(ArrayMode::Mut) => host::descriptor_tag(element, host::MODE_MUT),
    }
}

fn scalar(ty: &Type) -> bool {
    matches!(
        ty,
        Type::I32 | Type::U32 | Type::F32 | Type::F64 | Type::Bool | Type::Unit
    )
}

/// Array parameters a function may use: only those of a selected function.
fn admitted_arrays(f: &mir::Function, selected: bool) -> HashMap<&str, bool> {
    if !selected {
        return HashMap::new();
    }
    f.params
        .iter()
        .filter(|p| {
            matches!(
                p.ty,
                Type::ArrayI32 | Type::ArrayU32 | Type::ArrayF32 | Type::ArrayF64 | Type::ArrayBool
            )
        })
        .filter_map(|p| match p.mode {
            Some(Mode::Read) => Some((p.name.as_str(), false)),
            Some(Mode::Mut) => Some((p.name.as_str(), true)),
            _ => None,
        })
        .collect()
}

/// Admit the complete reachable MIR of the selection: the scalar fragment,
/// plus, in selected functions only, element reads, lengths and element
/// stores of their own `read`/`mut` `Array<i32|u32|f32|f64|bool>` parameters. An array
/// value may only feed `Index`; it is never passed, bound, stored or returned.
pub(super) fn admit(
    program: &mir::Program,
    selected: &[usize],
) -> Result<HashSet<usize>, ExportError> {
    let by_name = program
        .functions
        .iter()
        .enumerate()
        .map(|(i, f)| (f.name.as_str(), i))
        .collect::<HashMap<_, _>>();
    let mut pending = selected.to_vec();
    let mut reached = HashSet::new();
    let refuse = |message: &str| boundary("NATIVE_ABI", message.to_owned());
    while let Some(id) = pending.pop() {
        if !reached.insert(id) {
            continue;
        }
        let f = &program.functions[id];
        let arrays = admitted_arrays(f, selected.contains(&id));
        let values = f
            .blocks
            .iter()
            .map(|b| b.params.len() + b.instructions.iter().filter(|i| i.output.is_some()).count())
            .sum::<usize>();
        if f.foreign_oracle.is_some()
            || !scalar(&f.result)
            || f.params.iter().any(|p| {
                !arrays.contains_key(p.name.as_str()) && (!scalar(&p.ty) || p.mode.is_some())
            })
            || f.locals.iter().any(|(name, v)| {
                !arrays.contains_key(name.as_str()) && (!scalar(&v.ty) || v.mode.is_some())
            })
            || f.declaration_order.len() + values > 256
            || f.blocks.len() > 256
            || f.blocks.iter().map(|b| b.instructions.len()).sum::<usize>() > 2048
        {
            return Err(boundary(
                "NATIVE_ABI",
                format!("function '{}' exceeds scalar export admission", f.name),
            ));
        }
        let mut array_values: HashSet<ValueId> = HashSet::new();
        for b in &f.blocks {
            if b.params.iter().any(|(_, ty)| !scalar(ty)) {
                return Err(refuse("non-scalar block value"));
            }
            for instruction in &b.instructions {
                use mir::Operation as Op;
                let array_output = matches!(
                    &instruction.kind,
                    Op::ArrayRef { name, loan: None } if arrays.contains_key(name.as_str())
                );
                match &instruction.output {
                    Some((id, _)) if array_output => {
                        array_values.insert(*id);
                    }
                    Some((_, ty)) if !scalar(ty) => {
                        return Err(refuse("non-scalar instruction value"));
                    }
                    _ => {}
                }
                let uses_array = |ids: &[ValueId]| ids.iter().any(|id| array_values.contains(id));
                match &instruction.kind {
                    Op::Tick
                    | Op::I32(_)
                    | Op::U32(_)
                    | Op::F32(_)
                    | Op::F64(_)
                    | Op::Bool(_)
                    | Op::Read(_)
                    | Op::Drop(_)
                    | Op::ForNext(_) => {}
                    Op::Unary { value, .. } if !uses_array(&[*value]) => {}
                    Op::Binary { left, right, .. } if !uses_array(&[*left, *right]) => {}
                    Op::Bind { name, value } | Op::Store { name, value }
                        if !arrays.contains_key(name.as_str()) && !uses_array(&[*value]) => {}
                    Op::ArrayRef { .. } if array_output => {}
                    Op::Index { array, index }
                        if array_values.contains(array) && !uses_array(&[*index]) => {}
                    Op::Length(name) if arrays.contains_key(name.as_str()) => {}
                    Op::StoreIndex { name, index, value }
                        if arrays.get(name.as_str()) == Some(&true)
                            && !uses_array(&[*index, *value]) => {}
                    Op::Call { name, args }
                        if matches!(
                            (name.as_str(), args.len()),
                            ("wrapping_add" | "wrapping_mul", 2)
                                | ("f32_from_i32" | "f32_from_u32", 1)
                                | ("fma", 3)
                        ) && !uses_array(args) => {}
                    Op::Call { name, args } if !uses_array(args) => {
                        let callee = by_name
                            .get(name.as_str())
                            .ok_or_else(|| refuse("unsupported scalar export intrinsic"))?;
                        pending.push(*callee);
                    }
                    _ => {
                        return Err(refuse(
                            "reachable operation is outside scalar export fragment",
                        ));
                    }
                }
            }
            let terminator_uses_array = match &b.end {
                Some(mir::Terminator::Goto { args, .. }) => {
                    args.iter().any(|id| array_values.contains(id))
                }
                Some(mir::Terminator::Branch { condition, .. }) => array_values.contains(condition),
                Some(mir::Terminator::Return { value, .. }) => {
                    value.is_some_and(|id| array_values.contains(&id))
                }
                None => false,
            };
            if terminator_uses_array {
                return Err(refuse("array value escapes its element access"));
            }
        }
    }
    Ok(reached)
}

impl CompiledExports {
    /// Call an export with array parameters (HA-1.2). `args` are in parameter
    /// order: `I32`/`U32` scalars and `Read`/`Mut` `i32`/`u32` arrays.
    ///
    /// Native-serial runs at the caller's addresses: `mut` arrays hold the
    /// final contents after a normal return and their exact pre-call bits
    /// after any failure. The reference engine copies (HA-1 section 10) and
    /// writes `mut` arrays back only after a normal return.
    pub fn invoke_arrays(
        &mut self,
        name: &str,
        args: &mut [HostArgument<'_>],
        fuel: u64,
    ) -> Result<ArrayCallOutcome, ExportError> {
        if fuel == 0 {
            return Err(ExportError::Evaluation(eval::EvalError::Tool {
                code: "INVALID_FUEL",
                message: "fuel must be positive".into(),
            }));
        }
        let descriptor = self
            .signatures
            .iter()
            .find(|s| s.source_name == name)
            .ok_or_else(|| boundary("NATIVE_ABI", "unknown exported function"))?;
        let matches = |arg: &HostArgument<'_>, p: &ExportParameter| {
            use eval::host_read::{HostMutArray as M, HostReadArray as R};
            matches!(
                (arg, p.array, p.ty),
                (
                    HostArgument::Read(R::F32(_) | R::Float32(_)),
                    Some(ArrayMode::Read),
                    ScalarType::F32
                ) | (
                    HostArgument::Mut(M::F32(_) | M::Float32(_)),
                    Some(ArrayMode::Mut),
                    ScalarType::F32
                ) | (
                    HostArgument::Read(R::Float64(_)),
                    Some(ArrayMode::Read),
                    ScalarType::F64
                ) | (
                    HostArgument::Mut(M::Float64(_)),
                    Some(ArrayMode::Mut),
                    ScalarType::F64
                ) | (
                    HostArgument::Read(R::Bool(_)),
                    Some(ArrayMode::Read),
                    ScalarType::Bool
                ) | (
                    HostArgument::Mut(M::Bool(_)),
                    Some(ArrayMode::Mut),
                    ScalarType::Bool
                ) | (HostArgument::I32(_), None, ScalarType::I32)
                    | (HostArgument::U32(_), None, ScalarType::U32)
                    | (
                        HostArgument::Read(R::I32(_)),
                        Some(ArrayMode::Read),
                        ScalarType::I32
                    )
                    | (
                        HostArgument::Read(R::U32(_)),
                        Some(ArrayMode::Read),
                        ScalarType::U32
                    )
                    | (
                        HostArgument::Mut(M::I32(_)),
                        Some(ArrayMode::Mut),
                        ScalarType::I32
                    )
                    | (
                        HostArgument::Mut(M::U32(_)),
                        Some(ArrayMode::Mut),
                        ScalarType::U32
                    )
            )
        };
        if args.len() != descriptor.params.len()
            || !args
                .iter()
                .zip(&descriptor.params)
                .all(|(a, p)| matches(a, p))
        {
            return Err(boundary(
                "NATIVE_ABI",
                "export argument arity/type mismatch",
            ));
        }
        let result_type = descriptor.result;
        let value = match &mut self.implementation {
            Implementation::Reference => {
                let observation = observe_host_call(&self.checked, name, args, fuel);
                let value = match observation.result.map_err(ExportError::Evaluation)? {
                    eval::Value::I32(v) => ScalarValue::I32(v),
                    eval::Value::U32(v) => ScalarValue::U32(v),
                    _ => {
                        return Err(boundary(
                            "NATIVE_ABI",
                            "reference export returned a non-scalar",
                        ));
                    }
                };
                ArrayCallOutcome {
                    value,
                    zero_copy: false,
                    record_addresses: Vec::new(),
                }
            }
            #[cfg(feature = "native")]
            Implementation::Native(artifact) => {
                native::invoke(artifact, &descriptor.symbol, args, fuel, &self.source)?
            }
        };
        if value.value.ty() != result_type {
            return Err(boundary(
                "NATIVE_ABI",
                "export returned the wrong scalar tag",
            ));
        }
        Ok(value)
    }
}

#[cfg(feature = "native")]
mod native {
    use super::*;
    use crate::eval::host_read::{HostMutArray as M, HostReadArray as R};
    use invar_foreign::host_array::{HostArray, HostArrayCall, HostArrayError, HostCallError};

    fn host_array_error(error: HostArrayError) -> ExportError {
        let (code, message) = match error {
            HostArrayError::Count => ("NATIVE_ABI", "too many host arrays"),
            HostArrayError::Length => ("ARRAY_SIZE_LIMIT", "host array exceeds the element limit"),
            HostArrayError::Extent => ("HOST_ARRAY_EXTENT", "host array extent overflows"),
            HostArrayError::Alias => ("HOST_ARRAY_ALIAS", "host arrays overlap where forbidden"),
            HostArrayError::Live => (
                "ARRAY_SIZE_LIMIT",
                "host arrays exceed the live element limit",
            ),
            HostArrayError::Snapshot => (
                "HOST_STAGING_LIMIT",
                "rollback snapshot could not be reserved",
            ),
        };
        ExportError::Evaluation(eval::EvalError::Tool {
            code,
            message: message.into(),
        })
    }

    pub(super) fn invoke(
        artifact: &mut super::super::NativeArtifact,
        symbol: &str,
        args: &mut [HostArgument<'_>],
        fuel: u64,
        source: &[u8],
    ) -> Result<ArrayCallOutcome, ExportError> {
        let descriptor = artifact.descriptor(symbol)?.clone();
        let mut scalars = Vec::new();
        let mut arrays = Vec::new();
        for arg in args.iter_mut() {
            match arg {
                HostArgument::I32(v) => scalars.push(invar_foreign::Arg::I32(*v)),
                HostArgument::U32(v) => scalars.push(invar_foreign::Arg::U32(*v)),
                HostArgument::Read(R::I32(v)) => arrays.push(HostArray::ReadI32(v)),
                HostArgument::Read(R::U32(v)) => arrays.push(HostArray::ReadU32(v)),
                HostArgument::Mut(M::I32(v)) => arrays.push(HostArray::MutI32(v)),
                HostArgument::Mut(M::U32(v)) => arrays.push(HostArray::MutU32(v)),
                HostArgument::Read(R::Float32(v)) => arrays.push(HostArray::ReadF32(v)),
                HostArgument::Mut(M::Float32(v)) => arrays.push(HostArray::MutF32(v)),
                HostArgument::Read(R::Float64(v)) => arrays.push(HostArray::ReadF64(v)),
                HostArgument::Mut(M::Float64(v)) => arrays.push(HostArray::MutF64(v)),
                HostArgument::Read(R::Bool(v)) => arrays.push(HostArray::ReadBool(v)),
                HostArgument::Mut(M::Bool(v)) => arrays.push(HostArray::MutBool(v)),
                _ => {
                    return Err(boundary(
                        "NATIVE_ABI",
                        "export argument arity/type mismatch",
                    ));
                }
            }
        }
        if arrays.is_empty() {
            // A scalar-only export keeps its public scalar entry.
            let invocation = artifact
                .loaded
                .invoke(&descriptor, &scalars, fuel)
                .map_err(|e| {
                    boundary(
                        "NATIVE_ABI",
                        format!("export loader rejected invocation: {e:?}"),
                    )
                })?;
            let value = super::super::decode_native(invocation, source)?;
            return Ok(ArrayCallOutcome {
                value,
                zero_copy: false,
                record_addresses: Vec::new(),
            });
        }
        // Admission (extents, aliases, live union, snapshots) before any call.
        let call = HostArrayCall::prepare(arrays).map_err(host_array_error)?;
        call.invoke(
            &mut artifact.loaded,
            &descriptor,
            &scalars,
            fuel,
            |invocation, addresses| {
                super::super::decode_native(invocation, source).map(|value| ArrayCallOutcome {
                    value,
                    zero_copy: true,
                    record_addresses: addresses.to_vec(),
                })
            },
        )
        .map_err(|error| match error {
            HostCallError::Decode(error) => error,
            HostCallError::Loader(error) => boundary(
                "NATIVE_ABI",
                format!("export loader rejected invocation: {error:?}"),
            ),
        })
    }
}
