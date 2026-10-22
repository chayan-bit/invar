//! Independent aggregate storage for the CPU net oracle.
//!
//! Handles borrow storage; only verified affine moves and explicit cleanup
//! transfer or destroy owners. The flat arena and iterative traversal keep
//! nested cleanup independent of host recursion and Rust drops.

use super::{Frame, OracleError, Scalar, ToolFailure, arrays, internal, tool};
use crate::{
    eval::{CleanupEvent, EvalError, TrapCategory, Value},
    front::{Span, Type},
    mir,
};
use std::{cell::RefCell, collections::BTreeSet, rc::Rc};

enum Node {
    Blob { bytes: Vec<u8>, text: bool },
    Record(Vec<Option<Scalar>>),
    Variant { tag: usize, payload: Option<Scalar> },
}

impl Node {
    fn units(&self) -> usize {
        match self {
            Self::Blob { bytes, .. } => bytes.len(),
            Self::Record(fields) => fields.len(),
            Self::Variant { .. } => 2,
        }
    }
}

pub(super) struct Arena {
    slots: Vec<Option<Node>>,
}

fn reserve<T>(length: usize) -> Result<Vec<T>, OracleError> {
    let mut values = Vec::new();
    values.try_reserve_exact(length).map_err(|_| {
        tool(
            ToolFailure::Allocation,
            "aggregate storage allocation failed",
        )
    })?;
    Ok(values)
}

fn push<T>(values: &mut Vec<T>, value: T) -> Result<(), OracleError> {
    values.try_reserve(1).map_err(|_| {
        tool(
            ToolFailure::Allocation,
            "aggregate traversal allocation failed",
        )
    })?;
    values.push(value);
    Ok(())
}

fn aggregate(value: Scalar) -> Result<usize, OracleError> {
    match value {
        Scalar::Aggregate(id) => Ok(id),
        _ => Err(internal()),
    }
}

impl Arena {
    pub(super) fn new() -> Self {
        Self { slots: Vec::new() }
    }

    pub(super) fn import_record(
        &mut self,
        fields: Vec<Scalar>,
        arrays: &mut arrays::Arena,
    ) -> Result<Scalar, OracleError> {
        self.alloc(Node::Record(fields.into_iter().map(Some).collect()), arrays)
    }
    fn get(&self, id: usize) -> Result<&Node, OracleError> {
        self.slots
            .get(id)
            .and_then(Option::as_ref)
            .ok_or_else(internal)
    }

    fn get_mut(&mut self, id: usize) -> Result<&mut Node, OracleError> {
        self.slots
            .get_mut(id)
            .and_then(Option::as_mut)
            .ok_or_else(internal)
    }

    fn alloc(&mut self, node: Node, arrays: &mut arrays::Arena) -> Result<Scalar, OracleError> {
        self.slots
            .try_reserve(1)
            .map_err(|_| tool(ToolFailure::Allocation, "aggregate arena allocation failed"))?;
        // Children were admitted by their constructors. Admit only this shell.
        arrays.admit_units(node.units())?;
        let id = self.slots.len();
        self.slots.push(Some(node));
        Ok(Scalar::Aggregate(id))
    }

    fn drop_owner(
        &mut self,
        value: Scalar,
        name: &str,
        arrays: &mut arrays::Arena,
        trace: &mut Vec<CleanupEvent>,
    ) -> Result<(), OracleError> {
        if !matches!(value, Scalar::Array(_) | Scalar::Aggregate(_)) {
            return Ok(());
        }
        let mut pending = reserve(1)?;
        pending.push(value);
        let mut length = 0usize;
        while let Some(value) = pending.pop() {
            let units = match value {
                Scalar::Array(id) => arrays.release_owner(id)?,
                Scalar::Aggregate(id) => {
                    let node = self
                        .slots
                        .get_mut(id)
                        .and_then(Option::take)
                        .ok_or_else(internal)?;
                    let units = node.units();
                    arrays.release_units(units)?;
                    match node {
                        Node::Record(fields) => {
                            for field in fields.into_iter().flatten() {
                                push(&mut pending, field)?;
                            }
                        }
                        Node::Variant {
                            payload: Some(payload),
                            ..
                        } => push(&mut pending, payload)?,
                        Node::Variant { payload: None, .. } | Node::Blob { .. } => {}
                    }
                    units
                }
                _ => 0,
            };
            length = length.checked_add(units).ok_or_else(internal)?;
        }
        trace.try_reserve(1).map_err(|_| {
            tool(
                ToolFailure::Allocation,
                "aggregate cleanup trace allocation failed",
            )
        })?;
        trace.push(CleanupEvent {
            name: name.to_owned(),
            length,
        });
        Ok(())
    }

    pub(super) fn value(&self, id: usize, arrays: &arrays::Arena) -> Result<Value, OracleError> {
        enum Task {
            Visit(Scalar),
            Record(usize),
            Variant(usize, bool),
        }
        let mut pending = reserve(1)?;
        pending.push(Task::Visit(Scalar::Aggregate(id)));
        let mut values = Vec::new();
        while let Some(task) = pending.pop() {
            match task {
                Task::Visit(value) => {
                    let value = match value {
                        Scalar::F32(x) => Value::F32(x),
                        Scalar::I32(x) => Value::I32(x),
                        Scalar::U32(x) => Value::U32(x),
                        Scalar::Bool(x) => Value::Bool(x),
                        Scalar::Unit => Value::Unit,
                        Scalar::Array(id) => arrays.value(id)?,
                        Scalar::View(view) => arrays.view_value(view)?,
                        Scalar::Aggregate(id) => match self.get(id)? {
                            Node::Blob { bytes, text } => {
                                let mut owned = reserve(bytes.len())?;
                                owned.extend_from_slice(bytes);
                                if *text {
                                    Value::Text(Rc::new(
                                        String::from_utf8(owned).map_err(|_| internal())?,
                                    ))
                                } else {
                                    Value::Bytes(Rc::new(owned))
                                }
                            }
                            Node::Record(fields) => {
                                push(&mut pending, Task::Record(fields.len()))?;
                                for field in fields.iter().rev() {
                                    push(&mut pending, Task::Visit(field.unwrap_or(Scalar::Unit)))?;
                                }
                                continue;
                            }
                            Node::Variant { tag, payload } => {
                                push(&mut pending, Task::Variant(*tag, payload.is_some()))?;
                                if let Some(payload) = payload {
                                    push(&mut pending, Task::Visit(*payload))?;
                                }
                                continue;
                            }
                        },
                    };
                    push(&mut values, value)?;
                }
                Task::Record(length) => {
                    let start = values.len().checked_sub(length).ok_or_else(internal)?;
                    let mut fields = reserve(length)?;
                    fields.extend(values.drain(start..));
                    push(&mut values, Value::Record(Rc::new(RefCell::new(fields))))?;
                }
                Task::Variant(tag, has_payload) => {
                    let payload = if has_payload {
                        Some(values.pop().ok_or_else(internal)?)
                    } else {
                        None
                    };
                    push(
                        &mut values,
                        Value::Variant(Rc::new(RefCell::new((tag, payload)))),
                    )?;
                }
            }
        }
        if values.len() != 1 {
            return Err(internal());
        }
        values.pop().ok_or_else(internal)
    }
}

pub(super) fn is_text_intrinsic(name: &str) -> bool {
    matches!(
        name,
        "bytes_slice" | "utf8_slice" | "utf8_decode" | "utf8_encode"
    )
}

/// Byte/UTF-8 intrinsics over a borrowed named `Bytes`/`String` owner. Slices
/// and encodings are fresh owners; decoding and text slicing produce
/// `Result<String,Utf8Error>` with `Ok = 0`, `Err = 1` and a payload-free
/// `Utf8Error`. Each fresh shell is admitted against the shared live budget.
pub(super) fn text_intrinsic(
    name: &str,
    args: &[mir::ValueId],
    frame: &Frame,
    span: Span,
    arrays: &mut arrays::Arena,
    arena: &mut Arena,
) -> Result<Scalar, OracleError> {
    let bounds = || {
        OracleError::from(EvalError::Trap {
            category: TrapCategory::IndexOutOfBounds,
            span,
        })
    };
    let (source, rest) = args.split_first().ok_or_else(internal)?;
    let range = rest
        .iter()
        .map(|id| match frame.get(*id)? {
            Scalar::U32(bound) => usize::try_from(bound).map_err(|_| bounds()),
            _ => Err(internal()),
        })
        .collect::<Result<Vec<_>, _>>()?;
    let Node::Blob { bytes, text } = arena.get(aggregate(frame.get(*source)?)?)? else {
        return Err(internal());
    };
    let (bytes, text) = (bytes.as_slice(), *text);
    let fresh = |part: &[u8]| -> Result<Vec<u8>, OracleError> {
        let mut owned = reserve(part.len())?;
        owned.extend_from_slice(part);
        Ok(owned)
    };
    let decoded: Option<Vec<u8>> = match (name, text, range.as_slice()) {
        ("bytes_slice", false, [start, end]) => {
            let part = fresh(bytes.get(*start..*end).ok_or_else(bounds)?)?;
            return arena.alloc(
                Node::Blob {
                    bytes: part,
                    text: false,
                },
                arrays,
            );
        }
        ("utf8_encode", true, []) => {
            let part = fresh(bytes)?;
            return arena.alloc(
                Node::Blob {
                    bytes: part,
                    text: false,
                },
                arrays,
            );
        }
        ("utf8_decode", false, []) => std::str::from_utf8(bytes)
            .ok()
            .map(|t| fresh(t.as_bytes()))
            .transpose()?,
        ("utf8_slice", true, [start, end]) => {
            if start > end || *end > bytes.len() {
                return Err(bounds());
            }
            let whole = std::str::from_utf8(bytes).map_err(|_| internal())?;
            whole
                .get(*start..*end)
                .map(|t| fresh(t.as_bytes()))
                .transpose()?
        }
        _ => return Err(internal()),
    };
    let (tag, payload) = match decoded {
        Some(part) => (
            0,
            arena.alloc(
                Node::Blob {
                    bytes: part,
                    text: true,
                },
                arrays,
            )?,
        ),
        None => (
            1,
            arena.alloc(
                Node::Variant {
                    tag: 0,
                    payload: None,
                },
                arrays,
            )?,
        ),
    };
    arena.alloc(
        Node::Variant {
            tag,
            payload: Some(payload),
        },
        arrays,
    )
}

/// Admission follows exact verified declarations, including materialized Result
/// specializations. Unknown nominal types (views, floats, type-only records)
/// remain unsupported; names never substitute for declaration contents.
pub(super) fn accepts_type(ty: &Type, program: &mir::Program) -> bool {
    let mut pending = vec![ty.clone()];
    let mut seen = BTreeSet::new();
    while let Some(ty) = pending.pop() {
        match ty {
            Type::ArrayBool => return false,
            Type::F32
            | Type::ArrayF32
            | Type::I32
            | Type::U32
            | Type::Bool
            | Type::Unit
            | Type::ArrayI32
            | Type::ArrayU32 => {}
            // The GNET net contract has no binary64 words yet (v0.2 f64math).
            Type::F64 | Type::ArrayF64 => return false,
            // Utf8Error is the payload-free error of the checked UTF-8 intrinsics.
            Type::Named(name) if matches!(name.as_str(), "Bytes" | "String" | "Utf8Error") => {}
            // Integer borrowed views (`View<T>`, `MutView<T>`); f32 views do not check.
            Type::Named(name)
                if crate::front::view_parts(&name)
                    .is_some_and(|(element, _)| matches!(element, Type::I32 | Type::U32)) => {}
            Type::Named(name) => {
                if !seen.insert(name.clone()) {
                    continue;
                }
                if let Some(record) = program.records.iter().find(|record| record.name == name) {
                    pending.extend(record.fields.iter().map(|field| field.ty.clone()));
                } else if let Some(variant) =
                    program.variants.iter().find(|variant| variant.name == name)
                {
                    pending.extend(variant.arms.iter().filter_map(|arm| arm.payload.clone()));
                } else if let Some((ok, err)) = crate::front::result_parts(&name) {
                    // Intrinsic results such as Result<String,Utf8Error> are not declared.
                    pending.extend([ok, err]);
                } else {
                    return false;
                }
            }
        }
    }
    true
}

pub(super) fn cleanup(
    frame: &mut Frame,
    names: &[String],
    arrays: &mut arrays::Arena,
    arena: &mut Arena,
    trace: &mut Vec<CleanupEvent>,
) -> Result<(), OracleError> {
    for name in names {
        if let Some(slot) = frame.places.remove(name)
            && slot.owned
            && let Some(value) = slot.value
        {
            arena.drop_owner(value, name, arrays, trace)?;
        }
    }
    Ok(())
}

// Keep the dispatch hook aligned with the parent oracle contexts.
#[allow(clippy::too_many_arguments)]
pub(super) fn operation(
    op: &mir::Operation,
    frame: &mut Frame,
    _ty: Option<&Type>,
    span: Span,
    program: &mir::Program,
    arrays: &mut arrays::Arena,
    arena: &mut Arena,
    trace: &mut Vec<CleanupEvent>,
) -> Result<Option<Scalar>, OracleError> {
    use mir::Operation as O;
    Ok(match op {
        O::Blob { bytes, text } => {
            let mut owned = reserve(bytes.len())?;
            owned.extend_from_slice(bytes);
            if *text && std::str::from_utf8(&owned).is_err() {
                return Err(internal());
            }
            Some(arena.alloc(
                Node::Blob {
                    bytes: owned,
                    text: *text,
                },
                arrays,
            )?)
        }
        O::RecordRef { name, .. } => {
            let value = frame
                .places
                .get(name)
                .and_then(|slot| slot.value)
                .ok_or_else(internal)?;
            arena.get(aggregate(value)?)?;
            Some(value)
        }
        O::MoveField { name, index } => {
            let id = aggregate(
                frame
                    .places
                    .get(name)
                    .and_then(|slot| slot.value)
                    .ok_or_else(internal)?,
            )?;
            let Node::Record(fields) = arena.get_mut(id)? else {
                return Err(internal());
            };
            Some(
                fields
                    .get_mut(*index)
                    .and_then(Option::take)
                    .ok_or_else(internal)?,
            )
        }
        O::Record { name, fields } => {
            let declaration = program
                .records
                .iter()
                .find(|record| record.name == *name)
                .ok_or_else(internal)?;
            let mut values = reserve(declaration.fields.len())?;
            values.resize(declaration.fields.len(), None);
            for (index, id) in fields {
                let field = values.get_mut(*index).ok_or_else(internal)?;
                if field.is_some() {
                    return Err(internal());
                }
                *field = Some(frame.get(*id)?);
            }
            if values.iter().any(Option::is_none) {
                return Err(internal());
            }
            Some(arena.alloc(Node::Record(values), arrays)?)
        }
        O::Field { record, index } => {
            let Node::Record(fields) = arena.get(aggregate(frame.get(*record)?)?)? else {
                return Err(internal());
            };
            Some(fields.get(*index).copied().flatten().ok_or_else(internal)?)
        }
        O::Variant { tag, payload, .. } => {
            let payload = payload.map(|id| frame.get(id)).transpose()?;
            Some(arena.alloc(Node::Variant { tag: *tag, payload }, arrays)?)
        }
        O::VariantTag(value) => {
            let Node::Variant { tag, .. } = arena.get(aggregate(frame.get(*value)?)?)? else {
                return Err(internal());
            };
            Some(Scalar::U32(u32::try_from(*tag).map_err(|_| internal())?))
        }
        O::VariantPayload { value, tag } => {
            let Node::Variant {
                tag: actual,
                payload,
            } = arena.get(aggregate(frame.get(*value)?)?)?
            else {
                return Err(internal());
            };
            if actual != tag {
                return Err(internal());
            }
            Some(payload.ok_or_else(internal)?)
        }
        O::VariantTakePayload { value, tag } => {
            let Node::Variant {
                tag: actual,
                payload,
            } = arena.get_mut(aggregate(frame.get(*value)?)?)?
            else {
                return Err(internal());
            };
            if actual != tag {
                return Err(internal());
            }
            Some(payload.take().ok_or_else(internal)?)
        }
        O::DropValue(value) => {
            arena.drop_owner(frame.get(*value)?, "<result>", arrays, trace)?;
            None
        }
        O::Length(name) => {
            let id = aggregate(
                frame
                    .places
                    .get(name)
                    .and_then(|slot| slot.value)
                    .ok_or_else(internal)?,
            )?;
            let Node::Blob { bytes, .. } = arena.get(id)? else {
                return Err(internal());
            };
            Some(Scalar::U32(
                u32::try_from(bytes.len()).map_err(|_| internal())?,
            ))
        }
        O::Index { array, index } => {
            let id = aggregate(frame.get(*array)?)?;
            let Scalar::U32(index) = frame.get(*index)? else {
                return Err(internal());
            };
            let Node::Blob { bytes, text: false } = arena.get(id)? else {
                return Err(internal());
            };
            let bounds = || {
                OracleError::from(EvalError::Trap {
                    category: TrapCategory::IndexOutOfBounds,
                    span,
                })
            };
            let index = usize::try_from(index).map_err(|_| bounds())?;
            Some(Scalar::U32(u32::from(
                *bytes.get(index).ok_or_else(bounds)?,
            )))
        }
        _ => return Err(internal()),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn issue_gnet_02_aggregates_deep_cleanup_uses_flat_handles_and_one_event() {
        let mut arrays = arrays::Arena::new(20_003);
        let mut arena = Arena::new();
        let mut value = arena
            .alloc(
                Node::Blob {
                    bytes: vec![1, 2, 3],
                    text: false,
                },
                &mut arrays,
            )
            .unwrap();
        for _ in 0..10_000 {
            value = arena
                .alloc(
                    Node::Variant {
                        tag: 0,
                        payload: Some(value),
                    },
                    &mut arrays,
                )
                .unwrap();
        }
        let mut trace = Vec::new();
        arena
            .drop_owner(value, "outer", &mut arrays, &mut trace)
            .unwrap();
        assert_eq!(
            trace,
            vec![CleanupEvent {
                name: "outer".to_owned(),
                length: 20_003
            }]
        );
        // Shared budget is entirely reusable after the remaining tree is freed.
        arrays.admit_units(20_003).unwrap();
        arrays.release_units(20_003).unwrap();
        assert!(arena.get(10_000).is_err());
    }
}
