//! Interpreter operation dispatch, intrinsics and text results.

use super::*;

impl Interpreter<'_> {
    pub(super) fn operation(
        &mut self,
        function: &str,
        op: &Operation,
        output_type: Option<&Type>,
        span: Span,
        frame: &mut Frame,
    ) -> Result<Option<Value>, EvalError> {
        Ok(match op {
            Operation::Tick => {
                self.tick()?;
                None
            }
            Operation::I32(value) => Some(Value::I32(*value)),
            Operation::U32(value) => Some(Value::U32(*value)),
            Operation::F32(value) => Some(Value::F32(*value)),
            Operation::F64(value) => Some(Value::F64(*value)),
            Operation::Bool(value) => Some(Value::Bool(*value)),
            Operation::Blob { bytes, text } => {
                admit_live(&mut self.live_elements, bytes.len())?;
                Some(if *text {
                    Value::Text(Rc::new(
                        String::from_utf8(bytes.clone())
                            .map_err(|_| tool("MIR_TYPE", "verified UTF-8 literal invalid"))?,
                    ))
                } else {
                    Value::Bytes(Rc::new(bytes.clone()))
                })
            }
            Operation::Read(name) => Some(copied(frame.slot(name)?)),
            Operation::Move(name) => Some(frame.move_owner(name)?),
            Operation::MoveField { name, index } => Some(frame.move_field(name, *index)?),
            Operation::ArrayRef { name, .. } => Some(frame.array_ref(name)?),
            Operation::RecordRef { name, .. } => Some(frame.record_ref(name)?),
            Operation::Variant { tag, payload, .. } => {
                let payload = payload.map(|id| frame.value(id)).transpose()?;
                admit_live(&mut self.live_elements, 2)?;
                Some(Value::Variant(Rc::new(RefCell::new((*tag, payload)))))
            }
            Operation::VariantTag(value) => {
                let Value::Variant(values) = frame.value(*value)? else {
                    return Err(tool("MIR_TYPE", "variant tag base is not variant"));
                };
                Some(Value::U32(values.borrow().0 as u32))
            }
            Operation::VariantPayload { value, tag } => {
                let Value::Variant(values) = frame.value(*value)? else {
                    return Err(tool("MIR_TYPE", "variant payload base is not variant"));
                };
                let values = values.borrow();
                if values.0 != *tag {
                    return Err(tool("MIR_TYPE", "variant payload tag mismatch"));
                }
                Some(
                    values
                        .1
                        .as_ref()
                        .map(copied)
                        .ok_or_else(|| tool("MIR_TYPE", "variant payload missing"))?,
                )
            }
            Operation::VariantTakePayload { value, tag } => {
                let Value::Variant(storage) = frame.value(*value)? else {
                    return Err(tool("MIR_TYPE", "Result payload base missing"));
                };
                let mut storage = storage.borrow_mut();
                if storage.0 != *tag {
                    return Err(tool("MIR_TYPE", "Result payload tag mismatch"));
                }
                Some(
                    storage
                        .1
                        .take()
                        .ok_or_else(|| tool("MIR_TYPE", "Result payload missing"))?,
                )
            }
            Operation::DropValue(value) => {
                let owned = frame.value(*value)?;
                record_cleanup(&mut self.trace, &mut self.live_elements, "<result>", &owned)?;
                None
            }
            Operation::Record { name, fields } => {
                let declaration = self
                    .program
                    .records
                    .iter()
                    .find(|record| record.name == *name)
                    .ok_or_else(|| tool("MIR_TYPE", "verified record declaration missing"))?;
                let mut values = std::iter::repeat_with(|| Value::Unit)
                    .take(declaration.fields.len())
                    .collect::<Vec<_>>();
                for (index, id) in fields {
                    values[*index] = frame.value(*id)?;
                }
                admit_live(&mut self.live_elements, values.len())?;
                Some(Value::Record(Rc::new(RefCell::new(values))))
            }
            Operation::Field { record, index } => {
                let Value::Record(values) = frame.value(*record)? else {
                    return Err(tool("MIR_TYPE", "field base is not a record"));
                };
                let value = values
                    .borrow()
                    .get(*index)
                    .map(copied)
                    .ok_or_else(|| tool("MIR_TYPE", "record field missing"))?;
                Some(value)
            }
            Operation::Array(items) => {
                if items.len() > MAX_ARRAY_LEN {
                    return Err(tool("ARRAY_SIZE_LIMIT", "array length limit exceeded"));
                }
                match output_type {
                    Some(Type::ArrayI32) => {
                        let mut values = Vec::new();
                        values
                            .try_reserve_exact(items.len())
                            .map_err(|_| tool("RESOURCE_LIMIT", "array allocation failed"))?;
                        for id in items {
                            let Value::I32(value) = frame.value(*id)? else {
                                return Err(tool("MIR_TYPE", "invalid i32 array element"));
                            };
                            values.push(value);
                        }
                        admit_live(&mut self.live_elements, values.len())?;
                        Some(Value::ArrayI32(Rc::new(RefCell::new(values))))
                    }
                    Some(Type::ArrayU32) => {
                        let mut values = Vec::new();
                        values
                            .try_reserve_exact(items.len())
                            .map_err(|_| tool("RESOURCE_LIMIT", "array allocation failed"))?;
                        for id in items {
                            let Value::U32(value) = frame.value(*id)? else {
                                return Err(tool("MIR_TYPE", "invalid u32 array element"));
                            };
                            values.push(value);
                        }
                        admit_live(&mut self.live_elements, values.len())?;
                        Some(Value::ArrayU32(Rc::new(RefCell::new(values))))
                    }
                    Some(Type::ArrayF32) => {
                        let mut values = Vec::new();
                        values
                            .try_reserve_exact(items.len())
                            .map_err(|_| tool("RESOURCE_LIMIT", "array allocation failed"))?;
                        for id in items {
                            let Value::F32(value) = frame.value(*id)? else {
                                return Err(tool("MIR_TYPE", "invalid u32 array element"));
                            };
                            values.push(value);
                        }
                        admit_live(&mut self.live_elements, values.len())?;
                        Some(Value::ArrayF32(Rc::new(RefCell::new(values))))
                    }
                    Some(Type::ArrayF64) => {
                        let mut values = Vec::new();
                        values
                            .try_reserve_exact(items.len())
                            .map_err(|_| tool("RESOURCE_LIMIT", "array allocation failed"))?;
                        for id in items {
                            let Value::F64(value) = frame.value(*id)? else {
                                return Err(tool("MIR_TYPE", "invalid f64 array element"));
                            };
                            values.push(value);
                        }
                        admit_live(&mut self.live_elements, values.len())?;
                        Some(Value::ArrayF64(Rc::new(RefCell::new(values))))
                    }
                    Some(Type::ArrayBool) => {
                        let mut values = Vec::new();
                        values
                            .try_reserve_exact(items.len())
                            .map_err(|_| tool("RESOURCE_LIMIT", "array allocation failed"))?;
                        for id in items {
                            let Value::Bool(value) = frame.value(*id)? else {
                                return Err(tool("MIR_TYPE", "invalid u32 array element"));
                            };
                            values.push(value);
                        }
                        admit_live(&mut self.live_elements, values.len())?;
                        Some(Value::ArrayBool(Rc::new(RefCell::new(values))))
                    }
                    _ => return Err(tool("MIR_TYPE", "invalid array output type")),
                }
            }
            Operation::Unary {
                op,
                value,
                operator,
            } => Some(unary(*op, frame.value(*value)?, *operator)?),
            Operation::Binary {
                op,
                left,
                right,
                operator,
            } => Some(binary(
                *op,
                frame.value(*left)?,
                frame.value(*right)?,
                *operator,
            )?),
            Operation::Index { array, index } => {
                let array = frame.value(*array)?;
                let Value::U32(index) = frame.value(*index)? else {
                    return Err(tool("MIR_TYPE", "index must be u32"));
                };
                Some(indexed(&array, index, span)?)
            }
            Operation::Length(name) => {
                let length = array_length(frame.slot(name)?)?;
                Some(Value::U32(mir_len_u32(length)?))
            }
            Operation::Call { name, args } => {
                let values = args
                    .iter()
                    .map(|id| frame.value(*id))
                    .collect::<Result<Vec<_>, _>>()?;
                if crate::spatial::result(name).is_some() {
                    Some(if let Some(provider) = &mut self.sorts {
                        provider(function, name, &values, span, &mut self.live_elements)?
                    } else {
                        crate::spatial::invoke(name, &values, span, &mut self.live_elements)?
                    })
                } else if crate::check::sort_modes(name).is_some() {
                    Some(if let Some(provider) = &mut self.sorts {
                        provider(function, name, &values, span, &mut self.live_elements)?
                    } else {
                        crate::sorting::invoke(name, &values, span, &mut self.live_elements, 1)?
                    })
                } else if matches!(name.as_str(), "view" | "view_mut") {
                    let [source, Value::U32(start), Value::U32(end), rest @ ..] = values.as_slice()
                    else {
                        return Err(tool("MIR_TYPE", "invalid verified view operands"));
                    };
                    let stride = match rest {
                        [] => 1,
                        [Value::U32(value)] => *value,
                        _ => return Err(tool("MIR_TYPE", "invalid verified view stride")),
                    };
                    Some(make_view(
                        source,
                        *start,
                        *end,
                        stride,
                        name == "view_mut",
                        span,
                    )?)
                } else if matches!(
                    name.as_str(),
                    "bytes_slice" | "utf8_slice" | "utf8_decode" | "utf8_encode"
                ) {
                    Some(self.text_intrinsic(name, &values, span)?)
                } else if crate::eval::floating::handles(name) {
                    Some(crate::eval::floating::intrinsic(
                        name,
                        &values,
                        &mut self.live_elements,
                        span,
                    )?)
                } else if let Some((numerator, denominator)) = crate::check::scale_call(name) {
                    let [value] = <[Value; 1]>::try_from(values)
                        .map_err(|_| tool("MIR_TYPE", "checked scale arity mismatch"))?;
                    Some(crate::eval::checked_scale(
                        value,
                        numerator,
                        denominator,
                        span,
                    )?)
                } else if let Some(degrees_to_radians) = crate::check::angle_call(name) {
                    let [value] = <[Value; 1]>::try_from(values)
                        .map_err(|_| tool("MIR_TYPE", "checked angle arity mismatch"))?;
                    Some(crate::eval::checked_angle(value, degrees_to_radians, span)?)
                } else {
                    Some(self.invoke(name, values).map_err(|error| match error {
                        EvalError::Trap { category, .. } if crate::check::is_frame_helper(name) => {
                            EvalError::Trap { category, span }
                        }
                        error => error,
                    })?)
                }
            }
            Operation::Bind { name, value } => {
                frame.slots.insert(
                    name.clone(),
                    Slot {
                        value: Some(frame.value(*value)?),
                        owned: true,
                    },
                );
                None
            }
            Operation::Store { name, value } => {
                let new = frame.value(*value)?;
                let slot = frame
                    .slots
                    .get_mut(name)
                    .ok_or_else(|| tool("MIR_PLACE", "missing store place"))?;
                let old = slot
                    .value
                    .replace(new)
                    .ok_or_else(|| tool("MIR_PLACE", "moved store place"))?;
                record_cleanup(&mut self.trace, &mut self.live_elements, name, &old)?;
                None
            }
            Operation::StoreIndex { name, index, value } => {
                let Value::U32(index) = frame.value(*index)? else {
                    return Err(tool("MIR_TYPE", "index must be u32"));
                };
                let new = frame.value(*value)?;
                store_index(frame.slot(name)?, index, new, span)?;
                None
            }
            Operation::Drop(names) => {
                frame.cleanup(names, &mut self.trace, &mut self.live_elements)?;
                None
            }
            Operation::ForNext(value) => {
                let Value::U32(value) = frame.value(*value)? else {
                    return Err(tool("MIR_TYPE", "for step requires u32"));
                };
                Some(Value::U32(
                    value
                        .checked_add(1)
                        .ok_or_else(|| tool("MIR_FOR", "for increment overflow"))?,
                ))
            }
        })
    }

    fn invoke(&mut self, name: &str, args: Vec<Value>) -> Result<Value, EvalError> {
        match (name, args.as_slice()) {
            ("wrapping_add", [Value::U32(a), Value::U32(b)]) => Ok(Value::U32(a.wrapping_add(*b))),
            ("wrapping_mul", [Value::U32(a), Value::U32(b)]) => Ok(Value::U32(a.wrapping_mul(*b))),
            ("filled_i32", [Value::U32(length), Value::I32(value)]) => {
                filled(&mut self.live_elements, *length, Value::I32(*value))
            }
            ("filled_u32", [Value::U32(length), Value::U32(value)]) => {
                filled(&mut self.live_elements, *length, Value::U32(*value))
            }
            _ => self.call(name, args),
        }
    }

    fn text_intrinsic(
        &mut self,
        name: &str,
        args: &[Value],
        span: Span,
    ) -> Result<Value, EvalError> {
        let invalid = || trap(TrapCategory::IndexOutOfBounds, span);
        match (name, args) {
            ("bytes_slice", [Value::Bytes(value), Value::U32(start), Value::U32(end)]) => {
                let part = value
                    .get(*start as usize..*end as usize)
                    .ok_or_else(invalid)?;
                admit_live(&mut self.live_elements, part.len())?;
                Ok(Value::Bytes(Rc::new(part.to_vec())))
            }
            ("utf8_encode", [Value::Text(value)]) => {
                admit_live(&mut self.live_elements, value.len())?;
                Ok(Value::Bytes(Rc::new(value.as_bytes().to_vec())))
            }
            ("utf8_decode", [Value::Bytes(value)]) => {
                let text = std::str::from_utf8(value).ok();
                self.text_result(text)
            }
            ("utf8_slice", [Value::Text(value), Value::U32(start), Value::U32(end)]) => {
                if start > end || *end as usize > value.len() {
                    return Err(invalid());
                }
                self.text_result(value.get(*start as usize..*end as usize))
            }
            _ => Err(tool("MIR_TYPE", "verified text intrinsic operands invalid")),
        }
    }

    fn text_result(&mut self, text: Option<&str>) -> Result<Value, EvalError> {
        admit_live(&mut self.live_elements, 2 + text.map_or(2, str::len))?;
        let (tag, payload) = match text {
            Some(value) => (0, Value::Text(Rc::new(value.to_owned()))),
            None => (1, Value::Variant(Rc::new(RefCell::new((0, None))))),
        };
        Ok(Value::Variant(Rc::new(RefCell::new((tag, Some(payload))))))
    }
}
