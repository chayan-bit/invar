//! Array element access, fills and scalar operators on MIR values.

use super::*;

pub(super) fn array_length(value: &Value) -> Result<usize, EvalError> {
    if let Some(length) = view_len(value) {
        return usize::try_from(length)
            .map_err(|_| tool("MIR_SIZE", "view length exceeds host size"));
    }
    match value {
        Value::ArrayI32(values) => Ok(values
            .try_borrow()
            .map_err(|_| tool("MIR_LOAN", "array borrow conflict"))?
            .len()),
        Value::ArrayU32(values) => Ok(values
            .try_borrow()
            .map_err(|_| tool("MIR_LOAN", "array borrow conflict"))?
            .len()),
        Value::ArrayF32(values) => Ok(values
            .try_borrow()
            .map_err(|_| tool("MIR_LOAN", "array borrow conflict"))?
            .len()),
        Value::ArrayF64(values) => Ok(values
            .try_borrow()
            .map_err(|_| tool("MIR_LOAN", "array borrow conflict"))?
            .len()),
        Value::ArrayBool(values) => Ok(values
            .try_borrow()
            .map_err(|_| tool("MIR_LOAN", "array borrow conflict"))?
            .len()),
        _ => Err(tool("MIR_TYPE", "expected array")),
    }
}

pub(super) fn mir_len_u32(length: usize) -> Result<u32, EvalError> {
    u32::try_from(length).map_err(|_| tool("MIR_SIZE", "array length exceeds u32"))
}

pub(super) fn mir_host_index(index: u32, host_max: u64) -> Result<usize, EvalError> {
    if u64::from(index) > host_max {
        return Err(tool("MIR_INDEX", "index exceeds host size"));
    }
    usize::try_from(index).map_err(|_| tool("MIR_INDEX", "index exceeds host size"))
}

pub(super) fn indexed(array: &Value, index: u32, span: Span) -> Result<Value, EvalError> {
    let index = mir_host_index(index, usize::MAX as u64)?;
    let value = match array {
        Value::ArrayI32(values) => values
            .try_borrow()
            .map_err(|_| tool("MIR_LOAN", "array borrow conflict"))?
            .get(index)
            .copied()
            .map(Value::I32),
        Value::ArrayU32(values) => values
            .try_borrow()
            .map_err(|_| tool("MIR_LOAN", "array borrow conflict"))?
            .get(index)
            .copied()
            .map(Value::U32),
        Value::ArrayF32(values) => values
            .try_borrow()
            .map_err(|_| tool("MIR_LOAN", "array borrow conflict"))?
            .get(index)
            .copied()
            .map(Value::F32),
        Value::ArrayF64(values) => values
            .try_borrow()
            .map_err(|_| tool("MIR_LOAN", "array borrow conflict"))?
            .get(index)
            .copied()
            .map(Value::F64),
        Value::ArrayBool(values) => values
            .try_borrow()
            .map_err(|_| tool("MIR_LOAN", "array borrow conflict"))?
            .get(index)
            .copied()
            .map(Value::Bool),
        Value::Bytes(values) => values
            .get(index)
            .copied()
            .map(|byte| Value::U32(u32::from(byte))),
        Value::ViewI32(values, range) => values
            .try_borrow()
            .map_err(|_| tool("MIR_LOAN", "view borrow conflict"))?
            .get(view_index(*range, index as u32, span)?)
            .copied()
            .map(Value::I32),
        Value::ViewU32(values, range) => values
            .try_borrow()
            .map_err(|_| tool("MIR_LOAN", "view borrow conflict"))?
            .get(view_index(*range, index as u32, span)?)
            .copied()
            .map(Value::U32),
        _ => return Err(tool("MIR_TYPE", "indexed nonarray")),
    };
    value.ok_or_else(|| trap(TrapCategory::IndexOutOfBounds, span))
}

pub(super) fn store_index(
    array: &Value,
    index: u32,
    new: Value,
    span: Span,
) -> Result<(), EvalError> {
    let index = mir_host_index(index, usize::MAX as u64)?;
    let exists = match (array, new) {
        (Value::ArrayI32(values), Value::I32(value)) => values
            .try_borrow_mut()
            .map_err(|_| tool("MIR_LOAN", "array borrow conflict"))?
            .get_mut(index)
            .map(|slot| *slot = value)
            .is_some(),
        (Value::ArrayU32(values), Value::U32(value)) => values
            .try_borrow_mut()
            .map_err(|_| tool("MIR_LOAN", "array borrow conflict"))?
            .get_mut(index)
            .map(|slot| *slot = value)
            .is_some(),
        (Value::ArrayF32(values), Value::F32(value)) => values
            .try_borrow_mut()
            .map_err(|_| tool("MIR_LOAN", "array borrow conflict"))?
            .get_mut(index)
            .map(|slot| *slot = value)
            .is_some(),
        (Value::ArrayF64(values), Value::F64(value)) => values
            .try_borrow_mut()
            .map_err(|_| tool("MIR_LOAN", "array borrow conflict"))?
            .get_mut(index)
            .map(|slot| *slot = value)
            .is_some(),
        (Value::ArrayBool(values), Value::Bool(value)) => values
            .try_borrow_mut()
            .map_err(|_| tool("MIR_LOAN", "array borrow conflict"))?
            .get_mut(index)
            .map(|slot| *slot = value)
            .is_some(),
        (Value::ViewI32(values, range), Value::I32(value)) if range.mutable => values
            .try_borrow_mut()
            .map_err(|_| tool("MIR_LOAN", "view borrow conflict"))?
            .get_mut(view_index(*range, index as u32, span)?)
            .map(|slot| *slot = value)
            .is_some(),
        (Value::ViewU32(values, range), Value::U32(value)) if range.mutable => values
            .try_borrow_mut()
            .map_err(|_| tool("MIR_LOAN", "view borrow conflict"))?
            .get_mut(view_index(*range, index as u32, span)?)
            .map(|slot| *slot = value)
            .is_some(),
        _ => return Err(tool("MIR_TYPE", "invalid indexed store")),
    };
    if exists {
        Ok(())
    } else {
        Err(trap(TrapCategory::IndexOutOfBounds, span))
    }
}

pub(super) fn filled(live: &mut usize, length: u32, element: Value) -> Result<Value, EvalError> {
    let length = usize::try_from(length)
        .map_err(|_| tool("ARRAY_SIZE_LIMIT", "array length exceeds host size"))?;
    if length > MAX_ARRAY_LEN {
        return Err(tool("ARRAY_SIZE_LIMIT", "array length limit exceeded"));
    }
    if !matches!(element, Value::I32(_) | Value::U32(_)) {
        return Err(tool("MIR_TYPE", "invalid fill element"));
    }
    admit_live(live, length)?;
    match element {
        Value::I32(value) => {
            let mut items = Vec::new();
            items
                .try_reserve_exact(length)
                .map_err(|_| tool("RESOURCE_LIMIT", "array allocation failed"))?;
            items.resize(length, value);
            Ok(Value::ArrayI32(Rc::new(RefCell::new(items))))
        }
        Value::U32(value) => {
            let mut items = Vec::new();
            items
                .try_reserve_exact(length)
                .map_err(|_| tool("RESOURCE_LIMIT", "array allocation failed"))?;
            items.resize(length, value);
            Ok(Value::ArrayU32(Rc::new(RefCell::new(items))))
        }
        _ => Err(tool("MIR_TYPE", "invalid fill element")),
    }
}

pub(super) fn unary(op: UnaryOp, value: Value, span: Span) -> Result<Value, EvalError> {
    match (op, value) {
        (UnaryOp::Not, Value::Bool(value)) => Ok(Value::Bool(!value)),
        (UnaryOp::Neg, Value::F32(value)) => Ok(Value::F32(value.negate())),
        (UnaryOp::Neg, Value::F64(value)) => Ok(Value::F64(value.negate())),
        (UnaryOp::Neg, Value::I32(value)) => value
            .checked_neg()
            .map(Value::I32)
            .ok_or_else(|| trap(TrapCategory::IntegerOverflow, span)),
        _ => Err(tool("MIR_TYPE", "invalid unary value")),
    }
}

pub(super) fn binary(
    op: BinaryOp,
    left: Value,
    right: Value,
    span: Span,
) -> Result<Value, EvalError> {
    use BinaryOp as Op;
    match (left, right) {
        (Value::F32(a), Value::F32(b)) => crate::eval::floating::binary(a, op, b),
        (Value::F64(a), Value::F64(b)) => crate::eval::floating::binary64(a, op, b),
        (Value::I32(a), Value::I32(b)) => {
            let arithmetic = match op {
                Op::Add => a.checked_add(b),
                Op::Sub => a.checked_sub(b),
                Op::Mul => a.checked_mul(b),
                Op::Div if b != 0 => a.checked_div(b),
                Op::Rem if b != 0 && a == i32::MIN && b == -1 => Some(0),
                Op::Rem if b != 0 => a.checked_rem(b),
                _ => None,
            };
            if matches!(op, Op::Add | Op::Sub | Op::Mul | Op::Div | Op::Rem) {
                if b == 0 && matches!(op, Op::Div | Op::Rem) {
                    return Err(trap(TrapCategory::DivisionByZero, span));
                }
                return arithmetic
                    .map(Value::I32)
                    .ok_or_else(|| trap(TrapCategory::IntegerOverflow, span));
            }
            let compared = match op {
                Op::Eq => a == b,
                Op::Ne => a != b,
                Op::Lt => a < b,
                Op::Le => a <= b,
                Op::Gt => a > b,
                Op::Ge => a >= b,
                _ => return Err(tool("MIR_TYPE", "invalid i32 operator")),
            };
            Ok(Value::Bool(compared))
        }
        (Value::U32(a), Value::U32(b)) => {
            let arithmetic = match op {
                Op::Add => a.checked_add(b),
                Op::Sub => a.checked_sub(b),
                Op::Mul => a.checked_mul(b),
                Op::Div if b != 0 => Some(a / b),
                Op::Rem if b != 0 => Some(a % b),
                _ => None,
            };
            if matches!(op, Op::Add | Op::Sub | Op::Mul | Op::Div | Op::Rem) {
                if b == 0 && matches!(op, Op::Div | Op::Rem) {
                    return Err(trap(TrapCategory::DivisionByZero, span));
                }
                return arithmetic
                    .map(Value::U32)
                    .ok_or_else(|| trap(TrapCategory::IntegerOverflow, span));
            }
            let compared = match op {
                Op::Eq => a == b,
                Op::Ne => a != b,
                Op::Lt => a < b,
                Op::Le => a <= b,
                Op::Gt => a > b,
                Op::Ge => a >= b,
                _ => return Err(tool("MIR_TYPE", "invalid u32 operator")),
            };
            Ok(Value::Bool(compared))
        }
        (Value::Bool(a), Value::Bool(b)) => match op {
            Op::Eq => Ok(Value::Bool(a == b)),
            Op::Ne => Ok(Value::Bool(a != b)),
            _ => Err(tool("MIR_TYPE", "invalid Bool operator")),
        },
        _ => Err(tool("MIR_TYPE", "invalid MIR binary values")),
    }
}
