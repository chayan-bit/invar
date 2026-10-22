//! Evaluator values and borrowed-view arithmetic.

use super::*;

#[derive(Debug, PartialEq, Eq)]
pub enum Value {
    I32(i32),
    U32(u32),
    F32(crate::numeric_contract::F32Bits),
    F64(crate::numeric_contract::F64Bits),
    Bool(bool),
    Unit,
    ArrayI32(Rc<RefCell<Vec<i32>>>),
    ArrayU32(Rc<RefCell<Vec<u32>>>),
    ArrayF32(Rc<RefCell<Vec<crate::numeric_contract::F32Bits>>>),
    ArrayF64(Rc<RefCell<Vec<crate::numeric_contract::F64Bits>>>),
    ArrayBool(Rc<RefCell<Vec<bool>>>),
    Bytes(Rc<Vec<u8>>),
    Text(Rc<String>),
    ViewI32(Rc<RefCell<Vec<i32>>>, ViewRange),
    ViewU32(Rc<RefCell<Vec<u32>>>, ViewRange),
    VoxelGrid(Rc<crate::spatial::Grid>),
    Record(Rc<RefCell<Vec<Value>>>),
    Variant(Rc<RefCell<(usize, Option<Value>)>>),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ViewRange {
    pub start: u32,
    pub len: u32,
    pub stride: u32,
    pub mutable: bool,
}

pub(crate) fn view_len(value: &Value) -> Option<u32> {
    match value {
        Value::ArrayI32(values) => array_len_u32(values.borrow().len()).ok(),
        Value::ArrayU32(values) => array_len_u32(values.borrow().len()).ok(),
        Value::ArrayF32(values) => array_len_u32(values.borrow().len()).ok(),
        Value::ArrayF64(values) => array_len_u32(values.borrow().len()).ok(),
        Value::ArrayBool(values) => array_len_u32(values.borrow().len()).ok(),
        Value::Bytes(values) => array_len_u32(values.len()).ok(),
        Value::Text(value) => array_len_u32(value.len()).ok(),
        Value::ViewI32(_, range) | Value::ViewU32(_, range) => Some(range.len),
        _ => None,
    }
}

pub(crate) fn make_view(
    value: &Value,
    start: u32,
    end: u32,
    stride: u32,
    mutable: bool,
    span: Span,
) -> Result<Value, EvalError> {
    let invalid = || EvalError::Trap {
        category: TrapCategory::IndexOutOfBounds,
        span,
    };
    let base_len = view_len(value)
        .ok_or_else(|| EvalError::tool("INTERNAL_TYPE", "view source is not an array"))?;
    if start > end || end > base_len || stride == 0 {
        return Err(invalid());
    }
    let base = match value {
        Value::ViewI32(_, range) | Value::ViewU32(_, range) => *range,
        _ => ViewRange {
            start: 0,
            len: base_len,
            stride: 1,
            mutable: true,
        },
    };
    if mutable && !base.mutable {
        return Err(EvalError::tool(
            "INTERNAL_BORROW",
            "mutable view of read-only view",
        ));
    }
    let step = base.stride.checked_mul(stride).ok_or_else(invalid)?;
    let origin = base
        .start
        .checked_add(start.checked_mul(base.stride).ok_or_else(invalid)?)
        .ok_or_else(invalid)?;
    let len = (end - start).div_ceil(stride);
    if len > 0 {
        let last = origin
            .checked_add((len - 1).checked_mul(step).ok_or_else(invalid)?)
            .ok_or_else(invalid)?;
        let physical = match value {
            Value::ArrayI32(values) | Value::ViewI32(values, _) => values.borrow().len(),
            Value::ArrayU32(values) | Value::ViewU32(values, _) => values.borrow().len(),
            _ => unreachable!(),
        };
        if u64::from(last) >= physical as u64 {
            return Err(invalid());
        }
    }
    let range = ViewRange {
        start: origin,
        len,
        stride: step,
        mutable,
    };
    Ok(match value {
        Value::ArrayI32(values) | Value::ViewI32(values, _) => {
            Value::ViewI32(Rc::clone(values), range)
        }
        Value::ArrayU32(values) | Value::ViewU32(values, _) => {
            Value::ViewU32(Rc::clone(values), range)
        }
        _ => unreachable!(),
    })
}

pub(crate) fn view_index(range: ViewRange, index: u32, span: Span) -> Result<usize, EvalError> {
    let invalid = || EvalError::Trap {
        category: TrapCategory::IndexOutOfBounds,
        span,
    };
    if index >= range.len {
        return Err(invalid());
    }
    let physical = range
        .start
        .checked_add(index.checked_mul(range.stride).ok_or_else(invalid)?)
        .ok_or_else(invalid)?;
    usize::try_from(physical).map_err(|_| invalid())
}
