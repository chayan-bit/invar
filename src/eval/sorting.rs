//! Independent stable comparison-sort oracle. Production radix code is separate.
use super::{EvalError, TrapCategory, Value, admit_live};
use crate::{front::Span, numeric_contract::F32Bits};
use std::{cell::RefCell, ops::Range, rc::Rc};

fn invalid() -> EvalError {
    EvalError::tool("INTERNAL_TYPE", "invalid checked sort operands")
}
fn reserve<T>(length: usize) -> Result<Vec<T>, EvalError> {
    let mut out = Vec::new();
    out.try_reserve_exact(length)
        .map_err(|_| EvalError::tool("RESOURCE_LIMIT", "sort allocation failed"))?;
    Ok(out)
}
fn canonical(value: F32Bits) -> F32Bits {
    if value.is_nan() { F32Bits::NAN } else { value }
}
fn reorder<T: Copy>(input: &[T], order: &[usize]) -> Result<Vec<T>, EvalError> {
    let mut output = reserve(order.len())?;
    for &index in order {
        output.push(*input.get(index).ok_or_else(invalid)?);
    }
    Ok(output)
}

enum Prepared {
    I32(Vec<i32>),
    U32(Vec<u32>),
    F32(Vec<F32Bits>),
    Bool(Vec<bool>),
}
impl Prepared {
    fn new(value: &Value, order: &[usize]) -> Result<Self, EvalError> {
        Ok(match value {
            Value::ArrayI32(v) => Self::I32(reorder(&v.borrow(), order)?),
            Value::ArrayU32(v) => Self::U32(reorder(&v.borrow(), order)?),
            Value::ArrayF32(v) => Self::F32(reorder(&v.borrow(), order)?),
            Value::ArrayBool(v) => Self::Bool(reorder(&v.borrow(), order)?),
            _ => return Err(invalid()),
        })
    }
    fn commit(self, value: &Value) -> Result<(), EvalError> {
        match (self, value) {
            (Self::I32(out), Value::ArrayI32(v)) => *v.borrow_mut() = out,
            (Self::U32(out), Value::ArrayU32(v)) => *v.borrow_mut() = out,
            (Self::F32(out), Value::ArrayF32(v)) => *v.borrow_mut() = out,
            (Self::Bool(out), Value::ArrayBool(v)) => *v.borrow_mut() = out,
            _ => return Err(invalid()),
        }
        Ok(())
    }
}

pub(super) fn invoke(
    name: &str,
    args: &[Value],
    span: Span,
    live: &mut usize,
) -> Result<Value, EvalError> {
    let modes = crate::check::sort_modes(name).ok_or_else(invalid)?;
    if args.len() != modes.len() {
        return Err(invalid());
    }
    let keys = args.first().ok_or_else(invalid)?;
    let length = super::view_len(keys).ok_or_else(invalid)? as usize;
    let trap = || EvalError::Trap {
        category: TrapCategory::IndexOutOfBounds,
        span,
    };
    if name == "sort_by_key" && super::view_len(&args[1]) != Some(length as u32) {
        return Err(trap());
    }
    let ranges: Vec<Range<usize>> = if name == "segmented_sort" {
        let Value::ArrayU32(offsets) = &args[1] else {
            return Err(invalid());
        };
        let offsets = offsets.borrow();
        if offsets.first() != Some(&0) {
            return Err(trap());
        }
        let mut previous = 0;
        for &offset in offsets.iter() {
            if offset < previous || u64::from(offset) > length as u64 {
                return Err(trap());
            }
            previous = offset;
        }
        if offsets.last().copied().map(u64::from) != Some(length as u64) {
            return Err(trap());
        }
        let mut ranges = reserve(offsets.len() - 1)?;
        ranges.extend(
            offsets
                .windows(2)
                .map(|pair| pair[0] as usize..pair[1] as usize),
        );
        ranges
    } else {
        let mut ranges = reserve(1)?;
        ranges.push(0..length);
        ranges
    };
    let mut order = reserve(length)?;
    order.extend(0..length);
    for range in &ranges {
        match keys {
            Value::ArrayI32(v) => {
                let v = v.borrow();
                order[range.clone()].sort_by_key(|&i| v[i]);
            }
            Value::ArrayU32(v) => {
                let v = v.borrow();
                order[range.clone()].sort_by_key(|&i| v[i]);
            }
            Value::ArrayF32(v) => {
                let v = v.borrow();
                order[range.clone()].sort_by(|&a, &b| {
                    f32::from_bits(canonical(v[a]).to_bits())
                        .total_cmp(&f32::from_bits(canonical(v[b]).to_bits()))
                });
            }
            _ => return Err(invalid()),
        }
    }
    if name == "argsort" {
        let mut output = reserve(length)?;
        for index in order {
            output.push(u32::try_from(index).map_err(|_| invalid())?);
        }
        admit_live(live, length)?;
        return Ok(Value::ArrayU32(Rc::new(RefCell::new(output))));
    }
    let mut prepared = Prepared::new(keys, &order)?;
    if let Prepared::F32(v) = &mut prepared {
        for range in &ranges {
            for key in &mut v[range.clone()] {
                *key = canonical(*key);
            }
        }
    }
    // All validation and fallible staging finish before the first write.
    let payload = (name == "sort_by_key")
        .then(|| Prepared::new(&args[1], &order))
        .transpose()?;
    prepared.commit(keys)?;
    if let Some(payload) = payload {
        payload.commit(&args[1])?;
    }
    Ok(Value::Unit)
}

#[cfg(test)]
mod tests {
    use super::*;
    fn floats(bits: &[u32]) -> Value {
        Value::ArrayF32(Rc::new(RefCell::new(
            bits.iter().map(|&b| F32Bits::from_bits(b)).collect(),
        )))
    }
    #[test]
    fn raw_nan_order_stability_and_no_argsort_write() {
        let bits = [
            0xffc00001, 0, 0x80000000, 0x7f800001, 0xff800000, 1, 0x80000001,
        ];
        let input = floats(&bits);
        let Value::ArrayF32(original) = &input else {
            panic!()
        };
        let original = Rc::clone(original);
        let span = Span { start: 1, end: 2 };
        let ids = invoke("argsort", &[input], span, &mut 0).unwrap();
        let Value::ArrayU32(ids) = ids else { panic!() };
        assert_eq!(*ids.borrow(), [4, 6, 2, 1, 5, 0, 3]);
        assert_eq!(
            original
                .borrow()
                .iter()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>(),
            bits
        );
        let input = floats(&bits);
        let Value::ArrayF32(storage) = &input else {
            panic!()
        };
        let storage = Rc::clone(storage);
        invoke("sort", &[input], span, &mut 0).unwrap();
        assert_eq!(
            storage
                .borrow()
                .iter()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>(),
            [
                0xff800000, 0x80000001, 0x80000000, 0, 1, 0x7fc00000, 0x7fc00000
            ]
        );
    }
    #[test]
    fn invalid_late_offset_preserves_every_key_bit() {
        let input = floats(&[0xffc00001, 0x3f800000, 0]);
        let Value::ArrayF32(storage) = &input else {
            panic!()
        };
        let storage = Rc::clone(storage);
        let before = storage.borrow().clone();
        let offsets = Value::ArrayU32(Rc::new(RefCell::new(vec![0, 2, 3, 2])));
        assert!(matches!(
            invoke(
                "segmented_sort",
                &[input, offsets],
                Span { start: 4, end: 5 },
                &mut 0
            ),
            Err(EvalError::Trap {
                category: TrapCategory::IndexOutOfBounds,
                ..
            })
        ));
        assert_eq!(*storage.borrow(), before);
    }
}
