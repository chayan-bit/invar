//! Evaluation frames: owner slots, SSA values and cleanup accounting.

use super::*;

pub(super) fn copied(value: &Value) -> Value {
    match value {
        Value::I32(value) => Value::I32(*value),
        Value::U32(value) => Value::U32(*value),
        Value::F32(value) => Value::F32(*value),
        Value::F64(value) => Value::F64(*value),
        Value::Bool(value) => Value::Bool(*value),
        Value::Unit => Value::Unit,
        Value::ArrayI32(values) => Value::ArrayI32(Rc::clone(values)),
        Value::ArrayU32(values) => Value::ArrayU32(Rc::clone(values)),
        Value::ArrayF32(values) => Value::ArrayF32(Rc::clone(values)),
        Value::ArrayF64(values) => Value::ArrayF64(Rc::clone(values)),
        Value::ArrayBool(values) => Value::ArrayBool(Rc::clone(values)),
        Value::Bytes(values) => Value::Bytes(Rc::clone(values)),
        Value::Text(value) => Value::Text(Rc::clone(value)),
        Value::ViewI32(values, range) => Value::ViewI32(Rc::clone(values), *range),
        Value::ViewU32(values, range) => Value::ViewU32(Rc::clone(values), *range),
        Value::VoxelGrid(grid) => Value::VoxelGrid(Rc::clone(grid)),
        Value::Record(values) => Value::Record(Rc::clone(values)),
        Value::Variant(values) => Value::Variant(Rc::clone(values)),
    }
}

pub(super) struct Slot {
    pub(super) value: Option<Value>,
    pub(super) owned: bool,
}

/// Parks unknown work in its run's quarantine slot. When the slot is held or
/// already occupied the work is leaked instead: unknown work is never released.
pub(super) fn park<P>(slot: &RefCell<Option<P>>, pending: P) {
    match slot.try_borrow_mut() {
        Ok(mut slot) if slot.is_none() => *slot = Some(pending),
        _ => std::mem::forget(pending),
    }
}

#[derive(Default)]
pub(super) struct Frame {
    pub(super) slots: HashMap<String, Slot>,
    pub(super) values: HashMap<ValueId, Value>,
}

impl Frame {
    pub(super) fn new(function: &Function, args: Vec<Value>) -> Self {
        let slots = function
            .params
            .iter()
            .zip(args)
            .map(|(param, value)| {
                (
                    param.name.clone(),
                    Slot {
                        value: Some(value),
                        owned: !matches!(param.mode, Some(Mode::Read | Mode::Mut)),
                    },
                )
            })
            .collect();
        Self {
            slots,
            values: HashMap::new(),
        }
    }

    pub(super) fn value(&self, id: ValueId) -> Result<Value, EvalError> {
        self.values
            .get(&id)
            .map(copied)
            .ok_or_else(|| tool("MIR_VALUE", "missing verified MIR value"))
    }

    pub(super) fn slot(&self, name: &str) -> Result<&Value, EvalError> {
        self.slots
            .get(name)
            .and_then(|slot| slot.value.as_ref())
            .ok_or_else(|| tool("MIR_PLACE", "unavailable verified MIR place"))
    }

    pub(super) fn move_owner(&mut self, name: &str) -> Result<Value, EvalError> {
        self.slots
            .get_mut(name)
            .and_then(|slot| slot.value.take())
            .ok_or_else(|| tool("MIR_PLACE", "unavailable verified MIR owner"))
    }

    pub(super) fn move_field(&mut self, name: &str, index: usize) -> Result<Value, EvalError> {
        let Value::Record(fields) = self.slot(name)? else {
            return Err(tool("MIR_TYPE", "field move base is not a record"));
        };
        let mut fields = fields.borrow_mut();
        let field = fields
            .get_mut(index)
            .ok_or_else(|| tool("MIR_TYPE", "field move index missing"))?;
        if matches!(field, Value::Unit) {
            return Err(tool("MIR_TYPE", "field moved twice"));
        }
        Ok(std::mem::replace(field, Value::Unit))
    }

    pub(super) fn array_ref(&self, name: &str) -> Result<Value, EvalError> {
        match self.slot(name)? {
            Value::ArrayI32(values) => Ok(Value::ArrayI32(Rc::clone(values))),
            Value::ArrayU32(values) => Ok(Value::ArrayU32(Rc::clone(values))),
            Value::ArrayF32(values) => Ok(Value::ArrayF32(Rc::clone(values))),
            Value::ArrayF64(values) => Ok(Value::ArrayF64(Rc::clone(values))),
            Value::ArrayBool(values) => Ok(Value::ArrayBool(Rc::clone(values))),
            Value::Bytes(values) => Ok(Value::Bytes(Rc::clone(values))),
            Value::Text(value) => Ok(Value::Text(Rc::clone(value))),
            Value::ViewI32(values, range) => Ok(Value::ViewI32(Rc::clone(values), *range)),
            Value::ViewU32(values, range) => Ok(Value::ViewU32(Rc::clone(values), *range)),
            _ => Err(tool("MIR_TYPE", "array reference on scalar place")),
        }
    }

    pub(super) fn record_ref(&self, name: &str) -> Result<Value, EvalError> {
        match self.slot(name)? {
            Value::VoxelGrid(grid) => Ok(Value::VoxelGrid(Rc::clone(grid))),
            Value::Record(values) => Ok(Value::Record(Rc::clone(values))),
            Value::Variant(values) => Ok(Value::Variant(Rc::clone(values))),
            Value::Bytes(values) => Ok(Value::Bytes(Rc::clone(values))),
            Value::Text(value) => Ok(Value::Text(Rc::clone(value))),
            _ => Err(tool("MIR_TYPE", "record reference on nonrecord place")),
        }
    }

    pub(super) fn cleanup(
        &mut self,
        names: &[String],
        trace: &mut Vec<CleanupEvent>,
        live: &mut usize,
    ) -> Result<(), EvalError> {
        for name in names {
            if let Some(slot) = self.slots.remove(name)
                && slot.owned
                && let Some(value) = slot.value
            {
                record_cleanup(trace, live, name, &value)?;
            }
        }
        Ok(())
    }
}

pub(super) fn record_cleanup(
    trace: &mut Vec<CleanupEvent>,
    live: &mut usize,
    name: &str,
    value: &Value,
) -> Result<(), EvalError> {
    if matches!(
        value,
        Value::ArrayI32(_)
            | Value::ArrayU32(_)
            | Value::ArrayF32(_)
            | Value::ArrayF64(_)
            | Value::ArrayBool(_)
            | Value::Bytes(_)
            | Value::Text(_)
            | Value::VoxelGrid(_)
            | Value::Record(_)
            | Value::Variant(_)
    ) {
        fn size(value: &Value) -> usize {
            match value {
                Value::ArrayI32(values) => values.borrow().len(),
                Value::ArrayU32(values) => values.borrow().len(),
                Value::ArrayF32(values) => values.borrow().len(),
                Value::ArrayF64(values) => values.borrow().len(),
                Value::ArrayBool(values) => values.borrow().len(),
                Value::Bytes(values) => values.len(),
                Value::Text(value) => value.len(),
                Value::VoxelGrid(grid) => grid.storage_elements(),
                Value::Record(values) => {
                    values.borrow().len() + values.borrow().iter().map(size).sum::<usize>()
                }
                Value::Variant(storage) => 2 + storage.borrow().1.as_ref().map(size).unwrap_or(0),
                _ => 0,
            }
        }
        let length = size(value);
        // Every cleaned owner was admitted exactly once; moves transfer, never re-admit.
        *live = live.saturating_sub(length);
        trace.push(CleanupEvent {
            name: name.to_owned(),
            length,
        });
    }
    Ok(())
}
