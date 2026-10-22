//! A source-bound completed transpose ticket can replace only its reached loop.
use super::*;
impl Interpreter<'_> {
    pub(super) fn apply_transpose_override(
        &mut self,
        function: &Function,
        head: &mir::BasicBlock,
        condition: ValueId,
        frame: &mut Frame,
        completion: &crate::transpose_source::host::Completion,
    ) -> Result<bool, EvalError> {
        use crate::transpose_source::host::Words;
        let (contract, dimensions, input_length, values, ticks) = completion.parts();
        let Some(Terminator::Branch { span, .. }) = &head.end else {
            return Ok(false);
        };
        if contract.function() != function.name
            || contract.loop_span() != *span
            || contract.function_contract() != crate::tensor::function_fingerprint(function)
            || contract.loop_fuel(dimensions) != Some(ticks)
            || contract
                .dimensions()
                .iter()
                .zip(dimensions)
                .any(|(name, n)| !matches!(frame.slot(name),Ok(Value::U32(v)) if *v==n))
            || !matches!(frame.slot(contract.input()).and_then(array_length),Ok(n) if n==input_length)
        {
            return Ok(false);
        }
        let Some(Operation::Binary {
            op: BinaryOp::Lt,
            left,
            right,
            ..
        }) = head
            .instructions
            .iter()
            .find(|i| i.output.as_ref().is_some_and(|(id, _)| *id == condition))
            .map(|i| &i.kind)
        else {
            return Ok(false);
        };
        if !matches!(frame.value(*left), Ok(Value::U32(0)))
            || !matches!(frame.value(*right),Ok(Value::U32(n)) if n==dimensions[0])
            || ticks > self.fuel_limit.saturating_sub(self.steps_used)
        {
            return Ok(false);
        }
        let count = (dimensions[0] as usize).checked_mul(dimensions[1] as usize);
        match (
            contract.element(),
            values,
            frame.slot(contract.output())?,
            frame.slot(contract.input())?,
        ) {
            (Type::U32, Words::U32(values), Value::ArrayU32(target), Value::ArrayU32(input)) => {
                if Rc::ptr_eq(target, input) || count != Some(values.len()) {
                    return Ok(false);
                }
                let Ok(mut target) = target.try_borrow_mut() else {
                    return Ok(false);
                };
                if target.len() != values.len() {
                    return Ok(false);
                }
                target.copy_from_slice(values);
            }
            (Type::F32, Words::F32(values), Value::ArrayF32(target), Value::ArrayF32(input)) => {
                if Rc::ptr_eq(target, input) || count != Some(values.len()) {
                    return Ok(false);
                }
                let Ok(mut target) = target.try_borrow_mut() else {
                    return Ok(false);
                };
                if target.len() != values.len() {
                    return Ok(false);
                }
                target.copy_from_slice(values);
            }
            _ => return Ok(false),
        }
        self.steps_used += ticks;
        self.used_overrides += 1;
        Ok(true)
    }
}
