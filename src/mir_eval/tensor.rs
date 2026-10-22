//! Publish only a freshly checked completed tensor ticket, without changing
//! the source allocation/cleanup ledger or charging failed admission.
use super::*;
impl Interpreter<'_> {
    pub(super) fn apply_tensor_override(
        &mut self,
        function: &Function,
        head: &mir::BasicBlock,
        condition: ValueId,
        frame: &mut Frame,
        completion: &crate::tensor::host::Completion,
    ) -> Result<bool, EvalError> {
        let completion = match completion {
            crate::tensor::host::Completion::Contraction(c) => c,
            crate::tensor::host::Completion::Linear(c) => {
                return self.apply_f32_override(function, head, condition, frame, c);
            }
        };
        let (contract, dimensions, lengths, values, ticks) = completion.parts();
        let Some(Terminator::Branch { span, .. }) = &head.end else {
            return Ok(false);
        };
        if contract.function() != function.name || contract.loop_span() != *span
            || contract.function_contract() != crate::tensor::function_fingerprint(function)
            || contract.loop_fuel(dimensions) != Some(ticks)
            || contract.dimensions().iter().zip(dimensions).any(|(name,n)| !matches!(frame.slot(name),Ok(Value::U32(v)) if *v==n))
            || contract.inputs().iter().zip(lengths).any(|(name,n)| !matches!(frame.slot(name),Ok(Value::ArrayF32(v)) if v.try_borrow().is_ok_and(|v|v.len()==n))) {
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
        {
            return Ok(false);
        }
        let Value::ArrayF32(target) = frame.slot(contract.output())? else {
            return Ok(false);
        };
        for name in contract.inputs() {
            if matches!(frame.slot(name),Ok(Value::ArrayF32(input)) if Rc::ptr_eq(target,input)) {
                return Ok(false);
            }
        }
        let Some(output_len) = u64::from(dimensions[0]).checked_mul(u64::from(dimensions[2]))
        else {
            return Ok(false);
        };
        if output_len != values.len() as u64
            || ticks > self.fuel_limit.saturating_sub(self.steps_used)
        {
            return Ok(false);
        }
        let Ok(mut target) = target.try_borrow_mut() else {
            return Ok(false);
        };
        if target.len() != values.len() {
            return Ok(false);
        }
        target.copy_from_slice(values);
        self.steps_used += ticks;
        self.used_overrides += 1;
        Ok(true)
    }
}
