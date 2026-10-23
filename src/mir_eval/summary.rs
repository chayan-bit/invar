//! Atomic publication of a private checked-source CPU bounded scan completion.
use super::*;
impl Interpreter<'_> {
    pub(super) fn apply_summary_override(
        &mut self,
        function: &Function,
        head: &mir::BasicBlock,
        condition: ValueId,
        frame: &mut Frame,
        completion: &crate::summary_source::host::Completion,
    ) -> Result<bool, EvalError> {
        use crate::summary_source::Family;
        let (contract, prepared, plan, candidate, budget, steps) = completion.parts();
        let Some(Terminator::Branch { span, .. }) = &head.end else {
            return Ok(false);
        };
        let Some(ticks) = contract.loop_fuel(steps) else {
            return Ok(false);
        };
        if contract.function() != function.name
            || contract.loop_span() != *span
            || contract.function_contract() != crate::tensor::function_fingerprint(function)
            || ticks > self.fuel_limit.saturating_sub(self.steps_used)
            || !matches!(frame.slot(contract.count()), Ok(Value::U32(n)) if *n == steps)
        {
            return Ok(false);
        }
        if live_trips(head, condition, frame) != Some(steps) {
            return Ok(false);
        }
        let Value::ArrayU32(coefficients) = frame.slot(contract.coefficients())? else {
            return Ok(false);
        };
        let coefficients = Rc::clone(coefficients);
        let Ok(coefficients) = coefficients.try_borrow() else {
            return Ok(false);
        };
        let mut seed = [0; 4];
        let width = contract.family().state_width();
        match contract.family() {
            Family::Table(_) => {
                let Value::U32(n) = frame.slot(contract.seed())? else {
                    return Ok(false);
                };
                seed[0] = *n;
            }
            Family::Matrix(_) => {
                let Value::ArrayU32(values) = frame.slot(contract.seed())? else {
                    return Ok(false);
                };
                let Ok(values) = values.try_borrow() else {
                    return Ok(false);
                };
                if values.len() != width {
                    return Ok(false);
                }
                seed[..width].copy_from_slice(&values);
            }
        }
        if contract
            .verify_plan(prepared, plan, &coefficients, &seed[..width], steps, budget)
            .is_err()
            || candidate.coefficients() != coefficients.as_slice()
            || candidate.seed() != &seed[..width]
            || candidate.final_state().len() != width
            || candidate.prefixes().len() != steps as usize
            || candidate.prefixes().iter().any(|v| v.len() != width)
            || contract
                .state_names()
                .iter()
                .zip(&seed[..width])
                .any(|(name, n)| !matches!(frame.slot(name),Ok(Value::U32(v)) if v == n))
        {
            return Ok(false);
        }
        if let Some(name) = contract.output() {
            let Value::ArrayU32(output) = frame.slot(name)? else {
                return Ok(false);
            };
            if matches!(frame.slot(contract.coefficients()),Ok(Value::ArrayU32(input)) if Rc::ptr_eq(input,output))
                || matches!(frame.slot(contract.seed()),Ok(Value::ArrayU32(input)) if Rc::ptr_eq(input,output))
            {
                return Ok(false);
            }
            let Ok(mut output) = output.try_borrow_mut() else {
                return Ok(false);
            };
            if (steps as usize).checked_mul(width) != Some(output.len()) {
                return Ok(false);
            }
            for (to, from) in output.chunks_exact_mut(width).zip(candidate.prefixes()) {
                to.copy_from_slice(from);
            }
        }
        // Every scalar slot is validated above. No fallible operation follows
        // the first output write; source owners and allocation accounting stay live.
        for (name, final_value) in contract.state_names().iter().zip(candidate.final_state()) {
            if let Some(slot) = frame.slots.get_mut(name) {
                slot.value = Some(Value::U32(*final_value));
            }
        }
        self.steps_used += ticks;
        self.used_overrides += 1;
        Ok(true)
    }
}
