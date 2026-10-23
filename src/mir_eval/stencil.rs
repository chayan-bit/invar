//! A source-bound completed stencil ticket can replace only its reached loop.
//! Every live check runs before the single write; fuel is debited once, only on
//! publication. Any mismatch leaves ordinary source to run the loop.
use super::*;
use crate::numeric_contract::F32Bits;

/// Copy `values` into the fresh `target` only when it is a distinct, unborrowed
/// owner of exactly the published length. No write happens on refusal.
pub(super) fn publish_words(
    target: &Rc<RefCell<Vec<F32Bits>>>,
    input: &Rc<RefCell<Vec<F32Bits>>>,
    values: &[F32Bits],
) -> bool {
    if Rc::ptr_eq(target, input) {
        return false;
    }
    let Ok(mut target) = target.try_borrow_mut() else {
        return false;
    };
    if target.len() != values.len() {
        return false;
    }
    target.copy_from_slice(values);
    true
}

impl Interpreter<'_> {
    pub(super) fn apply_stencil_override(
        &mut self,
        function: &Function,
        head: &mir::BasicBlock,
        condition: ValueId,
        frame: &mut Frame,
        completion: &crate::stencil_source::host::Completion,
    ) -> Result<bool, EvalError> {
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
                .any(|(name, n)| !matches!(frame.slot(name), Ok(Value::U32(v)) if *v == n))
            || !matches!(frame.slot(contract.input()).and_then(array_length), Ok(n) if n == input_length)
        {
            return Ok(false);
        }
        if live_trips(head, condition, frame) != Some(dimensions[0])
            || ticks > self.fuel_limit.saturating_sub(self.steps_used)
        {
            return Ok(false);
        }
        let count = (dimensions[0] as usize).checked_mul(dimensions[1] as usize);
        if count != Some(values.len()) {
            return Ok(false);
        }
        let (Value::ArrayF32(target), Value::ArrayF32(input)) = (
            frame.slot(contract.output())?,
            frame.slot(contract.input())?,
        ) else {
            return Ok(false);
        };
        if !publish_words(target, input, values) {
            return Ok(false);
        }
        self.steps_used += ticks;
        self.used_overrides += 1;
        Ok(true)
    }
}

#[cfg(test)]
mod tests {
    use super::publish_words;
    use crate::numeric_contract::F32Bits;
    use std::{cell::RefCell, rc::Rc};

    fn words(n: &[u32]) -> Vec<F32Bits> {
        n.iter().map(|b| F32Bits::from_bits(*b)).collect()
    }

    #[test]
    fn stencil_publication_refuses_alias_borrow_and_length_without_writing() {
        let input = Rc::new(RefCell::new(words(&[1, 2])));
        let output = Rc::new(RefCell::new(words(&[0, 0])));
        let values = words(&[7, 8]);
        // Aliased output: refused, input untouched.
        assert!(!publish_words(&input, &input, &values));
        assert_eq!(*input.borrow(), words(&[1, 2]));
        // Output borrowed elsewhere: refused, untouched.
        {
            let _held = output.borrow();
            assert!(!publish_words(&output, &input, &values));
        }
        assert_eq!(*output.borrow(), words(&[0, 0]));
        // Length mismatch: refused, untouched.
        assert!(!publish_words(&output, &input, &words(&[7])));
        assert_eq!(*output.borrow(), words(&[0, 0]));
        // Distinct, unborrowed, exact length: published.
        assert!(publish_words(&output, &input, &values));
        assert_eq!(*output.borrow(), values);
    }
}
