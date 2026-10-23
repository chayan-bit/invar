//! Atomic publication of a source-bound, completed rank-1 f32 ticket.
use super::*;
use crate::f32_source::Completion;

impl Interpreter<'_> {
    pub(super) fn apply_f32_override(
        &mut self,
        function: &Function,
        head: &mir::BasicBlock,
        condition: ValueId,
        frame: &mut Frame,
        completion: &Completion,
    ) -> Result<bool, EvalError> {
        let r = &completion.region;
        let Some(Terminator::Branch { span, .. }) = &head.end else {
            return Ok(false);
        };
        if r.function != function.name
            || r.span != *span
            || r.fingerprint != crate::tensor::function_fingerprint(function)
            || r.ticks_per_iteration.checked_mul(completion.length as u64) != Some(completion.ticks)
            || completion.ticks > self.fuel_limit.saturating_sub(self.steps_used)
            || completion.input_lengths.iter().any(
                |(name, n)| !matches!(frame.slot(name), Ok(v) if array_length(v).ok() == Some(*n)),
            )
        {
            return Ok(false);
        }
        if live_trips(head, condition, frame).is_none_or(|n| n as usize != completion.length) {
            return Ok(false);
        }
        let plan = r
            .bind(
                completion.length as u32,
                invar_device::Capabilities::cuda_12_9_sm89(),
            )
            .map_err(|_| tool("MIR_INVALID", "f32 completion contract changed"))?;
        let program = invar_device::f32_ops::program(&plan)
            .map_err(|_| tool("MIR_INVALID", "f32 completion program changed"))?;
        if program.output_len() != completion.words.len() {
            return Ok(false);
        }
        let final_acc = if r.accumulator.is_some() {
            completion.words.last().copied().or(match program.kind {
                invar_device::f32_ops::Kind::Scan { initial } => Some(initial),
                _ => None,
            })
        } else {
            None
        };
        if let Some(acc) = &r.accumulator {
            let expected = match program.kind {
                invar_device::f32_ops::Kind::Scan { initial }
                | invar_device::f32_ops::Kind::Reduce { initial } => initial,
                _ => return Ok(false),
            };
            if !matches!(frame.slot(acc), Ok(Value::F32(v)) if v.to_bits() == expected) {
                return Ok(false);
            }
        }
        if r.accumulator.as_deref() != Some(r.output.as_str()) {
            match frame.slot(&r.output)? {
                Value::ArrayF32(a) => {
                    for (name, _) in &r.inputs {
                        if matches!(frame.slot(name), Ok(Value::ArrayF32(b)) if Rc::ptr_eq(a,b)) {
                            return Ok(false);
                        }
                    }
                    let Ok(mut a) = a.try_borrow_mut() else {
                        return Ok(false);
                    };
                    if a.len() != completion.words.len() {
                        return Ok(false);
                    }
                    for (a, w) in a.iter_mut().zip(&completion.words) {
                        *a = crate::numeric_contract::F32Bits::from_bits(*w);
                    }
                }
                Value::ArrayI32(a) => {
                    for (name, _) in &r.inputs {
                        if matches!(frame.slot(name), Ok(Value::ArrayI32(b)) if Rc::ptr_eq(a,b)) {
                            return Ok(false);
                        }
                    }
                    let Ok(mut a) = a.try_borrow_mut() else {
                        return Ok(false);
                    };
                    if a.len() != completion.words.len() {
                        return Ok(false);
                    }
                    for (a, w) in a.iter_mut().zip(&completion.words) {
                        *a = *w as i32;
                    }
                }
                Value::ArrayU32(a) => {
                    for (name, _) in &r.inputs {
                        if matches!(frame.slot(name), Ok(Value::ArrayU32(b)) if Rc::ptr_eq(a,b)) {
                            return Ok(false);
                        }
                    }
                    let Ok(mut a) = a.try_borrow_mut() else {
                        return Ok(false);
                    };
                    if a.len() != completion.words.len() {
                        return Ok(false);
                    }
                    a.copy_from_slice(&completion.words);
                }
                _ => return Ok(false),
            }
        }
        if let Some(acc) = &r.accumulator {
            let Some(word) = final_acc else {
                return Ok(false);
            };
            let slot = frame
                .slots
                .get_mut(acc)
                .ok_or_else(|| tool("MIR_INVALID", "f32 accumulator missing"))?;
            slot.value = Some(Value::F32(crate::numeric_contract::F32Bits::from_bits(
                word,
            )));
        }
        self.steps_used += completion.ticks;
        self.used_overrides += 1;
        Ok(true)
    }
}
