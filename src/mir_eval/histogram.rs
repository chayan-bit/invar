//! Private atomic publication for a pinned checked-source histogram attempt.
use super::*;
use crate::{
    general_net::compiled::FenceObservation,
    histogram_source::host::{Completion, Payload, Pending, failure},
};
pub(crate) struct Quarantined {
    frame: Frame,
    _payload: Rc<Payload>,
}
impl Quarantined {
    pub(crate) fn retained_arrays(&self) -> usize {
        self.frame
            .slots
            .values()
            .filter(|s| matches!(s.value, Some(Value::ArrayU32(_))))
            .count()
    }
}
impl Interpreter<'_> {
    pub(super) fn apply_histogram_override(
        &mut self,
        function: &Function,
        head: &mir::BasicBlock,
        condition: ValueId,
        frame: &mut Frame,
        completion: &Completion,
    ) -> Result<bool, EvalError> {
        if matches!(
            completion.status,
            FenceObservation::Unknown | FenceObservation::Pending
        ) {
            // Unknown is handled first: no owner borrow, proof check, work
            // accounting or fallible allocation can release reached context.
            let pending = Pending {
                resources: Some(Quarantined {
                    frame: std::mem::take(frame),
                    _payload: completion.payload.clone(),
                }),
            };
            park(&completion.quarantine, pending);
            return Err(failure("HISTOGRAM_UNKNOWN_COMPLETION"));
        }
        if completion.status != FenceObservation::Complete {
            return Err(failure("HISTOGRAM_FAILED_COMPLETION"));
        }
        let p = &completion.payload;
        let plan = &p.plan;
        let mut meter = p
            .meter
            .try_borrow_mut()
            .map_err(|_| failure("HISTOGRAM_METER"))?;
        // Complete publication work was prepaid in the persistent run ledger.
        meter
            .visit(
                plan.contract
                    .runtime_publish_header()
                    .map_err(|_| failure("HISTOGRAM_BUDGET"))?,
            )
            .map_err(|_| failure("HISTOGRAM_BUDGET"))?;
        if !p
            .ledger
            .try_borrow()
            .is_ok_and(|l| l.permits_attempt(plan, &p.meter, p.cost))
            || !std::ptr::eq(self.program, plan.program.as_ref())
            || !std::ptr::eq(function, plan.function())
            || !std::ptr::eq(head, &function.blocks[plan.head_index])
            || condition != plan.condition
        {
            return Err(failure("HISTOGRAM_SOURCE_BINDING"));
        }
        let i = p.data.candidate.source_accesses().len();
        if !matches!(frame.value(plan.left), Ok(Value::U32(0)))
            || !matches!(frame.value(plan.right),Ok(Value::U32(v)) if v as usize==i)
            || plan.loop_ticks(i) != Some(p.ticks)
            || p.ticks > self.fuel_limit.saturating_sub(self.steps_used)
        {
            return Err(failure("HISTOGRAM_LOOP_BINDING"));
        }
        let next_steps = self
            .steps_used
            .checked_add(p.ticks)
            .ok_or_else(|| failure("HISTOGRAM_FUEL"))?;
        let next_completed = self
            .used_overrides
            .checked_add(1)
            .ok_or_else(|| failure("HISTOGRAM_COUNT"))?;
        let Value::U32(n) = frame.slot(&plan.bins)? else {
            return Err(failure("HISTOGRAM_EXTENT"));
        };
        let Value::ArrayU32(values) = frame.slot(&plan.values)? else {
            return Err(failure("HISTOGRAM_OWNER"));
        };
        let Value::ArrayU32(indices) = frame.slot(&plan.indices)? else {
            return Err(failure("HISTOGRAM_OWNER"));
        };
        let Value::ArrayU32(output) = frame.slot(&plan.output)? else {
            return Err(failure("HISTOGRAM_OWNER"));
        };
        if !frame.slots.get(&plan.output).is_some_and(|s| s.owned)
            || frame.slots.get(&plan.values).is_none_or(|s| s.owned)
            || frame.slots.get(&plan.indices).is_none_or(|s| s.owned)
            || !Rc::ptr_eq(values, &p.owners[0])
            || !Rc::ptr_eq(indices, &p.owners[1])
            || !Rc::ptr_eq(output, &p.owners[2])
            || Rc::ptr_eq(values, output)
            || Rc::ptr_eq(indices, output)
        {
            return Err(failure("HISTOGRAM_OWNER"));
        }
        let values = values.try_borrow().map_err(|_| failure("HISTOGRAM_LOAN"))?;
        let indices = indices
            .try_borrow()
            .map_err(|_| failure("HISTOGRAM_LOAN"))?;
        let mut output = output
            .try_borrow_mut()
            .map_err(|_| failure("HISTOGRAM_LOAN"))?;
        let cost = plan
            .contract
            .runtime_cost(
                values.len(),
                indices.len(),
                *n as usize,
                crate::histogram_source::Budget::default(),
            )
            .map_err(|_| failure("HISTOGRAM_BUDGET"))?;
        if cost != p.cost
            || cost.visits != meter.limit
            || output.len() != *n as usize
            || output.len() != p.data.candidate.output().len()
        {
            return Err(failure("HISTOGRAM_EXTENT"));
        }
        // Validate independent source-order u64 sums and an exact permutation of
        // every duplicate occurrence. The immutable source proof was streamed
        // once before entry and remains pinned by the original run ledger.
        plan.contract
            .runtime_validate(&p.data, &values, &indices, *n, &mut meter)
            .map_err(|_| failure("HISTOGRAM_BAD_COMPLETION"))?;
        for &zero in output.iter() {
            meter.visit(1).map_err(|_| failure("HISTOGRAM_BUDGET"))?;
            if zero != 0 {
                return Err(failure("HISTOGRAM_FRESH_OUTPUT"));
            }
        }
        // Charge full copy BEFORE the first write. All remaining actions are
        // infallible: exact-length copy, semantic debit and publication count.
        meter
            .visit(output.len())
            .map_err(|_| failure("HISTOGRAM_BUDGET"))?;
        output.copy_from_slice(p.data.candidate.output());
        self.steps_used = next_steps;
        self.used_overrides = next_completed;
        Ok(true)
    }
}

#[cfg(test)]
mod atomic_tests;
#[cfg(test)]
mod tests;
