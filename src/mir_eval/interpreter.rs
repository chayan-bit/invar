//! Interpreter control flow: fuel, calls and block execution.

use super::*;

impl Interpreter<'_> {
    pub(super) fn tick(&mut self) -> Result<(), EvalError> {
        if self.steps_used == self.fuel_limit {
            return Err(EvalError::EvaluationLimit {
                fuel_limit: self.fuel_limit,
                steps_used: self.steps_used,
            });
        }
        self.steps_used += 1;
        Ok(())
    }

    pub(super) fn call(&mut self, name: &str, args: Vec<Value>) -> Result<Value, EvalError> {
        if self.call_depth == MAX_CALL_DEPTH {
            return Err(tool("CALL_DEPTH_LIMIT", "MIR call depth limit exceeded"));
        }
        let function = self
            .program
            .function(name)
            .ok_or_else(|| tool("MIR_CALL", "missing verified function"))?;
        if args.len() != function.params.len() {
            return Err(tool("MIR_CALL", "invalid call arity"));
        }
        if let Some(oracle) = &function.foreign_oracle {
            let oracle = oracle.clone();
            return self.call(&oracle, args);
        }
        let tc = crate::contraction_source::mir_is_tc(function)
            .then(|| crate::eval::floating::TcInputs::capture(&args))
            .transpose()?;
        let publications = self.used_overrides;
        self.call_depth += 1;
        let mut frame = Frame::new(function, args);
        let result = self.blocks(function, &mut frame);
        self.call_depth -= 1;
        match (tc, result) {
            (Some(tc), Ok(value)) if self.used_overrides == publications => tc.finish(value),
            (_, result) => result,
        }
    }

    fn blocks(&mut self, function: &Function, frame: &mut Frame) -> Result<Value, EvalError> {
        let mut block_id = function.entry;
        let mut definitions = None;
        loop {
            let block = function
                .blocks
                .get(block_id)
                .ok_or_else(|| tool("MIR_BLOCK", "missing verified block"))?;
            for instruction in &block.instructions {
                let value = match &instruction.kind {
                    Operation::Call { name, args }
                        if crate::fsum::Kind::from_name(name).is_some() =>
                    {
                        Some(self.reduce(function, name, args, instruction.span, frame)?)
                    }
                    kind => self.operation(
                        &function.name,
                        kind,
                        instruction.output.as_ref().map(|(_, ty)| ty),
                        instruction.span,
                        frame,
                    )?,
                };
                match (&instruction.output, value) {
                    (Some((id, _)), Some(value)) => {
                        frame.values.insert(*id, value);
                    }
                    (None, None) => {}
                    _ => return Err(tool("MIR_VALUE", "operation output mismatch")),
                }
            }
            match block
                .end
                .as_ref()
                .ok_or_else(|| tool("MIR_BLOCK", "missing verified terminator"))?
            {
                Terminator::Goto { target, args, .. } => {
                    let values = args
                        .iter()
                        .map(|id| frame.value(*id))
                        .collect::<Result<Vec<_>, _>>()?;
                    let next = function
                        .blocks
                        .get(*target)
                        .ok_or_else(|| tool("MIR_BLOCK", "missing verified successor"))?;
                    for ((id, _), value) in next.params.iter().zip(values) {
                        frame.values.insert(*id, value);
                    }
                    block_id = *target;
                }
                Terminator::Branch {
                    condition,
                    yes,
                    no,
                    span,
                } => {
                    let truth = matches!(frame.value(*condition)?, Value::Bool(true));
                    // An identity pair fold is a reached region even when its
                    // zero trip count skips the body. Preserve the existing
                    // provider boundary for every other empty source form.
                    let empty_pair = !truth
                        && self.provider.is_some()
                        && crate::pair_reduction::reduction_in(
                            function,
                            definitions
                                .get_or_insert_with(|| crate::counted::definitions(function)),
                            block_id,
                        )
                        .is_ok();
                    let bulk = if truth || empty_pair {
                        let live = live_loop(
                            function,
                            block,
                            *yes,
                            *condition,
                            *span,
                            frame,
                            self.fuel_limit.saturating_sub(self.steps_used),
                        );
                        if let (Some(provider), Some(live)) = (self.provider.as_mut(), live) {
                            provider(&live)?
                        } else {
                            None
                        }
                    } else {
                        None
                    };
                    let applied = match bulk.as_ref() {
                        Some(plan) => {
                            let before = self.used_overrides;
                            let applied =
                                self.apply_override(function, block, *condition, *yes, frame, plan);
                            // Every publication counts itself, including one that ends in a trap.
                            self.published.push(self.used_overrides != before);
                            applied?
                        }
                        None => false,
                    };
                    block_id = if applied || !truth { *no } else { *yes };
                }
                Terminator::Return { value, cleanup, .. } => {
                    let result = value
                        .map(|id| frame.value(id))
                        .transpose()?
                        .unwrap_or(Value::Unit);
                    frame.cleanup(cleanup, &mut self.trace, &mut self.live_elements)?;
                    return Ok(result);
                }
            }
        }
    }
}
