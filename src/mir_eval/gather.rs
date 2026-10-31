//! Checked-source gather publication. Only complete private results can write.
use super::*;
use crate::general_net::compiled::FenceObservation;
use crate::indexed_source::host::{Completion, Payload, Pending, failure};

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
    #[cfg(test)]
    pub(crate) fn output_weak(&self) -> std::rc::Weak<RefCell<Vec<u32>>> {
        let Value::ArrayU32(out) = self.frame.slot(&self._payload.plan.output).unwrap() else {
            panic!()
        };
        Rc::downgrade(out)
    }
}

impl Interpreter<'_> {
    pub(super) fn apply_gather_override(
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
            // Move the entire reached frame, without allocating or borrowing its
            // owners. Unknown work never causes release, publication or replay.
            let pending = Pending {
                resources: Some(Quarantined {
                    frame: std::mem::take(frame),
                    _payload: completion.payload.clone(),
                }),
            };
            park(&completion.quarantine, pending);
            return Err(failure("GATHER_UNKNOWN_COMPLETION"));
        }
        if completion.status != FenceObservation::Complete {
            return Err(failure("GATHER_FAILED_COMPLETION"));
        }
        let p = &completion.payload;
        p.plan.verify()?;
        p.plan.verify_function(function)?;
        if !p.plan.matches_source(p.source_hash)
            || self.program.contract_digest() != p.plan.program_digest
            || function.name != p.plan.function
            || !matches!(head.end,Some(Terminator::Branch{span,..}) if span==p.plan.contract.loop_span())
        {
            return Err(failure("GATHER_SOURCE_BINDING"));
        }
        let n = p.indices.len();
        if live_trips(head, condition, frame).is_none_or(|v| v as usize != n)
            || p.plan.loop_ticks(n) != Some(p.ticks)
            || p.ticks > self.fuel_limit.saturating_sub(self.steps_used)
        {
            return Err(failure("GATHER_LOOP_BINDING"));
        }
        let next_steps = self
            .steps_used
            .checked_add(p.ticks)
            .ok_or_else(|| failure("GATHER_FUEL"))?;
        let next_completed = self
            .used_overrides
            .checked_add(1)
            .ok_or_else(|| failure("GATHER_COUNT"))?;
        let Value::ArrayU32(values) = frame.slot(&p.plan.values)? else {
            return Err(failure("GATHER_OWNER"));
        };
        let Value::ArrayU32(indices) = frame.slot(&p.plan.indices)? else {
            return Err(failure("GATHER_OWNER"));
        };
        let Value::ArrayU32(output) = frame.slot(&p.plan.output)? else {
            return Err(failure("GATHER_OWNER"));
        };
        if !frame.slots.get(&p.plan.output).is_some_and(|s| s.owned)
            || Rc::ptr_eq(values, output)
            || Rc::ptr_eq(indices, output)
        {
            return Err(failure("GATHER_OWNER"));
        }
        let values = values.try_borrow().map_err(|_| failure("GATHER_LOAN"))?;
        let indices = indices.try_borrow().map_err(|_| failure("GATHER_LOAN"))?;
        let mut output = output
            .try_borrow_mut()
            .map_err(|_| failure("GATHER_LOAN"))?;
        if [
            std::ptr::from_ref(&*values).addr(),
            std::ptr::from_ref(&*indices).addr(),
            std::ptr::from_ref(&*output).addr(),
        ] != p.identities
            || *values != p.values
            || *indices != p.indices
            || output.len() != n
            || super::super::indexed_source::host::cost(
                values.len(),
                indices.len(),
                output.len(),
                p.budget,
            ) != Some(p.cost)
            || p.candidate.output().len() != n
            || p.candidate.accesses().len() != n
        {
            return Err(failure("GATHER_STALE_COMPLETION"));
        }
        // This is one indexed verification pass, also checking untouched output.
        for (i, ((&raw, &zero), access)) in indices
            .iter()
            .zip(output.iter())
            .zip(p.candidate.accesses())
            .enumerate()
        {
            let read = usize::try_from(raw).map_err(|_| failure("GATHER_BAD_COMPLETION"))?;
            if zero != 0
                || access.iteration != i
                || access.read_index != read
                || access.write_index != i
                || values.get(read) != p.candidate.output().get(i)
            {
                return Err(failure("GATHER_BAD_COMPLETION"));
            }
        }
        // Extents, loans, identities, source binding and all arithmetic are
        // validated. Nothing fallible follows the first write or source debit.
        output.copy_from_slice(p.candidate.output());
        self.steps_used = next_steps;
        self.used_overrides = next_completed;
        Ok(true)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::indexed_source::host::{self, Budget, Plan};
    const SOURCE:&[u8]=b"fn gather(a: read Array<u32>, ix: read Array<u32>) -> Array<u32> { var out = filled_u32(len(ix),0u32); for i in 0u32..len(ix) { out[i] = a[ix[i]]; } return out; }";

    #[test]
    fn issue_m4_05_gather_scatter_frontier_runtime_publication_no_writes_or_debit() {
        for fault in 0..7 {
            let checked = crate::check::check(crate::front::parse(SOURCE).unwrap(), false).unwrap();
            let program = mir::lower(&checked).unwrap();
            let plan = Rc::new(Plan::bind(SOURCE.into(), &checked, "gather", &program).unwrap());
            let function = &program.functions[0];
            let (head, condition, body) = function
                .blocks
                .iter()
                .find_map(|b| match &b.end {
                    Some(Terminator::Branch {
                        condition,
                        yes,
                        span,
                        ..
                    }) if *span == plan.contract.loop_span() => Some((b, *condition, *yes)),
                    _ => None,
                })
                .unwrap();
            let (left, right) = head
                .instructions
                .iter()
                .find_map(|i| match &i.kind {
                    Operation::Binary {
                        op: BinaryOp::Lt,
                        left,
                        right,
                        ..
                    } => Some((*left, *right)),
                    _ => None,
                })
                .unwrap();
            let values = Rc::new(RefCell::new(vec![7, 11, 19]));
            let indices = Rc::new(RefCell::new(vec![1, 0, 1]));
            let output = Rc::new(RefCell::new(vec![0; 3]));
            let mut frame = Frame::new(
                function,
                vec![
                    Value::ArrayU32(values.clone()),
                    Value::ArrayU32(indices.clone()),
                ],
            );
            frame.slots.insert(
                "out".into(),
                Slot {
                    value: Some(Value::ArrayU32(output.clone())),
                    owned: true,
                },
            );
            frame.values.insert(left, Value::U32(0));
            frame.values.insert(right, Value::U32(3));
            let live = super::super::bulk::live_loop(
                function,
                head,
                body,
                condition,
                plan.contract.loop_span(),
                &frame,
                1000,
            )
            .unwrap();
            let mut wrong_output = Plan::bind(SOURCE.into(), &checked, "gather", &program).unwrap();
            wrong_output.output = wrong_output.values.clone();
            assert!(!host::distinct(&live, &wrong_output)); // physical alias refusal before candidate entry
            let mut begun = 0;
            let mut refusals = Vec::new();
            let mut completion = host::prepare(
                plan,
                &live,
                Budget::default(),
                Rc::new(RefCell::new(None)),
                &mut begun,
                &mut refusals,
                host::produce_sync,
            )
            .unwrap()
            .unwrap();
            assert_eq!(begun, 1);
            match fault {
                0 => {
                    frame.slots.get_mut("out").unwrap().value =
                        Some(Value::ArrayU32(values.clone()))
                }
                1 => frame.slots.get_mut("out").unwrap().owned = false,
                2 => {
                    output.borrow_mut().pop();
                }
                3 => Rc::get_mut(&mut completion.payload).unwrap().source_hash[0] ^= 1,
                4 => Rc::get_mut(&mut completion.payload).unwrap().values[0] ^= 1,
                5 => Rc::get_mut(&mut completion.payload).unwrap().ticks += 1,
                _ => {}
            }
            let held = if fault == 6 {
                Some(output.borrow_mut())
            } else {
                None
            };
            let mut interpreter = Interpreter {
                program: &program,
                provider: None,
                sorts: None,
                reducer: None,
                fuel_limit: 1000,
                steps_used: 17,
                call_depth: 0,
                trace: Vec::new(),
                live_elements: 9,
                used_overrides: 0,
                published: Vec::new(),
            };
            let error = interpreter
                .apply_gather_override(function, head, condition, &mut frame, &completion)
                .unwrap_err();
            if fault == 0 || fault == 1 {
                assert!(matches!(
                    error,
                    EvalError::Tool {
                        code: "GATHER_OWNER",
                        ..
                    }
                ));
            }
            drop(held);
            assert!(output.borrow().iter().all(|&v| v == 0));
            assert_eq!(*values.borrow(), [7, 11, 19]);
            assert_eq!(interpreter.steps_used, 17);
            assert_eq!(interpreter.used_overrides, 0);
            assert!(interpreter.trace.is_empty());
        }
    }

    /// Real injections through the direct publication harness (review3 R2):
    /// owner replacement, held loans and both counters reach their typed
    /// codes with no write, debit or trace.
    #[test]
    fn machine_diagnostics_gather_publication_harness_faults() {
        let expected = [
            "GATHER_OWNER", // output slot aliases the values owner
            "GATHER_OWNER", // output slot is not owned
            "GATHER_LOAN",  // output held by a live borrow
            "GATHER_LOAN",  // values held mutably
            "GATHER_COUNT", // publication counter at its maximum
            "GATHER_COUNT", // host entry counter at its maximum
        ];
        for (fault, code) in expected.into_iter().enumerate() {
            let checked = crate::check::check(crate::front::parse(SOURCE).unwrap(), false).unwrap();
            let program = mir::lower(&checked).unwrap();
            let plan = Rc::new(Plan::bind(SOURCE.into(), &checked, "gather", &program).unwrap());
            let function = &program.functions[0];
            let (head, condition, body) = function
                .blocks
                .iter()
                .find_map(|b| match &b.end {
                    Some(Terminator::Branch {
                        condition,
                        yes,
                        span,
                        ..
                    }) if *span == plan.contract.loop_span() => Some((b, *condition, *yes)),
                    _ => None,
                })
                .unwrap();
            let (left, right) = head
                .instructions
                .iter()
                .find_map(|i| match &i.kind {
                    Operation::Binary {
                        op: BinaryOp::Lt,
                        left,
                        right,
                        ..
                    } => Some((*left, *right)),
                    _ => None,
                })
                .unwrap();
            let values = Rc::new(RefCell::new(vec![7, 11, 19]));
            let indices = Rc::new(RefCell::new(vec![1, 0, 1]));
            let output = Rc::new(RefCell::new(vec![0; 3]));
            let mut frame = Frame::new(
                function,
                vec![
                    Value::ArrayU32(values.clone()),
                    Value::ArrayU32(indices.clone()),
                ],
            );
            frame.slots.insert(
                "out".into(),
                Slot {
                    value: Some(Value::ArrayU32(output.clone())),
                    owned: true,
                },
            );
            frame.values.insert(left, Value::U32(0));
            frame.values.insert(right, Value::U32(3));
            let live = super::super::bulk::live_loop(
                function,
                head,
                body,
                condition,
                plan.contract.loop_span(),
                &frame,
                1000,
            )
            .unwrap();
            let mut begun = if fault == 5 { usize::MAX } else { 0 };
            let prepared = host::prepare(
                plan,
                &live,
                Budget::default(),
                Rc::new(RefCell::new(None)),
                &mut begun,
                &mut Vec::new(),
                host::produce_sync,
            );
            let error = if fault == 5 {
                assert_eq!(begun, usize::MAX); // no entry, no producer call
                let Err(error) = prepared else {
                    panic!("entry counter overflow was admitted");
                };
                error
            } else {
                let completion = prepared.unwrap().unwrap();
                match fault {
                    0 => {
                        frame.slots.get_mut("out").unwrap().value =
                            Some(Value::ArrayU32(values.clone()))
                    }
                    1 => frame.slots.get_mut("out").unwrap().owned = false,
                    _ => {}
                }
                let held_output = (fault == 2).then(|| output.borrow_mut());
                let held_values = (fault == 3).then(|| values.borrow_mut());
                let overrides = if fault == 4 { usize::MAX } else { 0 };
                let mut interpreter = Interpreter {
                    program: &program,
                    provider: None,
                    sorts: None,
                    reducer: None,
                    fuel_limit: 1000,
                    steps_used: 17,
                    call_depth: 0,
                    trace: Vec::new(),
                    live_elements: 9,
                    used_overrides: overrides,
                    published: Vec::new(),
                };
                let error = interpreter
                    .apply_gather_override(function, head, condition, &mut frame, &completion)
                    .unwrap_err();
                drop((held_output, held_values));
                assert_eq!(
                    (interpreter.steps_used, interpreter.used_overrides),
                    (17, overrides)
                );
                assert!(interpreter.trace.is_empty());
                error
            };
            assert!(
                matches!(&error, EvalError::Tool { code: c, .. } if *c == code),
                "fault {fault}: {error:?}"
            );
            assert!(output.borrow().iter().all(|&v| v == 0));
            assert_eq!(*values.borrow(), [7, 11, 19]);
            let document = error.to_diagnostics_json("gather.inv", SOURCE);
            assert_eq!(document["diagnostics"][0]["code"], code);
            assert_eq!(document["diagnostics"][0]["severity"], "error");
            assert_eq!(
                document["diagnostics"][0]["suggested_fixes"],
                serde_json::json!([])
            );
        }
        let registry: serde_json::Value = serde_json::from_str(include_str!(
            "../../schema/diagnostic-codes-indexed-v1.json"
        ))
        .unwrap();
        let mut registered = registry["codes"]
            .as_array()
            .unwrap()
            .iter()
            .filter(|e| e["fault_test"] == "machine_diagnostics_gather_publication_harness_faults")
            .map(|e| e["code"].as_str().unwrap())
            .collect::<Vec<_>>();
        registered.sort_unstable();
        let mut covered = expected.to_vec();
        covered.sort_unstable();
        covered.dedup();
        assert_eq!(covered, registered);
    }
}
