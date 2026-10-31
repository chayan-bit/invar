//! Full fresh-output publication for statically proved checked-source scatters.
use super::*;
use crate::{
    general_net::compiled::FenceObservation,
    scatter_proof::host::{self, Completion, Payload, Pending, failure},
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
    #[cfg(test)]
    pub(crate) fn output_weak(&self) -> std::rc::Weak<RefCell<Vec<u32>>> {
        let Value::ArrayU32(out) = self.frame.slot(&self._payload.plan.output).unwrap() else {
            panic!()
        };
        Rc::downgrade(out)
    }
}

impl Interpreter<'_> {
    pub(super) fn apply_scatter_override(
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
            let pending = Pending {
                resources: Some(Quarantined {
                    frame: std::mem::take(frame),
                    _payload: completion.payload.clone(),
                }),
            };
            park(&completion.quarantine, pending);
            return Err(failure("SCATTER_UNKNOWN_COMPLETION"));
        }
        if completion.status != FenceObservation::Complete {
            return Err(failure("SCATTER_FAILED_COMPLETION"));
        }
        let p = &completion.payload;
        p.plan.verify()?;
        p.plan.verify_function(function)?;
        if p.source_hash != p.plan.source_hash
            || self.program.contract_digest() != p.plan.program_digest
            || function.name != p.plan.function
            || !matches!(head.end,Some(Terminator::Branch { span, .. }) if span == p.plan.contract.certificate().loop_span())
        {
            return Err(failure("SCATTER_SOURCE_BINDING"));
        }
        let i = p.plan.iterations();
        let n = p.plan.output_len();
        if live_trips(head, condition, frame).is_none_or(|v| v as usize != i)
            || p.plan.loop_ticks() != Some(p.ticks)
            || p.ticks > self.fuel_limit.saturating_sub(self.steps_used)
        {
            return Err(failure("SCATTER_LOOP_BINDING"));
        }
        let next_steps = self
            .steps_used
            .checked_add(p.ticks)
            .ok_or_else(|| failure("SCATTER_FUEL"))?;
        let next_completed = self
            .used_overrides
            .checked_add(1)
            .ok_or_else(|| failure("SCATTER_COUNT"))?;
        let Value::ArrayU32(input) = frame.slot(&p.plan.input)? else {
            return Err(failure("SCATTER_OWNER"));
        };
        let Value::ArrayU32(output) = frame.slot(&p.plan.output)? else {
            return Err(failure("SCATTER_OWNER"));
        };
        if !frame.slots.get(&p.plan.output).is_some_and(|s| s.owned) || Rc::ptr_eq(input, output) {
            return Err(failure("SCATTER_OWNER"));
        }
        let input = input.try_borrow().map_err(|_| failure("SCATTER_LOAN"))?;
        let mut output = output
            .try_borrow_mut()
            .map_err(|_| failure("SCATTER_LOAN"))?;
        if [
            std::ptr::from_ref(&*input).addr(),
            std::ptr::from_ref(&*output).addr(),
        ] != p.identities
            || *input != p.input
            || output.len() != n
            || p.output.len() != n
            || p.accesses.len() != i
            || host::cost(input.len(), output.len(), i, p.budget) != Some(p.cost)
        {
            return Err(failure("SCATTER_STALE_COMPLETION"));
        }
        // Dense validation checks every candidate word, including all zero holes.
        // Algebraic injectivity already proves strictly increasing destinations
        // for i>1; exact ordered records must still bind this source expression.
        let mut k = 0;
        for (d, (&zero, &candidate)) in output.iter().zip(&p.output).enumerate() {
            if zero != 0 {
                return Err(failure("SCATTER_BAD_COMPLETION"));
            }
            if k < i && p.plan.destination(k) == Some(d) {
                let access = &p.accesses[k];
                if access.iteration != k
                    || access.read_index != k
                    || access.write_index != d
                    || k.checked_mul(size_of::<u32>()).is_none()
                    || d.checked_mul(size_of::<u32>()).is_none()
                    || input.get(k) != Some(&candidate)
                {
                    return Err(failure("SCATTER_BAD_COMPLETION"));
                }
                k += 1;
            } else if candidate != 0 {
                return Err(failure("SCATTER_BAD_COMPLETION"));
            }
        }
        if k != i {
            return Err(failure("SCATTER_BAD_COMPLETION"));
        }
        // All checks, loans and arithmetic precede publication; no fallible work
        // follows the first source write or debit.
        output.copy_from_slice(&p.output);
        self.steps_used = next_steps;
        self.used_overrides = next_completed;
        Ok(true)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use host::{Budget, MAX_STAGING_BYTES, MAX_VISITS, Plan};
    const SOURCE: &[u8] = b"fn scatter(a: read Array<u32>) -> Array<u32> { var out = filled_u32(5u32,0u32); for i in 0u32..3u32 { out[2u32*i] = a[i]; } return out; }";
    fn reach<'a>(
        function: &'a Function,
        plan: &Plan,
        input: Rc<RefCell<Vec<u32>>>,
        output: Rc<RefCell<Vec<u32>>>,
    ) -> (Frame, &'a mir::BasicBlock, ValueId, usize, ValueId, ValueId) {
        let (head, condition, body) = function
            .blocks
            .iter()
            .find_map(|b| match &b.end {
                Some(Terminator::Branch {
                    condition,
                    yes,
                    span,
                    ..
                }) if *span == plan.contract.certificate().loop_span() => {
                    Some((b, *condition, *yes))
                }
                _ => None,
            })
            .unwrap();
        let (left, right) = head
            .instructions
            .iter()
            .find_map(|ins| match ins.kind {
                Operation::Binary {
                    op: BinaryOp::Lt,
                    left,
                    right,
                    ..
                } => Some((left, right)),
                _ => None,
            })
            .unwrap();
        let mut frame = Frame::new(function, vec![Value::ArrayU32(input)]);
        frame.slots.insert(
            "out".into(),
            Slot {
                value: Some(Value::ArrayU32(output)),
                owned: true,
            },
        );
        frame.values.insert(left, Value::U32(0));
        frame.values.insert(right, Value::U32(3));
        (frame, head, condition, body, left, right)
    }
    #[test]
    fn issue_m4_05_gather_scatter_frontier_scatter_runtime_pre_entry_refusals() {
        for fault in 0..9 {
            let checked = crate::check::check(crate::front::parse(SOURCE).unwrap(), false).unwrap();
            let program = mir::lower(&checked).unwrap();
            let mut plan = Plan::bind(SOURCE.into(), &checked, "scatter", &program).unwrap();
            let input = Rc::new(RefCell::new(vec![7, 11, 19]));
            let output = Rc::new(RefCell::new(vec![0; 5]));
            let (frame, head, condition, body, _, _) =
                reach(&program.functions[0], &plan, input.clone(), output.clone());
            match fault {
                0 => plan.source_hash[0] ^= 1,
                1 => plan.function_digest[0] ^= 1,
                2 => plan.program_digest[0] ^= 1,
                3 => {
                    // All non-alias guards would admit this synthetic live
                    // footprint: correct extent, zero contents and read range.
                    input.borrow_mut().resize(5, 0);
                    input.borrow_mut().fill(0);
                    plan.output = plan.input.clone();
                }
                4 => {
                    input.borrow_mut().pop();
                }
                5 => {
                    output.borrow_mut().pop();
                }
                6 => output.borrow_mut()[1] = 42,
                _ => {}
            }
            let live = super::super::bulk::live_loop(
                &program.functions[0],
                head,
                body,
                condition,
                plan.contract.certificate().loop_span(),
                &frame,
                1000,
            )
            .unwrap();
            let budget = match fault {
                7 => Budget {
                    bytes: 0,
                    visits: MAX_VISITS,
                },
                8 => Budget {
                    bytes: MAX_STAGING_BYTES,
                    visits: 0,
                },
                _ => Budget::default(),
            };
            let before = output.borrow().clone();
            let mut begun = 0;
            let mut refusals = Vec::new();
            assert!(
                host::prepare(
                    Rc::new(plan),
                    &live,
                    budget,
                    Rc::new(RefCell::new(None)),
                    &mut begun,
                    &mut refusals,
                    host::produce_sync
                )
                .unwrap()
                .is_none()
            );
            assert_eq!(begun, 0);
            assert_eq!(*output.borrow(), before);
            assert_eq!(refusals.len(), 1);
        }
    }
    #[test]
    fn issue_m4_05_gather_scatter_frontier_scatter_runtime_publication_no_writes_or_debit() {
        for fault in 0..12 {
            let checked = crate::check::check(crate::front::parse(SOURCE).unwrap(), false).unwrap();
            let program = mir::lower(&checked).unwrap();
            let plan = Plan::bind(SOURCE.into(), &checked, "scatter", &program).unwrap();
            let input = Rc::new(RefCell::new(vec![7, 11, 19]));
            let output = Rc::new(RefCell::new(vec![0; 5]));
            let (mut frame, head, condition, body, left, right) =
                reach(&program.functions[0], &plan, input.clone(), output.clone());
            let live = super::super::bulk::live_loop(
                &program.functions[0],
                head,
                body,
                condition,
                plan.contract.certificate().loop_span(),
                &frame,
                1000,
            )
            .unwrap();
            let mut begun = 0;
            let mut refusals = Vec::new();
            let mut completion = host::prepare(
                Rc::new(plan),
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
                    frame.slots.get_mut("out").unwrap().value = Some(Value::ArrayU32(input.clone()))
                }
                1 => frame.slots.get_mut("out").unwrap().owned = false,
                2 => {
                    output.borrow_mut().pop();
                }
                3 => output.borrow_mut()[1] = 77,
                4 => input.borrow_mut()[0] ^= 1,
                5 => Rc::get_mut(&mut completion.payload).unwrap().source_hash[0] ^= 1,
                6 => Rc::get_mut(&mut completion.payload).unwrap().ticks += 1,
                7 => {
                    frame.values.insert(left, Value::U32(1));
                }
                8 => {
                    frame.values.insert(right, Value::U32(2));
                }
                9 => {
                    Rc::get_mut(&mut Rc::get_mut(&mut completion.payload).unwrap().plan)
                        .unwrap()
                        .function_digest[0] ^= 1
                }
                _ => {}
            }
            let before = output.borrow().clone();
            let input_before = input.borrow().clone();
            let held = if fault == 10 {
                Some(output.borrow_mut())
            } else {
                None
            };
            let mut wrong_program = program.clone();
            wrong_program.functions[0].name = "stale".into();
            let mut interpreter = Interpreter {
                program: if fault == 11 {
                    &wrong_program
                } else {
                    &program
                },
                provider: None,
                sorts: None,
                reducer: None,
                fuel_limit: 1000,
                steps_used: 17,
                call_depth: 0,
                trace: Vec::new(),
                live_elements: 8,
                used_overrides: 0,
                published: Vec::new(),
            };
            let error = interpreter
                .apply_scatter_override(
                    &program.functions[0],
                    head,
                    condition,
                    &mut frame,
                    &completion,
                )
                .unwrap_err();
            if fault == 0 || fault == 1 {
                assert!(matches!(
                    error,
                    EvalError::Tool {
                        code: "SCATTER_OWNER",
                        ..
                    }
                ));
            }
            drop(held);
            assert_eq!(*output.borrow(), before);
            assert_eq!(*input.borrow(), input_before);
            assert_eq!(interpreter.steps_used, 17);
            assert_eq!(interpreter.used_overrides, 0);
            assert!(interpreter.trace.is_empty());
        }
    }

    /// Real injections through the direct publication harness (review3 R2):
    /// owner replacement, held loans and both counters reach their typed
    /// codes with no write, debit or trace.
    #[test]
    fn machine_diagnostics_scatter_publication_harness_faults() {
        let expected = [
            "SCATTER_OWNER", // output slot aliases the input owner
            "SCATTER_OWNER", // output slot is not owned
            "SCATTER_LOAN",  // output held by a live borrow
            "SCATTER_LOAN",  // input held mutably
            "SCATTER_COUNT", // publication counter at its maximum
            "SCATTER_COUNT", // host entry counter at its maximum
        ];
        for (fault, code) in expected.into_iter().enumerate() {
            let checked = crate::check::check(crate::front::parse(SOURCE).unwrap(), false).unwrap();
            let program = mir::lower(&checked).unwrap();
            let plan = Plan::bind(SOURCE.into(), &checked, "scatter", &program).unwrap();
            let input = Rc::new(RefCell::new(vec![7, 11, 19]));
            let output = Rc::new(RefCell::new(vec![0; 5]));
            let (mut frame, head, condition, body, _, _) =
                reach(&program.functions[0], &plan, input.clone(), output.clone());
            let live = super::super::bulk::live_loop(
                &program.functions[0],
                head,
                body,
                condition,
                plan.contract.certificate().loop_span(),
                &frame,
                1000,
            )
            .unwrap();
            let mut begun = if fault == 5 { usize::MAX } else { 0 };
            let prepared = host::prepare(
                Rc::new(plan),
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
                            Some(Value::ArrayU32(input.clone()))
                    }
                    1 => frame.slots.get_mut("out").unwrap().owned = false,
                    _ => {}
                }
                let held_output = (fault == 2).then(|| output.borrow_mut());
                let held_input = (fault == 3).then(|| input.borrow_mut());
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
                    live_elements: 8,
                    used_overrides: overrides,
                    published: Vec::new(),
                };
                let error = interpreter
                    .apply_scatter_override(
                        &program.functions[0],
                        head,
                        condition,
                        &mut frame,
                        &completion,
                    )
                    .unwrap_err();
                drop((held_output, held_input));
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
            assert_eq!(*output.borrow(), [0; 5]);
            assert_eq!(*input.borrow(), [7, 11, 19]);
            let document = error.to_diagnostics_json("scatter.inv", SOURCE);
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
            .filter(|e| e["fault_test"] == "machine_diagnostics_scatter_publication_harness_faults")
            .map(|e| e["code"].as_str().unwrap())
            .collect::<Vec<_>>();
        registered.sort_unstable();
        let mut covered = expected.to_vec();
        covered.sort_unstable();
        covered.dedup();
        assert_eq!(covered, registered);
    }
}
