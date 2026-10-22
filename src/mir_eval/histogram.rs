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
                    frame: std::mem::replace(
                        frame,
                        Frame {
                            slots: HashMap::new(),
                            values: HashMap::new(),
                        },
                    ),
                    _payload: completion.payload.clone(),
                }),
            };
            if let Ok(mut slot) = completion.quarantine.try_borrow_mut() {
                if slot.is_none() {
                    *slot = Some(pending);
                } else {
                    std::mem::forget(pending);
                }
            } else {
                std::mem::forget(pending);
            }
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
mod tests {
    use super::*;
    use crate::{
        check, front,
        histogram_source::{
            self,
            host::{self, Ledger, Plan},
        },
    };
    const SOURCE: &str = "fn h(a: read Array<u32>, x: read Array<u32>, n: u32) -> Array<u32> { var o = filled_u32(n,0u32); for i in 0u32..len(x) { o[x[i]] = wrapping_add(o[x[i]],a[i]); } return o; }";
    fn context(s: &str) -> Rc<Plan> {
        let checked = check::check(front::parse(s.as_bytes()).unwrap(), false).unwrap();
        let program = Rc::new(mir::lower(&checked).unwrap());
        Rc::new(Plan::bind(s.as_bytes(), &checked, "h", program).unwrap())
    }
    fn compare(a: &crate::eval::Observation, b: &crate::eval::Observation) {
        assert_eq!(a.result, b.result);
        assert_eq!(a.steps_used, b.steps_used);
        assert_eq!(a.cleanup_trace, b.cleanup_trace);
    }
    fn mathematical(a: &[u32], ix: &[u32], n: u32) -> Vec<u32> {
        (0..n)
            .map(|bin| {
                ix.iter()
                    .enumerate()
                    .filter(|(_, b)| **b == bin)
                    .map(|(i, _)| u64::from(a[i]))
                    .sum::<u64>() as u32
            })
            .collect()
    }
    // Test-only library harness invokes the actual ordinary MIR allocation,
    // setup, loop, return and cleanup. It adds no production array binding ABI.
    // The test matrix varies fuel, caps, physical aliasing and receipt faults independently.
    #[allow(clippy::too_many_arguments, clippy::type_complexity)]
    fn run(
        plan: Rc<Plan>,
        a: &[u32],
        ix: &[u32],
        n: u32,
        fuel: u64,
        budget: histogram_source::Budget,
        alias: bool,
        receipt: impl Fn(&mut Completion),
    ) -> (
        crate::eval::Observation,
        usize,
        usize,
        usize,
        Option<Pending>,
    ) {
        let program = plan.program.clone();
        let ledger = Rc::new(RefCell::new(Ledger::new(vec![plan.clone()], budget)));
        let q = Rc::new(RefCell::new(None));
        let mut begun = 0;
        let mut refused = Vec::new();
        let mut provider = |live: &crate::mir_eval::LiveLoop<'_>| {
            if live.function != plan.function().name || live.span != plan.contract.loop_span() {
                return Ok(None);
            }
            let Some(mut c) = host::prepare(
                plan.clone(),
                ledger.clone(),
                live,
                budget,
                q.clone(),
                host::Progress {
                    begun: &mut begun,
                    refusals: &mut refused,
                },
                histogram_source::Contract::runtime_produce,
            )?
            else {
                return Ok(None);
            };
            receipt(&mut c);
            Ok(Some(crate::mir_eval::LoopOverride {
                histogram: Some(c),
                scatter: None,
                gather: None,
                iterations: live.iterations,
                input_lengths: Vec::new(),
                scalar: None,
                array: None,
                tensor: None,
                transpose: None,
                stencil: None,
                summary: None,
                region: None,
            }))
        };
        let values = Rc::new(RefCell::new(a.to_vec()));
        let indices = if alias {
            values.clone()
        } else {
            Rc::new(RefCell::new(ix.to_vec()))
        };
        let mut interpreter = Interpreter {
            program: &program,
            provider: if budget.visits == 0 {
                None
            } else {
                Some(&mut provider)
            },
            sorts: None,
            reducer: None,
            fuel_limit: fuel,
            steps_used: 0,
            call_depth: 0,
            trace: Vec::new(),
            live_elements: a.len() + if alias { 0 } else { ix.len() },
            used_overrides: 0,
            published: Vec::new(),
        };
        let result = if fuel == 0 {
            Err(tool("INVALID_FUEL", "fuel must be positive"))
        } else {
            interpreter.call(
                "h",
                vec![
                    Value::ArrayU32(values.clone()),
                    Value::ArrayU32(indices.clone()),
                    Value::U32(n),
                ],
            )
        };
        assert_eq!(*values.borrow(), a);
        assert_eq!(*indices.borrow(), if alias { a } else { ix });
        let obs = crate::eval::Observation {
            result,
            steps_used: interpreter.steps_used,
            cleanup_trace: interpreter.trace,
        };
        let completed = interpreter.used_overrides;
        let l = ledger.borrow();
        let visited = l.decision_visits + l.attempts.iter().map(|m| m.borrow().used).sum::<usize>();
        assert!(visited <= l.reserved);
        assert!(l.reserved <= budget.visits.min(65536));
        let pending = q.borrow_mut().take();
        (obs, begun, completed, l.reserved, pending)
    }
    #[test]
    fn issue_m4_05_gather_scatter_frontier_histogram_runtime_library_prefixes_and_independent_bins()
    {
        let shapes = [
            (vec![], vec![], 0),
            (vec![7], vec![0], 1),
            (vec![u32::MAX, 2, 7, 11, 31], vec![3, 3, 0, 3], 7),
            (vec![0; 6], vec![2, 1, 2, 1, 2, 2], 4),
            (vec![0, 1, 1, 0], vec![0, 1, 1, 0], 3),
        ];
        let mut total = 0;
        let mut pubs = 0;
        for source in [
            SOURCE.to_string(),
            SOURCE.replace("wrapping_add(o[x[i]],a[i])", "wrapping_add(a[i],o[x[i]])"),
            SOURCE.replace(
                "o[x[i]] = wrapping_add(o[x[i]],a[i])",
                "o[((x[(i)]))] = (wrapping_add((o[(x[i])]),(a[(i)])))",
            ),
            SOURCE.replace("return o;", "return (move o);"),
        ] {
            let plan = context(&source);
            for (a, ix, n) in &shapes {
                let done = run(
                    plan.clone(),
                    a,
                    ix,
                    *n,
                    10000,
                    Default::default(),
                    false,
                    |_| {},
                );
                let Ok(Value::ArrayU32(out)) = &done.0.result else {
                    panic!("{:?}", done.0.result)
                };
                assert_eq!(*out.borrow(), mathematical(a, ix, *n));
                assert_eq!(done.2, usize::from(!ix.is_empty()));
                assert_eq!(done.1, done.2);
                for fuel in 0..=done.0.steps_used + 1 {
                    let ordinary = run(
                        plan.clone(),
                        a,
                        ix,
                        *n,
                        fuel,
                        histogram_source::Budget {
                            bytes: 0,
                            visits: 0,
                        },
                        false,
                        |_| {},
                    )
                    .0;
                    let actual = run(
                        plan.clone(),
                        a,
                        ix,
                        *n,
                        fuel,
                        Default::default(),
                        false,
                        |_| {},
                    );
                    compare(&actual.0, &ordinary);
                    assert_eq!(actual.1, actual.2);
                    assert!(actual.4.is_none());
                    total += 1;
                    pubs += actual.2;
                }
            }
        }
        println!(
            "library optimized/ordinary MIR prefixes={total}, publications={pubs}; exact independent full bin arrays"
        );
    }
    #[test]
    fn issue_m4_05_gather_scatter_frontier_histogram_runtime_library_guards_alias_and_traps() {
        let plan = context(SOURCE);
        for (a, ix, n, alias) in [
            (vec![0, 1, 1, 0], vec![0, 1, 1, 0], 3, true),
            (vec![], vec![0], 1, false),
            (vec![3], vec![1], 1, false),
            (vec![3], vec![0], 0, false),
            (vec![3], vec![0, 0], 1, false),
        ] {
            let actual = run(
                plan.clone(),
                &a,
                &ix,
                n,
                10000,
                Default::default(),
                alias,
                |_| {},
            );
            if alias {
                assert_eq!(actual.2, 1);
                let Ok(Value::ArrayU32(out)) = actual.0.result else {
                    panic!()
                };
                assert_eq!(*out.borrow(), mathematical(&a, &ix, n));
                continue;
            }
            assert_eq!(actual.1, 0);
            assert_eq!(actual.2, 0);
            for fuel in 0..=actual.0.steps_used + 1 {
                let ordinary = run(
                    plan.clone(),
                    &a,
                    &ix,
                    n,
                    fuel,
                    histogram_source::Budget {
                        bytes: 0,
                        visits: 0,
                    },
                    false,
                    |_| {},
                )
                .0;
                compare(
                    &run(
                        plan.clone(),
                        &a,
                        &ix,
                        n,
                        fuel,
                        Default::default(),
                        false,
                        |_| {},
                    )
                    .0,
                    &ordinary,
                );
            }
        }
    }
    #[test]
    fn issue_m4_05_gather_scatter_frontier_histogram_runtime_complete_cap_boundary_and_max_refusal()
    {
        let plan = context(SOURCE);
        let a = [7, 11, 19, 101];
        let ix = [1, 0, 1];
        let full = run(
            plan.clone(),
            &a,
            &ix,
            4,
            10000,
            Default::default(),
            false,
            |_| {},
        );
        assert_eq!(full.2, 1);
        for limit in [full.3 - 1, full.3, full.3 + 1] {
            let actual = run(
                plan.clone(),
                &a,
                &ix,
                4,
                10000,
                histogram_source::Budget {
                    bytes: 160 * 1024,
                    visits: limit,
                },
                false,
                |_| {},
            );
            compare(&actual.0, &full.0);
            assert_eq!(actual.2, usize::from(limit >= full.3));
        }
        let cost = plan
            .contract
            .runtime_cost(a.len(), ix.len(), 4, Default::default())
            .unwrap();
        for bytes in [cost.bytes - 1, cost.bytes, cost.bytes + 1] {
            assert_eq!(
                run(
                    plan.clone(),
                    &a,
                    &ix,
                    4,
                    10000,
                    histogram_source::Budget {
                        bytes,
                        visits: 65536
                    },
                    false,
                    |_| {}
                )
                .2,
                usize::from(bytes >= cost.bytes)
            );
        }
        assert!(
            plan.contract
                .runtime_cost(
                    4096,
                    128,
                    4096,
                    histogram_source::Budget {
                        bytes: usize::MAX,
                        visits: usize::MAX
                    }
                )
                .is_err()
        );
        for (a, i, n) in [(50_000, 1, 1), (1, 2000, 1), (1, 1, 50_000)] {
            assert!(
                plan.contract
                    .runtime_cost(a, i, n, Default::default())
                    .is_err()
            );
        }
        println!(
            "library full reserve={}, data bytes={}, complete simultaneous maxima refused",
            full.3, cost.bytes
        );
    }
    #[test]
    fn issue_m4_05_gather_scatter_frontier_histogram_runtime_admitted_component_limits_prefixes() {
        let plan = context(SOURCE);
        let mut count = 0;
        let mut publications = 0;
        for (a_len, i_len, n) in [(4096, 64, 128), (128, 128, 128), (1, 1, 2048)] {
            let a = vec![u32::MAX; a_len];
            let ix = (0..i_len)
                .map(|i| (i % 7).min((n - 1) as usize) as u32)
                .collect::<Vec<_>>();
            let done = run(
                plan.clone(),
                &a,
                &ix,
                n,
                10000,
                Default::default(),
                false,
                |_| {},
            );
            assert_eq!(done.2, 1);
            let Ok(Value::ArrayU32(output)) = &done.0.result else {
                panic!()
            };
            assert_eq!(*output.borrow(), mathematical(&a, &ix, n));
            for fuel in 0..=done.0.steps_used + 1 {
                let expected = run(
                    plan.clone(),
                    &a,
                    &ix,
                    n,
                    fuel,
                    histogram_source::Budget {
                        bytes: 0,
                        visits: 0,
                    },
                    false,
                    |_| {},
                );
                let actual = run(
                    plan.clone(),
                    &a,
                    &ix,
                    n,
                    fuel,
                    Default::default(),
                    false,
                    |_| {},
                );
                compare(&actual.0, &expected.0);
                assert_eq!(actual.1, actual.2);
                count += 1;
                publications += actual.2;
            }
        }
        println!(
            "admitted component-limit optimized/ordinary MIR prefixes={count}, publications={publications}"
        );
    }
    #[test]
    fn issue_m4_05_gather_scatter_frontier_histogram_runtime_unknown_and_failed_completion() {
        for status in [
            FenceObservation::Unknown,
            FenceObservation::Pending,
            FenceObservation::FailedQuiescent,
        ] {
            let plan = context(SOURCE);
            let mut weak = None;
            // Receipt editing is test-only; Unknown must retain context even if
            // the payload's later checks would fail.
            let held = RefCell::new(None);
            let actual = run(
                plan,
                &[7, 11, 19],
                &[1, 0, 1],
                3,
                10000,
                Default::default(),
                false,
                |c| {
                    c.status = status;
                    let p = Rc::get_mut(&mut c.payload).unwrap();
                    p.ticks += 1;
                    *held.borrow_mut() = Some(Rc::downgrade(&p.owners[2]));
                },
            );
            assert_eq!(actual.1, 1);
            assert_eq!(actual.2, 0);
            assert!(matches!(actual.0.result, Err(EvalError::Tool { .. })));
            assert!(actual.0.cleanup_trace.is_empty());
            weak = held.borrow_mut().take().or(weak);
            if matches!(
                status,
                FenceObservation::Unknown | FenceObservation::Pending
            ) {
                assert!(actual.4.as_ref().unwrap().retained_arrays() >= 3);
                assert!(weak.as_ref().unwrap().upgrade().is_some());
                drop(actual.4);
                assert!(weak.unwrap().upgrade().is_some());
            } else {
                assert!(actual.4.is_none());
                assert!(weak.unwrap().upgrade().is_none());
            }
        }
    }
}
#[cfg(test)]
mod atomic_tests {
    use super::*;
    use crate::{
        check, front,
        histogram_source::{
            self,
            host::{self, Ledger, Plan},
        },
    };
    const SOURCE: &str = "fn h(a: read Array<u32>, x: read Array<u32>, n: u32) -> Array<u32> { var o = filled_u32(n,0u32); for i in 0u32..len(x) { o[x[i]] = wrapping_add(o[x[i]],a[i]); } return o; }";
    type Setup = (Rc<Plan>, Frame, Rc<RefCell<Ledger>>, Rc<RefCell<Vec<u32>>>);
    fn setup() -> Setup {
        let c = check::check(front::parse(SOURCE.as_bytes()).unwrap(), false).unwrap();
        let m = Rc::new(mir::lower(&c).unwrap());
        let plan = Rc::new(Plan::bind(SOURCE.as_bytes(), &c, "h", m).unwrap());
        let mut frame = Frame::new(
            plan.function(),
            vec![
                Value::ArrayU32(Rc::new(RefCell::new(vec![0; 5]))),
                Value::ArrayU32(Rc::new(RefCell::new(vec![3, 1, 3, 0]))),
                Value::U32(7),
            ],
        );
        let out = Rc::new(RefCell::new(vec![0; 7]));
        frame.slots.insert(
            "o".into(),
            Slot {
                value: Some(Value::ArrayU32(out.clone())),
                owned: true,
            },
        );
        frame.values.insert(plan.left, Value::U32(0));
        frame.values.insert(plan.right, Value::U32(4));
        let ledger = Rc::new(RefCell::new(Ledger::new(
            vec![plan.clone()],
            Default::default(),
        )));
        (plan, frame, ledger, out)
    }
    fn prepare(
        plan: Rc<Plan>,
        frame: &Frame,
        ledger: Rc<RefCell<Ledger>>,
        begun: &mut usize,
    ) -> Option<Completion> {
        let function = plan.function();
        let head = &function.blocks[plan.head_index];
        let Some(Terminator::Branch { yes, .. }) = head.end else {
            panic!()
        };
        let live = super::super::bulk::live_loop(
            function,
            head,
            yes,
            plan.condition,
            plan.contract.loop_span(),
            frame,
            10000,
        )
        .unwrap();
        host::prepare(
            plan.clone(),
            ledger,
            &live,
            Default::default(),
            Rc::new(RefCell::new(None)),
            host::Progress {
                begun,
                refusals: &mut Vec::new(),
            },
            histogram_source::Contract::runtime_produce,
        )
        .unwrap()
    }
    #[test]
    fn issue_m4_05_gather_scatter_frontier_histogram_runtime_atomic_forged_receipts_and_live_guards()
     {
        for fault in 0..37 {
            let (plan, mut frame, ledger, out) = setup();
            let mut begun = 0;
            let mut completion = prepare(plan.clone(), &frame, ledger.clone(), &mut begun).unwrap();
            assert_eq!(begun, 1);
            let p = Rc::get_mut(&mut completion.payload).unwrap();
            match fault {
                0..=20 => p.data.corrupt(fault),
                21 => p.ticks += 1,
                22 => p.cost.visits += 1,
                23 => p.meter.borrow_mut().used += 1,
                24 => p.meter.borrow_mut().limit += 1,
                25 => p.owners[0].borrow_mut()[4] = 1,
                26 => frame.slots.get_mut("o").unwrap().owned = false,
                27 => frame.slots.get_mut("a").unwrap().owned = true,
                28 => frame.slots.get_mut("x").unwrap().owned = true,
                29 => frame.slots.get_mut("n").unwrap().value = Some(Value::U32(8)),
                30 => {
                    // Preserve the exact allocation/words while substituting
                    // the Invar owner. A slice-address guard must not mask the
                    // required strong-owner identity check.
                    let pointer = p.owners[1].borrow().as_ptr();
                    let words = std::mem::take(&mut *p.owners[1].borrow_mut());
                    assert_eq!(words.as_ptr(), pointer);
                    frame.slots.get_mut("x").unwrap().value =
                        Some(Value::ArrayU32(Rc::new(RefCell::new(words))));
                }
                31 => {
                    let value = copied(frame.slot("a").unwrap());
                    frame.slots.get_mut("o").unwrap().value = Some(value);
                }
                32 => p.plan = setup().0,
                35 => {
                    p.ledger = Rc::new(RefCell::new(Ledger::new(
                        vec![setup().0],
                        Default::default(),
                    )))
                }
                36 => {
                    p.ledger = Rc::new(RefCell::new(Ledger::new(
                        vec![plan.clone()],
                        Default::default(),
                    )))
                }
                _ => {}
            }
            let held = if fault == 33 {
                Some(out.borrow_mut())
            } else {
                None
            };
            if fault == 34 {
                out.borrow_mut()[6] = 1;
            }
            let mut interpreter = Interpreter {
                program: &plan.program,
                provider: None,
                sorts: None,
                reducer: None,
                fuel_limit: 10000,
                steps_used: 17,
                call_depth: 0,
                trace: Vec::new(),
                live_elements: 16,
                used_overrides: 0,
                published: Vec::new(),
            };
            let error = interpreter
                .apply_histogram_override(
                    plan.function(),
                    &plan.function().blocks[plan.head_index],
                    plan.condition,
                    &mut frame,
                    &completion,
                )
                .unwrap_err();
            assert!(matches!(error, EvalError::Tool { .. }));
            assert_eq!(interpreter.steps_used, 17);
            assert_eq!(interpreter.used_overrides, 0);
            assert!(interpreter.trace.is_empty());
            drop(held);
            assert_eq!(
                *out.borrow(),
                if fault == 34 {
                    vec![0, 0, 0, 0, 0, 0, 1]
                } else {
                    vec![0; 7]
                }
            );
            assert!(ledger.borrow().reserved <= 65536);
        }
    }
    #[test]
    fn issue_m4_05_gather_scatter_frontier_histogram_runtime_pre_entry_owner_refusals() {
        for fault in 0..4 {
            let (plan, mut frame, ledger, out) = setup();
            match fault {
                0 => frame.slots.get_mut("o").unwrap().owned = false,
                1 => frame.slots.get_mut("a").unwrap().owned = true,
                2 => frame.slots.get_mut("x").unwrap().owned = true,
                _ => {
                    let input = copied(frame.slot("a").unwrap());
                    frame.slots.get_mut("o").unwrap().value = Some(input);
                    frame.slots.get_mut("n").unwrap().value = Some(Value::U32(5));
                }
            }
            let mut begun = 0;
            assert!(prepare(plan, &frame, ledger, &mut begun).is_none());
            assert_eq!(begun, 0);
            assert_eq!(*out.borrow(), vec![0; 7]);
        }
    }
    #[test]
    fn issue_m4_05_gather_scatter_frontier_histogram_runtime_failed_attempt_budget_never_resets() {
        let (plan, frame, ledger, out) = setup();
        let Value::ArrayU32(indices) = frame.slot("x").unwrap() else {
            panic!()
        };
        indices.borrow_mut()[0] = 7;
        let mut begun = 0;
        assert!(prepare(plan.clone(), &frame, ledger.clone(), &mut begun).is_none());
        assert_eq!(begun, 0);
        let first = ledger.borrow().reserved;
        assert!(first > 32769);
        indices.borrow_mut()[0] = 3;
        for _ in 0..16 {
            assert!(prepare(plan.clone(), &frame, ledger.clone(), &mut begun).is_none());
            assert_eq!(begun, 0);
            assert!(ledger.borrow().reserved >= first);
            assert!(ledger.borrow().reserved <= 65536);
        }
        assert_eq!(*out.borrow(), vec![0; 7]);
        println!("failed attempt complete reservation={first}; 16 repaired retries retain budget");
    }
    #[test]
    fn issue_m4_05_gather_scatter_frontier_histogram_runtime_entry_precedes_producer_error() {
        let (plan, frame, ledger, out) = setup();
        let f = plan.function();
        let head = &f.blocks[plan.head_index];
        let Some(Terminator::Branch { yes, .. }) = head.end else {
            panic!()
        };
        let live = super::super::bulk::live_loop(
            f,
            head,
            yes,
            plan.condition,
            plan.contract.loop_span(),
            &frame,
            10000,
        )
        .unwrap();
        let mut begun = 0;
        let error = host::prepare(
            plan.clone(),
            ledger,
            &live,
            Default::default(),
            Rc::new(RefCell::new(None)),
            host::Progress {
                begun: &mut begun,
                refusals: &mut Vec::new(),
            },
            |_, _, _| Err(histogram_source::Error::Allocation),
        )
        .err()
        .unwrap();
        assert!(matches!(
            error,
            EvalError::Tool {
                code: "HISTOGRAM_POST_ENTRY",
                ..
            }
        ));
        assert_eq!(begun, 1);
        assert_eq!(*out.borrow(), vec![0; 7]);
    }

    /// Real injections through the direct harness (review3 R2): the
    /// publication counter and the host entry counter at their maximum reach
    /// HISTOGRAM_COUNT with no write, debit, trace or producer entry.
    #[test]
    fn issue_ai_01_machine_diagnostics_histogram_publication_harness_faults() {
        let expected = [
            "HISTOGRAM_COUNT", // publication counter at its maximum
            "HISTOGRAM_COUNT", // host entry counter at its maximum
        ];
        for (fault, code) in expected.into_iter().enumerate() {
            let (plan, mut frame, ledger, out) = setup();
            let error = if fault == 1 {
                let f = plan.function();
                let head = &f.blocks[plan.head_index];
                let Some(Terminator::Branch { yes, .. }) = head.end else {
                    panic!()
                };
                let live = super::super::bulk::live_loop(
                    f,
                    head,
                    yes,
                    plan.condition,
                    plan.contract.loop_span(),
                    &frame,
                    10000,
                )
                .unwrap();
                let mut begun = usize::MAX;
                let prepared = host::prepare(
                    plan.clone(),
                    ledger.clone(),
                    &live,
                    Default::default(),
                    Rc::new(RefCell::new(None)),
                    host::Progress {
                        begun: &mut begun,
                        refusals: &mut Vec::new(),
                    },
                    |_, _, _| panic!("producer entered after entry counter overflow"),
                );
                assert_eq!(begun, usize::MAX);
                let Err(error) = prepared else {
                    panic!("entry counter overflow was admitted");
                };
                error
            } else {
                let mut begun = 0;
                let completion = prepare(plan.clone(), &frame, ledger.clone(), &mut begun).unwrap();
                let mut interpreter = Interpreter {
                    program: &plan.program,
                    provider: None,
                    sorts: None,
                    reducer: None,
                    fuel_limit: 10000,
                    steps_used: 17,
                    call_depth: 0,
                    trace: Vec::new(),
                    live_elements: 16,
                    used_overrides: usize::MAX,
                    published: Vec::new(),
                };
                let error = interpreter
                    .apply_histogram_override(
                        plan.function(),
                        &plan.function().blocks[plan.head_index],
                        plan.condition,
                        &mut frame,
                        &completion,
                    )
                    .unwrap_err();
                assert_eq!(
                    (interpreter.steps_used, interpreter.used_overrides),
                    (17, usize::MAX)
                );
                assert!(interpreter.trace.is_empty());
                error
            };
            assert!(
                matches!(&error, EvalError::Tool { code: c, .. } if *c == code),
                "fault {fault}: {error:?}"
            );
            assert_eq!(*out.borrow(), vec![0; 7]);
            let document = error.to_diagnostics_json("histogram.inv", SOURCE.as_bytes());
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
        let registered = registry["codes"]
            .as_array()
            .unwrap()
            .iter()
            .filter(|e| {
                e["fault_test"]
                    == "issue_ai_01_machine_diagnostics_histogram_publication_harness_faults"
            })
            .map(|e| e["code"].as_str().unwrap())
            .collect::<Vec<_>>();
        assert_eq!(registered, ["HISTOGRAM_COUNT"]);
    }
}
