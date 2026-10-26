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
fn issue_m4_05_gather_scatter_frontier_histogram_runtime_library_prefixes_and_independent_bins() {
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
fn issue_m4_05_gather_scatter_frontier_histogram_runtime_complete_cap_boundary_and_max_refusal() {
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
