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
fn issue_m4_05_gather_scatter_frontier_histogram_runtime_atomic_forged_receipts_and_live_guards() {
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
fn machine_diagnostics_histogram_publication_harness_faults() {
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
        "../../../schema/diagnostic-codes-indexed-v1.json"
    ))
    .unwrap();
    let registered = registry["codes"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|e| e["fault_test"] == "machine_diagnostics_histogram_publication_harness_faults")
        .map(|e| e["code"].as_str().unwrap())
        .collect::<Vec<_>>();
    assert_eq!(registered, ["HISTOGRAM_COUNT"]);
}
