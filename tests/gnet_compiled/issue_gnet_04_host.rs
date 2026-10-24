#![forbid(unsafe_code)]

use invar::{
    eval::Value,
    general_net::{
        compiled::{self, DeviceConfig, Target},
        device_host::{Limits, Plan, Progress, cpu::CpuModel},
        oracle,
    },
};
fn config() -> DeviceConfig {
    DeviceConfig {
        fuel: 1000,
        transitions: 10000,
        trace_events: 2000,
        quantum: 2,
        rounds: 1,
        capacity: 2,
    }
}
fn limits() -> Limits {
    Limits {
        owned: compiled::OwnedLimits {
            elements: 0,
            allocations: 1,
            cleanup_events: 0,
        },
        private: compiled::SpeculationLimits {
            slots: 2,
            ..compiled::SpeculationLimits::default()
        },
        ..Limits::default()
    }
}
#[test]
fn ordinary_source_runtime_inputs_and_continuations_match_ordered_oracle() {
    for target in [Target::Cuda, Target::Metal] {
        let plan=Plan::from_source(b"fn leaf(n:u32)->u32 { return wrapping_add(n,3u32); } fn run(n:u32)->u32 { let a=leaf(n); let b=leaf(n); return wrapping_add(a,b); }","run",target,limits()).unwrap();
        for input in [0, 1, 19, u32::MAX] {
            let values = [Value::U32(input)];
            let cfg = config();
            let reference = oracle::observe_call(
                plan.net(),
                plan.proof(),
                "run",
                &values,
                oracle::Limits {
                    fuel: cfg.fuel,
                    transitions: cfg.transitions,
                    trace_events: cfg.trace_events,
                    array_elements: 0,
                },
            )
            .unwrap();
            let mut backend = CpuModel::default();
            let mut session = plan.prepare(&mut backend, &values, cfg).unwrap();
            let mut continuations = 0;
            let got = loop {
                session.submit().unwrap();
                match session.poll().unwrap() {
                    Progress::Pending => panic!("synchronous model"),
                    Progress::Continuation { .. } => continuations += 1,
                    Progress::Complete(o) => break o,
                }
            };
            assert!(continuations > 0);
            assert_eq!(got.result, Some(reference.semantic.result.unwrap()));
            assert_eq!(got.steps, reference.semantic.steps_used);
            assert_eq!(got.transitions, reference.transitions);
            assert_eq!(got.max_frames, reference.max_frames);
            let events: Vec<[u64; 7]> = reference
                .events
                .iter()
                .map(|e| {
                    [
                        u64::from(e.kind == oracle::EventKind::Complete),
                        e.frame,
                        e.activation,
                        e.token.block.function as u64,
                        e.token.block.block as u64,
                        e.token.position as u64,
                        e.rule as u64,
                    ]
                })
                .collect();
            assert_eq!(got.events, events);
            assert_eq!(session.submitted(), session.completed());
            assert!(session.submit().is_err());
            drop(session.reclaim().unwrap());
        }
    }
}

#[test]
fn original_traps_and_every_source_fuel_prefix_match_actual_device_body() {
    use invar::eval::{EvalError, TrapCategory};
    for (source, value) in [
        ("fn run(n:i32)->i32{return 8/n;}", 0),
        ("fn run(n:i32)->i32{return n+1;}", i32::MAX),
    ] {
        for target in [Target::Cuda, Target::Metal] {
            let plan = Plan::from_source(source.as_bytes(), "run", target, limits()).unwrap();
            let values = [Value::I32(value)];
            let full = oracle::observe_call(
                plan.net(),
                plan.proof(),
                "run",
                &values,
                oracle::Limits::default(),
            )
            .unwrap();
            for fuel in 1..=full.semantic.steps_used + 1 {
                let cfg = DeviceConfig {
                    fuel,
                    capacity: 0,
                    ..config()
                };
                let reference = oracle::observe_call(
                    plan.net(),
                    plan.proof(),
                    "run",
                    &values,
                    oracle::Limits {
                        fuel,
                        transitions: cfg.transitions,
                        trace_events: cfg.trace_events,
                        array_elements: 0,
                    },
                )
                .unwrap();
                let mut backend = CpuModel::default();
                let mut session = plan.prepare(&mut backend, &values, cfg).unwrap();
                let got = loop {
                    session.submit().unwrap();
                    match session.poll().unwrap() {
                        Progress::Continuation { .. } => {}
                        Progress::Complete(value) => break value,
                        Progress::Pending => panic!("synchronous model"),
                    }
                };
                match reference.semantic.result {
                    Err(EvalError::Trap { category, span }) => {
                        assert_eq!(got.span, Some(span));
                        assert_eq!(
                            got.status,
                            match category {
                                TrapCategory::IntegerOverflow => compiled::Status::IntegerOverflow,
                                TrapCategory::DivisionByZero => compiled::Status::DivisionByZero,
                                _ => panic!("unexpected source trap"),
                            }
                        );
                    }
                    Err(EvalError::EvaluationLimit { .. }) => {
                        assert_eq!(got.status, compiled::Status::FuelLimit)
                    }
                    result => panic!("unexpected reference {result:?}"),
                }
                assert_eq!(got.steps, reference.semantic.steps_used);
                assert_eq!(got.transitions, reference.transitions);
                assert_eq!(got.events.len(), reference.events.len());
                for (actual, event) in got.events.iter().zip(reference.events) {
                    assert_eq!(
                        *actual,
                        [
                            u64::from(event.kind == oracle::EventKind::Complete),
                            event.frame,
                            event.activation,
                            event.token.block.function as u64,
                            event.token.block.block as u64,
                            event.token.position as u64,
                            event.rule as u64
                        ]
                    );
                }
                drop(session.reclaim().unwrap());
            }
        }
    }
}

#[test]
fn unsupported_numeric_and_owned_device_inputs_remain_explicit() {
    for source in [
        "fn run(n:f32)->f32{return n+1.0;}",
        "fn run(xs: read Array<i32>)->i32{return xs[0u32];}",
    ] {
        // ABI13 admits these numeric inputs and validated owner results.
        for target in [Target::Cuda, Target::Metal] {
            assert!(Plan::from_source(source.as_bytes(), "run", target, limits()).is_ok());
        }
    }
    for target in [Target::Cuda, Target::Metal] {
        let returned = Plan::from_source(
            b"fn run(n:u32)->Array<i32>{return filled_i32(n, 1);}",
            "run",
            target,
            limits(),
        )
        .unwrap();
        assert_eq!(returned.artifact().abi_version(), 13);
        let refused = Plan::from_source(
            b"fn run(xs: take Array<i32>)->i32{return xs[0u32];}",
            "run",
            target,
            limits(),
        )
        .unwrap_err();
        assert_eq!(
            invar::general_net::device_host::refusal_class(&refused),
            "unsupported_type"
        );
    }
    // GNET-03: an owner local to the program is a device data path.
    let owned = "fn run(n:u32)->i32{let xs:Array<i32> = [1,2];return xs[n];}";
    let storage = Limits {
        owned: compiled::OwnedLimits {
            elements: 4,
            allocations: 2,
            cleanup_events: 2,
        },
        ..limits()
    };
    assert!(Plan::from_source(owned.as_bytes(), "run", Target::Cuda, storage).is_ok());
}

#[test]
fn physical_trace_and_transition_caps_return_errors_without_source_success() {
    let plan = Plan::from_source(
        b"fn run(x:Bool)->Bool{return x==false;}",
        "run",
        Target::Cuda,
        limits(),
    )
    .unwrap();
    let values = [Value::Bool(false)];
    for (cfg, status) in [
        (
            DeviceConfig {
                transitions: 0,
                ..config()
            },
            compiled::Status::TransitionLimit,
        ),
        (
            DeviceConfig {
                trace_events: 0,
                ..config()
            },
            compiled::Status::TraceLimit,
        ),
    ] {
        let mut backend = CpuModel::default();
        let mut session = plan.prepare(&mut backend, &values, cfg).unwrap();
        let observation = loop {
            session.submit().unwrap();
            match session.poll().unwrap() {
                Progress::Complete(value) => break value,
                Progress::Continuation { .. } => {}
                Progress::Pending => panic!("synchronous model"),
            }
        };
        assert_eq!(observation.status, status);
        assert!(observation.result.is_none());
        assert!(observation.events.is_empty());
        drop(session.reclaim().unwrap());
    }
}
