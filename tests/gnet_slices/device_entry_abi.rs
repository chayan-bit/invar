#![forbid(unsafe_code)]

use crate::entry;

use invar::{
    eval::Value,
    general_net::{
        compiled::{self, Target},
        device_host::{self, EntryRefusal, Plan, cpu::CpuModel},
    },
};
use std::{
    cell::{Cell, RefCell},
    rc::Rc,
};
#[test]
fn cpu_group_models_match_oracle_for_arrays_records_and_strict_f32() {
    entry::corpus(&mut CpuModel::default(), Target::Cuda);
    entry::corpus(&mut CpuModel::default(), Target::Metal);
}
fn refusal(plan: &Plan, args: &[Value], expected: EntryRefusal) {
    let mut backend = CpuModel::with_compiler(std::path::Path::new("/nonexistent-p0b-compiler"));
    match plan.prepare(&mut backend, args, entry::config()) {
        Err(device_host::Error::Entry(got)) => assert_eq!(got, expected),
        Err(e) => panic!("wrong refusal {e:?}"),
        Ok(_) => panic!("unexpected admission"),
    };
}
#[test]
fn typed_refusals_precede_backend_allocation_and_keep_caller_unchanged() {
    let plan =
        Plan::from_source(entry::FK.as_bytes(), "fk", Target::Cuda, entry::limits()).unwrap();
    let mut args = entry::fk_args(2, false);
    let Value::ArrayF32(q) = &args[0] else {
        panic!()
    };
    args[2] = Value::ArrayF32(Rc::clone(q));
    refusal(&plan, &args, EntryRefusal::Alias);
    let observation = invar::general_net::oracle::observe_call(
        plan.net(),
        plan.proof(),
        "fk",
        &args,
        Default::default(),
    )
    .unwrap();
    assert!(matches!(
        observation.semantic.result,
        Err(invar::eval::EvalError::Tool {
            code: "HOST_ARRAY_ALIAS",
            ..
        })
    ));
    assert_eq!(observation.semantic.steps_used, 0);

    let mixed=Plan::from_source(b"fn run(a:read Array<f32>,b:read Array<i32>)->i32{return b[0u32];} fn main()->i32{return 0;}","run",Target::Cuda,entry::limits()).unwrap();
    let value = Rc::new(RefCell::new(vec![
        invar::numeric_contract::F32Bits::from_bits(1),
    ]));
    let bad = [
        Value::ArrayF32(Rc::clone(&value)),
        Value::ArrayF32(Rc::clone(&value)),
    ];
    let observed = invar::general_net::oracle::observe_call(
        mixed.net(),
        mixed.proof(),
        "run",
        &bad,
        Default::default(),
    )
    .unwrap();
    assert!(matches!(
        observed.semantic.result,
        Err(invar::eval::EvalError::Tool {
            code: "ENTRYPOINT_SIGNATURE",
            ..
        })
    ));
    assert_eq!(observed.semantic.steps_used, 0);

    let borrowed = entry::fk_args(1, false);
    let Value::ArrayF32(q) = &borrowed[0] else {
        panic!()
    };
    let lease = q.borrow_mut();
    refusal(&plan, &borrowed, EntryRefusal::Borrowed);
    drop(lease);

    let mut args = entry::fk_args(2049, false);
    refusal(&plan, &args, EntryRefusal::Oversize);
    args = entry::fk_args(2, false);
    args[3] = Value::Record(Rc::new(RefCell::new(vec![Value::I32(1)])));
    refusal(&plan, &args, EntryRefusal::WrongLayout);
    args[3] = Value::Record(Rc::new(RefCell::new(vec![
        Value::I32(1),
        Value::F32(invar::numeric_contract::F32Bits::POS_ZERO),
    ])));
    refusal(&plan, &args, EntryRefusal::WrongLayout);
    let plan = Plan::from_source(
        b"fn run(x:read Array<i32>)->u32{return len(x);} fn main()->i32{return 0;}",
        "run",
        Target::Cuda,
        entry::limits(),
    )
    .unwrap();
    let args = [Value::ArrayF32(Rc::new(RefCell::new(vec![])))];
    refusal(&plan, &args, EntryRefusal::UnsupportedType);
    let args = [Value::ViewI32(
        Rc::new(RefCell::new(vec![1, 2, 3])),
        invar::eval::ViewRange {
            start: 0,
            len: 2,
            stride: 2,
            mutable: false,
        },
    )];
    refusal(&plan, &args, EntryRefusal::WrongLayout);
    for source in [
        "struct P {x:Array<i32>} fn run(p:read P)->i32{return 0;} fn main()->i32{return 0;}",
        "fn run(x:take Array<f32>)->f32{return x[0u32];} fn main()->i32{return 0;}",
    ] {
        let error =
            Plan::from_source(source.as_bytes(), "run", Target::Cuda, entry::limits()).unwrap_err();
        assert_eq!(
            device_host::refusal_class(&error),
            "unsupported_type",
            "{source}: {error:?}"
        );
    }
}
#[test]
fn pending_or_cancelled_work_cannot_read_or_publish() {
    let observed = Cell::new(0);
    let mut gate = compiled::FencedExecution::new((), (), 2).unwrap();
    assert!(gate.observe(|_| observed.set(observed.get() + 1)).is_err());
    gate.submit(|_| compiled::Submission::Submitted).unwrap();
    assert_eq!(
        gate.poll(|_| compiled::FenceObservation::Pending),
        Err(compiled::FenceError::Pending)
    );
    assert!(gate.observe(|_| observed.set(observed.get() + 1)).is_err());
    gate.cancel();
    gate.poll(|_| compiled::FenceObservation::Complete)
        .unwrap_err();
    assert!(gate.observe(|_| observed.set(observed.get() + 1)).is_err());
    assert_eq!(observed.get(), 0);
}
#[test]
fn source_cap_and_checked_owned_bounds_cover_robotics_sizes() {
    assert_eq!(invar::array_limits::MAX_ARRAY_LEN, 1 << 24);
    assert_eq!(invar::array_limits::MAX_LIVE_ELEMENTS, 1 << 28);
    let checked = invar::check::check(
        invar::front::parse(
            b"fn main()->i32{let a=filled_i32(16777216u32,7);return a[16777215u32];}",
        )
        .unwrap(),
        true,
    )
    .unwrap();
    assert_eq!(
        invar::eval::observe(&checked, 100).result,
        Ok(Value::I32(7))
    );
    let mir = invar::mir::lower(&checked).unwrap();
    assert_eq!(
        invar::mir_eval::observe(&mir, 100).result,
        Ok(Value::I32(7))
    );
    let net = invar::general_net::Net::lower(&checked, Default::default()).unwrap();
    let observed = invar::general_net::oracle::observe(
        &net,
        &net.verify().unwrap(),
        invar::general_net::oracle::Limits {
            fuel: 100,
            array_elements: invar::array_limits::MAX_ARRAY_LEN,
            ..Default::default()
        },
    )
    .unwrap();
    assert_eq!(observed.semantic.result, Ok(Value::I32(7)));
    let checked = invar::check::check(
        invar::front::parse(b"fn main()->i32{let a=filled_i32(16777217u32,7);return 0;}").unwrap(),
        true,
    )
    .unwrap();
    assert!(invar::eval::observe(&checked, 100).result.is_err());
    let mut limits = entry::limits();
    limits.owned.elements = usize::MAX;
    assert!(Plan::from_source(entry::FK.as_bytes(), "fk", Target::Cuda, limits).is_err());
    limits.owned.elements = invar::array_limits::MAX_LIVE_ELEMENTS;
    limits.buffer_bytes = invar::array_limits::MAX_DEVICE_BYTES;
    let plan = Plan::from_source(entry::FK.as_bytes(), "fk", Target::Cuda, limits).unwrap();
    let args = entry::fk_args(1, false);
    let mut backend = CpuModel::with_compiler(std::path::Path::new("/nonexistent-p0b-compiler"));
    assert!(matches!(
        plan.prepare(&mut backend, &args, entry::config()),
        Err(device_host::Error::Limit)
    ));
}

#[test]
fn read_aliases_execute_with_a_pinned_shared_owner() {
    let plan=Plan::from_source(b"fn run(a:read Array<i32>,b:read Array<i32>)->i32{return a[0u32]+b[0u32];} fn main()->i32{return 0;}","run",Target::Cuda,entry::limits()).unwrap();
    let owner = Rc::new(RefCell::new(vec![7]));
    let args = [
        Value::ArrayI32(Rc::clone(&owner)),
        Value::ArrayI32(Rc::clone(&owner)),
    ];
    let mut backend = CpuModel::default();
    let mut session = plan.prepare(&mut backend, &args, entry::config()).unwrap();
    assert!(owner.try_borrow_mut().is_err());
    loop {
        session.submit().unwrap();
        match session.poll().unwrap() {
            device_host::Progress::Complete(o) => {
                assert_eq!(o.result, Some(Value::I32(14)));
                break;
            }
            device_host::Progress::Continuation { .. } => {}
            device_host::Progress::Pending => panic!("CPU group model submit is synchronous"),
        }
    }
    drop(session.reclaim().unwrap());
    assert!(owner.try_borrow_mut().is_ok());
}

#[test]
fn oracle_receipts_capture_input_before_mutation_and_freeze_flat_records() {
    use invar::general_net::{oracle, provenance};
    let source=b"struct P{x:f32} fn run(out:mut Array<f32>)->P{out[0u32]=2.0;return P{x:out[0u32]};} fn main()->i32{return 0;}";
    let artifact = provenance::compile_device_speculative(
        source,
        "run",
        Target::Metal,
        Default::default(),
        entry::limits().owned,
        entry::limits().private,
    )
    .unwrap();
    let cyclic = Rc::new(RefCell::new(Vec::new()));
    cyclic.borrow_mut().push(Value::Record(Rc::clone(&cyclic)));
    assert!(
        artifact
            .observe_oracle(&[Value::Record(Rc::clone(&cyclic))], Default::default())
            .is_err()
    );
    cyclic.borrow_mut().clear();
    let args = [entry::f(&[1])];
    let first = artifact
        .observe_oracle(&args, oracle::Limits::default())
        .unwrap();
    assert!(first.manifest()["input"].is_object());
    let input = first.manifest()["input"].clone();
    let second = artifact
        .observe_oracle(&args, oracle::Limits::default())
        .unwrap();
    assert_ne!(input, second.manifest()["input"]);
    let snapshot = first.observation();
    let Value::Record(fields) = snapshot.semantic.result.unwrap() else {
        panic!()
    };
    fields.borrow_mut()[0] = Value::I32(7);
    let Value::Record(fields) = first.observation().semantic.result.unwrap() else {
        panic!()
    };
    assert_eq!(
        fields.borrow()[0],
        Value::F32(invar::numeric_contract::F32Bits::from_bits(0x40000000))
    );
}

#[cfg(feature = "native")]
#[test]
fn native_integer_source_caps_match_reference() {
    let cap = invar::array_limits::MAX_ARRAY_LEN;
    let live = invar::array_limits::MAX_LIVE_ELEMENTS;
    let mut cases = Vec::new();
    for ty in ["i32", "u32"] {
        for length in [cap, cap + 1] {
            cases.push((
                format!("fn main()->i32{{let a=filled_{ty}({length}u32,7{ty});if a[{}u32]==7{ty}{{return 7;}}return 0;}}", length - 1),
                length == cap,
            ));
        }
    }
    for count in [live / cap, live / cap + 1] {
        let mut source = String::from("fn main()->i32{");
        for i in 0..count {
            source.push_str(&format!("let a{i}=filled_i32({cap}u32,7);"));
        }
        source.push_str(&format!("return a0[{}u32];}}", cap - 1));
        cases.push((source, count * cap == live));
    }
    for (index, (source, admitted)) in cases.iter().enumerate() {
        let checked =
            invar::check::check(invar::front::parse(source.as_bytes()).unwrap(), true).unwrap();
        let reference = invar::eval::observe(&checked, 1000);
        if *admitted {
            assert_eq!(reference.result, Ok(Value::I32(7)));
        } else {
            assert!(matches!(
                reference.result,
                Err(invar::eval::EvalError::Tool {
                    code: "ARRAY_SIZE_LIMIT",
                    ..
                })
            ));
        }
        let net = invar::general_net::Net::lower(&checked, Default::default()).unwrap();
        let program = net.compiled_mir(&net.verify().unwrap()).unwrap();
        let (status, document) = invar::native::run_program(&program, 1000).unwrap();
        assert_eq!(
            status,
            if *admitted { 0 } else { 2 },
            "case {index}: {document}"
        );
        if *admitted {
            assert_eq!(document["result"]["value"], 7, "case {index}");
        } else {
            assert_eq!(document["tool_code"], "ARRAY_SIZE_LIMIT", "case {index}");
        }
        assert_eq!(document["steps_used"], reference.steps_used, "case {index}");
        eprintln!("NATIVE_CAP_CASE {index} per_array={cap} live={live} status={status}");
    }
    assert_eq!(cases.len(), 6);
}
