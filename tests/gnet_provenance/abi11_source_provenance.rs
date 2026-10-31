//! Independent source observation and source-issued ABI11 receipt checks.
//! Issue label GNET-11.

use invar::{
    check, eval, front,
    general_net::{self, Net, compiled, oracle, provenance},
    numeric_contract::F32Bits,
};
use std::rc::Rc;
const SOURCE: &str = "fn run(x:f32)->Array<f32>{let a:Array<f32> = [x,-0.0];return a;}";
/// Direct i32/u32/f32 owner roots, empty roots, internal owner cleanup beside
/// a returned owner, and index/overflow traps inside owner-producing calls.
const OWNERS: &str = "fn ints(n:u32,x:i32)->Array<i32> {let tmp=filled_u32(2u32,9u32);var a=filled_i32(n,x);if n>0u32 {a[0u32]=a[0u32]+1;}return a;} \
fn words(n:u32)->Array<u32> {var a=filled_u32(n,0u32);for i in 0u32..n {a[i]=i*3u32;}return a;} \
fn floats(x:f32,n:u32)->Array<f32> {let scratch:Array<f32> = [x,-0.0];var a=filled_f32(n,x);if n>1u32 {a[1u32]=fma(scratch[0u32],x,scratch[1u32]);}return a;} \
fn pick(n:u32,i:u32)->Array<f32> {let a=filled_f32(n,1.0);let b=floats(a[i],n);return b;} \
fn see(xs:read Array<f32>)->f32 {return xs[0u32];} \
fn own(xs:take Array<i32>)->Array<i32> {return xs;} \
fn main()->i32 {return 7;}";

fn checked(source: &str) -> check::CheckedProgram {
    let c = check::check(front::parse(source.as_bytes()).unwrap(), false).unwrap();
    invar::ownership::check(&c).unwrap();
    c
}
/// Bit-exact carrier for results; floating values compare by encoding.
fn carrier(r: &Result<eval::Value, eval::EvalError>) -> String {
    match r {
        Ok(eval::Value::ArrayI32(v)) => format!("i32{:?}", v.borrow()),
        Ok(eval::Value::ArrayU32(v)) => format!("u32{:?}", v.borrow()),
        Ok(eval::Value::ArrayF32(v)) => format!(
            "f32{:?}",
            v.borrow().iter().map(|x| x.to_bits()).collect::<Vec<_>>()
        ),
        Ok(x) => panic!("unexpected scalar root {x:?}"),
        Err(e) => format!("{e:?}"),
    }
}
fn code(o: &eval::Observation) -> &'static str {
    match &o.result {
        Err(eval::EvalError::Tool { code, .. }) => code,
        x => panic!("expected a tool refusal, got {x:?}"),
    }
}
fn f(bits: u32) -> eval::Value {
    eval::Value::F32(F32Bits::from_bits(bits))
}

#[test]
fn source_owned_roots_are_fresh_and_scalar_admission_is_unchanged() {
    let checked = check::check(front::parse(SOURCE.as_bytes()).unwrap(), false).unwrap();
    for bits in [0, 1, 0x8000_0000, 0x7f80_0000, 0x7fa0_0001, 0xffc1_2345] {
        let args = [eval::Value::F32(F32Bits::from_bits(bits))];
        let a = eval::named::observe_owned_numeric_named(&checked, "run", &args, 1000);
        let b = eval::named::observe_owned_numeric_named(&checked, "run", &args, 1000);
        assert_eq!(a.steps_used, b.steps_used);
        assert_eq!(a.cleanup_trace, b.cleanup_trace);
        let (Ok(eval::Value::ArrayF32(a)), Ok(eval::Value::ArrayF32(b))) = (a.result, b.result)
        else {
            panic!("expected array roots")
        };
        assert!(!Rc::ptr_eq(&a, &b));
        assert_eq!(a.borrow()[0].to_bits(), bits);
        assert_eq!(a.borrow()[1].to_bits(), 0x8000_0000);
        a.borrow_mut()[0] = F32Bits::from_bits(0x3f80_0000);
        assert_eq!(b.borrow()[0].to_bits(), bits);
        let old = eval::named::observe_numeric_named(&checked, "run", &args, 1000);
        assert!(old.result.is_err());
        assert_eq!(old.steps_used, 0);
    }
    let wrong =
        eval::named::observe_owned_numeric_named(&checked, "run", &[eval::Value::I32(1)], 1000);
    assert!(wrong.result.is_err());
    assert_eq!(wrong.steps_used, 0);
}

#[test]
fn abi11_receipt_binds_owners_and_continuation() {
    let storage = compiled::OwnedLimits {
        elements: 16,
        allocations: 8,
        cleanup_events: 8,
    };
    let a = provenance::compile_resumable_owned_numeric(
        SOURCE.as_bytes(),
        "run",
        compiled::Target::Cpu,
        Default::default(),
        storage,
    )
    .unwrap();
    assert_eq!(a.artifact().abi_version(), 11);
    assert_eq!(a.manifest()["numeric"]["array_f32"]["kind"], 258);
    let c = &a.manifest()["numeric_owned_continuation"];
    assert_eq!(c["profile"], "cpu-owned-numeric-continuation-v1");
    let fields = c["task_fields"].as_array().unwrap();
    assert_eq!(fields.len(), 18);
    for (i, field) in fields.iter().enumerate() {
        assert_eq!(field["offset"], a.artifact().frame_bytes() + i * 8);
        assert_eq!(field["representation"], if i < 14 { "u64" } else { "i64" });
    }
    let mut tampered = a.manifest().clone();
    tampered["numeric_owned_continuation"]["task_fields"][8]["offset"] = 0.into();
    assert!(a.verify_manifest(&tampered).is_err());
    // ABI12 stage 2b (parent-approved minimal expectation update): CUDA now
    // yields a distinct ABI12 compile receipt only, never this ABI11 profile.
    let cuda = provenance::compile_resumable_owned_numeric(
        SOURCE.as_bytes(),
        "run",
        compiled::Target::Cuda,
        Default::default(),
        storage,
    )
    .unwrap();
    assert_eq!(cuda.artifact().abi_version(), 12);
    assert!(cuda.manifest().get("numeric_owned_continuation").is_none());
    assert_eq!(
        cuda.manifest()["cuda_owned_numeric_continuation"]["authority"],
        serde_json::json!({"execution": false, "module_load": false, "resume": false,
            "device_qualification": false})
    );
}

#[test]
fn owner_roots_every_fuel_prefix_match_independent_graph_oracle() {
    let c = checked(OWNERS);
    let net = Net::lower(&c, general_net::Limits::default()).unwrap();
    let proof = net.verify().unwrap();
    let u = eval::Value::U32;
    let i = eval::Value::I32;
    let cases: Vec<(&str, Vec<eval::Value>, &str)> = vec![
        ("ints", vec![u(3), i(-5)], "i32[-4, -5, -5]"),
        ("ints", vec![u(0), i(7)], "i32[]"),
        ("words", vec![u(4)], "u32[0, 3, 6, 9]"),
        ("words", vec![u(0)], "u32[]"),
        ("floats", vec![f(0x7fa0_0001), u(1)], "f32[2141192193]"),
        ("floats", vec![f(0x8000_0000), u(0)], "f32[]"),
        (
            "floats",
            vec![f(0x3fc0_0000), u(3)],
            "f32[1069547520, 1074790400, 1069547520]",
        ),
        ("floats", vec![f(0x8000_0000), u(2)], "f32[2147483648, 0]"),
        ("pick", vec![u(2), u(1)], "f32[1065353216, 1065353216]"),
        ("pick", vec![u(2), u(2)], "Trap"),
        ("ints", vec![u(1), i(i32::MAX)], "Trap"),
    ];
    let mut prefixes = 0;
    for (entry, args, expected) in &cases {
        let complete = eval::named::observe_owned_numeric_named(&c, entry, args, 100_000);
        let shown = carrier(&complete.result);
        assert!(shown.starts_with(expected), "{entry}{args:?}: {shown}");
        if shown.starts_with("Trap") {
            assert!(complete.steps_used > 0);
        } else {
            // Internal owners are cleaned before the returned owner escapes;
            // `words` creates only its returned owner.
            assert_eq!(
                complete.cleanup_trace.is_empty(),
                *entry == "words",
                "{entry}"
            );
        }
        for fuel in 1..=complete.steps_used + 1 {
            let source = eval::named::observe_owned_numeric_named(&c, entry, args, fuel);
            let graph = oracle::observe_call(
                &net,
                &proof,
                entry,
                args,
                oracle::Limits {
                    fuel,
                    transitions: 1_000_000,
                    trace_events: 100_000,
                    array_elements: 1_000,
                },
            )
            .unwrap();
            assert_eq!(graph.failure, None, "{entry} fuel {fuel}");
            assert_eq!(
                carrier(&source.result),
                carrier(&graph.semantic.result),
                "{entry}{args:?} fuel {fuel}"
            );
            assert_eq!(source.steps_used, graph.semantic.steps_used);
            assert_eq!(source.cleanup_trace, graph.semantic.cleanup_trace);
            // Strictly smaller fuel stops at the limit; enough fuel reproduces
            // the complete observation, including traps.
            if fuel < complete.steps_used {
                assert!(matches!(
                    source.result,
                    Err(eval::EvalError::EvaluationLimit { .. })
                ));
            } else {
                assert_eq!(carrier(&source.result), shown);
                assert_eq!(source.steps_used, complete.steps_used);
            }
            prefixes += 1;
        }
    }
    // Trap spans name the original operators.
    for (entry, args, text) in [
        ("pick", vec![u(2), u(2)], "a[i]"),
        ("ints", vec![u(1), i(i32::MAX)], "+"),
    ] {
        let o = eval::named::observe_owned_numeric_named(&c, entry, &args, 100_000);
        let Err(eval::EvalError::Trap { span, .. }) = o.result else {
            panic!("{entry}: {:?}", o.result)
        };
        assert_eq!(&OWNERS[span.start..span.end], text, "{entry}");
    }
    eprintln!("source owner observer: {prefixes} fuel prefixes equal to graph oracle");
}

#[test]
fn owner_observer_refusals_and_unchanged_scalar_observers() {
    let c = checked(OWNERS);
    let refuse = |o: eval::Observation, expected: &str| {
        assert_eq!(code(&o), expected);
        assert_eq!(o.steps_used, 0);
        assert!(o.cleanup_trace.is_empty());
    };
    let observe = eval::named::observe_owned_numeric_named;
    let u = eval::Value::U32;
    // Exact argument types and arity.
    for args in [
        vec![u(1), u(1)],
        vec![eval::Value::I32(1), eval::Value::I32(1)],
        vec![f(0), eval::Value::I32(1)],
        vec![eval::Value::Bool(true), eval::Value::I32(1)],
        vec![u(1)],
        vec![u(1), eval::Value::I32(1), u(1)],
    ] {
        refuse(observe(&c, "ints", &args, 1000), "NATIVE_ABI");
    }
    refuse(observe(&c, "floats", &[u(1), f(0)], 1000), "NATIVE_ABI");
    refuse(observe(&c, "missing", &[], 1000), "NATIVE_ABI");
    refuse(observe(&c, "words", &[u(1)], 0), "INVALID_FUEL");
    // External array parameters stay outside the observer, in every mode.
    let owner = eval::Value::ArrayI32(Rc::new(std::cell::RefCell::new(vec![1])));
    let floats = eval::Value::ArrayF32(Rc::new(std::cell::RefCell::new(vec![F32Bits::from_bits(
        0,
    )])));
    refuse(observe(&c, "own", &[owner], 1000), "NATIVE_ABI");
    refuse(observe(&c, "see", &[floats], 1000), "NATIVE_ABI");
    // Both existing scalar observers still refuse array-return signatures.
    for (entry, args) in [
        ("ints", vec![u(1), eval::Value::I32(1)]),
        ("words", vec![u(1)]),
        ("floats", vec![f(0), u(1)]),
    ] {
        refuse(
            eval::named::observe_named(&c, entry, &args, 1000),
            "NATIVE_ABI",
        );
        refuse(
            eval::named::observe_numeric_named(&c, entry, &args, 1000),
            "NATIVE_ABI",
        );
        assert!(observe(&c, entry, &args, 1000).result.is_ok());
    }
    // Scalar entries remain observable by the owner observer and agree.
    let scalar = eval::named::observe_numeric_named(&c, "main", &[], 1000);
    let owned = observe(&c, "main", &[], 1000);
    assert_eq!(
        (format!("{:?}", scalar.result), scalar.steps_used),
        (format!("{:?}", owned.result), owned.steps_used)
    );
}

#[test]
fn owner_roots_are_independent_across_calls_and_mutation() {
    let c = checked(OWNERS);
    let args = [eval::Value::U32(3), eval::Value::I32(4)];
    let observe = || eval::named::observe_owned_numeric_named(&c, "ints", &args, 1000);
    let first = observe();
    let Ok(eval::Value::ArrayI32(a)) = &first.result else {
        panic!("{:?}", first.result)
    };
    a.borrow_mut().iter_mut().for_each(|x| *x = -1);
    a.borrow_mut().push(99);
    for _ in 0..3 {
        let next = observe();
        assert_eq!(next.steps_used, first.steps_used);
        assert_eq!(next.cleanup_trace, first.cleanup_trace);
        let Ok(eval::Value::ArrayI32(b)) = &next.result else {
            panic!("{:?}", next.result)
        };
        assert!(!Rc::ptr_eq(a, b));
        assert_eq!(*b.borrow(), vec![5, 4, 4]);
        b.borrow_mut().clear();
    }
    assert_eq!(*a.borrow(), vec![-1, -1, -1, 99]);
}
