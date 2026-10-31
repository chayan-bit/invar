//! Issue label GNET-08.
#![forbid(unsafe_code)]

use invar::{
    check,
    eval::{self, Value},
    front,
    general_net::{self, Net, oracle},
    numeric_contract::F32Bits,
};
fn checked(source: &str) -> check::CheckedProgram {
    let c = check::check(front::parse(source.as_bytes()).unwrap(), false).unwrap();
    invar::ownership::check(&c).unwrap();
    c
}
fn equal(source: &eval::Observation, net: &eval::Observation) {
    match (&source.result, &net.result) {
        (Ok(Value::F32(a)), Ok(Value::F32(b))) => assert!(a.strict_eq(*b)),
        _ => assert_eq!(source.result, net.result),
    }
    assert_eq!(source.steps_used, net.steps_used);
    assert_eq!(source.cleanup_trace, net.cleanup_trace);
}
#[test]
fn private_numeric_named_source_preserves_bit_copies_and_keeps_export_signature_frozen() {
    let c = checked(
        "fn echo(x:f32)->f32{return x;} fn pred(x:f32)->Bool{return x==x;} fn unit()->Unit{return;}",
    );
    for bits in [
        0, 0x80000000, 1, 0x007fffff, 0x7f800000, 0xff800000, 0x7fa00123, 0xffc00123,
    ] {
        let args = [Value::F32(F32Bits::from_bits(bits))];
        let o = eval::named::observe_numeric_named(&c, "echo", &args, 2);
        let Ok(Value::F32(x)) = o.result else {
            panic!("{o:?}")
        };
        assert_eq!(x.to_bits(), bits);
        assert_eq!(o.steps_used, 2);
        let old = eval::named::observe_named(&c, "echo", &args, 100);
        assert!(matches!(
            old.result,
            Err(eval::EvalError::Tool {
                code: "NATIVE_ABI",
                ..
            })
        ));
        assert_eq!(old.steps_used, 0);
        let p = eval::named::observe_numeric_named(&c, "pred", &args, 100);
        assert_eq!(
            p.result,
            Ok(Value::Bool(!F32Bits::from_bits(bits).is_nan()))
        );
    }
    assert_eq!(
        eval::named::observe_numeric_named(&c, "unit", &[], 100).result,
        Ok(Value::Unit)
    );
}
#[test]
fn private_numeric_named_source_matches_every_net_fuel_prefix_and_original_trap_span() {
    for source in [
        "fn run(x:f32,n:u32)->f32{var y=x;for i in 0u32..n{y=fma(y,1.0,0.0);}return y;}",
        "fn run(x:f32,n:u32)->f32{if n==0u32{return -x;}return run(x,n-1u32);}",
        "fn run(x:f32,n:u32)->f32{let earlier=1/0;return x/f32_from_u32(n);}",
        "fn run(x:f32,n:u32)->f32{let earlier=2147483647+1;return x/f32_from_u32(n);}",
    ] {
        let c = checked(source);
        let net = Net::lower(&c, general_net::Limits::default()).unwrap();
        let proof = net.verify().unwrap();
        let args = [Value::F32(F32Bits::from_bits(0x80000000)), Value::U32(3)];
        let done = eval::named::observe_numeric_named(&c, "run", &args, 1000);
        for fuel in 1..=done.steps_used + 1 {
            let s = eval::named::observe_numeric_named(&c, "run", &args, fuel);
            let o = oracle::observe_call(
                &net,
                &proof,
                "run",
                &args,
                oracle::Limits {
                    fuel,
                    ..oracle::Limits::default()
                },
            )
            .unwrap();
            equal(&s, &o.semantic);
        }
    }
}
#[test]
fn private_numeric_named_source_refuses_wrong_types_owners_and_zero_fuel_before_work() {
    let c = checked(
        "fn echo(x:f32)->f32{return x;} fn owned(x:read Array<f32>)->f32{return x[0u32];} fn nine(a:f32,b:f32,c:f32,d:f32,e:f32,f:f32,g:f32,h:f32,i:f32)->f32{return a;}",
    );
    for (name, args, fuel, code) in [
        ("echo", vec![Value::I32(0)], 100, "NATIVE_ABI"),
        ("echo", vec![], 100, "NATIVE_ABI"),
        ("missing", vec![], 100, "NATIVE_ABI"),
        ("owned", vec![], 100, "NATIVE_ABI"),
        (
            "nine",
            (0..9).map(|_| Value::F32(F32Bits::POS_ZERO)).collect(),
            100,
            "NATIVE_ABI",
        ),
        (
            "echo",
            vec![Value::F32(F32Bits::POS_ZERO)],
            0,
            "INVALID_FUEL",
        ),
    ] {
        let o = eval::named::observe_numeric_named(&c, name, &args, fuel);
        assert!(matches!(o.result,Err(eval::EvalError::Tool{code:actual,..}) if actual==code));
        assert_eq!(o.steps_used, 0);
        assert!(o.cleanup_trace.is_empty());
    }
}
