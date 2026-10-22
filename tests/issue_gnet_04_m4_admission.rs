//! Integration-only receipt fixture: requires main's M4 source grammar.
use invar::{
    check, front,
    general_net::{Limits, Net, compiled, oracle},
};

#[test]
fn issue_gnet_04_m4_numeric_forms_are_checked_but_never_reinterpreted_as_integer() {
    for source in [
        "fn f(x: f32) -> f32 { return x+1.0; } fn main() -> i32 { return 1; }",
        "fn f() -> Array<f32> { return filled_f32(4u32,1.0); } fn main() -> i32 { return 1; }",
    ] {
        let checked = check::check(front::parse(source.as_bytes()).unwrap(), true).unwrap();
        let net = Net::lower(&checked, Limits::default()).unwrap();
        let proof = net.verify().unwrap();
        let observation = oracle::observe(&net, &proof, oracle::Limits::default()).unwrap();
        // The scalar/array oracle now observes these exact source types. The
        // legacy speculative emitter still rejects both whole candidates.
        assert_eq!(observation.failure, None);
        assert_eq!(observation.semantic.result, Ok(invar::eval::Value::I32(1)));
        assert!(!observation.events.is_empty());
        assert!(matches!(
            compiled::emit_speculative(
                &net,
                &proof,
                "main",
                compiled::Target::Cpu,
                compiled::Limits::default(),
                compiled::OwnedLimits::default(),
                compiled::SpeculationLimits::default()
            ),
            Err(compiled::Error::UnsupportedType)
        ));
    }
    let source = b"struct Numeric {x:f32} fn main()->i32{return 1;}";
    let checked = check::check(front::parse(source).unwrap(), true).unwrap();
    let net = Net::lower(&checked, Limits::default()).unwrap();
    let observation =
        oracle::observe(&net, &net.verify().unwrap(), oracle::Limits::default()).unwrap();
    // GNET-02 closed the nominal f32 member oracle gap; the source result is
    // observed exactly, never reinterpreted as integer.
    assert_eq!(observation.failure, None);
    assert_eq!(observation.semantic.result, Ok(invar::eval::Value::I32(1)));
    assert!(!observation.events.is_empty());
}
