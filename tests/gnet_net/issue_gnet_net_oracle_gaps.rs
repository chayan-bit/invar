//! Exact M5-02 corpus gaps, qualified against GNET-01/02 rather than its older
//! main base. Keep source bytes and hand-derived results tied to that receipt.
#![forbid(unsafe_code)]

use invar::{
    check,
    eval::{self, EvalError, TrapCategory, Value},
    front::{self, Span},
    general_net::{Limits, Net, oracle},
};

const VIEWS: &str = "fn main()->i32 { var a: Array<i32> = [1,2,3,4]; if true { let v=view_mut(a,0u32,2u32); v[1u32]=7; } let sample=view(a,1u32,4u32,2u32); return sample[0u32]+sample[1u32]; }";
const INEXACT: &str = include_str!("../fixtures/r1_01/inexact-conversion.inv");
const OVERFLOW: &str = include_str!("../fixtures/r1_01/overflow-conversion.inv");

fn checked(source: &str) -> check::CheckedProgram {
    check::check(front::parse(source.as_bytes()).unwrap(), true).unwrap()
}

fn agrees(source: &str, expected: Result<Value, EvalError>) -> eval::Observation {
    let checked = checked(source);
    let net = Net::lower(&checked, Limits::default()).unwrap();
    let proof = net.verify().unwrap();
    let complete = eval::observe(&checked, 100_000);
    assert_eq!(complete.result, expected, "{source}");
    // Every fuel prefix checks failure order and forbids premature cleanup.
    for fuel in 0..=complete.steps_used {
        let reference = eval::observe(&checked, fuel);
        let actual = oracle::observe(
            &net,
            &proof,
            oracle::Limits {
                fuel,
                ..oracle::Limits::default()
            },
        )
        .unwrap();
        assert_eq!(actual.failure, None, "fuel={fuel}: {source}");
        assert_eq!(
            actual.semantic.result, reference.result,
            "fuel={fuel}: {source}"
        );
        assert_eq!(actual.semantic.steps_used, reference.steps_used);
        assert_eq!(actual.semantic.cleanup_trace, reference.cleanup_trace);
    }
    complete
}

fn trap(source: &str, operation: &str, category: TrapCategory) -> Result<Value, EvalError> {
    let start = source.find(operation).unwrap();
    Err(EvalError::Trap {
        category,
        span: Span {
            start,
            end: start + operation.len(),
        },
    })
}

#[test]
fn issue_gnet_net_oracle_gaps_m502_views() {
    let observed = agrees(VIEWS, Ok(Value::I32(11)));
    assert_eq!(observed.steps_used, 30);
    assert_eq!(
        observed.cleanup_trace,
        vec![eval::CleanupEvent {
            name: "a".into(),
            length: 4
        }]
    );
    println!("M5-02 views: value=11 steps=30 cleanup=a[4], all fuel prefixes agree");
}

#[test]
fn issue_gnet_net_oracle_gaps_m502_unit_inexact() {
    let observed = agrees(
        INEXACT,
        trap(INEXACT, "to::<m>(mm)", TrapCategory::UnitConversionInexact),
    );
    assert_eq!(observed.steps_used, 5);
    assert!(observed.cleanup_trace.is_empty());
    println!("M5-02 unit-inexact: UNIT_CONVERSION_INEXACT span=108..119 steps=5");
}

#[test]
fn issue_gnet_net_oracle_gaps_m502_unit_overflow() {
    let observed = agrees(
        OVERFLOW,
        trap(OVERFLOW, "to::<mm>(metres)", TrapCategory::IntegerOverflow),
    );
    assert_eq!(observed.steps_used, 5);
    assert!(observed.cleanup_trace.is_empty());
    println!("M5-02 unit-overflow: INTEGER_OVERFLOW span=116..132 steps=5");
}

#[test]
fn issue_gnet_net_oracle_gaps_view_bounds_reborrows_and_return_lifetime() {
    agrees(
        include_str!("../fixtures/l1_02_views.inv"),
        Ok(Value::I32(65)),
    );
    agrees(
        "fn tail(xs: mut Array<i32>) -> MutView<i32> from xs { return view_mut(xs,1u32,3u32); } fn relay(xs: mut Array<i32>) -> MutView<i32> from xs { return tail(xs); } fn main()->i32 { var a: Array<i32> = [1,2,3]; if true { let v=relay(a); v[1u32]=9; } return a[2u32]; }",
        Ok(Value::I32(9)),
    );
    agrees(
        "fn main()->i32 { let a: Array<i32> = [1,2,3]; let x=view(a,0u32,3u32); let y=view(a,1u32,3u32); return x[1u32]+y[0u32]; }",
        Ok(Value::I32(4)),
    );
    for (source, operation) in [
        (
            "fn main()->i32 { let a: Array<i32> = [1,2]; let v=view(a,0u32,3u32); return 0; }",
            "view(a,0u32,3u32)",
        ),
        (
            "fn main()->i32 { let a: Array<i32> = [1,2]; let v=view(a,0u32,2u32,0u32); return 0; }",
            "view(a,0u32,2u32,0u32)",
        ),
        (
            "fn main()->i32 { let a: Array<i32> = [1,2,3]; let v=view(a,0u32,3u32,2u32); return v[2u32]; }",
            "v[2u32]",
        ),
        (
            "fn main()->i32 { let a: Array<i32> = [1]; let v=view(a,0u32,1u32,4294967295u32); let w=view(v,0u32,1u32,2u32); return 0; }",
            "view(v,0u32,1u32,2u32)",
        ),
    ] {
        agrees(
            source,
            trap(source, operation, TrapCategory::IndexOutOfBounds),
        );
    }
}

#[test]
fn issue_gnet_net_oracle_gaps_exact_nominal_scales() {
    for source in [
        "fn main()->i32 { let x: Quantity<i32,m> = quantity::<m>(-7); let y: Quantity<i32,mm> = to::<mm>(x); if y == quantity::<mm>(-7000) { return 7; } return 0; }",
        "fn main()->i32 { let x: Quantity<u32,mm> = quantity::<mm>(4294967000u32); let y: Quantity<u32,m> = to::<m>(x); if y == quantity::<m>(4294967u32) { return 7; } return 0; }",
        "fn main()->i32 { let x: Quantity<i32,mm> = quantity::<mm>(-7000); let y: Quantity<i32,m> = to::<m>(x); if y == quantity::<m>(-7) { return 7; } return 0; }",
    ] {
        agrees(source, Ok(Value::I32(7)));
    }
}

#[test]
fn issue_gnet_net_oracle_gaps_alias_lifetime_and_unit_refusals() {
    for (source, code) in [
        (
            include_str!("../fixtures/l1_02/overlap.inv"),
            "OWN_BORROW_CONFLICT",
        ),
        (
            include_str!("../fixtures/l1_02/owner-move.inv"),
            "OWN_BORROW_CONFLICT",
        ),
        (
            include_str!("../fixtures/l1_02/invalid-return.inv"),
            "OWN_RETURN_BORROW",
        ),
        (
            include_str!("../fixtures/l1_02/temporary-owner.inv"),
            "OWN_BORROW_SOURCE",
        ),
        (
            "fn main()->i32 { let x: Quantity<i32,m> = quantity::<m>(1); let y: Quantity<i32,s> = to::<s>(x); return 0; }",
            "UNIT_MISMATCH",
        ),
    ] {
        let error = check::check(front::parse(source.as_bytes()).unwrap(), true).unwrap_err();
        assert_eq!(error.code, code, "{source}");
        assert!(error.span.is_some());
    }
}
