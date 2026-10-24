#![forbid(unsafe_code)]
//! The Lemma B residue ring (L4/L5 `s = (a s + b) % P` on checked
//! integers) must agree with the reference on seeds outside `[0, P)` and on
//! coefficients whose product overflows, where `a s + b` itself traps.

use crate::support;

use invar::{
    eval::{self, EvalError, TrapCategory},
    general_net::{
        normal_form::{
            self, LawId,
            exec::{self, Mutation},
        },
        oracle,
    },
};
use std::{cell::RefCell, rc::Rc};

const L4_RESIDUE: &str = "fn f(xs: read Array<i32>, ys: read Array<i32>, n: u32, s0: i32) -> i32 { var s = s0; for i in 0u32 .. n { s = (s * xs[i] + ys[i]) % 10007; } return s; }";
const L4_GUARDED: &str = "fn f(xs: read Array<i32>, ys: read Array<i32>, n: u32, s0: i32) -> i32 { var s = s0; for i in 0u32 .. n { if xs[i] > 100 { s = (s * xs[i] + ys[i]) % 10007; } } return s; }";
const L5_RESIDUE: &str = "fn f(cs: read Array<i32>, n: u32, a0: i32, b0: i32) -> i32 { var a = a0; var b = b0; for i in 0u32 .. n { let na = (a * 2 + b * cs[i]) % 10007; let nb = (a + b) % 10007; a = na; b = nb; } return a * 3 + b; }";

fn ai32(v: &[i32]) -> eval::Value {
    eval::Value::ArrayI32(Rc::new(RefCell::new(v.to_vec())))
}

fn copy(v: &eval::Value) -> eval::Value {
    match v {
        eval::Value::ArrayI32(a) => {
            eval::Value::ArrayI32(Rc::new(RefCell::new(a.borrow().clone())))
        }
        eval::Value::I32(x) => eval::Value::I32(*x),
        eval::Value::U32(x) => eval::Value::U32(*x),
        _ => eval::Value::Unit,
    }
}

fn limits() -> oracle::Limits {
    oracle::Limits {
        fuel: 50_000_000,
        transitions: 500_000_000,
        trace_events: 1 << 24,
        array_elements: 1 << 20,
    }
}

/// Run `args` through the reference and the law executor; both must agree.
/// Returns the reference result and the executor's receipt.
fn agree(
    source: &str,
    law: LawId,
    args: &[eval::Value],
) -> (Result<eval::Value, EvalError>, exec::Receipt) {
    let (net, proof) = support::compile(source);
    let form = normal_form::extract(&net, &proof).expect("extract");
    assert!(
        form.certificates().any(|c| c.law == law),
        "{law:?} not certified: {form:#?}"
    );
    let oracle_args: Vec<eval::Value> = args.iter().map(copy).collect();
    let observed = oracle::observe_call(&net, &proof, "f", &oracle_args, limits()).expect("oracle");
    let ours = exec::run(&net, &proof, &form, "f", args, Mutation::None).expect("executor");
    assert_eq!(
        ours.result, observed.semantic.result,
        "{source} on {args:?}: receipt {:?}",
        ours.receipt
    );
    (observed.semantic.result, ours.receipt)
}

fn overflow(result: &Result<eval::Value, EvalError>) -> bool {
    matches!(
        result,
        Err(EvalError::Trap {
            category: TrapCategory::IntegerOverflow,
            ..
        })
    )
}

#[test]
fn l4_seed_above_the_modulus_traps_like_the_reference() {
    // The issue's reproducer: 1068844203 * 5 overflows i32 in lane 0.
    let args = [
        ai32(&[5]),
        ai32(&[14]),
        eval::Value::U32(1),
        eval::Value::I32(1_068_844_203),
    ];
    let (result, receipt) = agree(L4_RESIDUE, LawId::L4Affine, &args);
    assert!(overflow(&result), "{result:?}");
    assert!(
        receipt.guard_fallbacks.contains_key(&LawId::L4Affine),
        "{receipt:?}"
    );
}

#[test]
fn l4_seed_above_the_modulus_without_overflow_keeps_its_value() {
    for (seed, xs) in [(20_000, [1, 3]), (i32::MAX, [0, 7]), (10_007, [2, 2])] {
        let args = [
            ai32(&xs),
            ai32(&[4, 9]),
            eval::Value::U32(2),
            eval::Value::I32(seed),
        ];
        let (result, _) = agree(L4_RESIDUE, LawId::L4Affine, &args);
        assert!(result.is_ok(), "seed {seed}: {result:?}");
    }
}

#[test]
fn l4_untaken_update_returns_the_raw_seed() {
    // No lane updates `s`, so the source returns the seed itself, not its
    // residue.
    let args = [
        ai32(&[1, 2, 3]),
        ai32(&[0, 0, 0]),
        eval::Value::U32(3),
        eval::Value::I32(50_000),
    ];
    let (result, _) = agree(L4_GUARDED, LawId::L4Affine, &args);
    assert_eq!(result, Ok(eval::Value::I32(50_000)));
}

#[test]
fn l4_coefficient_overflow_below_the_modulus_traps() {
    // s < P, but s * xs[1] >= 2^31 in lane 1.
    let args = [
        ai32(&[3, 400_000, 1]),
        ai32(&[0, 0, 0]),
        eval::Value::U32(3),
        eval::Value::I32(9_000),
    ];
    let (result, receipt) = agree(L4_RESIDUE, LawId::L4Affine, &args);
    assert!(overflow(&result), "{result:?}");
    assert!(
        receipt.applied.contains_key(&LawId::L4Affine),
        "{receipt:?}"
    );
    let offset = [
        ai32(&[1, 1]),
        ai32(&[0, i32::MAX - 5]),
        eval::Value::U32(2),
        eval::Value::I32(9),
    ];
    assert!(overflow(&agree(L4_RESIDUE, LawId::L4Affine, &offset).0));
}

#[test]
fn l5_seeds_above_the_modulus_trap_like_the_reference() {
    // (a0, b0, cs, whether the reference traps on overflow)
    let cases = [
        (1 << 30, 3, [1, 1], true),
        (3, 1 << 29, [5, 1], true),
        (20_000, 30_000, [0, 0], false),
        (5, 7, [1, 200_000_000], true),
    ];
    for (a0, b0, cs, traps) in cases {
        let args = [
            ai32(&cs),
            eval::Value::U32(2),
            eval::Value::I32(a0),
            eval::Value::I32(b0),
        ];
        let (result, _) = agree(L5_RESIDUE, LawId::L5Matrix, &args);
        assert_eq!(overflow(&result), traps, "{a0} {b0} {cs:?}: {result:?}");
    }
}

#[test]
fn reduced_seeds_stay_on_the_law_path() {
    let args = [
        ai32(&[5, 7, 9]),
        ai32(&[14, 1, 2]),
        eval::Value::U32(3),
        eval::Value::I32(10_006),
    ];
    let (result, receipt) = agree(L4_RESIDUE, LawId::L4Affine, &args);
    assert!(result.is_ok(), "{result:?}");
    assert!(
        receipt.applied.contains_key(&LawId::L4Affine),
        "{receipt:?}"
    );
    assert!(receipt.guard_fallbacks.is_empty(), "{receipt:?}");
    let args = [
        ai32(&[2, 3]),
        eval::Value::U32(2),
        eval::Value::I32(0),
        eval::Value::I32(10_006),
    ];
    let (_, receipt) = agree(L5_RESIDUE, LawId::L5Matrix, &args);
    assert!(
        receipt.applied.contains_key(&LawId::L5Matrix),
        "{receipt:?}"
    );
    assert!(receipt.guard_fallbacks.is_empty(), "{receipt:?}");
}
