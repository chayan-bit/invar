#![forbid(unsafe_code)]
//! ENG-02 (#59): one loop iteration re-derived from the net as law-formula
//! terms, the net side of a lifting certificate. Each derived step is checked
//! against a hand-written form with the in-repo prover.

use crate::support;

use invar::{
    general_net::{
        normal_form::lift::{LoopSteps, loop_steps},
        regions,
    },
    laws_synth::{
        formula::{Cond, Formula, Sort, Term},
        prove::{Verdict, decide},
    },
};

const KADANE: &str = "fn f(xs: read Array<i32>, n: u32) -> i32 { var best = 0 - 2147483647; var cur = 0; for i in 0u32 .. n { cur = cur + xs[i]; if cur < 0 { cur = 0; } if cur > best { best = cur; } } return best; }";
const WRAPPING: &str = "fn f(xs: read Array<u32>, n: u32) -> u32 { var s = 7u32; for i in 0u32 .. n { s = wrapping_add(wrapping_mul(s, 3u32), xs[i]); } return s; }";
const GUARDED: &str = "fn f(xs: read Array<i32>, n: u32) -> i32 { var c = 0; var last = 0; for i in 0u32 .. n { let x = xs[i]; if x > 0 and c < 10 { c = c + 1; } else { last = x; } } return c + last; }";

fn steps(source: &str) -> Result<LoopSteps, String> {
    let (net, proof) = support::compile(source);
    let partition = regions::partition(&net, &proof).expect("partition");
    loop_steps(&partition, "f", 0)
}

fn v(name: &str) -> Term {
    Term::var(name)
}

fn proved(sort: Sort, derived: &Term, expected: Term) {
    let formula = Formula::new(sort, vec![(derived.clone(), expected)]);
    assert_eq!(decide(&formula), Verdict::Proved, "{derived:?}");
}

#[test]
fn issue_eng_02_lift_kadane_steps_are_max_forms() {
    let lifted = steps(KADANE).expect("kadane lifts");
    assert_eq!(lifted.sort, Sort::Int);
    assert_eq!(lifted.induction, "i");
    assert_eq!(Vec::from_iter(lifted.steps.keys()), ["best", "cur"]);
    assert_eq!(Vec::from_iter(&lifted.inputs), ["xs[i]"]);
    let cur = (v("cur") + v("xs[i]")).max(Term::Lit(0));
    proved(Sort::Int, &lifted.steps["cur"], cur.clone());
    proved(Sort::Int, &lifted.steps["best"], v("best").max(cur));
    // The step is not the unclamped sum.
    let formula = Formula::new(
        Sort::Int,
        vec![(lifted.steps["cur"].clone(), v("cur") + v("xs[i]"))],
    );
    assert!(matches!(decide(&formula), Verdict::Unproved(_)));
}

#[test]
fn issue_eng_02_lift_wrapping_and_guarded_steps() {
    let wrapping = steps(WRAPPING).expect("wrapping lifts");
    assert_eq!(wrapping.sort, Sort::Bv32);
    proved(
        Sort::Bv32,
        &wrapping.steps["s"],
        Term::Lit(3) * v("s") + v("xs[i]"),
    );
    let guarded = steps(GUARDED).expect("guarded lifts");
    assert_eq!(Vec::from_iter(guarded.steps.keys()), ["c", "last"]);
    let x = || v("xs[i]");
    let both = |yes: Term, no: Term| {
        let inner = Term::ite(Cond::Lt(v("c"), Term::Lit(10)), yes, no.clone());
        Term::ite(Cond::Lt(Term::Lit(0), x()), inner, no)
    };
    proved(
        Sort::Int,
        &guarded.steps["c"],
        both(v("c") + Term::Lit(1), v("c")),
    );
    proved(Sort::Int, &guarded.steps["last"], both(v("last"), x()));
}

#[test]
fn issue_eng_02_lift_refuses_outside_the_fragment() {
    let refused = [
        "fn f(xs: read Array<i32>, n: u32) -> i32 { var s = 0; for i in 0u32 .. n { s = s + xs[i + 1u32]; } return s; }",
        "fn f(xs: read Array<i32>, n: u32) -> i32 { var s = 1; for i in 0u32 .. n { s = s / xs[i]; } return s; }",
        "fn f(xs: mut Array<i32>, n: u32) -> i32 { var s = 0; for i in 0u32 .. n { s = s + xs[i]; xs[i] = s; } return s; }",
        "fn f(xs: read Array<u32>, n: u32) -> u32 { var s = 0u32; for i in 0u32 .. n { s = wrapping_add(s, xs[i]) + 1u32; } return s; }",
        "fn f(xs: read Array<i32>, n: u32) -> i32 { var s = 0; var i = 0u32; while i < n { s = s + xs[i]; i = i + 1u32; } return s; }",
        // A wrapping loop that orders an i32: Z/2^32 comparisons are unsigned.
        "fn f(xs: read Array<u32>, n: u32, k: i32) -> u32 { var s = 0u32; for i in 0u32 .. n { if k < 0 { s = wrapping_add(s, xs[i]); } } return s; }",
        // Float guards: integer comparisons cannot model float order or NaN.
        "fn f(xs: read Array<f32>, n: u32, m: f32) -> u32 { var at = 0u32; for i in 0u32 .. n { if xs[i] < m { at = i; } } return at; }",
        "fn f(n: u32, x: f32, y: f32) -> u32 { var s = 0u32; for i in 0u32 .. n { if x < y { s = s + 1u32; } } return s; }",
        // A guard no merge needs is still checked: this one reads away from i.
        "fn f(xs: read Array<i32>, n: u32) -> i32 { var s = 0; for i in 0u32 .. n { if xs[i + 1u32] > 0 { s = s; } s = s + xs[i]; } return s; }",
    ];
    for source in refused {
        assert!(steps(source).is_err(), "{source}");
    }
    let (net, proof) = support::compile(KADANE);
    let partition = regions::partition(&net, &proof).expect("partition");
    assert!(loop_steps(&partition, "f", 1).is_err());
    assert!(loop_steps(&partition, "g", 0).is_err());
}
