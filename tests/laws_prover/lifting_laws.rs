#![forbid(unsafe_code)]
//! ENG-02 (#59): laws beyond the sealed v0.1 catalogue. L8 finite-state maps
//! and L5 matrix groups above four variables are certified, re-checked by the
//! independent checker, and executed by the law executor equal to the CPU net
//! oracle (value or first source-order trap); they stay serial with
//! `law_cost_uncalibrated` until the cost model is calibrated (#52).

use crate::support;

use invar::{
    eval::{self, EvalError},
    general_net::{
        Net, Proof,
        normal_form::{
            self, COST_UNCALIBRATED, Guard, LawId, Node, NormalForm, Rep, RepExpr, Subject,
            certificate,
            exec::{self, Mutation},
        },
        oracle,
    },
};
use std::{cell::RefCell, rc::Rc};

use crate::split_mix::Rng;

impl Rng {
    fn below(&mut self, n: u64) -> u64 {
        self.next() % n.max(1)
    }
    fn chance(&mut self, percent: u64) -> bool {
        self.below(100) < percent
    }
}

fn au32(v: Vec<u32>) -> eval::Value {
    eval::Value::ArrayU32(Rc::new(RefCell::new(v)))
}

fn copy(v: &eval::Value) -> eval::Value {
    match v {
        eval::Value::ArrayU32(a) => au32(a.borrow().clone()),
        eval::Value::ArrayI32(a) => ai32(a.borrow().clone()),
        eval::Value::U32(x) => eval::Value::U32(*x),
        eval::Value::I32(x) => eval::Value::I32(*x),
        _ => unreachable!("the tests pass u32 and i32 values and arrays"),
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

/// A 3-state machine over 2 symbols, with an accept counter that reads the
/// state's prefix (a successive scan).
const FSM3: &str = "fn f(t: read Array<u32>, xs: read Array<u32>, n: u32, s0: u32) -> u32 { var s = s0; var hits = 0u32; for i in 0u32 .. n { s = t[s * 2u32 + xs[i]]; if s == 2u32 { hits = hits + 1u32; } } return s * 1000u32 + hits; }";
/// A 16-state machine over 4 symbols, with a second guarded transition.
const FSM16: &str = "fn f(t: read Array<u32>, xs: read Array<u32>, n: u32, s0: u32) -> u32 { var s = s0; for i in 0u32 .. n { if xs[i] < 9u32 { s = t[s * 4u32 + xs[i] % 4u32]; } else { s = t[s * 4u32 + 3u32]; } } return s; }";

struct Fsm {
    source: &'static str,
    states: u32,
    symbols: u32,
}

const MACHINES: [Fsm; 2] = [
    Fsm {
        source: FSM3,
        states: 3,
        symbols: 2,
    },
    Fsm {
        source: FSM16,
        states: 16,
        symbols: 4,
    },
];

/// Arguments: mostly in range, sometimes a bad entry (guard), a bad symbol or
/// seed (index trap), or a short input array (read trap).
fn args(rng: &mut Rng, fsm: &Fsm) -> Vec<eval::Value> {
    let mut table: Vec<u32> = (0..fsm.states * fsm.symbols)
        .map(|_| rng.below(u64::from(fsm.states)) as u32)
        .collect();
    if rng.chance(10) {
        let at = rng.below(table.len() as u64) as usize;
        table[at] = fsm.states + rng.below(3) as u32;
    }
    let n = rng.below(40) as u32;
    let len = if rng.chance(10) && n > 0 {
        rng.below(u64::from(n)) as usize
    } else {
        n as usize
    };
    let xs: Vec<u32> = (0..len)
        .map(|_| {
            if rng.chance(3) {
                fsm.symbols + rng.below(20) as u32
            } else {
                rng.below(u64::from(fsm.symbols)) as u32
            }
        })
        .collect();
    let s0 = if rng.chance(5) {
        fsm.states + rng.below(2) as u32
    } else {
        rng.below(u64::from(fsm.states)) as u32
    };
    vec![
        au32(table),
        au32(xs),
        eval::Value::U32(n),
        eval::Value::U32(s0),
    ]
}

fn compiled(source: &str) -> (Net, Proof, NormalForm) {
    let (net, proof) = support::compile(source);
    let form = normal_form::extract_uncalibrated(&net, &proof).expect("extract");
    let checked = certificate::check_all(&net, &proof, &form);
    assert!(
        checked.is_ok(),
        "checker verdict {checked:?}; form {form:#?}"
    );
    (net, proof, form)
}

/// `(agrees, receipt, oracle trapped)`, or `None` on an oracle tool limit.
fn differential(
    net: &Net,
    proof: &Proof,
    form: &NormalForm,
    args: &[eval::Value],
    mutation: Mutation,
) -> Option<(bool, exec::Receipt, bool)> {
    let oracle_args: Vec<eval::Value> = args.iter().map(copy).collect();
    let observed = oracle::observe_call(net, proof, "f", &oracle_args, limits()).expect("oracle");
    if matches!(
        observed.semantic.result,
        Err(EvalError::Tool { .. }) | Err(EvalError::EvaluationLimit { .. })
    ) {
        return None;
    }
    let exec_args: Vec<eval::Value> = args.iter().map(copy).collect();
    let ours = exec::run(net, proof, form, "f", &exec_args, mutation).expect("executor");
    let trapped = matches!(observed.semantic.result, Err(EvalError::Trap { .. }));
    Some((
        ours.result == observed.semantic.result,
        ours.receipt,
        trapped,
    ))
}

fn fsm_rep(form: &NormalForm) -> Option<Rep> {
    form.certificates()
        .find(|c| c.law == LawId::L8FiniteState)
        .and_then(|c| c.rep.clone())
}

#[test]
fn l8_fsm_runs_match_the_net_oracle() {
    for (m, fsm) in MACHINES.iter().enumerate() {
        let (net, proof, form) = compiled(fsm.source);
        assert!(
            matches!(
                fsm_rep(&form),
                Some(Rep::Lifted(RepExpr::FunctionTable { stride, .. })) if stride == fsm.symbols
            ),
            "{form:#?}"
        );
        let mut rng = Rng(0x0e02_0000 + m as u64);
        let (mut cases, mut traps, mut law_path, mut guarded) = (0, 0, 0, 0);
        for case in 0..1500 {
            let args = args(&mut rng, fsm);
            let Some((same, receipt, trapped)) =
                differential(&net, &proof, &form, &args, Mutation::None)
            else {
                continue;
            };
            assert!(same, "{}-state case {case}: {args:?}", fsm.states);
            assert!(receipt.violations.is_empty(), "{:?}", receipt.violations);
            cases += 1;
            traps += u64::from(trapped);
            law_path += receipt
                .applied
                .get(&LawId::L8FiniteState)
                .copied()
                .unwrap_or(0);
            guarded += receipt
                .guard_fallbacks
                .get(&LawId::L8FiniteState)
                .copied()
                .unwrap_or(0);
        }
        eprintln!(
            "{}-state: cases={cases} traps={traps} law_path={law_path} guard_fallbacks={guarded}",
            fsm.states
        );
        assert!(cases >= 1400, "{cases}");
        assert!(traps > 50, "{traps}");
        assert!(law_path > 1000, "{law_path}");
        assert!(guarded > 50, "{guarded}");
    }
}

#[test]
fn l8_table_entry_out_of_range_runs_serial() {
    let (net, proof, form) = compiled(FSM3);
    // Entry 3 of a 3-state table: the guard fails, the loop runs in source
    // order, and the source reads t[1 * 2 + 1] = 3, then traps on t[3 * 2 + 1].
    let table = vec![1, 2, 0, 3, 2, 1];
    let args = vec![
        au32(table),
        au32(vec![1, 1, 0, 0]),
        eval::Value::U32(4),
        eval::Value::U32(1),
    ];
    let (same, receipt, trapped) =
        differential(&net, &proof, &form, &args, Mutation::None).expect("no tool limit");
    assert!(same && trapped);
    assert_eq!(receipt.guard_fallbacks.get(&LawId::L8FiniteState), Some(&1));
    assert_eq!(receipt.applied.get(&LawId::L8FiniteState), None);
}

#[test]
fn l8_wrong_combine_is_caught() {
    for (m, fsm) in MACHINES.iter().enumerate() {
        let (net, proof, form) = compiled(fsm.source);
        let mut rng = Rng(0xbad_0e02 + m as u64);
        let wrong = (0..400)
            .filter_map(|_| {
                let args = args(&mut rng, fsm);
                differential(&net, &proof, &form, &args, Mutation::WrongCombine)
            })
            .filter(|(same, _, _)| !same)
            .count();
        assert!(wrong > 0, "{}-state: wrong combine not caught", fsm.states);
    }
}

#[test]
fn l8_is_cost_gated_by_default() {
    for fsm in &MACHINES {
        let (net, proof) = support::compile(fsm.source);
        let form = normal_form::extract(&net, &proof).expect("extract");
        let loop_form = &form.functions[0].loops[0];
        assert_eq!(
            loop_form.node,
            Node::Serial {
                reason: COST_UNCALIBRATED.to_owned()
            }
        );
        assert!(loop_form.certificates.is_empty());
        assert_eq!(certificate::check_all(&net, &proof, &form), Ok(0));
    }
}

#[test]
fn l8_tables_that_vary_in_the_loop_stay_serial() {
    for (source, reason) in [
        (
            "fn f(t: mut Array<u32>, xs: read Array<u32>, n: u32) -> u32 { var s = 0u32; for i in 0u32 .. n { s = t[s * 2u32 + xs[i]]; t[i] = s; } return s; }",
            "",
        ),
        (
            "fn f(xs: read Array<u32>, n: u32) -> u32 { var s = 0u32; for i in 0u32 .. n { let t = filled_u32(4u32, 1u32); s = t[s * 2u32 + xs[i]]; } return s; }",
            "finite-state table declared in the loop",
        ),
    ] {
        let (net, proof) = support::compile(source);
        let form = normal_form::extract_uncalibrated(&net, &proof).expect("extract");
        let loop_form = &form.functions[0].loops[0];
        let Node::Serial { reason: got } = &loop_form.node else {
            panic!("table varies in the loop but L8 applied: {form:#?}");
        };
        assert!(got.contains(reason), "{got}");
    }
}

#[test]
fn l8_forged_certificates_are_rejected() {
    let (net, proof, form) = compiled(FSM16);
    let at = form.functions[0].loops[0]
        .certificates
        .iter()
        .position(|c| c.law == LawId::L8FiniteState)
        .expect("L8 certificate");
    let forge = |edit: &dyn Fn(&mut normal_form::Certificate)| {
        let mut forged = form.clone();
        edit(&mut forged.functions[0].loops[0].certificates[at]);
        certificate::check_all(&net, &proof, &forged)
    };
    let stride = |k: u32| {
        Rep::Lifted(RepExpr::FunctionTable {
            table: "t".to_owned(),
            stride: k,
        })
    };
    type Edit<'a> = &'a dyn Fn(&mut normal_form::Certificate);
    let rejected: [(&str, Edit<'_>); 5] = [
        ("stride", &|c| c.rep = Some(stride(2))),
        ("table", &|c| {
            c.rep = Some(Rep::Lifted(RepExpr::FunctionTable {
                table: "xs".to_owned(),
                stride: 4,
            }))
        }),
        ("no guard", &|c| c.guards.clear()),
        ("guard seed", &|c| {
            c.guards = vec![Guard::StateTable {
                table: "t".to_owned(),
                stride: 4,
                seed: "n".to_owned(),
            }]
        }),
        ("law", &|c| c.law = LawId::L4Affine),
    ];
    for (name, edit) in rejected {
        assert!(forge(edit).is_err(), "forged {name} accepted");
    }
    let subject = &form.functions[0].loops[0].certificates[at].subject;
    assert_eq!(subject, &Subject::Var("s".to_owned()));
}

// ----- large-k L5 ------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq)]
enum Ring {
    Wrapping,
    Checked,
    Residue,
}

/// A `k`-variable recurrence whose variables form one cycle
/// (`v_j' = v_j * cs[i] + v_{j+1}`), so they are one jointly linear group.
fn recurrence(ring: Ring, k: usize) -> String {
    let (ty, one) = match ring {
        Ring::Wrapping => ("u32", "u32"),
        _ => ("i32", ""),
    };
    let mut body = String::new();
    for j in 0..k {
        let next = (j + 1) % k;
        body += &match ring {
            Ring::Wrapping => {
                format!("let n{j} = wrapping_add(wrapping_mul(v{j}, cs[i]), v{next}); ")
            }
            Ring::Checked => format!("let n{j} = v{j} * cs[i] + v{next}; "),
            Ring::Residue => format!("let n{j} = (v{j} * cs[i] + v{next}) % 10007; "),
        };
    }
    for j in 0..k {
        body += &format!("v{j} = n{j}; ");
    }
    let seeds: String = (1..k).map(|j| format!("var v{j} = {j}{one}; ")).collect();
    let result = match ring {
        Ring::Wrapping => format!("wrapping_add(v0, v{})", k - 1),
        _ => format!("v0 + v{}", k - 1),
    };
    format!(
        "fn f(cs: read Array<{ty}>, n: u32, s0: {ty}) -> {ty} {{ var v0 = s0; {seeds}for i in 0u32 .. n {{ {body}}} return {result}; }}"
    )
}

fn recurrence_args(rng: &mut Rng, ring: Ring) -> Vec<eval::Value> {
    let n = rng.below(24) as u32;
    let len = if rng.chance(10) && n > 0 {
        rng.below(u64::from(n)) as usize
    } else {
        n as usize
    };
    match ring {
        Ring::Wrapping => {
            let cs = (0..len).map(|_| rng.next() as u32).collect();
            vec![
                au32(cs),
                eval::Value::U32(n),
                eval::Value::U32(rng.next() as u32),
            ]
        }
        _ => {
            // Small coefficients, sometimes negative (residue guard) or huge
            // (overflow), and seeds above the modulus.
            let cs: Vec<i32> = (0..len)
                .map(|_| match rng.below(20) {
                    0 => -(rng.below(5) as i32),
                    1 => 400_000 + rng.below(1000) as i32,
                    _ => rng.below(4) as i32,
                })
                .collect();
            let s0 = match rng.below(10) {
                0 => 1_068_844_203,
                1 => 10_007 + rng.below(100) as i32,
                _ => rng.below(10_007) as i32,
            };
            vec![ai32(cs), eval::Value::U32(n), eval::Value::I32(s0)]
        }
    }
}

fn ai32(v: Vec<i32>) -> eval::Value {
    eval::Value::ArrayI32(Rc::new(RefCell::new(v)))
}

#[test]
fn large_k_l5_matches_the_net_oracle() {
    for (r, (ring, k)) in [(Ring::Wrapping, 6), (Ring::Checked, 6), (Ring::Residue, 5)]
        .into_iter()
        .enumerate()
    {
        let source = recurrence(ring, k);
        let (net, proof, form) = compiled(&source);
        let vars = form
            .certificates()
            .find_map(|c| match &c.rep {
                Some(Rep::Matrix { vars, .. }) if c.law == LawId::L5Matrix => Some(vars.len()),
                _ => None,
            })
            .unwrap_or_else(|| panic!("{ring:?} k={k}: no L5 certificate: {form:#?}"));
        assert_eq!(vars, k);
        let mut rng = Rng(0x0e02_1000 + r as u64);
        let (mut cases, mut traps, mut law_path) = (0, 0, 0);
        for case in 0..800 {
            let args = recurrence_args(&mut rng, ring);
            let Some((same, receipt, trapped)) =
                differential(&net, &proof, &form, &args, Mutation::None)
            else {
                continue;
            };
            assert!(same, "{ring:?} k={k} case {case}: {args:?}");
            assert!(receipt.violations.is_empty(), "{:?}", receipt.violations);
            cases += 1;
            traps += u64::from(trapped);
            law_path += receipt.applied.get(&LawId::L5Matrix).copied().unwrap_or(0);
        }
        eprintln!("{ring:?} k={k}: cases={cases} traps={traps} law_path={law_path}");
        assert!(cases >= 750 && law_path > 300, "{cases} {law_path}");
        if ring != Ring::Wrapping {
            assert!(traps > 20, "{ring:?}: {traps} traps");
        }
        let wrong = (0..300)
            .filter_map(|_| {
                let args = recurrence_args(&mut rng, ring);
                differential(&net, &proof, &form, &args, Mutation::WrongCombine)
            })
            .filter(|(same, _, _)| !same)
            .count();
        assert!(wrong > 0, "{ring:?} k={k}: wrong combine not caught");
    }
}

#[test]
fn large_k_l5_is_cost_gated_and_bounded() {
    for k in [5, 6] {
        let (net, proof) = support::compile(&recurrence(Ring::Wrapping, k));
        let form = normal_form::extract(&net, &proof).expect("extract");
        assert_eq!(
            form.functions[0].loops[0].node,
            Node::Serial {
                reason: COST_UNCALIBRATED.to_owned()
            },
            "k={k}"
        );
    }
    // The v0.1 cut is unchanged, and groups above the hard cap stay serial.
    let (net, proof) = support::compile(&recurrence(Ring::Wrapping, 4));
    let form = normal_form::extract(&net, &proof).expect("extract");
    assert!(form.certificates().any(|c| c.law == LawId::L5Matrix));
    let (net, proof) = support::compile(&recurrence(Ring::Wrapping, 17));
    let form = normal_form::extract_uncalibrated(&net, &proof).expect("extract");
    let Node::Serial { reason } = &form.functions[0].loops[0].node else {
        panic!("k=17 admitted: {form:#?}");
    };
    assert_eq!(reason, "matrix group of 17 variables");
}
