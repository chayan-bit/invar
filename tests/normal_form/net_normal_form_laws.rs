#![forbid(unsafe_code)]
//! P1 soundness: randomized differential tests of the law executor against the
//! CPU net oracle under observation contract v2 (value or first source-order
//! trap, plus committed `mut` arrays), at least 3,000 cases per v0.1 law
//! including trapping cases, and one mutation per law that must be caught.

use crate::law_programs::*;
use crate::support;

use invar::{
    eval::{self, EvalError},
    general_net::{
        Net, Proof,
        normal_form::{
            self, Annotated, Annotation, Certificate, Fact, Guard, IntTy, LawId, Node, NormalForm,
            Rep, Ring, Signedness, Subject, Target, certificate,
            exec::{self, Mutation},
        },
        oracle,
        regions::tree::Expr,
    },
    numeric_contract::F32Bits,
};
use std::{cell::RefCell, collections::BTreeMap, rc::Rc};

mod cases;

/// SplitMix64: a deterministic generator (no new dependency).
pub struct Rng(pub u64);

impl Rng {
    pub fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }
    pub fn below(&mut self, n: u64) -> u64 {
        self.next() % n.max(1)
    }
    pub fn range(&mut self, lo: i64, hi: i64) -> i64 {
        lo + self.below((hi - lo + 1) as u64) as i64
    }
    fn chance(&mut self, percent: u64) -> bool {
        self.below(100) < percent
    }
    /// Mostly small values, sometimes huge (overflow), sometimes zero.
    fn i32(&mut self) -> i32 {
        match self.below(10) {
            0 => [i32::MIN, i32::MAX, -1, 0, 1][self.below(5) as usize],
            1 | 2 => self.range(-(1 << 30), 1 << 30) as i32,
            _ => self.range(-50, 50) as i32,
        }
    }
    fn len(&mut self) -> u32 {
        self.below(24) as u32
    }
}

fn ai32(v: Vec<i32>) -> eval::Value {
    eval::Value::ArrayI32(Rc::new(RefCell::new(v)))
}
fn au32(v: Vec<u32>) -> eval::Value {
    eval::Value::ArrayU32(Rc::new(RefCell::new(v)))
}
fn af32(v: Vec<F32Bits>) -> eval::Value {
    eval::Value::ArrayF32(Rc::new(RefCell::new(v)))
}
fn i(x: i32) -> eval::Value {
    eval::Value::I32(x)
}
fn u(x: u32) -> eval::Value {
    eval::Value::U32(x)
}

/// Deep copy, so the oracle and the executor commit to separate arrays.
fn copy(v: &eval::Value) -> eval::Value {
    match v {
        eval::Value::ArrayI32(a) => ai32(a.borrow().clone()),
        eval::Value::ArrayU32(a) => au32(a.borrow().clone()),
        eval::Value::ArrayF32(a) => af32(a.borrow().clone()),
        eval::Value::I32(x) => i(*x),
        eval::Value::U32(x) => u(*x),
        eval::Value::F32(x) => eval::Value::F32(*x),
        eval::Value::Bool(x) => eval::Value::Bool(*x),
        _ => eval::Value::Unit,
    }
}

/// The length an array argument should have: usually `n`, sometimes shorter
/// (an index trap at the first missing element).
fn short(rng: &mut Rng, n: u32) -> usize {
    if rng.chance(15) && n > 0 {
        rng.below(u64::from(n)) as usize
    } else {
        n as usize
    }
}

fn vec_i32(rng: &mut Rng, len: usize, make: &mut dyn FnMut(&mut Rng) -> i32) -> Vec<i32> {
    (0..len).map(|_| make(rng)).collect()
}

struct Program {
    name: &'static str,
    source: &'static str,
    entry: &'static str,
    /// Laws the normal form must certify for this program.
    laws: &'static [LawId],
    args: fn(&mut Rng) -> Vec<eval::Value>,
}

fn small(rng: &mut Rng) -> i32 {
    rng.range(-3, 3) as i32
}

fn programs() -> Vec<Program> {
    use LawId::*;
    vec![
        Program {
            name: "l1-map",
            source: L1_MAP,
            entry: "f",
            laws: &[L1Map],
            args: |r| {
                let n = r.len();
                let (a, b) = (short(r, n), short(r, n));
                vec![
                    ai32(vec_i32(r, a, &mut Rng::i32)),
                    ai32(vec_i32(r, b, &mut |r| {
                        if r.chance(10) { 0 } else { r.i32() }
                    })),
                    u(n),
                ]
            },
        },
        Program {
            name: "l1-stride",
            source: L1_STRIDE,
            entry: "f",
            laws: &[L1Map],
            args: |r| {
                let n = r.len();
                let a = short(r, n);
                vec![
                    ai32(vec_i32(r, a, &mut Rng::i32)),
                    u(n),
                    u(r.below(4) as u32),
                ]
            },
        },
        Program {
            name: "l2-sealed",
            source: L2_SEALED,
            entry: "f",
            laws: &[L1Map, L2SealedFold],
            args: |r| {
                let n = r.len();
                let a = short(r, n);
                vec![au32((0..a).map(|_| r.next() as u32).collect()), u(n)]
            },
        },
        Program {
            name: "l3-sum",
            source: L3_SUM,
            entry: "f",
            laws: &[L1Map, L3TrapLiftedFold],
            args: |r| {
                let n = r.len();
                let a = short(r, n);
                vec![ai32(vec_i32(r, a, &mut Rng::i32)), u(n), i(r.i32())]
            },
        },
        Program {
            name: "l3-product",
            source: L3_PRODUCT,
            entry: "f",
            laws: &[L1Map, L3TrapLiftedFold],
            args: |r| {
                let n = r.len();
                let a = short(r, n);
                vec![
                    ai32(vec_i32(r, a, &mut |r| {
                        if r.chance(20) {
                            r.i32()
                        } else {
                            r.range(-9, 9) as i32
                        }
                    })),
                    u(n),
                    i(r.range(-3, 3) as i32),
                ]
            },
        },
        Program {
            name: "l4-checked",
            source: L4_CHECKED,
            entry: "f",
            laws: &[L1Map, L4Affine],
            args: |r| {
                let n = r.len();
                let (a, b) = (short(r, n), short(r, n));
                vec![
                    ai32(vec_i32(r, a, &mut small)),
                    ai32(vec_i32(r, b, &mut Rng::i32)),
                    u(n),
                    i(r.i32()),
                ]
            },
        },
        Program {
            name: "l4-wrapping",
            source: L4_WRAPPING,
            entry: "f",
            laws: &[L1Map, L4Affine],
            args: |r| {
                let n = r.len();
                let (a, b) = (short(r, n), short(r, n));
                vec![
                    au32((0..a).map(|_| r.next() as u32).collect()),
                    au32((0..b).map(|_| r.next() as u32).collect()),
                    u(n),
                    u(r.next() as u32),
                ]
            },
        },
        Program {
            name: "l4-residue",
            source: L4_RESIDUE,
            entry: "f",
            laws: &[L1Map, L4Affine],
            args: |r| {
                let n = r.len();
                let (a, b) = (short(r, n), short(r, n));
                let neg = r.chance(15);
                // Rarely a coefficient whose product overflows.
                let coef = |r: &mut Rng| match r.below(40) {
                    0 | 1 if neg => -(r.range(1, 50) as i32),
                    2 => r.range(60000, i64::from(i32::MAX)) as i32,
                    _ => r.range(0, 60000) as i32,
                };
                let mut c1 = coef;
                let mut c2 = coef;
                vec![
                    ai32(vec_i32(r, a, &mut c1)),
                    ai32(vec_i32(r, b, &mut c2)),
                    u(n),
                    i(match r.below(10) {
                        0 => -(r.range(1, 99) as i32),
                        1 => r.range(10007, i64::from(i32::MAX)) as i32,
                        _ => r.range(0, 10006) as i32,
                    }),
                ]
            },
        },
        Program {
            name: "l4-while",
            source: L4_WHILE,
            entry: "w",
            laws: &[L1Map, L4Affine],
            args: |r| vec![i(r.i32())],
        },
        Program {
            name: "l5-residue",
            source: L5_RESIDUE,
            entry: "f",
            laws: &[L1Map, L5Matrix],
            args: |r| {
                let n = r.len();
                let a = short(r, n);
                let neg = r.chance(10);
                vec![
                    // Seeds above P and overflowing coefficients.
                    ai32(vec_i32(r, a, &mut |r| match r.below(40) {
                        0..=11 if neg => -1,
                        12 => r.range(30000, i64::from(i32::MAX)) as i32,
                        _ => r.range(0, 30000) as i32,
                    })),
                    u(n),
                    i(match r.below(10) {
                        0 => -5,
                        1 => r.range(10007, i64::from(i32::MAX)) as i32,
                        _ => r.range(0, 10006) as i32,
                    }),
                    i(r.range(0, 10006) as i32),
                ]
            },
        },
        Program {
            name: "l5-checked",
            source: L5_CHECKED,
            entry: "f",
            laws: &[L1Map, L5Matrix],
            args: |r| {
                let n = r.len();
                let a = short(r, n);
                vec![
                    ai32(vec_i32(r, a, &mut small)),
                    u(n),
                    i(r.i32()),
                    i(r.i32()),
                ]
            },
        },
        Program {
            name: "l6-kadane",
            source: L6_KADANE,
            entry: "f",
            laws: &[L1Map, L7Select, L6Tropical],
            args: |r| {
                let n = r.len();
                let a = short(r, n);
                vec![ai32(vec_i32(r, a, &mut Rng::i32)), u(n)]
            },
        },
        Program {
            name: "l7-argmin-f32",
            source: L7_ARGMIN,
            entry: "f",
            laws: &[L1Map, L7Select],
            args: |r| {
                let n = r.len();
                let a = short(r, n);
                let pool = [
                    0.0f32,
                    -0.0,
                    1.5,
                    -2.25,
                    f32::NAN,
                    f32::INFINITY,
                    f32::NEG_INFINITY,
                    3.0,
                ];
                let pick = |r: &mut Rng| {
                    if r.chance(50) {
                        pool[r.below(8) as usize]
                    } else {
                        (r.range(-500, 500) as f32) / 100.0
                    }
                };
                let xs = (0..a)
                    .map(|_| F32Bits::from_bits(pick(r).to_bits()))
                    .collect();
                let seed = [f32::INFINITY, 0.0, -0.0, f32::NAN, 1.0][r.below(5) as usize];
                vec![
                    af32(xs),
                    u(n),
                    eval::Value::F32(F32Bits::from_bits(seed.to_bits())),
                ]
            },
        },
        Program {
            name: "l7-argmax-i32",
            source: L7_ARGMAX,
            entry: "f",
            laws: &[L1Map, L7Select],
            args: |r| {
                let n = r.len();
                let a = short(r, n);
                vec![
                    ai32(vec_i32(r, a, &mut |r| {
                        if r.chance(5) {
                            i32::MAX
                        } else {
                            r.range(-20, 20) as i32
                        }
                    })),
                    u(n),
                ]
            },
        },
        Program {
            name: "l9-pathsum",
            source: L9_PATHSUM,
            entry: "run",
            laws: &[L9PathSum],
            args: |r| {
                let lo = match r.below(10) {
                    0 => r.range(1 << 28, 1 << 30) as i32,
                    1 => r.range(-300, -1) as i32,
                    _ => r.range(0, 400) as i32,
                };
                vec![i(lo), i(r.range(0, 250) as i32), i(r.range(-1, 5) as i32)]
            },
        },
        Program {
            name: "l9-tail",
            source: L9_TAIL,
            entry: "t",
            laws: &[L9PathSum],
            args: |r| {
                let s = match r.below(10) {
                    0 => r.range(1 << 28, 1 << 30) as i32,
                    1 => r.range(-2000, -1) as i32,
                    _ => r.range(0, 1008) as i32,
                };
                vec![i(r.range(-1, 9) as i32), i(s)]
            },
        },
        Program {
            name: "l10-range",
            source: L10_RANGE,
            entry: "sum",
            laws: &[L10RangeSum],
            args: |r| {
                let lo = match r.below(6) {
                    0 => r.range(40000, 50000) as i32,
                    1 => r.range(-50000, -40000) as i32,
                    _ => r.range(-200, 200) as i32,
                };
                vec![i(lo), i(lo.saturating_add(r.range(0, 60) as i32))]
            },
        },
        Program {
            name: "l11-checked",
            source: L11_CHECKED,
            entry: "f",
            laws: &[L1Map, L11ScatterCombine],
            args: |r| {
                let n = r.len();
                let (a, b) = (short(r, n), short(r, n));
                let m = r.range(0, 6) as u32;
                let hl = (m as usize).max(1) - usize::from(r.chance(10));
                let h = vec_i32(r, hl, &mut Rng::i32);
                vec![
                    ai32(h),
                    au32((0..a).map(|_| r.below(40) as u32).collect()),
                    ai32(vec_i32(r, b, &mut Rng::i32)),
                    u(n),
                    u(m),
                ]
            },
        },
        Program {
            name: "l11-wrapping",
            source: L11_WRAPPING,
            entry: "f",
            laws: &[L1Map, L11ScatterCombine],
            args: |r| {
                let n = r.len();
                let (a, b) = (short(r, n), short(r, n));
                let m = r.range(0, 6) as u32;
                let h = (0..m.max(1)).map(|_| r.next() as u32).collect();
                vec![
                    au32(h),
                    au32((0..a).map(|_| r.below(40) as u32).collect()),
                    au32((0..b).map(|_| r.next() as u32).collect()),
                    u(n),
                    u(m),
                ]
            },
        },
        Program {
            name: "l12-ordered",
            source: L12_ORDERED,
            entry: "f",
            laws: &[L1Map, L12OrderedWrite],
            args: |r| {
                let n = r.len();
                let (a, b) = (short(r, n), short(r, n));
                let m = r.range(1, 8) as u32;
                vec![
                    ai32(vec![0; m as usize]),
                    au32((0..a).map(|_| r.below(u64::from(m) + 1) as u32).collect()),
                    ai32(vec_i32(r, b, &mut Rng::i32)),
                    u(n),
                ]
            },
        },
        Program {
            name: "l13-compaction",
            source: L13_COMPACT,
            entry: "f",
            laws: &[L1Map, L13Compaction],
            args: |r| {
                let n = r.len();
                let a = short(r, n);
                let ol = r.below(u64::from(n) + 2) as usize;
                vec![
                    ai32(vec![0; ol]),
                    ai32(vec_i32(r, a, &mut Rng::i32)),
                    u(n),
                    i(r.range(-10, 10) as i32),
                ]
            },
        },
        Program {
            name: "l14-frontier",
            source: L14_FRONTIER,
            entry: "expand_round",
            laws: &[L1Map, L3TrapLiftedFold, L14MonotoneFrontier],
            args: |r| {
                let n = r.range(0, 30) as u32;
                let len = short(r, n);
                let flags =
                    |r: &mut Rng, p| (0..len).map(|_| u32::from(r.chance(p))).collect::<Vec<_>>();
                let c = if r.chance(5) {
                    u32::MAX
                } else {
                    r.range(1, 9) as u32
                };
                vec![
                    au32(flags(r, 40)),
                    au32(flags(r, 30)),
                    au32(vec![0; len]),
                    u(n),
                    u(c),
                ]
            },
        },
        Program {
            name: "l15-first",
            source: L15_FIRST,
            entry: "f",
            laws: &[L1Map, L15FirstExit],
            args: |r| {
                let n = r.len();
                let a = short(r, n);
                vec![
                    ai32(vec_i32(r, a, &mut |r| r.range(-4, 4) as i32)),
                    u(n),
                    i(r.range(-30, 30) as i32),
                ]
            },
        },
        Program {
            name: "l15-scan",
            source: L15_SCAN,
            entry: "f",
            laws: &[L1Map, L3TrapLiftedFold, L15FirstExit],
            args: |r| {
                let n = r.len();
                let a = short(r, n);
                vec![ai32(vec_i32(r, a, &mut Rng::i32)), u(n), i(r.i32())]
            },
        },
        Program {
            name: "l19-fused",
            source: L19_FUSED,
            entry: "f",
            laws: &[L1Map, L19MapFusion],
            args: |r| {
                let n = r.len();
                let a = short(r, n);
                vec![ai32(vec_i32(r, a, &mut Rng::i32)), u(n)]
            },
        },
        Program {
            name: "l20-fork",
            source: L20_FORK,
            entry: "f",
            laws: &[L20ForkJoin],
            args: |r| {
                let pick = |r: &mut Rng| if r.chance(15) { 0 } else { r.i32() };
                vec![i(pick(r)), i(pick(r))]
            },
        },
    ]
}

fn limits() -> oracle::Limits {
    oracle::Limits {
        fuel: 50_000_000,
        transitions: 500_000_000,
        trace_events: 1 << 24,
        array_elements: 1 << 20,
    }
}

#[derive(Default, Debug)]
struct Tally {
    cases: u64,
    traps: u64,
    law_path: u64,
    fallbacks: u64,
    mismatches: Vec<String>,
}

/// One differential case: `Some(true)` when the executor agrees with the
/// oracle, `None` when the oracle hit a tool limit.
fn differential(
    net: &Net,
    proof: &Proof,
    form: &NormalForm,
    program: &Program,
    args: &[eval::Value],
    mutation: Mutation,
) -> Option<(bool, exec::Receipt, bool)> {
    let oracle_args: Vec<eval::Value> = args.iter().map(copy).collect();
    let exec_args: Vec<eval::Value> = args.iter().map(copy).collect();
    let observed =
        oracle::observe_call(net, proof, program.entry, &oracle_args, limits()).expect("oracle");
    if matches!(
        observed.semantic.result,
        Err(EvalError::Tool { .. }) | Err(EvalError::EvaluationLimit { .. })
    ) {
        return None;
    }
    let ours = exec::run(net, proof, form, program.entry, &exec_args, mutation).expect("executor");
    let trapped = matches!(observed.semantic.result, Err(EvalError::Trap { .. }));
    let same = ours.result == observed.semantic.result && oracle_args == exec_args;
    Some((same, ours.receipt, trapped))
}

fn compiled(program: &Program) -> (Net, Proof, NormalForm) {
    let (net, proof) = support::compile(program.source);
    let form = normal_form::extract(&net, &proof).expect("extract");
    let certified: Vec<LawId> = form.certificates().map(|c| c.law).collect();
    for law in program.laws {
        assert!(
            certified.contains(law),
            "{}: {law:?} not certified; form {form:#?}",
            program.name
        );
    }
    let checked = certificate::check_all(&net, &proof, &form);
    assert_eq!(
        checked,
        Ok(certified.len()),
        "{}: checker verdict; form {form:#?}",
        program.name
    );
    (net, proof, form)
}

const CASES: usize = 3200;
