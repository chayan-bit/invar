#![forbid(unsafe_code)]
//! P1 soundness: randomized differential tests of the law executor against the
//! CPU net oracle under observation contract v2 (value or first source-order
//! trap, plus committed `mut` arrays), at least 3,000 cases per v0.1 law
//! including trapping cases, and one mutation per law that must be caught.

#[path = "support/normal_form.rs"]
mod support;

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

/// SplitMix64: a deterministic generator (no new dependency).
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }
    fn below(&mut self, n: u64) -> u64 {
        self.next() % n.max(1)
    }
    fn range(&mut self, lo: i64, hi: i64) -> i64 {
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

const L1_MAP: &str = "fn f(xs: read Array<i32>, ys: read Array<i32>, n: u32) -> Array<i32> { var out = filled_i32(n, 0); for i in 0u32 .. n { out[i] = xs[i] / ys[i]; out[i] = out[i] + 1; } return move out; }";
const L1_STRIDE: &str = "fn f(xs: read Array<i32>, n: u32, k: u32) -> Array<i32> { var out = filled_i32(n * 3u32 + 1u32, 0); for i in 0u32 .. n { out[i * k] = out[i * k] + xs[i]; } return move out; }";
const L2_SEALED: &str = "fn f(xs: read Array<u32>, n: u32) -> u32 { var s = 7u32; var p = 3u32; for i in 0u32 .. n { s = wrapping_add(s, xs[i]); p = wrapping_mul(p, xs[i]); } return wrapping_add(s, p); }";
const L3_SUM: &str = "fn f(xs: read Array<i32>, n: u32, s0: i32) -> i32 { var s = s0; for i in 0u32 .. n { s = s + xs[i]; } return s; }";
const L3_PRODUCT: &str = "fn f(xs: read Array<i32>, n: u32, s0: i32) -> i32 { var p = s0; for i in 0u32 .. n { p = p * xs[i]; } return p; }";
const L4_CHECKED: &str = "fn f(xs: read Array<i32>, ys: read Array<i32>, n: u32, s0: i32) -> i32 { var s = s0; for i in 0u32 .. n { s = s * xs[i] + ys[i]; } return s; }";
const L4_WRAPPING: &str = "fn f(xs: read Array<u32>, ys: read Array<u32>, n: u32, s0: u32) -> u32 { var s = s0; for i in 0u32 .. n { s = wrapping_add(wrapping_mul(s, xs[i]), ys[i]); } return s; }";
const L4_RESIDUE: &str = "fn f(xs: read Array<i32>, ys: read Array<i32>, n: u32, s0: i32) -> i32 { var s = s0; for i in 0u32 .. n { s = (s * xs[i] + ys[i]) % 10007; } return s; }";
const L4_WHILE: &str = "fn w(x: i32) -> i32 { var s = x % 101; var i = 0; while i < x % 11 { s = (s * 2 + i * x) % 10007; i = i + 1; } return s; }";
const L5_RESIDUE: &str = "fn f(cs: read Array<i32>, n: u32, a0: i32, b0: i32) -> i32 { var a = a0; var b = b0; for i in 0u32 .. n { let na = (a * 2 + b * cs[i]) % 10007; let nb = (a + b) % 10007; a = na; b = nb; } return a * 3 + b; }";
const L5_CHECKED: &str = "fn f(cs: read Array<i32>, n: u32, a0: i32, b0: i32) -> i32 { var a = a0; var b = b0; for i in 0u32 .. n { let na = a + b * cs[i]; let nb = a - b; a = na; b = nb; } return a - b; }";
const L6_KADANE: &str = "fn f(xs: read Array<i32>, n: u32) -> i32 { var best = 0 - 2147483647; var cur = 0; for i in 0u32 .. n { cur = cur + xs[i]; if cur < 0 { cur = 0; } if cur > best { best = cur; } } return best; }";
const L7_ARGMIN: &str = "fn f(xs: read Array<f32>, n: u32, m0: f32) -> u32 { var m = m0; var at = 4294967295u32; for i in 0u32 .. n { if xs[i] < m { m = xs[i]; at = i; } } if m != m { return at + 1u32; } return at; }";
const L7_ARGMAX: &str = "fn f(xs: read Array<i32>, n: u32) -> i32 { var m = 0 - 1000; var at = 0u32; for i in 0u32 .. n { if xs[i] >= m { m = xs[i]; at = i; } } if at % 2u32 == 0u32 { return m; } return m + 1; }";
const L9_PATHSUM: &str = "fn leaf(x: i32) -> i32 { if x % 2 == 0 { return (x / 2 + 16) % 10007; } return (x * 3 + 1) % 10007; } fn dc(lo: i32, hi: i32, d: i32) -> i32 { if hi - lo <= 5 { return leaf(lo * 4 + hi); } if d <= 0 { return leaf(hi - lo + 7); } let m1 = lo + 1 + ((hi - lo - 2) * 3) / 9; let m2 = m1 + 1 + ((hi - m1 - 2) * 3) / 6; let a = dc(lo, m1, d - 1); let b = dc(m1, m2, d - 1); let c = dc(m2, hi, d - 1); return (a * 5 + b + c * 2) % 100003; } fn run(lo: i32, a: i32, b: i32) -> i32 { return dc(lo, lo + a, b); }";
const L9_TAIL: &str = "fn w(x: i32) -> i32 { return (x * 7 + 3) % 1009; } fn t(n: i32, s: i32) -> i32 { if n <= 0 { return w(s); } let sel = (s * 29 + n * 6) % 4; if sel == 0 { let v0 = t(n - 1, (s * 14 + 60) % 1009); let v1 = t(n - 1, (s * 12 + 66) % 1009); return (v0 * 7 + v1 + s) % 10007; } if sel == 1 { return (w(s) + n) % 10007; } return t(n - 1, (s * 4 + 22) % 1009); }";
const L10_RANGE: &str = "fn leaf(x: i32) -> i32 { if x % 2 == 0 { return x * x - 7; } return x * x * 3 + 1; } fn sum(lo: i32, hi: i32) -> i32 { if hi - lo <= 1 { return leaf(lo); } let m = lo + (hi - lo) / 2; let a = sum(lo, m); let b = sum(m, hi); return a + b; }";
const L11_CHECKED: &str = "fn f(h: mut Array<i32>, ix: read Array<u32>, w: read Array<i32>, n: u32, m: u32) -> Unit { for i in 0u32 .. n { h[ix[i] % m] = h[ix[i] % m] + w[i]; } }";
const L11_WRAPPING: &str = "fn f(h: mut Array<u32>, ix: read Array<u32>, w: read Array<u32>, n: u32, m: u32) -> Unit { for i in 0u32 .. n { h[ix[i] % m] = wrapping_add(h[ix[i] % m], w[i]); } }";
const L12_ORDERED: &str = "fn f(out: mut Array<i32>, ix: read Array<u32>, xs: read Array<i32>, n: u32) -> i32 { var last = 0 - 1; for i in 0u32 .. n { out[ix[i]] = xs[i] * 2; if xs[i] > 0 { last = xs[i]; } } return last; }";
const L13_COMPACT: &str = "fn f(out: mut Array<i32>, xs: read Array<i32>, n: u32, t: i32) -> u32 { var c = 0u32; for i in 0u32 .. n { if xs[i] > t { out[c] = xs[i]; c = c + 1u32; } } return c; }";
const L14_FRONTIER: &str = "fn succ(v: u32, k: u32, n: u32) -> u32 { return (v * 13u32 + k * 2u32 + 5u32) % n; } fn expand_round(frontier: read Array<u32>, seen: mut Array<u32>, next: mut Array<u32>, n: u32, c: u32) -> i32 { var count = 0; for u in 0u32 .. n { if frontier[u] == 1u32 and seen[u] == 0u32 { seen[u] = 1u32; count = count + 1; let degree = u % 4u32 + 1u32; for k in 0u32 .. degree { if seen[succ(u, k * c, n)] == 0u32 { next[succ(u, k * c, n)] = 1u32; } } } } return count; }";
const L15_FIRST: &str = "fn f(xs: read Array<i32>, n: u32, key: i32) -> u32 { for i in 0u32 .. n { if 100 / xs[i] == key { return i; } } return n; }";
const L15_SCAN: &str = "fn f(xs: read Array<i32>, n: u32, key: i32) -> u32 { var s = 0; for i in 0u32 .. n { s = s + xs[i]; if s > key { return i; } } return n + 1u32; }";
const L19_FUSED: &str = "fn f(xs: read Array<i32>, n: u32) -> Array<i32> { var a = filled_i32(n, 0); for i in 0u32 .. n { a[i] = xs[i] * 3; } var b = filled_i32(n, 0); for j in 0u32 .. n { b[j] = a[j] + xs[j] / 2; } return move b; }";
const L20_FORK: &str = "fn g(x: i32) -> i32 { return 1000 / x + x * x; } fn h(x: i32, y: i32) -> i32 { return x * y - 3; } fn f(a: i32, b: i32) -> i32 { let p = g(a); let q = h(a, b); let r = g(b); return p + q + r; }";

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

#[test]
fn every_law_agrees_with_the_net_oracle() {
    let mut per_law: BTreeMap<LawId, Tally> = BTreeMap::new();
    let mut report = Vec::new();
    let mut failures = Vec::new();
    for (p, program) in programs().iter().enumerate() {
        let (net, proof, form) = compiled(program);
        let mut rng = Rng(0x5eed_0000 + p as u64);
        let mut tally = Tally::default();
        for _ in 0..CASES {
            let args = (program.args)(&mut rng);
            let Some((same, receipt, trapped)) =
                differential(&net, &proof, &form, program, &args, Mutation::None)
            else {
                continue;
            };
            tally.cases += 1;
            tally.traps += u64::from(trapped);
            let used = program
                .laws
                .iter()
                .any(|law| receipt.applied.contains_key(law));
            tally.law_path += u64::from(used);
            tally.fallbacks +=
                u64::from(!receipt.guard_fallbacks.is_empty() || !receipt.violations.is_empty());
            if !receipt.violations.is_empty() && tally.mismatches.len() < 5 {
                tally
                    .mismatches
                    .push(format!("violation {:?} on {args:?}", receipt.violations));
            }
            if !same && tally.mismatches.len() < 5 {
                tally.mismatches.push(format!("mismatch on {args:?}"));
            }
        }
        report.push(format!(
            "{:<16} cases={} traps={} law_path={} fallbacks={}",
            program.name, tally.cases, tally.traps, tally.law_path, tally.fallbacks
        ));
        if !tally.mismatches.is_empty() {
            failures.push(format!("{}: {:#?}", program.name, tally.mismatches));
        }
        if tally.traps == 0 {
            failures.push(format!("{}: no trapping cases", program.name));
        }
        if tally.law_path * 2 <= tally.cases {
            failures.push(format!(
                "{}: law path used in only {} of {} cases",
                program.name, tally.law_path, tally.cases
            ));
        }
        for law in program.laws {
            let t = per_law.entry(*law).or_default();
            t.cases += tally.cases;
            t.traps += tally.traps;
            t.law_path += tally.law_path;
            t.fallbacks += tally.fallbacks;
        }
    }
    for (law, t) in &per_law {
        report.push(format!(
            "{:<4} cases={} traps={} law_path={} fallbacks={}",
            law.to_string(),
            t.cases,
            t.traps,
            t.law_path,
            t.fallbacks
        ));
    }
    eprintln!("{}", report.join("\n"));
    assert!(failures.is_empty(), "failures:\n{}", failures.join("\n"));
    let covered: Vec<LawId> = per_law.keys().copied().collect();
    assert_eq!(covered, LawId::ALL.to_vec(), "every v0.1 law is exercised");
    for (law, t) in &per_law {
        assert!(t.cases >= 3000, "{law:?}: {} cases", t.cases);
    }
}

/// Each mutation of the reference semantics must be caught by the oracle on
/// the programs of the law it breaks: the last trap instead of the first, a
/// dropped runtime guard, and a wrong combine (the right operand only, or a
/// reversed merge, or a stale round-start read).
#[test]
fn every_law_has_a_caught_mutation() {
    const MUTATION_CASES: usize = 800;
    let mutations = [
        Mutation::LastTrap,
        Mutation::DropGuards,
        Mutation::WrongCombine,
    ];
    let mut caught: BTreeMap<(LawId, &str), u64> = BTreeMap::new();
    let mut report = Vec::new();
    for (p, program) in programs().iter().enumerate() {
        let (net, proof, form) = compiled(program);
        let primary = *program.laws.last().expect("laws");
        let mut row = Vec::new();
        for mutation in mutations {
            let mut rng = Rng(0xbad_0000 + p as u64);
            let mut wrong = 0;
            for _ in 0..MUTATION_CASES {
                let args = (program.args)(&mut rng);
                if let Some((false, _, _)) =
                    differential(&net, &proof, &form, program, &args, mutation)
                {
                    wrong += 1;
                }
            }
            *caught
                .entry((primary, mutation_name(mutation)))
                .or_default() += wrong;
            row.push(format!("{}={wrong}", mutation_name(mutation)));
        }
        report.push(format!("{:<16} {}", program.name, row.join(" ")));
    }
    eprintln!("{}", report.join("\n"));
    let mut missing = Vec::new();
    for law in LawId::ALL {
        let total: u64 = mutations
            .iter()
            .map(|m| caught.get(&(law, mutation_name(*m))).copied().unwrap_or(0))
            .sum();
        if total == 0 {
            missing.push(law);
        }
    }
    assert!(missing.is_empty(), "no mutation caught for {missing:?}");
    for law in [LawId::L4Affine, LawId::L5Matrix, LawId::L9PathSum] {
        assert!(
            caught.get(&(law, "drop-guards")).copied().unwrap_or(0) > 0,
            "{law:?}: dropped guard not caught"
        );
    }
    let stride = programs()
        .into_iter()
        .find(|p| p.name == "l1-stride")
        .expect("l1-stride");
    let (net, proof, form) = compiled(&stride);
    let mut rng = Rng(0xbad_1000);
    let dropped = (0..MUTATION_CASES)
        .filter(|_| {
            let args = (stride.args)(&mut rng);
            matches!(
                differential(&net, &proof, &form, &stride, &args, Mutation::DropGuards),
                Some((false, _, _))
            )
        })
        .count();
    assert!(dropped > 0, "L1 stride: dropped guard not caught");
}

fn mutation_name(mutation: Mutation) -> &'static str {
    match mutation {
        Mutation::None => "none",
        Mutation::LastTrap => "last-trap",
        Mutation::DropGuards => "drop-guards",
        Mutation::WrongCombine => "wrong-combine",
    }
}

fn form_of(source: &str) -> (Net, Proof, NormalForm) {
    let (net, proof) = support::compile(source);
    let form = normal_form::extract(&net, &proof).expect("extract");
    (net, proof, form)
}

fn cert_where(form: &NormalForm, law: LawId, subject: &Subject) -> Certificate {
    form.certificates()
        .find(|c| c.law == law && c.subject == *subject)
        .cloned()
        .unwrap_or_else(|| panic!("no {law:?} certificate for {subject:?}"))
}

fn var(name: &str) -> Subject {
    Subject::Var(name.to_owned())
}

/// The checker re-derives every certified fact from the net: each forgery
/// below must be rejected (never accepted by name, hash or benchmark id).
#[test]
fn forged_certificates_are_rejected() {
    let mut accepted = Vec::new();
    let mut reject = |label: &str, net: &Net, proof: &Proof, cert: &Certificate| {
        if certificate::check(net, proof, cert).is_ok() {
            accepted.push(label.to_owned());
        }
    };
    let (net, proof, form) = form_of(L3_SUM);
    let sum = cert_where(&form, LawId::L3TrapLiftedFold, &var("s"));
    assert_eq!(certificate::check(&net, &proof, &sum), Ok(()));
    let forge = |f: &dyn Fn(&mut Certificate)| {
        let mut c = sum.clone();
        f(&mut c);
        c
    };
    reject(
        "fingerprint",
        &net,
        &proof,
        &forge(&|c| c.fingerprint[0] ^= 1),
    );
    reject("version", &net, &proof, &forge(&|c| c.version += 1));
    reject(
        "region function",
        &net,
        &proof,
        &forge(&|c| c.region.function += 1),
    );
    reject(
        "region head",
        &net,
        &proof,
        &forge(&|c| c.region.head = None),
    );
    reject(
        "law swap",
        &net,
        &proof,
        &forge(&|c| c.law = LawId::L4Affine),
    );
    reject(
        "rep swap",
        &net,
        &proof,
        &forge(&|c| c.rep = Some(Rep::SaturatedProduct(IntTy::I32))),
    );
    reject(
        "rep type",
        &net,
        &proof,
        &forge(&|c| c.rep = Some(Rep::LiftedSum(IntTy::U32))),
    );
    reject(
        "guard added",
        &net,
        &proof,
        &forge(&|c| c.guards.push(Guard::ReducedSeed("s".into()))),
    );
    reject(
        "subject renamed",
        &net,
        &proof,
        &forge(&|c| c.subject = var("zz")),
    );
    reject("facts altered", &net, &proof, &forge(&|c| c.facts.clear()));
    reject(
        "annotation added",
        &net,
        &proof,
        &forge(&|c| {
            c.annotations.push(Annotated::rederived(Annotation::Narrow {
                target: Target::Var("s".into()),
                bits: 32,
                signedness: Signedness::Signed,
                license: Vec::new(),
            }))
        }),
    );
    let (other_net, other_proof, _) = form_of(L3_PRODUCT);
    reject("certificate of another net", &other_net, &other_proof, &sum);

    let (net, proof, form) = form_of(L3_PRODUCT);
    let product = cert_where(&form, LawId::L3TrapLiftedFold, &var("p"));
    let mut lifted = product.clone();
    lifted.rep = Some(Rep::LiftedSum(IntTy::I32));
    reject("product claimed as a sum", &net, &proof, &lifted);

    let (net, proof, form) = form_of(L4_RESIDUE);
    let residue = cert_where(&form, LawId::L4Affine, &var("s"));
    assert_eq!(certificate::check(&net, &proof, &residue), Ok(()));
    let forge = |f: &dyn Fn(&mut Certificate)| {
        let mut c = residue.clone();
        f(&mut c);
        c
    };
    reject(
        "ring",
        &net,
        &proof,
        &forge(&|c| c.rep = Some(Rep::Affine(Ring::Checked64(IntTy::I32)))),
    );
    reject(
        "modulus",
        &net,
        &proof,
        &forge(&|c| {
            c.rep = Some(Rep::Affine(Ring::Residue {
                modulus: 10009,
                ty: IntTy::I32,
            }))
        }),
    );
    reject(
        "guard dropped",
        &net,
        &proof,
        &forge(&|c| {
            c.guards.retain(|g| !matches!(g, Guard::ReducedSeed(_)));
        }),
    );
    reject(
        "guard duplicated",
        &net,
        &proof,
        &forge(&|c| {
            let g = c.guards[0].clone();
            c.guards.push(g);
            c.guards.remove(1);
        }),
    );
    reject(
        "affine offset",
        &net,
        &proof,
        &forge(&|c| {
            if let Some(Fact::Affine { offset, .. }) = c.facts.first_mut() {
                *offset = Expr::I32(1);
            }
        }),
    );

    let (net, proof, form) = form_of(L1_STRIDE);
    let stride = cert_where(&form, LawId::L1Map, &Subject::Array("out".into()));
    assert_eq!(stride.guards, vec![Guard::StrideNonZero("k".into())]);
    let mut unguarded = stride.clone();
    unguarded.guards.clear();
    reject("stride guard dropped", &net, &proof, &unguarded);

    let (net, proof, form) = form_of(L2_SEALED);
    let mut sealed = cert_where(&form, LawId::L2SealedFold, &var("s"));
    sealed.rep = Some(Rep::Sealed(invar::laws::Law::WrappingProduct));
    reject("sealed sum as product", &net, &proof, &sealed);

    let (net, proof, form) = form_of(L7_ARGMIN);
    let select = cert_where(&form, LawId::L7Select, &var("m"));
    let mut flipped = select.clone();
    if let Some(Rep::Select { strict, .. }) = &mut flipped.rep {
        *strict = !*strict;
    }
    reject("select strictness", &net, &proof, &flipped);
    let mut dropped = form.clone();
    for function in &mut dropped.functions {
        for loop_form in &mut function.loops {
            for c in &mut loop_form.certificates {
                if let Some(Rep::Select { companions, .. }) = &mut c.rep {
                    companions.clear();
                }
            }
            if let Node::Parallel(node) = &mut loop_form.node {
                for carried in &mut node.carried {
                    if let Rep::Select { companions, .. } = &mut carried.rep {
                        companions.clear();
                    }
                }
            }
        }
    }
    assert!(
        certificate::check_all(&net, &proof, &dropped).is_err(),
        "companions dropped"
    );

    let (net, proof, form) = form_of(L11_CHECKED);
    let mut combine = cert_where(&form, LawId::L11ScatterCombine, &Subject::Array("h".into()));
    combine.law = LawId::L1Map;
    reject("scatter claimed owned", &net, &proof, &combine);
    let (net, proof, form) = form_of(L12_ORDERED);
    let mut ordered = cert_where(&form, LawId::L12OrderedWrite, &Subject::Array("out".into()));
    ordered.law = LawId::L14MonotoneFrontier;
    reject("ordered claimed monotone", &net, &proof, &ordered);

    let (net, proof, form) = form_of(L9_PATHSUM);
    let path = cert_where(&form, LawId::L9PathSum, &Subject::Recursion);
    let mut table = path.clone();
    table.guards = vec![
        Guard::TableRange { modulus: 100_019 },
        Guard::NonNegativeConstants,
        Guard::NoOverflowCombine,
    ];
    reject("path-sum table modulus", &net, &proof, &table);
    let mut coefficient = path.clone();
    if let Some(Fact::Combine { coefficients, .. }) = coefficient.facts.first_mut() {
        coefficients[0].1 += 1;
    }
    reject("path-sum coefficient", &net, &proof, &coefficient);
    let mut as_range = path.clone();
    as_range.law = LawId::L10RangeSum;
    as_range.guards.clear();
    reject("path sum claimed range sum", &net, &proof, &as_range);

    let (net, proof, form) = form_of(L20_FORK);
    let mut fork = cert_where(
        &form,
        LawId::L20ForkJoin,
        &form
            .certificates()
            .find(|c| c.law == LawId::L20ForkJoin)
            .expect("fork")
            .subject,
    );
    fork.facts = vec![Fact::Siblings(4)];
    reject("fork siblings + 1", &net, &proof, &fork);

    let (net, proof, form) = form_of(L19_FUSED);
    let fused = form
        .certificates()
        .find(|c| c.law == LawId::L19MapFusion)
        .cloned()
        .expect("fused");
    let mut self_fused = fused.clone();
    self_fused.subject = Subject::Fused(fused.region);
    reject("fusion with a non-adjacent loop", &net, &proof, &self_fused);

    let (net, proof, form) = form_of(L15_SCAN);
    let mut stripped = form.clone();
    for function in &mut stripped.functions {
        for loop_form in &mut function.loops {
            loop_form
                .certificates
                .retain(|c| !matches!(c.subject, Subject::Var(_)));
            if let Node::Parallel(node) = &mut loop_form.node {
                node.carried.clear();
            }
        }
    }
    assert!(
        certificate::check_all(&net, &proof, &stripped).is_err(),
        "carried state removed"
    );
    assert!(
        accepted.is_empty(),
        "forged certificates accepted: {accepted:?}"
    );
}
