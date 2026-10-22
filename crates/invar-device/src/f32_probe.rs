//! D1 per-process preflight probe for `strict-f32-v1` device execution.
//!
//! One fixed case table runs as a generated kernel on each device under the
//! same pinned compile options as strict plans. Strict device work is admitted
//! only when every strict-class case matches its exact IEEE 754 binary32
//! round-to-nearest-even result bit for bit. `/` and `sqrt` are reported but
//! never admitted by M4. A device that flushes subnormals fails the probe, which
//! makes strict regions ineligible on that target (D1). Expected bits are
//! literal constants, independently re-derived by unit and root oracle tests.

use crate::Capabilities;
use std::fmt::Write;

pub const VERSION: u32 = 1;
pub const KERNEL_NAME: &str = "invar_f32_probe_v1";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Op {
    Mul,
    Add,
    Sub,
    Fma,
    /// `c + a * b` written as source-order accumulation; must not fuse.
    Accumulate,
    Lt,
    Eq,
    FromU32,
    FromI32,
    TruncI32,
    Div,
    Sqrt,
}

impl Op {
    const ALL: [Op; 12] = [
        Op::Mul,
        Op::Add,
        Op::Sub,
        Op::Fma,
        Op::Accumulate,
        Op::Lt,
        Op::Eq,
        Op::FromU32,
        Op::FromI32,
        Op::TruncI32,
        Op::Div,
        Op::Sqrt,
    ];
    pub fn code(self) -> u32 {
        Self::ALL.iter().position(|&op| op == self).unwrap_or(0) as u32
    }
    /// Strict operations gate admission; `/` and `sqrt` stay CPU-only in M4.
    pub fn is_strict(self) -> bool {
        !matches!(self, Op::Div | Op::Sqrt)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Case {
    pub op: Op,
    pub a: u32,
    pub b: u32,
    pub c: u32,
    pub expected: u32,
    /// The exact result depends on gradual underflow (subnormal input/result).
    pub subnormal: bool,
    pub what: &'static str,
}

const fn case(op: Op, a: u32, b: u32, c: u32, expected: u32, subnormal: bool) -> Case {
    Case {
        op,
        a,
        b,
        c,
        expected,
        subnormal,
        what: "",
    }
}
const fn named(mut c: Case, what: &'static str) -> Case {
    c.what = what;
    c
}

const ONE: u32 = 0x3f80_0000;
const NEG_ZERO: u32 = 0x8000_0000;
const MIN_NORMAL: u32 = 0x0080_0000;
const HALF: u32 = 0x3f00_0000;
const TWO: u32 = 0x4000_0000;
const NAN: u32 = 0x7fc0_0000;

// One row per case keeps the exact table reviewable.
#[rustfmt::skip]
pub const CASES: &[Case] = &[
    named(case(Op::Mul, 1, ONE, 0, 1, true), "min subnormal * 1"),
    named(case(Op::Mul, 1, 0x7f80_0000, 0, 0x7f80_0000, true), "min subnormal * inf (DAZ gives NaN)"),
    named(case(Op::Mul, MIN_NORMAL, HALF, 0, 0x0040_0000, true), "min normal * 0.5 -> subnormal"),
    named(case(Op::Mul, MIN_NORMAL, 0x3f7f_ffff, 0, MIN_NORMAL, true), "subnormal-boundary tie rounds to min normal"),
    named(case(Op::Mul, 3, HALF, 0, 2, true), "subnormal tie to even"),
    named(case(Op::Mul, 0x3f80_0001, 0x3f80_0001, 0, 0x3f80_0002, false), "normal rounding"),
    named(case(Op::Mul, NEG_ZERO, ONE, 0, NEG_ZERO, false), "-0 * 1"),
    named(case(Op::Mul, 0x7f7f_ffff, TWO, 0, 0x7f80_0000, false), "overflow to +inf"),
    named(case(Op::Add, 1, 1, 0, 2, true), "subnormal + subnormal"),
    named(case(Op::Add, MIN_NORMAL, 0x8000_0001, 0, 0x007f_ffff, true), "normal - subnormal -> subnormal"),
    named(case(Op::Add, 0x0080_0001, 0x8080_0000, 0, 1, true), "normal + normal -> subnormal"),
    named(case(Op::Add, ONE, 0x3380_0000, 0, ONE, false), "tie to even (down)"),
    named(case(Op::Add, 0x3f80_0001, 0x3380_0000, 0, 0x3f80_0002, false), "tie to even (up)"),
    named(case(Op::Add, ONE, 0x33c0_0000, 0, 0x3f80_0001, false), "above half ulp"),
    named(case(Op::Add, NEG_ZERO, NEG_ZERO, 0, NEG_ZERO, false), "-0 + -0"),
    named(case(Op::Add, 0, NEG_ZERO, 0, 0, false), "+0 + -0"),
    named(case(Op::Add, ONE, 0xbf80_0000, 0, 0, false), "exact cancellation is +0"),
    named(case(Op::Add, 0x7f7f_ffff, 0x7f7f_ffff, 0, 0x7f80_0000, false), "sum overflow"),
    named(case(Op::Sub, 0x0080_0001, MIN_NORMAL, 0, 1, true), "normal - normal -> subnormal"),
    named(case(Op::Sub, ONE, ONE, 0, 0, false), "x - x is +0"),
    named(case(Op::Sub, NEG_ZERO, 0, 0, NEG_ZERO, false), "-0 - +0"),
    named(case(Op::Fma, 0x3f80_0800, 0x3f80_0800, 0xbf80_1000, 0x3380_0000, false), "single rounding"),
    named(case(Op::Fma, MIN_NORMAL, HALF, 0, 0x0040_0000, true), "fma subnormal result"),
    named(case(Op::Fma, 1, ONE, 1, 2, true), "fma subnormal operands"),
    named(case(Op::Accumulate, 0x3f80_0800, 0x3f80_0800, 0xbf80_1000, 0, false), "no implicit contraction"),
    named(case(Op::Accumulate, ONE, 0, 1, 1, true), "subnormal accumulator kept"),
    named(case(Op::Accumulate, 0, ONE, NEG_ZERO, 0, false), "-0 + (+0 * 1) is +0"),
    named(case(Op::Lt, 0, 1, 0, 1, true), "+0 < min subnormal"),
    named(case(Op::Lt, 1, 2, 0, 1, true), "subnormal ordering"),
    named(case(Op::Lt, NAN, ONE, 0, 0, false), "NaN is unordered"),
    named(case(Op::Lt, NEG_ZERO, 0, 0, 0, false), "-0 < +0 is false"),
    named(case(Op::Eq, NEG_ZERO, 0, 0, 1, false), "-0 == +0"),
    named(case(Op::Eq, NAN, NAN, 0, 0, false), "NaN != NaN"),
    named(case(Op::Eq, 1, 0, 0, 0, true), "min subnormal != 0"),
    named(case(Op::FromU32, 16_777_217, 0, 0, 0x4b80_0000, false), "u32 tie to even (down)"),
    named(case(Op::FromU32, 16_777_219, 0, 0, 0x4b80_0002, false), "u32 tie to even (up)"),
    named(case(Op::FromU32, u32::MAX, 0, 0, 0x4f80_0000, false), "u32 max rounds to 2^32"),
    named(case(Op::FromI32, 0xfeff_ffff, 0, 0, 0xcb80_0000, false), "i32 tie to even"),
    named(case(Op::FromI32, 0x8000_0000, 0, 0, 0xcf00_0000, false), "i32 min"),
    named(case(Op::TruncI32, 0x4020_0000, 0, 0, 2, false), "trunc 2.5"),
    named(case(Op::TruncI32, 0xc020_0000, 0, 0, 0xffff_fffe, false), "trunc -2.5"),
    named(case(Op::TruncI32, 0x4eff_ffff, 0, 0, 0x7fff_ff80, false), "trunc largest in-range"),
    named(case(Op::TruncI32, 1, 0, 0, 0, true), "trunc subnormal"),
    named(case(Op::TruncI32, 0xbf40_0000, 0, 0, 0, false), "trunc -0.75"),
    named(case(Op::Div, ONE, 0x4040_0000, 0, 0x3eaa_aaab, false), "1 / 3"),
    named(case(Op::Div, MIN_NORMAL, TWO, 0, 0x0040_0000, true), "div subnormal result"),
    named(case(Op::Div, 3, TWO, 0, 2, true), "div subnormal tie"),
    named(case(Op::Sqrt, TWO, 0, 0, 0x3fb5_04f3, false), "sqrt 2"),
    named(case(Op::Sqrt, 1, 0, 0, 0x1a35_04f3, true), "sqrt min subnormal"),
    named(case(Op::Sqrt, NEG_ZERO, 0, 0, NEG_ZERO, false), "sqrt -0"),
    named(case(Op::Sqrt, MIN_NORMAL, 0, 0, 0x2000_0000, false), "sqrt min normal"),
];

/// Kernel input: `[op, a, b, c]` per case. Values come from device memory so
/// no compiler can fold them.
pub fn words() -> Vec<u32> {
    CASES
        .iter()
        .flat_map(|c| [c.op.code(), c.a, c.b, c.c])
        .collect()
}

/// Bodies in `Op::ALL` order over floats `a`, `b`, `c` and their raw words
/// `aw`, `bw`, `cw`. `U(x)` is the target's binary32-to-word cast. Each body
/// sets the result word `r` and, for the operations the guarded Metal
/// contraction uses (`*`, `+`, accumulate), the device guard flag `f`.
const BODIES: [&str; 12] = [
    "r = U(a * b); f = invar_operand(aw) | invar_operand(bw) | invar_mul(aw, bw, r);",
    "r = U(a + b); f = invar_operand(aw) | invar_operand(bw) | invar_add(aw, bw, r);",
    "r = U(a - b); f = invar_operand(aw) | invar_operand(bw) | invar_add(aw, bw ^ 0x80000000u, r);",
    "r = U(FMA(a, b, c)); f = 1u;",
    "{ float s = c; s = s + a * b; r = U(s); const float p = a * b; f = invar_operand(aw) | invar_operand(bw) | invar_operand(cw) | invar_mul(aw, bw, U(p)) | invar_add(cw, U(p), r); }",
    "r = a < b ? 1u : 0u;",
    "r = a == b ? 1u : 0u;",
    "r = U(TO_FLOAT(aw));",
    "r = U(TO_FLOAT(TO_INT(aw)));",
    "r = (W)(TO_INT(a));",
    "r = U(a / b);",
    "r = U(SQRT(a));",
];

fn switch(cuda: bool) -> String {
    let rules: [(&str, &str); 6] = if cuda {
        [
            ("U(", "__float_as_uint("),
            ("FMA(", "fmaf("),
            ("TO_FLOAT(", "static_cast<float>("),
            ("TO_INT(", "static_cast<int>("),
            ("SQRT(", "sqrtf("),
            ("(W)", "(unsigned)"),
        ]
    } else {
        [
            ("U(", "as_type<uint>("),
            ("FMA(", "fma("),
            ("TO_FLOAT(", "float("),
            ("TO_INT(", "int("),
            ("SQRT(", "sqrt("),
            ("(W)", "as_type<uint>"),
        ]
    };
    let mut text = String::from("  switch (op) {\n");
    for (code, body) in BODIES.iter().enumerate() {
        let body = rules.iter().fold((*body).to_owned(), |text, (from, to)| {
            text.replace(from, to)
        });
        // Writing to a String cannot fail.
        let _ = writeln!(text, "    case {code}u: {body} break;");
    }
    text.push_str("    default: break;\n  }\n");
    text
}

/// MSL probe for the pinned strict Metal options (see `floating::MSL_OPTIONS`).
/// It shares the contraction's `A, B, C, F` buffer signature and grid-strided
/// launch, so runtimes submit it through the same fenced path: A = `words()`,
/// B = one unused word, C = two words per case (result, guard flag), F unused.
pub fn emit_msl() -> String {
    let n = CASES.len();
    let mut text = format!(
        "// invar-f32-probe-v{VERSION} numeric=strict-f32-v1 cases={n}\n#include <metal_stdlib>\n#pragma METAL fp contract(off)\nusing namespace metal;\n"
    );
    text.push_str(&crate::floating::guard_helpers(false));
    let _ = write!(
        text,
        "kernel void {KERNEL_NAME}(device const uint* A [[buffer(0)]], device const uint* B [[buffer(1)]], device uint* C [[buffer(2)]], device atomic_uint* F [[buffer(3)]], uint start [[thread_position_in_grid]], uint threads [[threads_per_grid]]) {{\n  (void)B; (void)F;\n  for (uint i = start; i < {n}u; i += threads) {{\n  const uint op = A[4u * i], aw = A[4u * i + 1u], bw = A[4u * i + 2u], cw = A[4u * i + 3u];\n  const float a = as_type<float>(aw), b = as_type<float>(bw), c = as_type<float>(cw);\n  uint r = 0xffffffffu, f = 0u;\n"
    );
    text.push_str(&switch(false));
    text.push_str("  C[2u * i] = r;\n  C[2u * i + 1u] = f;\n  }\n}\n");
    text
}

/// CUDA probe compiled with exactly `floating::cuda_strict_options(caps)`.
/// Plain operators deliberately test those options rather than intrinsics.
/// Same `A, B, C` layout as the MSL probe; CUDA never needs the guard, but the
/// flag word is reported identically.
pub fn emit_cuda(caps: Capabilities) -> Option<String> {
    let options = crate::floating::cuda_strict_options(caps)?;
    let n = CASES.len();
    let mut text = format!(
        "// invar-f32-probe-v{VERSION} numeric=strict-f32-v1 cases={n} caps={caps:?}\n// required-options: {}\n",
        options.join(" ")
    );
    text.push_str(&crate::floating::guard_helpers(true));
    let _ = write!(
        text,
        "extern \"C\" __global__ void {KERNEL_NAME}(const unsigned* A, const unsigned* B, unsigned* C) {{\n  (void)B;\n  for (unsigned i = blockIdx.x * blockDim.x + threadIdx.x; i < {n}u; i += blockDim.x * gridDim.x) {{\n  const unsigned op = A[4u * i], aw = A[4u * i + 1u], bw = A[4u * i + 2u], cw = A[4u * i + 3u];\n  const float a = __uint_as_float(aw), b = __uint_as_float(bw), c = __uint_as_float(cw);\n  unsigned r = 0xffffffffu, f = 0u;\n"
    );
    text.push_str(&switch(true));
    text.push_str("  C[2u * i] = r;\n  C[2u * i + 1u] = f;\n  }\n}\n");
    Some(text)
}

/// Operations a strict rank-2 contraction executes on a device. Only these
/// may be admitted under the Metal underflow guard; every other operation
/// stays refused on a flushing device.
pub const CONTRACTION_OPS: &[Op] = &[Op::Mul, Op::Add, Op::Accumulate];

/// Observed probe words compared with the exact table.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Report {
    observed: Vec<u32>,
    flags: Vec<bool>,
    mismatches: Vec<usize>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Error {
    WordCount { expected: usize, actual: usize },
    Guard(crate::floating::guard::InvalidWord),
}

/// Probe output staging: each case's result word starts at `u32::MAX` (no
/// case expects it) and its flag at `floating::guard::UNWRITTEN`.
pub fn unwritten() -> Vec<u32> {
    CASES
        .iter()
        .flat_map(|_| [u32::MAX, crate::floating::guard::UNWRITTEN])
        .collect()
}

impl Report {
    /// `observed` holds `(result, guard flag)` word pairs, one per case.
    /// Rejects wrong word counts and non-binary guard words before admission.
    pub fn evaluate(observed: &[u32]) -> Result<Self, Error> {
        if observed.len() != CASES.len() * 2 {
            return Err(Error::WordCount {
                expected: CASES.len() * 2,
                actual: observed.len(),
            });
        }
        let results: Vec<u32> = observed.iter().step_by(2).copied().collect();
        let flags: Vec<bool> = observed
            .iter()
            .skip(1)
            .step_by(2)
            .map(|&w| crate::floating::guard::decode(w).map_err(Error::Guard))
            .collect::<Result<_, _>>()?;
        let mismatches = CASES
            .iter()
            .zip(&results)
            .enumerate()
            .filter(|(_, (case, word))| case.expected != **word)
            .map(|(index, _)| index)
            .collect();
        Ok(Self {
            observed: results,
            flags,
            mismatches,
        })
    }
    pub fn observed(&self) -> &[u32] {
        &self.observed
    }
    /// The device guard flag per case; subtraction shares the add guard and
    /// explicit FMA conservatively requests exact replay on Metal.
    pub fn flags(&self) -> &[bool] {
        &self.flags
    }
    pub fn mismatches(&self) -> &[usize] {
        &self.mismatches
    }
    /// D1 admission: every strict-class case is exact.
    pub fn admits_strict(&self) -> bool {
        self.mismatches.iter().all(|&i| !CASES[i].op.is_strict())
    }
    /// Strict-with-fallback admission for a plan using `ops`: every case of
    /// those operations is exact or raised the device guard flag. A device
    /// that admits strict work passes trivially; a mismatch the guard misses,
    /// or a mismatch in an unguarded operation the plan uses, refuses it.
    pub fn admits_strict_with_fallback(&self, ops: &[Op]) -> bool {
        ops.iter().all(|op| {
            matches!(
                op,
                Op::Mul | Op::Add | Op::Sub | Op::Fma | Op::Accumulate | Op::FromU32 | Op::FromI32
            )
        }) && self
            .mismatches
            .iter()
            .all(|&i| !ops.contains(&CASES[i].op) || self.flags[i])
    }
    pub fn div_sqrt_exact(&self) -> bool {
        self.mismatches.iter().all(|&i| CASES[i].op.is_strict())
    }
    /// Some subnormal-dependent case failed (flush to zero or DAZ).
    pub fn flushes_subnormals(&self) -> bool {
        self.mismatches.iter().any(|&i| CASES[i].subnormal)
    }
    /// The accumulate case diverged: the compiler contracted `c + a * b`.
    pub fn contracts_implicitly(&self) -> bool {
        self.mismatches
            .iter()
            .any(|&i| CASES[i].op == Op::Accumulate && !CASES[i].subnormal)
    }
    /// Stable one-line verdict for diagnostics and refusals.
    pub fn summary(&self) -> String {
        let strict = self
            .mismatches
            .iter()
            .filter(|&&i| CASES[i].op.is_strict())
            .count();
        format!(
            "invar-f32-probe-v{VERSION}: cases={} strict_mismatches={strict} div_sqrt_mismatches={} flushes_subnormals={} implicit_contraction={} admits_strict={} admits_contraction_with_fallback={}",
            CASES.len(),
            self.mismatches.len() - strict,
            self.flushes_subnormals(),
            self.contracts_implicitly(),
            self.admits_strict(),
            self.admits_strict_with_fallback(CONTRACTION_OPS)
        )
    }
    /// One line per mismatching case, for evidence logs.
    pub fn details(&self) -> Vec<String> {
        self.mismatches
            .iter()
            .map(|&i| {
                let c = &CASES[i];
                format!(
                    "case {i} {:?} {}: a={:#010x} b={:#010x} c={:#010x} expected={:#010x} observed={:#010x} guard_flag={}",
                    c.op, c.what, c.a, c.b, c.c, c.expected, self.observed[i], self.flags[i]
                )
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Independent derivation with host binary32 arithmetic (IEEE RNE with
    /// gradual underflow on the supported x86_64/aarch64 hosts; Rust never
    /// enables FTZ/DAZ). `mul_add` is the correctly rounded fused operation.
    fn host(case: &Case) -> u32 {
        let (a, b, c) = (
            f32::from_bits(case.a),
            f32::from_bits(case.b),
            f32::from_bits(case.c),
        );
        let flag = |v: bool| u32::from(v);
        match case.op {
            Op::Mul => (a * b).to_bits(),
            Op::Add => (a + b).to_bits(),
            Op::Sub => (a - b).to_bits(),
            Op::Fma => a.mul_add(b, c).to_bits(),
            Op::Accumulate => {
                let product = std::hint::black_box(a * b);
                (c + product).to_bits()
            }
            Op::Lt => flag(a < b),
            Op::Eq => flag(a == b),
            Op::FromU32 => (case.a as f32).to_bits(),
            Op::FromI32 => (case.a as i32 as f32).to_bits(),
            Op::TruncI32 => {
                assert!(a.abs() < 2_147_483_648.0, "probe trunc stays in range");
                a as i32 as u32
            }
            Op::Div => (a / b).to_bits(),
            Op::Sqrt => a.sqrt().to_bits(),
        }
    }

    #[test]
    fn issue_m4_01_probe_table_matches_host_ieee_binary32() {
        assert_eq!(f32::from_bits(1) * 1.0, f32::from_bits(1), "host flushes");
        for (index, case) in CASES.iter().enumerate() {
            assert_eq!(host(case), case.expected, "case {index} {}", case.what);
            assert!(!case.what.is_empty());
        }
        let words = words();
        assert_eq!(words.len(), CASES.len() * 4);
        assert!(Op::ALL.iter().all(|op| CASES.iter().any(|c| c.op == *op)));
    }

    /// Pair each result word with a guard flag word.
    fn pairs(results: &[u32], flags: &[bool]) -> Vec<u32> {
        results
            .iter()
            .zip(flags)
            .flat_map(|(&r, &f)| [r, u32::from(f)])
            .collect()
    }

    #[test]
    fn issue_m4_01_probe_rejects_untouched_sentinel_buffer() {
        let words = vec![u32::MAX; CASES.len() * 2];
        assert_eq!(
            Report::evaluate(&words),
            Err(Error::Guard(crate::floating::guard::InvalidWord(u32::MAX)))
        );
    }

    #[test]
    fn issue_m4_01_probe_rejects_single_untouched_pair() {
        // Even one guarded contraction case must refuse instead of admitting
        // its unwritten result as an underflow that the CPU could repair.
        for at in 0..CASES.len() {
            let mut words = pairs(
                &CASES.iter().map(|c| c.expected).collect::<Vec<_>>(),
                &vec![false; CASES.len()],
            );
            words[2 * at..2 * at + 2].fill(u32::MAX);
            assert_eq!(
                Report::evaluate(&words),
                Err(Error::Guard(crate::floating::guard::InvalidWord(u32::MAX))),
                "case {at}"
            );
        }
    }

    #[test]
    fn issue_m4_01_probe_rejects_guard_word_two() {
        for at in 0..CASES.len() {
            let mut words = pairs(
                &CASES.iter().map(|c| c.expected).collect::<Vec<_>>(),
                &vec![false; CASES.len()],
            );
            words[2 * at + 1] = 2;
            assert_eq!(
                Report::evaluate(&words),
                Err(Error::Guard(crate::floating::guard::InvalidWord(2))),
                "case {at}"
            );
        }
        assert_eq!(crate::floating::guard::decode(0), Ok(false));
        assert_eq!(crate::floating::guard::decode(1), Ok(true));
        assert_eq!(
            crate::floating::guard::decode(2),
            Err(crate::floating::guard::InvalidWord(2))
        );
        assert_eq!(
            crate::floating::guard::decode(u32::MAX),
            Err(crate::floating::guard::InvalidWord(u32::MAX))
        );
    }

    #[test]
    fn issue_m4_01_probe_verdicts_classify_flush_contraction_and_div_sqrt() {
        let exact: Vec<u32> = CASES.iter().map(|c| c.expected).collect();
        let clear = vec![false; CASES.len()];
        let report = Report::evaluate(&pairs(&exact, &clear)).unwrap();
        assert!(report.admits_strict() && report.div_sqrt_exact());
        assert!(report.admits_strict_with_fallback(CONTRACTION_OPS));
        assert!(!report.flushes_subnormals() && !report.contracts_implicitly());
        assert!(Report::evaluate(&exact).is_err());
        // A flushing device: every subnormal-dependent strict case reads 0.
        let flushed: Vec<u32> = CASES
            .iter()
            .map(|c| if c.subnormal { 0 } else { c.expected })
            .collect();
        let report = Report::evaluate(&pairs(&flushed, &clear)).unwrap();
        assert!(!report.admits_strict() && report.flushes_subnormals());
        // Without guard flags, those mismatches refuse the fallback mode too.
        assert!(!report.admits_strict_with_fallback(CONTRACTION_OPS));
        assert!(report.summary().contains("admits_strict=false"));
        assert!(!report.details().is_empty());
        // Flags on exactly the guarded mismatches admit the contraction ops,
        // but never an unguarded operation such as `fma` or `<`.
        let flagged: Vec<bool> = CASES
            .iter()
            .map(|c| c.subnormal && CONTRACTION_OPS.contains(&c.op))
            .collect();
        let report = Report::evaluate(&pairs(&flushed, &flagged)).unwrap();
        assert!(report.admits_strict_with_fallback(CONTRACTION_OPS));
        assert!(!report.admits_strict_with_fallback(&[Op::Mul, Op::Fma]));
        assert!(!report.admits_strict_with_fallback(&[Op::Lt]));
        let mut missed = flagged.clone();
        let at = CASES
            .iter()
            .position(|c| c.op == Op::Mul && c.subnormal)
            .unwrap();
        missed[at] = false;
        let report = Report::evaluate(&pairs(&flushed, &missed)).unwrap();
        assert!(!report.admits_strict_with_fallback(CONTRACTION_OPS));
        // Implicit FMA in the accumulate shape.
        let mut fused = exact.clone();
        let at = CASES
            .iter()
            .position(|c| c.op == Op::Accumulate && !c.subnormal)
            .unwrap();
        fused[at] = 0x3380_0000;
        let report = Report::evaluate(&pairs(&fused, &clear)).unwrap();
        assert!(report.contracts_implicitly() && !report.admits_strict());
        assert!(!report.admits_strict_with_fallback(CONTRACTION_OPS));
        // Inexact division alone keeps strict admission.
        let mut div = exact;
        let at = CASES.iter().position(|c| c.op == Op::Div).unwrap();
        div[at] ^= 1;
        let report = Report::evaluate(&pairs(&div, &clear)).unwrap();
        assert!(report.admits_strict() && !report.div_sqrt_exact());
    }

    /// Device models. Every model reads subnormal inputs as signed zero (DAZ);
    /// they differ in what happens to a result below FLT_MIN.
    #[derive(Clone, Copy, Debug)]
    enum Model {
        /// Round to nearest even, then flush a result below FLT_MIN.
        RoundThenFlush,
        /// Flush whenever the exact value is below FLT_MIN, including results
        /// IEEE rounds up to FLT_MIN (the Apple M4, probe case 2).
        FlushExact,
        /// Keep gradual underflow in results: a subnormal result equals IEEE
        /// but reads as zero when a later operation consumes it.
        DazOnly,
    }
    const MODELS: [Model; 3] = [Model::RoundThenFlush, Model::FlushExact, Model::DazOnly];

    /// DAZ, decided independently of `guard::operand`: a mutated predicate
    /// must not also switch off the flushing it is meant to detect.
    fn daz(w: u32) -> f32 {
        if w & 0x7f80_0000 == 0 {
            f32::from_bits(w & 0x8000_0000)
        } else {
            f32::from_bits(w)
        }
    }
    fn flush(exact: f64, rounded: f32, model: Model) -> u32 {
        let tiny = match model {
            Model::RoundThenFlush => rounded != 0.0 && rounded.abs() < f32::MIN_POSITIVE,
            Model::FlushExact => exact != 0.0 && exact.abs() < f64::from(f32::MIN_POSITIVE),
            Model::DazOnly => false,
        };
        if tiny {
            rounded.to_bits() & 0x8000_0000
        } else {
            rounded.to_bits()
        }
    }
    fn device_mul(a: u32, b: u32, model: Model) -> u32 {
        let (x, y) = (daz(a), daz(b));
        flush(f64::from(x) * f64::from(y), x * y, model)
    }
    fn device_add(a: u32, b: u32, model: Model) -> u32 {
        let (x, y) = (daz(a), daz(b));
        flush(f64::from(x) + f64::from(y), x + y, model)
    }
    /// strict-f32-v1 compares NaN results by class only.
    fn same(x: u32, y: u32) -> bool {
        x == y || (f32::from_bits(x).is_nan() && f32::from_bits(y).is_nan())
    }

    /// Operand words concentrated where gradual underflow matters, plus
    /// ordinary values, and zeros and infinities of both signs (a subnormal
    /// times an infinity is where only the operand rule catches DAZ).
    fn operand_word(seed: &mut u64) -> u32 {
        *seed ^= *seed << 13;
        *seed ^= *seed >> 7;
        *seed ^= *seed << 17;
        let r = *seed;
        let sign = ((r >> 63) as u32) << 31;
        let mantissa = (r >> 8) as u32 & 0x007f_ffff;
        if r % 16 == 6 {
            return sign | 0x7f80_0000; // signed infinity
        }
        let exponent = match r % 8 {
            0 => 0,                          // zero or subnormal
            1 => 1,                          // FLT_MIN binade
            2 => 1 + (r >> 40) as u32 % 24,  // just above FLT_MIN
            3 => 64 + (r >> 40) as u32 % 64, // products land near FLT_MIN
            4 => 127,                        // around one
            5 => (r >> 40) as u32 % 254 + 1, // any normal
            6 => return sign,                // signed zero
            _ => 1 + (r >> 40) as u32 % 3,
        };
        sign | exponent << 23
            | if exponent == 0 && mantissa == 0 {
                1
            } else {
                mantissa
            }
    }

    /// The guard is sound per operation: on every model, each product or sum
    /// that differs from IEEE binary32 with gradual underflow is flagged.
    #[test]
    fn issue_m4_01_guard_flags_every_flushing_divergence() {
        let mut seed = 0x9e37_79b9_7f4a_7c15_u64;
        let (mut diverged, mut flagged_ok) = (0usize, 0usize);
        for _ in 0..400_000 {
            let (a, b) = (operand_word(&mut seed), operand_word(&mut seed));
            let (x, y) = (f32::from_bits(a), f32::from_bits(b));
            for model in MODELS {
                let input =
                    crate::floating::guard::operand(a) || crate::floating::guard::operand(b);
                let p = device_mul(a, b, model);
                let flag = input || crate::floating::guard::mul(a, b, p);
                if !same(p, (x * y).to_bits()) {
                    diverged += 1;
                    assert!(flag, "{model:?} mul {a:#010x} {b:#010x} device {p:#010x}");
                } else if !flag {
                    flagged_ok += 1;
                }
                let s = device_add(a, b, model);
                let flag = input || crate::floating::guard::add(a, b, s);
                if !same(s, (x + y).to_bits()) {
                    diverged += 1;
                    assert!(flag, "{model:?} add {a:#010x} {b:#010x} device {s:#010x}");
                }
            }
        }
        // The corpus exercises real divergence and leaves most work unflagged.
        assert!(diverged > 10_000, "{diverged}");
        assert!(flagged_ok > 100_000, "{flagged_ok}");
        // Exact cancellation and signed zeros stay unflagged.
        assert!(!crate::floating::guard::add(0x3fc0_0000, 0xbfc0_0000, 0));
        assert!(!crate::floating::guard::add(
            0x8000_0000,
            0x8000_0000,
            0x8000_0000
        ));
        assert!(!crate::floating::guard::mul(
            0x8000_0000,
            0x3f80_0000,
            0x8000_0000
        ));
        // A nonzero sum that the device flushed is flagged.
        assert!(crate::floating::guard::add(0x0080_0001, 0x8080_0000, 0));
    }

    /// The emitted loop on a model: ordered products and sums with the same
    /// predicates as the kernel, returning the output word and dispatch flag.
    fn device_dot(a: &[u32], b: &[u32], model: Model) -> (u32, bool) {
        use crate::floating::guard;
        let (mut sum, mut flag) = (0u32, false);
        for (&x, &y) in a.iter().zip(b) {
            let p = device_mul(x, y, model);
            let next = device_add(sum, p, model);
            flag |= guard::operand(x)
                || guard::operand(y)
                || guard::mul(x, y, p)
                || guard::add(sum, p, next);
            sum = next;
        }
        (sum, flag)
    }

    /// The guard is sound over the whole loop, by the induction in the
    /// evidence README: an unflagged dispatch equals the IEEE dot product.
    /// `DazOnly` needs the nonzero branch of the sum rule: a subnormal partial
    /// sum equals IEEE where it is produced and diverges only when consumed.
    #[test]
    fn issue_m4_01_guard_flags_every_divergent_dot_product() {
        let mut seed = 0x2545_f491_4f6c_dd1d_u64;
        let (mut diverged, mut clean) = (0usize, 0usize);
        for _ in 0..150_000 {
            let a: [u32; 3] = std::array::from_fn(|_| operand_word(&mut seed));
            let b: [u32; 3] = std::array::from_fn(|_| operand_word(&mut seed));
            let ieee = a.iter().zip(&b).fold(0.0f32, |s, (&x, &y)| {
                s + f32::from_bits(x) * f32::from_bits(y)
            });
            for model in MODELS {
                let (out, flag) = device_dot(&a, &b, model);
                if !same(out, ieee.to_bits()) {
                    diverged += 1;
                    assert!(flag, "{model:?} dot {a:08x?} {b:08x?} device {out:#010x}");
                } else if !flag {
                    clean += 1;
                }
            }
        }
        assert!(diverged > 10_000, "{diverged}");
        assert!(clean > 50_000, "{clean}");
    }

    #[test]
    fn issue_m4_01_probe_kernels_bind_pinned_options_and_every_op() {
        let msl = emit_msl();
        assert!(msl.contains("#pragma METAL fp contract(off)"));
        assert!(msl.contains(KERNEL_NAME));
        for code in 0..Op::ALL.len() {
            assert!(msl.contains(&format!("case {code}u:")));
        }
        let caps = Capabilities::cuda_12_9_sm89();
        let cuda = emit_cuda(caps).unwrap();
        for option in crate::floating::cuda_strict_options(caps).unwrap() {
            assert!(cuda.contains(&option));
        }
        assert!(cuda.contains(&format!("caps={caps:?}")));
        let metal = Capabilities::Metal {
            apple_family: 9,
            msl_major: 3,
            msl_minor: 1,
        };
        assert!(emit_cuda(metal).is_none());
    }
}
