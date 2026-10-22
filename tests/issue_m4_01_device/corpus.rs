//! Shared M4-01 device corpus: ordinary source programs, the independent
//! `F32Bits` oracle and binary32 word patterns. Used by the CUDA and Metal
//! hardware suites so both targets qualify against the same cases.
#![allow(dead_code)]

use invar::{check, eval, front, numeric_contract::F32Bits};

pub const MAT: &str = "fn mat(a: read Array<f32>, b: read Array<f32>, rows: u32, inner: u32, cols: u32) -> Array<f32> { var c = filled_f32(rows * cols,0.0); for i in 0u32..rows { for j in 0u32..cols { var sum: f32 = 0.0; for k in 0u32..inner { sum = sum + a[i * inner + k] * b[k * cols + j]; } c[i * cols + j] = sum; } } return c; }";

pub const FUEL: u64 = 10_000_000;

/// Zero, one, odd, rectangular and skinny axes.
pub const SHAPES: [[u32; 3]; 9] = [
    [0, 3, 2],
    [2, 0, 3],
    [3, 4, 0],
    [1, 1, 1],
    [3, 5, 3],
    [7, 1, 9],
    [1, 64, 1],
    [17, 13, 33],
    [31, 33, 2],
];

/// `pick` cycles through 2^-149, -0, ±1.5 (cancellation), 3 * 2^-149, 0.5,
/// 2^-126 and -2.25. `main` recomputes every cell in source order and compares
/// value and zero sign; it returns 7 only when every cell agrees.
pub fn program([rows, inner, cols]: [u32; 3]) -> String {
    format!(
        "{MAT} fn pick(q: u32, tiny: f32) -> f32 {{ let m = q % 8u32; if m == 0u32 {{ return tiny; }} if m == 1u32 {{ return -0.0; }} if m == 2u32 {{ return 1.5; }} if m == 3u32 {{ return -1.5; }} if m == 4u32 {{ return tiny * 3.0; }} if m == 5u32 {{ return 0.5; }} if m == 6u32 {{ return tiny * 8388608.0; }} return -2.25; }} \
         fn main() -> i32 {{ var tiny: f32 = 1.0; for s in 0u32..149u32 {{ tiny = tiny * 0.5; }} \
         var a = filled_f32({rows}u32 * {inner}u32, 0.0); for q in 0u32..len(a) {{ a[q] = pick(q * 3u32 + 1u32, tiny); }} \
         var b = filled_f32({inner}u32 * {cols}u32, 0.0); for p in 0u32..len(b) {{ b[p] = pick(p * 5u32 + 2u32, tiny); }} \
         let c = mat(a, b, {rows}u32, {inner}u32, {cols}u32); \
         for i in 0u32..{rows}u32 {{ for j in 0u32..{cols}u32 {{ var e: f32 = 0.0; for k in 0u32..{inner}u32 {{ e = e + a[i * {inner}u32 + k] * b[k * {cols}u32 + j]; }} let got = c[i * {cols}u32 + j]; if got != e or (1.0 / got) != (1.0 / e) {{ return 1; }} }} }} \
         return 7; }}"
    )
}

pub fn checked(source: &str) -> check::CheckedProgram {
    let program = check::check(front::parse(source.as_bytes()).unwrap(), true).unwrap();
    invar::ownership::check(&program).unwrap();
    program
}

pub fn reference(source: &str) -> eval::Observation {
    eval::observe(&checked(source), FUEL)
}

pub fn assert_parity(actual: &eval::Observation, expected: &eval::Observation) {
    assert_eq!(actual.result, expected.result);
    assert_eq!(actual.steps_used, expected.steps_used);
    assert_eq!(actual.cleanup_trace, expected.cleanup_trace);
}

/// Strict ordered reference: separate RN multiply and add, source order.
pub fn oracle([rows, inner, cols]: [u32; 3], a: &[u32], b: &[u32]) -> Vec<u32> {
    let [rows, inner, cols] = [rows, inner, cols].map(|n| n as usize);
    let mut out = Vec::with_capacity(rows * cols);
    for i in 0..rows {
        for j in 0..cols {
            let mut sum = F32Bits::POS_ZERO;
            for k in 0..inner {
                let product =
                    F32Bits::from_bits(a[i * inner + k]).mul(F32Bits::from_bits(b[k * cols + j]));
                sum = sum.add(product);
            }
            out.push(sum.to_bits());
        }
    }
    out
}

/// Normal-range words whose products and partial sums stay normal or exactly
/// zero, so a flushing device cannot be distinguished on them.
pub const NORMAL: [u32; 8] = [
    0x3f80_0000, // 1
    0xbfc0_0000, // -1.5
    0x3fc0_0000, // 1.5
    0x3f00_0000, // 0.5
    0x8000_0000, // -0
    0x4010_0000, // 2.25
    0x3f80_0001, // 1 + 2^-23 (rounding)
    0xc040_0000, // -3
];

/// Words that exercise gradual underflow: subnormal inputs and subnormal results.
pub const SUBNORMAL: [u32; 8] = [
    0x0000_0001, // 2^-149
    0x3f80_0000, // 1
    0x0080_0000, // 2^-126
    0x3f00_0000, // 0.5
    0x8000_0003, // -3 * 2^-149
    0x3fc0_0000, // 1.5
    0x007f_ffff, // largest subnormal
    0x8000_0000, // -0
];

pub fn words(pattern: &[u32], len: usize, stride: usize) -> Vec<u32> {
    (0..len)
        .map(|i| pattern[(i * stride + 1) % pattern.len()])
        .collect()
}

pub fn is_subnormal(word: u32) -> bool {
    word & 0x7f80_0000 == 0 && word & 0x007f_ffff != 0
}

/// The scheduler offers a loop only after its condition first holds, so a
/// zero-row contraction never reaches any device on any engine.
pub fn reached([rows, _, _]: [u32; 3]) -> bool {
    rows > 0
}

/// Intermediate underflow, normal final output: eight products of
/// 2^-64 * 2^-63 = 2^-127 are each subnormal, but their exact sum 2^-124 is
/// normal. A flushing device would publish +0; IEEE gives 2^-124.
pub const UNDERFLOW_A: u32 = 0x1f80_0000; // 2^-64
pub const UNDERFLOW_B: u32 = 0x2000_0000; // 2^-63
pub const UNDERFLOW_SUM: u32 = 0x0180_0000; // 2^-124

/// The same case as ordinary source, recomputed and sign-checked in `main`.
pub fn underflow_program() -> String {
    format!(
        "{MAT} fn main() -> i32 {{ var x: f32 = 1.0; for s in 0u32..64u32 {{ x = x * 0.5; }} let y = x * 2.0; \
         let a = filled_f32(8u32, x); let b = filled_f32(8u32, y); let c = mat(a, b, 1u32, 8u32, 1u32); \
         var e: f32 = 0.0; for k in 0u32..8u32 {{ e = e + a[k] * b[k]; }} \
         if c[0u32] != e or (1.0 / c[0u32]) != (1.0 / e) or c[0u32] == 0.0 {{ return 1; }} return 7; }}"
    )
}
