#![allow(dead_code)]
use coremath::records;
use invar::{
    math::{MathFn, eval_f32},
    numeric_contract::F32Bits,
};
use std::path::Path;

#[path = "coremath.rs"]
pub mod coremath;

pub const RANDOM: u64 = 1_000_000;

pub const STRIDE: u64 = 256;

pub const CHUNK: usize = 1 << 20;

/// The leading coefficient of sinf's `a` table, 0x1.921fb54442d17p-3, with
/// bit 40 flipped (a relative change near 2^-12).
pub const MUTATION: (&str, &str) = (
    "D::raw(0x3fc921fb54442d17ul)",
    "D::raw(0x3fc920fb54442d17ul)",
);

pub fn oracle(f: MathFn, x: u64, y: u64) -> u64 {
    let args = [F32Bits::from_bits(x as u32), F32Bits::from_bits(y as u32)];
    u64::from(eval_f32(f, &args[..f.arity()]).to_bits())
}

/// Binary32 specials plus `n` pseudo-random encodings, as (x, y) columns.
pub fn samples(n: usize) -> (Vec<u64>, Vec<u64>) {
    let specials: [f32; 14] = [
        0.0,
        -0.0,
        0.5,
        -0.5,
        1.5,
        -2.5,
        3.0,
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::NAN,
        1e-40,
        -1e-45,
        f32::MAX,
        8388609.0,
    ];
    let mut x: Vec<u64> = specials.iter().map(|v| u64::from(v.to_bits())).collect();
    let mut state = 0x9e37_79b9_7f4a_7c15u64;
    for _ in 0..n {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        x.push(state >> 32);
    }
    let y = x.iter().rev().copied().collect();
    (x, y)
}

/// (inputs, mismatches, first mismatch) of `run` against the upstream
/// driver's records for binary32 `f`.
pub fn against_upstream(
    driver: &Path,
    f: MathFn,
    random: u64,
    mut run: impl FnMut(&[u64], &[u64]) -> Vec<u64>,
) -> (u64, u64, String) {
    let (mut total, mut wrong, mut first) = (0u64, 0u64, String::new());
    records(driver, f, false, random, STRIDE, CHUNK, |batch| {
        let x: Vec<u64> = batch.iter().map(|r| r[0]).collect();
        let y: Vec<u64> = batch.iter().map(|r| r[1]).collect();
        for (record, got) in batch.iter().zip(run(&x, &y)) {
            total += 1;
            if !coremath::same(false, got, record[2]) {
                wrong += 1;
                if first.is_empty() {
                    first = format!(
                        "x={:#x} y={:#x} got={got:#x} upstream={:#x}",
                        record[0], record[1], record[2]
                    );
                }
            }
        }
    });
    (total, wrong, first)
}
