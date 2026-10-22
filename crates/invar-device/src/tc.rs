//! `tc-f16-f32-v1` (D5 as amended on 2026-10-14): explicit Tensor
//! Core and `simdgroup_matrix` matrix contraction.
//!
//! Inputs are rounded to binary16 (nearest even; a value that rounds past the
//! largest finite binary16, an infinity or a NaN is refused before any work).
//! Binary16 subnormals are admitted. The sm_120 Tensor Core aligns products
//! before normalizing a subnormal operand, so the CUDA schedule feeds the
//! `mma` only zero and normal binary16 operands and adds every product with a
//! subnormal operand on CUDA cores (second D5 amendment). Every product of
//! two binary16 values is exact in binary32. The device may
//! accumulate in any order; a result `r` is admissible when
//! `|r - exact| <= 2 * K * 2^-23 * sum_k |a_ik * b_kj|`, where `exact` is the
//! exact sum of the rounded products. The reference result is that exact sum
//! rounded once to nearest even. Exactness: every binary16 value is a multiple
//! of 2^-24 below 2^16, so each product is an integer multiple of 2^-48 below
//! 2^80 and `MAX_INNER` (2^20) of them sum below 2^100 in an `i128`.

use crate::Capabilities;
use crate::contraction::{self, atoms};
use crate::floating::{Error, Verified};
use crate::gemm::Geometry;

mod reference;
#[cfg(test)]
use reference::round_units48;
pub use reference::{
    Failure, InputRange, MAX_INNER, Report, f16_round, reference, round_inputs, verify,
};

pub const CUDA_SCHEDULE: &str = "invar-tc-f16-f32-mma-m16n8k16-v1";
pub const METAL_SCHEDULE: &str = "invar-tc-f16-f32-simdgroup8x8-v1";
/// CUDA: one warp per 16x8 output tile (`mma.sync` m16n8k16), 4 warps a block.
pub const CUDA_TILE: [u32; 3] = [16, 8, 16];
pub const CUDA_THREADS: u32 = 128;
/// Metal: one simdgroup per 8x8 output tile (`simdgroup_float8x8`).
pub const METAL_TILE: [u32; 3] = [8, 8, 8];
pub const METAL_THREADS: u32 = 32;
/// Groups are tile-strided, so this cap never changes which cells are written.
pub const MAX_GROUPS: u32 = 65_535;

/// Launch geometry of a verified `tc-f16-f32-v1` plan for its target.
pub fn geometry(plan: &Verified) -> Geometry {
    let [rows, _, cols] = plan.ir().dimensions;
    let (tile, per_group, threads) = match plan.capabilities() {
        Capabilities::Cuda { .. } => (CUDA_TILE, 4, CUDA_THREADS),
        Capabilities::Metal { .. } => (METAL_TILE, 1, METAL_THREADS),
    };
    let tiles = u64::from(rows.div_ceil(tile[0])) * u64::from(cols.div_ceil(tile[1]));
    Geometry {
        tiles,
        groups: u32::try_from(tiles.div_ceil(per_group).min(u64::from(MAX_GROUPS)))
            .unwrap_or(MAX_GROUPS),
        threads,
    }
}

fn admitted(plan: &Verified) -> Result<(), Error> {
    if plan.is_tc() {
        Ok(())
    } else {
        Err(Error::Numeric)
    }
}

/// CUDA C++ with inline-PTX `mma.sync` m16n8k16 (f16 inputs, f32 accumulator)
/// for a `tc-f16-f32-v1` plan. Inputs must already be binary16 values
/// (`round_inputs`), so the in-kernel `cvt.rn.f16x2.f32` is exact. The `mma`
/// sees binary16 subnormal operands as zero; a k-tile whose warp vote finds
/// one adds those exact products on CUDA cores in increasing k, and each cell
/// is the Tensor Core sum plus that correction, one nearest addition (second
/// D5 amendment). A zero inner extent omits the k loop and its helpers (cells
/// stay +0).
pub fn emit_cuda(plan: &Verified) -> Result<String, Error> {
    admitted(plan)?;
    let target = contraction::builtin_target(plan)?;
    atoms::emit_cuda(plan, &atoms::tc_cuda_v1(), target).map_err(contraction::Error::into_plan)
}

/// MSL 3.1 with `simdgroup_float8x8` multiply-accumulate over binary16-valued
/// words for a `tc-f16-f32-v1` plan: buffers 0/1/2 = A/B/C words, 3 = the
/// zeroed dispatch flag, raised only by a launch or bounds violation.
pub fn emit_msl(plan: &Verified) -> Result<String, Error> {
    admitted(plan)?;
    let target = contraction::builtin_target(plan)?;
    atoms::emit_msl(plan, &atoms::tc_metal_v1(), target).map_err(contraction::Error::into_plan)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Independent binary16 decoder: every finite binary16 value as f64.
    fn f16_values() -> Vec<f64> {
        let mut values: Vec<f64> = (0u32..0x7c00)
            .map(|h| {
                let e = (h >> 10) as i32;
                let m = f64::from(h & 0x3ff);
                if e == 0 {
                    m * 2f64.powi(-24)
                } else {
                    (1024.0 + m) * 2f64.powi(e - 25)
                }
            })
            .collect();
        values.dedup();
        values
    }

    /// Nearest-even binary16 by search over the decoded table.
    fn oracle(values: &[f64], word: u32) -> Option<u32> {
        let x = f32::from_bits(word);
        if !x.is_finite() {
            return None;
        }
        let v = f64::from(x).abs();
        if v >= 65520.0 {
            return None;
        }
        let i = values.partition_point(|&h| h < v);
        let pick = if i == values.len() {
            values[i - 1]
        } else if values[i] == v || i == 0 {
            values[i]
        } else {
            let (lo, hi) = (values[i - 1], values[i]);
            match (v - lo).partial_cmp(&(hi - v)).unwrap() {
                std::cmp::Ordering::Less => lo,
                std::cmp::Ordering::Greater => hi,
                // The even neighbour is the one with an even table index.
                std::cmp::Ordering::Equal if (i - 1) % 2 == 0 => lo,
                std::cmp::Ordering::Equal => hi,
            }
        };
        let r = pick as f32;
        Some(if x.is_sign_negative() { -r } else { r }.to_bits())
    }

    #[test]
    fn f16_rounding_matches_an_independent_table_search() {
        let values = f16_values();
        let mut words: Vec<u32> = (0..=u32::MAX).step_by(65_521).collect();
        for h in &values {
            let w = (*h as f32).to_bits();
            for d in [0u32, 1, 2] {
                words.extend([w.wrapping_add(d), w.wrapping_sub(d), w | 0x8000_0000]);
            }
        }
        // Exact midpoints between neighbours (ties) and the overflow edge.
        for pair in values.windows(2) {
            let mid = (pair[0] + pair[1]) / 2.0;
            words.push((mid as f32).to_bits());
        }
        words.extend([
            0x477f_e000, // 65504
            0x477f_efff, // just below 65520
            0x477f_f000, // 65520: ties to even 65536, overflow
            0x3300_0000, // 2^-25: tie between 0 and 2^-24, rounds to 0
            0x3300_0001, // just above: 2^-24
            0x7f80_0000,
            0xff80_0000,
            0x7fc0_0000,
        ]);
        for w in words {
            assert_eq!(f16_round(w), oracle(&values, w), "{w:#010x}");
        }
    }

    #[test]
    fn exact_rounding_of_wide_sums_matches_binary32_construction() {
        for (value, expected) in [
            (1i128 << 48, 0x3f80_0000),
            (-(1i128 << 48), 0xbf80_0000),
            ((1i128 << 48) + (1 << 24), 0x3f80_0000), // 1 + 2^-24 ties to even
            ((1i128 << 48) + (1 << 24) + 1, 0x3f80_0001), // just above the tie
            ((1i128 << 48) + 3 * (1 << 24), 0x3f80_0002), // 1 + 3*2^-24 ties up
            (1, 0x2780_0000),                         // 2^-48
        ] {
            assert_eq!(round_units48(value), expected, "{value}");
        }
        assert_eq!(round_units48(0), 0);
    }
}
