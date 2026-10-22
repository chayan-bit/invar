//! Tiled strict binary32 matrix contraction (M4-03) from the same verified
//! `floating` plan the direct M4-01 kernels use. One threadgroup owns a 64x64
//! output tile, stages 16-deep slices of A and B in shared memory and keeps a
//! 4x4 register tile per thread. Every output cell still adds its products one
//! at a time in increasing k with separately rounded multiply and add, so the
//! result is bit-identical to the source order; only independent cells run in
//! parallel (D6). The k loop stops at the exact inner extent, so masked tile
//! slots are loaded as zero but never accumulated (a `-0` sum stays `-0`).
//! Edge rows/columns are masked on load and store; every physical index is
//! still checked against its verified allocation.
//!
//! The kernel text comes from the generic contraction emitter (#61): the
//! `contraction::gemm_tiled_v1` move sequence, verified against the target
//! file the plan's live capabilities select.

use crate::contraction::{self, gemm_tiled_v1};
use crate::floating::{Error, Verified, strict_only};

/// Schedule identity, embedded in the emitted source and cache key.
pub const SCHEDULE: &str = "invar-f32-gemm-tiled-v1";
/// Output tile edge, k slice depth and threads per group (16x16, 4x4 each).
pub const TILE: u32 = 64;
pub const TILE_K: u32 = 16;
pub const THREADS: u32 = 256;
/// Groups are tile-strided, so this cap never changes which cells are written.
pub const MAX_GROUPS: u32 = 65_535;
/// Below this many multiply-adds the device launch is not worth its fixed cost.
pub const MIN_DEVICE_MULTIPLY_ADDS: u64 = 65_536;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Geometry {
    pub tiles: u64,
    pub groups: u32,
    pub threads: u32,
}

/// Launch geometry for a verified plan: one-dimensional groups of `THREADS`.
pub fn geometry(plan: &Verified) -> Geometry {
    let [rows, _, cols] = plan.ir().dimensions;
    let tiles = u64::from(rows.div_ceil(TILE)) * u64::from(cols.div_ceil(TILE));
    Geometry {
        tiles,
        groups: u32::try_from(tiles.min(u64::from(MAX_GROUPS))).unwrap_or(MAX_GROUPS),
        threads: THREADS,
    }
}

/// Why a shape stays on the CPU instead of the tiled device schedule, if it
/// does. This is a placement choice only; the CPU result is the same contract.
pub fn cpu_fallback([rows, inner, cols]: [u32; 3]) -> Option<&'static str> {
    if rows == 0 || cols == 0 {
        return Some("empty-output");
    }
    if inner == 0 {
        return Some("zero-inner-extent");
    }
    let work = u64::from(rows) * u64::from(inner) * u64::from(cols);
    (work < MIN_DEVICE_MULTIPLY_ADDS).then_some("below-device-work-threshold")
}

/// CUDA C++ for the tiled schedule. The header is the strict contraction
/// header, so the runtime's option and capability identity checks apply
/// unchanged; launch with `geometry(plan)` only.
pub fn emit_cuda(plan: &Verified) -> Result<String, Error> {
    strict_only(plan)?;
    let target = contraction::builtin_target(plan)?;
    contraction::emit_cuda(plan, &gemm_tiled_v1(), target).map_err(contraction::Error::into_plan)
}

/// MSL 3.1 for the tiled schedule, with the strict pragma and the shared
/// underflow guard: buffers 0/1/2 = A/B/C words, 3 = the zeroed dispatch flag.
/// A raised flag means the dispatch must be discarded and run on the CPU.
pub fn emit_msl(plan: &Verified) -> Result<String, Error> {
    strict_only(plan)?;
    let target = contraction::builtin_target(plan)?;
    contraction::emit_msl(plan, &gemm_tiled_v1(), target).map_err(contraction::Error::into_plan)
}
