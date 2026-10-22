//! Reached `sum_tree`/`dot_tree`/`sum_exact`/`dot_exact` calls on CPU
//! workers, with run receipts shared by the device engines. Trees split on
//! aligned power-of-two leaf chunks and exact sums merge integer partials,
//! so the bits never depend on the worker count.

use std::{ops::Range, thread};

use crate::{
    front::Span,
    fsum::{self, Kind},
    mir_eval::{ReduceCall, serial_reduction},
    numeric_contract::F32Bits,
    plan::{DeviceCost, DeviceCostModel},
};
use invar_device::fsum::Op;

use super::Error;

/// One reached reduction and where it ran.
#[derive(Clone, Debug, PartialEq)]
pub struct Reduction {
    pub function: String,
    pub span: Span,
    pub kind: Kind,
    /// Calibration family id `fsum.<op>.<type>` (ruling C14).
    pub family: &'static str,
    pub elements: usize,
    /// `cpu-parallel`, `cpu-serial`, `metal`, `cuda` or `mir-serial`.
    pub backend: &'static str,
    pub reason: &'static str,
    pub workers: usize,
    pub cost: Option<DeviceCost>,
    /// Device time for one completed submission, when a device ran it.
    pub kernel_ms: Option<f64>,
}

impl Reduction {
    pub(crate) fn new(call: &ReduceCall<'_>, backend: &'static str, reason: &'static str) -> Self {
        Self {
            function: call.function.to_owned(),
            span: call.span,
            kind: call.kind,
            family: family_id(call.kind),
            elements: call.a.len(),
            backend,
            reason,
            workers: 0,
            cost: None,
            kernel_ms: None,
        }
    }
}

fn word(value: F32Bits) -> f32 {
    f32::from_bits(value.to_bits())
}

/// Run `body` for each item on its own scoped thread, in item order.
fn scoped<T: Send>(items: usize, body: impl Fn(usize) -> T + Sync) -> Result<Vec<T>, Error> {
    let body = &body;
    thread::scope(|scope| {
        let handles = (0..items)
            .map(|item| {
                thread::Builder::new()
                    .spawn_scoped(scope, move || body(item))
                    .map_err(|_| Error::Spawn)
            })
            .collect::<Result<Vec<_>, _>>()?;
        handles
            .into_iter()
            .map(|handle| handle.join().map_err(|_| Error::WorkerPanic))
            .collect()
    })
}

fn tree(call: &ReduceCall<'_>, workers: usize) -> Result<u32, Error> {
    let (a, b, n) = (call.a, call.b.unwrap_or(call.a), call.a.len());
    let term = |i: usize| match call.kind {
        Kind::DotTree => word(a[i]) * word(b[i]),
        _ => word(a[i]),
    };
    let leaves = fsum::leaf_count(n);
    let level = fsum::chunk_level(leaves, workers);
    let chunks = leaves.div_ceil(1 << level);
    let nodes = scoped(chunks, |chunk| fsum::tree_node(n, level, chunk, &term))?;
    Ok(fsum::canonical(fsum::fold_aligned(chunks, |chunk| {
        nodes[chunk]
    })))
}

fn ranges(n: usize, parts: usize) -> Vec<Range<usize>> {
    let width = n.div_ceil(parts.max(1)).max(1);
    (0..n)
        .step_by(width)
        .map(|start| start..(start + width).min(n))
        .collect()
}

fn exact(call: &ReduceCall<'_>, workers: usize) -> Result<u32, Error> {
    let (a, b, dot) = (call.a, call.b.unwrap_or(call.a), call.kind.is_dot());
    let parts = ranges(a.len(), workers);
    let partials = scoped(parts.len(), |part| {
        let mut exact = fsum::Exact::new(dot);
        for i in parts[part].clone() {
            if dot {
                exact.add_product(a[i].to_bits(), b[i].to_bits());
            } else {
                exact.add(a[i].to_bits());
            }
        }
        exact
    })?;
    let mut total = fsum::Exact::new(dot);
    partials.iter().for_each(|partial| total.merge(partial));
    Ok(total.finish())
}

/// Receipt backend when the CPU cost model keeps a call on one of several
/// requested workers (the frozen explain schema has no worker field).
pub const SERIAL_BACKEND: &str = "cpu-serial";

/// A CPU receipt with the workers the cost model picked from `workers`.
pub fn cpu_receipt(call: &ReduceCall<'_>, workers: usize, reason: &'static str) -> Reduction {
    let chosen = cpu_workers(call.kind, call.a.len(), workers);
    let backend = if chosen < workers {
        SERIAL_BACKEND
    } else {
        "cpu-parallel"
    };
    let mut receipt = Reduction::new(call, backend, reason);
    receipt.workers = chosen;
    receipt
}

/// Measured coefficients of one reduction family. The values are recorded,
/// with their raw measurements, under the family id in
/// `calibration/fsum-v1.json`; a test keeps the file and this table equal.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Calibration {
    /// One worker's cost per term, in nanoseconds.
    pub ns_per_term: f64,
    /// Starting and joining one scoped worker, in nanoseconds.
    pub spawn_ns_per_worker: f64,
    /// Smallest call `auto` may place on Metal; below it Metal loses to the
    /// CPU. `usize::MAX` (profile `null`): Metal never won when measured.
    pub metal_min_terms: usize,
}

pub fn calibration(kind: Kind) -> Calibration {
    let (ns_per_term, metal_min_terms) = match kind {
        Kind::SumTree => (SUM_TREE_NS, METAL_MIN_TREE),
        Kind::DotTree => (DOT_TREE_NS, METAL_MIN_TREE),
        Kind::SumExact => (SUM_EXACT_NS, METAL_MIN_EXACT),
        Kind::DotExact => (DOT_EXACT_NS, METAL_MIN_EXACT),
    };
    Calibration {
        ns_per_term,
        spawn_ns_per_worker: SPAWN_NS,
        metal_min_terms,
    }
}

// Provisional: Apple M4, one worker at 2^26 terms, and the largest per-worker
// start-up seen at 2^10..2^12 on ten workers (load average 4.1). Linux CUDA-host rows
// are pending a quiet window; see `calibration/fsum-v1.json`.
const SUM_TREE_NS: f64 = 0.653;
const DOT_TREE_NS: f64 = 0.493;
const SUM_EXACT_NS: f64 = 3.109;
const DOT_EXACT_NS: f64 = 2.287;
const SPAWN_NS: f64 = 8930.0;
// End to end (upload, kernel, host finish), Metal on an Apple M4 lost to the
// CPU workers at every size from 2^10 to 2^26 for all four families; `null`
// in the profile. Re-measure when uploads stop copying.
const METAL_MIN_TREE: usize = usize::MAX;
const METAL_MIN_EXACT: usize = usize::MAX;

/// Workers the CPU cost model picks: all requested workers when splitting
/// repays their start-up, else one. Every count gives the same bits.
pub fn cpu_workers(kind: Kind, terms: usize, workers: usize) -> usize {
    let model = calibration(kind);
    let serial = model.ns_per_term * terms as f64;
    let split = model.spawn_ns_per_worker * workers as f64 + serial / workers.max(1) as f64;
    if workers > 1 && split < serial {
        workers
    } else {
        1
    }
}

/// The call's bits on the workers the cost model picks from `workers`.
pub fn cpu(call: &ReduceCall<'_>, workers: usize) -> Result<F32Bits, Error> {
    cpu_on(call, cpu_workers(call.kind, call.a.len(), workers))
}

/// The call's bits on exactly `workers` CPU threads (benchmarks and tests).
pub fn cpu_on(call: &ReduceCall<'_>, workers: usize) -> Result<F32Bits, Error> {
    if workers <= 1 {
        return Ok(serial_reduction(call));
    }
    let bits = if call.kind.is_exact() {
        exact(call, workers)?
    } else {
        tree(call, workers)?
    };
    Ok(F32Bits::from_bits(bits))
}

/// Device steps per SIMD group: `2^2` runs of 32 leaves, 1024 elements
/// (fastest of 0..=3 at 2^24 and 2^26 in the Metal resident benchmark).
pub const LEAF_SHIFT: u32 = 2;

/// Calibration family id of a reduction (ruling C14).
pub fn family_id(kind: Kind) -> &'static str {
    match kind {
        Kind::SumTree => "fsum.sum_tree.f32",
        Kind::DotTree => "fsum.dot_tree.f32",
        Kind::SumExact => "fsum.sum_exact.f32",
        Kind::DotExact => "fsum.dot_exact.f32",
    }
}

pub fn device_op(kind: Kind) -> Op {
    match kind {
        Kind::SumTree => Op::SumTree,
        Kind::DotTree => Op::DotTree,
        Kind::SumExact => Op::SumExact,
        Kind::DotExact => Op::DotExact,
    }
}

/// The operands as raw binary32 words for a device upload.
pub fn words(values: &[F32Bits]) -> Vec<u32> {
    values.iter().map(|value| value.to_bits()).collect()
}

/// Finish device partials with the shared host code: tree nodes of one level
/// fold as an aligned tree; exact digit rows merge and round once.
pub fn finish(call: &ReduceCall<'_>, nodes: &[u32], rows: &[i64]) -> F32Bits {
    let bits = if call.kind.is_exact() {
        fsum::finish_rows(rows, call.a.len() as u64, call.kind.is_dot())
    } else {
        fsum::canonical(fsum::fold_aligned(nodes.len(), |group| {
            f32::from_bits(nodes[group])
        }))
    };
    F32Bits::from_bits(bits)
}

/// Calibrated device and CPU cost of one reduction: operand bytes cross the
/// bus once; exact deposits cost about eight tree additions per term.
pub fn cost(call: &ReduceCall<'_>, model: DeviceCostModel) -> DeviceCost {
    let operands = 1 + usize::from(call.kind.is_dot());
    let operations = match call.kind {
        Kind::SumTree => 1,
        Kind::DotTree => 2,
        Kind::SumExact => 8,
        Kind::DotExact => 16,
    };
    model.estimate(call.a.len() * 4 * operands, call.a.len(), operations, 1)
}

/// Whether any function calls a float reduction, so a device run has work.
pub fn has_reduction(program: &crate::mir::Program) -> bool {
    program
        .functions
        .iter()
        .flat_map(|function| &function.blocks)
        .flat_map(|block| &block.instructions)
        .any(|instruction| {
            matches!(&instruction.kind, crate::mir::Operation::Call { name, .. } if Kind::from_name(name).is_some())
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    const KINDS: [Kind; 4] = [Kind::SumTree, Kind::DotTree, Kind::SumExact, Kind::DotExact];

    #[test]
    fn calibration_table_matches_the_recorded_profile() {
        let profile: serde_json::Value =
            serde_json::from_str(include_str!("../../calibration/fsum-v1.json")).unwrap();
        assert_eq!(profile["schema"], "invar-fsum-calibration-v1");
        for kind in KINDS {
            let entry = &profile["families"][family_id(kind)];
            let model = calibration(kind);
            assert_eq!(entry["ns_per_term"].as_f64(), Some(model.ns_per_term));
            let spawn = entry["spawn_ns_per_worker"].as_f64();
            assert_eq!(spawn, Some(model.spawn_ns_per_worker));
            let metal = &entry["metal_min_terms"];
            let terms = metal.as_u64().map_or(usize::MAX, |terms| terms as usize);
            assert!(metal.is_null() || metal.is_u64(), "{metal}");
            assert_eq!(terms, model.metal_min_terms);
        }
    }

    #[test]
    fn cost_model_keeps_small_calls_on_one_worker() {
        for kind in KINDS {
            assert_eq!(cpu_workers(kind, 0, 8), 1);
            assert_eq!(cpu_workers(kind, 64, 8), 1);
            assert_eq!(cpu_workers(kind, 1 << 28, 1), 1);
            assert_eq!(cpu_workers(kind, 1 << 28, 8), 8);
        }
    }
}
