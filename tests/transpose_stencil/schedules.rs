//! M4-02 contract entry for the transpose half of
//! `cargo test --locked schedules`.
//!
//! The stencil half of the selection is `tests/transpose_stencil/stencil_source.rs`
//! (source binding/mutations, ordinary source and one/several ping-pong slices
//! under `constant_zero` and `clamp`, failure prefixes, emitted CPU bits and
//! guards, FP32 goldens, special-value bits, the CUDA-text CPU barrier model).
//! Before this file the selection held no transpose test in this package.
//! Here the reached ordinary-source transpose is published on the CPU device for
//! every contract shape, and checked against the reference interpreter and an
//! independently derived index model, including a round trip.
//!
//! Related CPU coverage that keeps its own (unselected) names:
//! `transpose_source` (identity, alias/index-variant refusals,
//! physical bounds, exact loop fuel, emitted CPU bits and guard words),
//! `transpose_host` (fuel prefixes, pending/unknown/read/stale
//! device faults with no publication), `stencil_runtime` and
//! `combined_runtime` (CPU-model stencil and combined runtimes).
//!
//! Unavailable (GPU, not run here): `gpu_cuda::transpose_stencil` under `--features cuda`
//! (device, driver, transfers, completion and timing on real hardware), device
//! transpose/stencil runtime execution, and the 4096x4096 device transpose.
#![forbid(unsafe_code)]
use invar::{
    check, eval, front,
    transpose_source::host::{self, CpuDevice},
};
use invar_cuda::Runtime;
use invar_device::Capabilities;

const FUEL: u64 = 50_000_000;
const FLIP: &str = "fn flip(a: read Array<u32>, rows: u32, cols: u32) -> Array<u32> { var b = filled_u32(rows * cols,0u32); for r in 0u32..rows { for c in 0u32..cols { b[c * rows + r] = a[r * cols + c]; } } return b; }";
/// Every contract shape (`0xN`, `1x1`, `31x33`, `127x257`) plus bounded squares.
const SHAPES: [[u32; 2]; 8] = [
    [0, 33],
    [33, 0],
    [1, 1],
    [31, 33],
    [127, 257],
    [16, 16],
    [17, 17],
    [64, 64],
];

/// Nonuniform input words, recomputed (not reread) by every check.
const GEN: &str =
    "fn gen(i: u32) -> u32 { return wrapping_add(wrapping_mul(i, 2654435761u32), 40503u32); }";

fn program(rows: u32, cols: u32, body: &str) -> String {
    format!(
        "{FLIP} {GEN} fn main() -> i32 {{ let rows = {rows}u32; let cols = {cols}u32; var a = filled_u32(rows * cols,0u32); for g in 0u32..len(a) {{ a[g] = gen(g); }} {body} for k in 0u32..len(a) {{ if a[k] != gen(k) {{ return 0; }} }} return 7; }}"
    )
}

fn publish(source: &str) -> host::Run {
    let checked = check::check(front::parse(source.as_bytes()).unwrap(), true).unwrap();
    let reference = eval::observe(&checked, FUEL);
    let mut runtime = Runtime::new(CpuDevice, 2);
    let run = host::run_source(
        source.as_bytes(),
        FUEL,
        &mut runtime,
        Capabilities::cuda_12_9_sm89(),
    )
    .unwrap();
    assert_eq!(reference.result, Ok(eval::Value::I32(7)));
    assert_eq!(run.observation.result, reference.result);
    assert_eq!(run.observation.steps_used, reference.steps_used);
    assert_eq!(run.observation.cleanup_trace, reference.cleanup_trace);
    assert_eq!(runtime.quarantine_len(), 0);
    run
}

#[test]
fn schedules_reached_transpose_contract_shapes() {
    for [rows, cols] in SHAPES {
        // Output word j is input word (j % rows) * cols + j / rows: derived from
        // the destination index, not the producer's source-major loop.
        let body = "let b = flip(a, rows, cols); if len(b) != rows * cols { return 0; } for j in 0u32..len(b) { if b[j] != gen((j % rows) * cols + j / rows) { return 0; } }";
        let run = publish(&program(rows, cols, body));
        assert_eq!(
            run.completed,
            usize::from(rows > 0),
            "{rows}x{cols}: {:?}",
            run.refusals
        );
        println!(
            "reached_transpose dimensions={rows}x{cols} completed={}",
            run.completed
        );
    }
}

#[test]
fn schedules_reached_transpose_round_trip() {
    for [rows, cols] in SHAPES {
        let body = "let b = flip(a, rows, cols); let c = flip(b, cols, rows); if len(c) != len(a) { return 0; } for i in 0u32..len(c) { if c[i] != gen(i) { return 0; } }";
        let run = publish(&program(rows, cols, body));
        // Each flip publishes when its outer (row) axis is nonempty.
        assert_eq!(
            run.completed,
            usize::from(rows > 0) + usize::from(cols > 0),
            "{rows}x{cols}: {:?}",
            run.refusals
        );
        println!(
            "reached_transpose_round_trip dimensions={rows}x{cols} completed={}",
            run.completed
        );
    }
}
