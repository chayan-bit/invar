//! On CUDA, a program that also sorts (or runs a spatial query) runs its
//! float reductions where the program without the sort runs them, forced
//! (`--engine cuda`) and under auto's cost model, with the same receipts.
#![cfg(feature = "cuda")]

use invar::{
    check, eval, front, mir,
    plan::DeviceCostModel,
    schedule::{cuda, reductions::Reduction},
};
use std::path::Path;

const FUEL: u64 = 50_000_000;

const REDUCTIONS: &str = "let a: Array<f32> = filled_f32(70000u32, 0.5); let b: Array<f32> = filled_f32(70000u32, 2.0); let s = sum_tree(a); let d = dot_exact(a, b); let e = sum_exact(b); let t = dot_tree(a, b); if s != 35000.0 or d != 70000.0 or e != 140000.0 or t != 70000.0 { return 1; } return 0; }";
const SORT: &str =
    "var keys: Array<u32> = [3u32, 1u32, 2u32]; sort(keys); if keys[0u32] != 1u32 { return 2; } ";
const SPATIAL: &str = "let p: Array<f32> = [0.0]; let grid = voxel_grid_build(p, p, p, 1.0); let q = filled_f32(129u32, 0.0); let near = radius_neighbors(grid, q, q, q, 1.0, 1u32); ";

/// Auto models that make every reduction choose the device, or the CPU.
const DEVICE: DeviceCostModel = DeviceCostModel {
    cpu_bytes_per_ms: 1.,
    device_bytes_per_ms: 1e12,
    device_fixed_ms: 0.,
};
const CPU: DeviceCostModel = DeviceCostModel {
    cpu_bytes_per_ms: 1e12,
    device_bytes_per_ms: 1.,
    device_fixed_ms: 1e9,
};

fn compiler() -> &'static Path {
    Path::new(env!("CARGO_BIN_EXE_invar-nvrtc"))
}

fn lower(source: &str) -> mir::Program {
    let checked =
        check::check(front::parse(source.as_bytes()).expect("parse"), true).expect("check");
    mir::lower(&checked).expect("lower")
}

/// Receipts without spans (which move with the inserted prefix) and timings.
fn receipts(reductions: &[Reduction]) -> Vec<String> {
    reductions
        .iter()
        .map(|r| {
            let (family, backend, reason) = (r.family, r.backend, r.reason);
            format!("{} {family} {} {backend} {reason}", r.function, r.elements)
        })
        .collect()
}

fn programs() -> Vec<(String, String)> {
    let alone = format!("fn main() -> i32 {{ {REDUCTIONS}");
    [SORT, SPATIAL]
        .iter()
        .map(|prefix| {
            (
                alone.clone(),
                format!("fn main() -> i32 {{ {prefix}{REDUCTIONS}"),
            )
        })
        .collect()
}

#[test]
#[ignore = "requires a CUDA device and NVRTC"]
fn forced_cuda_sort_and_spatial_runs_reduce_on_the_device() {
    for (alone, with) in programs() {
        let expected = cuda::run_source(alone.as_bytes(), &lower(&alone), FUEL, compiler())
            .expect("fsum-only run");
        let actual = cuda::run_source(with.as_bytes(), &lower(&with), FUEL, compiler())
            .expect("intrinsic run");
        assert_eq!(actual.observation.result, Ok(eval::Value::I32(0)), "{with}");
        assert_eq!(expected.observation.result, actual.observation.result);
        assert_eq!(receipts(&actual.reductions), receipts(&expected.reductions));
        assert_eq!(actual.reductions.len(), 4, "{:?}", actual.reductions);
        assert!(
            actual.reductions.iter().all(|r| r.backend == "cuda"),
            "{:?}",
            actual.reductions
        );
    }
}

#[test]
#[ignore = "requires a CUDA device and NVRTC"]
fn auto_sort_and_spatial_runs_take_the_fsum_cost_choice() {
    for (model, backend) in [(DEVICE, "cuda"), (CPU, "cpu-")] {
        for (alone, with) in programs() {
            let run = |source: &str| {
                cuda::run_auto_source_with_fault(
                    source.as_bytes(),
                    &lower(source),
                    FUEL,
                    compiler(),
                    model,
                    cuda::Fault::None,
                )
                .expect("auto run")
            };
            let (expected, actual) = (run(&alone), run(&with));
            assert_eq!(actual.observation.result, Ok(eval::Value::I32(0)), "{with}");
            assert_eq!(receipts(&actual.reductions), receipts(&expected.reductions));
            assert_eq!(actual.reductions.len(), 4, "{:?}", actual.reductions);
            assert!(
                actual
                    .reductions
                    .iter()
                    .all(|r| r.backend.starts_with(backend)),
                "{backend}: {:?}",
                actual.reductions
            );
            assert!(!actual.executed.is_empty(), "the sort or query ran");
        }
    }
}
