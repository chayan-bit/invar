//! #195 on Metal: a program that also sorts (or runs a spatial query) runs its
//! float reductions where the program without the sort runs them, forced
//! (`--engine metal`) and under auto's cost model, with the same receipts.
#![cfg(all(feature = "metal", target_os = "macos"))]

use invar::{
    check, eval, front, mir,
    plan::DeviceCostModel,
    schedule::{metal, reductions::Reduction},
};
use invar_metal::{Fault, MetalDevice};

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
#[ignore = "requires live Metal hardware"]
fn forced_metal_sort_and_spatial_runs_reduce_on_the_device() {
    let mut device = MetalDevice::open_runtime_source().expect("Metal device");
    for (alone, with) in programs() {
        let mut run = |source: &str| {
            metal::run_source_with_device(source.as_bytes(), &lower(source), FUEL, &mut device)
                .expect("forced run")
        };
        let (expected, actual) = (run(&alone), run(&with));
        assert_eq!(actual.observation.result, Ok(eval::Value::I32(0)), "{with}");
        assert_eq!(expected.observation.result, actual.observation.result);
        assert_eq!(receipts(&actual.reductions), receipts(&expected.reductions));
        assert_eq!(actual.reductions.len(), 4, "{:?}", actual.reductions);
        assert!(
            actual.reductions.iter().any(|r| r.backend == "metal"),
            "{:?}",
            actual.reductions
        );
    }
}

#[test]
#[ignore = "requires live Metal hardware"]
fn auto_metal_sort_and_spatial_runs_take_the_fsum_cost_choice() {
    // Auto has no measured Metal crossover for any reduction yet
    // (`metal_min_terms` is unbounded), so even the device-favouring model
    // keeps every call on the CPU, as in the fsum-only program.
    for model in [DEVICE, CPU] {
        for (alone, with) in programs() {
            let run = |source: &str| {
                metal::run_auto_source_with_fault(
                    source.as_bytes(),
                    &lower(source),
                    FUEL,
                    metal::Kernels::RuntimeSource,
                    model,
                    Fault::None,
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
                    .all(|r| r.backend.starts_with("cpu-") && r.reason == "measured-cpu-cost"),
                "{:?}",
                actual.reductions
            );
            assert!(!actual.executed.is_empty(), "the sort or query ran");
        }
    }
}
