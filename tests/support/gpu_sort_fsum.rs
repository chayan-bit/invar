//! A program that also sorts (or runs a spatial query) runs its float
//! reductions where the program without the sort runs them, forced onto the
//! device and under auto's cost model, with the same receipts.
//! Shared by the CUDA and Metal binaries; the including module supplies
//! `open`, `forced`, `auto`, `on_device` and `auto_models`.
use super::{auto, auto_models, forced, on_device, open};
use invar::{check, eval, front, mir, plan::DeviceCostModel, schedule::reductions::Reduction};

pub const FUEL: u64 = 50_000_000;

const REDUCTIONS: &str = "let a: Array<f32> = filled_f32(70000u32, 0.5); let b: Array<f32> = filled_f32(70000u32, 2.0); let s = sum_tree(a); let d = dot_exact(a, b); let e = sum_exact(b); let t = dot_tree(a, b); if s != 35000.0 or d != 70000.0 or e != 140000.0 or t != 70000.0 { return 1; } return 0; }";
const SORT: &str =
    "var keys: Array<u32> = [3u32, 1u32, 2u32]; sort(keys); if keys[0u32] != 1u32 { return 2; } ";
const SPATIAL: &str = "let p: Array<f32> = [0.0]; let grid = voxel_grid_build(p, p, p, 1.0); let q = filled_f32(129u32, 0.0); let near = radius_neighbors(grid, q, q, q, 1.0, 1u32); ";

/// Auto models that make every reduction choose the device, or the CPU.
pub const DEVICE: DeviceCostModel = DeviceCostModel {
    cpu_bytes_per_ms: 1.,
    device_bytes_per_ms: 1e12,
    device_fixed_ms: 0.,
};
pub const CPU: DeviceCostModel = DeviceCostModel {
    cpu_bytes_per_ms: 1e12,
    device_bytes_per_ms: 1.,
    device_fixed_ms: 1e9,
};

pub fn lower(source: &str) -> mir::Program {
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
#[ignore = "requires a GPU (CUDA with NVRTC, or Apple Metal)"]
fn forced_sort_and_spatial_runs_reduce_on_the_device() {
    let mut device = open();
    for (alone, with) in programs() {
        let expected = forced(&mut device, &alone);
        let actual = forced(&mut device, &with);
        assert_eq!(actual.observation.result, Ok(eval::Value::I32(0)), "{with}");
        assert_eq!(expected.observation.result, actual.observation.result);
        assert_eq!(receipts(&actual.reductions), receipts(&expected.reductions));
        assert_eq!(actual.reductions.len(), 4, "{:?}", actual.reductions);
        assert!(on_device(&actual.reductions), "{:?}", actual.reductions);
    }
}

#[test]
#[ignore = "requires a GPU (CUDA with NVRTC, or Apple Metal)"]
fn auto_sort_and_spatial_runs_take_the_fsum_cost_choice() {
    for (model, placed) in auto_models() {
        for (alone, with) in programs() {
            let (expected, actual) = (auto(&alone, model), auto(&with, model));
            assert_eq!(actual.observation.result, Ok(eval::Value::I32(0)), "{with}");
            assert_eq!(receipts(&actual.reductions), receipts(&expected.reductions));
            assert_eq!(actual.reductions.len(), 4, "{:?}", actual.reductions);
            assert!(
                actual.reductions.iter().all(placed),
                "{:?}",
                actual.reductions
            );
            assert!(!actual.executed.is_empty(), "the sort or query ran");
        }
    }
}
