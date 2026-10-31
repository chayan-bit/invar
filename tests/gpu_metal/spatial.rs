//! Metal adapters for the spatial tests shared with CUDA, the C13 size list and
//! the normal-window witnesses only Metal has.
use invar::{mir, numeric_contract::F32Bits as F, schedule::metal};
use invar_device::{sorting, spatial};
use invar_metal::{Fault, MetalDevice};
use std::process::Command;

#[path = "../support/spatial_oracle.rs"]
#[allow(dead_code)]
mod oracle;

type Device = MetalDevice;

const IMPLEMENTATION: &str = "metal-sorted-grid-strict-f32";

fn open() -> Device {
    MetalDevice::open_runtime_source().expect("physical Metal required")
}

fn run_spatial(device: &mut Device, plan: &spatial::Plan, inputs: [&[u32]; 4]) -> body::Spatial {
    let r = device.run_spatial(plan, inputs).unwrap();
    body::Spatial {
        counts: r.counts,
        indices: r.indices,
        distances: r.distances,
        normal_window_enabled: Some(r.normal_window_enabled),
    }
}

fn run_sort(device: &mut Device, plan: &sorting::Plan, keys: &[u32], payload: &[u32]) -> Vec<u32> {
    device.run_sort(plan, keys, payload).unwrap().payload
}

fn run_source(device: &mut Device, source: &[u8], program: &mir::Program, fuel: u64) -> metal::Run {
    metal::run_source_with_device(source, program, fuel, device).unwrap()
}

fn run_auto_unmeasured(source: &[u8], program: &mir::Program, fuel: u64) -> metal::AutoRun {
    metal::run_auto_source(source, program, fuel, metal::Kernels::Missing).unwrap()
}

/// `invar run --engine metal` on the checker demo from a compiled-ahead library, as its JSON document.
fn cli_checker_document() -> serde_json::Value {
    let library =
        std::env::temp_dir().join(format!("spatial-checker-{}.metallib", std::process::id()));
    let built = Command::new(env!("CARGO_BIN_EXE_invar"))
        .args([
            "build",
            "demo/spatial-checker.inv",
            "--target",
            "metal",
            "--output",
        ])
        .arg(&library)
        .status()
        .unwrap();
    assert!(built.success());
    let output = Command::new(env!("CARGO_BIN_EXE_invar"))
        .args([
            "run",
            "demo/spatial-checker.inv",
            "--engine",
            "metal",
            "--artifact",
        ])
        .arg(&library)
        .args(["--format", "json"])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stdout)
    );
    serde_json::from_slice(&output.stdout).unwrap()
}

/// Injected faults fail the call, and an unconfirmed completion poisons the device.
fn fault_publication() {
    let p = spatial::Plan::mapping(1, 1.0f32.to_bits()).unwrap();
    for fault in [
        Fault::OutOfMemory,
        Fault::LaunchFailure,
        Fault::PartialCompletion,
    ] {
        let mut device = open();
        assert!(
            device
                .run_spatial_with_fault(&p, [&[0; 3], &[], &[], &[]], fault)
                .is_err()
        );
        if fault == Fault::PartialCompletion {
            assert!(device.is_poisoned());
        }
    }
}

/// This query forces the integer distance path while retaining the
/// normal-axis FMA and association witnesses in the point cloud.
fn extra_query_patterns() -> Vec<[F; 3]> {
    vec![[body::f(0.0), body::f(0.0), F::from_bits(0x17800000)]]
}

#[path = "../support/gpu_spatial.rs"]
mod body;

#[test]
#[ignore = "physical Metal; C13 raw harness through 2^22"]
fn spatial_metal_c13_large_periodic_cloud() {
    body::c13_periodic_cloud(&[0, 1, 31, 129, 4097, 65536, 1 << 20, 1 << 22]);
}

#[test]
#[ignore = "physical Metal"]
fn spatial_metal_normal_window_edge_is_proved_on_device() {
    let mut device = open();
    let f = body::f;
    let lower = F::from_bits(0x2b800000);
    let adjacent = F::from_bits(lower.to_bits() + 1);
    let edge = body::build(&mut device, &[[lower, f(0.0), f(0.0)]], lower);
    let actual = body::query(&mut device, &edge, &[[adjacent, f(0.0), f(0.0)]], None, 1);
    assert_eq!(actual.distances, [0x00800000]);
    eprintln!(
        "Metal proved normal window enabled: {:?}",
        actual.normal_window_enabled
    );
}

#[test]
fn spatial_metal_normal_window_boundaries_match_integer_oracle() {
    let low = F::from_bits(0x2b800000);
    let high = F::from_bits(0x5d800000);
    let adjacent = F::from_bits(low.to_bits() + 1);
    let dx = adjacent.sub(low);
    assert_eq!(dx.to_bits(), 0x20000000); // 2^-63
    assert_eq!(dx.mul(dx).to_bits(), 0x00800000); // min normal, never subnormal
    assert_eq!(
        oracle::squared_distance([high; 3], [high.negate(); 3]).to_bits(),
        0x7d400000
    );
    assert!(oracle::squared_distance([high; 3], [high.negate(); 3]).to_bits() < 0x7d800000); // <2^124
}
