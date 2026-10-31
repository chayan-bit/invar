//! CUDA adapters for the spatial tests shared with Metal, and the C13 size lists.
use invar::{mir, numeric_contract::F32Bits as F, schedule::cuda};
use invar_cuda::cuda::{CollectiveFault, CudaDevice};
use invar_device::{sorting, spatial};
use std::{path::Path, process::Command};

#[path = "../support/spatial_oracle.rs"]
#[allow(dead_code)]
mod oracle;

type Device = CudaDevice;

const IMPLEMENTATION: &str = "cuda-sorted-grid-strict-f32";

fn compiler() -> &'static Path {
    Path::new(env!("CARGO_BIN_EXE_invar-nvrtc"))
}

fn open() -> Device {
    CudaDevice::open().expect("physical CUDA required")
}

fn run_spatial(device: &mut Device, plan: &spatial::Plan, inputs: [&[u32]; 4]) -> body::Spatial {
    let r = device.run_spatial(plan, inputs, compiler()).unwrap();
    body::Spatial {
        counts: r.counts,
        indices: r.indices,
        distances: r.distances,
        normal_window_enabled: None,
    }
}

fn run_sort(device: &mut Device, plan: &sorting::Plan, keys: &[u32], payload: &[u32]) -> Vec<u32> {
    device
        .run_sort(plan, keys, payload, compiler())
        .unwrap()
        .payload
}

fn run_source(device: &mut Device, source: &[u8], program: &mir::Program, fuel: u64) -> cuda::Run {
    cuda::run_source_with_device(source, program, fuel, device, compiler()).unwrap()
}

fn run_auto_unmeasured(source: &[u8], program: &mir::Program, fuel: u64) -> cuda::AutoRun {
    let compiler = Path::new("/unavailable/spatial-nvrtc");
    cuda::run_auto_source(source, program, fuel, compiler).unwrap()
}

/// `invar run --engine cuda` on the checker demo, as its JSON document.
fn cli_checker_document() -> serde_json::Value {
    let output = Command::new(env!("CARGO_BIN_EXE_invar"))
        .args([
            "run",
            "demo/spatial-checker.inv",
            "--engine",
            "cuda",
            "--format",
            "json",
        ])
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
        CollectiveFault::OutOfMemory,
        CollectiveFault::LaunchFailure,
        CollectiveFault::PartialCompletion,
    ] {
        let mut device = open();
        assert!(
            device
                .run_spatial_with_fault(&p, [&[0; 3], &[], &[], &[]], compiler(), fault)
                .is_err()
        );
        if fault == CollectiveFault::PartialCompletion {
            assert!(device.is_poisoned());
        }
    }
}

fn extra_query_patterns() -> Vec<[F; 3]> {
    Vec::new()
}

#[path = "../support/gpu_spatial.rs"]
mod body;

#[test]
#[ignore = "physical CUDA; C13 raw harness through 2^18 (the per-push set)"]
fn spatial_cuda_c13_periodic_cloud_through_2_18() {
    body::c13_periodic_cloud(&[0, 1, 31, 129, 4097, 65536, 1 << 18]);
}

#[test]
#[ignore = "physical CUDA; C13 raw harness through 2^22, nightly long list"]
fn spatial_cuda_c13_large_periodic_cloud() {
    body::c13_periodic_cloud(&[0, 1, 31, 129, 4097, 65536, 1 << 18, 1 << 20, 1 << 22]);
}
