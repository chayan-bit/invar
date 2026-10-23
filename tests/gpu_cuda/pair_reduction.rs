//! CUDA adapters for the pair-reduction tests shared with Metal.
use invar::{device_ir::Ir, mir, schedule::cuda};
use invar_cuda::cuda::CudaDevice;
use std::path::Path;

#[path = "../support/pair_reduction.rs"]
mod support;

fn compiler() -> &'static Path {
    Path::new(env!("CARGO_BIN_EXE_invar-nvrtc"))
}

fn open() -> CudaDevice {
    CudaDevice::open().expect("CUDA hardware required")
}

fn run_source(
    source: &str,
    program: &mir::Program,
    fuel: u64,
    device: &mut CudaDevice,
) -> cuda::Run {
    cuda::run_source_with_device(source.as_bytes(), program, fuel, device, compiler()).unwrap()
}

/// CUDA runs the region IR as lowered.
fn prepare_ir(_ir: &mut Ir) {}

/// The pair, whether no prefixes were produced, and the source hash.
type Collected = (Option<[u32; 2]>, bool, u64);

fn run_collective(
    device: &mut CudaDevice,
    plan: &invar_device::Verified,
    inputs: &[&[u32]],
) -> Collected {
    let result = device.run_collective(plan, inputs, 0, compiler()).unwrap();
    (result.pair, result.prefixes.is_empty(), result.source_hash)
}

#[path = "../support/gpu_pair_reduction.rs"]
mod body;
