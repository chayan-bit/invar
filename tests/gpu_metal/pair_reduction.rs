//! Metal adapters for the pair-reduction tests shared with CUDA.
use invar::{device_ir::Ir, mir, schedule::metal};
use invar_metal::MetalDevice;

#[path = "../support/pair_reduction.rs"]
mod support;

fn open() -> MetalDevice {
    MetalDevice::open_runtime_source().expect("Metal hardware required")
}

fn run_source(
    source: &str,
    program: &mir::Program,
    fuel: u64,
    device: &mut MetalDevice,
) -> metal::Run {
    metal::run_source_with_device(source.as_bytes(), program, fuel, device).unwrap()
}

/// Metal plans its own geometry on the region IR before verification.
fn prepare_ir(ir: &mut Ir) {
    invar::device_ir::plan_metal(ir).unwrap();
}

/// The pair, whether no prefixes were produced, and the source hash.
type Collected = (Option<[u32; 2]>, bool, u64);

fn run_collective(
    device: &mut MetalDevice,
    plan: &invar_device::Verified,
    inputs: &[&[u32]],
) -> Collected {
    let result = device.run_collective(plan, inputs, 0).unwrap();
    (result.pair, result.prefixes.is_empty(), result.source_hash)
}

#[path = "../support/gpu_pair_reduction.rs"]
mod body;
