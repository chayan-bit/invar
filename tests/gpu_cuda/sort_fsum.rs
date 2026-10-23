//! On CUDA, a program that also sorts (or runs a spatial query) runs its
//! float reductions where the program without the sort runs them, forced
//! (`--engine cuda`) and under auto's cost model, with the same receipts.
use invar::{
    plan::DeviceCostModel,
    schedule::{cuda, reductions::Reduction},
};
use invar_cuda::cuda::CudaDevice;
use std::path::Path;

fn compiler() -> &'static Path {
    Path::new(env!("CARGO_BIN_EXE_invar-nvrtc"))
}

fn open() -> CudaDevice {
    CudaDevice::open().expect("a CUDA device with NVRTC")
}

fn forced(device: &mut CudaDevice, source: &str) -> cuda::Run {
    cuda::run_source_with_device(
        source.as_bytes(),
        &body::lower(source),
        body::FUEL,
        device,
        compiler(),
    )
    .expect("forced run")
}

fn auto(source: &str, model: DeviceCostModel) -> cuda::AutoRun {
    cuda::run_auto_source_with_fault(
        source.as_bytes(),
        &body::lower(source),
        body::FUEL,
        compiler(),
        model,
        cuda::Fault::None,
    )
    .expect("auto run")
}

/// Forced CUDA reduces every call on the device.
fn on_device(reductions: &[Reduction]) -> bool {
    reductions.iter().all(|r| r.backend == "cuda")
}

type Placement = (DeviceCostModel, fn(&Reduction) -> bool);

/// The device-favouring model places every reduction on CUDA, the CPU model on the CPU.
fn auto_models() -> [Placement; 2] {
    [
        (body::DEVICE, |r| r.backend.starts_with("cuda")),
        (body::CPU, |r| r.backend.starts_with("cpu-")),
    ]
}

#[path = "../support/gpu_sort_fsum.rs"]
mod body;
