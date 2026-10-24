//! On Metal, a program that also sorts (or runs a spatial query) runs its
//! float reductions where the program without the sort runs them, forced
//! (`--engine metal`) and under auto's cost model, with the same receipts.
use invar::{
    plan::DeviceCostModel,
    schedule::{metal, reductions::Reduction},
};
use invar_metal::{Fault, MetalDevice};

fn open() -> MetalDevice {
    MetalDevice::open_runtime_source().expect("Metal device")
}

fn forced(device: &mut MetalDevice, source: &str) -> metal::Run {
    metal::run_source_with_device(source.as_bytes(), &body::lower(source), body::FUEL, device)
        .expect("forced run")
}

fn auto(source: &str, model: DeviceCostModel) -> metal::AutoRun {
    metal::run_auto_source_with_fault(
        source.as_bytes(),
        &body::lower(source),
        body::FUEL,
        metal::Kernels::RuntimeSource,
        model,
        Fault::None,
    )
    .expect("auto run")
}

fn on_device(reductions: &[Reduction]) -> bool {
    reductions.iter().any(|r| r.backend == "metal")
}

type Placement = (DeviceCostModel, fn(&Reduction) -> bool);

/// Auto has no measured Metal crossover for any reduction yet
/// (`metal_min_terms` is unbounded), so even the device-favouring model
/// keeps every call on the CPU, as in the fsum-only program.
fn auto_models() -> [Placement; 2] {
    fn cpu_by_measurement(r: &Reduction) -> bool {
        r.backend.starts_with("cpu-") && r.reason == "measured-cpu-cost"
    }
    [
        (body::DEVICE, cpu_by_measurement),
        (body::CPU, cpu_by_measurement),
    ]
}

#[path = "../support/gpu_sort_fsum.rs"]
mod body;
