//! GNET-03 device fault injection on the CPU group model (feature
//! `gnet-fault-injection` only): thread 31 of the 32-thread group rewrites
//! shared slot words between group barriers in the emitted CUDA and Metal
//! bodies. `issue_gnet_03_gpu` and `issue_gnet_03_metal_gpu` run the same
//! injection on the actual devices.
#![cfg(feature = "gnet-fault-injection")]
#![forbid(unsafe_code)]

#[path = "support/gnet_device.rs"]
mod device;
#[path = "support/gnet_parallel.rs"]
mod parallel;

use invar::general_net::{
    compiled::{FAULT_HOOK, Fault, Target},
    device_host::cpu::CpuModel,
};

/// One test per fault so the per-run model compiles proceed in parallel.
fn both_targets(fault: Fault) {
    for target in [Target::Cuda, Target::Metal] {
        parallel::injected(
            target,
            &format!("cpu-model-{target:?}"),
            fault,
            CpuModel::default,
        );
    }
}

#[test]
fn issue_gnet_03_injection_cpu_model_stale_completion_stays_exact() {
    both_targets(Fault::StaleCompletion);
}

#[test]
fn issue_gnet_03_injection_cpu_model_poisoned_stale_stays_exact() {
    both_targets(Fault::PoisonedStale);
}

#[test]
fn issue_gnet_03_injection_cpu_model_contested_claim_stays_exact() {
    both_targets(Fault::ContestedClaim);
}

#[test]
fn issue_gnet_03_injection_cpu_model_cancel_race_stays_exact() {
    both_targets(Fault::CancelRace);
}

/// Positive control for the production check in `issue_gnet_03_device`: the
/// same byte search finds the hook in a binary built with the feature.
#[test]
fn issue_gnet_03_injection_build_contains_the_hook_text() {
    let binary = std::fs::read(env!("CARGO_BIN_EXE_invar")).unwrap();
    let hook = FAULT_HOOK.as_bytes();
    assert!(binary.windows(hook.len()).any(|w| w == hook));
}
