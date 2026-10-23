//! GNET-03 Metal hardware suite (ignored by default): the ordinary-source ABI5
//! route compiled by the Metal compiler and executed on the actual device, compared word
//! for word with the bounded CPU net oracle over the runnable corpus and the
//! adversarial dynamic-frontier cases. Run with
//! `cargo test --features metal --test gpu_metal gnet_03:: -- --ignored`.
#![forbid(unsafe_code)]

use crate::{device, parallel};

use invar::general_net::{compiled::Target, device_host::metal::Metal};

/// The generated MSL is program specific, so there is no compiled-ahead
/// library to load: compile it at run time, as forced Metal does with
/// `INVAR_METAL_RUNTIME_SOURCE=1`. `issue_metal_aot_gpu` covers the
/// compiled-ahead route (`invar build --target metal`, then `gnet-device --artifact`).
fn metal() -> Metal {
    Metal::open_runtime_source().expect("Metal device")
}

#[test]
#[ignore = "requires an Apple Metal device"]
fn issue_gnet_03_metal_gpu_corpus_executes_on_metal_with_exact_oracle_agreement() {
    let d = device::sharded(Target::Metal, "metal", metal, true, 4, false);
    d.print("metal");
    assert_eq!(d.fallback, 0);
    assert!(d.executed_total() >= 250, "{d:?}");
    assert!(d.executed["agree_trap"] > 0 && d.executed["agree_value"] > 0);
}

#[test]
#[ignore = "requires an Apple Metal device"]
fn issue_gnet_03_metal_gpu_adversarial_dynamic_frontiers_match_oracle() {
    let mut backend = metal;
    for case in device::adversarial() {
        let s = device::run_case(Target::Metal, "metal", &case, &mut backend);
        println!("gnet_03 metal {}: {s:?}", case.name);
        device::check_case(case.name, &s);
    }
    assert!(device::fuel_prefixes(Target::Metal, "metal", &mut backend) >= 10);
}

#[test]
#[ignore = "requires an Apple Metal device"]
fn issue_gnet_03_metal_gpu_fence_cancellation_and_quarantine() {
    device::lifecycle(Target::Metal, "metal", metal);
}

#[test]
#[ignore = "requires an Apple Metal device"]
fn issue_gnet_03_metal_gpu_parallel_slice_executes_wide_frontiers() {
    parallel::denominator(Target::Metal, "metal", metal);
}

/// Thread 31 rewrites slot words between group barriers in the real Metal
/// threadgroup; every run stays exact and the counters show the fault landed.
#[cfg(feature = "gnet-fault-injection")]
#[test]
#[ignore = "requires an Apple Metal device"]
fn issue_gnet_03_metal_gpu_injected_slot_races_stay_exact() {
    for fault in parallel::FAULTS {
        parallel::injected(Target::Metal, "metal", fault, metal);
    }
}
