//! GNET-03 CUDA hardware suite (ignored by default): the ordinary-source ABI5
//! route compiled ahead by NVRTC, loaded from its container through the Driver
//! API and executed on the actual device, compared word for word with the
//! bounded CPU net oracle over the runnable corpus and the adversarial
//! dynamic-frontier cases. Run with
//! `cargo test --features cuda --test issue_gnet_03_gpu -- --ignored`.
#![cfg(feature = "cuda")]
#![forbid(unsafe_code)]

#[path = "support/gnet_cuda_aot.rs"]
mod aot;
#[path = "support/gnet_device.rs"]
mod device;
#[path = "support/gnet_parallel.rs"]
mod parallel;

use invar::general_net::compiled::Target;

fn cuda() -> aot::Aot {
    aot::open()
}

#[test]
#[ignore = "requires a CUDA device and NVRTC"]
fn issue_gnet_03_gpu_corpus_executes_on_cuda_with_exact_oracle_agreement() {
    let d = device::sharded(Target::Cuda, "cuda", cuda, true, 4);
    d.print("cuda");
    assert_eq!(d.fallback, 0);
    assert!(d.executed_total() >= 250, "{d:?}");
    assert!(d.executed["agree_trap"] > 0 && d.executed["agree_value"] > 0);
}

#[test]
#[ignore = "requires a CUDA device and NVRTC"]
fn issue_gnet_03_gpu_adversarial_dynamic_frontiers_match_oracle() {
    let device = cuda();
    println!(
        "gnet_03 cuda device {} sm_{} driver {}",
        device.cuda.name(),
        device.cuda.sm(),
        device.cuda.driver_version()
    );
    let mut backend = cuda;
    for case in device::adversarial() {
        let s = device::run_case(Target::Cuda, "cuda", &case, &mut backend);
        println!("gnet_03 cuda {}: {s:?}", case.name);
        device::check_case(case.name, &s);
    }
    assert!(device::fuel_prefixes(Target::Cuda, "cuda", &mut backend) >= 10);
}

#[test]
#[ignore = "requires a CUDA device and NVRTC"]
fn issue_gnet_03_gpu_fence_cancellation_and_quarantine() {
    device::lifecycle(Target::Cuda, "cuda", cuda);
}

#[test]
#[ignore = "requires a CUDA device and NVRTC"]
fn issue_gnet_03_gpu_parallel_slice_executes_wide_frontiers() {
    parallel::denominator(Target::Cuda, "cuda", cuda);
}

/// Thread 31 rewrites slot words between group barriers in the real CUDA
/// block; every run stays exact and the counters show the fault landed.
#[cfg(feature = "gnet-fault-injection")]
#[test]
#[ignore = "requires a CUDA device and NVRTC"]
fn issue_gnet_03_gpu_injected_slot_races_stay_exact() {
    for fault in parallel::FAULTS {
        parallel::injected(Target::Cuda, "cuda", fault, cuda);
    }
}

/// The hook is a non-inlined device function: NVRTC keeps its symbol in the
/// injected image, and the production image of the same program has none.
/// Both are compiled-ahead containers, checked after decoding.
#[cfg(feature = "gnet-fault-injection")]
#[test]
#[ignore = "requires a CUDA device and NVRTC"]
fn issue_gnet_03_gpu_fault_hook_is_absent_from_production_cubin() {
    use invar::general_net::{
        compiled::FAULT_HOOK,
        device_host::{Plan, cuda_aot},
    };
    let sm = cuda().cuda.sm();
    let image = |plan: &Plan| {
        let bytes = cuda_aot::build(plan, aot::compiler(), sm).unwrap();
        cuda_aot::decode(&bytes).unwrap().cubin().to_vec()
    };
    let contains = |image: &[u8]| {
        image
            .windows(FAULT_HOOK.len())
            .any(|w| w == FAULT_HOOK.as_bytes())
    };
    for source in parallel::slice() {
        let production =
            Plan::from_source(&source.bytes, "run", Target::Cuda, parallel::limits()).unwrap();
        assert!(
            !contains(&image(&production)),
            "{}: production CUBIN",
            source.name
        );
        for fault in parallel::FAULTS {
            let faulted = Plan::from_source_with_fault(
                &source.bytes,
                "run",
                Target::Cuda,
                parallel::limits(),
                fault,
            )
            .unwrap();
            assert!(
                contains(&image(&faulted)),
                "{} {fault:?}: injected CUBIN",
                source.name
            );
        }
        println!(
            "gnet_03 cuda cubin hook check {}: production absent, {} faults present",
            source.name,
            parallel::FAULTS.len()
        );
    }
}
