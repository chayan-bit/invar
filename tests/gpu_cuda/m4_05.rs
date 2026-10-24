//! Indexed families on a real CUDA device: 20 oracle cases, fault paths, and the 2^28 capacity runs.
#[path = "../support/m4_05_device.rs"]
mod corpus;
#[test]
#[ignore = "requires real NVIDIA GPU and matching NVRTC; missing hardware is failure"]
fn issue_m4_05_gpu() {
    {
        let mut device = invar_cuda::cuda::CudaDevice::open()
            .expect("real CUDA driver and a local NVRTC required");
        let caps = device.capabilities();
        println!("target=cuda device={} caps={caps:?}", device.name());
        let compiler = std::path::Path::new(env!("CARGO_BIN_EXE_invar-nvrtc"));
        for case in corpus::cases(caps) {
            assert_eq!(corpus::sequential_reference(&case.bound), case.expected);
            let result = device.run_indexed(&case.bound, compiler).unwrap();
            assert_eq!(result.target, "cuda");
            assert_eq!(result.output, case.expected, "{}", case.name);
            println!(
                "{} target={} reason={} phases={} transfer_bytes={} transfer_mode={} source_hash={:016x}",
                case.name,
                result.target,
                result.reason,
                result.completed_phases,
                result.transfer_bytes,
                result.transfer_mode,
                result.source_hash
            );
        }
        assert_eq!(corpus::refusals(caps), 7);
        let case = corpus::cases(caps).remove(12);
        for fault in [
            invar_cuda::cuda::CollectiveFault::OutOfMemory,
            invar_cuda::cuda::CollectiveFault::LaunchFailure,
            invar_cuda::cuda::CollectiveFault::PartialCompletion,
        ] {
            assert!(
                device
                    .run_indexed_with_fault(&case.bound, compiler, fault)
                    .is_err()
            );
        }
        assert!(device.quarantine_len() > 0);
        println!(
            "20 oracle cases; 3 fault paths refused without output; unknown completion quarantined"
        );
    }
}

macro_rules! cap_test {
    ($name:ident, $family:literal) => {
        #[test]
        #[ignore = "requires real CUDA and a local NVRTC; 2^28 cells/elements"]
        fn $name() {
            {
                let mut device = invar_cuda::cuda::CudaDevice::open().expect("real CUDA required");
                let bound = corpus::large_bound($family, 1 << 28, device.capabilities());
                let compiler = std::path::Path::new(env!("CARGO_BIN_EXE_invar-nvrtc"));
                println!("family={} size=268435456 units={} target=cuda device={}", $family,
                    if $family == "frontier" { "adjacency-cells vertices=16384 rounds=16" } else { "elements" }, device.name());
                let result = device.run_indexed(&bound, compiler).expect("large device run must complete");
                assert!(result.output == corpus::sequential_reference(&bound), "{} bit mismatch", $family);
                println!("family={} size=268435456 bit_exact=true phases={} transfer_bytes={} source_hash={:016x}",
                    $family, result.completed_phases, result.transfer_bytes, result.source_hash);
            }
        }
    };
}
cap_test!(issue_m4_05_gpu_caps_2_28_gather, "gather");
cap_test!(issue_m4_05_gpu_caps_2_28_scatter, "scatter");
cap_test!(issue_m4_05_gpu_caps_2_28_histogram, "histogram");
cap_test!(issue_m4_05_gpu_caps_2_28_frontier, "frontier");
