//! Issue #5 (M4-05).
#![forbid(unsafe_code)]
#[cfg(target_os = "macos")]
#[path = "../../../tests/support/m4_05_device.rs"]
mod corpus;
#[test]
#[ignore = "requires real Apple7+ Metal GPU; missing hardware is failure"]
fn indexed_gpu() {
    #[cfg(target_os = "macos")]
    {
        let mut device =
            invar_metal::MetalDevice::open_runtime_source().expect("real Metal device required");
        let caps = device.capabilities();
        println!("target=metal device={} caps={caps:?}", device.name());
        for case in corpus::cases(caps) {
            assert_eq!(corpus::sequential_reference(&case.bound), case.expected);
            let result = device.run_indexed(&case.bound).unwrap();
            assert_eq!(result.target, "metal");
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
            invar_metal::Fault::OutOfMemory,
            invar_metal::Fault::LaunchFailure,
            invar_metal::Fault::PartialCompletion,
        ] {
            assert!(device.run_indexed_with_fault(&case.bound, fault).is_err());
        }
        assert!(device.quarantine_len() > 0);
        println!(
            "20 oracle cases; 3 fault paths refused without output; unknown completion quarantined"
        );
    }
    #[cfg(not(target_os = "macos"))]
    panic!("Metal qualification requires --features metal on macOS");
}

macro_rules! cap_test {
    ($name:ident, $family:literal) => {
        #[test]
        #[ignore = "one 2^28 Metal family at a time; requires real Apple GPU"]
        fn $name() {
            #[cfg(target_os = "macos")]
            {
                let mut device = invar_metal::MetalDevice::open_runtime_source().expect("real Metal required");
                let bound = corpus::large_bound($family, 1 << 28, device.capabilities());
                println!("family={} size=268435456 units={} target=metal device={}", $family,
                    if $family == "frontier" { "adjacency-cells vertices=16384 rounds=16" } else { "elements" }, device.name());
                let result = device.run_indexed(&bound).expect("large device run must complete");
                assert!(result.output == corpus::sequential_reference(&bound), "{} bit mismatch", $family);
                println!("family={} size=268435456 bit_exact=true phases={} transfer_bytes={} source_hash={:016x}",
                    $family, result.completed_phases, result.transfer_bytes, result.source_hash);
            }
            #[cfg(not(target_os = "macos"))]
            panic!("Metal qualification requires macOS");
        }
    };
}
cap_test!(cap_2_28_gather, "gather");
cap_test!(cap_2_28_scatter, "scatter");
cap_test!(cap_2_28_histogram, "histogram");
cap_test!(cap_2_28_frontier, "frontier");
