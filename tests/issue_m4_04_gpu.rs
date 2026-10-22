//! M4-04 CUDA qualification: segmented map, segmented reduction and stable
//! compaction from one verified IR, executed on the device and compared bit for
//! bit with the source expectation (CPU net oracle and reference evaluator),
//! including maximum sizes. Ignored by default; when run explicitly it fails,
//! never skips, without an authorized CUDA device.
#![cfg(feature = "cuda")]
#![forbid(unsafe_code)]

#[path = "support/m4_04_cases.rs"]
mod cases;

use invar_cuda::{Error, cuda::CudaDevice};
use invar_device::segmented as seg;
use std::path::Path;

fn compiler() -> &'static Path {
    Path::new(env!("CARGO_BIN_EXE_invar-nvrtc"))
}

#[test]
#[ignore = "requires an authorized CUDA device and NVRTC 12.4+"]
fn issue_m4_04_gpu_segmented_families_match_the_source_bit_exact() {
    let mut device =
        CudaDevice::open().expect("explicit CUDA qualification: no usable device or NVRTC");
    let caps = device.capabilities();
    println!("device={} caps={caps:?}", device.name());
    let mut executed = 0;
    for case in cases::cases(true) {
        let expected = cases::expected(&case);
        let plan = cases::plan(&case, caps);
        assert_eq!(
            seg::reference(&plan, &case.input, &case.offsets).unwrap(),
            expected,
            "{}",
            case.name
        );
        let result = device
            .run_segmented(&plan, &case.input, &case.offsets, compiler())
            .unwrap_or_else(|e| panic!("{}: {e:?}", case.name));
        assert_eq!(result.output, expected, "{}", case.name);
        assert_eq!(device.quarantine_len(), 0);
        println!(
            "{} elements={} segments={} outputs={} blocks={} kernel_ms={:.3} compile_ms={:.1} transfer_bytes={} source={:016x}",
            case.name,
            case.input.len(),
            case.offsets.len().saturating_sub(1),
            result.output.len(),
            plan.blocks(),
            result.kernel_ms,
            result.compile_ms,
            result.transfer_bytes,
            result.source_hash
        );
        executed += 1;
    }
    assert!(executed >= 55, "{executed}");
}

#[test]
#[ignore = "requires an authorized CUDA device and NVRTC 12.4+"]
fn issue_m4_04_gpu_admission_refusals_launch_nothing() {
    let mut device =
        CudaDevice::open().expect("explicit CUDA qualification: no usable device or NVRTC");
    let caps = device.capabilities();
    let case = cases::cases(false)
        .into_iter()
        .find(|case| case.name == "Sum/empty-interior")
        .unwrap();
    let plan = cases::plan(&case, caps);
    let mut decreasing = case.offsets.clone();
    // [0, 3, 3, ...] becomes [0, 5, 3, ...]: decreasing offsets.
    decreasing[1] = decreasing[2] + 2;
    for (input, offsets) in [
        (&case.input[..], &decreasing[..]),
        (&case.input[1..], &case.offsets[..]),
    ] {
        assert_eq!(
            device
                .run_segmented(&plan, input, offsets, compiler())
                .unwrap_err(),
            Error::Limit
        );
    }
    assert_eq!(device.quarantine_len(), 0);
    let result = device
        .run_segmented(&plan, &case.input, &case.offsets, compiler())
        .unwrap();
    assert_eq!(result.output, cases::expected(&case));
}

// CUDA requires the shared transfer-cap branch before this explicit
// 2^28 qualification can pass.
#[test]
#[ignore = "requires GPU hardware, sufficient memory and shared runtime caps"]
fn issue_m4_04_gpu_size_cap_bit_exact() {
    let mut device = CudaDevice::open().expect("explicit GPU size-cap qualification");
    println!("device={} caps={:?}", device.name(), device.capabilities());
    for (kind, n) in [
        (cases::Kind::Map, 1 << 28),
        (cases::Kind::Sum, 1 << 28),
        (cases::Kind::Product, 1 << 28),
        (cases::Kind::Compact("<"), 1 << 28),
    ] {
        let case = cases::Case {
            name: format!("{kind:?}/size-cap"),
            kind,
            input: (0..n).map(cases::word).collect(),
            offsets: if matches!(kind, cases::Kind::Compact(_)) {
                Vec::new()
            } else {
                // Empty interior segment and a long ragged tail.
                vec![0, n / 3, n / 3, n]
            },
            cutoff: u32::MAX / 2,
            stride: None,
        };
        let plan = cases::plan(&case, device.capabilities());
        // Shared evaluator limits prevent a source-interpreter run here.
        // The independent sequential test-support oracle validates every word.
        let expected = cases::sequential_expected(&case);
        let result = device
            .run_segmented(&plan, &case.input, &case.offsets, compiler())
            .unwrap_or_else(|e| panic!("{} elements={n}: {e:?}", case.name));
        assert_eq!(result.output, expected, "{}", case.name);
        assert_eq!(device.quarantine_len(), 0);
        println!(
            "{} elements={n} outputs={} blocks={} kernel_ms={:.3} bytes={} source={:016x} bit_exact=true",
            case.name,
            result.output.len(),
            plan.blocks(),
            result.kernel_ms,
            result.transfer_bytes,
            result.source_hash
        );
    }
}
