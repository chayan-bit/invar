//! M4-04 qualification: segmented map, segmented reduction and stable
//! compaction from one verified IR, executed on the device and compared bit for
//! bit with the source expectation (CPU net oracle and reference evaluator),
//! including maximum sizes. Ignored by default; when run explicitly it fails,
//! never skips, without a device. Shared by the CUDA and Metal binaries; the
//! including module supplies `Device`, `open` and `run_segmented`.
use super::{Device, cases, open, run_segmented};
use invar_device::segmented as seg;

/// What each backend's segmented result reports, device-independently.
pub struct Out {
    pub output: Vec<u32>,
    pub kernel_ms: f64,
    pub compile_ms: f64,
    pub transfer_bytes: usize,
    pub source_hash: u64,
}

#[test]
#[ignore = "requires a GPU (authorized CUDA with NVRTC 12.4+, or Apple7+ Metal)"]
fn issue_m4_04_gpu_segmented_families_match_the_source_bit_exact() {
    let mut device: Device = open();
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
        let result = run_segmented(&mut device, &plan, &case.input, &case.offsets)
            .unwrap_or_else(|e| panic!("{}: {e}", case.name));
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

// One family at a time: the largest buffers are 1 GiB each, and aggregate
// device allocation is checked against the queried device working set.
#[test]
#[ignore = "requires GPU hardware, sufficient memory and shared runtime caps"]
fn issue_m4_04_gpu_size_cap_bit_exact() {
    let mut device: Device = open();
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
        let result = run_segmented(&mut device, &plan, &case.input, &case.offsets)
            .unwrap_or_else(|e| panic!("{} elements={n}: {e}", case.name));
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
