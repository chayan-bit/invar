//! M4-04 Metal qualification: segmented map, segmented reduction and stable
//! compaction from the same verified IR as CUDA, executed on the local Apple
//! GPU and compared bit for bit with the source expectation (CPU net oracle and
//! reference evaluator), including maximum sizes, admission refusals and an
//! unconfirmed completion. Ignored by default; when run explicitly it fails,
//! never skips, without a Metal device.
#![cfg(all(feature = "metal", target_os = "macos"))]
#![forbid(unsafe_code)]

#[path = "support/m4_04_cases.rs"]
mod cases;

use invar_device::segmented as seg;
use invar_metal::{Error, Fault, MetalDevice};

#[test]
#[ignore = "requires a local Apple7+ Metal GPU"]
fn issue_m4_04_metal_gpu_segmented_families_match_the_source_bit_exact() {
    let mut device =
        MetalDevice::open_runtime_source().expect("explicit Metal qualification: no Metal device");
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
            .run_segmented(&plan, &case.input, &case.offsets)
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
#[ignore = "requires a local Apple7+ Metal GPU"]
fn issue_m4_04_metal_gpu_refusals_and_unconfirmed_completion_publish_nothing() {
    let mut device =
        MetalDevice::open_runtime_source().expect("explicit Metal qualification: no Metal device");
    let caps = device.capabilities();
    let case = cases::cases(false)
        .into_iter()
        .find(|case| case.name == "Map/empty-interior")
        .unwrap();
    let plan = cases::plan(&case, caps);
    let mut decreasing = case.offsets.clone();
    // [0, 3, 3, ...] becomes [0, 5, 3, ...]: decreasing offsets.
    decreasing[1] = decreasing[2] + 2;
    assert!(matches!(
        device.run_segmented(&plan, &case.input, &decreasing),
        Err(Error::Limit)
    ));
    for fault in [Fault::OutOfMemory, Fault::LaunchFailure] {
        assert!(
            device
                .run_segmented_with_fault(&plan, &case.input, &case.offsets, fault)
                .is_err()
        );
        assert_eq!(device.quarantine_len(), 0);
    }
    // Submitted but never confirmed: no output, the work is quarantined and
    // dispatch is refused until it is reaped after its real completion.
    assert!(matches!(
        device.run_segmented_with_fault(
            &plan,
            &case.input,
            &case.offsets,
            Fault::PartialCompletion
        ),
        Err(Error::Pending)
    ));
    assert_eq!(device.quarantine_len(), 1);
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
    while device.quarantine_len() > 0 && std::time::Instant::now() < deadline {
        device.reap();
        std::thread::sleep(std::time::Duration::from_millis(5));
    }
    assert_eq!(device.quarantine_len(), 0);
    let result = device
        .run_segmented(&plan, &case.input, &case.offsets)
        .unwrap();
    assert_eq!(result.output, cases::expected(&case));
}

// One family at a time: the largest buffers are 1 GiB each, and aggregate
// device allocation is checked against the queried Metal working set.
#[test]
#[ignore = "requires GPU hardware and sufficient memory"]
fn issue_m4_04_metal_gpu_size_cap_bit_exact() {
    let mut device =
        MetalDevice::open_runtime_source().expect("explicit GPU size-cap qualification");
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
            .run_segmented(&plan, &case.input, &case.offsets)
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
