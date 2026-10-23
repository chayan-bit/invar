//! M4-04 Metal qualification: the shared segmented tests plus the Metal refusals
//! and the unconfirmed completion.
use invar_device::segmented as seg;
use invar_metal::{Error, Fault, MetalDevice};

#[path = "../support/m4_04_cases.rs"]
mod cases;

type Device = MetalDevice;

fn open() -> Device {
    MetalDevice::open_runtime_source().expect("explicit Metal qualification: no Metal device")
}

fn run_segmented(
    device: &mut Device,
    plan: &seg::Verified,
    input: &[u32],
    offsets: &[u32],
) -> Result<body::Out, String> {
    let r = device
        .run_segmented(plan, input, offsets)
        .map_err(|e| format!("{e:?}"))?;
    Ok(body::Out {
        output: r.output,
        kernel_ms: r.kernel_ms,
        compile_ms: r.compile_ms,
        transfer_bytes: r.transfer_bytes,
        source_hash: r.source_hash,
    })
}

#[path = "../support/gpu_m4_04.rs"]
mod body;

#[test]
#[ignore = "requires a local Apple7+ Metal GPU"]
fn issue_m4_04_metal_gpu_refusals_and_unconfirmed_completion_publish_nothing() {
    let mut device = open();
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
