//! Live CUDA fused maps; the CPU graph and verifier tests stay in `tests/device/fused_map_device_plans.rs`.
//! Issue label M3-03.
use invar::{admission, device_ir};
use invar_device::{Capabilities, Error as VerifyError, Layout, Provenance};

#[path = "../support/m3_03_programs.rs"]
mod m3_03_programs;
use m3_03_programs::{CHAIN, LARGE_CHAIN, ZIP_DAG, checked_program};

#[test]
#[ignore = "requires a CUDA GPU and NVRTC 12.4+"]
fn gpu_fused_maps_lengths_and_zip_dag() {
    use invar_cuda::{
        Error as CudaError,
        cuda::{CudaDevice, FusedFault},
    };

    let compiler = std::path::Path::new(env!("CARGO_BIN_EXE_invar-nvrtc"));
    let mut device =
        CudaDevice::open().expect("CUDA_UNAVAILABLE: requires CUDA GPU and NVRTC 12.4+");
    let cases = [(CHAIN, false), (ZIP_DAG, true)];
    for (source, zip) in cases {
        let (_, program) = checked_program(source);
        let report = admission::analyze(source.as_bytes(), &program).unwrap();
        let region = &report.regions()[0];
        for len in [0usize, 1, 31, 32, 33, 255, 256, 257, 4093] {
            let ir = device_ir::from_region(region, len as u32).unwrap();
            let origin = Provenance {
                source_identity: ir.source_identity,
                graph_revision: ir.graph_revision,
            };
            let verified = invar_device::verify(ir, origin, device.capabilities()).unwrap();
            let inputs = (0..if zip { 3 } else { 1 })
                .map(|input| {
                    (0..len)
                        .map(|index| (index as u32).wrapping_mul(17).wrapping_add(input * 3 + 1))
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>();
            let borrowed = inputs.iter().map(Vec::as_slice).collect::<Vec<_>>();
            let result = device
                .run_fused_map(&verified, &borrowed, compiler)
                .unwrap();
            let expected = if zip {
                let first = (0..len)
                    .map(|i| inputs[0][i].wrapping_add(inputs[1][i]))
                    .collect::<Vec<_>>();
                let second = (0..len)
                    .map(|i| first[i].wrapping_mul(inputs[2][i]))
                    .collect::<Vec<_>>();
                vec![first, second]
            } else {
                let first = inputs[0]
                    .iter()
                    .map(|n| n.wrapping_add(3))
                    .collect::<Vec<_>>();
                let second = first.iter().map(|n| n.wrapping_mul(7)).collect::<Vec<_>>();
                let third = second
                    .iter()
                    .map(|n| n.wrapping_add(11))
                    .collect::<Vec<_>>();
                vec![first, second, third]
            };
            assert_eq!(result.outputs, expected, "zip={zip} len={len}");
            assert!(result.host_to_host_ms.is_finite());
            assert!(
                result.h2d_ms.is_finite()
                    && result.kernel_ms.is_finite()
                    && result.d2h_ms.is_finite()
            );
            eprintln!(
                "family={} len={} device={} caps={:?} source_hash={:016x} cubin_bytes={} geometry={}x{} transfer_bytes={} compile_ms={} host_to_host_ms={} h2d_ms={} kernel_ms={} d2h_ms={} residency=host-to-host completion=event-confirmed nvrtc_options=--gpu-architecture=sm_{},--std=c++17 nvrtc_log={:?}",
                if zip { "zip-dag" } else { "chain" },
                len,
                device.name(),
                device.capabilities(),
                result.source_hash,
                result.cubin_bytes,
                result.blocks,
                result.lanes,
                result.transfer_bytes,
                result.compile_ms,
                result.host_to_host_ms,
                result.h2d_ms,
                result.kernel_ms,
                result.d2h_ms,
                match device.capabilities() {
                    Capabilities::Cuda { sm, .. } => sm,
                    _ => unreachable!(),
                },
                result.nvrtc_log
            );
        }
    }
    let (_, program) = checked_program(CHAIN);
    let report = admission::analyze(CHAIN.as_bytes(), &program).unwrap();
    let ir = device_ir::from_region(&report.regions()[0], 3).unwrap();
    let origin = Provenance {
        source_identity: ir.source_identity,
        graph_revision: ir.graph_revision,
    };
    let verified = invar_device::verify(ir, origin, device.capabilities()).unwrap();
    let input = [1u32, 2, 3];
    let expected_cpu = match &report.regions()[0].execution() {
        invar::plan::Execution::DagProducer(chain) => chain.plan.run_dag(&[&input], 0..3).unwrap(),
        _ => panic!("missing graph-extracted chain"),
    };
    let mut unknown_layout = verified.ir().clone();
    unknown_layout.dag.as_mut().unwrap().inputs[0].layout = Layout::Unknown;
    unknown_layout.input.layout = Layout::Unknown;
    assert_eq!(
        invar_device::verify(unknown_layout, origin, device.capabilities()).err(),
        Some(VerifyError::Layout)
    );
    assert_eq!(
        device
            .run_fused_map(&verified, &[&input], compiler)
            .unwrap()
            .outputs,
        expected_cpu
    );
    eprintln!(
        "unknown_layout=pre_submit_reject cpu_fallback=reference-equal stale_device_use=none"
    );
    for (fault, expected) in [
        (FusedFault::FalseProfitability, CudaError::Unprofitable),
        (FusedFault::TransferOutOfMemory, CudaError::OutOfMemory),
        (FusedFault::LaunchFailure, CudaError::InvalidHandle),
    ] {
        assert_eq!(
            device
                .run_fused_map_with_fault(&verified, &[&input], compiler, fault)
                .err(),
            Some(expected)
        );
        assert_eq!(
            device
                .run_fused_map(&verified, &[&input], compiler)
                .unwrap()
                .outputs,
            expected_cpu
        );
        eprintln!(
            "pre_submit_fault={fault:?} cpu_fallback=reference-equal completion=no-device-output"
        );
    }
    let mut unknown = CudaDevice::open().unwrap();
    assert_eq!(
        unknown
            .run_fused_map_with_fault(
                &verified,
                &[&input],
                compiler,
                FusedFault::PartialCompletion,
            )
            .err(),
        Some(CudaError::Timeout)
    );
    // Unconfirmed work stays quarantined until its stream fence completes;
    // only then may a fresh submission run.
    assert!(unknown.is_poisoned());
    assert_eq!(unknown.quarantine_len(), 1);
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
    while unknown.quarantine_len() > 0 {
        assert!(
            std::time::Instant::now() < deadline,
            "CUDA_PENDING: fence never completed"
        );
        unknown.reap();
        std::thread::yield_now();
    }
    assert!(!unknown.is_poisoned());
    let fresh = unknown
        .run_fused_map(&verified, &[&input], compiler)
        .unwrap();
    assert_eq!(
        fresh.outputs.len(),
        verified.ir().dag.as_ref().unwrap().outputs.len()
    );
    eprintln!(
        "partial_completion=unknown quarantine=reaped-after-fence publication=none cpu_replay=none"
    );
}

#[test]
#[ignore = "requires a CUDA GPU and NVRTC 12.4+"]
fn gpu_bandwidth_gate() {
    use invar_cuda::cuda::CudaDevice;

    let compiler = std::path::Path::new(env!("CARGO_BIN_EXE_invar-nvrtc"));
    let (_, program) = checked_program(LARGE_CHAIN);
    let report = admission::analyze(LARGE_CHAIN.as_bytes(), &program).unwrap();
    let ir = device_ir::from_region(&report.regions()[0], 64 * 1024 * 1024).unwrap();
    let origin = Provenance {
        source_identity: ir.source_identity,
        graph_revision: ir.graph_revision,
    };
    let mut device =
        CudaDevice::open().expect("CUDA_UNAVAILABLE: requires CUDA GPU and NVRTC 12.4+");
    let verified = invar_device::verify(ir, origin, device.capabilities()).unwrap();
    let result = device.benchmark_fused_chain(&verified, compiler).unwrap();
    eprintln!("resident_benchmark={result:?}");
    assert!(result.outputs_correct);
    assert!(
        result.bandwidth_fraction >= 0.80,
        "copy bandwidth fraction={}",
        result.bandwidth_fraction
    );
    assert!(
        result.generated_to_handwritten_ratio <= 1.25,
        "handwritten ratio={}",
        result.generated_to_handwritten_ratio
    );
}
