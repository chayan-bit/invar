use invar::{
    admission, check::check, device_ir, eval, front::parse, mir, mir_eval, plan::PlanKind,
};
use invar::{laws, plan::Execution, typed_net::Prefix};
use invar_device::{
    Barrier, Capabilities, CollectiveKind, Error as VerifyError, Layout, Numeric, Provenance,
};

const SUM: &str = "fn main() -> i32 { let xs: Array<u32> = [1u32,2u32,3u32]; var acc: u32 = 7u32; for i in 0u32 .. len(xs) { acc = wrapping_add(acc,xs[i]); } if acc == 13u32 { return 0; } return 1; }";
const PRODUCT: &str = "fn main() -> i32 { let xs: Array<u32> = [2u32,3u32,5u32]; var acc: u32 = 3u32; for i in 0u32 .. len(xs) { acc = wrapping_mul(acc,xs[i]); } if acc == 90u32 { return 0; } return 1; }";
const INCLUSIVE: &str = "fn main() -> i32 { let aa: Array<u32> = [2u32,3u32,5u32]; let bb: Array<u32> = [7u32,11u32,13u32]; var state: u32 = 17u32; var out: Array<u32> = filled_u32(len(aa),0u32); for i in 0u32 .. len(aa) { state = wrapping_add(wrapping_mul(aa[i],state),bb[i]); out[i] = state; } if out[2u32] == 683u32 { return 0; } return 1; }";
const EXCLUSIVE: &str = "fn main() -> i32 { let aa: Array<u32> = [2u32,3u32,5u32]; let bb: Array<u32> = [7u32,11u32,13u32]; var state: u32 = 17u32; var out: Array<u32> = filled_u32(len(aa),0u32); for i in 0u32 .. len(aa) { out[i] = state; state = wrapping_add(wrapping_mul(aa[i],state),bb[i]); } if out[0u32] == 17u32 { return 0; } return 1; }";
const FINAL: &str = "fn main() -> i32 { let aa: Array<u32> = [2u32,3u32,5u32]; let bb: Array<u32> = [7u32,11u32,13u32]; var state: u32 = 17u32; for i in 0u32 .. len(aa) { state = wrapping_add(wrapping_mul(aa[i],state),bb[i]); } if state == 683u32 { return 0; } return 1; }";

#[test]
fn issue_m3_04_cuda_reductions_scans_graph_plans_lower_to_shared_ir() {
    for (source, kind) in [
        (SUM, PlanKind::Reduction),
        (PRODUCT, PlanKind::Reduction),
        (INCLUSIVE, PlanKind::Scan),
        (EXCLUSIVE, PlanKind::Scan),
        (FINAL, PlanKind::Scan),
    ] {
        let checked = check(parse(source.as_bytes()).unwrap(), true).unwrap();
        let program = mir::lower(&checked).unwrap();
        let report = admission::analyze(source.as_bytes(), &program).unwrap();
        let region = report
            .regions()
            .iter()
            .find(|r| r.plan_kind() == kind)
            .unwrap();
        assert_eq!(
            eval::observe(&checked, 100_000).result,
            Ok(eval::Value::I32(0))
        );
        assert_eq!(
            mir_eval::observe(&program, 100_000).result,
            Ok(eval::Value::I32(0))
        );
        let ir = device_ir::from_region(region, 3).expect("graph-extracted collective must lower");
        let origin = Provenance {
            source_identity: ir.source_identity,
            graph_revision: ir.graph_revision,
        };
        let verified = invar_device::verify(ir, origin, Capabilities::cuda_12_9_sm89()).unwrap();
        let source = invar_device::emit_cuda(&verified).unwrap();
        assert!(source.contains("invar_block_v1"));
        assert!(source.contains("invar_propagate_v1"));
        assert!(source.contains("invar_writeback_v1"));
        assert_eq!(source, invar_device::emit_cuda(&verified).unwrap());
    }
}

#[test]
fn issue_m3_04_cuda_reductions_scans_verifier_rejects_invalid_collectives() {
    let program = mir::lower(&check(parse(SUM.as_bytes()).unwrap(), true).unwrap()).unwrap();
    let report = admission::analyze(SUM.as_bytes(), &program).unwrap();
    let base = device_ir::from_region(&report.regions()[0], 3).unwrap();
    let origin = Provenance {
        source_identity: base.source_identity,
        graph_revision: base.graph_revision,
    };
    let mut wrong_identity = base.clone();
    wrong_identity.collective.as_mut().unwrap().kind = CollectiveKind::Reduction {
        law: invar_device::ReduceLaw::WrappingSum,
        identity: 1,
    };
    assert_eq!(
        invar_device::verify(wrong_identity, origin, Capabilities::cuda_12_9_sm89()).err(),
        Some(VerifyError::Operations)
    );
    let mut layout = base.clone();
    layout.input.layout = Layout::Unknown;
    layout.collective.as_mut().unwrap().inputs[0].layout = Layout::Unknown;
    assert_eq!(
        invar_device::verify(layout, origin, Capabilities::cuda_12_9_sm89()).err(),
        Some(VerifyError::Layout)
    );
    let mut overread = base.clone();
    overread.input.elements = 2;
    overread.collective.as_mut().unwrap().inputs[0].elements = 2;
    assert_eq!(
        invar_device::verify(overread, origin, Capabilities::cuda_12_9_sm89()).err(),
        Some(VerifyError::Bounds)
    );
    let mut geometry = base.clone();
    geometry.geometry.blocks = 2;
    assert_eq!(
        invar_device::verify(geometry, origin, Capabilities::cuda_12_9_sm89()).err(),
        Some(VerifyError::Operations)
    );
    let mut barrier = base.clone();
    barrier.barrier = Barrier::None;
    assert_eq!(
        invar_device::verify(barrier, origin, Capabilities::cuda_12_9_sm89()).err(),
        Some(VerifyError::Barrier)
    );
    let mut checked = base;
    checked.numeric = Numeric::CheckedU32;
    assert_eq!(
        invar_device::verify(checked, origin, Capabilities::cuda_12_9_sm89()).err(),
        Some(VerifyError::Numeric)
    );

    let program = mir::lower(&check(parse(INCLUSIVE.as_bytes()).unwrap(), true).unwrap()).unwrap();
    let report = admission::analyze(INCLUSIVE.as_bytes(), &program).unwrap();
    let mut scan = device_ir::from_region(&report.regions()[0], 4097).unwrap();
    let scan_origin = Provenance {
        source_identity: scan.source_identity,
        graph_revision: scan.graph_revision,
    };
    assert_eq!(scan.geometry.lanes, invar_device::collective::SCAN_LANES);
    scan.geometry.lanes = invar_device::collective::LANES;
    assert_eq!(
        invar_device::verify(scan, scan_origin, Capabilities::cuda_12_9_sm89()).err(),
        Some(VerifyError::Operations)
    );
}

#[test]
fn issue_m3_04_cuda_reductions_scans_cpu_serial_shape_corpus() {
    let plans = [SUM, PRODUCT].map(|source| {
        let program = mir::lower(&check(parse(source.as_bytes()).unwrap(), true).unwrap()).unwrap();
        let report = admission::analyze(source.as_bytes(), &program).unwrap();
        let Execution::Reduction(reduction) = report.regions()[0].execution() else {
            panic!("missing reduction")
        };
        reduction.clone()
    });
    let scans = [INCLUSIVE, EXCLUSIVE, FINAL].map(|source| {
        let program = mir::lower(&check(parse(source.as_bytes()).unwrap(), true).unwrap()).unwrap();
        let report = admission::analyze(source.as_bytes(), &program).unwrap();
        let Execution::Scan(scan) = report.regions()[0].execution() else {
            panic!("missing scan")
        };
        scan.clone()
    });
    for len in [
        0usize, 1, 31, 32, 33, 255, 256, 257, 4093, 4096, 4097, 10003,
    ] {
        let values = (0..len)
            .map(|i| (i as u32).wrapping_mul(17).wrapping_add(3))
            .collect::<Vec<_>>();
        for (law, plan) in [laws::Law::WrappingSum, laws::Law::WrappingProduct]
            .into_iter()
            .zip(&plans)
        {
            let serial = values
                .iter()
                .fold(19u32, |state, value| law.combine(state, *value));
            assert_eq!(
                plan.run(&values, 19).unwrap(),
                serial,
                "reduction len={len}"
            );
        }
        let aa = (0..len)
            .map(|i| (i as u32).wrapping_mul(3).wrapping_add(2))
            .collect::<Vec<_>>();
        let bb = (0..len)
            .map(|i| (i as u32).wrapping_mul(7).wrapping_add(5))
            .collect::<Vec<_>>();
        let arrays: [&[u32]; 2] = [&aa, &bb];
        let mut state = 17u32;
        let mut inclusive = Vec::new();
        let mut exclusive = Vec::new();
        for (&a, &b) in aa.iter().zip(&bb) {
            exclusive.push(state);
            state = a.wrapping_mul(state).wrapping_add(b);
            inclusive.push(state);
        }
        let empty = Vec::new();
        for (scan, expected) in scans.iter().zip([&inclusive, &exclusive, &empty]) {
            assert_eq!(
                scan.final_state(&arrays, 17, len).unwrap(),
                state,
                "scan final len={len}"
            );
            if scan.prefix() != Prefix::None {
                assert_eq!(
                    scan.published_states(&arrays, 17, len).unwrap(),
                    *expected,
                    "scan prefix len={len}"
                );
            }
        }
    }
}

#[cfg(feature = "cuda")]
#[test]
#[ignore = "requires a CUDA GPU and NVRTC 12.4+"]
fn issue_m3_04_gpu_reductions_scans_differential() {
    use invar_cuda::{
        Error as CudaError,
        cuda::{CollectiveFault, CudaDevice},
    };
    use invar_device::PrefixMode;
    let compiler = std::path::Path::new(env!("CARGO_BIN_EXE_invar-nvrtc"));
    let mut device =
        CudaDevice::open().expect("CUDA_UNAVAILABLE: requires CUDA GPU and NVRTC 12.4+");
    for (source, law, seed) in [
        (SUM, laws::Law::WrappingSum, 7u32),
        (PRODUCT, laws::Law::WrappingProduct, 3u32),
    ] {
        let program = mir::lower(&check(parse(source.as_bytes()).unwrap(), true).unwrap()).unwrap();
        let report = admission::analyze(source.as_bytes(), &program).unwrap();
        for len in [0usize, 1, 255, 256, 257, 4093, 4095, 4096, 4097, 10003] {
            let ir = device_ir::from_region(&report.regions()[0], len as u32).unwrap();
            let origin = Provenance {
                source_identity: ir.source_identity,
                graph_revision: ir.graph_revision,
            };
            let verified = invar_device::verify(ir, origin, device.capabilities()).unwrap();
            let values = (0..len)
                .map(|i| (i as u32).wrapping_mul(17).wrapping_add(3))
                .collect::<Vec<_>>();
            let expected = values
                .iter()
                .fold(seed, |state, value| law.combine(state, *value));
            let result = device
                .run_collective(&verified, &[&values], seed, compiler)
                .unwrap();
            assert_eq!(result.final_state, expected, "law={law:?} len={len}");
            assert!(result.prefixes.is_empty());
            eprintln!(
                "law={} identity={:?} len={} output={} source_hash={:016x} cubin_bytes={} geometry={}x{} transfer_bytes={} combines={} depth={} compile_ms={} host_to_host_ms={} h2d_ms={} block_ms={} propagate_ms={} writeback_ms={} d2h_ms={} completion=event-confirmed nvrtc_log={:?}",
                result.law_id,
                result.identity,
                len,
                result.final_state,
                result.source_hash,
                result.cubin_bytes,
                result.blocks,
                result.lanes,
                result.transfer_bytes,
                result.combine_count,
                result.scan_depth,
                result.compile_ms,
                result.host_to_host_ms,
                result.h2d_ms,
                result.block_ms,
                result.propagate_ms,
                result.writeback_ms,
                result.d2h_ms,
                result.nvrtc_log
            );
        }
    }
    for source in [INCLUSIVE, EXCLUSIVE, FINAL] {
        let program = mir::lower(&check(parse(source.as_bytes()).unwrap(), true).unwrap()).unwrap();
        let report = admission::analyze(source.as_bytes(), &program).unwrap();
        for len in [0usize, 1, 255, 256, 257, 4093, 4095, 4096, 4097, 10003] {
            let ir = device_ir::from_region(&report.regions()[0], len as u32).unwrap();
            let origin = Provenance {
                source_identity: ir.source_identity,
                graph_revision: ir.graph_revision,
            };
            let prefix = match ir.collective.as_ref().unwrap().kind {
                CollectiveKind::AffineScan { prefix, .. } => prefix,
                _ => panic!("not affine"),
            };
            let verified = invar_device::verify(ir, origin, device.capabilities()).unwrap();
            let aa = (0..len)
                .map(|i| (i as u32).wrapping_mul(3).wrapping_add(2))
                .collect::<Vec<_>>();
            let bb = (0..len)
                .map(|i| (i as u32).wrapping_mul(7).wrapping_add(5))
                .collect::<Vec<_>>();
            let mut state = 17u32;
            let mut expected = Vec::new();
            for (&a, &b) in aa.iter().zip(&bb) {
                if prefix == PrefixMode::Exclusive {
                    expected.push(state);
                }
                state = a.wrapping_mul(state).wrapping_add(b);
                if prefix == PrefixMode::Inclusive {
                    expected.push(state);
                }
            }
            let result = device
                .run_collective(&verified, &[&aa, &bb], 17, compiler)
                .unwrap();
            assert_eq!(result.final_state, state, "prefix={prefix:?} len={len}");
            assert_eq!(result.prefixes, expected, "prefix={prefix:?} len={len}");
            eprintln!(
                "law={} prefix={prefix:?} identity={:?} len={} output={} source_hash={:016x} cubin_bytes={} geometry={}x{} transfer_bytes={} combines={} depth={} compile_ms={} host_to_host_ms={} h2d_ms={} block_ms={} propagate_ms={} writeback_ms={} d2h_ms={} completion=event-confirmed nvrtc_log={:?}",
                result.law_id,
                result.identity,
                len,
                result.final_state,
                result.source_hash,
                result.cubin_bytes,
                result.blocks,
                result.lanes,
                result.transfer_bytes,
                result.combine_count,
                result.scan_depth,
                result.compile_ms,
                result.host_to_host_ms,
                result.h2d_ms,
                result.block_ms,
                result.propagate_ms,
                result.writeback_ms,
                result.d2h_ms,
                result.nvrtc_log
            );
        }
    }
    let program = mir::lower(&check(parse(SUM.as_bytes()).unwrap(), true).unwrap()).unwrap();
    let report = admission::analyze(SUM.as_bytes(), &program).unwrap();
    let ir = device_ir::from_region(&report.regions()[0], 3).unwrap();
    let origin = Provenance {
        source_identity: ir.source_identity,
        graph_revision: ir.graph_revision,
    };
    let verified = invar_device::verify(ir, origin, device.capabilities()).unwrap();
    let values = [1u32, 2, 3];
    for (fault, expected) in [
        (CollectiveFault::OutOfMemory, CudaError::OutOfMemory),
        (CollectiveFault::LaunchFailure, CudaError::InvalidHandle),
    ] {
        assert_eq!(
            device
                .run_collective_with_fault(&verified, &[&values], 7, compiler, fault)
                .err(),
            Some(expected)
        );
        assert_eq!(
            device
                .run_collective(&verified, &[&values], 7, compiler)
                .unwrap()
                .final_state,
            13
        );
        eprintln!(
            "collective_pre_submit_fault={fault:?} cpu_fallback=reference-equal stale_device_use=none"
        );
    }
    let mut unknown = CudaDevice::open().unwrap();
    assert_eq!(
        unknown
            .run_collective_with_fault(
                &verified,
                &[&values],
                7,
                compiler,
                CollectiveFault::PartialCompletion
            )
            .err(),
        Some(CudaError::Timeout)
    );
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
    let fresh = unknown
        .run_collective(&verified, &[&values], 7, compiler)
        .unwrap();
    assert_eq!(
        fresh.final_state,
        values
            .iter()
            .fold(7u32, |state, value| state.wrapping_add(*value))
    );
    eprintln!(
        "collective_partial_completion=unknown quarantine=reaped-after-fence publication=none cpu_replay=none"
    );
}

#[cfg(feature = "cuda")]
#[test]
#[ignore = "requires a CUDA GPU with memory for 64 Mi benchmark"]
fn issue_m3_04_gpu_resident_benchmark() {
    use invar_cuda::cuda::CudaDevice;
    let compiler = std::path::Path::new(env!("CARGO_BIN_EXE_invar-nvrtc"));
    let mut device =
        CudaDevice::open().expect("CUDA_UNAVAILABLE: requires CUDA GPU and NVRTC 12.4+");
    let mut gate_failures = Vec::new();
    for (source, seed, minimum) in [(SUM, 7u32, 0.75), (INCLUSIVE, 17u32, 0.60)] {
        let program = mir::lower(&check(parse(source.as_bytes()).unwrap(), true).unwrap()).unwrap();
        let report = admission::analyze(source.as_bytes(), &program).unwrap();
        let ir = device_ir::from_region(&report.regions()[0], 64 * 1024 * 1024).unwrap();
        let origin = Provenance {
            source_identity: ir.source_identity,
            graph_revision: ir.graph_revision,
        };
        let verified = invar_device::verify(ir, origin, device.capabilities()).unwrap();
        if let Ok(directory) = std::env::var("INVAR_M3_04_SOURCE_DIR") {
            std::fs::create_dir_all(&directory).unwrap();
            let name = if source == SUM {
                "sum"
            } else {
                "inclusive-scan"
            };
            std::fs::write(
                format!("{directory}/generated-{name}.cu"),
                invar_device::emit_cuda(&verified).unwrap(),
            )
            .unwrap();
        }
        let result = device
            .benchmark_collective(&verified, seed, compiler)
            .unwrap();
        eprintln!(
            "m3_04_resident law={:?} len={} geometry={}x{} workspace_bytes={} warmups={} samples={} copy={:?} generated={:?} handwritten={:?} copy_gbps={} generated_gbps={} bandwidth_fraction={} generated_to_handwritten_ratio={} source_hash={:016x} handwritten_hash={:016x} all_outputs_correct={}",
            verified.ir().collective.as_ref().unwrap().kind,
            result.length,
            result.blocks,
            result.lanes,
            result.workspace_bytes,
            result.warmups,
            result.samples,
            result.copy,
            result.generated,
            result.handwritten,
            result.copy_gbps,
            result.generated_gbps,
            result.bandwidth_fraction,
            result.generated_to_handwritten_ratio,
            result.generated_hash,
            result.handwritten_hash,
            result.outputs_correct,
        );
        assert!(result.outputs_correct);
        if result.bandwidth_fraction < minimum || result.generated_to_handwritten_ratio > 1.25 {
            gate_failures.push(format!(
                "{source}: bandwidth fraction {} below {minimum} or handwritten ratio {} above 1.25",
                result.bandwidth_fraction, result.generated_to_handwritten_ratio
            ));
        }
    }
    assert!(gate_failures.is_empty(), "{}", gate_failures.join("; "));
}

#[cfg(feature = "cuda")]
#[test]
#[ignore = "requires CUDA sanitizer and GPU"]
fn issue_m3_04_gpu_sanitizer_ragged_scan() {
    use invar_cuda::cuda::CudaDevice;
    let compiler = std::path::Path::new(env!("CARGO_BIN_EXE_invar-nvrtc"));
    let program = mir::lower(&check(parse(INCLUSIVE.as_bytes()).unwrap(), true).unwrap()).unwrap();
    let report = admission::analyze(INCLUSIVE.as_bytes(), &program).unwrap();
    let len = 4097usize;
    let ir = device_ir::from_region(&report.regions()[0], len as u32).unwrap();
    let origin = Provenance {
        source_identity: ir.source_identity,
        graph_revision: ir.graph_revision,
    };
    let aa = (0..len).map(|i| i as u32 + 1).collect::<Vec<_>>();
    let bb = (0..len).map(|i| i as u32 + 3).collect::<Vec<_>>();
    let mut expected = Vec::with_capacity(len);
    let mut state = 17u32;
    for (&a, &b) in aa.iter().zip(&bb) {
        state = a.wrapping_mul(state).wrapping_add(b);
        expected.push(state);
    }
    let mut device =
        CudaDevice::open().expect("CUDA_UNAVAILABLE: requires CUDA GPU and NVRTC 12.4+");
    let verified = invar_device::verify(ir, origin, device.capabilities()).unwrap();
    let result = device
        .run_collective(&verified, &[&aa, &bb], 17, compiler)
        .unwrap();
    assert_eq!(result.prefixes, expected);
    assert_eq!(result.final_state, state);
    eprintln!(
        "m3_04_sanitizer_case length={len} blocks={} completion=event-confirmed",
        result.blocks
    );
}

#[cfg(feature = "cuda")]
#[test]
#[ignore = "requires a CUDA GPU and NVRTC 12.4+"]
fn issue_m3_04_gpu_cold_process() {
    use invar_cuda::cuda::CudaDevice;
    if std::env::var_os("INVAR_M3_04_COLD_CHILD").is_some() {
        let compiler = std::path::Path::new(env!("CARGO_BIN_EXE_invar-nvrtc"));
        let program = mir::lower(&check(parse(SUM.as_bytes()).unwrap(), true).unwrap()).unwrap();
        let report = admission::analyze(SUM.as_bytes(), &program).unwrap();
        let ir = device_ir::from_region(&report.regions()[0], 4097).unwrap();
        let origin = Provenance {
            source_identity: ir.source_identity,
            graph_revision: ir.graph_revision,
        };
        let input = vec![1u32; 4097];
        let mut device =
            CudaDevice::open().expect("CUDA_UNAVAILABLE: requires CUDA GPU and NVRTC 12.4+");
        let verified = invar_device::verify(ir, origin, device.capabilities()).unwrap();
        let result = device
            .run_collective(&verified, &[&input], 7, compiler)
            .unwrap();
        assert_eq!(result.final_state, 4104);
        eprintln!(
            "cold_child compile_ms={} host_to_host_ms={} cubin_bytes={} h2d_ms={} block_ms={} propagate_ms={} writeback_ms={} d2h_ms={}",
            result.compile_ms,
            result.host_to_host_ms,
            result.cubin_bytes,
            result.h2d_ms,
            result.block_ms,
            result.propagate_ms,
            result.writeback_ms,
            result.d2h_ms,
        );
        return;
    }
    let started = std::time::Instant::now();
    let child = std::process::Command::new(std::env::current_exe().unwrap())
        .arg("--exact")
        .arg("issue_m3_04_gpu_cold_process")
        .arg("--ignored")
        .arg("--nocapture")
        .env("INVAR_M3_04_COLD_CHILD", "1")
        .output()
        .unwrap();
    let cold_process_ms = started.elapsed().as_secs_f64() * 1000.0;
    assert!(
        child.status.success(),
        "{}",
        String::from_utf8_lossy(&child.stderr)
    );
    eprintln!(
        "cold_process_wall_ms={cold_process_ms} child_stdout={:?} child_stderr={:?}",
        String::from_utf8_lossy(&child.stdout),
        String::from_utf8_lossy(&child.stderr)
    );
}
