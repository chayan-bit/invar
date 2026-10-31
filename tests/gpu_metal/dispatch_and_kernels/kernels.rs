//! Generated map, DAG, reduction and scan kernels on a real Apple GPU.

use super::*;

#[test]
#[ignore = "requires a real Apple7+ Metal GPU"]
fn metal_gpu_generated_map_matches_reference() {
    let _gpu_lock = GPU_LOCK
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let checked = check::check(front::parse(MAP.as_bytes()).unwrap(), true).unwrap();
    let program = mir::lower(&checked).unwrap();
    let report = admission::analyze(MAP.as_bytes(), &program).unwrap();
    let ir = device_ir::from_region(&report.regions()[0], 3).unwrap();
    let origin = Provenance {
        source_identity: ir.source_identity,
        graph_revision: ir.graph_revision,
    };
    let mut device =
        invar_metal::MetalDevice::open_runtime_source().expect("Metal Apple7+ GPU required");
    eprintln!(
        "Metal preflight device={} caps={:?} binding=objc2-metal@0.3.2",
        device.name(),
        device.capabilities()
    );
    let verified = invar_device::verify(ir, origin, device.capabilities()).unwrap();
    let result = device.run_map(&verified, &[1, 2, 3]).unwrap();
    assert_eq!(result.values, [6, 7, 8]);
    assert!(result.source_hash != 0);
    assert!(result.kernel_ms.is_finite() && result.kernel_ms > 0.0);
    assert_eq!(result.transfer_bytes, 24);
    assert!(!result.cache_hit);
    let warm = device.run_map(&verified, &[1, 2, 3]).unwrap();
    assert!(warm.cache_hit);
    assert_eq!(warm.values, result.values);
    let mut distinct_ir = verified.ir().clone();
    distinct_ir.source_identity ^= 1;
    let distinct_origin = Provenance {
        source_identity: distinct_ir.source_identity,
        graph_revision: distinct_ir.graph_revision,
    };
    let distinct =
        invar_device::verify(distinct_ir, distinct_origin, device.capabilities()).unwrap();
    // Pipelines are keyed by the exact MSL and function: a plan that emits
    // the same kernel reuses its pipeline, a different kernel never does.
    let other = device.run_map(&distinct, &[1, 2, 3]).unwrap();
    assert!(other.cache_hit, "an identical kernel was compiled again");
    assert_eq!(other.values, result.values);
}

#[test]
#[ignore = "requires a real Apple7+ Metal GPU"]
fn metal_gpu_failures_publish_nothing_and_quarantine() {
    let _gpu_lock = GPU_LOCK
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    use invar_metal::{Error, Fault};
    let checked = check::check(front::parse(MAP.as_bytes()).unwrap(), true).unwrap();
    let program = mir::lower(&checked).unwrap();
    let report = admission::analyze(MAP.as_bytes(), &program).unwrap();
    let ir = device_ir::from_region(&report.regions()[0], 3).unwrap();
    let origin = Provenance {
        source_identity: ir.source_identity,
        graph_revision: ir.graph_revision,
    };
    let mut device =
        invar_metal::MetalDevice::open_runtime_source().expect("Metal Apple7+ GPU required");
    let verified = invar_device::verify(ir, origin, device.capabilities()).unwrap();
    assert!(matches!(
        device.run_map_with_fault(&verified, &[1, 2, 3], Fault::OutOfMemory),
        Err(Error::OutOfMemory)
    ));
    assert!(matches!(
        device.run_map_with_fault(&verified, &[1, 2, 3], Fault::LaunchFailure),
        Err(Error::Device(_))
    ));
    assert!(!device.is_poisoned());
    assert_eq!(device.quarantine_len(), 0);
    assert!(matches!(
        device.run_map_with_fault(&verified, &[1, 2, 3], Fault::PartialCompletion),
        Err(Error::Pending)
    ));
    assert!(device.is_poisoned());
    assert_eq!(device.quarantine_len(), 1);
    assert!(matches!(
        device.run_map(&verified, &[1, 2, 3]),
        Err(Error::Pending)
    ));
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
    while device.is_poisoned() && std::time::Instant::now() < deadline {
        device.reap();
        std::thread::sleep(std::time::Duration::from_millis(1));
    }
    assert!(
        !device.is_poisoned(),
        "submitted work did not reach a terminal state"
    );
    assert_eq!(device.quarantine_len(), 0);
    assert_eq!(
        device.run_map(&verified, &[1, 2, 3]).unwrap().values,
        [6, 7, 8]
    );
    for (fault, code) in [
        (Fault::OutOfMemory, "METAL_UNAVAILABLE"),
        (Fault::LaunchFailure, "METAL_UNAVAILABLE"),
        (Fault::PartialCompletion, "METAL_PENDING"),
    ] {
        let failed = invar::schedule::metal::run_source_with_device_fault(
            MAP.as_bytes(),
            &program,
            1_000_000,
            &mut device,
            fault,
        )
        .unwrap();
        assert!(
            failed.executed.is_empty(),
            "injected failure published a result"
        );
        assert!(
            matches!(failed.observation.result, Err(invar::eval::EvalError::Tool { code: observed, .. }) if observed == code)
        );
        if fault == Fault::PartialCompletion {
            assert!(device.is_poisoned());
        }
    }

    for (source, inputs) in [
        (SUM, vec![&[1, 2, 3][..]]),
        (INCLUSIVE, vec![&[2, 3, 5][..], &[7, 11, 13][..]]),
    ] {
        let program =
            mir::lower(&check::check(front::parse(source.as_bytes()).unwrap(), true).unwrap())
                .unwrap();
        let report = admission::analyze(source.as_bytes(), &program).unwrap();
        let ir = device_ir::from_region(&report.regions()[0], 3).unwrap();
        let origin = Provenance {
            source_identity: ir.source_identity,
            graph_revision: ir.graph_revision,
        };
        let mut collective_device = invar_metal::MetalDevice::open_runtime_source().unwrap();
        let verified = invar_device::verify(ir, origin, collective_device.capabilities()).unwrap();
        for fault in [Fault::OutOfMemory, Fault::LaunchFailure] {
            assert!(
                collective_device
                    .run_collective_with_fault(&verified, &inputs, 17, fault)
                    .is_err()
            );
            assert!(!collective_device.is_poisoned());
        }
        assert!(matches!(
            collective_device.run_collective_with_fault(
                &verified,
                &inputs,
                17,
                Fault::PartialCompletion
            ),
            Err(Error::Pending)
        ));
        assert!(collective_device.is_poisoned());
        assert_eq!(collective_device.quarantine_len(), 1);
        assert!(matches!(
            collective_device.run_collective(&verified, &inputs, 17),
            Err(Error::Pending)
        ));
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
        while collective_device.is_poisoned() && std::time::Instant::now() < deadline {
            collective_device.reap();
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
        assert!(!collective_device.is_poisoned());
        assert_eq!(collective_device.quarantine_len(), 0);
        assert!(
            collective_device
                .run_collective(&verified, &inputs, 17)
                .is_ok()
        );
    }
}

#[test]
#[ignore = "requires a real Apple7+ Metal GPU and 1.25 GiB resident buffers"]
fn metal_gpu_map_bandwidth_gate() {
    let _gpu_lock = GPU_LOCK
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let program =
        mir::lower(&check::check(front::parse(LARGE_CHAIN.as_bytes()).unwrap(), true).unwrap())
            .unwrap();
    let report = admission::analyze(LARGE_CHAIN.as_bytes(), &program).unwrap();
    let mut ir = device_ir::from_region(&report.regions()[0], 64 * 1024 * 1024).unwrap();
    device_ir::retile_metal(&mut ir);
    let origin = Provenance {
        source_identity: ir.source_identity,
        graph_revision: ir.graph_revision,
    };
    let mut device =
        invar_metal::MetalDevice::open_runtime_source().expect("Metal Apple7+ GPU required");
    let verified = invar_device::verify(ir, origin, device.capabilities()).unwrap();
    let result = device.benchmark_fused_chain(&verified).unwrap();
    eprintln!("Metal map resident benchmark: {result:?}");
    assert!(result.outputs_correct);
    assert!(
        result.bandwidth_fraction >= 0.80,
        "copy bandwidth fraction={}",
        result.bandwidth_fraction
    );
}

#[test]
#[ignore = "requires a real Apple7+ Metal GPU and 64 Mi resident buffers"]
fn metal_gpu_collective_bandwidth_gates() {
    let _gpu_lock = GPU_LOCK
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    for (source, threshold, name) in [(SUM, 0.75, "reduction"), (INCLUSIVE, 0.60, "scan")] {
        let program =
            mir::lower(&check::check(front::parse(source.as_bytes()).unwrap(), true).unwrap())
                .unwrap();
        let report = admission::analyze(source.as_bytes(), &program).unwrap();
        let mut ir = device_ir::from_region(&report.regions()[0], 64 * 1024 * 1024).unwrap();
        device_ir::retile_metal(&mut ir);
        let origin = Provenance {
            source_identity: ir.source_identity,
            graph_revision: ir.graph_revision,
        };
        let mut device =
            invar_metal::MetalDevice::open_runtime_source().expect("Metal Apple7+ GPU required");
        let verified = invar_device::verify(ir, origin, device.capabilities()).unwrap();
        let result = device.benchmark_collective(&verified, 17).unwrap();
        eprintln!("Metal {name} resident benchmark: {result:?}");
        assert!(result.outputs_correct);
        assert!(
            result.bandwidth_fraction >= threshold,
            "{name} copy bandwidth fraction={}",
            result.bandwidth_fraction
        );
    }
}

#[test]
#[ignore = "requires a real Apple7+ Metal GPU"]
fn metal_gpu_fused_zip_dag_matches_reference() {
    let _gpu_lock = GPU_LOCK
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let checked = check::check(front::parse(ZIP_DAG.as_bytes()).unwrap(), true).unwrap();
    let program = mir::lower(&checked).unwrap();
    let report = admission::analyze(ZIP_DAG.as_bytes(), &program).unwrap();
    let region = report
        .regions()
        .iter()
        .find(|region| matches!(region.execution(), invar::plan::Execution::DagProducer(_)))
        .unwrap();
    let ir = device_ir::from_region(region, 3).unwrap();
    let origin = Provenance {
        source_identity: ir.source_identity,
        graph_revision: ir.graph_revision,
    };
    let mut device =
        invar_metal::MetalDevice::open_runtime_source().expect("Metal Apple7+ GPU required");
    let verified = invar_device::verify(ir, origin, device.capabilities()).unwrap();
    let invar::plan::Execution::DagProducer(chain) = region.execution() else {
        unreachable!()
    };
    let arrays = chain
        .inputs
        .iter()
        .map(|name| match name.as_str() {
            "xs" => &[1, 2, 3][..],
            "ys" => &[4, 5, 6][..],
            "zs" => &[7, 8, 9][..],
            other => panic!("unexpected graph input: {other}"),
        })
        .collect::<Vec<_>>();
    let result = device.run_fused_map(&verified, &arrays).unwrap();
    assert_eq!(result.outputs, vec![vec![5, 7, 9], vec![35, 56, 81]]);
}

#[test]
#[ignore = "requires a real Apple7+ Metal GPU"]
fn metal_gpu_fused_map_length_corpus() {
    let _gpu_lock = GPU_LOCK
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let mut device =
        invar_metal::MetalDevice::open_runtime_source().expect("Metal Apple7+ GPU required");
    for (source, zip) in [(CHAIN, false), (ZIP_DAG, true)] {
        let program =
            mir::lower(&check::check(front::parse(source.as_bytes()).unwrap(), true).unwrap())
                .unwrap();
        let report = admission::analyze(source.as_bytes(), &program).unwrap();
        let region = &report.regions()[0];
        for len in [0usize, 1, 31, 32, 33, 255, 256, 257, 4093] {
            let mut ir = device_ir::from_region(region, len as u32).unwrap();
            device_ir::retile_metal(&mut ir);
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
            let result = device.run_fused_map(&verified, &borrowed).unwrap();
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
        }
    }
}

#[test]
#[ignore = "requires a real Apple7+ Metal GPU"]
fn metal_gpu_reductions_match_reference() {
    let _gpu_lock = GPU_LOCK
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    for (source, values, seed, expected) in [
        (SUM, &[1, 2, 3][..], 7, 13),
        (PRODUCT, &[2, 3, 5][..], 3, 90),
    ] {
        let checked = check::check(front::parse(source.as_bytes()).unwrap(), true).unwrap();
        let program = mir::lower(&checked).unwrap();
        let report = admission::analyze(source.as_bytes(), &program).unwrap();
        let region = report
            .regions()
            .iter()
            .find(|region| region.plan_kind() == invar::plan::PlanKind::Reduction)
            .unwrap();
        let ir = device_ir::from_region(region, 3).unwrap();
        let origin = Provenance {
            source_identity: ir.source_identity,
            graph_revision: ir.graph_revision,
        };
        let mut device =
            invar_metal::MetalDevice::open_runtime_source().expect("Metal Apple7+ GPU required");
        let verified = invar_device::verify(ir, origin, device.capabilities()).unwrap();
        let result = device.run_collective(&verified, &[values], seed).unwrap();
        assert_eq!(result.final_state, expected);
        assert!(result.kernel_ms.is_finite() && result.kernel_ms > 0.0);
    }
}

#[test]
#[ignore = "requires a real Apple7+ Metal GPU"]
fn metal_gpu_affine_scans_match_reference() {
    let _gpu_lock = GPU_LOCK
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    for (source, expected) in [
        (INCLUSIVE, vec![41, 134, 683]),
        (EXCLUSIVE, vec![17, 41, 134]),
        (FINAL, vec![]),
    ] {
        let checked = check::check(front::parse(source.as_bytes()).unwrap(), true).unwrap();
        let program = mir::lower(&checked).unwrap();
        let report = admission::analyze(source.as_bytes(), &program).unwrap();
        let region = report
            .regions()
            .iter()
            .find(|region| region.plan_kind() == invar::plan::PlanKind::Scan)
            .unwrap();
        let ir = device_ir::from_region(region, 3).unwrap();
        let origin = Provenance {
            source_identity: ir.source_identity,
            graph_revision: ir.graph_revision,
        };
        let mut device =
            invar_metal::MetalDevice::open_runtime_source().expect("Metal Apple7+ GPU required");
        let verified = invar_device::verify(ir, origin, device.capabilities()).unwrap();
        let result = device
            .run_collective(&verified, &[&[2, 3, 5], &[7, 11, 13]], 17)
            .unwrap();
        assert_eq!(result.final_state, 683);
        assert_eq!(result.prefixes, expected);
        assert!(result.kernel_ms.is_finite() && result.kernel_ms > 0.0);
    }
}

#[test]
#[ignore = "requires a real Apple7+ Metal GPU"]
fn metal_gpu_reduction_scan_length_corpus() {
    let _gpu_lock = GPU_LOCK
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    use invar_device::ReduceLaw;
    let mut device =
        invar_metal::MetalDevice::open_runtime_source().expect("Metal Apple7+ GPU required");
    let lengths = [
        0usize, 1, 31, 32, 33, 255, 256, 257, 4093, 4095, 4096, 4097, 10003,
    ];
    for (source, law, seed) in [
        (SUM, ReduceLaw::WrappingSum, 7u32),
        (PRODUCT, ReduceLaw::WrappingProduct, 3u32),
    ] {
        let program =
            mir::lower(&check::check(front::parse(source.as_bytes()).unwrap(), true).unwrap())
                .unwrap();
        let report = admission::analyze(source.as_bytes(), &program).unwrap();
        for len in lengths {
            let ir = device_ir::from_region(&report.regions()[0], len as u32).unwrap();
            let origin = Provenance {
                source_identity: ir.source_identity,
                graph_revision: ir.graph_revision,
            };
            let verified = invar_device::verify(ir, origin, device.capabilities()).unwrap();
            let values = (0..len)
                .map(|i| (i as u32).wrapping_mul(17).wrapping_add(3))
                .collect::<Vec<_>>();
            let expected = values.iter().fold(seed, |state, value| match law {
                ReduceLaw::WrappingSum => state.wrapping_add(*value),
                ReduceLaw::WrappingProduct => state.wrapping_mul(*value),
            });
            let result = device.run_collective(&verified, &[&values], seed).unwrap();
            assert_eq!(result.final_state, expected, "law={law:?} len={len}");
            assert!(result.prefixes.is_empty());
        }
    }
    for (source, mode) in [
        (INCLUSIVE, "inclusive"),
        (EXCLUSIVE, "exclusive"),
        (FINAL, "final"),
    ] {
        let program =
            mir::lower(&check::check(front::parse(source.as_bytes()).unwrap(), true).unwrap())
                .unwrap();
        let report = admission::analyze(source.as_bytes(), &program).unwrap();
        // Chunk boundaries of the Metal scan schedule (64 blocks of 4096), and
        // both the 512-lane shared geometry and the 1024-lane Metal retiling.
        let chunk = 64 * 4096;
        let scan_lengths = lengths.into_iter().chain([
            chunk - 1,
            chunk,
            chunk + 1,
            2 * chunk + 4093,
            3 * chunk + 5,
        ]);
        for (len, retile) in scan_lengths.flat_map(|len| [(len, false), (len, true)]) {
            let mut ir = device_ir::from_region(&report.regions()[0], len as u32).unwrap();
            if retile {
                device_ir::retile_metal(&mut ir);
            }
            let origin = Provenance {
                source_identity: ir.source_identity,
                graph_revision: ir.graph_revision,
            };
            let verified = invar_device::verify(ir, origin, device.capabilities()).unwrap();
            // Odd multipliers keep every earlier state observable in later ones.
            let aa = (0..len)
                .map(|i| (i as u32).wrapping_mul(2_654_435_761) | 1)
                .collect::<Vec<_>>();
            let bb = (0..len)
                .map(|i| (i as u32).wrapping_mul(7).wrapping_add(5))
                .collect::<Vec<_>>();
            let mut state = 17u32;
            let mut inclusive = Vec::new();
            let mut exclusive = Vec::new();
            for (&a, &b) in aa.iter().zip(&bb) {
                exclusive.push(state);
                state = a.wrapping_mul(state).wrapping_add(b);
                inclusive.push(state);
            }
            let result = device.run_collective(&verified, &[&aa, &bb], 17).unwrap();
            assert_eq!(
                result.final_state, state,
                "mode={mode} len={len} retile={retile}"
            );
            let expected = match mode {
                "inclusive" => inclusive,
                "exclusive" => exclusive,
                _ => Vec::new(),
            };
            assert_eq!(
                result.prefixes, expected,
                "mode={mode} len={len} retile={retile}"
            );
        }
    }
}
