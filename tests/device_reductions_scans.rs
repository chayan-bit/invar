//! Issue label M3-04.
use invar::{
    admission, check::check, device_ir, eval, front::parse, mir, mir_eval, plan::PlanKind,
};
use invar::{laws, plan::Execution, typed_net::Prefix};
use invar_device::{
    Barrier, Capabilities, CollectiveKind, Error as VerifyError, Layout, Numeric, Provenance,
};

#[path = "support/m3_04_programs.rs"]
mod m3_04_programs;
use m3_04_programs::{EXCLUSIVE, FINAL, INCLUSIVE, PRODUCT, SUM};

#[test]
fn device_reductions_scans_graph_plans_lower_to_shared_ir() {
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
fn device_reductions_scans_verifier_rejects_invalid_collectives() {
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
fn device_reductions_scans_cpu_serial_shape_corpus() {
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
