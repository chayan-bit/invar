#![cfg(all(feature = "metal", target_os = "macos"))]
#[path = "support/pair_reduction.rs"]
mod support;
use invar::{eval, plan::PlanKind};
use invar_metal::MetalDevice;
use support::*;

#[test]
#[ignore = "requires Apple Metal hardware"]
fn pair_reduction_metal_gpu_source_differential() {
    let mut device = MetalDevice::open_runtime_source().expect("Metal hardware required");
    for n in [0, 1, 2, 3, 31, 32, 33, 255, 256, 257, 4093, 4096, 4097] {
        let (a, b) = inputs(n);
        for seed in [[1, 0], [u32::MAX, 17]] {
            let source = source(&a, &b, seed);
            let (checked, program) = compile(&source);
            let reference = eval::observe(&checked, 10_000_000);
            let run = invar::schedule::metal::run_source_with_device(
                source.as_bytes(),
                &program,
                10_000_000,
                &mut device,
            )
            .unwrap();
            assert_eq!(reference.result, Ok(eval::Value::I32(0)));
            assert_eq!(
                run.observation.result, reference.result,
                "n={n} seed={seed:?}"
            );
            assert_eq!(run.observation.steps_used, reference.steps_used);
            assert_eq!(run.observation.cleanup_trace, reference.cleanup_trace);
            assert!(run.executed.iter().any(|r| r.kind == PlanKind::Reduction));
        }
    }
    let source = source(&[2, 5], &[2, 5], [7, 11]).replace("pair(aa,bb,", "pair(aa,aa,");
    let (checked, program) = compile(&source);
    let expected = eval::observe(&checked, 10000);
    let actual = invar::schedule::metal::run_source_with_device(
        source.as_bytes(),
        &program,
        10000,
        &mut device,
    )
    .unwrap();
    assert_eq!(actual.observation.result, expected.result);
    assert_eq!(actual.observation.cleanup_trace, expected.cleanup_trace);
}

#[test]
#[ignore = "requires Apple Metal hardware"]
fn pair_reduction_metal_gpu_hierarchy_maximum_and_equal_pairs() {
    let mut device = MetalDevice::open_runtime_source().expect("Metal hardware required");
    let source = source(&[2, 5], &[3, 7], [1, 0]);
    let (_, program) = compile(&source);
    let report = region(&source, &program);
    eprintln!("device capabilities: {:?}", device.capabilities());
    for n in [
        1,
        33,
        4097,
        65537,
        262145,
        1_000_000,
        invar_device::collective::MAX_ELEMENTS as usize,
    ] {
        let (a, b) = if n == 33 {
            (vec![u32::MAX; n], vec![u32::MAX; n])
        } else {
            inputs(n)
        };
        let expected = oracle(&a, &b, [1, 0]);
        let mut ir = invar::device_ir::from_region(&report.regions()[0], n as u32).unwrap();
        invar::device_ir::plan_metal(&mut ir).unwrap();
        let origin = invar_device::Provenance {
            source_identity: ir.source_identity,
            graph_revision: ir.graph_revision,
        };
        let verified = invar_device::verify(ir, origin, device.capabilities()).unwrap();
        let result = device.run_collective(&verified, &[&a, &b], 0).unwrap();
        assert_eq!(result.pair, Some(expected), "n={n}");
        assert!(result.prefixes.is_empty());
        eprintln!(
            "pair n={n} complete=true output={expected:?} source_hash={:016x}",
            result.source_hash
        );
    }
}
