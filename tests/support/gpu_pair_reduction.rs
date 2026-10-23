//! Pair-reduction device tests shared by the CUDA and Metal binaries.
//! The including module supplies `open`, `run_source`, `prepare_ir` and `run_collective`.
use super::support::{compile, inputs, oracle, region, source};
use super::{open, prepare_ir, run_collective, run_source};
use invar::{eval, plan::PlanKind};

#[test]
#[ignore = "requires a GPU (CUDA with NVRTC, or Apple Metal)"]
fn pair_reduction_gpu_source_differential() {
    let mut device = open();
    for n in [0, 1, 2, 3, 31, 32, 33, 255, 256, 257, 4093, 4096, 4097] {
        let (a, b) = inputs(n);
        for seed in [[1, 0], [u32::MAX, 17]] {
            let source = source(&a, &b, seed);
            let (checked, program) = compile(&source);
            let reference = eval::observe(&checked, 10_000_000);
            let run = run_source(&source, &program, 10_000_000, &mut device);
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
    let actual = run_source(&source, &program, 10000, &mut device);
    assert_eq!(actual.observation.result, expected.result);
    assert_eq!(actual.observation.cleanup_trace, expected.cleanup_trace);
}

#[test]
#[ignore = "requires a GPU (CUDA with NVRTC, or Apple Metal)"]
fn pair_reduction_gpu_hierarchy_maximum_and_equal_pairs() {
    let mut device = open();
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
        prepare_ir(&mut ir);
        let origin = invar_device::Provenance {
            source_identity: ir.source_identity,
            graph_revision: ir.graph_revision,
        };
        let verified = invar_device::verify(ir, origin, device.capabilities()).unwrap();
        let (pair, no_prefixes, source_hash) = run_collective(&mut device, &verified, &[&a, &b]);
        assert_eq!(pair, Some(expected), "n={n}");
        assert!(no_prefixes);
        eprintln!("pair n={n} complete=true output={expected:?} source_hash={source_hash:016x}");
    }
}
