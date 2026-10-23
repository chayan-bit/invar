//! M4-04 device IR on CPU: the segmented-map recognizer, source-bound lowering
//! to one verified IR for CUDA and Metal, host admission refusals, and the IR's
//! independent reference model against the source expectation (CPU net oracle
//! and reference evaluator) for every adversarial case. Device execution is in
//! the ignored `issue_m4_04_gpu` (CUDA) and `issue_m4_04_metal_gpu` suites.
#![forbid(unsafe_code)]

#[path = "../support/m4_04_cases.rs"]
mod cases;

use cases::{Kind, function, word};
use invar::{segmented_device::MapContract, segmented_source};
use invar_device::{
    Capabilities,
    segmented::{self as seg, MapOp},
};

const CUDA_SM120: Capabilities = Capabilities::Cuda {
    major: 12,
    minor: 9,
    sm: 120,
};
const METAL: Capabilities = Capabilities::Metal {
    apple_family: 9,
    msl_major: 3,
    msl_minor: 1,
};

#[test]
fn issue_m4_04_segmented_flattening_compaction_device_ir_map_recognizer() {
    let source = function(Kind::Map);
    let contract = MapContract::from_source(source.as_bytes(), "run").unwrap();
    assert_eq!(
        contract.ops(),
        [
            MapOp::WrappingAdd(7),
            MapOp::WrappingMul(2_654_435_761),
            MapOp::AddSegment
        ]
    );
    let renamed = source
        .replace("xs", "samples")
        .replace("offsets", "bounds")
        .replace("count", "groups")
        .replace("out", "result");
    assert!(MapContract::from_source(renamed.as_bytes(), "run").is_ok());
    contract
        .verify_extraction(contract.graph(), contract.plan())
        .unwrap();
    // Alter a valid callback graph, leaving the original source binding intact.
    let mut changed = contract.graph().clone();
    let node = changed
        .nodes
        .iter_mut()
        .flatten()
        .find(|node| matches!(node.op, invar::typed_net::Op::Map(_)))
        .unwrap();
    node.op = invar::typed_net::Op::Map(invar::typed_net::MapOp::WrappingAdd(8));
    changed.seal_source();
    let proof = changed.verify(invar::loops::GRAPH_LIMITS).unwrap();
    let plan = changed.extract_dag(&proof, &[changed.entry]).unwrap();
    assert!(contract.verify_extraction(&changed, &plan).is_err());
    assert!(contract.verify_extraction(contract.graph(), &plan).is_err());
    let invar::typed_net::Plan::LiveDag { ops, .. } = contract.plan() else {
        panic!("callback DAG")
    };
    assert!(ops.len() <= 2 * contract.ops().len() + 1);
    assert_eq!(
        contract.plan().run_dag(&[&[1, 2], &[0, 1]], 0..2).unwrap()[0],
        vec![
            8u32.wrapping_mul(2_654_435_761),
            9u32.wrapping_mul(2_654_435_761).wrapping_add(1)
        ]
    );
    for (from, to) in [
        // Store to the segment slot, not the element.
        ("out[i] =", "out[s] ="),
        // A non-constant, non-segment operand.
        (", 7u32)", ", i)"),
        // Reading a different element.
        ("xs[i]", "xs[s]"),
        // Wrong segment bound.
        ("offsets[s + 1u32]", "offsets[s]"),
        // Output not sized by the input.
        ("filled_u32(len(xs), 0u32)", "filled_u32(count, 0u32)"),
        // Checked (trapping) arithmetic is not a wrapping map.
        ("wrapping_add(xs[i], 7u32)", "xs[i] + 7u32"),
    ] {
        let changed = source.replacen(from, to, 1);
        assert_ne!(changed, source);
        assert!(
            MapContract::from_source(changed.as_bytes(), "run").is_err(),
            "{from} -> {to}"
        );
    }
}

#[test]
fn issue_m4_04_segmented_flattening_compaction_device_ir_reference_matches_source() {
    let mut checked = 0;
    for case in cases::cases(false) {
        let expected = cases::expected(&case);
        assert_eq!(cases::sequential_expected(&case), expected, "{}", case.name);
        for caps in [Capabilities::cuda_12_9_sm89(), CUDA_SM120, METAL] {
            let plan = cases::plan(&case, caps);
            assert_eq!(
                seg::reference(&plan, &case.input, &case.offsets).unwrap(),
                expected,
                "{}",
                case.name
            );
            let source = match caps {
                Capabilities::Cuda { .. } => seg::emit_cuda(&plan).unwrap(),
                Capabilities::Metal { .. } => seg::emit_msl(&plan).unwrap(),
            };
            for kernel in plan.kernels() {
                assert!(source.contains(kernel), "{}: {kernel}", case.name);
            }
        }
        // The CPU candidates of the existing source contracts agree too.
        if matches!(case.kind, Kind::Sum | Kind::Product) {
            let function = function(case.kind);
            let contract =
                segmented_source::Contract::from_source(function.as_bytes(), "run").unwrap();
            let budget = segmented_source::Budget {
                max_elements: 1 << 20,
                max_segments: 1 << 20,
                max_output_bytes: 1 << 24,
                max_work: 1 << 24,
            };
            let prepared = contract
                .prepare(
                    &case.input,
                    &case.offsets,
                    (case.offsets.len() - 1) as u32,
                    budget,
                )
                .unwrap();
            assert_eq!(prepared.execute().unwrap(), expected, "{}", case.name);
        }
        checked += 1;
    }
    println!("m4_04 device IR reference cases agreeing with the source oracle: {checked}");
    assert!(checked >= 50);
}

#[test]
fn issue_m4_04_segmented_flattening_compaction_device_ir_admission_refusals() {
    let input: Vec<u32> = (0..10).map(word).collect();
    for kind in [Kind::Map, Kind::Sum] {
        let case = cases::Case {
            name: "admission".into(),
            kind,
            input: input.clone(),
            offsets: vec![0, 4, 4, 10],
            cutoff: 0,
            stride: None,
        };
        let plan = cases::plan(&case, CUDA_SM120);
        assert!(seg::admit_inputs(&plan, &input, &case.offsets).is_ok());
        // Decreasing, incomplete at either end, beyond the input, wrong
        // segment count, and a wrong input length: all fall back to the CPU.
        for offsets in [
            &[0, 5, 4, 10][..],
            &[1, 4, 4, 10],
            &[0, 4, 4, 9],
            &[0, 4, 4, 11],
            &[0, 4, 10],
            &[0, 4, 4, 10, 10],
        ] {
            assert_eq!(
                seg::admit_inputs(&plan, &input, offsets).unwrap_err(),
                seg::Error::Offsets,
                "{offsets:?}"
            );
        }
        assert_eq!(
            seg::admit_inputs(&plan, &input[..9], &case.offsets).unwrap_err(),
            seg::Error::Offsets
        );
    }
    let compact = cases::Case {
        name: "admission".into(),
        kind: Kind::Compact("<"),
        input: input.clone(),
        offsets: Vec::new(),
        cutoff: 9,
        stride: None,
    };
    let plan = cases::plan(&compact, METAL);
    assert!(seg::admit_inputs(&plan, &input, &[]).is_ok());
    assert!(seg::admit_inputs(&plan, &input, &[0]).is_err());
    assert!(seg::admit_inputs(&plan, &input[1..], &[]).is_err());
}
