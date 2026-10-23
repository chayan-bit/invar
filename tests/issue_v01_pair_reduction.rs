#[path = "support/pair_reduction.rs"]
mod support;

use invar::{
    eval,
    laws::{self, Affine},
    mir_eval,
    pair_reduction::{self, Error},
    plan::{Execution, PlanKind},
    schedule, typed_net,
};
use support::*;

#[test]
fn pair_reduction_cost_facts_include_pair_summaries_and_metal_chunks() {
    let source = source(&[2, 5], &[3, 7], [1, 0]);
    let (_, program) = compile(&source);
    let report = region(&source, &program);
    for (caps, temporary_bytes, launches) in [
        (invar_device::Capabilities::cuda_12_9_sm89(), 1040, 3),
        (
            invar_device::Capabilities::Metal {
                apple_family: 7,
                msl_major: 3,
                msl_minor: 1,
            },
            1084,
            6,
        ),
    ] {
        // 65 blocks cross Metal's 64-block chunk boundary; two u32s per summary.
        let ir = invar::device_ir::from_region(&report.regions()[0], 262_145).unwrap();
        let origin = invar_device::Provenance {
            source_identity: ir.source_identity,
            graph_revision: ir.graph_revision,
        };
        let facts = invar_device::verify(ir, origin, caps)
            .unwrap()
            .cost_facts()
            .unwrap();
        assert_eq!((facts.input_bytes, facts.output_bytes), (2_097_160, 8));
        assert_eq!(
            (facts.temporary_bytes, facts.launches),
            (temporary_bytes, launches)
        );
    }
}

#[test]
fn pair_reduction_source_reference_mir_and_chunked_cpu() {
    for n in [0, 1, 2, 3, 31, 32, 33, 255, 256, 257, 4093, 4096, 4097] {
        let (a, b) = inputs(n);
        for seed in [[1, 0], [u32::MAX, 17]] {
            let source = source(&a, &b, seed);
            let (checked, program) = compile(&source);
            let candidate = pair_reduction::extract(&program).unwrap();
            assert_eq!(
                candidate
                    .run(
                        &a,
                        &b,
                        Affine {
                            a: seed[0],
                            b: seed[1]
                        },
                        n
                    )
                    .unwrap(),
                Affine {
                    a: oracle(&a, &b, seed)[0],
                    b: oracle(&a, &b, seed)[1]
                }
            );
            let report = region(&source, &program);
            let schema: serde_json::Value =
                serde_json::from_str(include_str!("../schema/explain-v5.schema.json")).unwrap();
            let mut document = report.to_json();
            document["schema"] = serde_json::json!("invar-explain-v5");
            assert!(jsonschema::is_valid(&schema, &document));
            assert!(matches!(
                report.regions()[0].execution(),
                Execution::PairReduction(_)
            ));
            let reference = eval::observe(&checked, 10_000_000);
            assert_eq!(reference.result, Ok(eval::Value::I32(0)), "n={n}");
            let serial = mir_eval::observe(&program, 10_000_000);
            assert_eq!(serial.result, reference.result);
            assert_eq!(serial.steps_used, reference.steps_used);
            assert_eq!(serial.cleanup_trace, reference.cleanup_trace);
            for workers in [1, 2, 4] {
                let run =
                    schedule::run_source(source.as_bytes(), &program, 10_000_000, workers).unwrap();
                assert_eq!(
                    run.observation.result, reference.result,
                    "n={n} workers={workers}"
                );
                assert_eq!(run.observation.steps_used, reference.steps_used);
                assert_eq!(run.observation.cleanup_trace, reference.cleanup_trace);
                assert!(run.executed.iter().any(|r| r.kind == PlanKind::Reduction));
            }
        }
    }
}

#[test]
fn pair_reduction_identity_laws_equal_pairs_and_order() {
    let pairs = [0, 1, 2, u32::MAX]
        .into_iter()
        .flat_map(|a| [0, 1, 2, u32::MAX].map(|b| Affine { a, b }))
        .collect::<Vec<_>>();
    for &x in &pairs {
        assert_eq!(Affine::IDENTITY.then(x), x);
        assert_eq!(x.then(Affine::IDENTITY), x);
        let expected = oracle(&[x.a], &[x.b], [x.a, x.b]);
        assert_eq!(
            x.then(x),
            Affine {
                a: expected[0],
                b: expected[1]
            }
        );
        for &y in &pairs {
            for &z in &pairs {
                assert_eq!(x.then(y).then(z), x.then(y.then(z)));
            }
        }
    }
    // A reversed tree is observably wrong, even though repeated equal pairs tie.
    let source = source(&[2, 5], &[3, 7], [1, 0]);
    let (_, program) = compile(&source);
    let pair = pair_reduction::extract(&program).unwrap();
    assert_eq!(
        pair.run(&[2, 5], &[3, 7], Affine::IDENTITY, 2).unwrap(),
        Affine { a: 10, b: 22 }
    );
    assert_ne!(
        Affine { a: 2, b: 3 }.then(Affine { a: 5, b: 7 }),
        Affine { a: 5, b: 7 }.then(Affine { a: 2, b: 3 })
    );
    let commuted = source
        .replace("wrapping_mul(aa[i],sa)", "wrapping_mul(sa,aa[i])")
        .replace(
            "wrapping_add(wrapping_mul(aa[i],sb),bb[i])",
            "wrapping_add(bb[i],wrapping_mul(sb,aa[i]))",
        );
    let (checked, program) = compile(&commuted);
    assert!(pair_reduction::extract(&program).is_ok());
    assert_eq!(
        schedule::run_source(commuted.as_bytes(), &program, 10000, 2)
            .unwrap()
            .observation
            .result,
        eval::observe(&checked, 10000).result
    );
}

#[test]
fn pair_reduction_read_alias_store_order_and_floating_refusal() {
    let aliased = source(&[2, 5], &[2, 5], [7, 11]).replace("pair(aa,bb,", "pair(aa,aa,");
    let reverse_stores = aliased.replace(
        "sa = wrapping_mul(aa[i],sa); sb = wrapping_add(wrapping_mul(aa[i],sb),bb[i]);",
        "sb = wrapping_add(wrapping_mul(aa[i],sb),bb[i]); sa = wrapping_mul(aa[i],sa);",
    );
    for source in [aliased, reverse_stores] {
        let (checked, program) = compile(&source);
        assert!(pair_reduction::extract(&program).is_ok());
        let expected = eval::observe(&checked, 10000);
        assert_eq!(expected.result, Ok(eval::Value::I32(0)));
        let run = schedule::run_source(source.as_bytes(), &program, 10000, 4).unwrap();
        assert_eq!(run.observation.result, expected.result);
        assert_eq!(run.observation.cleanup_trace, expected.cleanup_trace);
    }
    for (a, b) in [
        ([1, 1, 1], [0, 0, 0]),
        ([0, 0, 0], [1, 3, 7]),
        ([u32::MAX; 3], [u32::MAX; 3]),
    ] {
        let source = source(&a, &b, [7, 11]);
        let (checked, program) = compile(&source);
        assert!(pair_reduction::extract(&program).is_ok());
        let expected = eval::observe(&checked, 10000);
        assert_eq!(expected.result, Ok(eval::Value::I32(0)));
        assert_eq!(
            schedule::run_source(source.as_bytes(), &program, 10000, 4)
                .unwrap()
                .observation
                .result,
            expected.result
        );
    }
    let floating = "fn main() -> i32 { let xs: Array<f32> = [1.0]; var a: f32 = 1.0; var b: f32 = 0.0; for i in 0u32..len(xs) { a = xs[i] * a; b = xs[i] * b + xs[i]; } return 0; }";
    let (_, program) = compile(floating);
    assert_eq!(
        pair_reduction::extract(&program).unwrap_err(),
        Error::UnlawfulOperator
    );
}

#[cfg(feature = "native")]
#[test]
fn pair_reduction_native_serial_matches_source() {
    for n in [0, 1, 2, 33, 257] {
        let (a, b) = inputs(n);
        let source = source(&a, &b, [u32::MAX, 17]);
        let (checked, program) = compile(&source);
        let expected = eval::observe(&checked, 1_000_000);
        let (status, result) = invar::native::run_program(&program, 1_000_000).unwrap();
        assert_eq!(status, 0);
        assert_eq!(expected.result, Ok(eval::Value::I32(0)));
        assert_eq!(
            result["result"],
            serde_json::json!({"type":"i32","value":0})
        );
        assert_eq!(result["steps_used"], expected.steps_used);
        assert_eq!(
            result["cleanup_trace"],
            serde_json::json!(
                expected
                    .cleanup_trace
                    .iter()
                    .map(|e| serde_json::json!({"name":e.name,"length":e.length}))
                    .collect::<Vec<_>>()
            )
        );
    }
}

#[test]
fn pair_reduction_maximum_bounds_and_guard_fallback() {
    let (_, program) = compile(&source(&[1], &[7], [1, 0]));
    let pair = pair_reduction::extract(&program).unwrap();
    let a = vec![1; laws::MAX_ELEMENTS];
    let b = vec![7; laws::MAX_ELEMENTS];
    assert_eq!(
        pair.run(&a, &b, Affine::IDENTITY, a.len()).unwrap(),
        Affine { a: 1, b: 7_000_000 }
    );
    assert_eq!(
        pair.run(&[], &[], Affine::IDENTITY, laws::MAX_ELEMENTS + 1),
        Err(Error::ResourceLimit)
    );
    assert_eq!(
        pair.run(&[1], &[], Affine::IDENTITY, 1),
        Err(Error::InvalidPlan)
    );
    let bad = source(&[2, 5], &[3], [1, 0]);
    let (checked, program) = compile(&bad);
    let reference = eval::observe(&checked, 10000);
    let run = schedule::run_source(bad.as_bytes(), &program, 10000, 4).unwrap();
    assert!(reference.result.is_err());
    assert_eq!(run.observation.result, reference.result);
    assert_eq!(run.observation.steps_used, reference.steps_used);
    assert_eq!(run.observation.cleanup_trace, reference.cleanup_trace);
    assert!(run.executed.is_empty());
    let source = source(&[2, 5], &[3, 7], [1, 0]);
    let (checked, program) = compile(&source);
    for fuel in [0, 1, 10, 25, 40, 70, 100] {
        let expected = eval::observe(&checked, fuel);
        let actual = schedule::run_source(source.as_bytes(), &program, fuel, 4)
            .unwrap()
            .observation;
        assert_eq!(actual.result, expected.result, "fuel={fuel}");
        assert_eq!(actual.steps_used, expected.steps_used);
        assert_eq!(actual.cleanup_trace, expected.cleanup_trace);
    }
}

#[test]
fn pair_reduction_unlawful_operators_refused() {
    let valid = source(&[2, 5], &[3, 7], [1, 0]);
    for bad in [
        valid.replace(
            "wrapping_add(wrapping_mul(aa[i],sb),bb[i])",
            "wrapping_mul(wrapping_mul(aa[i],sb),bb[i])",
        ),
        valid.replace("wrapping_mul(aa[i],sa)", "aa[i] * sa"),
        valid.replace(
            "wrapping_add(wrapping_mul(aa[i],sb),bb[i])",
            "wrapping_add(wrapping_mul(aa[i],sb),wrapping_mul(sb,bb[i]))",
        ),
        valid.replace("wrapping_mul(aa[i],sa)", "wrapping_mul(bb[i],sa)"),
    ] {
        let (_, program) = compile(&bad);
        assert!(matches!(
            pair_reduction::extract(&program),
            Err(Error::UnlawfulOperator | Error::Shape)
        ));
        let report = invar::admission::analyze(bad.as_bytes(), &program).unwrap();
        assert!(
            !report
                .regions()
                .iter()
                .any(|r| matches!(r.execution(), Execution::PairReduction(_)))
        );
    }
    let bad = valid.replace("wrapping_mul(aa[i],sa)", "aa[i] * sa");
    let (_, program) = compile(&bad);
    assert_eq!(
        pair_reduction::extract(&program).unwrap_err().code(),
        "NUMERIC_POLICY"
    );
}

#[test]
fn pair_reduction_graph_and_device_mutations_reject() {
    let source = source(&[2, 5], &[3, 7], [1, 0]);
    let (_, program) = compile(&source);
    let mut pair = pair_reduction::extract(&program).unwrap();
    let proof = pair.graph.verify(invar::loops::GRAPH_LIMITS).unwrap();
    pair.graph.nodes[pair.graph.entry].as_mut().unwrap().op = typed_net::Op::PairReduce {
        rule: "non-associative-pair",
    };
    assert!(pair.graph.verify(invar::loops::GRAPH_LIMITS).is_err());
    assert!(pair.graph.extract(&proof).is_err());
    assert_eq!(
        pair.run(&[2, 5], &[3, 7], Affine::IDENTITY, 2),
        Err(Error::InvalidPlan)
    );
    let report = region(&source, &program);
    for caps in [
        invar_device::Capabilities::cuda_12_9_sm89(),
        invar_device::Capabilities::Metal {
            apple_family: 7,
            msl_major: 3,
            msl_minor: 1,
        },
    ] {
        let ir = invar::device_ir::from_region(&report.regions()[0], 2).unwrap();
        let origin = invar_device::Provenance {
            source_identity: ir.source_identity,
            graph_revision: ir.graph_revision,
        };
        let verified = invar_device::verify(ir.clone(), origin, caps).unwrap();
        let emitted = match caps {
            invar_device::Capabilities::Cuda { .. } => invar_device::emit_cuda(&verified).unwrap(),
            _ => invar_device::emit_msl(&verified).unwrap(),
        };
        assert!(emitted.contains("output[0] = total.a"));
        assert!(emitted.contains("output[1] = total.b"));
        assert!(!emitted.contains('@'));
        let mut wrong_identity = ir.clone();
        wrong_identity.collective.as_mut().unwrap().kind =
            invar_device::CollectiveKind::PairReduction {
                identity_a: 0,
                identity_b: 0,
            };
        assert_eq!(
            invar_device::verify(wrong_identity, origin, caps).err(),
            Some(invar_device::Error::Operations)
        );
        let mut short = ir;
        short.output.elements = 1;
        short.collective.as_mut().unwrap().output.elements = 1;
        assert_eq!(
            invar_device::verify(short, origin, caps).err(),
            Some(invar_device::Error::Bounds)
        );
        let maximum = invar::device_ir::from_region(
            &report.regions()[0],
            invar_device::collective::MAX_ELEMENTS,
        )
        .unwrap();
        assert!(invar_device::verify(maximum, origin, caps).is_ok());
        assert!(
            invar::device_ir::from_region(
                &report.regions()[0],
                invar_device::collective::MAX_ELEMENTS + 1
            )
            .is_err()
        );
    }
}
