#![forbid(unsafe_code)]
#[path = "support/normal_form.rs"]
mod support;

use invar::{
    eval,
    general_net::{
        normal_form::{self, LawId, Node, certificate, exec},
        oracle,
    },
};
use std::{cell::RefCell, rc::Rc};

const MIXED: &str = "fn run(a: read Array<u32>, n: u32, seed: u32) -> u32 { var s=seed; for i in 0u32..n { s=wrapping_add(s*3u32,a[i]); } return s; }";
const FUSED: &str = "fn run(a: read Array<u32>) -> Array<u32> { var t=filled_u32(len(a),0u32); for i in 0u32..len(a) { t[i]=wrapping_add(a[i],1u32); } let b=move t; var out=filled_u32(len(b),0u32); for j in 0u32..len(b) { out[j]=wrapping_mul(b[j],3u32); } return out; }";

#[test]
fn mixed_checked_wrapping_updates_remain_serial_and_checkable() {
    let (mut cases, mut traps) = (0, 0);
    for entry in ["run", "renamed"] {
        for update in [
            "wrapping_add(s*3u32,a[i])",
            "wrapping_mul(s+a[i],3u32)",
            "wrapping_mul(s,3u32)+a[i]",
            "s*wrapping_add(a[i],1u32)",
        ] {
            let source = MIXED
                .replace("run", entry)
                .replace("wrapping_add(s*3u32,a[i])", update);
            let (net, proof) = support::compile(&source);
            let form = normal_form::extract(&net, &proof).unwrap();
            assert!(
                certificate::check_all(&net, &proof, &form).is_ok(),
                "{update}"
            );
            assert!(
                matches!(form.functions[0].loops[0].node, Node::Serial { .. }),
                "{update}"
            );
            for n in 0..9u32 {
                for short in 0..3 {
                    for seed in [0, 1, u32::MAX] {
                        let values = vec![u32::MAX; n.saturating_sub(short) as usize];
                        let args = || {
                            vec![
                                array(values.clone()),
                                eval::Value::U32(n),
                                eval::Value::U32(seed),
                            ]
                        };
                        let expected = oracle::observe_call(
                            &net,
                            &proof,
                            entry,
                            &args(),
                            oracle::Limits::default(),
                        )
                        .unwrap();
                        let actual =
                            exec::run(&net, &proof, &form, entry, &args(), exec::Mutation::None)
                                .unwrap();
                        assert_eq!(expected.failure, None);
                        assert_eq!(
                            actual.result, expected.semantic.result,
                            "{update} n={n} short={short} seed={seed}"
                        );
                        assert!(actual.receipt.violations.is_empty());
                        cases += 1;
                        traps += usize::from(expected.semantic.result.is_err());
                    }
                }
            }
        }
    }
    println!("mixed arithmetic: {cases} oracle comparisons, {traps} traps");
}

#[test]
fn moved_fresh_owner_lengths_license_only_equal_range_fusion() {
    let mut cases = 0;
    for entry in ["run", "renamed"] {
        let source = FUSED.replace("run", entry);
        let (net, proof) = support::compile(&source);
        let form = normal_form::extract(&net, &proof).unwrap();
        certificate::check_all(&net, &proof, &form).unwrap();
        let fusion = form.functions[0].loops[0]
            .certificates
            .iter()
            .find(|c| c.law == LawId::L19MapFusion)
            .expect("moved owner equal-length L19");
        // Same structure and move, but a different allocation length. A forged
        // L19 must be rejected independently of the extractor's decision.
        let (other, other_proof) =
            support::compile(&source.replace("filled_u32(len(a)", "filled_u32(7u32"));
        let other_form = normal_form::extract(&other, &other_proof).unwrap();
        certificate::check_all(&other, &other_proof, &other_form).unwrap();
        assert!(
            other_form.functions[0].loops[0]
                .certificates
                .iter()
                .all(|c| c.law != LawId::L19MapFusion)
        );
        let mut forged = fusion.clone();
        forged.fingerprint = other_form.fingerprint;
        assert!(
            certificate::check(&other, &other_proof, &forged).is_err(),
            "unequal moved-owner lengths"
        );
        for n in 0..9 {
            for value in [0, 1, u32::MAX] {
                let args = || vec![array(vec![value; n])];
                let expected =
                    oracle::observe_call(&net, &proof, entry, &args(), oracle::Limits::default())
                        .unwrap();
                let actual =
                    exec::run(&net, &proof, &form, entry, &args(), exec::Mutation::None).unwrap();
                assert_eq!(expected.failure, None);
                assert_eq!(actual.result, expected.semantic.result);
                assert!(actual.receipt.violations.is_empty());
                cases += 1;
            }
        }
    }
    println!("moved-owner fusion: {cases} oracle comparisons, unequal-length mutation rejected");
}

fn array(values: Vec<u32>) -> eval::Value {
    eval::Value::ArrayU32(Rc::new(RefCell::new(values)))
}
