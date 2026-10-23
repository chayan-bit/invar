use super::*;

#[test]
fn core_op_sets_reject_checked_and_strict_f32_folds() {
    for (source, kind) in [
        (MAP, Family::Map),
        (REDUCE, Family::Reduce),
        (SCAN, Family::Scan),
    ] {
        assert!(
            report(source).admissions.iter().any(|a| a.family == kind),
            "{kind:?}"
        );
    }
    for s in [
        REDUCE.replace("wrapping_add(s,a[i])", "s+a[i]"),
        SCAN.replace("wrapping_mul(s,3u32)", "s*3u32"),
        TENSOR.to_owned(),
    ] {
        let (net, proof) = support::compile(&s);
        let form = normal_form::extract(&net, &proof).unwrap();
        let r = family::analyze(&net, &proof, &form).expect("serial fallback remains checkable");
        assert!(
            !r.admissions
                .iter()
                .any(|a| matches!(a.family, Family::Reduce | Family::Scan))
        );
    }
}

#[test]
fn invalid_form_mutations_fail_before_admission() {
    let (net, proof) = support::compile(MAP);
    let form = normal_form::extract(&net, &proof).unwrap();
    let mut bad = form.clone();
    bad.fingerprint[0] ^= 1;
    assert!(
        family::analyze(&net, &proof, &bad).is_err(),
        "stale_fingerprint"
    );
    let mut bad = form.clone();
    bad.functions[0].loops.clear();
    assert_eq!(
        family::analyze(&net, &proof, &bad),
        Err(family::Error::FormMismatch),
        "omit_loop"
    );
    let mut bad = form.clone();
    bad.functions[0].loops[0].certificates.clear();
    assert!(
        family::analyze(&net, &proof, &bad).is_err(),
        "drop_certificates"
    );
    let mut bad = form.clone();
    if let Node::Parallel(n) = &mut bad.functions[0].loops[0].node {
        n.carried.push(normal_form::Carried {
            var: "forged".into(),
            law: normal_form::LawId::L4Affine,
            rep: Rep::Affine(Ring::Wrapping32),
            scan: true,
        });
    }
    assert!(
        family::analyze(&net, &proof, &bad).is_err(),
        "invent_representation"
    );
    let mut bad = form.clone();
    bad.functions[0].loops[0].node = Node::Serial {
        reason: "hidden".into(),
    };
    bad.functions[0].loops[0].certificates.clear();
    assert_eq!(
        family::analyze(&net, &proof, &bad),
        Err(family::Error::FormMismatch),
        "hide_parallel_loop"
    );
    let (other, other_proof) = support::compile(&MAP.replace("3u32", "4u32"));
    assert!(
        family::analyze(&other, &other_proof, &form).is_err(),
        "different_net"
    );
}

#[test]
fn checked_lowering_annotations_remain_accepted() {
    let (net, proof) = support::compile(MAP);
    let mut form = normal_form::extract(&net, &proof).unwrap();
    let partition = regions::partition(&net, &proof).unwrap();
    let c = &mut form.functions[0].loops[0].certificates[0];
    c.annotations = certificate::rederive(&partition, c.region).unwrap();
    assert!(!c.annotations.is_empty());
    assert!(family::analyze(&net, &proof, &form).is_ok());
}

#[test]
fn finite_state_summary_requires_a_pnf_law() {
    let c = checked(TABLE);
    assert!(invar::summary_source::Contract::from_checked(&c, "run").is_ok());
    assert!(
        report(TABLE)
            .admissions
            .iter()
            .all(|a| a.family != Family::Summary)
    );
    let refused = report(TABLE).refusals;
    assert_eq!(refused.len(), 1);
    assert_eq!(
        refused[0].reason,
        family::RefusalReason::MissingFiniteStateLaw
    );
    assert_eq!(refused[0].candidate.family, Family::Summary);
    assert!(report(&TABLE.replace("4u32", "17u32")).refusals.is_empty());
    for width in 1..=16 {
        for entry in ["run", "renamed"] {
            let source = TABLE
                .replace("4u32", &format!("{width}u32"))
                .replace("run", entry);
            let (diff, _, refused) = compare("table-shape", &source);
            assert!(diff.is_empty(), "{diff:?}");
            assert_eq!(refused.len(), 1);
        }
    }
}

#[test]
fn followup_findings_have_no_unresolved_admission_difference() {
    let mut differences = Vec::new();
    let mut cases = 0;
    for (id, source) in focused().into_iter().filter(|(id, _)| {
        [
            "family/table",
            "family/fused",
            "near-miss/scan-mixed-checked",
        ]
        .iter()
        .any(|prefix| id == prefix || id == &format!("{prefix}-renamed"))
    }) {
        cases += 1;
        differences.extend(compare(&id, &source).0);
    }
    assert_eq!(cases, 6);
    assert!(differences.is_empty(), "{differences:#?}");
}

fn array(a: Vec<u32>) -> eval::Value {
    eval::Value::ArrayU32(Rc::new(RefCell::new(a)))
}

#[test]
fn trap_heavy_cpu_net_oracle_agreement_and_first_trap_mutation() {
    let mut cases = 0;
    let mut traps = 0;
    for (source, kind) in [
        (MAP, Family::Map),
        (REDUCE, Family::Reduce),
        (SCAN, Family::Scan),
        (GATHER, Family::Gather),
        (SCATTER, Family::Scatter),
        (HISTOGRAM, Family::Histogram),
    ] {
        let (net, proof) = support::compile(source);
        let form = normal_form::extract(&net, &proof).unwrap();
        assert!(
            family::analyze(&net, &proof, &form)
                .unwrap()
                .admissions
                .iter()
                .any(|a| a.family == kind),
            "{kind:?}"
        );
        for n in 0..12u32 {
            for short in 0..4u32 {
                let values = (0..n.saturating_sub(short))
                    .map(|i| if i % 2 == 0 { u32::MAX } else { i })
                    .collect::<Vec<_>>();
                let args = || match kind {
                    Family::Map => vec![array(values.clone())],
                    Family::Reduce | Family::Scan => vec![
                        array(values.clone()),
                        eval::Value::U32(n),
                        eval::Value::U32(u32::MAX),
                    ],
                    _ => {
                        let indices = (0..n)
                            .map(|i| if short == 0 { i } else { i + short })
                            .collect();
                        let mut a = vec![array(values.clone()), array(indices)];
                        if kind != Family::Gather {
                            a.push(eval::Value::U32(n));
                        }
                        a
                    }
                };
                let oracle = general_net::oracle::observe_call(
                    &net,
                    &proof,
                    "run",
                    &args(),
                    general_net::oracle::Limits::default(),
                )
                .unwrap();
                assert_eq!(oracle.failure, None);
                let actual =
                    exec::run(&net, &proof, &form, "run", &args(), exec::Mutation::None).unwrap();
                assert_eq!(
                    actual.result, oracle.semantic.result,
                    "{kind:?} n={n} short={short}"
                );
                assert!(actual.receipt.violations.is_empty());
                traps += usize::from(matches!(
                    oracle.semantic.result,
                    Err(eval::EvalError::Trap { .. })
                ));
                cases += 1;
            }
        }
    }
    assert_eq!(cases, 288);
    assert!(traps > 100);
    // Later lanes have different index spans. Returning the last event must
    // disagree with the CPU net's first source-order trap.
    let source = "fn run(a: read Array<u32>, b: read Array<u32>, n: u32) -> u32 { var s=0u32; for i in 0u32..n { s=wrapping_add(wrapping_mul(a[i],s),b[i]); } return s; }";
    let (net, proof) = support::compile(source);
    let form = normal_form::extract(&net, &proof).unwrap();
    let args = || vec![array(vec![3]), array(vec![]), eval::Value::U32(3)];
    let oracle = general_net::oracle::observe_call(
        &net,
        &proof,
        "run",
        &args(),
        general_net::oracle::Limits::default(),
    )
    .unwrap();
    let mutated = exec::run(
        &net,
        &proof,
        &form,
        "run",
        &args(),
        exec::Mutation::LastTrap,
    )
    .unwrap();
    assert_ne!(
        mutated.result, oracle.semantic.result,
        "last_trap mutation survived"
    );
    println!("oracle_cases={cases} traps={traps}; first_trap mutation killed");
}

#[test]
fn strict_f32_family_bodies_match_cpu_net_bits_and_trap_spans() {
    use numeric_contract::F32Bits;
    fn floats(n: usize) -> eval::Value {
        let bits = [
            0, 0x80000000, 1, 0x007fffff, 0x3f800001, 0xbf800000, 0x7fc12345, 0x7f800000,
        ];
        eval::Value::ArrayF32(Rc::new(RefCell::new(
            (0..n)
                .map(|i| F32Bits::from_bits(bits[i % bits.len()]))
                .collect(),
        )))
    }
    let mut count = 0;
    let mut traps = 0;
    for (source, kind, tensor) in [
        (TRANSPOSE.to_owned(), Family::Transpose, false),
        (TENSOR.to_owned(), Family::Tensor, true),
        (DIRECT.to_owned(), Family::Tensor, true),
        (stencil(false), Family::Stencil, false),
        (stencil(true), Family::Stencil, false),
    ] {
        let (net, proof) = support::compile(&source);
        let form = normal_form::extract(&net, &proof).unwrap();
        assert!(
            family::analyze(&net, &proof, &form)
                .unwrap()
                .admissions
                .iter()
                .any(|a| a.family == kind),
            "{kind:?}"
        );
        for rows in 0..5usize {
            for cols in 0..5usize {
                for short in 0..2usize {
                    let args = || {
                        if tensor {
                            vec![
                                floats((rows * 3).saturating_sub(short)),
                                floats((cols * 3).saturating_sub(short)),
                                eval::Value::U32(rows as u32),
                                eval::Value::U32(3),
                                eval::Value::U32(cols as u32),
                            ]
                        } else {
                            vec![
                                floats((rows * cols).saturating_sub(short)),
                                eval::Value::U32(rows as u32),
                                eval::Value::U32(cols as u32),
                            ]
                        }
                    };
                    let oracle = general_net::oracle::observe_call(
                        &net,
                        &proof,
                        "run",
                        &args(),
                        general_net::oracle::Limits::default(),
                    )
                    .unwrap();
                    assert_eq!(oracle.failure, None);
                    let actual =
                        exec::run(&net, &proof, &form, "run", &args(), exec::Mutation::None)
                            .unwrap();
                    assert_eq!(
                        actual.result, oracle.semantic.result,
                        "{kind:?} rows={rows} cols={cols} short={short}"
                    );
                    assert!(actual.receipt.violations.is_empty());
                    traps += usize::from(matches!(
                        oracle.semantic.result,
                        Err(eval::EvalError::Trap { .. })
                    ));
                    count += 1;
                }
            }
        }
    }
    assert_eq!(count, 250);
    assert!(traps > 50);
    println!("f32_oracle_cases={count} traps={traps}");
}

#[test]
fn serial_round_wrapper_consumes_checked_frontier_children() {
    let frozen = include_str!("../../bench/general-net-v01/current/frontier_graph_a.inv");
    let mut count = 0;
    for n in 1..=5u32 {
        let source = frozen.replace("5u32", &format!("{n}u32"));
        let (net, proof) = support::compile(&source);
        let form = normal_form::extract(&net, &proof).unwrap();
        let report = family::analyze(&net, &proof, &form).unwrap();
        assert!(
            report
                .admissions
                .iter()
                .any(|a| a.family == Family::Frontier)
        );
        let oracle =
            general_net::oracle::observe(&net, &proof, general_net::oracle::Limits::default())
                .unwrap();
        assert_eq!(oracle.failure, None);
        let actual = exec::run(&net, &proof, &form, "main", &[], exec::Mutation::None).unwrap();
        assert_eq!(actual.result, oracle.semantic.result);
        assert!(actual.receipt.violations.is_empty());
        count += 1;
        let mut bad = form.clone();
        for function in &mut bad.functions {
            for l in &mut function.loops {
                l.certificates
                    .retain(|c| c.law != normal_form::LawId::L14MonotoneFrontier);
            }
        }
        assert!(
            family::analyze(&net, &proof, &bad).is_err(),
            "drop_frontier_child_law"
        );
    }
    println!("frontier_oracle_cases={count}; child-law mutations killed={count}");
}

#[test]
fn dropped_runtime_guard_is_rejected_even_without_a_family_match() {
    let source = "fn run(a: read Array<u32>, n: u32, stride: u32) -> Array<u32> { var out=filled_u32(n,0u32); for i in 0u32..n { out[i*stride]=wrapping_add(a[i],1u32); } return out; }";
    let (net, proof) = support::compile(source);
    let mut form = normal_form::extract(&net, &proof).unwrap();
    assert!(family::analyze(&net, &proof, &form).is_ok());
    let c = form.functions[0].loops[0]
        .certificates
        .iter_mut()
        .find(|c| !c.guards.is_empty())
        .expect("stride guard");
    c.guards.clear();
    assert!(
        matches!(
            family::analyze(&net, &proof, &form),
            Err(family::Error::Certificate(certificate::Rejection::Guards))
        ),
        "drop_runtime_guard"
    );
}
