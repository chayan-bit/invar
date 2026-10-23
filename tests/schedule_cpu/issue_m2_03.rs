use invar::{check::check, eval, front::parse, laws, mir, mir_eval, schedule};

#[test]
fn issue_m2_03_lawful_reduction_scan_source_wrapping_sum() {
    let source = "fn main() -> i32 { let xs: Array<u32> = [4294967295u32, 1u32, 7u32]; var acc: u32 = 0u32; for i in 0u32 .. len(xs) { acc = wrapping_add(acc, xs[i]); } if acc == 7u32 { return 0; } return 1; }";
    let checked = check(parse(source.as_bytes()).unwrap(), true).unwrap();
    let program = mir::lower(&checked).unwrap();
    assert_eq!(
        eval::observe(&checked, 10_000).result,
        Ok(eval::Value::I32(0))
    );
    assert_eq!(
        mir_eval::observe(&program, 10_000).result,
        Ok(eval::Value::I32(0))
    );
    let candidate = laws::extract_reduction(&program).unwrap().unwrap();
    assert_eq!(candidate.law, laws::Law::WrappingSum);
    assert_eq!(candidate.run(&[u32::MAX, 1, 7], 0).unwrap(), 7);
    assert_eq!(candidate.input, "xs");
    assert_eq!(candidate.state, "acc");
}

fn assert_cpu_agrees_with_reference(source: &str) {
    let checked = check(parse(source.as_bytes()).unwrap(), true).unwrap();
    let program = mir::lower(&checked).unwrap();
    let reference = eval::observe(&checked, 1_000_000);
    for workers in 1..=schedule::MAX_WORKERS {
        let run = schedule::run_source(source.as_bytes(), &program, 1_000_000, workers).unwrap();
        assert_eq!(
            run.observation.result, reference.result,
            "{workers} {source}"
        );
        assert_eq!(
            run.observation.steps_used, reference.steps_used,
            "{workers}"
        );
        assert_eq!(run.observation.cleanup_trace, reference.cleanup_trace);
    }
}

#[test]
fn issue_m2_03_lawful_reduction_scan_reads_seed_live_after_preloop_store() {
    let source = "fn main() -> i32 { let xs: Array<u32> = [2u32, 3u32]; var acc: u32 = 0u32; acc = 9u32; for i in 0u32 .. len(xs) { acc = wrapping_add(acc, xs[i]); } if acc == 14u32 { return 0; } return 1; }";
    let checked = check(parse(source.as_bytes()).unwrap(), true).unwrap();
    let program = mir::lower(&checked).unwrap();
    assert_eq!(
        eval::observe(&checked, 10_000).result,
        Ok(eval::Value::I32(0))
    );
    let candidate = laws::extract_reduction(&program).unwrap().unwrap();
    assert_eq!(candidate.run(&[2, 3], 9).unwrap(), 14);
    assert_cpu_agrees_with_reference(source);
}

#[test]
fn issue_m2_03_lawful_reduction_scan_exhaustive_bounded_laws() {
    for modulus in [4u32, 8] {
        for law in [laws::Law::WrappingSum, laws::Law::WrappingProduct] {
            for left in 0..modulus {
                assert_eq!(law.combine(law.identity(), left) % modulus, left);
                assert_eq!(law.combine(left, law.identity()) % modulus, left);
                for middle in 0..modulus {
                    for right in 0..modulus {
                        let x = law.combine(law.combine(left, middle), right) % modulus;
                        let y = law.combine(left, law.combine(middle, right)) % modulus;
                        assert_eq!(x, y);
                    }
                }
            }
        }
    }
    for modulus in [4u32, 8] {
        for a in 0..modulus {
            for b in 0..modulus {
                let x = laws::Affine { a, b };
                let normalized = |f: laws::Affine| laws::Affine {
                    a: f.a % modulus,
                    b: f.b % modulus,
                };
                assert_eq!(normalized(laws::Affine::IDENTITY.then(x)), x);
                assert_eq!(normalized(x.then(laws::Affine::IDENTITY)), x);
                for c in 0..modulus {
                    for d in 0..modulus {
                        let y = laws::Affine { a: c, b: d };
                        for seed in 0..modulus {
                            let direct = (c * ((a * seed + b) % modulus) + d) % modulus;
                            assert_eq!(x.then(y).apply(seed) % modulus, direct);
                        }
                        for e in 0..modulus {
                            for f in 0..modulus {
                                let z = laws::Affine { a: e, b: f };
                                assert_eq!(
                                    normalized(x.then(y).then(z)),
                                    normalized(x.then(y.then(z)))
                                );
                            }
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn issue_m2_03_lawful_reduction_scan_ordered_tree_shapes_and_cost() {
    let left = laws::Affine { a: 2, b: 3 };
    let right = laws::Affine { a: 5, b: 7 };
    assert_ne!(left.then(right), right.then(left));
    for len in [0usize, 1, 2, 3, 5, 7, 9, 17, 257] {
        let values = (0..len)
            .map(|i| laws::Affine {
                a: (i as u32).wrapping_mul(3).wrapping_add(1),
                b: i as u32 + 4,
            })
            .collect::<Vec<_>>();
        let scan = laws::scan_affine(&values).unwrap();
        let mut state = 11u32;
        let direct = values
            .iter()
            .map(|f| {
                state = f.a.wrapping_mul(state).wrapping_add(f.b);
                state
            })
            .collect::<Vec<_>>();
        assert_eq!(
            scan.prefixes
                .iter()
                .map(|f| f.apply(11))
                .collect::<Vec<_>>(),
            direct,
            "len {len}"
        );
        let size = len.max(1).next_power_of_two();
        assert!(scan.combines <= 3 * size, "len {len}");
        assert!(
            scan.depth <= 2 * size.trailing_zeros() as usize + 1,
            "len {len}"
        );
        if len == 0 {
            assert_eq!(scan.prefixes, Vec::<laws::Affine>::new());
        }
    }
}

#[test]
fn issue_m2_03_lawful_reduction_scan_seeded_differential() {
    let mut state = 0x9e3779b97f4a7c15u64;
    let mut next = || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        (state >> 16) as u32
    };
    for _ in 0..5_011 {
        let len = (next() % 97) as usize;
        let xs = (0..len).map(|_| next()).collect::<Vec<_>>();
        for law in [laws::Law::WrappingSum, laws::Law::WrappingProduct] {
            let scan = laws::scan_u32(law, &xs).unwrap();
            let mut acc = law.identity();
            let direct = xs
                .iter()
                .map(|&x| {
                    acc = law.combine(acc, x);
                    acc
                })
                .collect::<Vec<_>>();
            assert_eq!(scan.prefixes, direct);
            assert!(scan.combines <= 3 * len.max(1).next_power_of_two());
        }
        let transforms = (0..len)
            .map(|_| laws::Affine {
                a: next(),
                b: next(),
            })
            .collect::<Vec<_>>();
        let seed = next();
        let scan = laws::scan_affine(&transforms).unwrap();
        let mut acc = seed;
        let direct = transforms
            .iter()
            .map(|f| {
                acc = f.a.wrapping_mul(acc).wrapping_add(f.b);
                acc
            })
            .collect::<Vec<_>>();
        assert_eq!(
            scan.prefixes
                .iter()
                .map(|f| f.apply(seed))
                .collect::<Vec<_>>(),
            direct
        );
    }
}

#[test]
fn issue_m2_03_lawful_reduction_scan_public_source_shapes_and_negatives() {
    for (law, operation, identity) in [
        (laws::Law::WrappingSum, "wrapping_add", 0u32),
        (laws::Law::WrappingProduct, "wrapping_mul", 1u32),
    ] {
        for values in [
            vec![],
            vec![1],
            vec![0, 1, 2],
            vec![3, 4, 5, 6, 7],
            (0..17).collect(),
        ] {
            let expected = values
                .iter()
                .copied()
                .fold(identity, |a, b| law.combine(a, b));
            let literal = values
                .iter()
                .map(|x| format!("{x}u32"))
                .collect::<Vec<_>>()
                .join(",");
            let source = format!(
                "fn main() -> i32 {{ let source: Array<u32> = [{literal}]; var acc: u32 = {identity}u32; for j in 0u32 .. len(source) {{ acc = {operation}(acc, source[j]); }} if acc == {expected}u32 {{ return 0; }} return 1; }}"
            );
            let checked = check(parse(source.as_bytes()).unwrap(), true).unwrap();
            let program = mir::lower(&checked).unwrap();
            assert_eq!(
                eval::observe(&checked, 100_000).result,
                Ok(eval::Value::I32(0))
            );
            assert_eq!(
                mir_eval::observe(&program, 100_000).result,
                Ok(eval::Value::I32(0))
            );
            let candidate = laws::extract_reduction(&program).unwrap().unwrap();
            assert_eq!(candidate.law, law);
            assert_eq!(candidate.run(&values, identity).unwrap(), expected);
            assert_eq!(candidate.rule_revision, laws::RULE_REVISION);
            assert_eq!(candidate.fallback_reason, "mir-serial");
            assert!(
                source[candidate.source_span.start..candidate.source_span.end].starts_with("for ")
            );
        }
    }
    let checked = "fn main() -> i32 { let xs: Array<i32> = [2147483647, 1]; var acc: i32 = 0; for i in 0u32 .. len(xs) { acc = acc + xs[i]; } return acc; }";
    let program = mir::lower(&check(parse(checked.as_bytes()).unwrap(), true).unwrap()).unwrap();
    assert!(laws::extract_reduction(&program).unwrap().is_none());
}

#[test]
fn issue_m2_03_lawful_reduction_scan_rejects_unproved_source_laws() {
    let base = "fn main() -> i32 { let xs: Array<u32> = [2u32, 3u32]; var acc: u32 = 0u32; for i in 0u32 .. len(xs) { acc = wrapping_add(acc, xs[i]); } return 0; }";
    let variants = [
        format!(
            "fn step(x: u32) -> u32 {{ return x; }} {}",
            base.replace("wrapping_add(acc, xs[i])", "step(xs[i])")
        ),
        base.replace("wrapping_add(acc, xs[i])", "acc + xs[i]"),
        base.replace("wrapping_add(acc, xs[i])", "wrapping_add(acc, acc)"),
        base.replace("wrapping_add(acc, xs[i])", "wrapping_add(xs[i], xs[i])"),
        base.replace("wrapping_add(acc, xs[i])", "wrapping_add(acc, xs[0u32])"),
        base.replace(
            "acc = wrapping_add(acc, xs[i]);",
            "acc = wrapping_add(acc, xs[i]); acc = wrapping_add(acc, 1u32);",
        ),
    ];
    for source in variants {
        let checked = check(parse(source.as_bytes()).unwrap(), true).unwrap();
        let program = mir::lower(&checked).unwrap();
        assert!(
            laws::extract_reduction(&program).unwrap().is_none(),
            "{source}"
        );
        assert_cpu_agrees_with_reference(&source);
    }
}

/// Commuted operands, live seeds, mutable inputs and constant bounds are lawful.
#[test]
fn issue_m2_03_lawful_reduction_scan_accepts_lawful_source_variants() {
    let base = "fn main() -> i32 { let xs: Array<u32> = [2u32, 3u32, 4294967295u32]; var acc: u32 = 0u32; for i in 0u32 .. len(xs) { acc = wrapping_add(acc, xs[i]); } if acc == 4u32 { return 0; } return 1; }";
    let variants = [
        base.replace("wrapping_add(acc, xs[i])", "wrapping_add(xs[i], acc)"),
        base.replace("var acc: u32 = 0u32", "var acc: u32 = 9u32")
            .replace("acc == 4u32", "acc == 13u32"),
        base.replace("let xs: Array<u32>", "var xs: Array<u32>"),
        base.replace("0u32 .. len(xs)", "0u32 .. 4u32"),
        base.replace("wrapping_add(acc, xs[i])", "wrapping_mul(xs[i], acc)")
            .replace("var acc: u32 = 0u32", "var acc: u32 = 5u32"),
    ];
    for source in variants {
        let checked = check(parse(source.as_bytes()).unwrap(), true).unwrap();
        let program = mir::lower(&checked).unwrap();
        assert!(
            laws::extract_reduction(&program).unwrap().is_some(),
            "{source}"
        );
        assert_cpu_agrees_with_reference(&source);
    }
}

/// Reductions read live inputs and seeds: parameters, filled and moved owners.
#[test]
fn issue_m2_03_lawful_reduction_scan_live_inputs_and_seeds() {
    let data = (0..257)
        .map(|i| format!("{}u32", (i as u32).wrapping_mul(2_654_435_761)))
        .collect::<Vec<_>>()
        .join(",");
    let sources = [
        format!(
            "fn fold(xs: read Array<u32>, seed: u32) -> u32 {{ var acc: u32 = seed; for i in 0u32 .. len(xs) {{ acc = wrapping_add(xs[i], acc); }} return acc; }} \
             fn main() -> i32 {{ let values: Array<u32> = [{data}]; if fold(values, 7u32) == 0u32 {{ return 1; }} return 0; }}"
        ),
        "fn main() -> i32 { let xs: Array<u32> = filled_u32(100u32, 3u32); var acc: u32 = 2u32; for i in 0u32 .. len(xs) { acc = wrapping_mul(acc, xs[i]); } if acc == 0u32 { return 1; } return 0; }".to_string(),
        format!(
            "fn main() -> i32 {{ let first: Array<u32> = [{data}]; let xs: Array<u32> = move first; var acc: u32 = 1u32; acc = 5u32; for i in 0u32 .. len(xs) {{ acc = wrapping_add(acc, xs[i]); }} if acc == 0u32 {{ return 1; }} return 0; }}"
        ),
    ];
    for source in sources {
        let checked = check(parse(source.as_bytes()).unwrap(), true).unwrap();
        let program = mir::lower(&checked).unwrap();
        assert!(
            laws::extract_reduction(&program).unwrap().is_some(),
            "{source}"
        );
        assert_cpu_agrees_with_reference(&source);
        let report = invar::admission::analyze(source.as_bytes(), &program).unwrap();
        assert_eq!(
            report.regions()[0].plan_kind(),
            schedule::PlanKind::Reduction
        );
    }
}

/// The executed reduction comes from the graph: retargeting its array input or
/// staling its law revision changes the live result or rejects the graph.
#[test]
fn issue_m2_03_lawful_reduction_scan_graph_rewiring_and_stale_rule() {
    let source = "fn main() -> i32 { let xs: Array<u32> = [1u32, 2u32, 3u32]; let ys: Array<u32> = [10u32, 20u32, 30u32]; var acc: u32 = 4u32; for i in 0u32 .. len(xs) { acc = wrapping_add(acc, xs[i]); } if acc == 10u32 { return 0; } return 1; }";
    let checked = check(parse(source.as_bytes()).unwrap(), true).unwrap();
    let program = mir::lower(&checked).unwrap();
    assert_eq!(
        eval::observe(&checked, 10_000).result,
        Ok(eval::Value::I32(0))
    );
    let candidate = laws::extract_reduction(&program).unwrap().unwrap();
    let (xs, ys): (&[u32], &[u32]) = (&[1, 2, 3], &[10, 20, 30]);
    assert_eq!(candidate.plan.run_reduce(&[xs, ys], &[4], 3).unwrap(), 10);

    let mut rewired = candidate.graph.clone();
    let root = rewired.entry;
    let old = rewired.nodes[root].as_ref().unwrap().inputs[1];
    let retarget = rewired.push(invar::typed_net::Node {
        source_id: 99,
        ty: invar::typed_net::Type::ArrayU32,
        effect: invar::typed_net::Effect::Total,
        inputs: Vec::new(),
        op: invar::typed_net::Op::InputU32(1),
    });
    rewired.nodes[root].as_mut().unwrap().inputs[1] = retarget;
    rewired.nodes[old] = None;
    let plan = rewired
        .extract(&rewired.verify(invar::loops::GRAPH_LIMITS).unwrap())
        .unwrap();
    assert_eq!(plan.run_reduce(&[xs, ys], &[4], 3).unwrap(), 64);

    let mut stale = candidate.graph.clone();
    let root = stale.entry;
    if let Some(invar::typed_net::Op::Reduce { rule, .. }) =
        stale.nodes[root].as_mut().map(|node| &mut node.op)
    {
        *rule = "sealed-wrapping-u32-laws-v0";
    }
    assert_eq!(
        stale.verify(invar::loops::GRAPH_LIMITS).unwrap_err(),
        invar::typed_net::Error::RuleRejected
    );
    let mut wrong_type = candidate.graph.clone();
    let root = wrong_type.entry;
    wrong_type.nodes[root].as_mut().unwrap().inputs.swap(0, 1);
    assert!(wrong_type.verify(invar::loops::GRAPH_LIMITS).is_err());
}
