use invar::{check::check, eval, front::parse, mir, mir_eval, recurrence};

#[test]
fn issue_m2_04_affine_recurrence_infers_source_coefficients_and_prefixes() {
    let source = "fn main() -> i32 { let aa: Array<u32> = [2u32, 3u32, 5u32]; let bb: Array<u32> = [7u32, 11u32, 13u32]; var state: u32 = 17u32; var out: Array<u32> = filled_u32(len(aa), 0u32); for i in 0u32 .. len(aa) { state = wrapping_add(wrapping_mul(aa[i], state), bb[i]); out[i] = state; } if out[0u32] != 41u32 { return 1; } if out[1u32] != 134u32 { return 1; } if out[2u32] != 683u32 { return 1; } return 0; }";
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
    let admission = invar::admission::analyze(source.as_bytes(), &program)
        .unwrap()
        .to_json();
    let obligations = admission["regions"][0]["obligations"].as_array().unwrap();
    for code in ["SEED", "PUBLICATION"] {
        assert!(
            obligations.iter().any(|item| item["name"] == code),
            "{code}"
        );
    }
    let candidate = recurrence::infer(&program).unwrap().candidate.unwrap();
    assert_eq!(candidate.state, "state");
    assert_eq!(candidate.arrays, vec!["aa", "bb"]);
    let arrays: [&[u32]; 2] = [&[2, 3, 5], &[7, 11, 13]];
    assert_eq!(
        candidate.plan.affine_coefficients(&arrays, 3).unwrap(),
        vec![
            invar::laws::Affine { a: 2, b: 7 },
            invar::laws::Affine { a: 3, b: 11 },
            invar::laws::Affine { a: 5, b: 13 },
        ]
    );
    assert_eq!(
        candidate.prefix_states(&arrays, 17, 3).unwrap(),
        vec![41, 134, 683]
    );
    assert_eq!(candidate.final_state(&arrays, 17, 3).unwrap(), 683);
    assert_eq!(
        mir_eval::observe(&candidate.fallback, 100_000).result,
        Ok(eval::Value::I32(0))
    );
}

type Case = (String, Vec<u32>, u32, [Vec<u32>; 2]);

fn source_case(len: usize, prefixes: bool, reordered: bool) -> Case {
    let a = (0..len)
        .map(|i| (i as u32).wrapping_mul(3).wrapping_add(2))
        .collect::<Vec<_>>();
    let b = (0..len + 2)
        .map(|i| (i as u32).wrapping_mul(7).wrapping_add(5))
        .collect::<Vec<_>>();
    let literal = |xs: &[u32]| {
        xs.iter()
            .map(|x| format!("{x}u32"))
            .collect::<Vec<_>>()
            .join(",")
    };
    let (a_name, b_name, state, index) = if reordered {
        ("factors", "offsets", "carry", "row")
    } else {
        ("aa", "bb", "state", "i")
    };
    let a_decl = format!("let {a_name}: Array<u32> = [{}];", literal(&a));
    let b_decl = format!("let {b_name}: Array<u32> = [{}];", literal(&b));
    let mut source = format!(
        "fn main() -> i32 {{ {} {} var {state}: u32 = 17u32;",
        if reordered { &b_decl } else { &a_decl },
        if reordered { &a_decl } else { &b_decl }
    );
    if prefixes {
        source.push_str(&format!(
            "var out: Array<u32> = filled_u32(len({a_name}), 0u32);"
        ));
    }
    source.push_str(&format!("for {index} in 0u32 .. len({a_name}) {{ {state} = wrapping_add(wrapping_mul({a_name}[{index}], {state}), {b_name}[{index}]);"));
    if prefixes {
        source.push_str(&format!("out[{index}] = {state};"));
    }
    source.push('}');
    let mut current = 17u32;
    let expected = (0..len)
        .map(|i| {
            current = a[i].wrapping_mul(current).wrapping_add(b[i]);
            current
        })
        .collect::<Vec<_>>();
    if prefixes {
        for (i, value) in expected.iter().enumerate() {
            source.push_str(&format!("if out[{i}u32] != {value}u32 {{ return 1; }}"));
        }
    }
    source.push_str(&format!(
        "if {state} != {current}u32 {{ return 1; }} return 0; }}"
    ));
    (source, expected, current, [a, b])
}

#[test]
fn issue_m2_04_affine_recurrence_shapes_final_and_every_prefix() {
    for len in [0, 1, 3, 7, 17, 257] {
        for prefixes in [false, true] {
            let (source, expected, final_state, [a, b]) = source_case(len, prefixes, len % 2 == 1);
            let arrays: [&[u32]; 2] = [&a, &b];
            let checked = check(parse(source.as_bytes()).unwrap(), true).unwrap();
            let program = mir::lower(&checked).unwrap();
            let reference = eval::observe(&checked, 1_000_000);
            let serial = mir_eval::observe(&program, 1_000_000);
            assert_eq!(reference.result, Ok(eval::Value::I32(0)), "len {len}");
            assert_eq!(serial.result, reference.result, "len {len}");
            assert_eq!(serial.cleanup_trace, reference.cleanup_trace, "len {len}");
            let report = recurrence::infer(&program).unwrap();
            assert!(report.rejected.is_empty(), "{:?}", report.rejected);
            let candidate = report.candidate.unwrap();
            assert_eq!(
                candidate
                    .plan
                    .affine_coefficients(&arrays, len)
                    .unwrap()
                    .len(),
                len
            );
            assert_eq!(candidate.prefix_states(&arrays, 17, len).unwrap(), expected);
            assert_eq!(
                candidate.final_state(&arrays, 17, len).unwrap(),
                final_state
            );
            assert_eq!(candidate.publishes_prefixes, prefixes);
            assert_eq!(candidate.rule_revision, recurrence::RULE_REVISION);
            assert!(
                source[candidate.source_span.start..candidate.source_span.end].starts_with("for ")
            );
            let (combines, depth) = candidate.scan_cost(&arrays, len).unwrap();
            assert!(combines <= 3 * len.max(1).next_power_of_two());
            assert!(depth <= 2 * len.max(1).next_power_of_two().trailing_zeros() as usize + 1);
        }
    }
}

#[test]
fn issue_m2_04_affine_recurrence_rejects_near_matches_with_reasons() {
    let (base, _, _, _) = source_case(3, true, false);
    let variants = [
        (
            base.replace("wrapping_mul(aa[i], state)", "wrapping_mul(state, state)"),
            "STATE_DEPENDENCE",
        ),
        (
            base.replace(
                "wrapping_mul(aa[i], state)",
                "wrapping_mul(wrapping_add(state, 1u32), state)",
            ),
            "SHAPE",
        ),
        (
            base.replace(
                "wrapping_add(wrapping_mul(aa[i], state), bb[i])",
                "wrapping_mul(aa[i], state) + bb[i]",
            ),
            "NUMERIC_POLICY",
        ),
        (
            format!(
                "fn step(x: u32) -> u32 {{ return x; }} {}",
                base.replace("bb[i]", "step(bb[i])")
            ),
            "UNKNOWN_EFFECT",
        ),
        (
            base.replace("out[i] = state;", "out[i] = wrapping_add(state, 0u32);"),
            "SHAPE",
        ),
        (base.replace("bb[i]", "bb[0u32]"), "COEFFICIENT"),
    ];
    for (source, reason) in variants {
        let checked = check(parse(source.as_bytes()).unwrap(), true).unwrap();
        let program = mir::lower(&checked).unwrap();
        let report = recurrence::infer(&program).unwrap();
        assert!(report.candidate.is_none(), "{source}");
        assert_eq!(report.rejected.len(), 1, "{source}");
        assert_eq!(report.rejected[0].reason, reason, "{source}");
        assert_cpu_agrees_with_reference(&source);
    }
    // Commuted operands, a preloop seed store and a short output are lawful; the
    // short output falls back at run time and traps exactly like the reference.
    for source in [
        base.replace(
            "wrapping_add(wrapping_mul(aa[i], state), bb[i])",
            "wrapping_add(bb[i], wrapping_mul(state, aa[i]))",
        ),
        base.replace(
            "var state: u32 = 17u32;",
            "var state: u32 = 17u32; state = 9u32;",
        ),
        base.replace("filled_u32(len(aa), 0u32)", "filled_u32(2u32, 0u32)"),
    ] {
        let checked = check(parse(source.as_bytes()).unwrap(), true).unwrap();
        let program = mir::lower(&checked).unwrap();
        assert!(
            recurrence::infer(&program).unwrap().candidate.is_some(),
            "{source}"
        );
        assert_cpu_agrees_with_reference(&source);
    }
}

#[test]
fn issue_m2_04_affine_recurrence_full_width_and_constant_coefficients() {
    let sources = [
        (
            "fn main() -> i32 { let aa: Array<u32> = [4294967295u32, 2u32]; let bb: Array<u32> = [1u32, 4294967295u32]; var state: u32 = 4294967295u32; var out: Array<u32> = filled_u32(len(aa), 0u32); for i in 0u32 .. len(aa) { state = wrapping_add(wrapping_mul(aa[i], state), bb[i]); out[i] = state; } if out[0u32] != 2u32 { return 1; } if out[1u32] != 3u32 { return 1; } return 0; }",
            vec![vec![u32::MAX, 2], vec![1, u32::MAX]],
            u32::MAX,
            vec![2, 3],
        ),
        (
            "fn main() -> i32 { let xs: Array<u32> = [0u32, 0u32, 0u32]; var state: u32 = 17u32; for i in 0u32 .. len(xs) { state = wrapping_add(wrapping_mul(3u32, state), 7u32); } if state == 550u32 { return 0; } return 1; }",
            vec![],
            17,
            vec![58, 181, 550],
        ),
    ];
    for (source, inputs, seed, expected) in sources {
        let arrays = inputs.iter().map(Vec::as_slice).collect::<Vec<_>>();
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
        let candidate = recurrence::infer(&program).unwrap().candidate.unwrap();
        let len = expected.len();
        assert_eq!(
            candidate.prefix_states(&arrays, seed, len).unwrap(),
            expected
        );
        assert_eq!(
            candidate.final_state(&arrays, seed, len).unwrap(),
            *expected.last().unwrap()
        );
    }
}

fn assert_cpu_agrees_with_reference(source: &str) {
    let checked = check(parse(source.as_bytes()).unwrap(), true).unwrap();
    let program = mir::lower(&checked).unwrap();
    let reference = eval::observe(&checked, 1_000_000);
    for workers in 1..=invar::schedule::MAX_WORKERS {
        let run =
            invar::schedule::run_source(source.as_bytes(), &program, 1_000_000, workers).unwrap();
        assert_eq!(
            run.observation.result, reference.result,
            "{workers} {source}"
        );
        assert_eq!(
            run.observation.steps_used, reference.steps_used,
            "{workers}"
        );
        assert_eq!(
            run.observation.cleanup_trace, reference.cleanup_trace,
            "{workers}"
        );
    }
}

/// `ys[i] = s` before the update publishes the exclusive prefix (seed first).
#[test]
fn issue_m2_04_affine_recurrence_regression_store_before_update_is_exclusive() {
    for len in [0usize, 1, 63, 64, 65, 257] {
        let xs = (0..len)
            .map(|i| format!("{}u32", (i as u32).wrapping_mul(2_654_435_761)))
            .collect::<Vec<_>>()
            .join(", ");
        let mut expected_exclusive = Vec::new();
        let mut expected_inclusive = Vec::new();
        let mut state = 9u32;
        for i in 0..len {
            expected_exclusive.push(state);
            state = 5u32
                .wrapping_mul(state)
                .wrapping_add((i as u32).wrapping_mul(2_654_435_761));
            expected_inclusive.push(state);
        }
        for (update_first, expected) in [(false, &expected_exclusive), (true, &expected_inclusive)]
        {
            let body = if update_first {
                "s = wrapping_add(wrapping_mul(5u32, s), xs[i]); ys[i] = s;"
            } else {
                "ys[i] = s; s = wrapping_add(wrapping_mul(5u32, s), xs[i]);"
            };
            let mut source = format!(
                "fn main() -> i32 {{ let xs: Array<u32> = [{xs}]; var s: u32 = 9u32; \
                 var ys: Array<u32> = filled_u32(len(xs), 0u32); \
                 for i in 0u32 .. len(xs) {{ {body} }}"
            );
            for (i, value) in expected.iter().enumerate() {
                source.push_str(&format!("if ys[{i}u32] != {value}u32 {{ return 1; }}"));
            }
            source.push_str(&format!("if s != {state}u32 {{ return 2; }} return 0; }}"));
            assert_cpu_agrees_with_reference(&source);
            let checked = check(parse(source.as_bytes()).unwrap(), true).unwrap();
            assert_eq!(
                eval::observe(&checked, 1_000_000).result,
                Ok(eval::Value::I32(0))
            );
        }
    }
    // A later checked add sees the exclusive prefix: 9 + 4294967286 fits, a published 45 traps.
    assert_cpu_agrees_with_reference(
        "fn main() -> i32 { let xs: Array<u32> = [0u32, 0u32]; var s: u32 = 9u32; \
         var ys: Array<u32> = filled_u32(len(xs), 0u32); \
         for i in 0u32 .. len(xs) { ys[i] = s; s = wrapping_add(wrapping_mul(5u32, s), xs[i]); } \
         let t: u32 = ys[0u32] + 4294967286u32; return 0; }",
    );
}

/// Scans read live coefficient arrays and seeds; the graph drives execution.
#[test]
fn issue_m2_04_affine_recurrence_live_inputs_and_graph_rewiring() {
    let source = "fn run(aa: read Array<u32>, bb: read Array<u32>, seed: u32) -> u32 { var s: u32 = seed; var out: Array<u32> = filled_u32(len(aa), 0u32); for i in 0u32 .. len(aa) { s = wrapping_add(wrapping_mul(s, aa[i]), bb[i]); out[i] = s; } return wrapping_add(s, out[0u32]); } \
        fn main() -> i32 { let aa: Array<u32> = filled_u32(70u32, 3u32); let bb: Array<u32> = [1u32, 2u32, 3u32, 4u32, 5u32, 6u32, 7u32, 8u32, 9u32, 10u32, 11u32, 12u32, 13u32, 14u32, 15u32, 16u32, 17u32, 18u32, 19u32, 20u32, 21u32, 22u32, 23u32, 24u32, 25u32, 26u32, 27u32, 28u32, 29u32, 30u32, 31u32, 32u32, 33u32, 34u32, 35u32, 36u32, 37u32, 38u32, 39u32, 40u32, 41u32, 42u32, 43u32, 44u32, 45u32, 46u32, 47u32, 48u32, 49u32, 50u32, 51u32, 52u32, 53u32, 54u32, 55u32, 56u32, 57u32, 58u32, 59u32, 60u32, 61u32, 62u32, 63u32, 64u32, 65u32, 66u32, 67u32, 68u32, 69u32, 70u32]; \
        if run(aa, bb, 11u32) == 0u32 { return 1; } return 0; }";
    assert_cpu_agrees_with_reference(source);
    let checked = check(parse(source.as_bytes()).unwrap(), true).unwrap();
    let program = mir::lower(&checked).unwrap();
    let report = invar::admission::analyze(source.as_bytes(), &program).unwrap();
    assert_eq!(
        report.regions()[0].plan_kind(),
        invar::schedule::PlanKind::Scan
    );
    let candidate = recurrence::infer(&program).unwrap().candidate.unwrap();
    assert_eq!(candidate.arrays, vec!["aa", "bb"]);
    let (aa, bb): (&[u32], &[u32]) = (&[2, 3], &[5, 7]);
    assert_eq!(
        candidate.prefix_states(&[aa, bb], 1, 2).unwrap(),
        vec![7, 28]
    );

    // Swapping the coefficient inputs is verifier-valid and changes the live result.
    let mut swapped = candidate.graph.clone();
    let root = swapped.entry;
    swapped.nodes[root].as_mut().unwrap().inputs.swap(1, 2);
    let plan = swapped
        .extract(&swapped.verify(invar::loops::GRAPH_LIMITS).unwrap())
        .unwrap();
    let coefficients = plan.affine_coefficients(&[aa, bb], 2).unwrap();
    assert_eq!(
        coefficients,
        vec![
            invar::laws::Affine { a: 5, b: 2 },
            invar::laws::Affine { a: 7, b: 3 }
        ]
    );

    let mut stale = candidate.graph.clone();
    if let Some(invar::typed_net::Op::AffineScan { rule, .. }) =
        stale.nodes[root].as_mut().map(|node| &mut node.op)
    {
        *rule = "wrapping-affine-recurrence-v0";
    }
    assert_eq!(
        stale.verify(invar::loops::GRAPH_LIMITS).unwrap_err(),
        invar::typed_net::Error::RuleRejected
    );
}
