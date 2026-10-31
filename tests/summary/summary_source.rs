#![forbid(unsafe_code)] // Issue #6 (M4-07).
#[path = "../support/summary_source_host.rs"]
mod reached_host;
use invar::{
    bounded_summary::Budget,
    check, eval, front,
    general_net::{self, oracle},
    mir, mir_eval,
    summary_source::{Contract, Error, Family, Observation},
    typed_net::{BoundedFamily, Plan, Prefix},
};

fn budget() -> Budget {
    Budget {
        max_steps: 1024,
        max_bytes: 1_000_000,
        max_composition_ops: 1_000_000,
    }
}
fn checked(s: &str) -> check::CheckedProgram {
    check::check(front::parse(s.as_bytes()).unwrap(), false).unwrap()
}
fn source(family: Family, prefix: bool) -> String {
    match family {
        Family::Table(k) => format!(
            "fn run(coeff: read Array<u32>, seed: u32, count: u32) -> {} {{ var state = seed; {} for i in 0u32..count {{ state = coeff[i * {}u32 + state]; {} }} return {}; }}",
            if prefix { "Array<u32>" } else { "u32" },
            if prefix {
                "var out = filled_u32(count,0u32);"
            } else {
                ""
            },
            k,
            if prefix { "out[i] = state;" } else { "" },
            if prefix { "out" } else { "state" }
        ),
        Family::Matrix(d) => {
            let d = usize::from(d);
            let mut s = "fn run(coeff: read Array<u32>, seed: read Array<u32>, count: u32) -> Array<u32> { ".to_owned();
            for row in 0..d {
                s.push_str(&format!("var x{row} = seed[{row}u32]; "));
            }
            if prefix {
                s.push_str(&format!("var out = filled_u32(count * {d}u32,0u32); "));
            }
            s.push_str("for i in 0u32..count { ");
            for row in 0..d {
                let mut dot = "0u32".to_owned();
                for col in 0..d {
                    dot = format!(
                        "wrapping_add({dot},wrapping_mul(coeff[i * {}u32 + {}u32],x{col}))",
                        d * d,
                        row * d + col
                    );
                }
                s.push_str(&format!("let y{row} = {dot}; "));
            }
            for row in 0..d {
                s.push_str(&format!("x{row} = y{row}; "));
            }
            if prefix {
                for row in 0..d {
                    s.push_str(&format!("out[i * {d}u32 + {row}u32] = x{row}; "));
                }
            }
            s.push_str("} return ");
            if prefix {
                s.push_str("out;");
            } else {
                s.push_str(&format!(
                    "[{}];",
                    (0..d)
                        .map(|j| format!("x{j}"))
                        .collect::<Vec<_>>()
                        .join(",")
                ));
            }
            s.push_str(" }");
            s
        }
    }
}
fn data(family: Family, steps: u32) -> (Vec<u32>, Vec<u32>) {
    match family {
        Family::Table(k) => (
            (0..steps)
                .flat_map(|row| {
                    (0..u32::from(k)).map(move |col| (col * (row + 1) + row + 1) % u32::from(k))
                })
                .collect(),
            vec![u32::from(k) - 1],
        ),
        Family::Matrix(d) => (
            (0..steps * u32::from(d) * u32::from(d))
                .map(|i| {
                    if i % 3 == 0 {
                        u32::MAX
                    } else {
                        i.wrapping_mul(747796405).wrapping_add(2891336453)
                    }
                })
                .collect(),
            [u32::MAX, 0x80000000, 3, 11][..usize::from(d)].to_vec(),
        ),
    }
}
fn independent(family: Family, coeff: &[u32], seed: &[u32]) -> Vec<Vec<u32>> {
    let width = match family {
        Family::Table(k) => usize::from(k),
        Family::Matrix(d) => usize::from(d) * usize::from(d),
    };
    let mut state = seed.to_vec();
    let mut all = Vec::new();
    for step in coeff.chunks_exact(width) {
        state = match family {
            Family::Table(_) => vec![step[state[0] as usize]],
            Family::Matrix(d) => (0..usize::from(d))
                .map(|row| {
                    let sum = (0..usize::from(d)).fold(0u128, |sum, col| {
                        sum + u128::from(step[row * usize::from(d) + col]) * u128::from(state[col])
                    });
                    (sum % (1u128 << 32)) as u32
                })
                .collect(),
        };
        all.push(state.clone());
    }
    all
}
fn literals(values: &[u32]) -> String {
    values
        .iter()
        .map(|n| format!("{n}u32"))
        .collect::<Vec<_>>()
        .join(",")
}
fn caller(
    function: &str,
    family: Family,
    steps: u32,
    coeff: &[u32],
    seed: &[u32],
    expected: Option<&[u32]>,
) -> String {
    let seed_decl = match family {
        Family::Table(_) => format!("let initial = {}u32;", seed[0]),
        Family::Matrix(_) => format!("let initial: Array<u32> = [{}];", literals(seed)),
    };
    let validation=expected.map_or(String::new(),|expected| {
        if function.contains("-> u32") {format!("if answer != {}u32 {{ return 0; }}",expected[0])}
        else {format!("let expected: Array<u32> = [{}]; if len(answer) != {}u32 {{ return 0; }} for j in 0u32..len(answer) {{ if answer[j] != expected[j] {{ return 0; }} }}",literals(expected),expected.len())}
    });
    format!(
        "{function} fn main() -> i32 {{ let input: Array<u32> = [{}]; {seed_decl} let answer = run(input,initial,{steps}u32); {validation} return 7; }}",
        literals(coeff)
    )
}
fn families() -> [Family; 6] {
    [
        Family::Table(1),
        Family::Table(2),
        Family::Table(16),
        Family::Matrix(1),
        Family::Matrix(2),
        Family::Matrix(4),
    ]
}

#[test]
fn summary_source_real_checked_graph_and_every_bounded_prefix() {
    let mut fixtures = 0;
    for family in families() {
        for prefix in [false, true] {
            for steps in 0..=7 {
                let f = source(family, prefix);
                let (coeff, seed) = data(family, steps);
                let independent = independent(family, &coeff, &seed);
                let final_state = independent.last().unwrap_or(&seed);
                let expected = if prefix {
                    independent.iter().flatten().copied().collect::<Vec<_>>()
                } else {
                    final_state.clone()
                };
                let s = caller(&f, family, steps, &coeff, &seed, Some(&expected));
                let p = checked(&s);
                let c = Contract::from_checked(&p, "run").unwrap();
                let prepared = c.prepare(&coeff, &seed, steps, budget()).unwrap();
                assert_eq!(c.family(), family);
                assert_eq!(
                    c.state_names(),
                    match family {
                        Family::Table(_) => vec!["state".to_owned()],
                        Family::Matrix(d) =>
                            (0..d).map(|row| format!("x{row}")).collect::<Vec<_>>(),
                    }
                );
                assert_eq!(prepared.prefixes(), independent);
                assert_eq!(prepared.final_state(), final_state);
                assert_eq!(prepared.coefficients(), coeff);
                assert_eq!(
                    c.observation(),
                    if prefix {
                        Observation::InclusivePrefixes
                    } else {
                        Observation::FinalState
                    }
                );
                assert_eq!(prepared.observation(), c.observation());
                assert_eq!(prepared.source_span(), c.loop_span());
                assert_eq!(prepared.function_contract(), c.function_contract());
                assert_eq!(prepared.source_fuel(), c.loop_fuel(steps).unwrap());
                assert!(s[c.loop_span().start..c.loop_span().end].starts_with("for i"));
                let m = c.verified_mir().unwrap();
                mir::verify(&m).unwrap();
                let source = eval::observe(&p, 100_000);
                let serial = mir_eval::observe(&m, 100_000);
                assert_eq!(
                    source.result,
                    Ok(eval::Value::I32(7)),
                    "family={family:?} prefix={prefix} steps={steps}"
                );
                assert_eq!(serial.result, source.result);
                assert_eq!(serial.steps_used, source.steps_used);
                assert_eq!(serial.cleanup_trace, source.cleanup_trace);
                let (net, proof) = c.checked_net();
                let actual = oracle::observe(
                    net,
                    proof,
                    oracle::Limits {
                        fuel: 100_000,
                        ..oracle::Limits::default()
                    },
                )
                .unwrap();
                assert_eq!(actual.failure, None);
                assert_eq!(actual.semantic.result, source.result);
                assert_eq!(actual.semantic.steps_used, source.steps_used);
                assert_eq!(actual.semantic.cleanup_trace, source.cleanup_trace);
                c.verify_prepared(&prepared).unwrap();
                let plan = prepared.typed_plan().unwrap();
                c.verify_plan(&prepared, &plan, &coeff, &seed, steps, budget())
                    .unwrap();
                let scalars = [seed[0], steps];
                let arrays = [&coeff[..], &seed[..]];
                let candidate = plan.run_bounded_scan(&arrays, &scalars, budget()).unwrap();
                assert_eq!(candidate.seed(), prepared.seed());
                assert_eq!(candidate.prefixes(), prepared.prefixes());
                assert_eq!(candidate.final_state(), prepared.final_state());
                assert_eq!(candidate.required_bytes(), prepared.required_bytes());
                assert_eq!(candidate.required_work(), prepared.required_work());
                fixtures += 1;
            }
        }
    }
    println!(
        "actual_source_mir_net_fixtures={fixtures} typed_table_matrix_candidate_available=true reached_source_publication=false"
    );
    assert_eq!(fixtures, 96);
}

#[test]
fn summary_source_renaming_groups_and_observations_are_source_bound() {
    for family in families() {
        for prefix in [false, true] {
            let original = source(family, prefix);
            // Add transparent grouping at independently selected operand/bound sites.
            let grouped = original
                .replace("i * ", "(i) * ")
                .replace("0u32..count", "(0u32)..(count)")
                .replace("var state = seed;", "var state = (seed);")
                .replace("seed[0u32]", "(seed[(0u32)])");
            let renamed = grouped
                .replace("run(", "fold(")
                .replace("coeff", "transition_data")
                .replace("count", "length")
                .replace("seed", "initial_state");
            let c = Contract::from_checked(&checked(&original), "run").unwrap();
            let g = Contract::from_checked(&checked(&grouped), "run").unwrap();
            let r = Contract::from_checked(&checked(&renamed), "fold").unwrap();
            assert_eq!(r.coefficients(), "transition_data");
            assert_eq!(r.seed(), "initial_state");
            assert_eq!(r.count(), "length");
            assert_eq!(r.function(), "fold");
            assert_eq!(c.loop_fuel(7), g.loop_fuel(7));
            assert_eq!(g.loop_fuel(7), r.loop_fuel(7));
            let (coeff, seed) = data(family, 3);
            let prepared = c.prepare(&coeff, &seed, 3, budget()).unwrap();
            assert_eq!(g.verify_prepared(&prepared), Err(Error::RuntimeMismatch));
            assert_eq!(r.verify_prepared(&prepared), Err(Error::RuntimeMismatch));
            assert_ne!(c.function_contract(), r.function_contract());
            assert_eq!(c.output().is_some(), prefix);
        }
    }
}

#[test]
fn summary_source_unknown_effects_alias_policy_and_shapes_refuse() {
    let t = source(Family::Table(2), true);
    let m = source(Family::Matrix(2), true);
    // The exact family bound and state-dependent coefficients keep a
    // distinct reason; every other near miss is simply not a scan form.
    for (s, expected) in [
        (source(Family::Table(17), false), Error::FamilyBound),
        (source(Family::Table(17), true), Error::FamilyBound),
        (source(Family::Matrix(5), false), Error::FamilyBound),
        (source(Family::Matrix(5), true), Error::FamilyBound),
        (
            t.replace("coeff[i * 2u32 + state]", "coeff[state * 2u32 + state]"),
            Error::StateDependent,
        ),
        (
            m.replace(
                "wrapping_mul(coeff[i * 4u32 + 1u32],x1)",
                "wrapping_mul(coeff[x0 * 4u32 + 1u32],x1)",
            ),
            Error::StateDependent,
        ),
        (
            m.replace(
                "wrapping_mul(coeff[i * 4u32 + 0u32],x0)",
                "wrapping_mul(x1,x0)",
            ),
            Error::StateDependent,
        ),
    ] {
        let p = checked(&s);
        assert_eq!(
            Contract::from_checked(&p, "run").err(),
            Some(expected),
            "{s}"
        );
    }
    for s in [
        t.replace("coeff[i * 2u32 + state]", "coeff[i * 2u32 + 0u32]"),
        t.replace("0u32..count", "1u32..count"),
        t.replace("out[i] = state;", "out[i] = seed;"),
        t.replace(
            "out[i] = state;",
            "out[i] = state; let ignored = len(coeff);",
        ),
        t.replace("coeff: read", "coeff: mut"),
        t.replace(
            "state = coeff[i * 2u32 + state]; out[i] = state;",
            "out[i] = state; state = coeff[i * 2u32 + state];",
        ),
        m.replace("i * 4u32 + 3u32", "i * 4u32 + 2u32"),
        m.replace(
            "wrapping_mul(coeff[i * 4u32 + 0u32],x0)",
            "coeff[i * 4u32 + 0u32] * x0",
        ),
        m.replace("x0 = y0;", "x0 = y1;"),
        m.replace("count * 2u32", "count + 2u32"),
        m.replace("seed: read", "seed: mut"),
    ] {
        let p = checked(&s);
        assert!(
            matches!(Contract::from_checked(&p, "run"), Err(Error::Unrecognized)),
            "{s}"
        );
    }
    let helper = t.replace(
        "state = coeff[i * 2u32 + state]",
        "state = opaque(coeff[i * 2u32 + state])",
    ) + " fn opaque(n: u32) -> u32 { return n; }";
    assert!(matches!(
        Contract::from_checked(&checked(&helper), "run"),
        Err(Error::Unrecognized)
    ));
    let invalid = t.replace("var out = filled_u32(count,0u32);", "var out = coeff;");
    let e = check::check(front::parse(invalid.as_bytes()).unwrap(), false).unwrap_err();
    assert_eq!(e.code, "OWN_MOVE_REQUIRED");
    assert!(matches!(
        Contract::from_checked(&checked(&t), "absent"),
        Err(Error::Unrecognized)
    ));
}

#[test]
fn summary_source_runtime_mutation_all_frozen_budgets_and_empty() {
    for family in families() {
        let c = Contract::from_checked(&checked(&source(family, true)), "run").unwrap();
        let (mut coeff, seed) = data(family, 3);
        let original = coeff.clone();
        let p = c.prepare(&coeff, &seed, 3, budget()).unwrap();
        let exact = Budget {
            max_steps: 3,
            max_bytes: p.required_bytes(),
            max_composition_ops: p.required_work(),
        };
        let exact_p = c.prepare(&coeff, &seed, 3, exact).unwrap();
        assert_eq!(p.prefixes(), exact_p.prefixes());
        for field in 0..3 {
            let mut l = exact;
            match field {
                0 => l.max_steps -= 1,
                1 => l.max_bytes -= 1,
                2 => l.max_composition_ops -= 1,
                _ => unreachable!(),
            }
            assert!(matches!(
                c.prepare(&coeff, &seed, 3, l),
                Err(Error::Budget(_))
            ));
            assert_eq!(
                exact_p.verify_inputs(&coeff, &seed, 3, l),
                Err(Error::RuntimeMismatch)
            );
        }
        exact_p.verify_inputs(&coeff, &seed, 3, exact).unwrap();
        coeff[0] ^= 1;
        assert_eq!(p.coefficients(), original);
        assert_eq!(
            p.verify_inputs(&coeff, &seed, 3, budget()),
            Err(Error::RuntimeMismatch)
        );
        let mut changed = seed.clone();
        changed[0] ^= 1;
        assert_eq!(
            p.verify_inputs(&original, &changed, 3, budget()),
            Err(Error::RuntimeMismatch)
        );
        assert_eq!(
            p.verify_inputs(&original, &seed, 2, budget()),
            Err(Error::RuntimeMismatch)
        );
        assert!(matches!(
            c.prepare(&original[..original.len() - 1], &seed, 3, budget()),
            Err(Error::Shape)
        ));
        assert!(matches!(
            c.prepare(&original, &[], 3, budget()),
            Err(Error::Seed)
        ));
        let empty = c
            .prepare(
                &[],
                &seed,
                0,
                Budget {
                    max_steps: 0,
                    max_bytes: 0,
                    max_composition_ops: 0,
                },
            )
            .unwrap();
        assert!(empty.prefixes().is_empty());
        assert_eq!(empty.final_state(), seed);
        assert_eq!(empty.source_fuel(), 0);
        assert_eq!(empty.required_bytes(), 0);
        assert_eq!(empty.required_work(), 0);
        let typed = p.typed_plan().unwrap();
        c.verify_plan(&p, &typed, &original, &seed, 3, budget())
            .unwrap();
        let candidate = typed
            .run_bounded_scan(&[&original, &seed], &[seed[0], 3], exact)
            .unwrap();
        assert_eq!(candidate.prefixes(), p.prefixes());
        let empty_candidate = empty
            .typed_plan()
            .unwrap()
            .run_bounded_scan(&[&[], &seed], &[seed[0], 0], empty.budget())
            .unwrap();
        assert_eq!(empty_candidate.final_state(), seed);
        println!(
            "family={family:?} steps=3 exact_bytes={} exact_work={} typed_candidate_available=true reached_source_publication=false",
            p.required_bytes(),
            p.required_work()
        );
    }
    let c = Contract::from_checked(&checked(&source(Family::Table(2), false)), "run").unwrap();
    assert!(matches!(
        c.prepare(&[0, 2], &[0], 1, budget()),
        Err(Error::Shape)
    ));
    assert!(matches!(
        c.prepare(&[0, 1], &[2], 1, budget()),
        Err(Error::Seed)
    ));
    assert!(matches!(
        c.prepare(&[], &[2], 0, budget()),
        Err(Error::Seed)
    ));
    let c = Contract::from_checked(&checked(&source(Family::Matrix(4), true)), "run").unwrap();
    assert!(matches!(
        c.prepare(&[], &[0; 4], u32::MAX, budget()),
        Err(Error::Bounds)
    ));
    let c = Contract::from_checked(&checked(&source(Family::Matrix(1), false)), "run").unwrap();
    let same = [3u32];
    assert!(matches!(
        c.prepare(&same, &same, 1, budget()),
        Err(Error::Alias)
    ));
}

#[test]
fn summary_source_exact_source_fuel_and_all_small_failure_prefixes() {
    let mut fuel_cases = 0;
    for family in families() {
        for prefix in [false, true] {
            for steps in [0, 1, 2] {
                let f = source(family, prefix);
                let (coeff, seed) = data(family, steps);
                let s = caller(&f, family, steps, &coeff, &seed, None);
                let p = checked(&s);
                let c = Contract::from_checked(&p, "run").unwrap();
                let m = c.verified_mir().unwrap();
                let (net, proof) = c.checked_net();
                let prepared = c.prepare(&coeff, &seed, steps, budget()).unwrap();
                let plan = prepared.typed_plan().unwrap();
                c.verify_plan(&prepared, &plan, &coeff, &seed, steps, budget())
                    .unwrap();
                let candidate = plan
                    .run_bounded_scan(&[&coeff, &seed], &[seed[0], steps], budget())
                    .unwrap();
                assert_eq!(candidate.prefixes(), independent(family, &coeff, &seed));
                let full = eval::observe(&p, 100_000);
                assert_eq!(full.result, Ok(eval::Value::I32(7)));
                let mut baseline = s.clone();
                baseline.replace_range(c.loop_span().start..c.loop_span().end, "");
                let empty = eval::observe(&checked(&baseline), 100_000);
                assert_eq!(empty.result, Ok(eval::Value::I32(7)));
                assert_eq!(
                    full.steps_used - empty.steps_used - 3,
                    c.loop_fuel(steps).unwrap()
                );
                for fuel in 1..=full.steps_used + 1 {
                    let a = eval::observe(&p, fuel);
                    let b = mir_eval::observe(&m, fuel);
                    let g = oracle::observe(
                        net,
                        proof,
                        oracle::Limits {
                            fuel,
                            ..oracle::Limits::default()
                        },
                    )
                    .unwrap();
                    assert_eq!(
                        b.result, a.result,
                        "family={family:?} prefix={prefix} steps={steps} fuel={fuel}"
                    );
                    assert_eq!(b.steps_used, a.steps_used);
                    assert_eq!(b.cleanup_trace, a.cleanup_trace);
                    assert_eq!(g.failure, None);
                    assert_eq!(g.semantic.result, a.result);
                    assert_eq!(g.semantic.steps_used, a.steps_used);
                    assert_eq!(g.semantic.cleanup_trace, a.cleanup_trace);
                    fuel_cases += 1;
                }
            }
        }
    }
    println!("actual_source_mir_net_fuel_prefix_comparisons={fuel_cases}");
    assert!(fuel_cases > 1000);
}

#[test]
fn summary_source_refusals_retain_original_traps_and_spans() {
    let cases = [
        (Family::Table(2), vec![0], vec![1], 1),
        (Family::Matrix(2), vec![0; 4], vec![1], 1),
        (Family::Matrix(2), vec![0], vec![1, 2], 1),
    ];
    for (family, coeff, seed, steps) in cases {
        let f = source(family, false);
        let s = caller(&f, family, steps, &coeff, &seed, None);
        let p = checked(&s);
        let c = Contract::from_checked(&p, "run").unwrap();
        let m = c.verified_mir().unwrap();
        assert!(c.prepare(&coeff, &seed, steps, budget()).is_err());
        let complete = eval::observe(&p, 1000);
        assert!(matches!(
            complete.result,
            Err(eval::EvalError::Trap {
                category: eval::TrapCategory::IndexOutOfBounds,
                ..
            })
        ));
        for fuel in 1..=complete.steps_used + 1 {
            let a = eval::observe(&p, fuel);
            let b = mir_eval::observe(&m, fuel);
            assert_eq!(a.result, b.result);
            assert_eq!(a.steps_used, b.steps_used);
            assert_eq!(a.cleanup_trace, b.cleanup_trace);
        }
        println!(
            "private_refusal family={family:?} source_failure={:?}",
            complete.result
        );
    }
    // Invalid final table entries/seeds need not trap in the original source;
    // reject optimization without promoting the refusal to a source failure.
    let f = source(Family::Table(2), false);
    for (coeff, seed, steps) in [(vec![4, 4], vec![0], 1), (vec![], vec![16], 0)] {
        let s = caller(&f, Family::Table(2), steps, &coeff, &seed, None);
        let p = checked(&s);
        let c = Contract::from_checked(&p, "run").unwrap();
        assert!(c.prepare(&coeff, &seed, steps, budget()).is_err());
        assert_eq!(eval::observe(&p, 1000).result, Ok(eval::Value::I32(7)));
    }
    let f = source(Family::Matrix(1), false).replace(
        "wrapping_mul(coeff[i * 1u32 + 0u32],x0)",
        "coeff[i * 1u32 + 0u32] * x0",
    );
    let s = caller(&f, Family::Matrix(1), 1, &[u32::MAX], &[u32::MAX], None);
    let p = checked(&s);
    assert!(matches!(
        Contract::from_checked(&p, "run"),
        Err(Error::Unrecognized)
    ));
    assert!(matches!(
        eval::observe(&p, 1000).result,
        Err(eval::EvalError::Trap {
            category: eval::TrapCategory::IntegerOverflow,
            ..
        })
    ));
    let f = source(Family::Matrix(1), false);
    let s = format!(
        "{f} fn main() -> i32 {{ let input: Array<u32> = [3u32]; let answer = run(input,input,1u32); if answer[0u32] != 9u32 {{ return 0; }} return 7; }}"
    );
    let p = checked(&s);
    let c = Contract::from_checked(&p, "run").unwrap();
    let source = eval::observe(&p, 1000);
    let serial = mir_eval::observe(&c.verified_mir().unwrap(), 1000);
    assert_eq!(source.result, Ok(eval::Value::I32(7)));
    assert_eq!(serial.result, source.result);
    assert_eq!(serial.cleanup_trace, source.cleanup_trace);
}

#[test]
fn summary_source_checked_annotations_net_policy_and_copy_mutations() {
    let s = source(Family::Matrix(2), false);
    let p = checked(&s);
    let c = Contract::from_checked(&p, "run").unwrap();
    let m = c.verified_mir().unwrap();
    let original = m.contract_digest();
    let mut changed = m.clone();
    changed.functions[0].name = "changed".to_owned();
    assert_ne!(changed.contract_digest(), original);
    assert_eq!(c.verified_mir().unwrap().contract_digest(), original);
    changed = m.clone();
    changed.layout_version += 1;
    assert!(general_net::Net::encode(&changed, general_net::Limits::default()).is_err());
    assert!(invar::numeric_contract::NumericPolicy::from_parts(2, "strict-f32").is_err());
    assert!(invar::numeric_contract::NumericPolicy::from_parts(1, "relaxed-f32").is_err());
    let (net, proof) = c.checked_net();
    let mut stale = net.clone();
    stale.graph.revision += 1;
    assert!(stale.compiled_mir(proof).is_err());
    let mut stale = net.clone();
    stale.contract_version += 1;
    assert!(stale.verify().is_err());
    let mut changed = checked(&s);
    changed.expression_types.clear();
    assert!(matches!(
        Contract::from_checked(&changed, "run"),
        Err(Error::CheckedMismatch)
    ));
    let mut changed = checked(&s);
    changed
        .function_expression_types
        .get_mut("run")
        .unwrap()
        .pop();
    assert!(matches!(
        Contract::from_checked(&changed, "run"),
        Err(Error::CheckedMismatch)
    ));
    let mut changed = checked(&s);
    changed.program.functions[0].name = "changed".to_owned();
    assert!(matches!(
        Contract::from_checked(&changed, "run"),
        Err(Error::CheckedMismatch)
    ));
}

#[test]
fn summary_source_typed_plan_requires_exact_private_source_and_runtime_binding() {
    let mut mutations = 0;
    for family in families() {
        for prefix in [false, true] {
            let p = checked(&source(family, prefix));
            let contract = Contract::from_checked(&p, "run").unwrap();
            let sibling = Contract::from_checked(&p, "run").unwrap();
            let (coeff, seed) = data(family, 3);
            let prepared = contract.prepare(&coeff, &seed, 3, budget()).unwrap();
            let foreign = sibling.prepare(&coeff, &seed, 3, budget()).unwrap();
            let plan = prepared.typed_plan().unwrap();
            // Equal graph hashes and public plan values never replace the source instance.
            assert_eq!(foreign.typed_plan().unwrap(), plan);
            assert_eq!(
                contract.verify_plan(&foreign, &plan, &coeff, &seed, 3, budget()),
                Err(Error::RuntimeMismatch)
            );
            assert_eq!(
                sibling.verify_plan(&prepared, &plan, &coeff, &seed, 3, budget()),
                Err(Error::RuntimeMismatch)
            );
            let graph = prepared.typed_graph().unwrap();
            let proof = graph
                .verify(invar::typed_net::Limits {
                    nodes: 4,
                    ports: 7,
                    depth: 2,
                    array_len: 1_000_000,
                    map_ops: 0,
                })
                .unwrap();
            assert_eq!(graph.extract(&proof).unwrap(), plan);
            let mut plans = Vec::new();
            for index in 0..7 {
                let mut changed = plan.clone();
                if let Plan::BoundedScan {
                    family,
                    prefix,
                    seed,
                    input,
                    steps,
                    source_identity,
                    graph_revision,
                } = &mut changed
                {
                    match index {
                        0 => {
                            *family = match family {
                                BoundedFamily::Table(_) => BoundedFamily::Table(3),
                                BoundedFamily::Matrix(_) => BoundedFamily::Matrix(3),
                            }
                        }
                        1 => {
                            *prefix = if *prefix == Prefix::None {
                                Prefix::Inclusive
                            } else {
                                Prefix::None
                            }
                        }
                        2 => *seed += 1,
                        3 => *input += 1,
                        4 => *steps += 1,
                        5 => *source_identity ^= 1,
                        6 => *graph_revision += 1,
                        _ => unreachable!(),
                    }
                }
                plans.push(changed);
            }
            for changed in plans {
                assert_eq!(prepared.verify_plan(&changed), Err(Error::RuntimeMismatch));
                assert_eq!(
                    contract.verify_plan(&prepared, &changed, &coeff, &seed, 3, budget()),
                    Err(Error::RuntimeMismatch)
                );
                mutations += 1;
            }
            let mut bad_coeff = coeff.clone();
            bad_coeff[0] ^= 1;
            assert_eq!(
                contract.verify_plan(&prepared, &plan, &bad_coeff, &seed, 3, budget()),
                Err(Error::RuntimeMismatch)
            );
            let mut bad_seed = seed.clone();
            bad_seed[0] ^= 1;
            assert_eq!(
                contract.verify_plan(&prepared, &plan, &coeff, &bad_seed, 3, budget()),
                Err(Error::RuntimeMismatch)
            );
            assert_eq!(
                contract.verify_plan(&prepared, &plan, &coeff, &seed, 2, budget()),
                Err(Error::RuntimeMismatch)
            );
            for altered in [
                Budget {
                    max_steps: 999,
                    ..budget()
                },
                Budget {
                    max_bytes: 999_999,
                    ..budget()
                },
                Budget {
                    max_composition_ops: 999_999,
                    ..budget()
                },
            ] {
                assert_eq!(
                    contract.verify_plan(&prepared, &plan, &coeff, &seed, 3, altered),
                    Err(Error::RuntimeMismatch)
                );
            }
            contract
                .verify_plan(&prepared, &plan, &coeff, &seed, 3, budget())
                .unwrap();
            // Prepared storage does not borrow mutable caller storage.
            coeff
                .iter()
                .zip(prepared.coefficients())
                .for_each(|(a, b)| assert_eq!(a, b));
        }
    }
    println!(
        "public_plan_fact_mutations_denied={mutations} same_hash_cross_source_instance_denied=24"
    );
    assert_eq!(mutations, 84);
}
