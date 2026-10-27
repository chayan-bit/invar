#![forbid(unsafe_code)]
use invar::segmented_source::{self, Budget, Contract, Error};
use invar::{check, front, general_net, laws, loops, mir, typed_net};
use invar::{eval, mir_eval};

fn budget() -> Budget {
    Budget {
        max_elements: 1024,
        max_segments: 1024,
        max_output_bytes: 4096,
        max_work: 2048,
    }
}
fn function(product: bool, renamed: bool) -> String {
    let mut source = format!(
        "fn run(xs: read Array<u32>, offsets: read Array<u32>, count: u32) -> Array<u32> {{
          var out = filled_u32(count, 0u32);
          for s in 0u32..count {{
            var acc = {}u32;
            for i in offsets[s]..offsets[s + 1u32] {{
              acc = {}(acc, xs[i]);
            }}
            out[s] = acc;
          }}
          return move out;
        }}",
        if product { 1 } else { 0 },
        if product {
            "wrapping_mul"
        } else {
            "wrapping_add"
        }
    );
    if renamed {
        for (old, new) in [
            ("xs", "samples"),
            ("offsets", "boundaries"),
            ("count", "groups"),
            ("out", "result"),
            ("acc", "subtotal"),
        ] {
            source = source.replace(old, new);
        }
    }
    source
}
fn literals(values: &[u32]) -> String {
    values
        .iter()
        .map(|n| format!("{n}u32"))
        .collect::<Vec<_>>()
        .join(",")
}
fn caller(f: &str, input: &[u32], offsets: &[u32], count: u32, expected: Option<&[u32]>) -> String {
    let checks = expected.map_or(String::new(), |expected| {
        format!(
            "let expected: Array<u32> = [{}];
          if len(answer) != {}u32 {{ return 0; }}
          for j in 0u32..len(answer) {{ if answer[j] != expected[j] {{ return 0; }} }}",
            literals(expected),
            expected.len()
        )
    });
    format!(
        "{f} fn main() -> i32 {{ let input: Array<u32> = [{}];
      let bounds: Array<u32> = [{}]; let answer = run(input,bounds,{count}u32);
      {checks} return 7; }}",
        literals(input),
        literals(offsets)
    )
}
fn scalar_oracle(product: bool, input: &[u32], offsets: &[u32]) -> Vec<u32> {
    offsets
        .windows(2)
        .map(|p| {
            // Mathematical modulo arithmetic, independent of laws::combine/Plan.
            let mut state = if product { 1u128 } else { 0 };
            for &n in &input[p[0] as usize..p[1] as usize] {
                state = if product {
                    state * u128::from(n)
                } else {
                    state + u128::from(n)
                };
                state %= 1u128 << 32;
            }
            state as u32
        })
        .collect()
}
fn checked(s: &str) -> check::CheckedProgram {
    check::check(front::parse(s.as_bytes()).unwrap(), true).unwrap()
}
fn equal(a: &eval::Observation, b: &eval::Observation) {
    assert_eq!(a.result, b.result);
    assert_eq!(a.steps_used, b.steps_used);
    assert_eq!(a.cleanup_trace, b.cleanup_trace);
}
fn source_observations(s: &str, prefixes: bool) -> eval::Observation {
    let p = checked(s);
    let m = mir::lower(&p).unwrap();
    mir::verify(&m).unwrap();
    let net = general_net::Net::lower(&p, general_net::Limits::default()).unwrap();
    let proof = net.verify().unwrap();
    let done = eval::observe(&p, 100_000);
    let upper = if prefixes {
        done.steps_used + 1
    } else {
        100_000
    };
    let lower = if prefixes { 1 } else { upper };
    for fuel in lower..=upper {
        let a = eval::observe(&p, fuel);
        let b = mir_eval::observe(&m, fuel);
        equal(&a, &b);
        let c = general_net::oracle::observe(
            &net,
            &proof,
            general_net::oracle::Limits {
                fuel,
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(c.failure, None);
        equal(&a, &c.semantic);
    }
    done
}

#[test]
fn issue_m4_04_segmented_flattening_compaction_source_cpu_boundaries() {
    let shapes: &[&[u32]] = &[
        &[0],
        &[0, 0],
        &[0, 0, 0, 0],
        &[0, 1],
        &[0, 1, 2, 3],
        &[0, 0, 2, 2, 5],
        &[0, 1, 8, 9],
        &[0, 0, 0, 9, 9, 9],
        &[0, 3, 6, 9],
        &[0, 1, 1, 2, 7, 9],
    ];
    let mut fixtures = 0;
    for product in [false, true] {
        for renamed in [false, true] {
            for &offsets in shapes {
                let input: Vec<u32> = (0..*offsets.last().unwrap())
                    .map(|i| [u32::MAX, 0x80000000, 0, 1, 3, 11, 19][i as usize % 7])
                    .collect();
                let expected = scalar_oracle(product, &input, offsets);
                let source = caller(
                    &function(product, renamed),
                    &input,
                    offsets,
                    (offsets.len() - 1) as u32,
                    Some(&expected),
                );
                let contract = Contract::from_source(source.as_bytes(), "run").unwrap();
                contract.verify_source(source.as_bytes()).unwrap();
                assert_eq!(contract.function(), "run");
                assert_eq!(
                    contract.law(),
                    if product {
                        laws::Law::WrappingProduct
                    } else {
                        laws::Law::WrappingSum
                    }
                );
                assert!(contract.loop_span().start < contract.loop_span().end);
                assert_ne!(contract.mir_contract(), [0; 32]);
                assert_ne!(contract.source_hash(), [0; 32]);
                assert_eq!(contract.names()[3], if renamed { "result" } else { "out" });
                let m = contract.verified_mir().unwrap();
                contract.verify_mir(&m).unwrap();
                let prepared = contract
                    .prepare(&input, offsets, (offsets.len() - 1) as u32, budget())
                    .unwrap();
                assert_eq!(prepared.offsets(), offsets);
                assert_eq!(prepared.output_bytes(), expected.len() as u64 * 4);
                assert_eq!(prepared.work(), (input.len() + expected.len()) as u64);
                assert_eq!(prepared.rule(), segmented_source::RULE);
                assert_eq!(prepared.execute().unwrap(), expected);
                assert_eq!(contract.graph().nodes.len(), 3);
                assert_eq!(contract.plan().loop_count(), 1);
                let done = source_observations(&source, false);
                assert_eq!(done.result, Ok(eval::Value::I32(7)));
                fixtures += 1;
            }
        }
    }
    assert_eq!(fixtures, 40);
}

#[test]
fn issue_m4_04_segmented_flattening_compaction_exhaustive_small_boundaries() {
    let mut cases = 0;
    for product in [false, true] {
        let c = Contract::from_source(function(product, false).as_bytes(), "run").unwrap();
        // Enumerate every placement of internal boundaries for lengths 1..7,
        // inserting empty leading/interior/trailing segments deterministically.
        for n in 0..=7u32 {
            let input: Vec<u32> = (0..n).map(|i| u32::MAX - i).collect();
            for mask in 0..1u32 << n.saturating_sub(1) {
                let mut offsets = vec![0, 0];
                for i in 1..n {
                    if mask & (1 << (i - 1)) != 0 {
                        offsets.extend([i, i]);
                    }
                }
                offsets.extend([n, n]);
                let prepared = c
                    .prepare(&input, &offsets, (offsets.len() - 1) as u32, budget())
                    .unwrap();
                assert_eq!(
                    prepared.execute().unwrap(),
                    scalar_oracle(product, &input, &offsets)
                );
                cases += 1;
            }
        }
    }
    assert_eq!(cases, 256);
}

#[test]
fn issue_m4_04_segmented_flattening_compaction_every_source_fuel_prefix_and_cleanup() {
    let mut observations = 0;
    for (input, offsets) in [
        (&[][..], &[0][..]),
        (&[][..], &[0, 0, 0][..]),
        (&[u32::MAX, 3, 0, 7][..], &[0, 1, 1, 4][..]),
    ] {
        for product in [false, true] {
            let source = caller(
                &function(product, false),
                input,
                offsets,
                (offsets.len() - 1) as u32,
                None,
            );
            let c = Contract::from_source(source.as_bytes(), "run").unwrap();
            let p = c
                .prepare(input, offsets, (offsets.len() - 1) as u32, budget())
                .unwrap();
            let done = source_observations(&source, true);
            assert_eq!(done.result, Ok(eval::Value::I32(7)));
            // Main setup + callee allocation/bounds/return + main return is
            // 21 + input literals + segment count. Remaining nested ticks are
            // checked against actual independently interpreted source execution.
            assert_eq!(
                done.steps_used,
                21 + input.len() as u64
                    + (offsets.len() - 1) as u64
                    + p.remaining_loop_ticks().unwrap()
            );
            assert_eq!(
                done.cleanup_trace
                    .iter()
                    .map(|e| e.name.as_str())
                    .collect::<Vec<_>>(),
                vec!["answer", "bounds", "input"]
            );
            observations += done.steps_used + 1;
        }
    }
    println!("all-prefix source/MIR/general-net observations: {observations}");
}

#[test]
fn issue_m4_04_segmented_flattening_compaction_invalid_offsets_keep_serial_meaning() {
    let c = Contract::from_source(function(false, false).as_bytes(), "run").unwrap();
    let input = [2, 3, 5];
    for (offsets, count, error) in [
        (vec![], 0, Error::OffsetShape),
        (vec![0], 1, Error::OffsetShape),
        (vec![0, 3, 2], 2, Error::OffsetOrder),
        (vec![0, 4], 1, Error::OffsetBounds),
        (vec![0, u32::MAX], 1, Error::OffsetBounds),
        (vec![1, 3], 1, Error::IncompleteCoverage),
        (vec![0, 2], 1, Error::IncompleteCoverage),
    ] {
        assert_eq!(
            c.prepare(&input, &offsets, count, budget()).unwrap_err(),
            error
        );
        // No completion was produced. The original source still executes,
        // including legitimate decreasing/partial ranges and original traps.
        let source = caller(&function(false, false), &input, &offsets, count, None);
        let done = source_observations(&source, true);
        if matches!(error, Error::OffsetBounds | Error::OffsetShape) && count > 0 {
            assert!(matches!(
                done.result,
                Err(eval::EvalError::Trap {
                    category: eval::TrapCategory::IndexOutOfBounds,
                    ..
                })
            ));
            assert!(done.cleanup_trace.is_empty());
        } else {
            assert_eq!(done.result, Ok(eval::Value::I32(7)));
        }
    }
    assert_eq!(
        c.prepare(&[], &[0], 0, budget())
            .unwrap()
            .execute()
            .unwrap(),
        Vec::<u32>::new()
    );
    assert_eq!(
        c.prepare(&[], &[0], u32::MAX, budget()).unwrap_err(),
        Error::Budget
    );
}

#[test]
fn issue_m4_04_segmented_flattening_compaction_exact_budget_and_fresh_completion() {
    let c = Contract::from_source(function(false, false).as_bytes(), "run").unwrap();
    let input = [u32::MAX, 2, 5];
    let offsets = [0, 0, 2, 3];
    let exact = Budget {
        max_elements: 3,
        max_segments: 3,
        max_output_bytes: 12,
        max_work: 6,
    };
    assert_eq!(
        c.prepare(&input, &offsets, 3, exact)
            .unwrap()
            .execute()
            .unwrap(),
        vec![0, 1, 5]
    );
    for refused in [
        Budget {
            max_elements: 2,
            ..exact
        },
        Budget {
            max_segments: 2,
            ..exact
        },
        Budget {
            max_output_bytes: 11,
            ..exact
        },
        Budget {
            max_work: 5,
            ..exact
        },
    ] {
        assert_eq!(
            c.prepare(&input, &offsets, 3, refused).unwrap_err(),
            Error::Budget
        );
    }
    let mut first = c
        .prepare(&input, &offsets, 3, exact)
        .unwrap()
        .execute()
        .unwrap();
    first[1] = 99;
    assert_eq!(
        c.prepare(&input, &offsets, 3, exact)
            .unwrap()
            .execute()
            .unwrap(),
        vec![0, 1, 5]
    );
    let changed = [7, 8, 9];
    assert_eq!(
        c.prepare(&changed, &offsets, 3, exact)
            .unwrap()
            .execute()
            .unwrap(),
        vec![0, 15, 9]
    );
    assert_eq!(input, [u32::MAX, 2, 5]);
    assert_eq!(offsets, [0, 0, 2, 3]);
}

#[test]
fn issue_m4_04_segmented_flattening_compaction_source_near_matches_and_invalid_source() {
    let f = function(false, false);
    for changed in [
        f.replace("wrapping_add(acc, xs[i])", "acc + xs[i]"),
        f.replace("var acc = 0u32", "var acc = 1u32"),
        f.replace("out[s] = acc", "out[count - s - 1u32] = acc"),
        f.replace("offsets[s + 1u32]", "offsets[s + 2u32]"),
        f.replace(
            "wrapping_add(acc, xs[i])",
            "wrapping_add(acc, xs[i] / 1u32)",
        ),
        f.replace(
            "acc = wrapping_add(acc, xs[i]);",
            "acc = wrapping_add(acc, xs[i]); out[s] = acc;",
        ),
        f.replace("filled_u32(count, 0u32)", "filled_u32(count + 1u32, 0u32)"),
        f.replace("xs: read", "xs: mut"),
    ] {
        assert_eq!(
            Contract::from_source(changed.as_bytes(), "run").unwrap_err(),
            Error::Unrecognized
        );
        let source =
            caller(&changed, &[2, 3], &[0, 1, 2, 2], 2, None).replace("let input:", "var input:");
        assert_eq!(
            source_observations(&source, false).result,
            Ok(eval::Value::I32(7))
        );
    }
    let invalid = f.replace("read Array<u32>", "Array<u32>");
    assert!(matches!(
        Contract::from_source(invalid.as_bytes(), "run"),
        Err(Error::Source(_))
    ));
    assert_eq!(
        Contract::from_source(f.as_bytes(), "absent").unwrap_err(),
        Error::Unrecognized
    );
    // A total-looking user callback is still unknown, not silently admitted.
    let hidden = format!(
        "{} fn callback(a:u32,b:u32)->u32 {{ return wrapping_add(a,b); }}",
        f.replace("wrapping_add(acc, xs[i])", "callback(acc, xs[i])")
    );
    assert_eq!(
        Contract::from_source(hidden.as_bytes(), "run").unwrap_err(),
        Error::Unrecognized
    );
}

#[test]
fn issue_m4_04_segmented_flattening_compaction_source_mir_mutations() {
    let source = function(false, false);
    let c = Contract::from_source(source.as_bytes(), "run").unwrap();
    assert_eq!(
        c.verify_source(source.replace("0u32", "1u32").as_bytes()),
        Err(Error::SourceMismatch)
    );
    assert_eq!(
        c.verify_source(format!("{source} // new bytes").as_bytes()),
        Err(Error::SourceMismatch)
    );
    let mut m = c.verified_mir().unwrap();
    let f = m.functions.iter_mut().find(|f| f.name == "run").unwrap();
    let call = f
        .blocks
        .iter_mut()
        .flat_map(|b| &mut b.instructions)
        .find(|i| matches!(&i.kind,mir::Operation::Call{name,..} if name == "wrapping_add"))
        .unwrap();
    if let mir::Operation::Call { name, .. } = &mut call.kind {
        *name = "wrapping_mul".into();
    }
    mir::verify(&m).unwrap(); // A valid alternative program still cannot keep this binding.
    assert_eq!(c.verify_mir(&m), Err(Error::MirMismatch));
    let mut stale = c.verified_mir().unwrap();
    stale.layout_version += 1;
    assert_eq!(c.verify_mir(&stale), Err(Error::MirMismatch));
    let mut span = c.verified_mir().unwrap();
    span.functions[0].span.end += 1;
    assert_eq!(c.verify_mir(&span), Err(Error::MirMismatch));
}

#[test]
fn issue_m4_04_segmented_flattening_compaction_typed_graph_plan_mutations() {
    let c = Contract::from_source(function(false, false).as_bytes(), "run").unwrap();
    c.verify_extraction(c.graph(), c.plan()).unwrap();
    let mut graph = c.graph().clone();
    graph.nodes[2].as_mut().unwrap().inputs.swap(0, 1);
    assert!(c.verify_extraction(&graph, c.plan()).is_err());
    let mut graph = c.graph().clone();
    graph.nodes[2].as_mut().unwrap().op = typed_net::Op::Reduce {
        law: laws::Law::WrappingProduct,
        rule: laws::RULE_REVISION,
    };
    graph.seal_source();
    let proof = graph.verify(loops::GRAPH_LIMITS).unwrap();
    let altered = graph.extract(&proof).unwrap();
    assert_eq!(
        c.verify_extraction(&graph, &altered),
        Err(Error::ExtractionMismatch)
    );
    let mut graph = c.graph().clone();
    graph.nodes[1].as_mut().unwrap().op = typed_net::Op::InputU32(1);
    graph.seal_source();
    assert!(c.verify_extraction(&graph, c.plan()).is_err());
    let mut graph = c.graph().clone();
    graph.nodes[2].as_mut().unwrap().effect = typed_net::Effect::MayFail;
    assert!(c.verify_extraction(&graph, c.plan()).is_err());
    let mut graph = c.graph().clone();
    graph.nodes[2].as_mut().unwrap().op = typed_net::Op::Reduce {
        law: laws::Law::WrappingSum,
        rule: "unreviewed",
    };
    graph.seal_source();
    assert!(c.verify_extraction(&graph, c.plan()).is_err());
    let mut plan = c.plan().clone();
    if let typed_net::Plan::Reduce { seed, .. } = &mut plan {
        *seed = 1;
    }
    assert_eq!(
        c.verify_extraction(c.graph(), &plan),
        Err(Error::ExtractionMismatch)
    );
}

#[test]
fn issue_m4_04_segmented_flattening_compaction_source_failure_order_and_cleanup() {
    let f = function(false, false);
    let source = caller(&f, &[7], &[0, 2], 1, None);
    let done = source_observations(&source, true);
    let failed_index = source.find("xs[i]").unwrap();
    assert_eq!(
        done.result,
        Err(eval::EvalError::Trap {
            category: eval::TrapCategory::IndexOutOfBounds,
            span: front::Span {
                start: failed_index,
                end: failed_index + "xs[i]".len()
            }
        })
    );
    assert!(done.cleanup_trace.is_empty());
    // A failure before the candidate is reached must keep its exact source span.
    let before = source.replace("let answer = run", "let bad = 1 / 0; let answer = run");
    let done = source_observations(&before, true);
    let slash = before.find("/ 0").unwrap();
    assert_eq!(
        done.result,
        Err(eval::EvalError::Trap {
            category: eval::TrapCategory::DivisionByZero,
            span: front::Span {
                start: slash,
                end: slash + 1
            }
        })
    );
    assert!(done.cleanup_trace.is_empty());
}

fn grouped_function(product: bool, renamed: bool) -> String {
    let mut s = format!(
        "fn run(xs: read Array<u32>, offsets: read Array<u32>, count: u32) -> Array<u32> {{
          var out = (filled_u32((count), (0u32)));
          for s in (0u32)..(count) {{
            var acc = ({}u32);
            for i in ((offsets)[(s)])..((offsets)[((s) + (1u32))]) {{
              acc = ({}((acc), ((xs)[(i)])));
            }}
            out[(s)] = (acc);
          }}
          return move out;
        }}",
        if product { 1 } else { 0 },
        if product {
            "wrapping_mul"
        } else {
            "wrapping_add"
        }
    );
    if renamed {
        for (old, new) in [
            ("xs", "samples"),
            ("offsets", "boundaries"),
            ("count", "groups"),
            ("out", "result"),
            ("acc", "subtotal"),
            ("for s", "for segment_index"),
            ("(s)", "(segment_index)"),
            ("for i", "for flat_index"),
            ("(i)", "(flat_index)"),
        ] {
            s = s.replace(old, new);
        }
    }
    s
}

#[test]
fn issue_m4_04_segmented_flattening_compaction_library_grouped_renamed_fuel_prefixes() {
    let mut fixtures = 0;
    let mut prefixes = 0;
    for product in [false, true] {
        for renamed in [false, true] {
            for grouped in [false, true] {
                for offsets in [&[0][..], &[0, 0, 0][..], &[0, 1][..], &[0, 1, 1, 4][..]] {
                    let input: Vec<u32> = (0..*offsets.last().unwrap())
                        .map(|i| [u32::MAX, 3, 0, 7][i as usize])
                        .collect();
                    let f = if grouped {
                        grouped_function(product, renamed)
                    } else {
                        function(product, renamed)
                    };
                    let count = (offsets.len() - 1) as u32;
                    let mut s = caller(&f, &input, offsets, count, None);
                    let entry = if renamed {
                        s = s.replace("run(", "fold_segments(");
                        "fold_segments"
                    } else {
                        "run"
                    };
                    let c = Contract::from_source(s.as_bytes(), entry).unwrap();
                    assert_eq!(c.function(), entry);
                    let ast = checked(&s);
                    let f = ast.program.function(entry).unwrap();
                    assert_eq!(c.loop_span(), f.body.statements[1].span);
                    c.verify_source(s.as_bytes()).unwrap();
                    c.verify_mir(&c.verified_mir().unwrap()).unwrap();
                    let p = c.prepare(&input, offsets, count, budget()).unwrap();
                    // Hand-derived independent expectation remains in the tests; the
                    // library derives these weights from this specific checked AST.
                    assert_eq!(
                        p.remaining_loop_ticks(),
                        Some(15 * u64::from(count) + 7 * input.len() as u64)
                    );
                    assert_eq!(
                        p.execute().unwrap(),
                        scalar_oracle(product, &input, offsets)
                    );
                    let done = source_observations(&s, true);
                    assert_eq!(done.result, Ok(eval::Value::I32(7)));
                    assert_eq!(
                        done.steps_used,
                        21 + input.len() as u64
                            + u64::from(count)
                            + p.remaining_loop_ticks().unwrap()
                    );
                    prefixes += done.steps_used + 1;
                    fixtures += 1;
                    let value_check = caller(
                        &if grouped {
                            grouped_function(product, renamed)
                        } else {
                            function(product, renamed)
                        },
                        &input,
                        offsets,
                        count,
                        Some(&scalar_oracle(product, &input, offsets)),
                    );
                    assert_eq!(
                        source_observations(&value_check, false).result,
                        Ok(eval::Value::I32(7))
                    );
                }
            }
        }
    }
    assert_eq!(fixtures, 32);
    println!("library grouped/renamed fixtures: {fixtures}; normal fuel prefixes: {prefixes}");
}

#[test]
fn issue_m4_04_segmented_flattening_compaction_library_grouped_near_matches() {
    let f = grouped_function(false, true);
    for changed in [
        f.replace("(1u32)", "(2u32)"),
        f.replace(
            "wrapping_add((subtotal), ((samples)[(flat_index)]))",
            "subtotal + samples[flat_index]",
        ),
        f.replace(
            "result[(segment_index)]",
            "result[groups-segment_index-1u32]",
        ),
        f.replace("var subtotal = (0u32)", "var subtotal = (1u32)"),
    ] {
        assert_ne!(changed, f);
        assert_eq!(
            Contract::from_source(changed.as_bytes(), "run").unwrap_err(),
            Error::Unrecognized
        );
        let s = caller(&changed, &[2, 3, 5], &[0, 1, 2, 3], 2, None);
        assert_eq!(
            source_observations(&s, true).result,
            Ok(eval::Value::I32(7))
        );
    }
}

#[test]
fn issue_m4_04_segmented_flattening_compaction_library_source_loop_span_revalidation() {
    let s = caller(&grouped_function(false, true), &[2, 3], &[0, 0, 2], 2, None);
    let c = Contract::from_source(s.as_bytes(), "run").unwrap();
    let other = s
        .replace("for segment_index", "for another_segment")
        .replace("(segment_index)", "(another_segment)");
    Contract::from_source(other.as_bytes(), "run").unwrap();
    assert_eq!(
        c.verify_source(other.as_bytes()),
        Err(Error::SourceMismatch)
    );
    let other_c = Contract::from_source(other.as_bytes(), "run").unwrap();
    assert_eq!(
        c.verify_mir(&other_c.verified_mir().unwrap()),
        Err(Error::MirMismatch)
    );
    let mut m = c.verified_mir().unwrap();
    let tick = m
        .functions
        .iter_mut()
        .find(|f| f.name == "run")
        .unwrap()
        .blocks
        .iter_mut()
        .flat_map(|b| &mut b.instructions)
        .find(|i| matches!(i.kind, mir::Operation::Tick) && i.span == c.loop_span())
        .unwrap();
    tick.span.end += 1;
    mir::verify(&m).unwrap();
    assert_eq!(c.verify_mir(&m), Err(Error::MirMismatch));
    let mut m = c.verified_mir().unwrap();
    let call = m
        .functions
        .iter_mut()
        .find(|f| f.name == "run")
        .unwrap()
        .blocks
        .iter_mut()
        .flat_map(|b| &mut b.instructions)
        .find(|i| {
            matches!(&i.kind,
            mir::Operation::Call{name,..} if name=="wrapping_add")
        })
        .unwrap();
    if let mir::Operation::Call { name, .. } = &mut call.kind {
        *name = "wrapping_mul".into();
    }
    mir::verify(&m).unwrap();
    assert_eq!(c.verify_mir(&m), Err(Error::MirMismatch));
    // Revalidation checks the immutable candidate's source and structure. Even
    // successful checks and prepare confer no reached-source publication permit.
    assert_eq!(
        c.prepare(&[2, 3], &[0, 0, 2], 2, budget())
            .unwrap()
            .execute()
            .unwrap(),
        vec![0, 5]
    );
}
