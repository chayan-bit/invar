//! Issue #4 (M4-04).
#![forbid(unsafe_code)]
use invar::compaction_source::{self, Budget, Contract, Descriptor, Error, Predicate};
use invar::{check, front, general_net, loops, mir, typed_net};
use invar::{eval, mir_eval};

fn budget() -> Budget {
    Budget {
        max_elements: 1024,
        max_bytes: 20_480,
        max_work: 7168,
    }
}
fn predicate(op: Predicate) -> &'static str {
    match op {
        Predicate::Less => "<",
        Predicate::LessEqual => "<=",
        Predicate::Equal => "==",
        Predicate::NotEqual => "!=",
        Predicate::GreaterEqual => ">=",
        Predicate::Greater => ">",
    }
}
fn families() -> [Predicate; 6] {
    [
        Predicate::Less,
        Predicate::LessEqual,
        Predicate::Equal,
        Predicate::NotEqual,
        Predicate::GreaterEqual,
        Predicate::Greater,
    ]
}
fn source(op: Predicate, renamed: bool) -> String {
    let mut s = format!(
        "fn run(xs: read Array<u32>, count: u32, cutoff: u32) -> Array<u32> {{
      var selected=0u32;
      for i in 0u32..count {{ if xs[i] {} cutoff {{ selected=selected+1u32; }} }}
      var out=filled_u32(selected,0u32);
      var pos=0u32;
      for j in 0u32..count {{ if xs[j] {} cutoff {{ out[pos]=xs[j]; pos=pos+1u32; }} }}
      return move out;
    }}",
        predicate(op),
        predicate(op)
    );
    if renamed {
        for (old, new) in [
            ("xs", "values"),
            ("count", "extent"),
            ("cutoff", "threshold"),
            ("selected", "retained"),
            ("out", "result"),
            ("pos", "position"),
        ] {
            s = s.replace(old, new);
        }
    }
    s
}
fn literals(v: &[u32]) -> String {
    v.iter()
        .map(|n| format!("{n}u32"))
        .collect::<Vec<_>>()
        .join(",")
}
fn caller(f: &str, input: &[u32], count: u32, cutoff: u32, expected: Option<&[u32]>) -> String {
    let check = expected.map_or(String::new(), |expected| {
        format!(
            "let expected: Array<u32> = [{}]; if len(answer)!={}u32 {{return 0;}}
        for k in 0u32..len(answer) {{if answer[k]!=expected[k] {{return 0;}}}}",
            literals(expected),
            expected.len()
        )
    });
    format!(
        "{f} fn main()->i32 {{let input: Array<u32> = [{}];
        let answer=run(input,{count}u32,{cutoff}u32); {check} return 7;}}",
        literals(input)
    )
}
fn oracle(
    op: Predicate,
    input: &[u32],
    count: usize,
    cutoff: u32,
) -> (Vec<u32>, Vec<u32>, Vec<u32>) {
    let indices: Vec<usize> = (0..count)
        .filter(|&i| match op {
            Predicate::Less => input[i] < cutoff,
            Predicate::LessEqual => input[i] <= cutoff,
            Predicate::Equal => input[i] == cutoff,
            Predicate::NotEqual => input[i] != cutoff,
            Predicate::GreaterEqual => input[i] >= cutoff,
            Predicate::Greater => input[i] > cutoff,
        })
        .collect();
    // Positions derive from the independently selected indices, not the plan's
    // affine recurrence, scanned flags or any compaction implementation helper.
    (
        (0..count)
            .map(|i| u32::from(indices.contains(&i)))
            .collect(),
        (0..count)
            .map(|i| indices.iter().filter(|&&j| j < i).count() as u32)
            .collect(),
        indices.into_iter().map(|i| input[i]).collect(),
    )
}
fn checked(s: &str) -> check::CheckedProgram {
    check::check(front::parse(s.as_bytes()).unwrap(), true).unwrap()
}
fn equal(a: &eval::Observation, b: &eval::Observation) {
    assert_eq!(a.result, b.result);
    assert_eq!(a.steps_used, b.steps_used);
    assert_eq!(a.cleanup_trace, b.cleanup_trace);
}
fn observe(s: &str, prefixes: bool) -> eval::Observation {
    let p = checked(s);
    let m = mir::lower(&p).unwrap();
    mir::verify(&m).unwrap();
    let net = general_net::Net::lower(&p, general_net::Limits::default()).unwrap();
    let proof = net.verify().unwrap();
    let done = eval::observe(&p, 100_000);
    let last = if prefixes {
        done.steps_used + 1
    } else {
        100_000
    };
    let first = if prefixes { 1 } else { last };
    for fuel in first..=last {
        let a = eval::observe(&p, fuel);
        equal(&a, &mir_eval::observe(&m, fuel));
        let b = general_net::oracle::observe(
            &net,
            &proof,
            general_net::oracle::Limits {
                fuel,
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(b.failure, None);
        equal(&a, &b.semantic);
    }
    done
}

#[test]
fn segmented_flattening_compaction_stable_source_cpu_boundaries() {
    let shapes: &[&[u32]] = &[
        &[],
        &[0],
        &[u32::MAX],
        &[0, 1],
        &[u32::MAX, 0, 1, u32::MAX, 0x80000000, 1, 19],
        &[8, 2, 7, 1, 9, 3, 0, 11, 4, 8, 5],
    ];
    let mut fixtures = 0;
    for op in families() {
        for renamed in [false, true] {
            for &input in shapes {
                let count = input.len();
                let cutoff = if fixtures % 3 == 0 { u32::MAX } else { 3 };
                let (flags, positions, expected) = oracle(op, input, count, cutoff);
                let s = caller(
                    &source(op, renamed),
                    input,
                    count as u32,
                    cutoff,
                    Some(&expected),
                );
                let c = Contract::from_source(s.as_bytes(), "run").unwrap();
                assert_eq!(c.function(), "run");
                assert_eq!(c.predicate(), op);
                assert_eq!(c.names()[3], if renamed { "result" } else { "out" });
                assert_ne!(c.source_hash(), [0; 32]);
                assert_ne!(c.mir_contract(), [0; 32]);
                assert!(c.loop_spans()[0].end < c.loop_spans()[1].start);
                c.verify_source(s.as_bytes()).unwrap();
                c.verify_mir(&c.verified_mir().unwrap()).unwrap();
                c.verify_extraction(c.graph(), c.plan()).unwrap();
                assert_eq!(c.graph().nodes.len(), 3);
                assert!(matches!(
                    c.plan(),
                    typed_net::Plan::AffineScan {
                        prefix: typed_net::Prefix::Exclusive,
                        ..
                    }
                ));
                let p = c.prepare(input, count as u32, cutoff, budget()).unwrap();
                assert_eq!(p.flags(), flags);
                assert_eq!(p.positions(), positions);
                assert_eq!(p.retained(), expected.len() as u32);
                assert_eq!(
                    p.required_bytes(),
                    16 * count as u64 + 4 * expected.len() as u64
                );
                assert_eq!(p.required_work(), 6 * count as u64 + expected.len() as u64);
                assert_eq!(p.rule(), compaction_source::RULE);
                p.verify_descriptor(p.descriptor()).unwrap();
                assert_eq!(p.execute().unwrap(), expected);
                assert_eq!(observe(&s, false).result, Ok(eval::Value::I32(7)));
                fixtures += 1;
            }
        }
    }
    assert_eq!(fixtures, 72);
}

#[test]
fn segmented_flattening_compaction_stable_selection_percentages_and_order() {
    let input: Vec<u32> = (0..100).map(|i| (i * 37) % 100).collect();
    let mut selections = Vec::new();
    for cutoff in [100, 99, 50, 1, 0] {
        let (_, _, expected) = oracle(Predicate::GreaterEqual, &input, 100, cutoff);
        let s = caller(
            &source(Predicate::GreaterEqual, false),
            &input,
            100,
            cutoff,
            Some(&expected),
        );
        let c = Contract::from_source(s.as_bytes(), "run").unwrap();
        let p = c.prepare(&input, 100, cutoff, budget()).unwrap();
        let output = p.execute().unwrap();
        assert_eq!(output, expected);
        assert_eq!(observe(&s, false).result, Ok(eval::Value::I32(7)));
        selections.push(output.len());
    }
    assert_eq!(selections, vec![0, 1, 50, 99, 100]);
}

#[test]
fn segmented_flattening_compaction_stable_exhaustive_flag_patterns() {
    let c =
        Contract::from_source(source(Predicate::GreaterEqual, false).as_bytes(), "run").unwrap();
    let mut cases = 0;
    for n in 0..=8 {
        for mask in 0..1u32 << n {
            let input: Vec<u32> = (0..n)
                .map(|i| if mask & (1 << i) != 0 { i + 1 } else { 0 })
                .collect();
            let p = c.prepare(&input, n, 1, budget()).unwrap();
            let expected: Vec<u32> = (0..n)
                .filter(|&i| mask & (1 << i) != 0)
                .map(|i| i + 1)
                .collect();
            let positions: Vec<u32> = (0..n)
                .map(|i| (mask & ((1 << i) - 1)).count_ones())
                .collect();
            assert_eq!(p.positions(), positions);
            assert_eq!(p.execute().unwrap(), expected);
            cases += 1;
        }
    }
    assert_eq!(cases, 511);
}

#[test]
fn segmented_flattening_compaction_stable_every_fuel_prefix_and_cleanup() {
    let mut prefixes = 0;
    for op in families() {
        for input in [&[][..], &[0][..], &[u32::MAX, 0, 1, 4][..]] {
            let count = input.len();
            let (_, _, expected) = oracle(op, input, count, 1);
            let s = caller(&source(op, false), input, count as u32, 1, None);
            let c = Contract::from_source(s.as_bytes(), "run").unwrap();
            let p = c.prepare(input, count as u32, 1, budget()).unwrap();
            let done = observe(&s, true);
            assert_eq!(done.result, Ok(eval::Value::I32(7)));
            assert_eq!(
                done.steps_used,
                25 + input.len() as u64 + 14 * count as u64 + 13 * expected.len() as u64
            );
            assert_eq!(
                done.steps_used,
                16 + input.len() as u64 + p.remaining_region_ticks().unwrap()
            );
            assert_eq!(
                done.cleanup_trace
                    .iter()
                    .map(|e| e.name.as_str())
                    .collect::<Vec<_>>(),
                vec!["answer", "input"]
            );
            prefixes += done.steps_used + 1;
        }
    }
    println!("stable compaction normal all-fuel-prefix observations: {prefixes}");
}

#[test]
fn segmented_flattening_compaction_stable_length_budget_fences_and_repeats() {
    let c =
        Contract::from_source(source(Predicate::GreaterEqual, false).as_bytes(), "run").unwrap();
    let input = [7, 0, 9, 2, 11];
    let exact = Budget {
        max_elements: 3,
        max_bytes: 56,
        max_work: 20,
    };
    let p = c.prepare(&input, 3, 5, exact).unwrap();
    assert_eq!(p.execute().unwrap(), vec![7, 9]);
    for b in [
        Budget {
            max_elements: 2,
            ..exact
        },
        Budget {
            max_bytes: 55,
            ..exact
        },
        Budget {
            max_work: 19,
            ..exact
        },
    ] {
        assert_eq!(c.prepare(&input, 3, 5, b).unwrap_err(), Error::Budget);
        let s = caller(
            &source(Predicate::GreaterEqual, false),
            &input,
            3,
            5,
            Some(&[7, 9]),
        );
        assert_eq!(observe(&s, false).result, Ok(eval::Value::I32(7)));
    }
    assert_eq!(
        c.prepare(&input, 6, 5, budget()).unwrap_err(),
        Error::Bounds
    );
    assert_eq!(
        c.prepare(&input, u32::MAX, 5, budget()).unwrap_err(),
        Error::Budget
    );
    assert_eq!(
        c.prepare(
            &input,
            0,
            5,
            Budget {
                max_elements: 0,
                max_bytes: 0,
                max_work: 0
            }
        )
        .unwrap()
        .execute()
        .unwrap(),
        Vec::<u32>::new()
    );
    let mut out = p.execute().unwrap();
    out[0] = 0;
    assert_eq!(p.execute().unwrap(), vec![7, 9]);
    assert_eq!(
        c.prepare(&input, 3, 8, budget())
            .unwrap()
            .execute()
            .unwrap(),
        vec![9]
    );
    assert_eq!(
        c.prepare(&[8, 9, 0], 3, 5, budget())
            .unwrap()
            .execute()
            .unwrap(),
        vec![8, 9]
    );
    assert_eq!(input, [7, 0, 9, 2, 11]);
}

#[test]
fn segmented_flattening_compaction_stable_descriptor_mutations() {
    let c =
        Contract::from_source(source(Predicate::GreaterEqual, false).as_bytes(), "run").unwrap();
    let p = c.prepare(&[7, 0, 9, 2], 4, 5, budget()).unwrap();
    let d = p.descriptor();
    let changed = [8, 0, 9, 2];
    let flags = [1, 0, 2, 0];
    let reversed = [1, 1, 0, 2];
    let short = [0, 1, 1];
    for mutation in [
        Descriptor {
            source_hash: [0; 32],
            ..d
        },
        Descriptor {
            mir_hash: [0; 32],
            ..d
        },
        Descriptor {
            predicate: Predicate::Less,
            ..d
        },
        Descriptor {
            input: &changed,
            ..d
        },
        Descriptor { count: 3, ..d },
        Descriptor { cutoff: 6, ..d },
        Descriptor { flags: &flags, ..d },
        Descriptor {
            positions: &reversed,
            ..d
        },
        Descriptor {
            positions: &short,
            ..d
        },
        Descriptor { retained: 3, ..d },
    ] {
        assert_eq!(
            p.verify_descriptor(mutation),
            Err(Error::DescriptorMismatch)
        );
    }
    assert_eq!(p.execute().unwrap(), vec![7, 9]);
}

#[test]
fn segmented_flattening_compaction_stable_source_near_matches_and_invalid_alias() {
    let f = source(Predicate::GreaterEqual, false);
    for changed in [
        f.replace("xs[j] >= cutoff", "xs[j] > cutoff"),
        f.replace("out[pos]=xs[j]", "out[selected-pos-1u32]=xs[j]"),
        f.replace("selected=selected+1u32", "selected=selected+2u32"),
        f.replace("filled_u32(selected,0u32)", "filled_u32(count,0u32)"),
        f.replace("for j in 0u32..count", "for j in 1u32..count"),
        f.replace("pos=pos+1u32", "pos=pos+2u32"),
        f.replace("var pos=0u32", "var pos=1u32"),
        f.replace("xs: read", "xs: mut"),
        f.replace("if xs[i] >= cutoff", "if xs[i] / 1u32 >= cutoff"),
    ] {
        assert_eq!(
            Contract::from_source(changed.as_bytes(), "run").unwrap_err(),
            Error::Unrecognized
        );
        let s = caller(&changed, &[7, 0, 9], 3, 5, None).replace("let input:", "var input:");
        observe(&s, true); // Some altered writes legitimately trap; exact serial outcome is retained.
    }
    let invalid = f.replace("out[pos]=xs[j]", "xs[j]=out[pos]");
    assert!(matches!(
        Contract::from_source(invalid.as_bytes(), "run"),
        Err(Error::Source(_))
    ));
    let alias = f.replace("var out=filled_u32(selected,0u32)", "var out=move xs");
    assert!(matches!(
        Contract::from_source(alias.as_bytes(), "run"),
        Err(Error::Source(_))
    ));
    assert_eq!(
        Contract::from_source(f.as_bytes(), "absent").unwrap_err(),
        Error::Unrecognized
    );
    let callback = format!(
        "{} fn pred(a:u32,b:u32)->Bool {{ return a>=b; }}",
        f.replace("xs[i] >= cutoff", "pred(xs[i],cutoff)")
            .replace("xs[j] >= cutoff", "pred(xs[j],cutoff)")
    );
    assert_eq!(
        Contract::from_source(callback.as_bytes(), "run").unwrap_err(),
        Error::Unrecognized
    );
    assert_eq!(
        observe(&caller(&callback, &[7, 0, 9], 3, 5, Some(&[7, 9])), false).result,
        Ok(eval::Value::I32(7))
    );
}

#[test]
fn segmented_flattening_compaction_stable_source_mir_and_scan_mutations() {
    let s = source(Predicate::GreaterEqual, false);
    let c = Contract::from_source(s.as_bytes(), "run").unwrap();
    assert_eq!(
        c.verify_source(s.replace(">=", ">").as_bytes()),
        Err(Error::SourceMismatch)
    );
    assert_eq!(
        c.verify_source(format!("{s} // changed bytes").as_bytes()),
        Err(Error::SourceMismatch)
    );
    let mut m = c.verified_mir().unwrap();
    m.layout_version += 1;
    assert_eq!(c.verify_mir(&m), Err(Error::MirMismatch));
    let mut m = c.verified_mir().unwrap();
    m.functions[0].span.end += 1;
    assert_eq!(c.verify_mir(&m), Err(Error::MirMismatch));
    let mut m = c.verified_mir().unwrap();
    let instruction = m.functions[0]
        .blocks
        .iter_mut()
        .flat_map(|b| &mut b.instructions)
        .find(|i| {
            matches!(
                &i.kind,
                mir::Operation::Binary {
                    op: front::BinaryOp::Ge,
                    ..
                }
            )
        })
        .unwrap();
    if let mir::Operation::Binary { op, .. } = &mut instruction.kind {
        *op = front::BinaryOp::Gt;
    }
    mir::verify(&m).unwrap();
    assert_eq!(c.verify_mir(&m), Err(Error::MirMismatch));
    let mut graph = c.graph().clone();
    graph.nodes[2].as_mut().unwrap().inputs.swap(0, 1);
    assert!(c.verify_extraction(&graph, c.plan()).is_err());
    let mut graph = c.graph().clone();
    if let typed_net::Op::AffineScan { prefix, .. } = &mut graph.nodes[2].as_mut().unwrap().op {
        *prefix = typed_net::Prefix::Inclusive;
    }
    graph.seal_source();
    let proof = graph.verify(loops::GRAPH_LIMITS).unwrap();
    assert_eq!(
        c.verify_extraction(&graph, &graph.extract(&proof).unwrap()),
        Err(Error::ExtractionMismatch)
    );
    let mut graph = c.graph().clone();
    if let typed_net::Op::AffineScan { a, .. } = &mut graph.nodes[2].as_mut().unwrap().op {
        *a = typed_net::Coefficient::Constant(2);
    }
    graph.seal_source();
    assert!(c.verify_extraction(&graph, c.plan()).is_err());
    let mut graph = c.graph().clone();
    graph.nodes[1].as_mut().unwrap().op = typed_net::Op::InputU32(1);
    graph.seal_source();
    assert!(c.verify_extraction(&graph, c.plan()).is_err());
    let mut graph = c.graph().clone();
    graph.nodes[2].as_mut().unwrap().effect = typed_net::Effect::MayFail;
    assert!(c.verify_extraction(&graph, c.plan()).is_err());
    let mut plan = c.plan().clone();
    if let typed_net::Plan::AffineScan { seed, .. } = &mut plan {
        *seed = 1;
    }
    assert_eq!(
        c.verify_extraction(c.graph(), &plan),
        Err(Error::ExtractionMismatch)
    );
}

#[test]
fn segmented_flattening_compaction_stable_exact_traps_fuel_and_cleanup() {
    let f = source(Predicate::GreaterEqual, false);
    let c = Contract::from_source(f.as_bytes(), "run").unwrap();
    assert_eq!(c.prepare(&[7], 2, 5, budget()).unwrap_err(), Error::Bounds);
    let s = caller(&f, &[7], 2, 5, None);
    let done = observe(&s, true);
    let start = s.find("xs[i]").unwrap();
    assert_eq!(
        done.result,
        Err(eval::EvalError::Trap {
            category: eval::TrapCategory::IndexOutOfBounds,
            span: front::Span {
                start,
                end: start + 5
            }
        })
    );
    assert!(done.cleanup_trace.is_empty());
    let earlier = s.replace("let answer=run", "let bad=1 / 0; let answer=run");
    let done = observe(&earlier, true);
    let start = earlier.find("/ 0").unwrap();
    assert_eq!(
        done.result,
        Err(eval::EvalError::Trap {
            category: eval::TrapCategory::DivisionByZero,
            span: front::Span {
                start,
                end: start + 1
            }
        })
    );
    assert!(done.cleanup_trace.is_empty());
    let trapping = f.replace("xs[i] >= cutoff", "xs[i] / cutoff >= 1u32");
    assert_eq!(
        Contract::from_source(trapping.as_bytes(), "run").unwrap_err(),
        Error::Unrecognized
    );
    let s = caller(&trapping, &[7], 1, 0, None);
    let done = observe(&s, true);
    let start = s.find("/ cutoff").unwrap();
    assert_eq!(
        done.result,
        Err(eval::EvalError::Trap {
            category: eval::TrapCategory::DivisionByZero,
            span: front::Span {
                start,
                end: start + 1
            }
        })
    );
    assert!(done.cleanup_trace.is_empty());
}
