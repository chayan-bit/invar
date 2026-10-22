#[path = "support/language_limits.rs"]
mod language_limits;
#[cfg(feature = "native")]
use invar::native;
use invar::{admission, check::check, eval, front::parse, loops, mir, mir_eval, typed_net};

const SOURCE: &str = r#"
fn main() -> i32 {
    let data: Array<u32> = [1u32, 2u32, 4294967295u32];
    var first: Array<u32> = filled_u32(len(data), 0u32);
    for i in 0u32 .. len(data) { first[i] = wrapping_add(data[i], 3u32); }
    let stable: Array<u32> = move first;
    var second: Array<u32> = filled_u32(len(stable), 0u32);
    for j in 0u32 .. len(stable) { second[j] = wrapping_mul(stable[j], 7u32); }
    return 0;
}
"#;

#[test]
fn issue_m2_02_fused_map_extraction_public_source_graph_and_mutation() {
    let checked = check(parse(SOURCE.as_bytes()).unwrap(), true).unwrap();
    let mir = mir::lower(&checked).unwrap();
    let extracted = loops::extract_fused(SOURCE.as_bytes(), &mir)
        .unwrap()
        .unwrap();
    let report = admission::analyze(SOURCE.as_bytes(), &mir)
        .unwrap()
        .to_json();
    let obligations = report["regions"][0]["obligations"].as_array().unwrap();
    for code in ["FUSION", "FUSION_GUARD"] {
        assert!(
            obligations.iter().any(|item| item["name"] == code),
            "{code}"
        );
    }
    let source_loop_count = mir
        .functions
        .iter()
        .flat_map(|f| &f.blocks)
        .flat_map(|b| &b.instructions)
        .filter(|i| matches!(i.kind, mir::Operation::ForNext(_)))
        .count();
    assert_eq!(source_loop_count, 2);
    assert_eq!(extracted.plan.loop_count(), 1);
    assert_eq!(
        extracted.plan.run_live(&[&[1, 2, u32::MAX]], 3).unwrap(),
        vec![28, 35, 14]
    );
    for span in extracted.source_spans {
        assert!(SOURCE[span.start..span.end].starts_with("for "));
    }
    assert_eq!(
        eval::observe(&checked, 10_000).result,
        mir_eval::observe(&mir, 10_000).result
    );

    let mut changed = extracted.graph.clone();
    changed.nodes[changed.entry].as_mut().unwrap().inputs[0] = changed.nodes.len();
    assert!(matches!(
        changed.verify(loops::GRAPH_LIMITS),
        Err(typed_net::Error::InvalidNode(_))
    ));
}

fn source_for(
    values: &[u32],
    add: u32,
    mul: u32,
    renamed: bool,
    expected_first: bool,
) -> (String, Vec<u32>) {
    let (data, first, stable, second, i, j) = if renamed {
        ("xs", "out", "middle", "finals", "row", "col")
    } else {
        ("data", "first", "stable", "second", "i", "j")
    };
    let expected: Vec<u32> = values
        .iter()
        .map(|n| n.wrapping_add(add).wrapping_mul(mul))
        .collect();
    let array = |items: &[u32]| {
        items
            .iter()
            .map(|n| format!("{n}u32"))
            .collect::<Vec<_>>()
            .join(", ")
    };
    let expected_declaration = format!("let expected: Array<u32> = [{}];", array(&expected));
    let mut source = format!(
        "fn main() -> i32 {{ let {data}: Array<u32> = [{}];",
        array(values)
    );
    if expected_first {
        source.push_str(&expected_declaration);
    }
    source.push_str(&format!(
        "var {first}: Array<u32> = filled_u32(len({data}), 0u32); \
         for {i} in 0u32 .. len({data}) {{ {first}[{i}] = wrapping_add({data}[{i}], {add}u32); }} \
         let {stable}: Array<u32> = move {first}; \
         var {second}: Array<u32> = filled_u32(len({stable}), 0u32); \
         for {j} in 0u32 .. len({stable}) {{ {second}[{j}] = wrapping_mul({stable}[{j}], {mul}u32); }}"
    ));
    if !expected_first {
        source.push_str(&expected_declaration);
    }
    for index in 0..values.len() {
        source.push_str(&format!(
            "if {second}[{index}u32] != expected[{index}u32] {{ return 1; }}"
        ));
    }
    source.push_str("return 0; }");
    (source, expected)
}

#[test]
fn issue_m2_02_fused_map_extraction_shapes_names_constants_and_reordering() {
    let cases = [
        vec![],
        vec![0],
        vec![u32::MAX, 17, 3],
        (0..7).collect(),
        (0..257).collect(),
    ];
    for (case, values) in cases.iter().enumerate() {
        let (source, expected) = source_for(
            values,
            case as u32 + 3,
            case as u32 + 7,
            case % 2 == 0,
            case % 2 == 1,
        );
        let checked = check(parse(source.as_bytes()).unwrap(), true).unwrap();
        let mir = mir::lower(&checked).unwrap();
        let reference = eval::observe(&checked, 1_000_000);
        let serial = mir_eval::observe(&mir, 1_000_000);
        assert_eq!(reference.result, Ok(eval::Value::I32(0)), "case {case}");
        assert_eq!(serial.result, reference.result, "case {case}");
        assert_eq!(serial.cleanup_trace, reference.cleanup_trace, "case {case}");
        let extracted = loops::extract_fused(source.as_bytes(), &mir)
            .unwrap()
            .unwrap();
        assert_eq!(
            extracted
                .plan
                .run_live(&[values.as_slice()], values.len())
                .unwrap(),
            expected,
            "case {case}"
        );
        assert_eq!(extracted.plan.loop_count(), 1);
        let source_loops = mir
            .functions
            .iter()
            .flat_map(|f| &f.blocks)
            .flat_map(|b| &b.instructions)
            .filter(|i| matches!(i.kind, mir::Operation::ForNext(_)))
            .count();
        assert_eq!(source_loops, 2);
    }
}

#[test]
fn issue_m2_02_fused_map_extraction_preserves_unwritten_tail() {
    let source = SOURCE
        .replace("filled_u32(len(data), 0u32)", "filled_u32(4u32, 0u32)")
        .replace(
            "return 0;",
            "if second[3u32] == 0u32 { return 0; } else { return 1; }",
        );
    let checked = check(parse(source.as_bytes()).unwrap(), true).unwrap();
    let mir = mir::lower(&checked).unwrap();
    assert_eq!(
        eval::observe(&checked, 10_000).result,
        Ok(eval::Value::I32(0))
    );
    assert_eq!(
        mir_eval::observe(&mir, 10_000).result,
        Ok(eval::Value::I32(0))
    );
    assert!(
        loops::extract_fused(source.as_bytes(), &mir)
            .unwrap()
            .is_none()
    );
}

#[test]
fn issue_m2_02_fused_map_extraction_rejects_intervening_mutation() {
    let source = SOURCE.replace(
        "let stable: Array<u32> = move first;",
        "first[0u32] = 9u32; let stable: Array<u32> = move first;",
    );
    let checked = check(parse(source.as_bytes()).unwrap(), true).unwrap();
    let program = mir::lower(&checked).unwrap();
    assert_eq!(
        eval::observe(&checked, 10_000).result,
        mir_eval::observe(&program, 10_000).result
    );
    assert!(
        loops::extract_fused(source.as_bytes(), &program)
            .unwrap()
            .is_none()
    );
}

#[test]
fn issue_m2_02_fused_map_extraction_near_matches_keep_serial_with_reasons() {
    let variants = [
        (
            SOURCE.replace("wrapping_add(data[i], 3u32)", "data[i] + 3u32"),
            "TOTALITY",
        ),
        (
            SOURCE.replace(
                "wrapping_add(data[i], 3u32)",
                "wrapping_add(first[i], 3u32)",
            ),
            "ALIAS",
        ),
        (
            SOURCE.replace(
                "wrapping_add(data[i], 3u32)",
                "wrapping_add(data[0u32], 3u32)",
            ),
            "DEPENDENCE",
        ),
        (
            format!(
                "fn step(x: u32) -> u32 {{ return x; }} {}",
                SOURCE.replace("wrapping_add(data[i], 3u32)", "step(data[i])")
            ),
            "UNKNOWN_EFFECT",
        ),
    ];
    for (source, reason) in variants {
        let checked = check(parse(source.as_bytes()).unwrap(), true).unwrap();
        let mir = mir::lower(&checked).unwrap();
        let report = admission::analyze(source.as_bytes(), &mir).unwrap();
        assert_eq!(
            report.regions()[0].first_rejected_obligation(),
            Some(reason)
        );
        assert_eq!(report.to_json()["regions"][0]["chosen_plan"], "mir-serial");
        assert!(
            loops::extract_fused(source.as_bytes(), &mir)
                .unwrap()
                .is_none()
        );
    }
}

#[test]
fn issue_m2_02_fused_map_extraction_unequal_inputs_with_true_guard() {
    let source = r#"
fn main() -> i32 {
    let left: Array<u32> = [1u32, 2u32, 3u32];
    let right: Array<u32> = [4u32, 5u32, 6u32, 7u32, 8u32];
    var tmp: Array<u32> = filled_u32(len(left), 0u32);
    for i in 0u32 .. len(left) { tmp[i] = wrapping_add(left[i], right[i]); }
    let stable: Array<u32> = move tmp;
    var output: Array<u32> = filled_u32(len(stable), 0u32);
    for j in 0u32 .. len(stable) { output[j] = wrapping_mul(stable[j], 7u32); }
    if output[0u32] != 35u32 { return 1; }
    if output[1u32] != 49u32 { return 1; }
    if output[2u32] != 63u32 { return 1; }
    return 0;
}
"#;
    let checked = check(parse(source.as_bytes()).unwrap(), true).unwrap();
    let mir = mir::lower(&checked).unwrap();
    assert_eq!(
        eval::observe(&checked, 10_000).result,
        Ok(eval::Value::I32(0))
    );
    let extracted = loops::extract_fused(source.as_bytes(), &mir)
        .unwrap()
        .unwrap();
    assert_eq!(extracted.plan.loop_count(), 1);
    assert_eq!(
        extracted
            .plan
            .run_live(&[&[1, 2, 3], &[4, 5, 6, 7, 8]], 3)
            .unwrap(),
        vec![35, 49, 63]
    );
    let mut changed = extracted.graph.clone();
    let root = changed.entry;
    let left = changed.nodes[root].as_ref().unwrap().inputs[0];
    changed.nodes[root].as_mut().unwrap().inputs[1] = left;
    assert!(matches!(
        changed.verify(loops::GRAPH_LIMITS),
        Err(typed_net::Error::SharedOrOrphan(_))
    ));
}

#[cfg(feature = "native")]
#[test]
fn issue_m2_02_fused_map_extraction_native_serial_and_trap() {
    for source in [
        SOURCE.to_string(),
        SOURCE.replace("filled_u32(len(data), 0u32)", "filled_u32(2u32, 0u32)"),
    ] {
        let checked = check(parse(source.as_bytes()).unwrap(), true).unwrap();
        let program = mir::lower(&checked).unwrap();
        let reference = eval::observe(&checked, 10_000);
        let serial = mir_eval::observe(&program, 10_000);
        let (native_status, native_json) = native::run_program(&program, 10_000).unwrap();
        assert_eq!(serial.result, reference.result);
        assert_eq!(serial.cleanup_trace, reference.cleanup_trace);
        assert_eq!(
            native_json["cleanup_trace"],
            serde_json::json!(
                serial
                    .cleanup_trace
                    .iter()
                    .map(|event| serde_json::json!({"name":event.name,"length":event.length}))
                    .collect::<Vec<_>>()
            )
        );
        if source == SOURCE {
            assert_eq!(native_status, 0);
            assert_eq!(
                native_json["result"],
                serde_json::json!({"type":"i32","value":0})
            );
            assert!(
                loops::extract_fused(source.as_bytes(), &program)
                    .unwrap()
                    .is_some()
            );
        } else {
            assert_eq!(native_status, 3);
            assert_eq!(native_json["trap"]["category"], "INDEX_OUT_OF_BOUNDS");
            assert!(
                loops::extract_fused(source.as_bytes(), &program)
                    .unwrap()
                    .is_none()
            );
        }
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

fn u32_literal(len: usize) -> String {
    (0..len)
        .map(|i| format!("{}u32", (i as u32).wrapping_mul(2_654_435_761)))
        .collect::<Vec<_>>()
        .join(", ")
}

/// Loop 1 may not run, or may run under different control flow, before loop 2.
#[test]
fn issue_m2_02_fused_map_extraction_regression_first_loop_must_execute() {
    let second = "let stable: Array<u32> = move first; \
        var second: Array<u32> = filled_u32(len(stable), 0u32); \
        for j in 0u32 .. len(stable) { second[j] = wrapping_mul(stable[j], 7u32); } \
        var h: u32 = 0u32; \
        for k in 0u32 .. len(second) { h = wrapping_add(wrapping_mul(h, 31u32), second[k]); } \
        if h == 0u32 { return 10; } if second[0u32] == 0u32 { return 20; } return 30; }";
    for len in [3, 64, 257] {
        let data = u32_literal(len);
        let prefix = format!(
            "fn main() -> i32 {{ let data: Array<u32> = [{data}]; \
             var first: Array<u32> = filled_u32(len(data), 0u32);"
        );
        let body = "first[i] = wrapping_add(data[i], 3u32);";
        let cases = [
            format!(
                "{prefix} let c: u32 = 5u32; if c == 6u32 {{ for i in 0u32 .. len(data) {{ {body} }} }} {second}"
            ),
            format!("{prefix} for i in 0u32 .. 0u32 {{ {body} }} {second}"),
            format!(
                "{prefix} var n: u32 = 0u32; while n < 0u32 {{ for i in 0u32 .. len(data) {{ {body} }} n = n + 1u32; }} {second}"
            ),
            format!(
                "{prefix} var n: u32 = 0u32; while n < 2u32 {{ for i in 0u32 .. len(data) {{ {body} }} n = n + 1u32; }} {second}"
            ),
            format!(
                "fn main() -> i32 {{ let data: Array<u32> = filled_u32({len}u32, 4294967295u32); \
                 var first: Array<u32> = filled_u32(len(data), 0u32); \
                 for i in 0u32 .. len(data) {{ {body} }} {second}"
            ),
            format!(
                "fn run(data: read Array<u32>) -> i32 {{ var first: Array<u32> = filled_u32(len(data), 0u32); \
                 for i in 0u32 .. len(data) {{ {body} }} {second} \
                 fn main() -> i32 {{ let values: Array<u32> = [{data}]; return run(values); }}"
            ),
            format!(
                "{prefix} for i in 0u32 .. len(data) {{ {body} }} first[0u32] = 9u32; {second}"
            ),
        ];
        for source in cases {
            assert_cpu_agrees_with_reference(&source);
        }
    }
}

/// Rewiring a verifier-valid graph built from public source changes live output.
#[test]
fn issue_m2_02_fused_map_extraction_regression_live_graph_rewiring() {
    let data = [1u32, 2, u32::MAX, 40, 7];
    let expected = data
        .iter()
        .map(|x| x.wrapping_add(3).wrapping_mul(7))
        .collect::<Vec<_>>();
    let mut source = SOURCE.replace(
        "[1u32, 2u32, 4294967295u32]",
        "[1u32, 2u32, 4294967295u32, 40u32, 7u32]",
    );
    let checks = expected
        .iter()
        .enumerate()
        .map(|(i, value)| format!("if second[{i}u32] != {value}u32 {{ return 1; }}"))
        .collect::<String>();
    source = source.replace("return 0;", &format!("{checks} return 0;"));
    let checked = check(parse(source.as_bytes()).unwrap(), true).unwrap();
    let program = mir::lower(&checked).unwrap();
    assert_eq!(
        eval::observe(&checked, 100_000).result,
        Ok(eval::Value::I32(0))
    );
    let fused = loops::extract_fused(source.as_bytes(), &program)
        .unwrap()
        .unwrap();
    assert_eq!(fused.plan.run_live(&[&data], data.len()).unwrap(), expected);
    assert_eq!(fused.unfused.nodes.iter().flatten().count(), 3);

    // Swap which map stage reads the input: input -> mul -> add instead of input -> add -> mul.
    let mut rewired = fused.unfused.clone();
    let consumer = rewired.entry;
    let producer = rewired.nodes[consumer].as_ref().unwrap().inputs[0];
    let input = rewired.nodes[producer].as_ref().unwrap().inputs[0];
    rewired.nodes[consumer].as_mut().unwrap().inputs = vec![input];
    rewired.nodes[producer].as_mut().unwrap().inputs = vec![consumer];
    rewired.entry = producer;
    let proof = rewired.verify(loops::GRAPH_LIMITS).unwrap();
    let plan = rewired.extract(&proof).unwrap();
    let output = plan.run_live(&[&data], data.len()).unwrap();
    assert_ne!(output, expected);
    assert_eq!(
        output,
        data.iter()
            .map(|x| x.wrapping_mul(7).wrapping_add(3))
            .collect::<Vec<_>>()
    );
    // Reordering the fused operations also changes the live result.
    let mut reordered = fused.graph.clone();
    let root = reordered.entry;
    if let Some(typed_net::Op::FusedMap(ops)) =
        reordered.nodes[root].as_mut().map(|node| &mut node.op)
    {
        ops.reverse();
    }
    let proof = reordered.verify(loops::GRAPH_LIMITS).unwrap();
    let plan = reordered.extract(&proof).unwrap();
    assert_ne!(plan.run_live(&[&data], data.len()).unwrap(), expected);
    // A stale proof from the unmodified graph is rejected.
    let original = fused.graph.verify(loops::GRAPH_LIMITS).unwrap();
    assert!(matches!(
        reordered.extract(&original),
        Err(typed_net::Error::StaleProof)
    ));
}

fn producer_publication(source: &str, workers: usize) -> (Option<String>, String) {
    let checked = check(parse(source.as_bytes()).unwrap(), true).unwrap();
    let program = mir::lower(&checked).unwrap();
    let report = admission::analyze(source.as_bytes(), &program).unwrap();
    let marker = report.to_json()["regions"][0]["execution"]["intermediate"]
        .as_str()
        .unwrap()
        .to_string();
    let run =
        invar::schedule::run_source(source.as_bytes(), &program, 50_000_000, workers).unwrap();
    let published = run
        .executed
        .first()
        .and_then(|step| step.published_array.clone());
    (published, marker)
}

/// An intermediate only the consumer reads is not written; results, cleanup
/// traces and allocation-limit outcomes still match the reference.
#[test]
fn issue_m2_02_fused_map_extraction_elides_unobservable_intermediate() {
    let checks = "var h: u32 = 0u32; for k in 0u32 .. len(second) { h = wrapping_add(wrapping_mul(h, 31u32), second[k]); } if h == 7u32 { return 1; } return 0; }";
    for len in [1usize, 64, 257] {
        let source = format!(
            "fn main() -> i32 {{ let data: Array<u32> = [{}]; \
             var first: Array<u32> = filled_u32(len(data), 0u32); \
             for i in 0u32 .. len(data) {{ first[i] = wrapping_add(data[i], 3u32); }} \
             let stable: Array<u32> = move first; \
             var second: Array<u32> = filled_u32(len(stable), 0u32); \
             for j in 0u32 .. len(stable) {{ second[j] = wrapping_mul(stable[j], 7u32); }} {checks}",
            u32_literal(len)
        );
        assert_cpu_agrees_with_reference(&source);
        for workers in [1, 4] {
            assert_eq!(
                producer_publication(&source, workers),
                (None, "elided".to_string())
            );
        }
        // Reading the intermediate after the consumer keeps it materialized.
        let observed = source.replace(
            "if h == 7u32",
            "if stable[0u32] == 7u32 { return 2; } if h == 7u32",
        );
        assert_cpu_agrees_with_reference(&observed);
        assert_eq!(
            producer_publication(&observed, 4),
            (Some("first".to_string()), "materialized".to_string())
        );
    }
    // The consumer's allocation exceeds the live-element limit after the producer ran.
    let limit = "fn main() -> i32 { let data: Array<u32> = filled_u32(6u32, 1u32); \
        var first: Array<u32> = filled_u32(len(data), 0u32); \
        for i in 0u32 .. len(data) { first[i] = wrapping_add(data[i], 3u32); } \
        let stable: Array<u32> = move first; \
        var second: Array<u32> = filled_u32(len(stable), 0u32); \
        for j in 0u32 .. len(stable) { second[j] = wrapping_mul(stable[j], 7u32); } return 0; }";
    // Leave twelve live units: data and producer fit; consumer's six do not.
    let limit = limit.replace(
        "fn main() -> i32 {",
        &format!(
            "fn main() -> i32 {{ {}",
            language_limits::padding(invar::array_limits::MAX_LIVE_ELEMENTS - 12),
        ),
    );
    let checked = check(parse(limit.as_bytes()).unwrap(), true).unwrap();
    let program = mir::lower(&checked).unwrap();
    let reference = eval::observe(&checked, 50_000_000);
    assert!(
        matches!(
            &reference.result,
            Err(eval::EvalError::Tool {
                code: "ARRAY_SIZE_LIMIT",
                ..
            })
        ),
        "{:?}",
        reference.result
    );
    for workers in [1, 4] {
        let run =
            invar::schedule::run_source(limit.as_bytes(), &program, 50_000_000, workers).unwrap();
        assert_eq!(run.observation.result, reference.result);
        assert_eq!(run.observation.steps_used, reference.steps_used);
        assert_eq!(run.observation.cleanup_trace, reference.cleanup_trace);
        assert_eq!(run.executed.len(), 1);
        assert_eq!(run.executed[0].published_array, None);
    }
}
