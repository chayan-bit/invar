#![forbid(unsafe_code)]

use invar::{check, eval, front, general_net, mir, mir_net};

use general_net::{
    Error, Limits, Net, Route,
    coverage::{self, Rule},
    oracle,
};
use invar::{
    eval::{EvalError, Value},
    front::Type,
    mir_eval,
};
use std::collections::BTreeSet;

fn checked(source: &str) -> check::CheckedProgram {
    check::check(front::parse(source.as_bytes()).unwrap(), true).unwrap()
}

fn lower(source: &str) -> Net {
    Net::lower(&checked(source), Limits::default()).unwrap()
}

fn observation(source: &str, fuel: u64) -> oracle::Observation {
    let net = lower(source);
    oracle::observe(
        &net,
        &net.verify().unwrap(),
        oracle::Limits {
            fuel,
            ..oracle::Limits::default()
        },
    )
    .unwrap()
}

fn differential(source: &str, fuel: u64) -> oracle::Observation {
    let reference = eval::observe(&checked(source), fuel);
    let actual = observation(source, fuel);
    assert_eq!(actual.semantic.result, reference.result, "{source}");
    assert_eq!(actual.semantic.steps_used, reference.steps_used, "{source}");
    assert_eq!(
        actual.semantic.cleanup_trace, reference.cleanup_trace,
        "{source}"
    );
    actual
}

#[test]
fn issue_gnet_01_checked_source_boundary_retains_all_existing_data_forms() {
    for source in [
        include_str!("../fixtures/l1_01/modules-forward.inv"),
        include_str!("../fixtures/l1_01/record-fields.inv"),
        include_str!("../fixtures/l1_01/record-moved-array.inv"),
        include_str!("../fixtures/l1_01/record-borrow-take.inv"),
        include_str!("../fixtures/l1_01/variant-match.inv"),
        include_str!("../fixtures/l1_01/variant-owned-pattern.inv"),
        include_str!("../fixtures/l1_01/result-propagation.inv"),
        include_str!("../fixtures/l1_02_views.inv"),
        "fn relay(xs: read Array<i32>) -> View<i32> from xs { let part = view(xs,0u32,1u32); return part; } fn main() -> i32 { let xs: Array<i32> = [7i32]; let part = relay(xs); return part[0u32]; }",
        include_str!("../fixtures/l1_03_generics.inv"),
        include_str!("../fixtures/l1_03_closures.inv"),
        include_str!("../fixtures/l1_04/basic.inv"),
        include_str!("../fixtures/l1_04/foreign-oracle.inv"),
    ] {
        let checked = checked(source);
        let net = Net::lower(&checked, Limits::default()).unwrap();
        let compiled = net.compiled_mir(&net.verify().unwrap()).unwrap();
        let reference = eval::observe(&checked, 10_000);
        let extracted = mir_eval::observe(&compiled, 10_000);
        assert_eq!(extracted.result, reference.result, "{source}");
        assert_eq!(extracted.steps_used, reference.steps_used);
        assert_eq!(extracted.cleanup_trace, reference.cleanup_trace);
    }
}

#[test]
fn issue_gnet_01_wire_mutations_are_rejected_or_change_real_computation() {
    let mut net = lower("fn main() -> i32 { return 10 - 3; }");
    let proof = net.verify().unwrap();
    let agent = net
        .agents
        .iter()
        .find(|agent| agent.rule == Rule::Binary)
        .unwrap();
    let site = agent.token;
    let positions: Vec<_> = net
        .graph
        .wires
        .iter()
        .enumerate()
        .filter_map(|(i, wire)| match wire.site {
            mir_net::Site::Instruction {
                function,
                block,
                index,
                ..
            } if function == site.block.function
                && block == site.block.block
                && index == site.position =>
            {
                Some(i)
            }
            _ => None,
        })
        .collect();
    assert_eq!(positions.len(), 2);
    let original = net.clone();
    net.graph.wires[positions[0]].producer = original.graph.wires[positions[1]].producer;
    net.graph.wires[positions[1]].producer = original.graph.wires[positions[0]].producer;
    assert_eq!(net.compiled_mir(&proof).unwrap_err(), Error::StaleProof);
    assert_eq!(net.verify().unwrap_err(), Error::Annotation);
    let graph_proof = net.graph.verify().unwrap();
    let rewritten =
        Net::encode(&net.graph.extract(&graph_proof).unwrap(), Limits::default()).unwrap();
    assert_eq!(
        oracle::observe(
            &rewritten,
            &rewritten.verify().unwrap(),
            oracle::Limits::default()
        )
        .unwrap()
        .semantic
        .result,
        Ok(Value::I32(-7))
    );

    let mut missing = original.clone();
    missing.graph.wires.pop();
    assert!(matches!(
        missing.verify(),
        Err(Error::Graph(mir_net::Error::MissingWire(_)))
    ));
    let mut duplicate = original.clone();
    duplicate.graph.wires.push(duplicate.graph.wires[0].clone());
    assert!(matches!(
        duplicate.verify(),
        Err(Error::Graph(mir_net::Error::DuplicateWire(_)))
    ));
    let mut bad_type = original.clone();
    bad_type.graph.wires[0].ty = Type::Bool;
    assert!(matches!(
        bad_type.verify(),
        Err(Error::Graph(mir_net::Error::Type(_)))
    ));
    let mut bad_id = original;
    bad_id.graph.wires[0].producer = usize::MAX;
    assert!(matches!(
        bad_id.verify(),
        Err(Error::Graph(mir_net::Error::Type(_)))
    ));
}

#[test]
fn issue_gnet_01_activation_effect_cleanup_and_call_annotations_cannot_be_forged() {
    let net = lower(
        "fn f() -> i32 { return 3; } fn main() -> i32 { if true { return f(); } else { return 1 / 0; } }",
    );
    let proof = net.verify().unwrap();
    let mut forged = net.clone();
    forged.agents[0].next.position += 1;
    assert_eq!(forged.verify().unwrap_err(), Error::Annotation);
    assert_eq!(forged.compiled_mir(&proof).unwrap_err(), Error::StaleProof);
    let mut forged = net.clone();
    let call = forged
        .agents
        .iter_mut()
        .find(|a| a.rule == Rule::Call)
        .unwrap();
    call.call = Some(general_net::CallTarget::Intrinsic("wrapping_add".into()));
    assert_eq!(forged.verify().unwrap_err(), Error::Annotation);
    let mut forged = net.clone();
    forged
        .agents
        .iter_mut()
        .find(|a| a.rule == Rule::Binary)
        .unwrap()
        .observation = coverage::Observation::Value;
    assert_eq!(forged.verify().unwrap_err(), Error::Annotation);
    let mut forged = net;
    let branch = forged
        .control
        .iter_mut()
        .find(|c| c.rule == Rule::Branch)
        .unwrap();
    if let Route::Branch { yes, no, .. } = &mut branch.route {
        std::mem::swap(yes, no);
    }
    assert_eq!(forged.verify().unwrap_err(), Error::Annotation);

    let mut net = lower("fn main() -> i32 { let xs: Array<i32> = [1]; return xs[0u32]; }");
    for control in &mut net.control {
        if let Route::Return { cleanup, .. } = &mut control.route {
            cleanup.clear();
        }
    }
    assert_eq!(net.verify().unwrap_err(), Error::Annotation);
}

#[test]
fn issue_gnet_01_contract_and_graph_versions_are_separate_and_checked() {
    let mut net = lower("fn main() -> i32 { return 1; }");
    let proof = net.verify().unwrap();
    net.contract_version += 1;
    assert_eq!(net.verify().unwrap_err(), Error::Version);
    assert_eq!(net.compiled_mir(&proof).unwrap_err(), Error::StaleProof);
    net.contract_version = general_net::CONTRACT_VERSION;
    net.graph.schema_version += 1;
    assert_eq!(
        net.verify().unwrap_err(),
        Error::Graph(mir_net::Error::Schema)
    );
}

#[test]
fn issue_gnet_01_structural_limits_count_record_variant_and_control_ports() {
    let source = "struct Pair { x: i32, y: i32 } fn main() -> i32 { let pair = Pair { x: 1, y: 2 }; return pair.x; }";
    let program = mir::lower(&checked(source)).unwrap();
    let net = Net::encode(&program, Limits::default()).unwrap();
    let wires = net.graph.wires.len();
    assert!(wires > 2);
    assert_eq!(
        Net::encode(
            &program,
            Limits {
                wires: wires - 1,
                ..Limits::default()
            }
        )
        .unwrap_err(),
        Error::Limit
    );
    Net::encode(
        &program,
        Limits {
            wires,
            ..Limits::default()
        },
    )
    .unwrap()
    .verify()
    .unwrap();
    for limits in [
        Limits {
            functions: 0,
            ..Limits::default()
        },
        Limits {
            blocks: 0,
            ..Limits::default()
        },
        Limits {
            instructions: 0,
            ..Limits::default()
        },
    ] {
        assert_eq!(Net::encode(&program, limits).unwrap_err(), Error::Limit);
    }
    let blob = mir::lower(&checked(
        "fn main() -> i32 { let text = \"abc\"; return 1; }",
    ))
    .unwrap();
    assert_eq!(
        Net::encode(
            &blob,
            Limits {
                literal_bytes: 2,
                ..Limits::default()
            }
        )
        .unwrap_err(),
        Error::Limit
    );
}

#[test]
fn issue_gnet_01_all_current_mir_variants_have_explicit_inventory_entries() {
    use mir::Operation as O;
    let span = front::Span { start: 0, end: 1 };
    let operations = [
        O::Tick,
        O::I32(1),
        O::U32(1),
        O::Bool(true),
        O::Blob {
            bytes: vec![1],
            text: false,
        },
        O::Read("x".into()),
        O::Move("x".into()),
        O::MoveField {
            name: "x".into(),
            index: 0,
        },
        O::ArrayRef {
            name: "x".into(),
            loan: None,
        },
        O::RecordRef {
            name: "x".into(),
            loan: None,
        },
        O::Array(vec![1]),
        O::Record {
            name: "x".into(),
            fields: vec![(0, 1)],
        },
        O::Field {
            record: 1,
            index: 0,
        },
        O::Variant {
            name: "x".into(),
            tag: 0,
            payload: Some(1),
        },
        O::VariantTag(1),
        O::VariantPayload { value: 1, tag: 0 },
        O::VariantTakePayload { value: 1, tag: 0 },
        O::DropValue(1),
        O::Unary {
            op: front::UnaryOp::Neg,
            value: 1,
            operator: span,
        },
        O::Binary {
            op: front::BinaryOp::Add,
            left: 1,
            right: 2,
            operator: span,
        },
        O::Index { array: 1, index: 2 },
        O::Length("x".into()),
        O::Call {
            name: "x".into(),
            args: vec![1],
        },
        O::Bind {
            name: "x".into(),
            value: 1,
        },
        O::Store {
            name: "x".into(),
            value: 1,
        },
        O::StoreIndex {
            name: "x".into(),
            index: 1,
            value: 2,
        },
        O::Drop(vec!["x".into()]),
        O::ForNext(1),
        O::F32(invar::numeric_contract::F32Bits::POS_ZERO),
        O::F64(invar::numeric_contract::F64Bits::POS_ZERO),
    ];
    let terms = [
        mir::Terminator::Goto {
            target: 0,
            args: vec![1],
            span,
        },
        mir::Terminator::Branch {
            condition: 1,
            yes: 0,
            no: 1,
            span,
        },
        mir::Terminator::Return {
            value: Some(1),
            cleanup: Vec::new(),
            span,
        },
    ];
    let rules: BTreeSet<_> = operations
        .iter()
        .map(|op| coverage::operation(op).0)
        .chain(terms.iter().map(|term| coverage::terminator(term).0))
        .collect();
    assert_eq!(rules, coverage::ALL_RULES.into_iter().collect());
    assert_eq!(rules.len(), coverage::ALL_RULES.len());
    assert_eq!(coverage::operands(&operations[11]), vec![1]);
    assert_eq!(coverage::operands(&operations[25]), vec![1, 2]);
    assert_eq!(
        rules.iter().filter(|rule| oracle::supports(**rule)).count(),
        32
    );
}

#[test]
fn issue_gnet_01_host_boundary_is_explicit_and_not_marked_pure() {
    let net = lower(include_str!("../fixtures/l1_04/foreign-oracle.inv"));
    let host: Vec<_> = net
        .agents
        .iter()
        .filter(|a| matches!(a.call, Some(general_net::CallTarget::Host { .. })))
        .collect();
    assert!(!host.is_empty());
    assert!(
        host.iter()
            .all(|a| a.observation == coverage::Observation::HostBoundary)
    );
    let observed =
        oracle::observe(&net, &net.verify().unwrap(), oracle::Limits::default()).unwrap();
    assert!(matches!(
        observed.semantic.result,
        Err(EvalError::Tool {
            code: "UNSUPPORTED_FEATURE",
            ..
        })
    ));
    assert_eq!(observed.semantic.steps_used, 0);
    assert_eq!(observed.failure, Some(oracle::ToolFailure::Unsupported));
    assert!(observed.events.is_empty());
}

#[test]
fn issue_gnet_01_scalar_oracle_loops_recursion_short_circuit_and_early_return() {
    for source in [
        "fn fib(n: u32) -> i32 { if n < 2u32 { return 1; } return fib(n - 1u32) + fib(n - 2u32); } fn main() -> i32 { return fib(8u32); }",
        "fn main() -> i32 { var n: i32 = 0; for i in 0u32..7u32 { n = n + 2; } while n > 10 { n = n - 1; } return n; }",
        "fn bad() -> Bool { return 1 / 0 == 0; } fn main() -> i32 { if false and bad() { return 3; } if true or bad() { return 7; } return 1 / 0; }",
        "fn done() -> Unit { return; } fn main() -> i32 { done(); if true { return 19; } return 1 / 0; }",
        "fn main() -> i32 { for i in 4294967295u32..4294967295u32 { return 1 / 0; } return 7; }",
        include_str!("../fixtures/l1_03_generics.inv"),
    ] {
        assert!(differential(source, 10_000).semantic.result.is_ok());
    }
}

#[test]
fn issue_gnet_01_scalar_oracle_integer_edges_and_ordered_traps() {
    for source in [
        "fn main() -> i32 { return -2147483648 % -1; }",
        "fn main() -> i32 { return -2147483648 / -1; }",
        "fn main() -> i32 { return 2147483647 + 1; }",
        "fn main() -> i32 { let x: i32 = -2147483648; return -x; }",
        "fn main() -> i32 { let x: u32 = 0u32 - 1u32; return 3; }",
        "fn main() -> i32 { let x: u32 = wrapping_mul(4294967295u32, 3u32); if x == 4294967293u32 { return 1; } return 0; }",
        "fn f() -> i32 { return 1 / 0; } fn g() -> i32 { return 2147483647 + 1; } fn main() -> i32 { return f() + g(); }",
        "fn main() -> i32 { return -7 / 3 + -7 % 3; }",
        "fn main() -> i32 { let x = 7u32 % 0u32; return 0; }",
    ] {
        differential(source, 1000);
    }
    let observed = observation("fn main() -> i32 { return 1 / 0; }", 1000);
    assert_eq!(observed.events.last().unwrap().rule, Rule::Binary);
    assert_eq!(
        observed.events.last().unwrap().kind,
        oracle::EventKind::Demand
    );
}

#[test]
fn issue_gnet_01_source_fuel_is_independent_of_physical_transitions() {
    let source = "fn f(x: i32) -> i32 { return x + 1; } fn main() -> i32 { var n = 0; for i in 0u32..3u32 { n = f(n); } return n; }";
    let complete = differential(source, 1000);
    assert!(complete.transitions > complete.semantic.steps_used);
    for fuel in 1..=complete.semantic.steps_used + 1 {
        differential(source, fuel);
    }
    let zero = observation(source, 0);
    assert!(matches!(
        zero.semantic.result,
        Err(EvalError::Tool {
            code: "INVALID_FUEL",
            ..
        })
    ));
}

#[test]
fn issue_gnet_01_divergent_oldest_call_never_demands_later_trap() {
    let source = "fn f() -> i32 { while true {} return 0; } fn g() -> i32 { return 1 / 0; } fn main() -> i32 { let a = f(); let b = g(); return a + b; }";
    let net = lower(source);
    let result = differential(source, 80);
    assert_eq!(
        result.semantic.result,
        Err(EvalError::EvaluationLimit {
            fuel_limit: 80,
            steps_used: 80
        })
    );
    let program = net.compiled_mir(&net.verify().unwrap()).unwrap();
    let g = program
        .functions
        .iter()
        .position(|f| f.name == "g")
        .unwrap();
    assert!(
        result
            .events
            .iter()
            .all(|event| event.token.block.function != g)
    );
    assert!(result.events.iter().any(|event| event.activation > 1));
}

#[test]
fn issue_gnet_01_tool_exhaustion_is_not_a_source_trap_or_divergence() {
    let net = lower("fn main() -> i32 { while true {} return 0; }");
    for (limits, expected) in [
        (
            oracle::Limits {
                transitions: 3,
                ..oracle::Limits::default()
            },
            oracle::ToolFailure::TransitionLimit,
        ),
        (
            oracle::Limits {
                trace_events: 1,
                ..oracle::Limits::default()
            },
            oracle::ToolFailure::TraceLimit,
        ),
    ] {
        let result = oracle::observe(&net, &net.verify().unwrap(), limits).unwrap();
        assert!(matches!(
            result.semantic.result,
            Err(EvalError::Tool {
                code: "RESOURCE_LIMIT",
                ..
            })
        ));
        assert_eq!(result.failure, Some(expected));
        assert!(result.semantic.steps_used < limits.fuel);
    }
}

#[test]
fn issue_gnet_01_call_depth_uses_bounded_heap_frames_and_preserves_following_runs() {
    let source = "fn f(n: u32) -> i32 { if n == 0u32 { return 7; } return f(n - 1u32); } fn main() -> i32 { return f(63u32); }";
    let result = observation(source, 10_000);
    assert!(matches!(
        result.semantic.result,
        Err(EvalError::Tool {
            code: "CALL_DEPTH_LIMIT",
            ..
        })
    ));
    assert_eq!(
        observation(&source.replace("f(63u32)", "f(62u32)"), 10_000)
            .semantic
            .result,
        Ok(Value::I32(7))
    );
    assert_eq!(
        observation("fn main() -> i32 { return 5; }", 100)
            .semantic
            .result,
        Ok(Value::I32(5))
    );
}

#[test]
fn issue_gnet_01_admitted_aggregates_and_remaining_gaps_are_explicit() {
    for source in [
        "struct Pair { x: i32 } fn unused() -> i32 { let pair = Pair { x: 1 }; return pair.x; } fn main() -> i32 { return 7; }",
        include_str!("../fixtures/l1_01/variant-match.inv"),
    ] {
        assert!(differential(source, 1000).semantic.result.is_ok());
    }
    // GNET-02 closed the view and byte/UTF-8 intrinsic gaps.
    for source in [
        include_str!("../fixtures/l1_02_views.inv"),
        "fn unused() -> i32 { let b: Bytes = b\"abc\"; let sliced = bytes_slice(b, 0u32, 1u32); return 0; } fn main() -> i32 { return 7; }",
    ] {
        assert!(differential(source, 1000).semantic.result.is_ok());
    }
    // Declared foreign calls remain explicit host boundaries.
    let result = observation(include_str!("../fixtures/l1_04/foreign-oracle.inv"), 1000);
    assert!(matches!(
        result.semantic.result,
        Err(EvalError::Tool {
            code: "UNSUPPORTED_FEATURE",
            ..
        })
    ));
    assert_eq!(result.semantic.steps_used, 0);
    assert_eq!(result.failure, Some(oracle::ToolFailure::Unsupported));
    assert!(result.events.is_empty());
}

#[test]
fn issue_gnet_01_bounded_generated_scalar_compositions_match_source_reference() {
    let mut seed = 0x67ab91u32;
    for _ in 0..64 {
        seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
        let n = seed % 9;
        let bias = (seed >> 8) % 7;
        let source = format!(
            "fn f(n: u32) -> i32 {{ if n == 0u32 {{ return {bias}; }} return f(n - 1u32) + 2; }} fn main() -> i32 {{ var acc = 0; for i in 0u32..{n}u32 {{ if i % 2u32 == 0u32 {{ acc = acc + f(i); }} else {{ acc = acc - f(i); }} }} return acc; }}"
        );
        differential(&source, 10_000);
    }
}

#[test]
fn issue_gnet_01_logical_complete_follows_nested_calls_and_only_selected_branches() {
    let result = differential(
        "fn f() -> i32 { return 3; } fn main() -> i32 { if true { return f(); } else { return 1 / 0; } }",
        1000,
    );
    let mut demanded = Vec::new();
    for event in &result.events {
        match event.kind {
            oracle::EventKind::Demand => {
                demanded.push((event.frame, event.activation, event.token, event.rule))
            }
            oracle::EventKind::Complete => assert_eq!(
                demanded.pop(),
                Some((event.frame, event.activation, event.token, event.rule))
            ),
        }
    }
    assert!(demanded.is_empty());
    assert!(!result.events.iter().any(|event| event.rule == Rule::Binary));
}

#[cfg(feature = "native")]
#[test]
fn issue_gnet_01_checked_compiler_boundary_native_cpu_matches_scalar_oracle() {
    let source = "fn f(n: u32) -> i32 { if n == 0u32 { return 3; } return f(n - 1u32) + 1; } fn main() -> i32 { return f(7u32); }";
    let net = lower(source);
    let program = net.compiled_mir(&net.verify().unwrap()).unwrap();
    let observed = differential(source, 1000);
    let (status, document) = invar::native::run_program(&program, 1000).unwrap();
    assert_eq!(status, 0);
    assert_eq!(document["result"]["value"], 10);
    assert_eq!(document["steps_used"], observed.semantic.steps_used);
}
