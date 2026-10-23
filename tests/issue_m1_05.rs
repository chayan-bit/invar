use std::collections::BTreeMap;

#[path = "../test-support/term_oracle.rs"]
mod term;

use invar::{
    check::check,
    eval,
    front::{self, BinaryOp},
    mir, mir_eval, mir_net,
    typed_net::{self, Effect, Error, Limits, MapOp, Net, Node, Op, Type},
};

use term::{Kind, Term};

#[test]
fn issue_m1_05_rewrite_extraction_equivalence_public_mir_graph_differential() {
    let source = "fn main() -> i32 { return 10 - 3; }";
    let checked = check(front::parse(source.as_bytes()).unwrap(), true).unwrap();
    let program = mir::lower(&checked).unwrap();
    let limits = mir_net::Limits {
        functions: 8,
        blocks: 64,
        instructions: 512,
        wires: 1024,
    };
    let mut net = mir_net::Net::encode(&program, limits).unwrap();
    let proof = net.verify().unwrap();
    let extracted = net.extract(&proof).unwrap();
    let reference = eval::observe(&checked, 1000);
    let direct = mir_eval::observe(&program, 1000);
    let through_graph = mir_eval::observe(&extracted, 1000);
    assert_eq!(reference.result, direct.result);
    assert_eq!(reference.result, through_graph.result);
    assert_eq!(through_graph.steps_used, reference.steps_used);
    assert_eq!(through_graph.cleanup_trace, reference.cleanup_trace);

    // Swap the real operands of subtraction. The modified graph still verifies
    // but extraction now computes the opposite subtraction.
    let instruction = program.functions[0].blocks[0]
        .instructions
        .iter()
        .position(|i| {
            matches!(
                i.kind,
                mir::Operation::Binary {
                    op: BinaryOp::Sub,
                    ..
                }
            )
        })
        .unwrap();
    let left = net
        .wires
        .iter()
        .position(|wire| {
            wire.site
                == mir_net::Site::Instruction {
                    function: 0,
                    block: 0,
                    index: instruction,
                    port: 0,
                }
        })
        .unwrap();
    let right = net
        .wires
        .iter()
        .position(|wire| {
            wire.site
                == mir_net::Site::Instruction {
                    function: 0,
                    block: 0,
                    index: instruction,
                    port: 1,
                }
        })
        .unwrap();
    let old = net.wires[left].producer;
    net.wires[left].producer = net.wires[right].producer;
    net.wires[right].producer = old;
    assert_eq!(net.extract(&proof).unwrap_err(), mir_net::Error::StaleProof);
    let mutated = net.extract(&net.verify().unwrap()).unwrap();
    assert_eq!(
        mir_eval::observe(&mutated, 1000).result,
        Ok(eval::Value::I32(-7))
    );
    assert_ne!(mir_eval::observe(&mutated, 1000).result, reference.result);
}

#[cfg(feature = "native")]
#[test]
fn issue_m1_05_rewrite_extraction_equivalence_native_cpu() {
    use invar::native;
    for source in [
        "fn main() -> i32 { return 10 - 3; }",
        "fn main() -> i32 { if true { return 8; } else { return 1 / 0; } }",
        "fn main() -> i32 { let xs: Array<i32> = [4]; return xs[0u32]; }",
    ] {
        let checked = check(front::parse(source.as_bytes()).unwrap(), true).unwrap();
        let direct = mir::lower(&checked).unwrap();
        let net = mir_net::Net::encode(
            &direct,
            mir_net::Limits {
                functions: 8,
                blocks: 64,
                instructions: 512,
                wires: 1024,
            },
        )
        .unwrap();
        let extracted = net.extract(&net.verify().unwrap()).unwrap();
        let (direct_status, direct_json) = native::run_program(&direct, 1000).unwrap();
        let (graph_status, graph_json) = native::run_program(&extracted, 1000).unwrap();
        assert_eq!(graph_status, direct_status);
        assert_eq!(graph_json, direct_json);
        let source_result = eval::observe(&checked, 1000);
        assert_eq!(
            mir_eval::observe(&extracted, 1000).result,
            source_result.result
        );
        if source.contains("10 - 3") {
            let mut changed = net.clone();
            let index = direct.functions[0].blocks[0]
                .instructions
                .iter()
                .position(|instruction| {
                    matches!(
                        instruction.kind,
                        mir::Operation::Binary {
                            op: BinaryOp::Sub,
                            ..
                        }
                    )
                })
                .unwrap();
            let left = changed
                .wires
                .iter()
                .position(|wire| {
                    wire.site
                        == mir_net::Site::Instruction {
                            function: 0,
                            block: 0,
                            index,
                            port: 0,
                        }
                })
                .unwrap();
            let right = changed
                .wires
                .iter()
                .position(|wire| {
                    wire.site
                        == mir_net::Site::Instruction {
                            function: 0,
                            block: 0,
                            index,
                            port: 1,
                        }
                })
                .unwrap();
            let old = changed.wires[left].producer;
            changed.wires[left].producer = changed.wires[right].producer;
            changed.wires[right].producer = old;
            let mutated = changed.extract(&changed.verify().unwrap()).unwrap();
            let (_, output) = native::run_program(&mutated, 1000).unwrap();
            assert_eq!(output["result"]["value"], -7);
        }
    }
}

const LIMITS: Limits = Limits {
    nodes: 64,
    ports: 128,
    depth: 16,
    array_len: 64,
    map_ops: 4,
};
const RUN: term::Limits = term::Limits {
    nodes: 64,
    depth: 16,
    steps: 32,
};

fn t(id: u32, kind: Kind) -> Term {
    Term { id, kind }
}

fn maps(input: Vec<u32>) -> Net {
    let mut net = Net::empty(0);
    let source = net.push(Node {
        source_id: 1,
        ty: Type::ArrayU32,
        effect: Effect::Total,
        inputs: vec![],
        op: Op::ArrayU32(input),
    });
    let first = net.push(Node {
        source_id: 2,
        ty: Type::ArrayU32,
        effect: Effect::Total,
        inputs: vec![source],
        op: Op::Map(MapOp::WrappingAdd(3)),
    });
    net.entry = net.push(Node {
        source_id: 3,
        ty: Type::ArrayU32,
        effect: Effect::Total,
        inputs: vec![first],
        op: Op::Map(MapOp::WrappingMul(7)),
    });
    net.seal_source();
    net
}

#[test]
fn issue_m1_05_rewrite_extraction_equivalence_beta_changes_extracted_body() {
    let source = t(
        0,
        Kind::Apply(
            Box::new(t(
                1,
                Kind::Lambda {
                    binder: 9,
                    body: Box::new(t(2, Kind::Var(9))),
                },
            )),
            Box::new(t(3, Kind::Nat(41))),
        ),
    );
    let types = [
        (0, Type::Nat),
        (1, Type::Function(Box::new(Type::Nat), Box::new(Type::Nat))),
        (2, Type::Nat),
        (3, Type::Nat),
    ]
    .into();
    let mut net = typed_net::encode(
        &term::Program::default(),
        &source,
        &types,
        &BTreeMap::new(),
        LIMITS,
    )
    .unwrap();
    let old_proof = net.verify(LIMITS).unwrap();
    let direct = net.extract(&old_proof).unwrap();
    assert!(matches!(
        &direct,
        typed_net::Plan::Term {
            entry: Term {
                kind: Kind::Apply(..),
                ..
            },
            ..
        }
    ));
    let change = net.rewrite_beta(LIMITS).unwrap().unwrap();
    assert_eq!(change.rule, typed_net::BETA_RULE);
    assert_eq!(change.before, 0);
    assert_eq!(change.after, 1);
    assert_eq!(net.extract(&old_proof).unwrap_err(), Error::StaleProof);
    let proof = net.verify(LIMITS).unwrap();
    let extracted = net.extract(&proof).unwrap();
    assert!(matches!(
        &extracted,
        typed_net::Plan::Term {
            entry: Term {
                kind: Kind::Let { .. },
                ..
            },
            ..
        }
    ));
    assert_eq!(
        term::run_term(&direct, RUN).unwrap().result,
        term::run_term(&extracted, RUN).unwrap().result
    );
    assert_eq!(
        term::run_term(&extracted, RUN).unwrap().result,
        Ok(term::Value::Nat(41))
    );
    net.rule_history[0] = "stale-beta-v0";
    assert_eq!(net.verify(LIMITS), Err(Error::RuleRejected));

    // Capturing an outer affine value would move it when the closure is made;
    // the closed-lambda beta rule therefore declines this shape.
    let captured = t(
        10,
        Kind::Let {
            binder: 20,
            value: Box::new(t(
                11,
                Kind::Lambda {
                    binder: 21,
                    body: Box::new(t(12, Kind::Var(21))),
                },
            )),
            body: Box::new(t(
                13,
                Kind::Apply(
                    Box::new(t(
                        14,
                        Kind::Lambda {
                            binder: 22,
                            body: Box::new(t(15, Kind::Var(20))),
                        },
                    )),
                    Box::new(t(16, Kind::Nat(0))),
                ),
            )),
        },
    );
    let f = Type::Function(Box::new(Type::Nat), Box::new(Type::Nat));
    let g = Type::Function(Box::new(Type::Nat), Box::new(f.clone()));
    let types = [
        (10, f.clone()),
        (11, f.clone()),
        (12, Type::Nat),
        (13, f.clone()),
        (14, g),
        (15, f),
        (16, Type::Nat),
    ]
    .into();
    let mut net = typed_net::encode(
        &term::Program::default(),
        &captured,
        &types,
        &BTreeMap::new(),
        LIMITS,
    )
    .unwrap();
    assert_eq!(net.rewrite_beta(LIMITS).unwrap(), None);

    let trapping = t(
        30,
        Kind::Apply(
            Box::new(t(
                31,
                Kind::Lambda {
                    binder: 32,
                    body: Box::new(t(33, Kind::Nat(1))),
                },
            )),
            Box::new(t(34, Kind::Fail("arg"))),
        ),
    );
    let types = [
        (30, Type::Nat),
        (31, Type::Function(Box::new(Type::Nat), Box::new(Type::Nat))),
        (33, Type::Nat),
        (34, Type::Nat),
    ]
    .into();
    let mut net = typed_net::encode(
        &term::Program::default(),
        &trapping,
        &types,
        &BTreeMap::new(),
        LIMITS,
    )
    .unwrap();
    assert_eq!(net.rewrite_beta(LIMITS).unwrap(), None);
    let original = net.extract(&net.verify(LIMITS).unwrap()).unwrap();
    assert_eq!(
        term::run_term(&original, RUN).unwrap().result,
        Err(term::Error::Trap {
            at: 34,
            label: "arg"
        })
    );

    // A beta redex inside a returned closure stays intact in v1, since the
    // oracle exposes closure bodies in its internal value observations.
    let outer = t(
        40,
        Kind::Lambda {
            binder: 50,
            body: Box::new(source),
        },
    );
    let types = [
        (40, Type::Function(Box::new(Type::Nat), Box::new(Type::Nat))),
        (0, Type::Nat),
        (1, Type::Function(Box::new(Type::Nat), Box::new(Type::Nat))),
        (2, Type::Nat),
        (3, Type::Nat),
    ]
    .into();
    let mut net = typed_net::encode(
        &term::Program::default(),
        &outer,
        &types,
        &BTreeMap::new(),
        LIMITS,
    )
    .unwrap();
    assert_eq!(net.rewrite_beta(LIMITS).unwrap(), None);
    let proof = net.verify(LIMITS).unwrap();
    assert_eq!(
        term::observe_net(&net, &proof, RUN).unwrap(),
        term::observe(&term::Program::default(), &outer, RUN)
    );
}

#[test]
fn issue_m1_05_rewrite_extraction_equivalence_fusion_changes_loop_count() {
    for input in [vec![], vec![0], vec![1, u32::MAX, 42]] {
        let mut net = maps(input.clone());
        let proof = net.verify(LIMITS).unwrap();
        let direct = net.extract(&proof).unwrap();
        assert_eq!(direct.loop_count(), 2);
        let expected: Vec<u32> = input
            .iter()
            .map(|v| v.wrapping_add(3).wrapping_mul(7))
            .collect();
        assert_eq!(direct.run_array().unwrap(), expected);
        assert_eq!(
            term::observe_array_net(&net, &proof, RUN).unwrap().result,
            Ok(expected.clone())
        );
        let rewrite = net.rewrite_map_fusion(LIMITS).unwrap().unwrap();
        assert_eq!(rewrite.rule, typed_net::MAP_FUSION_RULE);
        assert_eq!(net.extract(&proof).unwrap_err(), Error::StaleProof);
        let proof = net.verify(LIMITS).unwrap();
        let extracted = net.extract(&proof).unwrap();
        assert_eq!(extracted.loop_count(), 1);
        assert_eq!(extracted.run_array().unwrap(), expected);
        assert_eq!(
            term::observe_array_net(&net, &proof, RUN).unwrap().result,
            Ok(expected)
        );
    }
}

#[test]
fn issue_m1_05_rewrite_extraction_equivalence_invalid_and_fallback() {
    let mut net = maps(vec![2, 4]);
    let original = net.extract(&net.verify(LIMITS).unwrap()).unwrap();
    let too_small = Limits {
        map_ops: 1,
        ..LIMITS
    };
    assert_eq!(net.rewrite_map_fusion(too_small).unwrap(), None);
    assert_eq!(net.revision, 0);
    assert_eq!(net.extract(&net.verify(LIMITS).unwrap()).unwrap(), original);

    // A content mutation is visible to the proof, and once reverified it
    // changes the actual extracted result rather than being an unused label.
    let proof = net.verify(LIMITS).unwrap();
    net.nodes[0].as_mut().unwrap().op = Op::ArrayU32(vec![8, 9]);
    assert_eq!(net.extract(&proof).unwrap_err(), Error::StaleProof);
    let changed = net.extract(&net.verify(LIMITS).unwrap()).unwrap();
    assert_ne!(changed.run_array().unwrap(), original.run_array().unwrap());
    net.nodes[0].as_mut().unwrap().effect = Effect::Unknown;
    assert_eq!(net.verify(LIMITS), Err(Error::Effect(0)));
}
