//! Issue label M1-04.
use std::collections::BTreeMap;

use crate::term;

use invar::{
    check::check,
    front::parse,
    mir, mir_eval, mir_net,
    typed_net::{self, Effect, Error, Limits, Signature, Type},
};

use term::{Event, Kind, Term};

const GRAPH_LIMITS: Limits = Limits {
    nodes: 64,
    ports: 128,
    depth: 16,
    array_len: 64,
    map_ops: 4,
};
const RUN_LIMITS: term::Limits = term::Limits {
    nodes: 64,
    depth: 16,
    steps: 32,
};

fn t(id: u32, kind: Kind) -> Term {
    Term { id, kind }
}

#[test]
fn typed_net_verifier_public_mir_ports_and_ownership() {
    let source = "fn main() -> i32 { let xs: Array<i32> = [4]; return xs[0u32]; }";
    let checked = check(parse(source.as_bytes()).unwrap(), true).unwrap();
    let program = mir::lower(&checked).unwrap();
    let limits = mir_net::Limits {
        functions: 8,
        blocks: 64,
        instructions: 512,
        wires: 1024,
    };
    let net = mir_net::Net::encode(&program, limits).unwrap();
    let proof = net.verify().unwrap();
    let extracted = net.extract(&proof).unwrap();
    let direct = mir_eval::observe(&program, 1000);
    let extracted_result = mir_eval::observe(&extracted, 1000);
    assert_eq!(direct.result, extracted_result.result);
    assert_eq!(direct.steps_used, extracted_result.steps_used);
    assert_eq!(direct.cleanup_trace, extracted_result.cleanup_trace);
    assert_eq!(
        mir_eval::observe(&extracted, 1000).result,
        Ok(invar::eval::Value::I32(4))
    );
    assert!(matches!(
        mir_net::Net::encode(&program, mir_net::Limits { wires: 1, ..limits }),
        Err(mir_net::Error::Limit)
    ));

    let mut mutated = net.clone();
    mutated.wires.pop();
    assert!(matches!(
        mutated.verify(),
        Err(mir_net::Error::MissingWire(_))
    ));
    let mut mutated = net.clone();
    mutated.wires.push(mutated.wires[0].clone());
    assert!(matches!(
        mutated.verify(),
        Err(mir_net::Error::DuplicateWire(_))
    ));
    let mut mutated = net.clone();
    mutated.wires[0].ty = invar::front::Type::Bool;
    assert!(matches!(mutated.verify(), Err(mir_net::Error::Type(_))));
    let mut mutated = net;
    mutated.schema_version += 1;
    assert_eq!(mutated.verify(), Err(mir_net::Error::Schema));

    let mut bad_cleanup = program.clone();
    for block in &mut bad_cleanup.functions[0].blocks {
        if let Some(mir::Terminator::Return { cleanup, .. }) = &mut block.end {
            cleanup.clear();
        }
    }
    assert!(matches!(
        mir_net::Net::encode(&bad_cleanup, limits),
        Err(mir_net::Error::Mir(_))
    ));
}

fn call(id: u32, name: &str, args: Vec<Term>) -> Term {
    t(
        id,
        Kind::Call {
            name: name.into(),
            args,
        },
    )
}

fn spin() -> (term::Program, BTreeMap<String, Signature>) {
    (
        term::Program {
            functions: [(
                "spin".into(),
                term::Function {
                    params: vec![],
                    body: call(90, "spin", vec![]),
                },
            )]
            .into(),
        },
        [(
            "spin".into(),
            Signature {
                params: vec![],
                result: Type::Nat,
            },
        )]
        .into(),
    )
}

fn annotations(items: &[(u32, Type)]) -> BTreeMap<u32, Type> {
    items.iter().cloned().collect()
}

#[test]
fn typed_net_verifier_cbv_demand_and_failure_provenance() {
    let (program, signatures) = spin();
    let discarded = t(
        0,
        Kind::Apply(
            Box::new(t(
                1,
                Kind::Lambda {
                    binder: 10,
                    body: Box::new(t(2, Kind::Nat(7))),
                },
            )),
            Box::new(call(3, "spin", vec![])),
        ),
    );
    let types = annotations(&[
        (0, Type::Nat),
        (1, Type::Function(Box::new(Type::Nat), Box::new(Type::Nat))),
        (2, Type::Nat),
        (3, Type::Nat),
        (90, Type::Nat),
    ]);
    let graph = typed_net::encode(&program, &discarded, &types, &signatures, GRAPH_LIMITS).unwrap();
    let proof = graph.verify(GRAPH_LIMITS).unwrap();
    assert_eq!(
        graph.decode(&proof).unwrap(),
        (program.clone(), discarded.clone())
    );
    let bounded = term::Limits {
        steps: 7,
        ..RUN_LIMITS
    };
    assert_eq!(
        term::observe_net(&graph, &proof, bounded).unwrap(),
        term::observe(&program, &discarded, bounded)
    );
    assert_eq!(
        term::observe_net(&graph, &proof, bounded).unwrap().result,
        Err(term::Error::StepLimit(90))
    );

    let branch = t(
        4,
        Kind::If {
            condition: Box::new(t(5, Kind::Bool(true))),
            yes: Box::new(t(6, Kind::Nat(5))),
            no: Box::new(call(7, "spin", vec![])),
        },
    );
    let types = annotations(&[
        (4, Type::Nat),
        (5, Type::Bool),
        (6, Type::Nat),
        (7, Type::Nat),
        (90, Type::Nat),
    ]);
    let graph = typed_net::encode(&program, &branch, &types, &signatures, GRAPH_LIMITS).unwrap();
    let proof = graph.verify(GRAPH_LIMITS).unwrap();
    let observed = term::observe_net(&graph, &proof, RUN_LIMITS).unwrap();
    assert_eq!(observed, term::observe(&program, &branch, RUN_LIMITS));
    assert!(!observed.events.contains(&Event::Demand(7)));

    let first = t(
        8,
        Kind::Apply(
            Box::new(t(9, Kind::Fail("function"))),
            Box::new(t(10, Kind::Fail("argument"))),
        ),
    );
    let types = annotations(&[
        (8, Type::Nat),
        (9, Type::Function(Box::new(Type::Nat), Box::new(Type::Nat))),
        (10, Type::Nat),
    ]);
    let empty = term::Program::default();
    let graph = typed_net::encode(&empty, &first, &types, &BTreeMap::new(), GRAPH_LIMITS).unwrap();
    let proof = graph.verify(GRAPH_LIMITS).unwrap();
    let observed = term::observe_net(&graph, &proof, RUN_LIMITS).unwrap();
    assert_eq!(observed, term::observe(&empty, &first, RUN_LIMITS));
    assert_eq!(
        observed.result,
        Err(term::Error::Trap {
            at: 9,
            label: "function"
        })
    );
    assert!(!observed.events.contains(&Event::Demand(10)));

    let named = call(
        11,
        "f",
        vec![
            t(12, Kind::Unit),
            t(13, Kind::Fail("second")),
            t(14, Kind::Fail("third")),
        ],
    );
    let named_program = term::Program {
        functions: [(
            "f".into(),
            term::Function {
                params: vec![31, 32, 33],
                body: t(91, Kind::Nat(0)),
            },
        )]
        .into(),
    };
    let named_signatures = [(
        "f".into(),
        Signature {
            params: vec![(31, Type::Unit), (32, Type::Nat), (33, Type::Nat)],
            result: Type::Nat,
        },
    )]
    .into();
    let types = annotations(&[
        (11, Type::Nat),
        (12, Type::Unit),
        (13, Type::Nat),
        (14, Type::Nat),
        (91, Type::Nat),
    ]);
    let graph = typed_net::encode(
        &named_program,
        &named,
        &types,
        &named_signatures,
        GRAPH_LIMITS,
    )
    .unwrap();
    let observed =
        term::observe_net(&graph, &graph.verify(GRAPH_LIMITS).unwrap(), RUN_LIMITS).unwrap();
    assert_eq!(observed, term::observe(&named_program, &named, RUN_LIMITS));
    assert_eq!(
        observed.result,
        Err(term::Error::Trap {
            at: 13,
            label: "second"
        })
    );
    assert!(!observed.events.contains(&Event::Demand(14)));

    let wrong_arity = call(20, "spin", vec![t(21, Kind::Nat(1))]);
    let types = annotations(&[(20, Type::Nat), (21, Type::Nat), (90, Type::Nat)]);
    assert!(matches!(
        typed_net::encode(&program, &wrong_arity, &types, &signatures, GRAPH_LIMITS),
        Err(Error::Type(_))
    ));
}

#[test]
fn typed_net_verifier_rejects_types_affine_duplication_and_stale_facts() {
    let pair = t(
        0,
        Kind::Pair(Box::new(t(1, Kind::Nat(2))), Box::new(t(2, Kind::Nat(9)))),
    );
    let pair_type = Type::Pair(Box::new(Type::Nat), Box::new(Type::Nat));
    let types = annotations(&[(0, pair_type), (1, Type::Nat), (2, Type::Nat)]);
    let mut graph = typed_net::encode(
        &term::Program::default(),
        &pair,
        &types,
        &BTreeMap::new(),
        GRAPH_LIMITS,
    )
    .unwrap();
    let proof = graph.verify(GRAPH_LIMITS).unwrap();
    let before = term::observe_net(&graph, &proof, RUN_LIMITS)
        .unwrap()
        .result;
    graph.nodes[graph.entry].as_mut().unwrap().inputs.swap(0, 1);
    assert_eq!(graph.check_proof(&proof), Err(Error::StaleProof));
    let after_proof = graph.verify(GRAPH_LIMITS).unwrap();
    assert_ne!(
        term::observe_net(&graph, &after_proof, RUN_LIMITS)
            .unwrap()
            .result,
        before
    );
    graph.nodes[graph.entry].as_mut().unwrap().inputs[0] =
        graph.nodes[graph.entry].as_ref().unwrap().inputs[1];
    assert!(matches!(
        graph.verify(GRAPH_LIMITS),
        Err(Error::SharedOrOrphan(_))
    ));

    let duplicate = t(
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
                Kind::Pair(
                    Box::new(t(14, Kind::Var(20))),
                    Box::new(t(15, Kind::Var(20))),
                ),
            )),
        },
    );
    let function_type = Type::Function(Box::new(Type::Nat), Box::new(Type::Nat));
    let types = annotations(&[
        (
            10,
            Type::Pair(
                Box::new(function_type.clone()),
                Box::new(function_type.clone()),
            ),
        ),
        (11, function_type.clone()),
        (12, Type::Nat),
        (
            13,
            Type::Pair(
                Box::new(function_type.clone()),
                Box::new(function_type.clone()),
            ),
        ),
        (14, function_type.clone()),
        (15, function_type),
    ]);
    assert_eq!(
        typed_net::encode(
            &term::Program::default(),
            &duplicate,
            &types,
            &BTreeMap::new(),
            GRAPH_LIMITS
        )
        .unwrap_err(),
        Error::DuplicatedOwner(20)
    );

    let mut graph = typed_net::encode(
        &term::Program::default(),
        &pair,
        &annotations(&[
            (0, Type::Pair(Box::new(Type::Nat), Box::new(Type::Nat))),
            (1, Type::Nat),
            (2, Type::Nat),
        ]),
        &BTreeMap::new(),
        GRAPH_LIMITS,
    )
    .unwrap();
    graph.schema_version += 1;
    assert_eq!(graph.verify(GRAPH_LIMITS), Err(Error::Schema));
    graph.schema_version = typed_net::SCHEMA_VERSION;
    graph.nodes[0].as_mut().unwrap().effect = Effect::MayFail;
    assert_eq!(graph.verify(GRAPH_LIMITS), Err(Error::Effect(0)));
}

mod term_oracle_self_tests {
    use super::term::{Error, Event, Function, Kind, Limits, Program, Term, Value, observe};

    const LIMITS: Limits = Limits {
        nodes: 64,
        depth: 16,
        steps: 32,
    };

    fn t(id: u32, kind: Kind) -> Term {
        Term { id, kind }
    }

    fn call(id: u32, name: &str, args: Vec<Term>) -> Term {
        t(
            id,
            Kind::Call {
                name: name.into(),
                args,
            },
        )
    }

    fn spin() -> Program {
        Program {
            functions: [(
                "spin".into(),
                Function {
                    params: vec![],
                    body: call(90, "spin", vec![]),
                },
            )]
            .into(),
        }
    }

    #[test]
    fn typed_net_verifier_term_oracle_preserves_demand_and_provenance() {
        // A discarded argument remains demanded under call-by-value.
        let discarded = t(
            0,
            Kind::Apply(
                Box::new(t(
                    1,
                    Kind::Lambda {
                        binder: 10,
                        body: Box::new(t(2, Kind::Nat(7))),
                    },
                )),
                Box::new(call(3, "spin", vec![])),
            ),
        );
        let observation = observe(&spin(), &discarded, Limits { steps: 7, ..LIMITS });
        assert_eq!(observation.result, Err(Error::StepLimit(90)));
        assert_eq!(
            &observation.events[..4],
            &[
                Event::Demand(0),
                Event::Demand(1),
                Event::Complete(1),
                Event::Demand(3),
            ]
        );
        assert!(!observation.events.contains(&Event::Demand(2)));

        // The unselected branch is never demanded, even if divergent.
        let branch = t(
            4,
            Kind::If {
                condition: Box::new(t(5, Kind::Bool(true))),
                yes: Box::new(t(6, Kind::Nat(5))),
                no: Box::new(call(7, "spin", vec![])),
            },
        );
        let observation = observe(&spin(), &branch, LIMITS);
        assert_eq!(observation.result, Ok(Value::Nat(5)));
        assert_eq!(observation.steps, 3);
        assert!(!observation.events.contains(&Event::Demand(7)));
        assert!(!observation.events.contains(&Event::Demand(90)));

        // Function position is evaluated before the argument, including traps.
        let first = t(
            8,
            Kind::Apply(
                Box::new(t(9, Kind::Fail("function"))),
                Box::new(t(10, Kind::Fail("argument"))),
            ),
        );
        let observation = observe(&Program::default(), &first, LIMITS);
        assert_eq!(
            observation.result,
            Err(Error::Trap {
                at: 9,
                label: "function"
            })
        );
        assert!(!observation.events.contains(&Event::Demand(10)));

        let function_diverges = t(
            20,
            Kind::Apply(
                Box::new(call(21, "spin", vec![])),
                Box::new(t(22, Kind::Fail("argument"))),
            ),
        );
        let observation = observe(&spin(), &function_diverges, Limits { steps: 7, ..LIMITS });
        assert_eq!(observation.result, Err(Error::StepLimit(90)));
        assert!(!observation.events.contains(&Event::Demand(22)));

        // Named-call arguments are left-to-right; a later trap is hidden.
        let named = call(
            11,
            "missing",
            vec![
                t(12, Kind::Unit),
                t(13, Kind::Fail("second")),
                t(14, Kind::Fail("third")),
            ],
        );
        let observation = observe(&Program::default(), &named, LIMITS);
        assert_eq!(
            observation.result,
            Err(Error::Trap {
                at: 13,
                label: "second"
            })
        );
        assert!(!observation.events.contains(&Event::Demand(14)));
    }

    #[test]
    fn typed_net_verifier_term_oracle_affine_binding_and_limits() {
        let identity = t(
            1,
            Kind::Lambda {
                binder: 20,
                body: Box::new(t(2, Kind::Var(20))),
            },
        );
        let applied = t(
            0,
            Kind::Apply(Box::new(identity), Box::new(t(3, Kind::Nat(9)))),
        );
        assert_eq!(
            observe(&Program::default(), &applied, LIMITS).result,
            Ok(Value::Nat(9))
        );

        let duplicate = t(
            4,
            Kind::Let {
                binder: 21,
                value: Box::new(t(
                    5,
                    Kind::Lambda {
                        binder: 22,
                        body: Box::new(t(6, Kind::Var(22))),
                    },
                )),
                body: Box::new(t(
                    7,
                    Kind::Pair(Box::new(t(8, Kind::Var(21))), Box::new(t(9, Kind::Var(21)))),
                )),
            },
        );
        assert_eq!(
            observe(&Program::default(), &duplicate, LIMITS).result,
            Err(Error::MissingBinder { at: 9, binder: 21 })
        );

        // Capturing an affine closure consumes it from the surrounding scope.
        let capture = t(
            24,
            Kind::Let {
                binder: 25,
                value: Box::new(t(
                    26,
                    Kind::Lambda {
                        binder: 27,
                        body: Box::new(t(28, Kind::Var(27))),
                    },
                )),
                body: Box::new(t(
                    29,
                    Kind::Let {
                        binder: 30,
                        value: Box::new(t(
                            31,
                            Kind::Lambda {
                                binder: 32,
                                body: Box::new(t(33, Kind::Var(25))),
                            },
                        )),
                        body: Box::new(t(34, Kind::Var(25))),
                    },
                )),
            },
        );
        assert_eq!(
            observe(&Program::default(), &capture, LIMITS).result,
            Err(Error::MissingBinder { at: 34, binder: 25 })
        );

        let copyable = t(
            10,
            Kind::Let {
                binder: 23,
                value: Box::new(t(11, Kind::Nat(4))),
                body: Box::new(t(
                    12,
                    Kind::Pair(
                        Box::new(t(13, Kind::Var(23))),
                        Box::new(t(14, Kind::Var(23))),
                    ),
                )),
            },
        );
        assert_eq!(
            observe(&Program::default(), &copyable, LIMITS).result,
            Ok(Value::Pair(
                Box::new(Value::Nat(4)),
                Box::new(Value::Nat(4))
            ))
        );

        let recursive = call(15, "spin", vec![]);
        assert_eq!(
            observe(&spin(), &recursive, Limits { depth: 3, ..LIMITS }).result,
            Err(Error::DepthLimit(90))
        );
        assert_eq!(
            observe(
                &Program::default(),
                &copyable,
                Limits { nodes: 2, ..LIMITS }
            )
            .result,
            Err(Error::NodeLimit)
        );
    }
}
