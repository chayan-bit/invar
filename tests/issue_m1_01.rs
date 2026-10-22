use invar::{check::check, eval, front::parse, mir, mir_eval};

fn lower(source: &str) -> mir::Program {
    let checked = check(parse(source.as_bytes()).unwrap(), true).unwrap();
    mir::lower(&checked).unwrap()
}

#[test]
fn issue_m1_01_lowering_verifier_smoke() {
    for source in [
        "fn main() -> i32 { return 42; }",
        "fn main() -> i32 { if true { return 1; } else { return 2; } }",
        "fn main() -> i32 { var x: i32 = 0; while x < 2 { x = x + 1; } return x; }",
        "fn main() -> i32 { var x: u32 = 0u32; for i in 0u32 .. 2u32 { x = x + i; } return 1; }",
        "fn main() -> i32 { for i in 0u32 .. 2u32 { let a: Array<i32> = [1]; let b: Array<i32> = move a; } return 0; }",
        include_str!("../examples/sum.inv"),
        include_str!("../examples/ownership.inv"),
    ] {
        let program = lower(source);
        mir::verify(&program).unwrap();
    }
}

#[test]
fn issue_m1_01_typed_mir_independent_evaluator() {
    // Fixed corpus; no random seed or native-oracle claim.
    for (source, fuel) in [
        ("fn main() -> i32 { return 42; }", 10_000),
        ("fn main() -> i32 { return 42; }", 1),
        (
            "fn main() -> i32 { if true { return 1; } else { return 1 / 0; } }",
            10_000,
        ),
        (
            "fn main() -> i32 { if false and (1 / 0 == 0) { return 1; } else { return 2; } }",
            10_000,
        ),
        (
            "fn main() -> i32 { if true or (1 / 0 == 0) { return 3; } else { return 4; } }",
            10_000,
        ),
        (
            "fn main() -> i32 { var x: i32 = 0; while x < 2 { x = x + 1; } return x; }",
            10_000,
        ),
        (
            "fn main() -> i32 { var x: u32 = 0u32; for i in 0u32 .. 2u32 { x = x + i; } return 1; }",
            10_000,
        ),
        (
            "fn main() -> i32 { for i in 0u32 .. 2u32 { let a: Array<i32> = [1]; let b: Array<i32> = move a; } return 0; }",
            10_000,
        ),
        (
            "fn main() -> i32 { let xs: Array<i32> = []; if len(xs) == 0u32 { return 1; } else { return 0; } }",
            10_000,
        ),
        (
            "fn main() -> i32 { let xs: Array<i32> = filled_i32(3u32, 7); return xs[2u32]; }",
            10_000,
        ),
        (
            "fn f(n: i32) -> i32 { if n == 0 { return 1; } else { return n * f(n - 1); } } fn main() -> i32 { return f(5); }",
            10_000,
        ),
        (
            "fn main() -> i32 { return -2147483648i32 / -1i32; }",
            10_000,
        ),
        (
            "fn main() -> i32 { return -2147483648i32 % -1i32; }",
            10_000,
        ),
        (
            "fn main() -> i32 { let xs: Array<i32> = [1]; return xs[2u32]; }",
            10_000,
        ),
        (
            "fn main() -> i32 { var xs: Array<i32> = [1]; xs[2u32] = 1 / 0; return 0; }",
            10_000,
        ),
        (
            "fn main() -> i32 { var xs: Array<i32> = [1]; xs[2u32] = 7; return 0; }",
            10_000,
        ),
        (
            "fn main() -> i32 { for i in 4294967295u32 .. 4294967295u32 {} return 8; }",
            10_000,
        ),
        (
            "fn make() -> Array<i32> { return [9]; } fn main() -> i32 { let xs: Array<i32> = make(); return xs[0u32]; }",
            10_000,
        ),
        ("fn main() -> i32 { while true {} return 0; }", 8),
        (include_str!("../examples/sum.inv"), 10_000),
        (include_str!("../examples/ownership.inv"), 10_000),
        (
            "fn read_one(a: read Array<i32>) -> i32 { return a[0u32]; } fn bump(a: mut Array<i32>) -> Unit { a[0u32] = a[0u32] + read_one(a); return; } fn main() -> i32 { var xs: Array<i32> = [2]; bump(xs); return xs[0u32]; }",
            10_000,
        ),
    ] {
        let checked = check(parse(source.as_bytes()).unwrap(), true).unwrap();
        let lowered = mir::lower(&checked).unwrap();
        let reference = eval::observe(&checked, fuel);
        let mir = mir_eval::observe(&lowered, fuel);
        assert_eq!(mir.result, reference.result, "{source}");
        assert_eq!(mir.steps_used, reference.steps_used, "{source}");
        assert_eq!(mir.cleanup_trace, reference.cleanup_trace, "{source}");
    }
}

#[test]
fn issue_m1_01_typed_mir_verifier_rejects_mutations() {
    let original = lower("fn main() -> i32 { return 1; }");
    let mut missing = original.clone();
    missing.functions[0].blocks[0].end = None;
    assert!(
        mir::verify(&missing)
            .unwrap_err()
            .message
            .contains("missing terminator")
    );

    let mut target = original.clone();
    target.functions[0].blocks[0].end = Some(mir::Terminator::Goto {
        target: usize::MAX,
        args: vec![],
        span: target.functions[0].span,
    });
    assert!(
        mir::verify(&target)
            .unwrap_err()
            .message
            .contains("missing successor")
    );

    let mut type_mutation = original.clone();
    let output = type_mutation.functions[0].blocks[0]
        .instructions
        .iter_mut()
        .find_map(|instruction| instruction.output.as_mut())
        .unwrap();
    output.1 = invar::front::Type::U32;
    assert!(
        mir::verify(&type_mutation)
            .unwrap_err()
            .message
            .contains("output type")
    );

    let mut stale = original.clone();
    stale.functions[0].blocks[0].instructions[0].span.start = stale.functions[0].span.end + 1;
    assert!(
        mir::verify(&stale)
            .unwrap_err()
            .message
            .contains("stale source span")
    );

    let mut unknown = original.clone();
    unknown.schema_version += 1;
    assert!(
        mir::verify(&unknown)
            .unwrap_err()
            .message
            .contains("unknown MIR schema")
    );

    let mut arity = lower("fn main() -> i32 { if true { return 1; } else { return 2; } }");
    let yes = match &arity.functions[0].blocks[0].end {
        Some(mir::Terminator::Branch { yes, .. }) => *yes,
        other => panic!("expected branch, got {other:?}"),
    };
    arity.functions[0].blocks[yes]
        .params
        .push((999, invar::front::Type::I32));
    assert!(
        mir::verify(&arity)
            .unwrap_err()
            .message
            .contains("arity mismatch")
    );

    let mut unsupported = original.clone();
    let unsupported_span = unsupported.functions[0].span;
    unsupported.functions[0].blocks[0]
        .instructions
        .push(mir::Instruction {
            output: Some((999, invar::front::Type::I32)),
            kind: mir::Operation::Call {
                name: "unknown_effect".into(),
                args: vec![],
            },
            span: unsupported_span,
        });
    assert!(
        mir::verify(&unsupported)
            .unwrap_err()
            .message
            .contains("unknown MIR call")
    );

    let mut dominance =
        lower("fn main() -> i32 { var x: i32 = 0; if true { x = 1; } else { x = 2; } return x; }");
    let branch_value = dominance.functions[0]
        .blocks
        .iter()
        .flat_map(|block| &block.instructions)
        .find_map(|instruction| {
            matches!(instruction.kind, mir::Operation::I32(1))
                .then(|| instruction.output.as_ref().unwrap().0)
        })
        .unwrap();
    let returned = dominance.functions[0]
        .blocks
        .iter_mut()
        .find_map(|block| match &mut block.end {
            Some(mir::Terminator::Return {
                value: Some(id), ..
            }) => Some(id),
            _ => None,
        })
        .unwrap();
    *returned = branch_value;
    assert!(
        mir::verify(&dominance)
            .unwrap_err()
            .message
            .contains("dominate")
    );

    let mut ownership = lower(
        "fn main() -> i32 { let a: Array<i32> = [1]; if true { let b: Array<i32> = move a; return b[0u32]; } else {} return a[0u32]; }",
    );
    let ownership_span = ownership.functions[0].span;
    let join = ownership.functions[0]
        .blocks
        .iter()
        .find_map(|block| match &block.end {
            Some(mir::Terminator::Goto { target, args, .. }) if args.is_empty() && *target > 1 => {
                Some(*target)
            }
            _ => None,
        })
        .unwrap();
    let returning = ownership.functions[0]
        .blocks
        .iter_mut()
        .find(|block| matches!(block.end, Some(mir::Terminator::Return { .. })))
        .unwrap();
    returning.end = Some(mir::Terminator::Goto {
        target: join,
        args: vec![],
        span: ownership_span,
    });
    assert!(
        mir::verify(&ownership)
            .unwrap_err()
            .message
            .contains("moved MIR place")
    );

    let mut cleanup = lower("fn main() -> i32 { let xs: Array<i32> = [1]; return 0; }");
    for block in &mut cleanup.functions[0].blocks {
        if let Some(mir::Terminator::Return { cleanup, .. }) = &mut block.end {
            cleanup.clear();
        }
    }
    assert!(
        mir::verify(&cleanup)
            .unwrap_err()
            .message
            .contains("return cleanup")
    );

    let mut loans = lower(
        "fn both(a: read Array<i32>, b: read Array<i32>) -> Unit { return; } fn main() -> i32 { var xs: Array<i32> = [1]; both(xs, xs); return 0; }",
    );
    loans.functions[0].params[1].mode = Some(invar::front::Mode::Mut);
    loans.functions[0].locals.get_mut("b").unwrap().mode = Some(invar::front::Mode::Mut);
    let main = &mut loans.functions[1];
    let refs: Vec<_> = main
        .blocks
        .iter_mut()
        .flat_map(|block| &mut block.instructions)
        .filter_map(|instruction| match &mut instruction.kind {
            mir::Operation::ArrayRef { loan, .. } if loan.is_some() => Some(loan),
            _ => None,
        })
        .collect();
    *refs.into_iter().nth(1).unwrap() = Some(invar::front::Mode::Mut);
    assert!(
        mir::verify(&loans)
            .unwrap_err()
            .message
            .contains("loan conflict")
    );
}

#[test]
fn issue_m1_01_differential_detects_semantic_mutation() {
    let source = "fn main() -> i32 { return 2 + 3; }";
    let checked = check(parse(source.as_bytes()).unwrap(), true).unwrap();
    let mut lowered = mir::lower(&checked).unwrap();
    assert_eq!(eval::run(&checked, 100).unwrap().value, eval::Value::I32(5));
    let operation = lowered.functions[0]
        .blocks
        .iter_mut()
        .flat_map(|block| &mut block.instructions)
        .find_map(|instruction| match &mut instruction.kind {
            mir::Operation::Binary { op, .. } => Some(op),
            _ => None,
        })
        .unwrap();
    *operation = invar::front::BinaryOp::Sub;
    mir::verify(&lowered).unwrap();
    assert_eq!(
        mir_eval::run(&lowered, 100).unwrap().value,
        eval::Value::I32(-1)
    );
}
