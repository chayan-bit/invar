use invar::{check::check, eval, front::parse, mir, mir_eval};

use crate::corpus::CORPUS;

fn lower(source: &str) -> mir::Program {
    let checked = check(parse(source.as_bytes()).unwrap(), true).unwrap();
    mir::lower(&checked).unwrap()
}

#[test]
fn issue_m1_01_typed_mir_independent_evaluator() {
    for &(source, fuel) in CORPUS {
        let checked = check(parse(source.as_bytes()).unwrap(), true).unwrap();
        let lowered = mir::lower(&checked).unwrap();
        mir::verify(&lowered).unwrap();
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
