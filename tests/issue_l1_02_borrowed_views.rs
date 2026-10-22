use std::process::Command;

use invar::{check, eval, front, mir, mir_eval};

const VIEWS: &str = include_str!("fixtures/l1_02_views.inv");

fn checked(source: &str) -> check::CheckedProgram {
    check::check(front::parse(source.as_bytes()).unwrap(), true).unwrap()
}

#[test]
fn issue_l1_02_borrowed_views_reference_mir_native_agree() {
    let checked = checked(VIEWS);
    let reference = eval::run(&checked, 10_000).unwrap();
    assert_eq!(reference.value, eval::Value::I32(65));
    assert_eq!(
        reference.cleanup_trace,
        vec![eval::CleanupEvent {
            name: "xs".into(),
            length: 6
        }]
    );
    let lowered = mir::lower(&checked).unwrap();
    assert_eq!(mir_eval::run(&lowered, 10_000).unwrap(), reference);
    #[cfg(feature = "native")]
    {
        let (status, document) = invar::native::run_program(&lowered, 10_000).unwrap();
        assert_eq!(status, 0);
        assert_eq!(
            document["result"],
            serde_json::json!({"type":"i32","value":65})
        );
        assert_eq!(document["steps_used"], reference.steps_used);
        assert_eq!(
            document["cleanup_trace"],
            serde_json::json!([{"name":"xs","length":6}])
        );
    }
}

#[test]
fn issue_l1_02_borrowed_views_read_return_from_mut_parameter() {
    let source = "fn peek(xs: mut Array<i32>) -> View<i32> from xs { return view(xs,0u32,1u32); } fn main() -> i32 { var xs: Array<i32> = [8i32]; let part = peek(xs); return part[0u32]; }";
    let checked = checked(source);
    let reference = eval::run(&checked, 1000).unwrap();
    assert_eq!(reference.value, eval::Value::I32(8));
    let lowered = mir::lower(&checked).unwrap();
    assert_eq!(mir_eval::run(&lowered, 1000).unwrap(), reference);
    #[cfg(feature = "native")]
    {
        let (status, document) = invar::native::run_program(&lowered, 1000).unwrap();
        assert_eq!(status, 0);
        assert_eq!(document["result"]["value"], 8);
        assert_eq!(document["steps_used"], reference.steps_used);
    }
}

#[test]
fn issue_l1_02_borrowed_views_nested_reborrow_chain() {
    let source = "fn main() -> i32 { var xs: Array<i32> = [1i32]; let a = view_mut(xs,0u32,1u32); if true { let b = view_mut(a,0u32,1u32); if true { let c = view_mut(b,0u32,1u32); c[0u32] = 9i32; } b[0u32] = b[0u32] + 1i32; } return a[0u32]; }";
    let checked = checked(source);
    let reference = eval::run(&checked, 1000).unwrap();
    assert_eq!(reference.value, eval::Value::I32(10));
    let lowered = mir::lower(&checked).unwrap();
    assert_eq!(mir_eval::run(&lowered, 1000).unwrap(), reference);
    #[cfg(feature = "native")]
    {
        let (status, document) = invar::native::run_program(&lowered, 1000).unwrap();
        assert_eq!(status, 0);
        assert_eq!(document["result"]["value"], 10);
    }
}

#[test]
fn issue_l1_02_borrowed_views_public_check_and_negative_sources() {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/l1_02_views.inv"
    );
    let output = Command::new(env!("CARGO_BIN_EXE_invar"))
        .args(["check", path, "--format", "json"])
        .output()
        .unwrap();
    assert!(output.status.success());
    let document: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(document["status"], "ok");
    for (fixture, code) in [
        ("overlap.inv", "OWN_BORROW_CONFLICT"),
        ("owner-move.inv", "OWN_BORROW_CONFLICT"),
        ("temporary-owner.inv", "OWN_BORROW_SOURCE"),
        ("invalid-return.inv", "OWN_RETURN_BORROW"),
    ] {
        let path = format!(
            "{}/tests/fixtures/l1_02/{fixture}",
            env!("CARGO_MANIFEST_DIR")
        );
        let output = Command::new(env!("CARGO_BIN_EXE_invar"))
            .args(["check", &path, "--format", "json"])
            .output()
            .unwrap();
        assert_eq!(output.status.code(), Some(1), "{fixture}");
        let document: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
        assert_eq!(document["status"], "source_error", "{fixture}");
        assert_eq!(document["diagnostics"][0]["code"], code, "{fixture}");
    }
    for (source, code) in [
        (
            "fn main() -> i32 { var xs: Array<i32> = [1i32,2i32,3i32]; let a = view_mut(xs,0u32,2u32); let b = view_mut(xs,1u32,3u32); return a[0u32]; }",
            "OWN_BORROW_CONFLICT",
        ),
        (
            "fn main() -> i32 { var xs: Array<i32> = [1i32,2i32,3i32]; let n: u32 = 1u32; let a = view_mut(xs,0u32,n); let b = view_mut(xs,n,3u32); return a[0u32]; }",
            "OWN_BORROW_CONFLICT",
        ),
        (
            "fn main() -> i32 { let xs: Array<i32> = [1i32]; let a = view(xs,0u32,1u32); let other: Array<i32> = move xs; return a[0u32]; }",
            "OWN_BORROW_CONFLICT",
        ),
        (
            "fn main() -> i32 { var xs: Array<i32> = [1i32]; let a = view(xs,0u32,1u32); xs[0u32] = 2i32; return a[0u32]; }",
            "OWN_BORROW_CONFLICT",
        ),
        (
            "fn main() -> i32 { let a = view(filled_i32(1u32,1i32),0u32,1u32); return a[0u32]; }",
            "OWN_BORROW_SOURCE",
        ),
        (
            "fn bad(xs: read Array<i32>) -> View<i32> from xs { let local: Array<i32> = [1i32]; return view(local,0u32,1u32); } fn main() -> i32 { return 0i32; }",
            "OWN_RETURN_BORROW",
        ),
        (
            "fn main() -> i32 { let xs: Array<i32> = [1i32]; let a = view_mut(xs,0u32,1u32); return a[0u32]; }",
            "OWN_MUT_REQUIRED",
        ),
        (
            "fn main() -> i32 { var xs: Array<i32> = [1i32]; let a = view_mut(xs,0u32,1u32); if true { let b = view_mut(a,0u32,1u32); a[0u32] = 2i32; } return 0i32; }",
            "OWN_BORROW_CONFLICT",
        ),
        (
            "fn main() -> i32 { var xs: Array<i32> = [1i32]; let a = view_mut(xs,0u32,1u32); let alias = a; return alias[0u32]; }",
            "UNSUPPORTED_FEATURE",
        ),
        (
            "fn main() -> i32 { let n: i32 = 1i32; let a = view(n,0u32,1u32); return 0i32; }",
            "TYPE_MISMATCH",
        ),
    ] {
        let error = check::check(front::parse(source.as_bytes()).unwrap(), true).unwrap_err();
        assert_eq!(error.code, code, "{source}");
    }
}

#[test]
fn issue_l1_02_borrowed_views_range_and_stride_trap_identically() {
    for source in [
        "fn main() -> i32 { let xs: Array<i32> = [1i32,2i32]; let a = view(xs,0u32,3u32); return a[0u32]; }",
        "fn main() -> i32 { let xs: Array<i32> = [1i32,2i32]; let a = view(xs,0u32,2u32,0u32); return a[0u32]; }",
        "fn main() -> i32 { let xs: Array<i32> = [1i32,2i32]; let a = view(xs,0u32,2u32,4294967295u32); let b = view(a,0u32,1u32,2u32); return b[0u32]; }",
        "fn main() -> i32 { let xs: Array<i32> = [1i32,2i32]; let a = view(xs,0u32,2u32,2u32); return a[1u32]; }",
    ] {
        let checked = checked(source);
        let reference = eval::observe(&checked, 10_000);
        assert!(matches!(
            &reference.result,
            Err(eval::EvalError::Trap {
                category: eval::TrapCategory::IndexOutOfBounds,
                ..
            })
        ));
        let lowered = mir::lower(&checked).unwrap();
        let mir = mir_eval::observe(&lowered, 10_000);
        assert_eq!(mir.steps_used, reference.steps_used);
        assert_eq!(mir.result, reference.result);
        #[cfg(feature = "native")]
        {
            let (status, document) = invar::native::run_program(&lowered, 10_000).unwrap();
            assert_eq!(status, 3);
            assert_eq!(document["trap"]["category"], "INDEX_OUT_OF_BOUNDS");
            assert_eq!(document["steps_used"], reference.steps_used);
        }
    }
}

#[test]
fn issue_l1_02_borrowed_views_mir_rejects_forged_view_source() {
    let checked = checked(VIEWS);
    let mut lowered = mir::lower(&checked).unwrap();
    let mut changed = false;
    for function in &mut lowered.functions {
        for block in &mut function.blocks {
            for instruction in &mut block.instructions {
                if let mir::Operation::Call { name, args } = &mut instruction.kind
                    && name == "view"
                {
                    args[0] = args[1];
                    changed = true;
                    break;
                }
            }
        }
    }
    assert!(changed);
    assert_eq!(mir::verify(&lowered).unwrap_err().code, "MIR_INVALID");
}

#[test]
fn issue_l1_02_borrowed_views_mir_rejects_owner_drop_and_return_escape() {
    let program = checked(VIEWS);
    let mut lowered = mir::lower(&program).unwrap();
    let main = lowered
        .functions
        .iter_mut()
        .find(|function| function.name == "main")
        .unwrap();
    let block = main
        .blocks
        .iter_mut()
        .find(|block| {
            block.instructions.iter().any(
                |item| matches!(&item.kind, mir::Operation::Bind { name, .. } if name == "empty"),
            )
        })
        .unwrap();
    let position = block
        .instructions
        .iter()
        .position(|item| matches!(&item.kind, mir::Operation::Bind { name, .. } if name == "empty"))
        .unwrap();
    let span = block.instructions[position].span;
    block.instructions.insert(
        position + 1,
        mir::Instruction {
            output: None,
            kind: mir::Operation::Drop(vec!["xs".into()]),
            span,
        },
    );
    assert_eq!(mir::verify(&lowered).unwrap_err().code, "MIR_INVALID");

    let source = "fn choose(a: read Array<i32>, b: read Array<i32>) -> View<i32> from a { return view(a,0u32,1u32); } fn main() -> i32 { let x: Array<i32> = [1i32]; let y: Array<i32> = [2i32]; let chosen = choose(x,y); return chosen[0u32]; }";
    let checked = checked(source);
    let mut lowered = mir::lower(&checked).unwrap();
    let choose = lowered
        .functions
        .iter_mut()
        .find(|function| function.name == "choose")
        .unwrap();
    choose.return_from = Some("b".into());
    assert_eq!(mir::verify(&lowered).unwrap_err().code, "MIR_INVALID");
}

#[test]
fn issue_l1_02_borrowed_views_mir_consumes_materialized_view_once() {
    let source = "fn relay(xs: read Array<i32>) -> View<i32> from xs { let part = view(xs,0u32,1u32); return part; } fn main() -> i32 { let xs: Array<i32> = [7i32]; let part = relay(xs); return part[0u32]; }";
    let checked = checked(source);
    let mut lowered = mir::lower(&checked).unwrap();
    let relay = lowered
        .functions
        .iter_mut()
        .find(|function| function.name == "relay")
        .unwrap();
    let original = relay
        .blocks
        .iter()
        .flat_map(|block| &block.instructions)
        .find_map(|item| {
            matches!(&item.kind, mir::Operation::Call { name, .. } if name == "view")
                .then(|| item.output.as_ref().unwrap().0)
        })
        .unwrap();
    let return_block = relay
        .blocks
        .iter_mut()
        .find(|block| matches!(block.end, Some(mir::Terminator::Return { .. })))
        .unwrap();
    let Some(mir::Terminator::Return { value, .. }) = &mut return_block.end else {
        unreachable!()
    };
    *value = Some(original);
    assert_eq!(mir::verify(&lowered).unwrap_err().code, "MIR_INVALID");
}
