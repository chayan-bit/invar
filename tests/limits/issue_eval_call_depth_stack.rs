//! The reference evaluator reaches its documented 64-call limit and reports
//! the typed `CALL_DEPTH_LIMIT`, never a host stack overflow, when called from
//! a default 2 MiB thread, in Debug and Release. This includes a recursive call
//! nested at the checked maximum expression height in every frame, and every
//! root entry (main, named scalar exports, host calls).
#![forbid(unsafe_code)]

use invar::{
    check,
    eval::{
        self, EvalError, Value,
        host_read::{HostArgument, HostMutArray},
        named,
    },
    front,
};

const DEFAULT_THREAD_STACK: usize = 2 * 1024 * 1024;

fn on_default_thread<T: Send>(body: impl FnOnce() -> T + Send) -> T {
    std::thread::scope(|scope| {
        std::thread::Builder::new()
            .stack_size(DEFAULT_THREAD_STACK)
            .spawn_scoped(scope, body)
            .unwrap()
            .join()
            .expect("the evaluator must not overflow a 2 MiB caller thread")
    })
}

fn checked(source: &str, main: bool) -> Option<check::CheckedProgram> {
    check::check(front::parse(source.as_bytes()).ok()?, main).ok()
}

fn depth_limited(result: &Result<Value, EvalError>) -> bool {
    matches!(
        result,
        Err(EvalError::Tool {
            code: "CALL_DEPTH_LIMIT",
            ..
        })
    )
}

/// `r(n)` with its recursive call as the innermost leaf of a left-nested
/// `+` chain of `terms` levels.
fn recursion(terms: usize, n: u32) -> String {
    let tail = " + 1".repeat(terms);
    format!(
        "fn r(n: u32) -> i32 {{ if n == 0u32 {{ return 1; }} return r(n - 1u32){tail}; }} fn main() -> i32 {{ return r({n}u32); }}"
    )
}

#[test]
fn issue_eval_call_depth_limit_is_typed_on_a_default_thread() {
    on_default_thread(|| {
        for (n, limited) in [
            (10, false),
            (62, false),
            (63, true),
            (64, true),
            (500, true),
        ] {
            let program = checked(&recursion(1, n), true).unwrap();
            let observed = eval::observe(&program, 1_000_000);
            assert_eq!(
                depth_limited(&observed.result),
                limited,
                "{n}: {observed:?}"
            );
            if !limited {
                assert_eq!(observed.result, Ok(Value::I32(1 + n as i32)));
            }
        }
    });
}

#[test]
fn issue_eval_call_depth_limit_at_maximum_expression_height_in_every_frame() {
    // The largest accepted chain: one more level is a checked height refusal.
    let terms = (1..=1000)
        .rev()
        .find(|terms| checked(&recursion(*terms, 1), true).is_some())
        .unwrap();
    assert!(checked(&recursion(terms + 1, 1), true).is_none());
    assert!(terms >= 100, "{terms}");
    on_default_thread(|| {
        let program = checked(&recursion(terms, 500), true).unwrap();
        let observed = eval::observe(&program, 10_000_000);
        assert!(depth_limited(&observed.result), "{observed:?}");
        let program = checked(&recursion(terms, 62), true).unwrap();
        let observed = eval::observe(&program, 10_000_000);
        assert!(observed.result.is_ok(), "{observed:?}");
    });
    println!("maximum chain height per frame: {terms}");
}

#[test]
fn issue_eval_call_depth_limit_named_and_host_entries_on_a_default_thread() {
    let source = "fn descend(n: u32) -> i32 { if n == 0u32 { return 7; } return descend(n - 1u32) + 0; } fn fill(xs: mut Array<i32>, n: u32) -> i32 { xs[0u32] = 9; return descend(n); }";
    let program = checked(source, false).unwrap();
    invar::ownership::check(&program).unwrap();
    on_default_thread(|| {
        let deep = named::observe_named(&program, "descend", &[Value::U32(500)], 1_000_000);
        assert!(depth_limited(&deep.result), "{deep:?}");
        let fine = named::observe_named(&program, "descend", &[Value::U32(60)], 1_000_000);
        assert_eq!(fine.result, Ok(Value::I32(7)));
        let mut buffer = [1i32, 2];
        let mut args = [
            HostArgument::Mut(HostMutArray::I32(&mut buffer)),
            HostArgument::U32(500),
        ];
        let host = eval::host_read::observe_host_call(&program, "fill", &mut args, 1_000_000);
        assert!(depth_limited(&host.result), "{host:?}");
        // A limit is not a normal return: the caller buffer is not committed.
        assert_eq!(buffer, [1, 2]);
    });
}
