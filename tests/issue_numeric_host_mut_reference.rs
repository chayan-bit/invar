#![forbid(unsafe_code)]
//! HA-1.1 stage 1: `mut` host array arguments on the copying reference engine.
//!
//! The independent comparator is the unchanged in-language call: a wrapper
//! declares one `var` owner per `mut` buffer and one owner per distinct `read`
//! buffer, calls the function, and returns either the result or a final `mut`
//! owner. Steps are compared by differencing against stubs of the same
//! signature and outcome class, anchored by parameterless twins. A normal
//! return (including `Result::Err`) commits; every other outcome and every
//! refusal must leave each caller buffer bit-identical.
use invar::{
    check,
    eval::{
        self, EvalError, Observation, Value,
        host_read::{
            HostArgument as A, HostMutArray as M, HostReadArgument as RA, HostReadArray as R,
            observe_host_call, observe_host_read_call,
        },
    },
    front,
    numeric_contract::F32Bits,
};

const SOURCE: &str = "\
fn store2(xs: mut Array<i32>, i: u32, v: i32) -> i32 { xs[0u32] = xs[0u32] + 1; xs[i] = v; return xs[i]; }
fn div_after(xs: mut Array<i32>, d: i32) -> i32 { xs[0u32] = 1; return xs[1u32] / d; }
fn over(xs: mut Array<i32>) -> i32 { xs[0u32] = 2147483647; xs[1u32] = xs[0u32] + 1; return 0; }
fn inner_mut(xs: mut Array<i32>) -> i32 { xs[1u32] = xs[1u32] * 3; return 0; }
fn inner_read(xs: read Array<i32>) -> i32 { return xs[0u32] + xs[1u32]; }
fn nested(xs: mut Array<i32>) -> i32 { xs[0u32] = 10; let a = inner_mut(xs); return inner_read(xs); }
fn two(a: mut Array<u32>, b: mut Array<u32>, r1: read Array<u32>, r2: read Array<u32>) -> u32 { a[0u32] = r1[0u32] + r2[1u32]; b[0u32] = a[0u32] * 2u32; return b[0u32]; }
fn fcopy(xs: mut Array<f32>, ys: read Array<f32>, i: u32) -> f32 { xs[0u32] = ys[0u32]; xs[1u32] = ys[1u32]; xs[i] = ys[2u32]; return xs[1u32]; }
fn grow_after(xs: mut Array<i32>, ys: read Array<i32>) -> u32 { xs[0u32] = 7; let t = filled_u32(1u32, 0u32); return len(t) + len(ys); }
fn deep(xs: mut Array<i32>, n: u32) -> i32 { xs[0u32] = xs[0u32] + 1; return deep(xs, n + 1u32); }
fn settle(xs: mut Array<i32>, k: i32) -> Result<i32, u32> { xs[0u32] = k; if k < 0 { return Result::Err(7u32); } return Result::Ok(k); }
fn flag(xs: mut Array<u32>) -> Result<Bool, i32> { xs[0u32] = 1u32; return Result::Ok(true); }
fn wide(xs: mut Array<i32>) -> Result<Result<i32, u32>, u32> { xs[0u32] = 1; return Result::Err(1u32); }
fn eat(xs: take Array<i32>) -> u32 { return len(xs); }
fn peek(xs: read Array<i32>) -> Result<i32, u32> { if len(xs) == 0u32 { return Result::Err(0u32); } return Result::Ok(xs[0u32]); }
fn empty_ok(xs: mut Array<i32>, ys: read Array<i32>) -> u32 { return len(xs) + len(ys); }
fn main() -> i32 { return 7; }";

fn checked(source: &str) -> check::CheckedProgram {
    let c = check::check(front::parse(source.as_bytes()).unwrap(), false).unwrap();
    invar::ownership::check(&c).unwrap();
    c
}

fn code(o: &Observation) -> &'static str {
    match &o.result {
        Err(EvalError::Tool { code, .. }) => code,
        x => panic!("expected tool refusal, got {x:?}"),
    }
}

/// Owned copy of host buffers, for building fresh arguments per call.
#[derive(Clone, Debug, PartialEq)]
enum Buf {
    I32(Vec<i32>),
    U32(Vec<u32>),
    F32(Vec<F32Bits>),
}

/// One argument description: scalars, a read of buffer `b`, or a mut of
/// buffer `b`. Reads of the same buffer index are exact aliases.
#[derive(Clone, Copy, Debug)]
enum Arg {
    I32(i32),
    U32(u32),
    Read(usize),
    Mut(usize),
}

fn literal(b: &Buf) -> (String, String) {
    match b {
        Buf::I32(v) if v.len() > 16 && v.iter().all(|x| *x == v[0]) => (
            "Array<i32>".into(),
            format!("filled_i32({}u32, {})", v.len(), v[0]),
        ),
        Buf::I32(v) if v.is_empty() => ("Array<i32>".into(), "filled_i32(0u32, 0)".into()),
        Buf::I32(v) => (
            "Array<i32>".into(),
            format!(
                "[{}]",
                v.iter().map(i32::to_string).collect::<Vec<_>>().join(",")
            ),
        ),
        Buf::U32(v) => (
            "Array<u32>".into(),
            format!(
                "[{}]",
                v.iter()
                    .map(|x| format!("{x}u32"))
                    .collect::<Vec<_>>()
                    .join(",")
            ),
        ),
        Buf::F32(v) => (
            "Array<f32>".into(),
            format!(
                "[{}]",
                v.iter()
                    .map(|x| format!("{:?}", f32::from_bits(x.to_bits())))
                    .collect::<Vec<_>>()
                    .join(",")
            ),
        ),
    }
}

/// Call the host API with fresh mutable copies of `bufs`; returns the
/// observation and the buffers afterwards. A buffer is either read or `mut`
/// within one call (safe borrows cannot overlap).
fn host(
    program: &check::CheckedProgram,
    name: &str,
    args: &[Arg],
    bufs: &[Buf],
    fuel: u64,
) -> (Observation, Vec<Buf>) {
    let reads = bufs.to_vec();
    let mut after = bufs.to_vec();
    let mut slots: Vec<Option<&mut Buf>> = after.iter_mut().map(Some).collect();
    let mut host_args: Vec<A<'_>> = Vec::new();
    for arg in args {
        host_args.push(match arg {
            Arg::I32(x) => A::I32(*x),
            Arg::U32(x) => A::U32(*x),
            Arg::Read(b) => A::Read(match &reads[*b] {
                Buf::I32(v) => R::I32(v),
                Buf::U32(v) => R::U32(v),
                Buf::F32(v) => R::F32(v),
            }),
            Arg::Mut(b) => A::Mut(match slots[*b].take().expect("one mut per buffer") {
                Buf::I32(v) => M::I32(v),
                Buf::U32(v) => M::U32(v),
                Buf::F32(v) => M::F32(v),
            }),
        });
    }
    let o = observe_host_call(program, name, &mut host_args, fuel);
    drop(host_args);
    (o, after)
}

fn signature(name: &str) -> (String, String) {
    let line = SOURCE
        .lines()
        .find(|l| l.starts_with(&format!("fn {name}(")))
        .unwrap();
    let signature = &line[..line.find(" {").unwrap()];
    let result = signature[signature.rfind("-> ").unwrap() + 3..].to_owned();
    (signature.to_owned(), result)
}

/// The in-source comparator for a scalar-result call: result, final `mut`
/// contents, cleanup, outcome-matched step delta and absolute stub steps.
fn compare(
    name: &str,
    args: &[Arg],
    bufs: &[Buf],
    fuel: u64,
    label: &str,
) -> (Observation, Vec<Buf>) {
    let (signature, result_ty) = signature(name);
    let body = match result_ty.as_str() {
        "i32" => "0",
        "u32" => "0u32",
        "f32" => "0.0",
        other => panic!("{other}"),
    };
    let renamed = |prefix: &str| {
        signature.replacen(&format!("fn {name}("), &format!("fn {prefix}_{name}("), 1)
    };
    let trap = "let z = 0u32; let t = 1u32 / z;";
    let mut decls = String::new();
    let mut declared = vec![false; bufs.len()];
    let mut call = Vec::new();
    for arg in args {
        match arg {
            Arg::I32(x) => call.push(format!("{x}")),
            Arg::U32(x) => call.push(format!("{x}u32")),
            Arg::Read(b) | Arg::Mut(b) => {
                if !declared[*b] {
                    let (ty, lit) = literal(&bufs[*b]);
                    let kw = if args.iter().any(|a| matches!(a, Arg::Mut(m) if m == b)) {
                        "var"
                    } else {
                        "let"
                    };
                    decls.push_str(&format!("{kw} a{b}: {ty} = {lit}; "));
                    declared[*b] = true;
                }
                call.push(format!("a{b}"));
            }
        }
    }
    let call = call.join(", ");
    let mut source = format!(
        "{SOURCE}\n{} {{ let s = {body}; return s; }}\n{} {{ {trap} let s = {body}; return s; }}\n\
fn k_stub() -> {result_ty} {{ let s = {body}; return s; }}\nfn k_trap() -> {result_ty} {{ {trap} let s = {body}; return s; }}\n\
fn w_result() -> {result_ty} {{ {decls}let r = {name}({call}); return r; }}\n\
fn w_stub() -> {result_ty} {{ {decls}let r = stub_{name}({call}); return r; }}\n\
fn w_trap() -> {result_ty} {{ {decls}let r = trap_{name}({call}); return r; }}\n",
        renamed("stub"),
        renamed("trap")
    );
    let muts: Vec<usize> = args
        .iter()
        .filter_map(|a| if let Arg::Mut(b) = a { Some(*b) } else { None })
        .collect();
    for b in &muts {
        let (ty, _) = literal(&bufs[*b]);
        source.push_str(&format!(
            "fn w_after{b}() -> {ty} {{ {decls}let r = {name}({call}); return a{b}; }}\n"
        ));
    }
    let program = checked(&source);
    let (o, after) = host(&program, name, args, bufs, fuel);
    let wrapped = eval::named::observe_owned_numeric_named(&program, "w_result", &[], fuel);
    assert_eq!(o.result, wrapped.result, "{label}: result");
    let (prefix, stub_wrapper) = if o.result.is_ok() {
        ("stub", "w_stub")
    } else {
        ("trap", "w_trap")
    };
    let (host_stub, stub_after) = host(&program, &format!("{prefix}_{name}"), args, bufs, fuel);
    assert_eq!(stub_after, bufs, "{label}: stub writes nothing");
    let wrapped_stub = eval::named::observe_owned_numeric_named(&program, stub_wrapper, &[], fuel);
    assert_eq!(
        host_stub.result, wrapped_stub.result,
        "{label}: stub result"
    );
    let anchor =
        eval::named::observe_owned_numeric_named(&program, &format!("k_{prefix}"), &[], fuel);
    assert_eq!(
        anchor.steps_used, host_stub.steps_used,
        "{label}: absolute stub steps"
    );
    assert_eq!(
        i128::from(o.steps_used) - i128::from(host_stub.steps_used),
        i128::from(wrapped.steps_used) - i128::from(wrapped_stub.steps_used),
        "{label}: step delta"
    );
    let callee = &o.cleanup_trace;
    if o.result.is_ok() {
        assert_eq!(
            &wrapped.cleanup_trace[..callee.len()],
            callee.as_slice(),
            "{label}: cleanup"
        );
        for b in &muts {
            let w = eval::named::observe_owned_numeric_named(
                &program,
                &format!("w_after{b}"),
                &[],
                fuel,
            );
            let got = match (&w.result, &after[*b]) {
                (Ok(Value::ArrayI32(v)), Buf::I32(h)) => *v.borrow() == *h,
                (Ok(Value::ArrayU32(v)), Buf::U32(h)) => *v.borrow() == *h,
                (Ok(Value::ArrayF32(v)), Buf::F32(h)) => *v.borrow() == *h,
                _ => false,
            };
            assert!(
                got,
                "{label}: committed buffer {b} equals the in-source owner: {:?} vs {:?}",
                w.result, after[*b]
            );
        }
    } else {
        assert_eq!(
            &wrapped.cleanup_trace, callee,
            "{label}: no cleanup after failure"
        );
        assert_eq!(
            after, bufs,
            "{label}: every buffer unchanged after a non-normal outcome"
        );
    }
    (o, after)
}

fn f32s(values: &[f32]) -> Vec<F32Bits> {
    values
        .iter()
        .map(|x| F32Bits::from_bits(x.to_bits()))
        .collect()
}

#[test]
fn normal_returns_commit_exactly_like_in_source_owners() {
    let cases: Vec<(&str, Vec<Arg>, Vec<Buf>)> = vec![
        (
            "store2",
            vec![Arg::Mut(0), Arg::U32(2), Arg::I32(-9)],
            vec![Buf::I32(vec![1, 2, 3])],
        ),
        ("nested", vec![Arg::Mut(0)], vec![Buf::I32(vec![0, 5])]),
        (
            "two",
            vec![Arg::Mut(0), Arg::Mut(1), Arg::Read(2), Arg::Read(2)],
            vec![
                Buf::U32(vec![0, 1]),
                Buf::U32(vec![9]),
                Buf::U32(vec![3, 4]),
            ],
        ),
        (
            "fcopy",
            vec![Arg::Mut(0), Arg::Read(1), Arg::U32(2)],
            vec![
                Buf::F32(f32s(&[1.0, 2.0, 3.0])),
                Buf::F32(f32s(&[-0.5, 0.25, 8.0])),
            ],
        ),
        (
            "empty_ok",
            vec![Arg::Mut(0), Arg::Read(1)],
            vec![Buf::I32(vec![]), Buf::I32(vec![])],
        ),
    ];
    for (name, args, bufs) in &cases {
        let (o, _) = compare(name, args, bufs, 100_000, name);
        assert!(o.result.is_ok(), "{name}: {:?}", o.result);
    }
}

#[test]
fn non_normal_outcomes_leave_every_buffer_unchanged_with_exact_spans() {
    let cases: Vec<(&str, Vec<Arg>, Vec<Buf>, &str)> = vec![
        (
            "store2",
            vec![Arg::Mut(0), Arg::U32(3), Arg::I32(4)],
            vec![Buf::I32(vec![1, 2, 3])],
            "xs[i]",
        ),
        (
            "div_after",
            vec![Arg::Mut(0), Arg::I32(0)],
            vec![Buf::I32(vec![5, 6])],
            "/",
        ),
        ("over", vec![Arg::Mut(0)], vec![Buf::I32(vec![5, 6])], "+"),
        (
            "fcopy",
            vec![Arg::Mut(0), Arg::Read(1), Arg::U32(9)],
            vec![
                Buf::F32(f32s(&[1.0, 2.0, 3.0])),
                Buf::F32(f32s(&[-0.5, 0.25, 8.0])),
            ],
            "xs[i]",
        ),
    ];
    for (name, args, bufs, span_text) in &cases {
        let (o, _) = compare(name, args, bufs, 100_000, name);
        let Err(EvalError::Trap { span, .. }) = &o.result else {
            panic!("{name}: {:?}", o.result)
        };
        assert_eq!(
            &SOURCE[span.start..span.end],
            *span_text,
            "{name}: original span"
        );
    }
    // Tool limits after writes. Call depth: a wrapper adds one frame, so only
    // the host outcome is checked here.
    let program = checked(SOURCE);
    let bufs = vec![Buf::I32(vec![0])];
    let (o, after) = host(
        &program,
        "deep",
        &[Arg::Mut(0), Arg::U32(0)],
        &bufs,
        100_000,
    );
    assert_eq!(code(&o), "CALL_DEPTH_LIMIT");
    assert!(o.steps_used > 0 && o.cleanup_trace.is_empty());
    assert_eq!(after, bufs);
    let million = vec![1; 1_000_000];
    let (o, _) = compare(
        "grow_after",
        &[Arg::Mut(0), Arg::Read(1)],
        &[Buf::I32(million.clone()), Buf::I32(million)],
        100_000,
        "grow_after",
    );
    assert_eq!(o.result, Ok(Value::U32(1_000_001)));
    assert!(o.steps_used > 0);
}

#[test]
fn every_fuel_prefix_leaves_buffers_unchanged_and_full_fuel_commits() {
    let bufs = vec![Buf::I32(vec![1, 2, 3])];
    let args = [Arg::Mut(0), Arg::U32(2), Arg::I32(-9)];
    let program = checked(SOURCE);
    let (done, committed) = host(&program, "store2", &args, &bufs, 100_000);
    assert_eq!(done.result, Ok(Value::I32(-9)));
    assert_eq!(committed, vec![Buf::I32(vec![2, 2, -9])]);
    for fuel in 1..done.steps_used {
        let (o, after) = host(&program, "store2", &args, &bufs, fuel);
        assert_eq!(
            o.result,
            Err(EvalError::EvaluationLimit {
                fuel_limit: fuel,
                steps_used: fuel
            })
        );
        assert!(o.cleanup_trace.is_empty());
        assert_eq!(after, bufs, "fuel {fuel}");
    }
    let (o, after) = host(&program, "store2", &args, &bufs, done.steps_used);
    assert_eq!((o.result, after), (done.result, committed));
    compare("store2", &args, &bufs, 100_000, "full fuel");
}

#[test]
fn result_err_is_a_normal_committing_return() {
    let program = checked(&format!(
        "{SOURCE}\n\
fn stub_settle(xs: mut Array<i32>, k: i32) -> Result<i32, u32> {{ if k < 0 {{ return Result::Err(0u32); }} return Result::Ok(0); }}\n\
fn w_tag(k: i32) -> i32 {{ var a: Array<i32> = [3, 4]; let r = settle(a, k); return match r {{ Result::Ok(v) => v, Result::Err(e) => 0 - 1 }}; }}\n\
fn w_stub(k: i32) -> i32 {{ var a: Array<i32> = [3, 4]; let r = stub_settle(a, k); return match r {{ Result::Ok(v) => v, Result::Err(e) => 0 - 1 }}; }}\n\
fn w_after(k: i32) -> Array<i32> {{ var a: Array<i32> = [3, 4]; let r = settle(a, k); return a; }}"
    ));
    for (k, tag, payload) in [(5, 0usize, Value::I32(5)), (-2, 1, Value::U32(7))] {
        let bufs = vec![Buf::I32(vec![3, 4])];
        let (o, after) = host(&program, "settle", &[Arg::Mut(0), Arg::I32(k)], &bufs, 1000);
        let Ok(Value::Variant(v)) = &o.result else {
            panic!("{k}: {:?}", o.result)
        };
        assert_eq!(*v.borrow(), (tag, Some(payload)), "{k}: exact variant");
        assert_eq!(
            after,
            vec![Buf::I32(vec![k, 4])],
            "{k}: committed, Err included"
        );
        let in_source =
            eval::named::observe_owned_numeric_named(&program, "w_after", &[Value::I32(k)], 1000);
        let Ok(Value::ArrayI32(a)) = &in_source.result else {
            panic!("{in_source:?}")
        };
        assert_eq!(*a.borrow(), vec![k, 4]);
        let (s, _) = host(
            &program,
            "stub_settle",
            &[Arg::Mut(0), Arg::I32(k)],
            &bufs,
            1000,
        );
        let w = eval::named::observe_owned_numeric_named(&program, "w_tag", &[Value::I32(k)], 1000);
        let ws =
            eval::named::observe_owned_numeric_named(&program, "w_stub", &[Value::I32(k)], 1000);
        assert_eq!(w.result, Ok(Value::I32(if k < 0 { -1 } else { k })));
        assert_eq!(
            i128::from(o.steps_used) - i128::from(s.steps_used),
            i128::from(w.steps_used) - i128::from(ws.steps_used),
            "{k}: step delta"
        );
        assert!(o.cleanup_trace.is_empty());
    }
    let (o, after) = host(&program, "flag", &[Arg::Mut(0)], &[Buf::U32(vec![0])], 1000);
    let Ok(Value::Variant(v)) = &o.result else {
        panic!("{:?}", o.result)
    };
    assert_eq!(*v.borrow(), (0, Some(Value::Bool(true))));
    assert_eq!(after, vec![Buf::U32(vec![1])]);
}

#[test]
fn nan_payloads_and_signed_zero_are_committed_or_kept_bit_exactly() {
    let program = checked(SOURCE);
    let payloads = [0x7fa0_0001, 0xffc1_2345, 0x8000_0000];
    let original = [0x7f80_0001, 0x8000_0000, 0x7fc0_0000];
    let bits = |v: &[u32]| v.iter().map(|b| F32Bits::from_bits(*b)).collect::<Vec<_>>();
    let bufs = vec![Buf::F32(bits(&original)), Buf::F32(bits(&payloads))];
    let (o, after) = host(
        &program,
        "fcopy",
        &[Arg::Mut(0), Arg::Read(1), Arg::U32(2)],
        &bufs,
        1000,
    );
    assert!(o.result.is_ok());
    assert_eq!(
        after,
        vec![Buf::F32(bits(&payloads)), Buf::F32(bits(&payloads))]
    );
    let (o, after) = host(
        &program,
        "fcopy",
        &[Arg::Mut(0), Arg::Read(1), Arg::U32(3)],
        &bufs,
        1000,
    );
    assert!(matches!(o.result, Err(EvalError::Trap { .. })));
    assert_eq!(after, bufs, "restored bit-exactly after writes then a trap");
}

/// Function, arguments, buffers, fuel and the expected refusal code.
type Refusal<'a> = (&'a str, Vec<Arg>, &'a Vec<Buf>, u64, &'a str);

#[test]
fn refusals_take_zero_steps_and_leave_buffers_byte_identical() {
    let program = checked(SOURCE);
    let base = vec![Buf::I32(vec![1, 2, 3])];
    let over = vec![Buf::I32(vec![0; invar::array_limits::MAX_ARRAY_LEN + 1])];
    let cases: Vec<Refusal<'_>> = vec![
        (
            "store2",
            vec![Arg::Mut(0), Arg::U32(0), Arg::I32(1)],
            &base,
            0,
            "INVALID_FUEL",
        ),
        ("eat", vec![Arg::Mut(0)], &base, 10, "HOST_TAKE_UNAVAILABLE"),
        ("wide", vec![Arg::Mut(0)], &base, 10, "NATIVE_ABI"),
        (
            "store2",
            vec![Arg::Read(0), Arg::U32(0), Arg::I32(1)],
            &base,
            10,
            "NATIVE_ABI",
        ),
        ("inner_read", vec![Arg::Mut(0)], &base, 10, "NATIVE_ABI"),
        (
            "store2",
            vec![Arg::Mut(0), Arg::U32(0)],
            &base,
            10,
            "NATIVE_ABI",
        ),
        (
            "store2",
            vec![Arg::Mut(0), Arg::U32(0), Arg::I32(1)],
            &over,
            10,
            "ARRAY_SIZE_LIMIT",
        ),
    ];
    for (name, args, bufs, fuel, expected) in cases {
        let (o, after) = host(&program, name, &args, bufs, fuel);
        assert_eq!(
            (code(&o), o.steps_used, o.cleanup_trace.len()),
            (expected, 0, 0),
            "{name} {args:?}"
        );
        assert_eq!(&after, bufs, "{name}: byte-identical");
    }
    // The read-only API keeps its approved refusals for mut, take and Result
    // results; `peek` is read-only and returns a scalar Result.
    let xs = vec![1, 2];
    for name in ["store2", "eat", "settle", "peek"] {
        let args: Vec<RA<'_>> = match name {
            "store2" => vec![RA::Array(R::I32(&xs)), RA::U32(0), RA::I32(1)],
            "settle" => vec![RA::Array(R::I32(&xs)), RA::I32(1)],
            _ => vec![RA::Array(R::I32(&xs))],
        };
        let o = observe_host_read_call(&program, name, &args, 10);
        assert_eq!((code(&o), o.steps_used), ("NATIVE_ABI", 0), "{name}");
    }
    let (o, _) = host(
        &program,
        "peek",
        &[Arg::Read(0)],
        &[Buf::I32(vec![4, 5])],
        10,
    );
    let Ok(Value::Variant(v)) = &o.result else {
        panic!("{:?}", o.result)
    };
    assert_eq!(*v.borrow(), (0, Some(Value::I32(4))));
}

#[test]
fn live_union_counts_mut_buffers_once_with_reads() {
    let program = checked(SOURCE);
    // 1M mut + 999,999 read: the callee's temporary fits exactly at 2M.
    let bufs = vec![Buf::I32(vec![1; 1_000_000]), Buf::I32(vec![1; 999_999])];
    let (o, after) = host(
        &program,
        "grow_after",
        &[Arg::Mut(0), Arg::Read(1)],
        &bufs,
        100,
    );
    assert_eq!(o.result, Ok(Value::U32(1_000_000)));
    let Buf::I32(v) = &after[0] else {
        unreachable!()
    };
    assert_eq!((v[0], v[1]), (7, 1));
}
