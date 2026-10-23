#![forbid(unsafe_code)]
//! Read-only host array arguments (HA-1). The independent comparator is the
//! unchanged in-language call: a source wrapper builds one owner per distinct
//! host buffer (exact aliases share one owner, as an in-source call passing one
//! owner twice) and passes it as read loans. Steps are compared by differencing
//! against a stub with the same signature and the same outcome class (a stub
//! that returns for normal results, one that traps for failures, since a
//! failing call never runs the wrapper's post-call steps):
//! `S_host(f) - S_host(stub) == S_wrap(f) - S_wrap(stub)`.
use invar::{
    check,
    eval::{
        self, EvalError, Observation, Value,
        host_read::{HostReadArgument as A, HostReadArray as H, observe_host_read_call},
    },
    front,
    numeric_contract::F32Bits,
};

const SOURCE: &str = "\
fn total_f32(xs: read Array<f32>) -> f32 { var s = 0.0; for i in 0u32..len(xs) { s = s + xs[i]; } return s; }
fn total_i32(xs: read Array<i32>) -> i32 { var s = 0; for i in 0u32..len(xs) { s = s + xs[i]; } return s; }
fn outer(xs: read Array<f32>) -> f32 { return total_f32(xs); }
fn copy_out(xs: read Array<f32>) -> Array<f32> { var b = filled_f32(len(xs), 0.0); for i in 0u32..len(xs) { b[i] = xs[i]; } return b; }
fn dot(a: read Array<u32>, b: read Array<u32>) -> u32 { var s = 0u32; for i in 0u32..len(a) { s = s + a[i] * b[i]; } return s; }
fn pick(xs: read Array<i32>, at: u32) -> i32 { let t = filled_i32(2u32, 5); return xs[at] + t[0u32]; }
fn scaled(xs: read Array<f32>, k: f32, keep: Bool) -> Array<f32> { var b = filled_f32(len(xs), 0.0); for i in 0u32..len(xs) { if keep { b[i] = xs[i] * k; } } return b; }
fn lens(a: read Array<i32>, b: read Array<i32>) -> u32 { return len(a) + len(b); }
fn grow2(a: read Array<i32>, b: read Array<i32>) -> u32 { let t = filled_u32(1u32, 0u32); return len(a) + len(b) + len(t); }
fn bump(xs: mut Array<i32>) -> i32 { xs[0u32] = 1; return 0; }
fn eat(xs: take Array<i32>) -> u32 { return len(xs); }
fn nine(a: i32, b: i32, c: i32, d: i32, e: i32, f: i32, g: i32, h: i32, i: read Array<i32>) -> i32 { return 0; }
fn main() -> i32 { return 7; }";

fn checked(source: &str) -> check::CheckedProgram {
    let c = check::check(front::parse(source.as_bytes()).unwrap(), false).unwrap();
    invar::ownership::check(&c).unwrap();
    c
}
fn bits(values: &[u32]) -> Vec<F32Bits> {
    values.iter().map(|b| F32Bits::from_bits(*b)).collect()
}
fn floats(values: &[f32]) -> Vec<F32Bits> {
    values
        .iter()
        .map(|x| F32Bits::from_bits(x.to_bits()))
        .collect()
}
fn code(o: &Observation) -> &'static str {
    match &o.result {
        Err(EvalError::Tool { code, .. }) => code,
        x => panic!("expected tool refusal, got {x:?}"),
    }
}

#[test]
fn read_arguments_observe_exact_bits_and_leave_buffers_untouched() {
    let p = checked(SOURCE);
    let payloads = bits(&[0x7fa0_0001, 0xffc1_2345, 0x8000_0000, 1, 0x7f80_0000]);
    let o = observe_host_read_call(&p, "copy_out", &[A::Array(H::F32(&payloads))], 1000);
    let Ok(Value::ArrayF32(copy)) = &o.result else {
        panic!("{o:?}")
    };
    assert_eq!(
        *copy.borrow(),
        payloads,
        "fresh owner holds the exact encodings"
    );
    assert!(
        o.cleanup_trace.is_empty(),
        "the result owner escapes; the borrowed input is never cleaned"
    );

    let xs = vec![5, -3, 10];
    let o = observe_host_read_call(&p, "total_i32", &[A::Array(H::I32(&xs))], 1000);
    assert_eq!(o.result, Ok(Value::I32(12)));
    assert_eq!(xs, vec![5, -3, 10]);

    // Nested reborrow through another read parameter.
    let fs = floats(&[0.5, -1.25, 3.0]);
    let o = observe_host_read_call(&p, "outer", &[A::Array(H::F32(&fs))], 1000);
    assert_eq!(
        o.result,
        Ok(Value::F32(F32Bits::from_bits(2.25f32.to_bits())))
    );

    // The same host buffer may be passed to two read parameters.
    let us = vec![1u32, 2, 3];
    let o = observe_host_read_call(
        &p,
        "dot",
        &[A::Array(H::U32(&us)), A::Array(H::U32(&us))],
        1000,
    );
    assert_eq!(o.result, Ok(Value::U32(14)));

    // Mixed scalar and array arguments, internal cleanup of a temporary.
    let o = observe_host_read_call(&p, "pick", &[A::Array(H::I32(&xs)), A::U32(2)], 1000);
    assert_eq!(o.result, Ok(Value::I32(15)));
    assert_eq!(
        o.cleanup_trace
            .iter()
            .map(|c| (c.name.as_str(), c.length))
            .collect::<Vec<_>>(),
        vec![("t", 2)]
    );

    // Empty buffers.
    let o = observe_host_read_call(&p, "total_f32", &[A::Array(H::F32(&[]))], 1000);
    assert_eq!(o.result, Ok(Value::F32(F32Bits::from_bits(0))));
}

#[test]
fn traps_limits_and_fuel_prefixes_report_no_cleanup_after_failure() {
    let p = checked(SOURCE);
    let xs = vec![1, 2];
    // An index trap after a temporary was created: no cleanup is recorded.
    let o = observe_host_read_call(&p, "pick", &[A::Array(H::I32(&xs)), A::U32(2)], 1000);
    let Err(EvalError::Trap { span, .. }) = &o.result else {
        panic!("{o:?}")
    };
    assert_eq!(&SOURCE[span.start..span.end], "xs[at]");
    assert!(o.cleanup_trace.is_empty());
    // Every fuel prefix: exhaustion strictly below the step count, then completion.
    let fs = floats(&[0.25, -4.0, 8.0]);
    let args = [
        A::Array(H::F32(&fs)),
        A::F32(F32Bits::from_bits(2.0f32.to_bits())),
        A::Bool(true),
    ];
    let done = observe_host_read_call(&p, "scaled", &args, 100_000);
    assert!(done.result.is_ok());
    for fuel in 1..=done.steps_used {
        let o = observe_host_read_call(&p, "scaled", &args, fuel);
        if fuel < done.steps_used {
            assert_eq!(
                o.result,
                Err(EvalError::EvaluationLimit {
                    fuel_limit: fuel,
                    steps_used: fuel
                })
            );
            assert!(o.cleanup_trace.is_empty());
        } else {
            assert_eq!(o.result, done.result);
            assert_eq!(o.cleanup_trace, done.cleanup_trace);
        }
    }
    assert_eq!(fs, floats(&[0.25, -4.0, 8.0]));
}

/// Literal text for values expressible exactly in source.
fn literal(a: &H<'_>) -> (String, String) {
    match a {
        H::I32(v) if v.len() > 16 && v.iter().all(|x| *x == v[0]) => (
            "Array<i32>".into(),
            format!("filled_i32({}u32, {})", v.len(), v[0]),
        ),
        H::I32([]) => ("Array<i32>".into(), "filled_i32(0u32, 0)".into()),
        H::U32([]) => ("Array<u32>".into(), "filled_u32(0u32, 0u32)".into()),
        H::F32([]) | H::Float32([]) => ("Array<f32>".into(), "filled_f32(0u32, 0.0)".into()),
        H::I32(v) => (
            "Array<i32>".into(),
            format!(
                "[{}]",
                v.iter().map(i32::to_string).collect::<Vec<_>>().join(",")
            ),
        ),
        H::U32(v) => (
            "Array<u32>".into(),
            format!(
                "[{}]",
                v.iter()
                    .map(|x| format!("{x}u32"))
                    .collect::<Vec<_>>()
                    .join(",")
            ),
        ),
        H::Float32(v) => (
            "Array<f32>".into(),
            format!(
                "[{}]",
                v.iter()
                    .map(|x| format!("{x:?}"))
                    .collect::<Vec<_>>()
                    .join(",")
            ),
        ),
        H::F32(v) => (
            "Array<f32>".into(),
            format!(
                "[{}]",
                v.iter()
                    .map(|x| format!("{:?}", f32::from_bits(x.to_bits())))
                    .collect::<Vec<_>>()
                    .join(",")
            ),
        ),
        // The reference corpus holds only i32, u32 and f32 buffers; f64 and Bool have their own tests
        // (src/eval/host_read/tests.rs).
        H::Float64(_) | H::Bool(_) => unreachable!("no f64 or Bool buffer in this corpus"),
    }
}

/// Whether two host buffers are the same buffer: same type, address, length.
fn exact_alias(a: &H<'_>, b: &H<'_>) -> bool {
    match (a, b) {
        (H::I32(x), H::I32(y)) => std::ptr::eq(*x, *y),
        (H::U32(x), H::U32(y)) => std::ptr::eq(*x, *y),
        (H::F32(x), H::F32(y)) => std::ptr::eq(*x, *y),
        (H::Float32(x), H::Float32(y)) => std::ptr::eq(*x, *y),
        _ => false,
    }
}

/// `fn name(params) -> result` text from the source, and two stubs with that
/// signature: `stub_name` returns, `trap_name` traps on its first statement;
/// `k_stub_name`/`k_trap_name` are parameterless twins with the same bodies.
fn stub(name: &str) -> (String, String) {
    let line = SOURCE
        .lines()
        .find(|l| l.starts_with(&format!("fn {name}(")))
        .unwrap();
    let signature = &line[..line.find(" {").unwrap()];
    let result = &signature[signature.rfind("-> ").unwrap() + 3..];
    let body = match result {
        "i32" => "0",
        "u32" => "0u32",
        "f32" => "0.0",
        "Array<f32>" => "filled_f32(0u32, 0.0)",
        other => panic!("{other}"),
    };
    let renamed = |prefix: &str| {
        signature.replacen(&format!("fn {name}("), &format!("fn {prefix}_{name}("), 1)
    };
    (
        result.to_owned(),
        format!(
            "{} {{ let s = {body}; return s; }}\n{} {{ let z = 0u32; let t = 1u32 / z; let s = {body}; return s; }}\n\
fn k_stub_{name}() -> {result} {{ let s = {body}; return s; }}\n\
fn k_trap_{name}() -> {result} {{ let z = 0u32; let t = 1u32 / z; let s = {body}; return s; }}",
            renamed("stub"),
            renamed("trap")
        ),
    )
}

/// Host call against the in-language single-owner wrapper: result, cleanup
/// and step delta.
fn compare(name: &str, args: &[A<'_>], fuel: u64, label: &str) -> Observation {
    let (result_ty, stub_fn) = stub(name);
    let mut decls = String::new();
    let mut call = Vec::new();
    let mut owners = 0;
    for (i, arg) in args.iter().enumerate() {
        match arg {
            A::Array(a) => {
                let shared = args[..i]
                    .iter()
                    .position(|earlier| matches!(earlier, A::Array(e) if exact_alias(e, a)));
                if let Some(j) = shared {
                    call.push(format!("a{j}"));
                } else {
                    let (ty, lit) = literal(a);
                    decls.push_str(&format!("let a{i}: {ty} = {lit}; "));
                    call.push(format!("a{i}"));
                    owners += 1;
                }
            }
            A::U32(x) => call.push(format!("{x}u32")),
            A::F32(x) => call.push(format!("{:?}", f32::from_bits(x.to_bits()))),
            A::Bool(x) => call.push(x.to_string()),
            x => panic!("{x:?}"),
        }
    }
    let call = call.join(", ");
    let program = checked(&format!(
        "{SOURCE}\n{stub_fn}\nfn w_result() -> {result_ty} {{ {decls}let r = {name}({call}); return r; }}\nfn w_stub() -> {result_ty} {{ {decls}let r = stub_{name}({call}); return r; }}\nfn w_trap() -> {result_ty} {{ {decls}let r = trap_{name}({call}); return r; }}"
    ));
    let host = observe_host_read_call(&program, name, args, fuel);
    let wrapped = eval::named::observe_owned_numeric_named(&program, "w_result", &[], fuel);
    let (prefix, stub_wrapper) = if host.result.is_ok() {
        ("stub", "w_stub")
    } else {
        ("trap", "w_trap")
    };
    let host_stub = observe_host_read_call(&program, &format!("{prefix}_{name}"), args, fuel);
    let wrapped_stub = eval::named::observe_owned_numeric_named(&program, stub_wrapper, &[], fuel);
    assert_eq!(
        host_stub.result.is_ok(),
        host.result.is_ok(),
        "{label}: stub outcome"
    );
    assert_eq!(
        host_stub.result, wrapped_stub.result,
        "{label}: stub result"
    );
    // Anchor the absolute count, which differencing alone would cancel: the
    // host stub costs exactly its body, as a parameterless twin does.
    let anchor = eval::named::observe_owned_numeric_named(
        &program,
        &format!("k_{prefix}_{name}"),
        &[],
        fuel,
    );
    assert_eq!(
        (anchor.result.is_ok(), anchor.steps_used),
        (host.result.is_ok(), host_stub.steps_used),
        "{label}: absolute stub steps"
    );
    assert_eq!(host.result, wrapped.result, "{label}: result");
    assert_eq!(
        i128::from(host.steps_used) - i128::from(host_stub.steps_used),
        i128::from(wrapped.steps_used) - i128::from(wrapped_stub.steps_used),
        "{label}: step delta"
    );
    let callee = &host.cleanup_trace;
    if host.result.is_ok() {
        assert_eq!(
            &wrapped.cleanup_trace[..callee.len()],
            callee.as_slice(),
            "{label}: cleanup"
        );
        let rest = &wrapped.cleanup_trace[callee.len()..];
        assert!(
            rest.iter()
                .all(|c| c.name == "r" || c.name.starts_with('a')),
            "{label}"
        );
        assert_eq!(
            rest.iter().filter(|c| c.name.starts_with('a')).count(),
            owners,
            "{label}: owners"
        );
    } else {
        assert_eq!(
            &wrapped.cleanup_trace, callee,
            "{label}: no cleanup after a trap"
        );
    }
    host
}

#[test]
fn host_read_call_equals_in_language_read_loan() {
    let fs = floats(&[0.5, -1.25, 3.0]);
    let xs = vec![7, -8, 9];
    let us = vec![3u32, 4, 5];
    let empty: Vec<i32> = Vec::new();
    let two = F32Bits::from_bits((-2.0f32).to_bits());
    let cases: Vec<(&str, Vec<A<'_>>)> = vec![
        ("total_f32", vec![A::Array(H::F32(&fs))]),
        ("total_i32", vec![A::Array(H::I32(&empty))]),
        ("outer", vec![A::Array(H::F32(&fs))]),
        ("copy_out", vec![A::Array(H::F32(&fs))]),
        // Exact alias: one owner in the wrapper.
        ("dot", vec![A::Array(H::U32(&us)), A::Array(H::U32(&us))]),
        ("lens", vec![A::Array(H::I32(&xs)), A::Array(H::I32(&xs))]),
        // Partial overlap and a same-address prefix: separate owners.
        (
            "dot",
            vec![A::Array(H::U32(&us[..2])), A::Array(H::U32(&us[1..]))],
        ),
        (
            "lens",
            vec![A::Array(H::I32(&xs)), A::Array(H::I32(&xs[..1]))],
        ),
        // Empty slices at one address never alias.
        (
            "lens",
            vec![A::Array(H::I32(&xs[..0])), A::Array(H::I32(&xs[..0]))],
        ),
        ("pick", vec![A::Array(H::I32(&xs)), A::U32(1)]),
        ("pick", vec![A::Array(H::I32(&xs)), A::U32(3)]),
        (
            "scaled",
            vec![A::Array(H::F32(&fs)), A::F32(two), A::Bool(false)],
        ),
        (
            "scaled",
            vec![A::Array(H::F32(&fs)), A::F32(two), A::Bool(true)],
        ),
    ];
    for (index, (name, args)) in cases.iter().enumerate() {
        compare(name, args, 100_000, &format!("case {index} {name}"));
    }
    eprintln!(
        "read-only host reference: {} calls equal to in-language read loans",
        cases.len()
    );
}

#[test]
fn refusals_take_zero_steps_and_mut_take_signatures_stay_unadmitted() {
    let p = checked(SOURCE);
    let xs = vec![1, 2];
    let us = vec![1u32];
    let nine: Vec<A<'_>> = (0..8)
        .map(|_| A::I32(0))
        .chain([A::Array(H::I32(&xs))])
        .collect();
    let cases: Vec<(&str, Vec<A<'_>>, u64, &str)> = vec![
        ("total_i32", vec![A::Array(H::I32(&xs))], 0, "INVALID_FUEL"),
        ("missing", vec![], 10, "NATIVE_ABI"),
        ("total_i32", vec![A::Array(H::U32(&us))], 10, "NATIVE_ABI"),
        ("total_i32", vec![A::I32(1)], 10, "NATIVE_ABI"),
        ("total_i32", vec![], 10, "NATIVE_ABI"),
        (
            "total_i32",
            vec![A::Array(H::I32(&xs)), A::Array(H::I32(&xs))],
            10,
            "NATIVE_ABI",
        ),
        (
            "pick",
            vec![A::Array(H::I32(&xs)), A::I32(0)],
            10,
            "NATIVE_ABI",
        ),
        ("bump", vec![A::Array(H::I32(&xs))], 10, "NATIVE_ABI"),
        ("eat", vec![A::Array(H::I32(&xs))], 10, "NATIVE_ABI"),
        ("nine", nine, 10, "NATIVE_ABI"),
    ];
    for (name, args, fuel, expected) in cases {
        let o = observe_host_read_call(&p, name, &args, fuel);
        assert_eq!(code(&o), expected, "{name}");
        assert_eq!(o.steps_used, 0, "{name}");
        assert!(o.cleanup_trace.is_empty(), "{name}");
    }
    assert_eq!(xs, vec![1, 2]);
    assert_eq!(
        observe_host_read_call(&p, "main", &[], 10).result,
        Ok(Value::I32(7))
    );
}

#[test]
fn host_arrays_count_once_against_element_limits() {
    let p = checked(SOURCE);
    let million = vec![1; 1_000_000];
    let other = vec![1; 1_000_000];
    let almost = vec![1; 999_999];
    let over = vec![1; invar::array_limits::MAX_ARRAY_LEN + 1];
    // The raised per-array limit is refused before any step.
    let o = observe_host_read_call(
        &p,
        "lens",
        &[A::Array(H::I32(&over)), A::Array(H::I32(&[]))],
        100,
    );
    assert_eq!((code(&o), o.steps_used), ("ARRAY_SIZE_LIMIT", 0));
    // One buffer passed twice counts once, as one in-source owner passed twice.
    let o = compare(
        "grow2",
        &[A::Array(H::I32(&million)), A::Array(H::I32(&million))],
        100,
        "exact alias",
    );
    assert_eq!(o.result, Ok(Value::U32(2_000_001)));
    // Distinct buffers count twice; this former 2M boundary now fits.
    let o = compare(
        "grow2",
        &[A::Array(H::I32(&million)), A::Array(H::I32(&other))],
        100,
        "distinct",
    );
    assert_eq!(o.result, Ok(Value::U32(2_000_001)));
    assert!(o.steps_used > 0);
    let o = compare(
        "grow2",
        &[A::Array(H::I32(&million)), A::Array(H::I32(&almost))],
        100,
        "distinct below",
    );
    assert_eq!(o.result, Ok(Value::U32(2_000_000)));
    let o = compare(
        "lens",
        &[A::Array(H::I32(&million)), A::Array(H::I32(&other))],
        100,
        "distinct at limit",
    );
    assert_eq!(o.result, Ok(Value::U32(2_000_000)));
    // A partial overlap counts its union: 1.6M live elements, so the
    // temporary fits although the per-argument sum is 2M. There is no
    // in-source partially overlapping owner; this is the HA-1 definition.
    let backing = vec![1; 1_600_000];
    let o = observe_host_read_call(
        &p,
        "grow2",
        &[
            A::Array(H::I32(&backing[..1_000_000])),
            A::Array(H::I32(&backing[600_000..])),
        ],
        100,
    );
    assert_eq!(o.result, Ok(Value::U32(2_000_001)));
    // The union of distinct buffers is not reduced by an alias of one of them.
    let o = observe_host_read_call(
        &p,
        "grow2",
        &[A::Array(H::I32(&million)), A::Array(H::I32(&other))],
        100,
    );
    assert_eq!(o.result, Ok(Value::U32(2_000_001)));
    assert_eq!(million, vec![1; 1_000_000]);
}
