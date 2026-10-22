//! v0.2 f64math: executable f64 and invar-math-cr-v1 from ordinary source,
//! with identical outcomes on the reference evaluator, MIR and native.
#![forbid(unsafe_code)]
use invar::{check, eval, front, mir, mir_eval};

fn checked(source: &str) -> check::CheckedProgram {
    let checked = check::check(front::parse(source.as_bytes()).unwrap(), true)
        .unwrap_or_else(|error| panic!("{error:?}: {source}"));
    invar::ownership::check(&checked).unwrap_or_else(|error| panic!("{error:?}: {source}"));
    checked
}

/// Every engine reaches the same result, trap, fuel and cleanup at every fuel.
fn qualify(source: &str, expected: Result<i32, (&str, &str)>) {
    let checked = checked(source);
    let program = mir::lower(&checked).unwrap();
    let complete = eval::observe(&checked, 100_000);
    match (&complete.result, expected) {
        (Ok(eval::Value::I32(value)), Ok(want)) => assert_eq!(*value, want, "{source}"),
        (Err(eval::EvalError::Trap { category, span }), Err((code, text))) => {
            assert_eq!(format!("{category:?}"), code, "{source}");
            assert_eq!(&source[span.start..span.end], text, "{source}");
        }
        (other, _) => panic!("{other:?}: {source}"),
    }
    for fuel in [
        0,
        complete.steps_used / 2,
        complete.steps_used,
        complete.steps_used + 1,
    ] {
        let reference = eval::observe(&checked, fuel);
        let actual = mir_eval::observe(&program, fuel);
        assert_eq!(actual.result, reference.result, "fuel={fuel}: {source}");
        assert_eq!(actual.steps_used, reference.steps_used, "{source}");
        assert_eq!(actual.cleanup_trace, reference.cleanup_trace, "{source}");
    }
    #[cfg(feature = "native")]
    {
        let (status, doc) = invar::native::run_program(&program, complete.steps_used + 1).unwrap();
        assert_eq!(doc["steps_used"], complete.steps_used, "{doc}: {source}");
        match &complete.result {
            Ok(eval::Value::I32(value)) => {
                assert_eq!(status, 0, "{doc}");
                assert_eq!(doc["result"]["value"], *value, "{source}");
            }
            Err(eval::EvalError::Trap { span, .. }) => {
                assert_eq!(status, 3, "{doc}");
                assert_eq!(doc["trap"]["category"], "INTEGER_OVERFLOW", "{source}");
                assert_eq!(doc["trap"]["span"]["start"], span.start, "{source}");
            }
            other => panic!("{other:?}"),
        }
    }
}

#[test]
fn v02_f64_strict_arithmetic_literals_and_predicates() {
    for source in [
        "fn main() -> i32 { let a: f64 = 0.1; let b: f64 = 0.2; if a + b == 0.30000000000000004 and a + b != 0.3 { return 7; } return 0; }",
        "fn main() -> i32 { let x = 1.5f64; let y: f64 = -x * 2.0; if y == -3.0 and 1.0 / (y - y) > 0.0 { return 7; } return 0; }",
        "fn main() -> i32 { let n: f64 = 0.0 / 0.0; let z: f64 = -0.0; if n != n and not (n < 1.0) and z == 0.0 and 1.0 / z < 0.0 { return 7; } return 0; }",
        "fn main() -> i32 { let a = 1.0000000000000002f64; let b = 0.9999999999999998f64; if a * b - 1.0 == 0.0 and fma(a, b, -1.0) < 0.0 { return 7; } return 0; }",
        "fn main() -> i32 { let big = 9007199254740992.0f64; if big + 1.0 == big and f64_from_u32(4294967295u32) == 4294967295.0 { return 7; } return 0; }",
        "fn main() -> i32 { let single = 0.1; let double = 0.1f64; if f64_from_f32(single) != double and f32_from_f64(double) == single { return 7; } return 0; }",
    ] {
        qualify(source, Ok(7));
    }
}

#[test]
fn v02_float_to_int_truncates_toward_zero_and_traps_in_source_order() {
    qualify(
        "fn main() -> i32 { if i32_from_f64(-2.9f64) == -2 and u32_from_f64(4294967295.5f64) == 4294967295u32 and i32_from_f32(-0.5) == 0 and u32_from_f32(3.75) == 3u32 { return 7; } return 0; }",
        Ok(7),
    );
    for (source, text) in [
        (
            "fn main() -> i32 { let x: f64 = 2147483648.0; return i32_from_f64(x); }",
            "i32_from_f64(x)",
        ),
        (
            "fn main() -> i32 { let n: f64 = 0.0 / 0.0; let k = u32_from_f64(n); return 0; }",
            "u32_from_f64(n)",
        ),
        (
            "fn main() -> i32 { let x: f32 = -1.0; let k = u32_from_f32(x); return 0; }",
            "u32_from_f32(x)",
        ),
    ] {
        qualify(source, Err(("IntegerOverflow", text)));
    }
}

#[test]
fn v02_f64_arrays_records_variants_results_and_generics() {
    for source in [
        "fn scale(a: mut Array<f64>, k: f64) -> Unit { for i in 0u32..len(a) { a[i] = a[i] * k; } } fn main() -> i32 { var a: Array<f64> = [1.0, 2.0, 3.0]; scale(a, 0.5); let b = filled_f64(2u32, 0.25); if a[2u32] == 1.5 and b[1u32] == 0.25 and len(b) == 2u32 { return 7; } return 0; }",
        "struct P { x: f64, y: f64 } fn norm2(p: read P) -> f64 { return p.x * p.x + p.y * p.y; } fn main() -> i32 { let p = P { x: 3.0, y: 4.0 }; if norm2(p) == 25.0 { return 7; } return 0; }",
        "enum Reading { Missing, Value(f64) } fn get(r: read Reading) -> f64 { return match r { Reading::Missing => -1.0, Reading::Value(v) => v }; } fn main() -> i32 { let a = Reading::Value(2.5); let b = Reading::Missing(); if get(a) == 2.5 and get(b) == -1.0 { return 7; } return 0; }",
        "fn safe(x: f64) -> Result<f64, i32> { if x < 0.0 { return Result::Err(1); } return Result::Ok(sqrt(x)); } fn main() -> i32 { let r = safe(16.0); let v: f64 = match r { Result::Ok(root) => root, Result::Err(e) => 0.0 }; if v == 4.0 { return 7; } return 0; }",
        "fn twice<T: Value>(x: T) -> T { return x; } fn main() -> i32 { let x: f64 = twice(2.0f64); if x == 2.0 { return 7; } return 0; }",
    ] {
        qualify(source, Ok(7));
    }
}

#[test]
fn v02_math_library_is_correctly_rounded_on_every_cpu_engine() {
    for source in [
        "fn main() -> i32 { if sin(1.0f64) == 0.8414709848078965 and cos(1.0f64) == 0.5403023058681398 and exp(1.0f64) == 2.718281828459045 and log(2.0f64) == 0.6931471805599453 { return 7; } return 0; }",
        "fn main() -> i32 { let q = atan2(1.0f64, 1.0); if q * 4.0 == 3.141592653589793 and hypot(3.0f64, 4.0) == 5.0 and cbrt(27.0f64) == 3.0 and pow(2.0f64, 10.0) == 1024.0 { return 7; } return 0; }",
        "fn main() -> i32 { if atan(1.0) == 0.7853982 and exp2(3.0) == 8.0 and log10(1000.0) == 3.0 and tanh(0.0) == 0.0 { return 7; } return 0; }",
        "fn main() -> i32 { let x: f64 = -2.5; if floor(x) == -3.0 and ceil(x) == -2.0 and trunc(x) == -2.0 and round(x) == -3.0 and abs(x) == 2.5 and copysign(1.0f64, x) == -1.0 { return 7; } return 0; }",
        "fn main() -> i32 { let n: f64 = 0.0 / 0.0; if min(n, 1.0f64) == 1.0 and max(-0.0f64, 0.0) == 0.0 and 1.0 / min(-0.0f64, 0.0) < 0.0 and log(-1.0f64) != log(-1.0f64) { return 7; } return 0; }",
        "fn main() -> i32 { if asin(1.0f64) * 2.0 == acos(-1.0f64) and atan(1.0f64) == 0.7853981633974483 and sinh(0.0f64) == 0.0 and cosh(0.0f64) == 1.0 and expm1(0.0f64) == 0.0 and log1p(0.0f64) == 0.0 and log2(8.0f64) == 3.0 and tan(0.0f64) == 0.0 { return 7; } return 0; }",
    ] {
        qualify(source, Ok(7));
    }
}

#[test]
fn v02_numeric_contracts_name_trig_for_f32_sin_cos_and_sqrt() {
    // C4 interim: f32 sin, cos and sqrt are invar-trig-f32-v1; f64 and every
    // other library function are invar-math-cr-v1.
    for (source, expected) in [
        (
            "fn main() -> i32 { let x: f32 = sin(1.0) + cos(1.0) + sqrt(2.0); return 0; }",
            serde_json::json!({"trig": "invar-trig-f32-v1"}),
        ),
        (
            "fn main() -> i32 { let x: f32 = sin(1.0) + exp(1.0); return 0; }",
            serde_json::json!({"math": "invar-math-cr-v1", "trig": "invar-trig-f32-v1"}),
        ),
        (
            "fn main() -> i32 { let x: f64 = sin(1.0f64); return 0; }",
            serde_json::json!({"binary64": "strict-f64-v1", "math": "invar-math-cr-v1"}),
        ),
    ] {
        let checked = check::check(front::parse(source.as_bytes()).unwrap(), true).unwrap();
        let program = invar::mir::lower(&checked).unwrap();
        assert_eq!(invar::math::contracts(&program), Some(expected), "{source}");
    }
}

#[test]
fn v02_f64_rejects_implicit_conversion_mixing_and_float_remainder() {
    for source in [
        "fn main() -> i32 { let x: f64 = 1; return 0; }",
        "fn main() -> i32 { let a: f64 = 1.0; let b: f32 = 1.0; let c = a + b; return 0; }",
        "fn main() -> i32 { let x = 1.0f64 % 2.0; return 0; }",
        "fn main() -> i32 { let x: f32 = 1.0f64; return 0; }",
        "fn main() -> i32 { let x = sin(1.0f64, 2.0); return 0; }",
        "fn main() -> i32 { let x = sin(1); return 0; }",
        "fn main() -> i32 { let x = i32_from_f64(1.0f32); return 0; }",
        "fn main() -> i32 { let a: Array<f64> = [1.0]; let v = view(a, 0u32, 1u32); return 0; }",
        "fn sin(x: f64) -> f64 { return x; } fn main() -> i32 { return 0; }",
    ] {
        let rejected = front::parse(source.as_bytes())
            .and_then(|parsed| check::check(parsed, true))
            .map(|checked| invar::ownership::check(&checked).is_err())
            .unwrap_or(true);
        assert!(rejected, "{source}");
    }
    for bad in ["1.0f16", "1.0x", "1.0f"] {
        let source = format!("fn main() -> i32 {{ let x = {bad}; return 0; }}");
        assert!(front::parse(source.as_bytes()).is_err(), "{source}");
    }
}

#[test]
fn v02_variables_may_share_library_function_names() {
    for source in [
        // Main's GNET-05 frontier corpus counts BFS rounds in `var round`.
        "fn main() -> i32 { var round: u32 = 0u32; while round < 3u32 { round = round + 1u32; } if round == 3u32 { return 7; } return 0; }",
        "fn main() -> i32 { let min: i32 = 2; let max: i32 = 5; let abs: f64 = 1.5; if min + max == 7 and abs == 1.5 { return 7; } return 0; }",
        // A call still names the library function with a same-named variable in scope.
        "fn main() -> i32 { let round: f64 = 0.25; let x: f64 = 2.5; if round(x) == 3.0 and round == 0.25 { return 7; } return 0; }",
        "fn main() -> i32 { let min: f64 = 9.0; if min(1.0f64, min) == 1.0 and max(min, 2.0f64) == 9.0 { return 7; } return 0; }",
    ] {
        qualify(source, Ok(7));
    }
}

#[test]
fn v02_user_functions_may_not_take_library_function_names() {
    for name in ["round", "min", "sin"] {
        let source =
            format!("fn {name}(x: f64) -> f64 {{ return x; }} fn main() -> i32 {{ return 0; }}");
        let error = check::check(front::parse(source.as_bytes()).unwrap(), true).unwrap_err();
        assert_eq!(error.code, "NAME_DUPLICATE", "{source}");
        assert_eq!(
            error.message,
            format!(
                "function name '{name}' is reserved for the invar-math-cr-v1 library function '{name}'"
            ),
            "{source}"
        );
    }
}

#[test]
fn v02_f64_literals_lower_to_exact_binary64_constants() {
    let program = mir::lower(&checked(
        "fn main() -> i32 { let a: f64 = 0.1; let b = 0.3f64; let c: f32 = 0.1; return 7; }",
    ))
    .unwrap();
    let constants: Vec<String> = program
        .functions
        .iter()
        .flat_map(|f| &f.blocks)
        .flat_map(|b| &b.instructions)
        .filter_map(|i| match i.kind {
            mir::Operation::F64(value) => Some(format!("f64:{:016x}", value.to_bits())),
            mir::Operation::F32(value) => Some(format!("f32:{:08x}", value.to_bits())),
            _ => None,
        })
        .collect();
    assert_eq!(
        constants,
        [
            "f64:3fb999999999999a",
            "f64:3fd3333333333333",
            "f32:3dcccccd"
        ]
    );
    mir::verify(&program).unwrap();
}
