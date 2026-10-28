//! ENG-03: storage-only arrays `Array<f16|bf16|i8|u8>` on every CPU
//! engine (reference, MIR, cpu-parallel, and native-serial with
//! `--features native`). Loads widen exactly; a store rounds once to nearest
//! even (f16, bf16) or traps `INTEGER_OVERFLOW` (i8, u8), as
//! `numeric_contract::storage` says.

use invar::eval::{EvalError, TrapCategory};
use invar::numeric_contract::F32Bits as F;
use invar::numeric_contract::storage::{Bf16Bits, F16Bits};
use invar::{check, eval, front, mir, mir_eval, schedule};

const FUEL: u64 = 10_000_000;

use crate::split_mix::Rng;

/// A source expression with exactly the value `x` (NaN as the class).
fn literal(x: F) -> String {
    if x.is_nan() {
        return "(0.0 / 0.0)".into();
    }
    if x.is_infinite() {
        return if x.is_negative() {
            "(-1.0 / 0.0)"
        } else {
            "(1.0 / 0.0)"
        }
        .into();
    }
    // Display is the shortest round-trip decimal, so it denotes the same word.
    let mut text = format!("{}", f32::from_bits(x.to_bits()).abs());
    if !text.contains('.') {
        text.push_str(".0");
    }
    if x.is_negative() {
        format!("(-{text})")
    } else {
        text
    }
}

/// A source predicate that holds exactly when `r` has the class and bits of `x`.
fn matches(r: &str, x: F) -> String {
    if x.is_nan() {
        format!("{r} != {r}")
    } else if x.is_zero() {
        let sign = if x.is_negative() { "<" } else { ">" };
        format!("{r} == 0.0 and 1.0 / {r} {sign} 0.0")
    } else {
        format!("{r} == {}", literal(x))
    }
}

fn checked(source: &str) -> check::CheckedProgram {
    check::check(front::parse(source.as_bytes()).unwrap(), true)
        .unwrap_or_else(|error| panic!("{error:?}\n{source}"))
}

/// Run on every CPU engine this build has; all must observe the same result
/// and fuel. Returns that result and the number of engines run.
fn every_engine(source: &str) -> (Result<eval::Value, EvalError>, usize) {
    let checked = checked(source);
    invar::ownership::check(&checked).unwrap();
    let program = mir::lower(&checked).unwrap();
    let reference = eval::observe(&checked, FUEL);
    let lowered = mir_eval::observe(&program, FUEL);
    assert_eq!(lowered.result, reference.result, "mir: {source}");
    assert_eq!(lowered.steps_used, reference.steps_used);
    let mut engines = 2;
    for workers in [1, 2, 7] {
        let parallel = schedule::run_source(source.as_bytes(), &program, FUEL, workers).unwrap();
        assert_eq!(
            parallel.observation.result, reference.result,
            "cpu-parallel {workers}"
        );
        assert_eq!(parallel.observation.steps_used, reference.steps_used);
        engines += 1;
    }
    #[cfg(feature = "native")]
    {
        let (_, doc) = invar::native::run_program(&program, FUEL).unwrap();
        match &reference.result {
            Ok(eval::Value::I32(value)) => {
                assert_eq!(doc["result"]["value"], serde_json::json!(value), "{doc}")
            }
            Err(EvalError::Trap { category, span }) => {
                assert_eq!(
                    doc["trap"]["category"], "INTEGER_OVERFLOW",
                    "{category:?} {doc}"
                );
                assert_eq!(doc["trap"]["span"]["start"], span.start, "{doc}");
            }
            other => panic!("unexpected reference result {other:?}"),
        }
        assert_eq!(doc["steps_used"], reference.steps_used, "{doc}");
        engines += 1;
    }
    (reference.result, engines)
}

/// Words that exercise one narrow float format: signed zeros, infinities,
/// NaN, the overflow threshold, subnormals, exact ties both ways, and
/// seeded words of every class.
fn words(rng: &mut Rng, ties: &[u32]) -> Vec<F> {
    let mut words: Vec<u32> = vec![
        0,
        0x8000_0000,
        0x7f80_0000,
        0xff80_0000,
        0x7fc0_0000,
        0x3f80_0000,
        0xbf80_0000,
        0x477f_e000,
        0x477f_efff,
        0x477f_f000,
        0x7f7f_ffff,
        0xff7f_ffff,
        0x3380_0000,
        0x3300_0000,
        0x3300_0001,
        0x0000_0001,
        0x8000_0001,
        0x007f_ffff,
        0x0080_0000,
    ];
    for &tie in ties {
        words.extend([tie, tie | 0x8000_0000, tie + 1, tie - 1]);
    }
    words.extend((0..96).map(|_| rng.next() as u32));
    words.extend((0..32).map(|_| 0x3800_0000 + (rng.next() as u32 & 0x0fff_ffff)));
    words.into_iter().map(F::from_bits).collect()
}

/// Store every word into one storage array, then compare each load with the
/// oracle's rounding.
fn round_trip(ty: &str, words: &[F], oracle: impl Fn(F) -> F) -> String {
    let stores: String = words
        .iter()
        .enumerate()
        .map(|(k, &x)| format!("a[{k}u32] = {}; ", literal(x)))
        .collect();
    let checks: String = words
        .iter()
        .enumerate()
        .map(|(k, &x)| {
            format!(
                "if {} {{ good = good + 1u32; }} ",
                matches(&format!("a[{k}u32]"), oracle(x))
            )
        })
        .collect();
    format!(
        "fn main() -> i32 {{ var a = filled_{ty}({}u32, 0.0); {stores} var good: u32 = 0u32; {checks} \
         if good == len(a) {{ return 1; }} return 0; }}",
        words.len()
    )
}

#[test]
fn issue_eng_03_f16_and_bf16_stores_round_like_the_oracle_on_every_engine() {
    let mut rng = Rng(0x2800_f16b);
    // Ties to even: f16 at 1 + 2^-11 and 1 + 3 * 2^-11, bf16 at 1 + 2^-8 and 1 + 3 * 2^-8.
    let f16 = words(&mut rng, &[0x3f80_1000, 0x3f80_3000]);
    let bf16 = words(&mut rng, &[0x3f80_8000, 0x3f81_8000]);
    let source = round_trip("f16", &f16, |x| F16Bits::round(x).widen());
    assert_eq!(every_engine(&source).0, Ok(eval::Value::I32(1)), "{source}");
    let source = round_trip("bf16", &bf16, |x| Bf16Bits::round(x).widen());
    assert_eq!(every_engine(&source).0, Ok(eval::Value::I32(1)), "{source}");
    // The fill value rounds too: 65520 is past the largest f16.
    let fill = "fn main() -> i32 { let a = filled_f16(2u32, 65520.0); let b = filled_bf16(1u32, 1.00390625); \
                if a[1u32] == 1.0 / 0.0 and b[0u32] == 1.0 { return 1; } return 0; }";
    let (result, engines) = every_engine(fill);
    assert_eq!(result, Ok(eval::Value::I32(1)));
    assert!(engines >= 5);
}

#[test]
fn issue_eng_03_i8_and_u8_loads_widen_and_in_range_stores_keep_their_value() {
    let source = "fn main() -> i32 { var b = filled_i8(3u32, -128); b[1u32] = 127; \
                  let w: i32 = b[0u32] + b[1u32]; var c = filled_u8(2u32, 255u32); c[0u32] = 0u32; \
                  if w == -1 and c[1u32] == 255u32 and c[0u32] == 0u32 and len(b) == 3u32 { return 1; } return 0; }";
    assert_eq!(every_engine(source).0, Ok(eval::Value::I32(1)));
}

#[test]
fn issue_eng_03_i8_and_u8_narrowing_traps_at_the_first_trap_in_source_order() {
    let cases = [
        (
            "fn main() -> i32 { var b = filled_i8(2u32, 0); b[0u32] = 100; b[1u32] = -129; b[0u32] = 300; return 1; }",
            "-129",
        ),
        (
            "fn main() -> i32 { var b = filled_i8(2u32, 0); var v: i32 = 120; \
             for i in 0u32..2u32 { b[i] = v + 0; v = v + 8; } b[0u32] = 1000; return 1; }",
            "v + 0",
        ),
        (
            "fn main() -> i32 { var c = filled_u8(1u32, 256u32); c[0u32] = 999u32; return 1; }",
            "256u32",
        ),
        (
            "fn main() -> i32 { var c = filled_u8(2u32, 7u32); c[1u32] = 255u32 + 1u32; return 1; }",
            "255u32 + 1u32",
        ),
    ];
    for (source, first) in cases {
        let (result, _) = every_engine(source);
        let start = source.find(first).unwrap();
        match result {
            Err(EvalError::Trap {
                category: TrapCategory::IntegerOverflow,
                span,
            }) => assert_eq!(span.start, start, "{source}"),
            other => panic!("{other:?}: {source}"),
        }
    }
}

#[test]
fn issue_eng_03_storage_stores_parallelize_and_keep_fuel_prefixes() {
    let mut rng = Rng(0x5eed_0003);
    let xs: Vec<F> = (0..61)
        .map(|_| F::from_bits(0x3c00_0000 + (rng.next() as u32 & 0x07ff_ffff)))
        .collect();
    let three = F::from_bits(3.0f32.to_bits());
    let mut sum = F::from_bits(0);
    for &x in &xs {
        sum = sum.add(F16Bits::round(x.mul(three)).widen());
    }
    let items: Vec<String> = xs.iter().map(|&x| literal(x)).collect();
    let source = format!(
        "fn main() -> i32 {{ let xs: Array<f32> = [{}]; var a = filled_f16(len(xs), 0.0); \
         for i in 0u32..len(xs) {{ a[i] = xs[i] * 3.0; }} var s: f32 = 0.0; \
         for j in 0u32..len(a) {{ s = s + a[j]; }} if {} {{ return 1; }} return 0; }}",
        items.join(", "),
        matches("s", sum)
    );
    assert_eq!(every_engine(&source).0, Ok(eval::Value::I32(1)));
    let checked = checked(&source);
    let lowered = mir::lower(&checked).unwrap();
    let complete = eval::observe(&checked, FUEL);
    for fuel in (0..=complete.steps_used + 1).step_by(7) {
        let reference = eval::observe(&checked, fuel);
        let actual = mir_eval::observe(&lowered, fuel);
        assert_eq!(actual.result, reference.result, "fuel={fuel}");
        assert_eq!(actual.steps_used, reference.steps_used);
    }
}

fn code(source: &str) -> &'static str {
    match front::parse(source.as_bytes()).and_then(|program| check::check(program, true)) {
        Ok(_) => "OK",
        Err(error) => error.code,
    }
}

#[test]
fn issue_eng_03_storage_arrays_refuse_arithmetic_crossings_and_views() {
    let main = "fn main() -> i32 { return 1; }";
    let refused = [
        // A storage array never crosses a function boundary yet.
        (format!("fn f(a: read Array<f16>) -> u32 {{ return len(a); }} {main}"), "UNSUPPORTED_FEATURE"),
        (format!("fn f() -> Array<i8> {{ return filled_i8(1u32, 0); }} {main}"), "UNSUPPORTED_FEATURE"),
        (format!("struct S {{ a: Array<u8> }} {main}"), "UNSUPPORTED_FEATURE"),
        (
            "fn main() -> i32 { let b: Array<i8> = filled_i8(2u32, 0); \
             let c = closure [read b: Array<i8>] (x: i32) -> i32 { return b[0u32] + x; }; return c(1); }".to_owned(),
            "UNSUPPORTED_FEATURE",
        ),
        // A view would store unconverted values.
        (
            "fn main() -> i32 { var b = filled_i8(2u32, 0); let v = view_mut(b, 0u32, 1u32); return 1; }".to_owned(),
            "UNSUPPORTED_FEATURE",
        ),
        (
            "fn main() -> i32 { let b = filled_u8(2u32, 0u32); let v = view((b), 0u32, 1u32); return 1; }".to_owned(),
            "UNSUPPORTED_FEATURE",
        ),
        // Loads are i32/u32/f32, so a narrow value never meets an operator; a
        // store takes the widened type and no other array converts.
        ("fn main() -> i32 { var b = filled_i8(1u32, 0); b[0u32] = 1.0; return 1; }".to_owned(), "TYPE_MISMATCH"),
        ("fn main() -> i32 { var c = filled_u8(1u32, 0u32); let x: i32 = 1; c[0u32] = x; return 1; }".to_owned(), "TYPE_MISMATCH"),
        ("fn main() -> i32 { let a: Array<f16> = [1.0]; return 1; }".to_owned(), "TYPE_MISMATCH"),
        ("fn main() -> i32 { let a: Array<bf16> = filled_f32(1u32, 0.0); return 1; }".to_owned(), "TYPE_MISMATCH"),
        ("fn main() -> i32 { let a = filled_f16(1u32, 0.0); let b: Array<f32> = a; return 1; }".to_owned(), "TYPE_MISMATCH"),
    ];
    for (source, expected) in &refused {
        assert_eq!(code(source), *expected, "{source}");
    }
    assert_eq!(refused.len(), 11);
}
