//! invar-math-cr-v1 lowering differential: every library function, in f32 and
//! f64, on signed zeros, infinities, NaN, ties and ordinary values, through
//! source programs on the reference, MIR and native engines. Expected values
//! come from `invar::math`, which `math_differential` checks against
//! upstream CORE-MATH; this file checks that each engine lowers to it exactly.
use invar::{
    check, eval, front,
    math::{MathFn, eval_f32, eval_f64},
    mir, mir_eval,
    numeric_contract::{F32Bits, F64Bits},
};

const VALUES: [&str; 24] = [
    "0.0",
    "-0.0",
    "0.5",
    "-0.5",
    "1.0",
    "-1.0",
    "2.5",
    "-2.5",
    "3.0",
    "0.1",
    "0.00001",
    "10.0",
    "100.5",
    "-700.25",
    "1000.0",
    "123456.789",
    "0.75",
    "-0.3",
    "4.5",
    "-5.5",
    "INF",
    "-INF",
    "NAN",
    "0.7853981633974483",
];
const PAIRS: [&str; 9] = [
    "0.0", "-0.0", "1.0", "-2.5", "0.5", "3.0", "INF", "-INF", "NAN",
];

/// Source text for a named value at type `ty`.
fn operand(value: &str, ty: &str) -> String {
    match value {
        "INF" => format!("(1.0{ty} / 0.0{ty})"),
        "-INF" => format!("(-1.0{ty} / 0.0{ty})"),
        "NAN" => format!("(0.0{ty} / 0.0{ty})"),
        _ => format!("{value}{ty}"),
    }
}

fn bits64(value: &str) -> F64Bits {
    match value {
        "INF" => F64Bits::INFINITY,
        "-INF" => F64Bits::NEG_INFINITY,
        "NAN" => F64Bits::NAN,
        _ if value.starts_with('-') => F64Bits::from_decimal(&value[1..]).unwrap().negate(),
        _ => F64Bits::from_decimal(value).unwrap(),
    }
}

fn bits32(value: &str) -> F32Bits {
    match value {
        "INF" => F32Bits::from_bits(0x7f80_0000),
        "-INF" => F32Bits::from_bits(0xff80_0000),
        "NAN" => F32Bits::NAN,
        _ if value.starts_with('-') => F32Bits::from_decimal(&value[1..]).unwrap().negate(),
        _ => F32Bits::from_decimal(value).unwrap(),
    }
}

/// An exact plain decimal for a finite result, or `None` when the literal
/// would exceed the shared 256-digit cap.
fn literal(value: f64, ty: &str) -> Option<String> {
    if value.is_nan() {
        return Some(operand("NAN", ty));
    }
    if value.is_infinite() {
        return Some(operand(if value > 0.0 { "INF" } else { "-INF" }, ty));
    }
    if value != 0.0 && !(1e-200..1e200).contains(&value.abs()) {
        return None;
    }
    let text = format!("{value}");
    let text = if text.contains('.') {
        text
    } else {
        format!("{text}.0")
    };
    Some(match text.strip_prefix('-') {
        Some(magnitude) => format!("(-{magnitude}{ty})"),
        None => format!("{text}{ty}"),
    })
}

/// One program per function and type; `main` counts the cases that match.
fn program(f: MathFn, binary64: bool) -> (String, i32) {
    let ty = if binary64 { "f64" } else { "f32" };
    let inputs: Vec<Vec<&str>> = if f.arity() == 1 {
        VALUES.iter().map(|v| vec![*v]).collect()
    } else {
        PAIRS
            .iter()
            .flat_map(|a| PAIRS.iter().map(move |b| vec![*a, *b]))
            .collect()
    };
    let mut body = String::new();
    let mut cases = 0;
    for args in inputs {
        let expected = if binary64 {
            let parsed: Vec<F64Bits> = args.iter().map(|v| bits64(v)).collect();
            f64::from_bits(eval_f64(f, &parsed).to_bits())
        } else {
            let parsed: Vec<F32Bits> = args.iter().map(|v| bits32(v)).collect();
            f64::from(f32::from_bits(eval_f32(f, &parsed).to_bits()))
        };
        let Some(expected) = literal(expected, ty) else {
            continue;
        };
        let call = args
            .iter()
            .map(|v| operand(v, ty))
            .collect::<Vec<_>>()
            .join(", ");
        body.push_str(&format!(
            "if same({}({call}), {expected}) {{ ok = ok + 1; }} ",
            f.name()
        ));
        cases += 1;
    }
    let source = format!(
        "fn same(a: {ty}, b: {ty}) -> Bool {{ if a != a {{ return b != b; }} if a == 0.0 {{ return b == 0.0 and 1.0 / a == 1.0 / b; }} return a == b; }} \
         fn main() -> i32 {{ var ok: i32 = 0; {body}return ok; }}"
    );
    (source, cases)
}

#[test]
fn every_math_function_lowers_identically_on_every_cpu_engine() {
    for binary64 in [false, true] {
        for f in MathFn::ALL {
            let (source, cases) = program(f, binary64);
            let context = format!("{} f{}", f.name(), if binary64 { 64 } else { 32 });
            let checked = check::check(front::parse(source.as_bytes()).unwrap(), true)
                .unwrap_or_else(|e| panic!("{context}: {e:?}\n{source}"));
            let reference = eval::run(&checked, 1_000_000).unwrap();
            assert_eq!(reference.value, eval::Value::I32(cases), "{context}");
            let program = mir::lower(&checked).unwrap();
            assert_eq!(
                mir_eval::run(&program, 1_000_000).unwrap(),
                reference,
                "{context}"
            );
            #[cfg(feature = "native")]
            {
                let (status, doc) = invar::native::run_program(&program, 1_000_000).unwrap();
                assert_eq!(status, 0, "{context}: {doc}");
                assert_eq!(doc["result"]["value"], cases, "{context}: {doc}");
            }
        }
    }
}
