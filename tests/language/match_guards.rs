//! Issue #260: match guards and let-else. A guard is tried in arm order after its
//! pattern binds, with the ordinary trap order and fuel, and never covers its case;
//! let-else reaches every engine as plain statements and its else block must diverge.
use crate::loop_control::{agree, checked};
use crate::surface_sugar::{assert_same_program_renamed, first_code};
use invar::{eval, front};

const SHAPES: &str = "enum Shape { Circle(i32), Square(i32), Empty }

fn area(s: read Shape) -> i32 {
    return match s {
        Shape::Circle(r) if r > 10 => 1000,
        Shape::Circle(q) => q * q * 3,
        Shape::Square(w) if w == 0 => -1,
        Shape::Square(v) => v * v,
        Shape::Empty => 0,
    };
}

fn main() -> i32 {
    let a: Shape = Shape::Circle(2);
    let b: Shape = Shape::Circle(11);
    let c: Shape = Shape::Square(0);
    let d: Shape = Shape::Square(3);
    let e: Shape = Shape::Empty();
    return area(a) + area(b) + area(c) + area(d) + area(e);
}
";

const PICK: &str = "enum Pick { A(u32), B(u32) }\n";

#[test]
fn guards_pick_the_first_passing_arm_identically_on_every_engine() {
    agree(SHAPES, Ok(12 + 1000 - 1 + 9), true);
    // A guard of another case never runs; the passing guard after a failing one wins.
    let order = format!(
        "{PICK}fn main() -> i32 {{
    let zero: u32 = 0u32;
    let p: Pick = Pick::B(4u32);
    return match p {{
        Pick::A(a) if 1u32 / zero == 0u32 => 1,
        Pick::A(b) => 2,
        Pick::B(c) if c > 5u32 => 3,
        Pick::B(d) if d > 3u32 => 4,
        Pick::B(e) => 5,
    }};
}}"
    );
    agree(&order, Ok(4), true);
    let moved = format!(
        "{PICK}fn main() -> i32 {{
    let p: Pick = Pick::B(4u32);
    let v: u32 = match move p {{ Pick::A(a) => a, Pick::B(c) if c > 9u32 => 0u32, Pick::B(d) => d + 1u32 }};
    if v == 5u32 {{ return 1; }}
    return 0;
}}"
    );
    agree(&moved, Ok(1), true);
    let result = "fn half(x: u32) -> Result<u32, u32> {
    if x % 2u32 == 0u32 { return Result::Ok(x / 2u32); }
    return Result::Err(x);
}

fn main() -> i32 {
    let r = half(8u32);
    return match r { Result::Ok(h) if h > 3u32 => 1, Result::Ok(g) => 2, Result::Err(e) => 3 };
}";
    agree(result, Ok(1), true);
    let looped = format!(
        "{PICK}fn main() -> i32 {{
    var acc: u32 = 0u32;
    for i in 0u32 .. 32u32 {{
        let o: Pick = Pick::A(i);
        acc = acc + match o {{ Pick::A(a) if a % 3u32 == 0u32 => a, Pick::A(b) => 1u32, Pick::B(c) => c }};
    }}
    if acc == 186u32 {{ return 1; }}
    return 0;
}}"
    );
    agree(&looped, Ok(1), true);
}

#[test]
fn a_trapping_guard_traps_at_its_operator_with_the_reference_fuel_prefix() {
    let source = "enum Shape { Circle(i32), Empty }

fn main() -> i32 {
    let s: Shape = Shape::Circle(0);
    return match s {
        Shape::Circle(r) if 10 / r > 1 => 1,
        Shape::Circle(q) => 2,
        Shape::Empty => 3,
    };
}";
    agree(source, Err(eval::TrapCategory::DivisionByZero), true);
    let observed = eval::observe(&checked(source), 1_000_000);
    let Err(eval::EvalError::Trap { span, .. }) = observed.result else {
        panic!("{:?}", observed.result)
    };
    assert_eq!(span.start, source.find("/ r").unwrap());
}

/// Steps of `main` returning `body`'s match, with a `let no: Bool = false;` before it.
fn steps(arms: &str) -> u64 {
    let source = format!(
        "{PICK}fn main() -> i32 {{\n    let no: Bool = false;\n    let p: Pick = Pick::B(4u32);\n    return match p {{ {arms} }};\n}}"
    );
    let observed = eval::observe(&checked(&source), 1_000_000);
    assert_eq!(observed.result, Ok(eval::Value::I32(5)), "{source}");
    observed.steps_used
}

#[test]
fn each_evaluated_guard_costs_its_own_steps_and_an_unreached_one_none() {
    let plain = steps("Pick::B(e) => 5, Pick::A(a) => 1");
    let two =
        steps("Pick::B(c) if no => 3, Pick::B(d) if no => 4, Pick::B(e) => 5, Pick::A(a) => 1");
    let other_case = steps("Pick::A(x) if no => 0, Pick::B(e) => 5, Pick::A(a) => 1");
    assert_eq!(two, plain + 2, "a name guard is one step each time it runs");
    assert_eq!(
        other_case, plain,
        "a guard of another case is never evaluated"
    );
}

const VERDICT: &str = "enum Shape { Circle(i32), Empty }
fn main() -> i32 {
    let s: Shape = Shape::Circle(1);
    return match s { ARMS };
}";

#[test]
fn a_guard_that_moves_an_owner_moves_it_only_on_its_own_case() {
    let program = |later_a: &str| {
        format!(
            "{PICK}fn big(a: take Array<u32>) -> Bool {{ return len(a) > 1u32; }}
fn first(a: take Array<u32>) -> i32 {{ return 3; }}
fn main() -> i32 {{
    let xs: Array<u32> = [1u32, 2u32];
    let p: Pick = Pick::B(4u32);
    return match p {{
        Pick::A(a) if big(move xs) => 1,
        Pick::A(b) => {later_a},
        Pick::B(c) => first(move xs),
    }};
}}"
        )
    };
    // Case B never runs case A's guard, so it may still move `xs`.
    agree(&program("2"), Ok(3), true);
    // A later arm of case A runs after that guard, so `xs` is gone there.
    let source = program("first(move xs)");
    assert_eq!(first_code(&source), "OWN_USE_AFTER_MOVE");
}

#[test]
fn guarded_arms_never_cover_their_case_and_refusals_have_exact_codes() {
    for (arms, code, at) in [
        (
            "Shape::Circle(r) if r > 1 => 1, Shape::Empty => 3",
            "MATCH_NONEXHAUSTIVE",
            "match s",
        ),
        (
            "Shape::Circle(r) => 1, Shape::Circle(q) if q > 1 => 2, Shape::Empty => 3",
            "NAME_DUPLICATE",
            "Shape::Circle(q)",
        ),
        (
            "Shape::Circle(r) if r => 1, Shape::Circle(q) => 2, Shape::Empty => 3",
            "TYPE_MISMATCH",
            "r =>",
        ),
        (
            "Shape::Circle(r) if r > 1 => 1, Shape::Circle(r) => 2, Shape::Empty => 3",
            "NAME_DUPLICATE",
            "Shape::Circle(r) =>",
        ),
        (
            "Shape::Circle(r) if => 1, Shape::Empty => 3",
            "PARSE_EXPECTED_TOKEN",
            "=> 1",
        ),
    ] {
        let source = VERDICT.replace("ARMS", arms);
        let error = front::parse(source.as_bytes())
            .and_then(|parsed| invar::check::check(parsed, true).map(|_| ()))
            .unwrap_err();
        assert_eq!(error.code, code, "{arms}");
        assert_eq!(
            error.span.map(|s| s.start),
            source.find(at),
            "{arms}: {error:?}"
        );
    }
    let owned = "fn main() -> i32 {
    let packet: Result<Result<i32, u32>, u32> = Result::Ok(Result::Ok(3i32));
    return match move packet { Result::Ok(inner) if true => 1i32, Result::Ok(other) => 2i32, Result::Err(code) => 0i32 };
}";
    assert_eq!(first_code(owned), "UNSUPPORTED_FEATURE");
    // A payload-free arm may be guarded too.
    let bare = VERDICT.replace(
        "ARMS",
        "Shape::Empty if true => 4, Shape::Circle(r) => r, Shape::Empty => 3",
    );
    agree(&bare, Ok(1), true);
}

/// The hidden names of the one let-else in `sugared` (flag, check arms `@k`, extraction
/// arms `@xk`, all keyed by the byte offset of its `let`) with the plain names spelled out.
fn renames(sugared: &str, plain: &[(&str, &'static str)]) -> Vec<(String, &'static str)> {
    let start = sugared[..sugared.find(" else {").unwrap()]
        .rfind("let ")
        .unwrap();
    plain
        .iter()
        .map(|(suffix, name)| (format!("else@{start}{suffix}"), *name))
        .collect()
}

#[test]
fn let_else_is_the_check_the_diverging_block_and_the_extraction() {
    let names = [
        ("", "ok"),
        ("@0", "c0"),
        ("@1", "c1"),
        ("@x0", "e0"),
        ("@x1", "e1"),
    ];
    let cases = [
        (
            format!("{PICK}fn get(p: read Pick) -> u32 {{\n    let Pick::B(v) = p else {{ return 7u32; }};\n    return v + 1u32;\n}}\nfn main() -> i32 {{ let p: Pick = Pick::A(1u32); if get(p) == 7u32 {{ return 1; }} return 0; }}"),
            format!("{PICK}fn get(p: read Pick) -> u32 {{\n    let ok: Bool = match p {{ Pick::B(c0) => true, Pick::A(c1) => false }};\n    if not ok {{ return 7u32; }}\n    let v: u32 = match p {{ Pick::B(e0) => e0, Pick::A(e1) => 0u32 }};\n    return v + 1u32;\n}}\nfn main() -> i32 {{ let p: Pick = Pick::A(1u32); if get(p) == 7u32 {{ return 1; }} return 0; }}"),
        ),
        (
            "enum F { Some(f32), None }\nfn main() -> i32 {\n    let f: F = F::Some(1.5f32);\n    let F::Some(x) = move f else { assert(false, \"none\"); };\n    if x == 1.5f32 { return 1; }\n    return 0;\n}".to_owned(),
            "enum F { Some(f32), None }\nfn main() -> i32 {\n    let f: F = F::Some(1.5f32);\n    let ok: Bool = match f { F::Some(c0) => true, F::None => false };\n    if not ok { assert(false, \"none\"); }\n    let x: f32 = match move f { F::Some(e0) => e0, F::None => 0.0f32 };\n    if x == 1.5f32 { return 1; }\n    return 0;\n}".to_owned(),
        ),
        (
            "fn half(x: u32) -> Result<u32, u32> {\n    if x % 2u32 == 0u32 { return Result::Ok(x / 2u32); }\n    return Result::Err(x);\n}\nfn main() -> i32 {\n    var acc: u32 = 0u32;\n    for i in 0u32 .. 10u32 {\n        let r = half(i);\n        let Result::Ok(h) = r else { continue; };\n        acc += h;\n    }\n    if acc == 10u32 { return 1; }\n    return 0;\n}".to_owned(),
            "fn half(x: u32) -> Result<u32, u32> {\n    if x % 2u32 == 0u32 { return Result::Ok(x / 2u32); }\n    return Result::Err(x);\n}\nfn main() -> i32 {\n    var acc: u32 = 0u32;\n    for i in 0u32 .. 10u32 {\n        let r = half(i);\n        let ok: Bool = match r { Result::Ok(c0) => true, Result::Err(c1) => false };\n        if not ok { continue; }\n        let h: u32 = match r { Result::Ok(e0) => e0, Result::Err(e1) => 0u32 };\n        acc += h;\n    }\n    if acc == 10u32 { return 1; }\n    return 0;\n}".to_owned(),
        ),
        (
            format!("{PICK}enum Mode {{ Fast, Slow }}\nfn main() -> i32 {{\n    let m: Mode = Mode::Slow();\n    let Mode::Fast = m else {{ return 2; }};\n    return 3;\n}}"),
            format!("{PICK}enum Mode {{ Fast, Slow }}\nfn main() -> i32 {{\n    let m: Mode = Mode::Slow();\n    let ok: Bool = match m {{ Mode::Fast => true, Mode::Slow => false }};\n    if not ok {{ return 2; }}\n    return 3;\n}}"),
        ),
    ];
    for (sugared, plain) in &cases {
        assert_same_program_renamed(sugared, plain, &renames(sugared, &names));
    }
    agree(&cases[0].0, Ok(1), true);
    agree(&cases[2].0, Ok(1), true);
    agree(&cases[3].0, Ok(2), true);
    let trapping = cases[1]
        .0
        .replace("1.5f32);\n", "1.5f32);\n    let g: F = F::None();\n")
        .replace("move f else", "move g else");
    agree(&trapping, Err(eval::TrapCategory::AssertionFailed), true);
}

#[test]
fn let_else_refusals_have_exact_codes_and_spans() {
    let base = |stmt: &str| {
        format!(
            "{PICK}fn main() -> i32 {{\n    var acc: u32 = 0u32;\n    let o: Pick = Pick::A(1u32);\n    {stmt}\n    return 0;\n}}"
        )
    };
    for (stmt, code, at) in [
        (
            "let Pick::A(v) = o else { acc = 1u32; };",
            "PARSE_LET_ELSE_FALLTHROUGH",
            "{ acc",
        ),
        (
            "let Pick::A(v) = o else { if true { return 1; } };",
            "PARSE_LET_ELSE_FALLTHROUGH",
            "{ if",
        ),
        (
            "let Pick::A(v) = o else { assert(true, \"x\"); };",
            "PARSE_LET_ELSE_FALLTHROUGH",
            "{ assert",
        ),
        (
            "let Pick::A(v) = o else { break; };",
            "PARSE_LOOP_CONTROL",
            "break",
        ),
        (
            "let Pick::A(v) = Pick::A(2u32) else { return 1; };",
            "PARSE_EXPECTED_TOKEN",
            "Pick::A(2u32)",
        ),
        (
            "var Pick::A(v) = o else { return 1; };",
            "PARSE_EXPECTED_TOKEN",
            "::A(v) = o",
        ),
        (
            "let Pick::C(v) = o else { return 1; };",
            "NAME_UNKNOWN",
            "Pick::C(v)",
        ),
        (
            "let r = Result::Ok(1u32);\n    let Result::Ok(h) = r else { return 1; };",
            "TYPE_MISMATCH",
            "r else",
        ),
    ] {
        let source = base(stmt);
        let error = front::parse(source.as_bytes())
            .and_then(|parsed| invar::check::check(parsed, true).map(|_| ()))
            .unwrap_err();
        assert_eq!(error.code, code, "{stmt}");
        assert_eq!(
            error.span.map(|s| s.start),
            source.find(at),
            "{stmt}: {error:?}"
        );
    }
    let owned = "fn main() -> i32 {
    let packet: Result<Result<i32, u32>, u32> = Result::Ok(Result::Ok(3i32));
    let Result::Ok(inner) = move packet else { return 1; };
    return 0;
}";
    assert_eq!(first_code(owned), "UNSUPPORTED_FEATURE");
    // A `move` of a typed owner declares the Result type for the pattern.
    let moved = "fn half(x: u32) -> Result<u32, u32> { return Result::Ok(x); }
fn main() -> i32 {
    let q = half(4u32);
    let r = move q;
    let Result::Ok(h) = r else { return 1; };
    return 0;
}";
    assert_eq!(first_code(moved), "OK");
    // The parser never splits `let name` and `let V::P`: a plain binding is unchanged.
    assert_eq!(first_code(&base("let w: u32 = 1u32;")), "OK");
}
