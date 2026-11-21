//! Surface forms of the language-surface epic (#194) that are spelled
//! differently but must reach every engine as the same program: literal
//! spellings (#253) and the sugar that the parser desugars before checking
//! and family recognition (N11.4.3).
use invar::{check, engine, eval, front, mir, mir_eval};

/// Replace every `Span { start: N, end: M }` in a Debug rendering with `Span`.
fn scrub_spans(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut rest = text;
    while let Some(at) = rest.find("Span { start: ") {
        out.push_str(&rest[..at]);
        out.push_str("Span");
        let tail = &rest[at..];
        rest = &tail[tail.find('}').map_or(tail.len(), |end| end + 1)..];
    }
    out.push_str(rest);
    out
}

/// Receipt fields derived from source spans: `span` itself, and the
/// normal-form digests, which hash span-carrying certificates (see
/// `receipt_digests_follow_spans_alone`).
const SPAN_DERIVED: [&str; 3] = ["span", "fingerprint", "certificate_digest"];

/// Remove span-derived JSON fields, recursively.
fn scrub_json(value: &mut serde_json::Value) {
    match value {
        serde_json::Value::Object(map) => {
            for key in SPAN_DERIVED {
                map.remove(key);
            }
            map.values_mut().for_each(scrub_json);
        }
        serde_json::Value::Array(items) => items.iter_mut().for_each(scrub_json),
        _ => {}
    }
}

fn lower(source: &str) -> (check::CheckedProgram, mir::Program) {
    let parsed = front::parse(source.as_bytes()).unwrap_or_else(|e| panic!("{source}: {e:?}"));
    let checked = check::check(parsed, true).unwrap_or_else(|e| panic!("{source}: {e:?}"));
    let program = mir::lower(&checked).unwrap();
    (checked, program)
}

/// MIR of every function with source spans and hash-map order removed.
fn mir_shape(program: &mir::Program, renames: &[(String, &str)]) -> String {
    let rename = |name: &String| {
        renames
            .iter()
            .find(|(hidden, _)| hidden == name)
            .map_or(name.as_str(), |(_, plain)| plain)
            .to_owned()
    };
    let functions = program.functions.iter().map(|f| {
        let mut locals = f
            .locals
            .iter()
            .map(|(name, local)| (rename(name), local))
            .collect::<Vec<_>>();
        locals.sort_by(|a, b| a.0.cmp(&b.0));
        format!(
            "{:?}",
            (
                &f.name,
                &f.params,
                &f.result,
                f.entry,
                &f.blocks,
                locals,
                &f.declaration_order
            )
        )
    });
    scrub_spans(&functions.collect::<Vec<_>>().join("\n"))
}

/// Reference and MIR results without spans: value or trap category, steps and cleanup.
fn outcome(checked: &check::CheckedProgram, program: &mir::Program, fuel: u64) -> String {
    let reference = scrub_spans(&format!("{:?}", eval::run(checked, fuel)));
    let lowered = scrub_spans(&format!("{:?}", mir_eval::run(program, fuel)));
    assert_eq!(reference, lowered, "reference and MIR disagree");
    reference
}

/// The smallest fuel at which the run no longer ends in an evaluation limit:
/// the fuel prefix of a trap or completion.
fn fuel_prefix(checked: &check::CheckedProgram) -> u64 {
    let (mut low, mut high) = (1u64, 1_000_000u64);
    while low < high {
        let mid = low + (high - low) / 2;
        match eval::run(checked, mid) {
            Err(eval::EvalError::EvaluationLimit { .. }) => low = mid + 1,
            _ => high = mid,
        }
    }
    low
}

fn receipt(source: &str, program: &mir::Program) -> serde_json::Value {
    let mut receipt = engine::run_mir(
        source.as_bytes(),
        program,
        engine::Engine::CpuParallel,
        1_000_000,
        4,
        None,
    )
    .unwrap_or_else(|e| serde_json::json!(scrub_spans(&format!("{e:?}"))));
    scrub_json(&mut receipt);
    receipt
}

/// `sugared` and `plain` must give the same MIR, the same reference and MIR
/// outcome and fuel prefix, and the same cpu-parallel receipt, spans aside.
pub fn assert_same_program(sugared: &str, plain: &str) {
    assert_same_program_renamed(sugared, plain, &[]);
}

/// [`assert_same_program`] where the sugared MIR names the hidden locals in
/// `renames` (hidden name, plain name) and the plain program spells them out.
pub fn assert_same_program_renamed(sugared: &str, plain: &str, renames: &[(String, &str)]) {
    let (sugared_checked, sugared_mir) = lower(sugared);
    let (plain_checked, plain_mir) = lower(plain);
    let shape = renames
        .iter()
        .fold(mir_shape(&sugared_mir, renames), |shape, (hidden, name)| {
            shape.replace(&format!("{hidden:?}"), &format!("{name:?}"))
        });
    assert_eq!(shape, mir_shape(&plain_mir, &[]), "MIR\n{sugared}\n{plain}");
    assert_eq!(
        outcome(&sugared_checked, &sugared_mir, 1_000_000),
        outcome(&plain_checked, &plain_mir, 1_000_000),
        "outcome\n{sugared}"
    );
    assert_eq!(
        fuel_prefix(&sugared_checked),
        fuel_prefix(&plain_checked),
        "fuel\n{sugared}"
    );
    assert_eq!(
        receipt(sugared, &sugared_mir),
        receipt(plain, &plain_mir),
        "receipt\n{sugared}"
    );
}

/// The first diagnostic code `invar check` reports for `source`.
pub fn first_code(source: &str) -> &'static str {
    match front::parse(source.as_bytes()) {
        Err(error) => error.code,
        Ok(parsed) => check::check(parsed, true)
            .err()
            .map_or("OK", |error| error.code),
    }
}

#[test]
fn receipt_digests_follow_spans_alone() {
    let source = "fn main() -> i32 { var acc: u32 = 0u32; for i in 0u32 .. 64u32 { acc = wrapping_add(acc, i); } if acc == 0u32 { return 1; } return 0; }";
    let respaced = source.replace("var acc", "var   acc");
    let digest = |text: &str| {
        let (_, program) = lower(text);
        let receipt = engine::run_mir(
            text.as_bytes(),
            &program,
            engine::Engine::CpuParallel,
            1_000_000,
            4,
            None,
        )
        .unwrap();
        receipt["normal_form"]["certificate_digest"].clone()
    };
    assert!(digest(source).is_string());
    assert_ne!(
        digest(source),
        digest(&respaced),
        "whitespace alone moves the digest"
    );
}

#[test]
fn literal_separators_and_radix_prefixes_lex_to_the_same_token_value() {
    let accepted = [
        ("1_000_000u32", "1000000u32"),
        ("0xFFFF_0000u32", "4294901760u32"),
        ("0xffu32", "255u32"),
        ("0o17", "15"),
        ("0b1010_1010u32", "170u32"),
        ("0x7FFF_FFFFi32", "2147483647i32"),
        ("1_2_3", "123"),
        ("007", "007"),
    ];
    for (spelling, value) in accepted {
        let source = format!("fn main() -> i32 {{ let x = {spelling}; return 0; }}");
        let program =
            front::parse(source.as_bytes()).unwrap_or_else(|e| panic!("{spelling}: {e:?}"));
        let front::StmtKind::Binding { value: expr, .. } =
            &program.functions[0].body.statements[0].kind
        else {
            panic!("binding")
        };
        assert_eq!(
            expr.kind,
            front::ExprKind::Integer(value.into()),
            "{spelling}"
        );
        let start = source.find(spelling).unwrap();
        assert_eq!(
            (expr.span.start, expr.span.end),
            (start, start + spelling.len()),
            "{spelling}"
        );
    }
    let refused = [
        "1_",
        "1__0",
        "0x_FF",
        "0xFF_",
        "0xFF_u32",
        "1_u32",
        "0x",
        "0xu32",
        "0b102",
        "0o8",
        "0xG1",
        "0X10",
        "0b",
        "1_0.5",
        "0x1.5",
        "0x1_0000_0000_0000_0000_0000_0000_0000_0000",
    ];
    for spelling in refused {
        let source = format!("fn main() -> i32 {{ let x = {spelling}; return 0; }}");
        let error = front::parse(source.as_bytes()).expect_err(spelling);
        assert_eq!(error.code, "LEX_INVALID_LITERAL", "{spelling}");
        assert_eq!(
            error.span.map(|s| s.start),
            source.find(spelling),
            "{spelling}"
        );
    }
}

#[test]
fn literal_range_checks_are_unchanged_by_the_spelling() {
    for (spelling, code) in [
        ("0x8000_0000i32", "TYPE_MISMATCH"),
        ("2_147_483_648i32", "TYPE_MISMATCH"),
        ("0x1_0000_0000u32", "TYPE_MISMATCH"),
        ("0xFFFF_FFFF_FFFF_FFFF_FFFFu32", "TYPE_MISMATCH"),
        ("0xFFFF_FFFFu32", "OK"),
    ] {
        let source = format!(
            "fn main() -> i32 {{ let x: u32 = 0u32; if false {{ let y = {spelling}; }} return 0; }}"
        );
        assert_eq!(first_code(&source), code, "{spelling}");
    }
    let negative = "fn main() -> i32 { return -0x8000_0000; }";
    assert_eq!(first_code(negative), "OK");
}

#[test]
fn literal_spellings_give_identical_mir_outcomes_and_receipts() {
    let body = |mask: &str, size: &str, step: &str| {
        format!(
            "fn main() -> i32 {{
                var xs: Array<u32> = filled_u32({size}, 0u32);
                for i in 0u32 .. len(xs) {{ xs[i] = wrapping_add(i, {mask}); }}
                var acc: u32 = 0u32;
                for j in 0u32 .. len(xs) {{ acc = wrapping_add(acc, xs[j] % {step}); }}
                if acc == 0u32 {{ return 1; }}
                return 0;
            }}"
        )
    };
    assert_same_program(
        &body("0xFFFF_0000u32", "1_024u32", "0b1_0000u32"),
        &body("4294901760u32", "1024u32", "16u32"),
    );
    let trapping = |big: &str| format!("fn main() -> i32 {{ let x: i32 = {big}; return x + 1; }}");
    assert_same_program(&trapping("0x7FFF_FFFF"), &trapping("2147483647"));
}

/// A `main` over `ys = [1, 2, 3, 4]` and `acc`, running `body` and returning a checksum.
fn compound_main(body: &str) -> String {
    format!(
        "fn main() -> i32 {{
            var ys: Array<u32> = [1u32, 2u32, 3u32, 4u32];
            var acc: u32 = 7u32;
            var k: u32 = 1u32;
            {body}
            if acc == 0u32 {{ return 1; }}
            return 0;
        }}"
    )
}

#[test]
fn compound_assignment_is_the_plain_assignment() {
    let cases = [
        (
            "for i in 0u32 .. len(ys) { ys[i] *= 3u32; }",
            "for i in 0u32 .. len(ys) { ys[i] = ys[i] * 3u32; }",
        ),
        (
            "for i in 0u32 .. len(ys) { acc += ys[i]; }",
            "for i in 0u32 .. len(ys) { acc = acc + ys[i]; }",
        ),
        ("acc -= 2u32 + 1u32;", "acc = acc - (2u32 + 1u32);"),
        ("acc *= k * 2u32;", "acc = acc * (k * 2u32);"),
        ("acc /= 2u32;", "acc = acc / 2u32;"),
        ("acc %= 4u32;", "acc = acc % 4u32;"),
        ("ys[2u32] += k;", "ys[2u32] = ys[2u32] + k;"),
        ("ys[k] -= 1u32;", "ys[k] = ys[k] - 1u32;"),
        // Each of these traps, at the same operation and fuel prefix.
        ("acc -= 8u32;", "acc = acc - 8u32;"),
        ("acc /= k - 1u32;", "acc = acc / (k - 1u32);"),
        (
            "for i in 0u32 .. len(ys) { ys[i] *= 2147483648u32; }",
            "for i in 0u32 .. len(ys) { ys[i] = ys[i] * 2147483648u32; }",
        ),
        ("ys[9u32] += 1u32;", "ys[9u32] = ys[9u32] + 1u32;"),
    ];
    for (sugared, plain) in cases {
        assert_same_program(&compound_main(sugared), &compound_main(plain));
    }
    let sum = |update: &str| {
        format!(
            "fn sum(xs: read Array<i32>) -> i32 {{
                var result: i32 = 0;
                for i in 0 .. len(xs) {{ {update} }}
                return result;
            }}
            fn main() -> i32 {{ let values: Array<i32> = [1, 2, 3, 4]; return sum(values); }}"
        )
    };
    assert_same_program(&sum("result += xs[i];"), &sum("result = result + xs[i];"));
}

#[test]
fn compound_assignment_evaluates_a_computed_index_once() {
    let hidden = |source: &str, index: &str| format!("index@{}", source.find(index).unwrap());
    for (sugared, plain) in [
        (
            "ys[k + 1u32] += 5u32;",
            "let index_t: u32 = k + 1u32; ys[index_t] = ys[index_t] + 5u32;",
        ),
        // A trapping index traps once, before the right side is evaluated.
        (
            "ys[k - 2u32] += 1u32 / 0u32;",
            "let index_t: u32 = k - 2u32; ys[index_t] = ys[index_t] + 1u32 / 0u32;",
        ),
        (
            "ys[k + 8u32] *= 2u32;",
            "let index_t: u32 = k + 8u32; ys[index_t] = ys[index_t] * 2u32;",
        ),
        (
            "for i in 0u32 .. 3u32 { ys[i + 1u32] -= ys[i]; }",
            "for i in 0u32 .. 3u32 { let index_t: u32 = i + 1u32; ys[index_t] = ys[index_t] - ys[i]; }",
        ),
    ] {
        let source = compound_main(sugared);
        let index = &sugared[sugared.find('[').unwrap() + 1..sugared.find(']').unwrap()];
        assert_same_program_renamed(
            &source,
            &compound_main(plain),
            &[(hidden(&source, index), "index_t")],
        );
    }
    // Repeating a five-step index costs ten steps; binding it once costs one
    // dispatch, five steps and two reads.
    let fuel = |source: &str| fuel_prefix(&lower(&compound_main(source)).0);
    assert!(
        fuel("ys[k + 1u32 + 0u32] += 5u32;")
            < fuel("ys[k + 1u32 + 0u32] = ys[k + 1u32 + 0u32] + 5u32;"),
        "the index is evaluated once"
    );
}

#[test]
fn compound_assignment_refuses_what_assignment_refuses() {
    for (sugared, plain) in [
        (
            "let x: u32 = 1u32; x += 1u32;",
            "let x: u32 = 1u32; x = x + 1u32;",
        ),
        ("acc += 1i32;", "acc = acc + 1i32;"),
        ("ys[1i32] += 1u32;", "ys[1i32] = ys[1i32] + 1u32;"),
        ("nope += 1u32;", "nope = nope + 1u32;"),
    ] {
        let code = first_code(&compound_main(plain));
        assert_ne!(code, "OK", "{plain}");
        assert_eq!(first_code(&compound_main(sugared)), code, "{sugared}");
    }
    for bad in [
        "1u32 += 2u32;",
        "len(ys) += 1u32;",
        "acc += ;",
        "acc + = 1u32;",
        "acc += 1u32",
    ] {
        assert_eq!(
            first_code(&compound_main(bad)),
            "PARSE_EXPECTED_TOKEN",
            "{bad}"
        );
    }
}
