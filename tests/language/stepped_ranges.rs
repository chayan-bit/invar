//! Issue #256: `for i in lo .. hi step k` counts `ceil((hi - lo) / k)` trips,
//! the view stride rule, and is the counted loop it desugars to.
use crate::surface_sugar::{assert_same_program_renamed, first_code};
use invar::{check, eval, front};

/// `for i in lo .. hi step k`, spelled out as its desugaring.
fn plain(lo: &str, hi: &str, k: &str, body: &str) -> String {
    format!(
        "let lo_t: u32 = {lo}; let hi_t: u32 = {hi}; let step_t: u32 = {k}; var trips_t: u32 = 0u32; \
         if hi_t > lo_t {{ trips_t = (hi_t - lo_t - 1u32) / step_t + 1u32; }} \
         for trip_t in 0u32 .. trips_t {{ let i: u32 = lo_t + trip_t * step_t; {body} }}"
    )
}

/// `body` in a `main` with an array `xs`, an accumulator `sum` and a zero
/// `n` known only at run time.
fn program(body: &str) -> String {
    format!(
        "fn main() -> i32 {{\n    let xs: Array<u32> = [5u32, 6u32, 7u32, 8u32, 9u32, 10u32, 11u32];\n    \
         var sum: u32 = 0u32;\n    var n: u32 = 0u32;\n    {body}\n    if sum == 0u32 {{ return 0; }}\n    return 1;\n}}\n"
    )
}

fn renames(source: &str) -> Vec<(String, &'static str)> {
    let at = source.find("for i in ").unwrap() + 4;
    ["lo", "hi", "step", "trips", "trip"]
        .iter()
        .zip(["lo_t", "hi_t", "step_t", "trips_t", "trip_t"])
        .map(|(role, name)| (format!("{role}@{at}"), name))
        .collect()
}

#[test]
fn trip_counts_follow_the_view_stride_rule_at_the_boundaries() {
    for (lo, hi, k) in [
        (0u32, 10u32, 3u32),
        (0, 9, 3),
        (2, 3, 1),
        (5, 5, 2),
        (7, 4, 1),
        (1, 4, 10),
        (0, 1, 4_000_000_000),
        (4_294_967_290, 4_294_967_295, 2),
        (0, 4_294_967_295, 2_147_483_648),
    ] {
        // Element j is lo + j*k for j < ceil((hi - lo) / k); none when hi <= lo.
        let count = if hi > lo { (hi - lo).div_ceil(k) } else { 0 };
        let checks: String = (0..count)
            .map(|j| format!(" and seen[{j}u32] == {}u32", lo + j * k))
            .collect();
        let source = format!(
            "fn main() -> i32 {{ var seen: Array<u32> = filled_u32(8u32, 0u32); var n: u32 = 0u32; \
             for i in {lo}u32 .. {hi}u32 step {k}u32 {{ seen[n] = i; n += 1u32; }} \
             if n == {count}u32{checks} {{ return 1; }} return 0; }}"
        );
        crate::engines::agree(&source, 1_000_000, 1);
        if lo <= hi && hi <= 64 {
            let view = format!(
                "fn main() -> i32 {{ let ys: Array<u32> = filled_u32(64u32, 0u32); \
                 let v = view(ys, {lo}u32, {hi}u32, {k}u32); if len(v) == {count}u32 {{ return 1; }} return 0; }}"
            );
            crate::engines::agree(&view, 1_000_000, 1);
        }
    }
}

#[test]
fn a_stepped_loop_is_its_counted_loop() {
    for (lo, hi, k, body) in [
        ("0u32", "len(xs)", "2u32", "sum += xs[i];"),
        ("1u32", "len(xs)", "3u32", "sum += xs[i] * i;"),
        ("4u32", "4u32", "2u32", "sum += xs[i];"),
        ("6u32", "2u32", "1u32", "sum += xs[i];"),
        ("0u32", "len(xs)", "100u32", "sum += xs[i];"),
        // The body's own trap, in source order, at the same fuel prefix.
        ("0u32", "9u32", "4u32", "sum += xs[i];"),
        // A zero step known only at run time traps on a non-empty range.
        ("0u32", "len(xs)", "n", "sum += xs[i];"),
        ("3u32", "3u32", "n", "sum += xs[i];"),
    ] {
        let sugared = program(&format!("for i in {lo} .. {hi} step {k} {{ {body} }}"));
        assert_same_program_renamed(
            &sugared,
            &program(&plain(lo, hi, k, body)),
            &renames(&sugared),
        );
    }
}

#[test]
fn a_zero_step_is_refused_and_a_runtime_zero_traps_over_the_header() {
    for step in ["0u32", "0", "0x0u32", "0_000u32"] {
        let source = program(&format!(
            "for i in 0u32 .. 3u32 step {step} {{ sum += i; }}"
        ));
        assert_eq!(first_code(&source), "PARSE_ZERO_STEP", "{step}");
    }
    for (body, code) in [
        ("for i in 0u32 .. 3u32 step 2 {}", "OK"),
        ("for i in 0u32 .. 3u32 step -1 {}", "TYPE_MISMATCH"),
        ("for i in 0u32 .. 3u32 step 1i32 {}", "TYPE_MISMATCH"),
        ("for i in 0u32 .. 3u32 step {}", "PARSE_EXPECTED_TOKEN"),
        ("for i in xs step 2u32 {}", "PARSE_EXPECTED_TOKEN"),
        (
            "for i in 0u32 .. 3u32 step 1u32 { i = 2u32; }",
            first_code(&program("let i: u32 = 0u32; i = 2u32;")),
        ),
        (
            "let step: u32 = 2u32; for i in 0u32 .. 6u32 step step { sum += i; }",
            "OK",
        ),
    ] {
        assert_eq!(first_code(&program(body)), code, "{body}");
    }
    let source = program("for i in 0u32 .. 3u32 step n { sum += i; }");
    let checked = check::check(front::parse(source.as_bytes()).unwrap(), true).unwrap();
    let start = source.find("for i").unwrap();
    let end = source.find(" step n").unwrap() + " step n".len();
    assert_eq!(
        eval::run(&checked, 1_000).unwrap_err(),
        eval::EvalError::Trap {
            category: eval::TrapCategory::DivisionByZero,
            span: front::Span { start, end },
        }
    );
}
