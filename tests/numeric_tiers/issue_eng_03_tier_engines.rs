//! ENG-03: source folds under the Tree and Exact tiers on every CPU
//! engine (reference, MIR, cpu-parallel at 1, 2, 7 and 64 workers, and
//! native-serial with `--features native`), against the numeric oracle.

use invar::numeric_contract::{F32Bits as F, FoldTier, fold::relaxed_admits};
use invar::{check, eval, front, mir, mir_eval, schedule};

const FUEL: u64 = 50_000_000;

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
    // Display is the shortest round-trip positional decimal; the source rounds
    // a decimal once to nearest even, so it denotes the same word.
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
fn matches(x: F) -> String {
    if x.is_nan() {
        "r != r".into()
    } else if x.is_zero() {
        format!(
            "r == 0.0 and 1.0 / r {} 0.0",
            if x.is_negative() { "<" } else { ">" }
        )
    } else {
        format!("r == {}", literal(x))
    }
}

fn program(tier: &str, init: F, data: &str, expected: F) -> String {
    format!(
        "#[numeric(fold = \"{tier}\")] fn fold_sum(xs: read Array<f32>, init: f32) -> f32 {{ \
         var acc: f32 = init; for i in 0u32..len(xs) {{ acc = acc + xs[i]; }} return acc; }} \
         fn main() -> i32 {{ {data} let r = fold_sum(xs, {}); if {} {{ return 1; }} return 0; }}",
        literal(init),
        matches(expected)
    )
}

fn listed(terms: &[F]) -> String {
    let items: Vec<String> = terms.iter().map(|&x| literal(x)).collect();
    if items.is_empty() {
        "let xs = filled_f32(0u32, 0.0);".into()
    } else {
        format!("let xs: Array<f32> = [{}];", items.join(", "))
    }
}

/// Run on every CPU engine this build has; each must return 1.
fn every_engine(source: &str, workers: &[usize]) -> usize {
    let checked = check::check(front::parse(source.as_bytes()).unwrap(), true)
        .unwrap_or_else(|error| panic!("{error:?}\n{source}"));
    invar::ownership::check(&checked).unwrap();
    let program = mir::lower(&checked).unwrap();
    let reference = eval::observe(&checked, FUEL);
    assert_eq!(
        reference.result,
        Ok(eval::Value::I32(1)),
        "reference: {source}"
    );
    let lowered = mir_eval::observe(&program, FUEL);
    assert_eq!(lowered.result, reference.result, "mir: {source}");
    assert_eq!(lowered.steps_used, reference.steps_used);
    let mut engines = 2;
    for &n in workers {
        let parallel = schedule::run_source(source.as_bytes(), &program, FUEL, n).unwrap();
        assert_eq!(
            parallel.observation.result, reference.result,
            "cpu-parallel {n}: {source}"
        );
        assert_eq!(parallel.observation.steps_used, reference.steps_used);
        engines += 1;
    }
    #[cfg(feature = "native")]
    {
        let (status, doc) = invar::native::run_program(&program, FUEL).unwrap();
        assert_eq!(
            (status, &doc["result"]["value"]),
            (0, &serde_json::json!(1)),
            "{doc}"
        );
        engines += 1;
    }
    engines
}

fn word(rng: &mut Rng) -> F {
    match rng.next() % 8 {
        0 => [
            F::POS_ZERO,
            F::NEG_ZERO,
            F::INFINITY,
            F::NEG_INFINITY,
            F::NAN,
        ][(rng.next() % 5) as usize],
        // Subnormals and the smallest normals.
        1 => F::from_bits(rng.next() as u32 & 0x80ff_ffff),
        // Cancelling magnitudes.
        2 | 3 => {
            let big = F::from_bits(0x4b00_0000 + (rng.next() as u32 & 0x00ff_ffff));
            if rng.next().is_multiple_of(2) {
                big
            } else {
                big.negate()
            }
        }
        _ => loop {
            let x = F::from_bits(rng.next() as u32);
            if !x.is_nan() && !x.is_infinite() {
                break x;
            }
        },
    }
}

#[test]
fn issue_eng_03_tree_and_exact_source_folds_match_the_oracle_on_every_engine() {
    let mut rng = Rng(0x0e03_7ee5);
    let mut runs = 0;
    for case in 0..80 {
        let len = (rng.next() % 40) as usize;
        let terms: Vec<F> = (0..len).map(|_| word(&mut rng)).collect();
        let init = if case % 3 == 0 {
            word(&mut rng)
        } else {
            F::POS_ZERO
        };
        for (name, tier) in [
            ("tree", FoldTier::Tree),
            ("exact", FoldTier::Exact),
            ("strict", FoldTier::Strict),
        ] {
            let expected = tier.reference(init, &terms).unwrap();
            assert!(relaxed_admits(init, &terms, expected) || terms.iter().any(|x| x.is_nan()));
            let workers: &[usize] = if case % 10 == 0 { &[1, 2, 7, 64] } else { &[2] };
            runs += every_engine(&program(name, init, &listed(&terms), expected), workers);
        }
    }
    assert!(runs >= 80 * 3 * 3, "{runs}");
}

#[test]
fn issue_eng_03_tiers_differ_where_the_oracle_says_they_do() {
    let one = F::from_bits(1.0f32.to_bits());
    let big = F::from_bits(1.0e8f32.to_bits());
    // 1e8 + 1 - 1e8 + 0.5: the left fold and the one-leaf tree lose the 1.
    let terms = [big, one, big.negate(), F::from_bits(0.5f32.to_bits())];
    let tree = FoldTier::Tree.reference(F::POS_ZERO, &terms).unwrap();
    let exact = FoldTier::Exact.reference(F::POS_ZERO, &terms).unwrap();
    assert_eq!(
        (tree.to_bits(), exact.to_bits()),
        (0.5f32.to_bits(), 1.5f32.to_bits())
    );
    let data = listed(&terms);
    every_engine(&program("tree", F::POS_ZERO, &data, tree), &[1, 2, 7, 64]);
    every_engine(&program("exact", F::POS_ZERO, &data, exact), &[1, 2, 7, 64]);
    // No iterations: every tier leaves `init`, -0 included.
    for tier in ["tree", "exact", "strict"] {
        every_engine(&program(tier, F::NEG_ZERO, &listed(&[]), F::NEG_ZERO), &[2]);
    }
}

/// 100,003 generated terms span many leaves and every worker split.
#[test]
fn issue_eng_03_large_tree_and_exact_folds_are_bit_identical_across_workers() {
    let n = 100_003u32;
    let value = |i: u32| {
        let h = i.wrapping_mul(2_654_435_761) % 2_000_001;
        F::from_bits((h as f32 / 1024.0 - 976.5625).to_bits())
    };
    let terms: Vec<F> = (0..n).map(value).collect();
    let data = format!(
        "var xs = filled_f32({n}u32, 0.0); for i in 0u32..{n}u32 {{ \
         xs[i] = f32_from_u32(wrapping_mul(i, 2654435761u32) % 2000001u32) / 1024.0 - 976.5625; }}"
    );
    for (name, tier) in [("tree", FoldTier::Tree), ("exact", FoldTier::Exact)] {
        let expected = tier.reference(F::POS_ZERO, &terms).unwrap();
        every_engine(&program(name, F::POS_ZERO, &data, expected), &[1, 2, 7, 64]);
    }
}

fn code(source: &str) -> &'static str {
    match check::check(front::parse(source.as_bytes()).unwrap(), true) {
        Err(error) => error.code,
        Ok(_) => "OK",
    }
}

#[test]
fn issue_eng_03_folds_no_tier_can_take_are_refused_not_run_strict() {
    let wrap = |body: &str| {
        format!(
            "#[numeric(fold = \"tree\")] fn f(xs: read Array<f32>) -> f32 {{ {body} }} \
             fn main() -> i32 {{ let xs: Array<f32> = [1.0, 2.0]; let r = f(xs); return 0; }}"
        )
    };
    let mut refused = 0;
    for body in [
        // Read elsewhere inside the loop.
        "var acc: f32 = 0.0; var out: f32 = 0.0; for i in 0u32..len(xs) { acc = acc + xs[i]; out = acc; } return out;",
        // Updated twice.
        "var acc: f32 = 0.0; for i in 0u32..len(xs) { acc = acc + xs[i]; acc = acc + 1.0; } return acc;",
        // A conditional update.
        "var acc: f32 = 0.0; for i in 0u32..len(xs) { if xs[i] > 1.0 { acc = acc + xs[i]; } } return acc;",
        // Nested loops.
        "var acc: f32 = 0.0; for j in 0u32..2u32 { for i in 0u32..len(xs) { acc = acc + xs[i]; } } return acc;",
        // A while loop.
        "var acc: f32 = 0.0; var i = 0u32; while i < len(xs) { acc = acc + xs[i]; i = i + 1u32; } return acc;",
        // Not `acc + term`, and an accumulator typed only by inference.
        "var acc: f32 = 0.0; for i in 0u32..len(xs) { acc = acc + xs[i] + 1.0; } return acc;",
        "var acc: f32 = 0.0; for i in 0u32..len(xs) { acc = acc - xs[i]; } return acc;",
        "var acc = xs[0u32]; for i in 0u32..len(xs) { acc = acc + xs[i]; } return acc;",
    ] {
        assert_eq!(code(&wrap(body)), "NUMERIC_POLICY", "{body}");
        refused += 1;
    }
    for body in [
        // A per-iteration accumulator is not a fold of this loop.
        "var out: f32 = 0.0; for i in 0u32..len(xs) { var s: f32 = 0.0; s = s + xs[i]; out = s; } return out;",
        // Integer folds and products keep their v1 rules.
        "var k = 0u32; var p: f32 = 1.0; for i in 0u32..len(xs) { k = k + 1u32; p = p * xs[i]; } return p;",
        // Two independent tiered folds in one loop.
        "var a: f32 = 0.0; var b = 0.0; for i in 0u32..len(xs) { a = a + xs[i]; b = xs[i] + b; } return a + b;",
    ] {
        assert_eq!(code(&wrap(body)), "OK", "{body}");
    }
    assert_eq!(refused, 8);
}

/// The source evaluator and MIR agree at every fuel prefix of a tiered fold.
#[test]
fn issue_eng_03_tiered_folds_keep_fuel_prefixes_on_reference_and_mir() {
    for tier in ["tree", "exact"] {
        let source = program(
            tier,
            F::POS_ZERO,
            "let xs: Array<f32> = [1.0, 2.0, 3.0];",
            F::from_bits(6.0f32.to_bits()),
        );
        let checked = check::check(front::parse(source.as_bytes()).unwrap(), true).unwrap();
        let lowered = mir::lower(&checked).unwrap();
        let complete = eval::observe(&checked, FUEL);
        assert_eq!(complete.result, Ok(eval::Value::I32(1)));
        for fuel in 0..=complete.steps_used + 1 {
            let reference = eval::observe(&checked, fuel);
            let actual = mir_eval::observe(&lowered, fuel);
            assert_eq!(actual.result, reference.result, "{tier} fuel={fuel}");
            assert_eq!(actual.steps_used, reference.steps_used);
            assert_eq!(actual.cleanup_trace, reference.cleanup_trace);
        }
    }
}

#[test]
fn issue_eng_03_compact_functions_with_many_folds_are_not_refused() {
    // Each fold generates dozens of nodes from fewer source bytes; their spans
    // must not run out in a compact function. (Counters are `j0`.. because
    // `i32` is a type name.)
    for tier in ["tree", "exact"] {
        let folds: String = (0..64)
            .map(|k| format!("var a{k}=0.0;for j{k} in 0u32..len(x){{a{k}=a{k}+x[j{k}];}}"))
            .collect();
        let total: String = (0..64).map(|k| format!("t=t+a{k};")).collect();
        let source = format!(
            "#[numeric(fold=\"{tier}\")]fn f(x:read Array<f32>)->f32{{{folds}var t=0.0;{total}return t;}}\
             fn main()->i32{{let x: Array<f32> = [1.0,2.0,3.0];if f(x)==384.0{{return 1;}}return 0;}}",
        );
        assert!(every_engine(&source, &[2]) >= 3, "{tier}");
    }
}

#[test]
fn issue_eng_03_generated_names_never_collide_with_source_names() {
    // A function may already use the names the lowering would generate.
    for tier in ["tree", "exact"] {
        let source = format!(
            "#[numeric(fold = \"{tier}\")] fn f(xs: read Array<f32>) -> f32 {{ \
             let __fold0_lo: u32 = 0u32; var __fold0_acc: f32 = 0.0; var acc: f32 = 0.0; \
             for i in __fold0_lo..len(xs) {{ acc = acc + xs[i]; }} \
             __fold0_acc = acc; return __fold0_acc; }} \
             fn __fold_0_n() -> u32 {{ return 0u32; }} \
             fn main() -> i32 {{ let xs: Array<f32> = [1.0, 2.0, 3.0]; \
             if f(xs) == 6.0 and __fold_0_n() == 0u32 {{ return 1; }} return 0; }}"
        );
        assert!(every_engine(&source, &[2]) >= 3, "{tier}");
    }
}

#[test]
fn issue_eng_03_folds_past_the_array_limit_trap_array_size_limit() {
    // The slot buffer of a fold over every u32 is past the array limit; the
    // Exact init slot must not overflow the count first (review finding).
    for tier in ["tree", "exact"] {
        let source = format!(
            "#[numeric(fold = \"{tier}\")] fn f() -> f32 {{ var acc: f32 = 0.0; \
             for i in 0u32..4294967295u32 {{ acc = acc + 1.0; }} return acc; }} \
             fn main() -> i32 {{ if f() == 0.0 {{ return 1; }} return 0; }}"
        );
        let checked = check::check(front::parse(source.as_bytes()).unwrap(), true).unwrap();
        let program = mir::lower(&checked).unwrap();
        let reference = eval::observe(&checked, FUEL);
        assert!(
            matches!(
                reference.result,
                Err(eval::EvalError::Tool {
                    code: "ARRAY_SIZE_LIMIT",
                    ..
                })
            ),
            "{tier}: {:?}",
            reference.result
        );
        let lowered = mir_eval::observe(&program, FUEL);
        assert_eq!(lowered.result, reference.result, "{tier}");
        assert_eq!(lowered.steps_used, reference.steps_used);
        let parallel = schedule::run_source(source.as_bytes(), &program, FUEL, 2).unwrap();
        assert_eq!(parallel.observation.result, reference.result, "{tier}");
    }
}

/// Under Relaxed, a fold the lowering can take runs as Tree on every engine,
/// and a shape it cannot take runs strict, which the contract admits, instead
/// of being refused (ENG-03 step 5').
#[test]
fn issue_eng_03_relaxed_folds_run_as_tree_and_other_sums_run_strict() {
    let mut rng = Rng(0x5e1a_0005);
    let mut engines = 0;
    for _ in 0..12 {
        let terms: Vec<F> = (0..rng.next() % 40).map(|_| word(&mut rng)).collect();
        let init = word(&mut rng);
        let expected = FoldTier::Tree.reference(init, &terms).unwrap();
        let source = program("relaxed", init, &listed(&terms), expected);
        engines += every_engine(&source, &[2, 7]);
    }
    assert!(engines >= 12 * 4);
    let mut strict = 0;
    for (body, sum) in [
        (
            "var acc: f32 = 0.0; var out: f32 = 0.0; for i in 0u32..len(xs) { acc = acc + xs[i]; out = acc; } return out;",
            "3.0",
        ),
        (
            "var acc: f32 = 0.0; for i in 0u32..len(xs) { acc = acc + xs[i]; acc = acc + 1.0; } return acc;",
            "5.0",
        ),
        (
            "var acc: f32 = 0.0; for i in 0u32..len(xs) { if xs[i] > 1.0 { acc = acc + xs[i]; } } return acc;",
            "2.0",
        ),
        (
            "var acc: f32 = 0.0; for j in 0u32..2u32 { for i in 0u32..len(xs) { acc = acc + xs[i]; } } return acc;",
            "6.0",
        ),
        (
            "var acc: f32 = 0.0; var i = 0u32; while i < len(xs) { acc = acc + xs[i]; i = i + 1u32; } return acc;",
            "3.0",
        ),
        (
            "var acc: f32 = 0.0; for i in 0u32..len(xs) { acc = acc + xs[i] + 1.0; } return acc;",
            "5.0",
        ),
        (
            "var acc: f32 = 0.0; for i in 0u32..len(xs) { acc = acc - xs[i]; } return acc;",
            "(-3.0)",
        ),
        (
            "var acc = xs[0u32]; for i in 0u32..len(xs) { acc = acc + xs[i]; } return acc;",
            "4.0",
        ),
    ] {
        let source = format!(
            "#[numeric(fold = \"relaxed\")] fn f(xs: read Array<f32>) -> f32 {{ {body} }} \
             fn main() -> i32 {{ let xs: Array<f32> = [1.0, 2.0]; if f(xs) == {sum} {{ return 1; }} return 0; }}"
        );
        assert!(every_engine(&source, &[2]) >= 3, "{body}");
        strict += 1;
    }
    assert_eq!(strict, 8);
}
