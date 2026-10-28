//! ENG-03: numeric policy v2 fold tiers and their oracle.

#[path = "../support/normal_form.rs"]
mod support;

use invar::numeric_contract::{
    F32Bits as F, FoldTier, NumericPolicy, PolicyError, fold::relaxed_admits, round_rational,
};
use num_bigint::BigInt;
use num_rational::BigRational;

use crate::split_mix::Rng;

impl Rng {
    fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }
}

fn f(value: f32) -> F {
    F::from_bits(value.to_bits())
}

/// Exact value of a finite word, decoded independently of the oracle.
fn exact(x: F) -> BigRational {
    let bits = x.to_bits();
    let exponent = (bits >> 23) & 0xff;
    let fraction = i64::from(bits & 0x007f_ffff);
    let (significand, power) = if exponent == 0 {
        (fraction, -149)
    } else {
        (fraction | 1 << 23, exponent as i32 - 150)
    };
    let significand = if x.is_negative() {
        -significand
    } else {
        significand
    };
    let numerator = BigInt::from(significand);
    if power >= 0 {
        BigRational::from_integer(numerator << power as usize)
    } else {
        BigRational::new(numerator, BigInt::from(1) << (-power) as usize)
    }
}

#[test]
fn issue_eng_03_policy_v2_accepts_the_four_tiers_and_refuses_the_rest() {
    let tiers = [
        ("f32-strict", NumericPolicy::STRICT_F32, FoldTier::Strict),
        ("f32-tree", NumericPolicy::TREE_F32, FoldTier::Tree),
        ("f32-exact", NumericPolicy::EXACT_F32, FoldTier::Exact),
        ("f32-relaxed", NumericPolicy::RELAXED_F32, FoldTier::Relaxed),
    ];
    for (kind, policy, tier) in tiers {
        assert_eq!(NumericPolicy::from_parts(2, kind), Ok(policy));
        assert_eq!(policy.fold(), tier);
        assert_eq!(NumericPolicy::f32_fold(tier), policy);
        assert_eq!(FoldTier::from_name(tier.name()), Some(tier));
        // The serialized fields select the same policy again.
        let json = policy.to_json();
        let version = u32::try_from(json["version"].as_u64().unwrap()).unwrap();
        assert_eq!(
            NumericPolicy::from_parts(version, json["kind"].as_str().unwrap()),
            Ok(policy)
        );
    }
    assert_eq!(
        NumericPolicy::from_parts(2, "f64-strict"),
        Ok(NumericPolicy::STRICT_F64)
    );

    // Strict v2 is the v1 identity, so frozen receipt ids do not move.
    assert_eq!(
        format!("{:?}", NumericPolicy::STRICT_F32),
        "NumericPolicy { version: 1 }"
    );
    assert_eq!(
        format!("{:?}", NumericPolicy::STRICT_F64),
        "NumericPolicy { version: 1, binary64: true }"
    );
    assert_eq!(
        format!("{:?}", NumericPolicy::TC_F16_F32),
        "NumericPolicy { version: 1, tc: true }"
    );
    assert_eq!(
        format!("{:?}", NumericPolicy::TREE_F32),
        "NumericPolicy { version: 2, fold: Tree }"
    );
    let ids = [
        NumericPolicy::TREE_F32,
        NumericPolicy::EXACT_F32,
        NumericPolicy::RELAXED_F32,
    ]
    .map(NumericPolicy::id);
    assert_eq!(ids, ["f32-tree-v2", "f32-exact-v2", "f32-relaxed-v2"]);
    assert_eq!(
        NumericPolicy::RELAXED_F32.to_json(),
        serde_json::json!({"version": 2, "kind": "f32-relaxed"})
    );

    let mut refused = 0;
    for kind in [
        "",
        "fast",
        "f32-fast",
        "f32-reassoc_add",
        "tree",
        "f32-tree-v2",
        "F32-tree",
        "f64-tree",
        "f64-relaxed",
        "strict-f32",
        "tc-f16-f32",
    ] {
        assert_eq!(
            NumericPolicy::from_parts(2, kind),
            Err(PolicyError::UnsupportedKind),
            "{kind}"
        );
        refused += 1;
    }
    for kind in ["f32-tree", "f32-strict", "fast"] {
        assert_eq!(
            NumericPolicy::from_parts(1, kind),
            Err(PolicyError::UnsupportedKind)
        );
        refused += 1;
    }
    for version in [0, 3, u32::MAX] {
        for kind in ["f32-tree", "strict-f32"] {
            assert_eq!(
                NumericPolicy::from_parts(version, kind),
                Err(PolicyError::UnsupportedVersion(version))
            );
            refused += 1;
        }
    }
    assert_eq!(FoldTier::from_name("fast"), None);
    assert_eq!(refused, 20);
}

#[test]
fn issue_eng_03_tier_references_name_one_result() {
    let init = F::POS_ZERO;
    let cancel = [f(1.0e8), f(1.0), f(-1.0e8)];
    assert_eq!(FoldTier::Strict.reference(init, &cancel), Some(F::POS_ZERO));
    assert_eq!(FoldTier::Tree.reference(init, &cancel), Some(F::POS_ZERO));
    assert_eq!(FoldTier::Exact.reference(init, &cancel), Some(f(1.0)));
    assert_eq!(FoldTier::Relaxed.reference(init, &cancel), None);

    // No terms: every tier returns `init`, including -0.
    for tier in [FoldTier::Strict, FoldTier::Tree, FoldTier::Exact] {
        assert_eq!(tier.reference(F::NEG_ZERO, &[]), Some(F::NEG_ZERO));
        assert_eq!(tier.reference(f(2.5), &[]), Some(f(2.5)));
    }

    // Tree is `init + sum_tree(terms)`: 17 terms span three leaves.
    let terms: Vec<F> = (0..17).map(|i| f(1.0 + i as f32 * 2.0e-7)).collect();
    let tree = invar::fsum_reference::sum_tree(&terms);
    assert_eq!(
        FoldTier::Tree.reference(f(3.0), &terms),
        Some(f(3.0).add(tree))
    );
    assert_ne!(
        FoldTier::Tree.reference(F::POS_ZERO, &terms),
        FoldTier::Strict.reference(F::POS_ZERO, &terms),
        "the fixture separates the tree shape from the left fold"
    );
}

/// One seeded fold input: `init`, then the terms.
fn sample(rng: &mut Rng, case: u64) -> (F, Vec<F>) {
    let len = rng.below(24) as usize;
    let word = |rng: &mut Rng| -> F {
        match case {
            // Cancellation: large opposite pairs around small values.
            0 => {
                let big = f((rng.below(1 << 20) as f32 + 1.0) * 1.0e6);
                if rng.below(3) == 0 {
                    f(rng.below(1000) as f32 * 1.0e-3)
                } else if rng.below(2) == 0 {
                    big
                } else {
                    big.negate()
                }
            }
            // Subnormals and the smallest normals.
            1 => F::from_bits(rng.next() as u32 & 0x80ff_ffff),
            // Signed zeros only, or with one small value.
            2 => [F::POS_ZERO, F::NEG_ZERO, F::NEG_ZERO, f(1.0e-30)][rng.below(4) as usize],
            // Infinities among finite values.
            3 => [F::INFINITY, F::NEG_INFINITY, f(1.0), f(-7.5)][rng.below(4) as usize],
            // NaN class among finite values.
            4 => [F::NAN, F::from_bits(0xffc0_1234), f(4.0), f(-0.0)][rng.below(4) as usize],
            // Any finite word, including the largest magnitudes.
            _ => loop {
                let x = F::from_bits(rng.next() as u32);
                if !x.is_nan() && !x.is_infinite() {
                    break x;
                }
            },
        }
    };
    let init = if rng.below(2) == 0 {
        F::POS_ZERO
    } else {
        word(rng)
    };
    (init, (0..len).map(|_| word(rng)).collect())
}

#[test]
fn issue_eng_03_strict_tree_and_exact_stay_inside_the_relaxed_bound() {
    let mut rng = Rng(0x3e03_0280);
    let mut by_case = [0usize; 6];
    let mut exact_checked = 0;
    for n in 0..10_000u64 {
        let case = n % 6;
        let (init, terms) = sample(&mut rng, case);
        for tier in [FoldTier::Strict, FoldTier::Tree, FoldTier::Exact] {
            let result = tier.reference(init, &terms).unwrap();
            assert!(
                relaxed_admits(init, &terms, result),
                "case {case} {tier:?} init {:08x} terms {:08x?} result {:08x}",
                init.to_bits(),
                terms.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
                result.to_bits()
            );
        }
        // Exact equals the once-rounded rational sum on finite inputs.
        let summands: Vec<F> = std::iter::once(init).chain(terms.iter().copied()).collect();
        if summands.iter().all(|x| !x.is_nan() && !x.is_infinite()) {
            let total: BigRational = summands.iter().map(|&x| exact(x)).sum();
            let all_negative_zero = summands
                .iter()
                .all(|x| x.to_bits() == F::NEG_ZERO.to_bits());
            let expected = round_rational(total, all_negative_zero);
            let got = FoldTier::Exact.reference(init, &terms).unwrap();
            assert!(got.strict_eq(expected), "exact {:08x}", got.to_bits());
            exact_checked += 1;
        }
        by_case[case as usize] += 1;
    }
    assert!(by_case.iter().all(|&count| count > 1_600), "{by_case:?}");
    assert!(exact_checked > 6_000, "{exact_checked}");
}

#[test]
fn issue_eng_03_relaxed_refuses_results_outside_the_contract() {
    let one = f(1.0);
    let mut refused = 0;
    let mut refuse = |init: F, terms: &[F], result: F| {
        assert!(
            !relaxed_admits(init, terms, result),
            "{:08x}",
            result.to_bits()
        );
        refused += 1;
    };
    // NaN and infinities: the class is fixed.
    refuse(F::POS_ZERO, &[one, F::NAN], one);
    refuse(F::POS_ZERO, &[one, F::NAN], F::INFINITY);
    refuse(F::POS_ZERO, &[F::INFINITY, F::NEG_INFINITY], F::INFINITY);
    refuse(F::POS_ZERO, &[F::INFINITY, one], F::NEG_INFINITY);
    refuse(F::POS_ZERO, &[F::INFINITY, one], one);
    refuse(F::POS_ZERO, &[one, one], F::INFINITY);
    refuse(F::POS_ZERO, &[one, one], F::NAN);
    // Signed zero: -0 exactly when every summand is -0.
    refuse(F::NEG_ZERO, &[F::NEG_ZERO, F::NEG_ZERO], F::POS_ZERO);
    refuse(F::POS_ZERO, &[one, one.negate()], F::NEG_ZERO);
    // N = 4 summands (init and 3 terms) with magnitude sum 2^25 + 1: the
    // bound is 3 * (2^25 + 1) * 2^-24, just above 6, around the exact sum 1.
    let big = f(16_777_216.0);
    let terms = [big, one, big.negate()];
    assert!(relaxed_admits(F::POS_ZERO, &terms, f(7.0)));
    assert!(relaxed_admits(F::POS_ZERO, &terms, f(-5.0)));
    refuse(F::POS_ZERO, &terms, f(7.25));
    refuse(F::POS_ZERO, &terms, f(-5.25));
    // Exact sums leave no room: a single term is exact.
    assert!(relaxed_admits(F::POS_ZERO, &[f(0.1)], f(0.1)));
    refuse(F::POS_ZERO, &[f(0.1)], F::from_bits(f(0.1).to_bits() + 1));
    assert_eq!(refused, 12);
    // Outside the finite domain any IEEE value is admitted.
    let max = F::from_bits(0x7f7f_ffff);
    for result in [F::NAN, F::INFINITY, F::NEG_INFINITY, F::POS_ZERO] {
        assert!(relaxed_admits(
            F::POS_ZERO,
            &[max, max, max.negate()],
            result
        ));
    }
}

fn front_code(source: &str) -> &'static str {
    match invar::front::parse(source.as_bytes()) {
        Err(error) => error.code,
        Ok(parsed) => match invar::check::check(parsed, true) {
            Err(error) => error.code,
            Ok(_) => "OK",
        },
    }
}

const TOTAL: &str = "fn fold_sum(xs: read Array<f32>) -> f32 { var acc: f32 = 0.0; \
    for i in 0u32..len(xs) { acc = UPDATE; } return acc; }";

fn fold_program(attribute: &str, update: &str) -> String {
    format!(
        "{attribute} {} fn main() -> i32 {{ let xs: Array<f32> = [100000000.0, 1.0, -100000000.0, 0.5]; \
         if fold_sum(xs) == 0.5 {{ return 7; }} return 0; }}",
        TOTAL.replace("UPDATE", update)
    )
}

#[test]
fn issue_eng_03_numeric_attribute_declares_a_tier_and_refuses_the_rest() {
    use invar::front::parse;
    let tiers = [
        ("strict", FoldTier::Strict),
        ("relaxed", FoldTier::Relaxed),
        ("tree", FoldTier::Tree),
        ("exact", FoldTier::Exact),
    ];
    for (name, tier) in tiers {
        let source = fold_program(&format!("#[numeric(fold = \"{name}\")]"), "acc + xs[i]");
        let parsed = parse(source.as_bytes()).unwrap();
        let folds: Vec<_> = parsed
            .functions
            .iter()
            .map(|f| (f.name.as_str(), f.fold))
            .collect();
        assert_eq!(folds, [("fold_sum", tier), ("main", FoldTier::Strict)]);
    }
    let public = "module m { #[numeric(fold = \"relaxed\")] pub fn one() -> f32 { return 1.0; } }";
    assert_eq!(
        parse(public.as_bytes()).unwrap().functions[0].fold,
        FoldTier::Relaxed
    );

    let mut refused = 0;
    for (attribute, code) in [
        ("#[numeric(fold = \"fast\")]", "NUMERIC_POLICY"),
        ("#[numeric(fold = \"\")]", "NUMERIC_POLICY"),
        ("#[numeric(fold = \"Tree\")]", "NUMERIC_POLICY"),
        ("#[numeric(fold = \"reassoc_add\")]", "NUMERIC_POLICY"),
        ("#[numeric(fold = tree)]", "PARSE_EXPECTED_TOKEN"),
        ("#[numeric(order = \"tree\")]", "UNSUPPORTED_FEATURE"),
        ("#[inline]", "UNSUPPORTED_FEATURE"),
        (
            "#[numeric(fold = \"relaxed\")] struct S { x: f32 }",
            "UNSUPPORTED_FEATURE",
        ),
        ("#[numeric(fold = \"relaxed\")]", "OK"),
        ("#[numeric(fold = \"strict\")]", "OK"),
        ("#[numeric(fold = \"tree\")]", "OK"),
        ("#[numeric(fold = \"exact\")]", "OK"),
    ] {
        let source = fold_program(attribute, "acc + xs[i]");
        assert_eq!(front_code(&source), code, "{attribute}");
        refused += usize::from(code != "OK");
    }
    assert_eq!(refused, 8);
}

/// A Relaxed fold, and a `reassoc_add` fold in a Strict or Relaxed function,
/// takes the Tree lowering (ENG-03 step 5'): on the source evaluator and MIR it
/// has the Tree program's outcome and fuel at every fuel prefix, while a plain
/// `+` fold in a Strict function stays strict.
#[test]
fn issue_eng_03_relaxed_folds_lower_as_tree_with_identical_fuel() {
    use invar::{check, eval, front, mir, mir_eval};
    let terms = [f(1.0e8), f(1.0), f(-1.0e8), f(0.5)];
    let tree_sum = FoldTier::Tree.reference(F::POS_ZERO, &terms).unwrap();
    let strict_sum = FoldTier::Strict.reference(F::POS_ZERO, &terms).unwrap();
    // `fold_program` returns 7 exactly when the fold gives 0.5.
    let outcome = |sum: F| Ok(eval::Value::I32(if sum == f(0.5) { 7 } else { 0 }));
    let observe = |source: &str, expected: Result<eval::Value, eval::EvalError>| {
        let checked = check::check(front::parse(source.as_bytes()).unwrap(), true).unwrap();
        invar::ownership::check(&checked).unwrap();
        let program = mir::lower(&checked).unwrap();
        let complete = eval::observe(&checked, 100_000);
        assert_eq!(complete.result, expected, "{source}");
        let runs: Vec<_> = (0..=complete.steps_used + 1)
            .map(|fuel| {
                let reference = eval::observe(&checked, fuel);
                let actual = mir_eval::observe(&program, fuel);
                assert_eq!(actual.result, reference.result, "fuel={fuel}: {source}");
                assert_eq!(actual.steps_used, reference.steps_used, "{source}");
                (format!("{:?}", reference.result), reference.steps_used)
            })
            .collect();
        runs
    };
    let tree = observe(
        &fold_program("#[numeric(fold = \"tree\")]", "acc + xs[i]"),
        outcome(tree_sum),
    );
    let plain = observe(&fold_program("", "acc + xs[i]"), outcome(strict_sum));
    assert_ne!(plain, tree, "the Tree program runs a slot buffer");
    let mut variants = 0;
    for (attribute, update) in [
        ("", "reassoc_add(acc, xs[i])"),
        ("#[numeric(fold = \"relaxed\")]", "acc + xs[i]"),
        ("#[numeric(fold = \"relaxed\")]", "reassoc_add(acc, xs[i])"),
        ("#[numeric(fold = \"strict\")]", "reassoc_add((acc), xs[i])"),
    ] {
        let source = fold_program(attribute, update);
        assert_eq!(
            observe(&source, outcome(tree_sum)),
            tree,
            "{attribute} {update}"
        );
        variants += 1;
    }
    assert_eq!(variants, 4);
    // Both results are inside the Relaxed contract of the same fold.
    assert!(relaxed_admits(F::POS_ZERO, &terms, tree_sum));
    assert!(relaxed_admits(F::POS_ZERO, &terms, strict_sum));
}

#[test]
fn issue_eng_03_reassoc_add_relaxes_only_an_additive_fold_update() {
    let mut refused = 0;
    for update in [
        "reassoc_add(acc * 2.0, xs[i])",
        "reassoc_add(xs[i], acc)",
        "acc * reassoc_add(acc, xs[i])",
        "reassoc_add(acc, reassoc_add(acc, xs[i]))",
        "reassoc_add(acc, xs[i], 1.0)",
        "reassoc_add(acc)",
        // The loop uses `acc` elsewhere.
        "reassoc_add(acc, acc)",
        "reassoc_add(acc, xs[i] + acc)",
    ] {
        assert_eq!(
            front_code(&fold_program("", update)),
            "NUMERIC_POLICY",
            "{update}"
        );
        refused += 1;
    }
    for source in [
        // An expression, not a fold update.
        "fn main() -> i32 { var acc: f32 = 0.0; let y = reassoc_add(acc, 1.0); return 7; }",
        "fn main() -> i32 { var acc: f32 = 0.0; if reassoc_add(acc, 1.0) == 1.0 { return 7; } return 0; }",
        // Not an f32 fold.
        "fn main() -> i32 { var acc: i32 = 0; for i in 0u32..3u32 { acc = reassoc_add(acc, 1); } return acc; }",
        // Outside a loop, or the loop reads `acc` again.
        "fn main() -> i32 { var acc: f32 = 0.0; acc = reassoc_add(acc, 1.0); return 7; }",
        "fn main() -> i32 { var acc: f32 = 0.0; var last: f32 = 0.0; \
         for i in 0u32..3u32 { acc = reassoc_add(acc, 1.0); last = acc; } return 7; }",
        "fn main() -> i32 { var acc: f32 = 0.0; while acc < 3.0 { acc = reassoc_add(acc, 1.0); } return 7; }",
    ] {
        assert_eq!(front_code(source), "NUMERIC_POLICY", "{source}");
        refused += 1;
    }
    for source in [
        // An update under `if`, and a read after the inner loop ends.
        "fn main() -> i32 { var acc: f32 = 0.0; \
         for i in 0u32..3u32 { if i > 0u32 { acc = reassoc_add(acc, 1.0); } } return 7; }",
        "fn main() -> i32 { var acc: f32 = 0.0; var last: f32 = 0.0; \
         for r in 0u32..2u32 { for c in 0u32..3u32 { acc = reassoc_add(acc, 1.0); } last = acc; } return 7; }",
    ] {
        assert_eq!(front_code(source), "OK", "{source}");
    }
    // The name is reserved.
    let shadow =
        "fn reassoc_add(a: f32, b: f32) -> f32 { return a; } fn main() -> i32 { return 7; }";
    assert_ne!(front_code(shadow), "OK");
    assert_eq!(refused, 14);
}

#[test]
fn issue_eng_03_a_caller_tier_never_reaches_its_callee() {
    let source = "fn callee(x: f32) -> f32 { var acc: f32 = 0.0; acc = acc + x; return acc; } \
        #[numeric(fold = \"relaxed\")] fn main() -> i32 { \
        let k: i32 = 1; \
        let c = closure [copy k: i32] (y: f32) -> f32 { return y + f32_from_i32(k); }; \
        let y = callee(c(0.0)); var acc: f32 = 0.0; for j in 0u32..1u32 { acc = reassoc_add(acc, y); } \
        if acc == 1.0 { return 7; } return 0; }";
    let checked = invar::check::check(invar::front::parse(source.as_bytes()).unwrap(), true);
    let checked = checked.unwrap();
    let folds: std::collections::BTreeMap<_, _> = checked
        .program
        .functions
        .iter()
        .map(|f| (f.name.clone(), f.fold))
        .collect();
    assert_eq!(folds["main"], FoldTier::Relaxed);
    assert_eq!(folds["callee"], FoldTier::Strict);
    // The lifted closure keeps strict folds too.
    let lifted: Vec<_> = folds
        .iter()
        .filter(|(name, _)| !matches!(name.as_str(), "main" | "callee"))
        .collect();
    assert!(
        !lifted.is_empty() && lifted.iter().all(|(_, fold)| **fold == FoldTier::Strict),
        "{folds:?}"
    );
}

use invar::numeric_contract::{
    NumericTrap,
    storage::{Bf16Bits, F16Bits, narrow_i8, narrow_u8},
};

#[test]
fn issue_eng_03_every_f16_and_bf16_pattern_round_trips() {
    let (mut f16_values, mut bf16_values) = (0, 0);
    for bits in 0..=u16::MAX {
        let half = F16Bits::from_bits(bits);
        let wide = half.widen();
        if half.is_nan() {
            assert!(wide.is_nan() && F16Bits::round(wide).is_nan(), "{bits:04x}");
        } else {
            assert_eq!(F16Bits::round(wide), half, "{bits:04x}");
            let check = half::f16::from_bits(bits).to_f32().to_bits();
            assert_eq!(wide.to_bits(), check, "f16 {bits:04x}");
            f16_values += 1;
        }
        let brain = Bf16Bits::from_bits(bits);
        let wide = brain.widen();
        if brain.is_nan() {
            assert!(
                wide.is_nan() && Bf16Bits::round(wide).is_nan(),
                "{bits:04x}"
            );
        } else {
            assert_eq!(Bf16Bits::round(wide), brain, "{bits:04x}");
            assert_eq!(wide.to_bits(), u32::from(bits) << 16, "bf16 {bits:04x}");
            let check = half::bf16::from_bits(bits).to_f32().to_bits();
            assert_eq!(wide.to_bits(), check, "bf16 {bits:04x}");
            bf16_values += 1;
        }
    }
    // Every encoding but the 2 * (2^10 - 1) and 2 * (2^7 - 1) NaNs.
    assert_eq!(f16_values, 65_536 - 2_046);
    assert_eq!(bf16_values, 65_536 - 254);
}

/// Round `value` with the oracle and with `half` (non-NaN results compared).
fn rounds(value: F) -> (u16, u16) {
    let host = f32::from_bits(value.to_bits());
    let f16 = F16Bits::round(value).to_bits();
    let bf16 = Bf16Bits::round(value).to_bits();
    if !value.is_nan() {
        assert_eq!(
            f16,
            half::f16::from_f32(host).to_bits(),
            "f16 {:08x}",
            value.to_bits()
        );
        assert_eq!(
            bf16,
            half::bf16::from_f32(host).to_bits(),
            "bf16 {:08x}",
            value.to_bits()
        );
    }
    (f16, bf16)
}

#[test]
fn issue_eng_03_f32_rounds_to_nearest_even_at_every_boundary() {
    let mut boundaries = 0;
    // Each pair of adjacent positive narrow values, including the subnormals
    // and the overflow threshold (one step past the largest finite value).
    let pairs =
        |widen: fn(u16) -> F, top: u16| (0..top).map(move |low| (low, widen(low), widen(low + 1)));
    let f16_pairs = pairs(|b| F16Bits::from_bits(b).widen(), 0x7c00);
    let bf16_pairs = pairs(|b| Bf16Bits::from_bits(b).widen(), 0x7f80);
    for (is_f16, (low, below, above)) in f16_pairs
        .map(|p| (true, p))
        .chain(bf16_pairs.map(|p| (false, p)))
    {
        let high = low + 1;
        // Past the largest finite f16 the next step would be 2^16; bf16's
        // would be 2^128, whose midpoint with the largest finite is below.
        let above = if is_f16 && above.is_infinite() {
            F::from_bits(0x4780_0000)
        } else {
            above
        };
        let mid = if above.is_infinite() {
            F::from_bits(0x7f7f_8000)
        } else {
            // Halving first keeps the top binade finite; every halving is exact.
            below.mul(f(0.5)).add(above.mul(f(0.5)))
        };
        let pick = |value: F| {
            let (f16, bf16) = rounds(value);
            if is_f16 { f16 } else { bf16 }
        };
        let even = if low % 2 == 0 { low } else { high };
        for (value, want) in [
            (mid, even),
            (F::from_bits(mid.to_bits() - 1), low),
            (F::from_bits(mid.to_bits() + 1), high),
        ] {
            assert_eq!(pick(value), want, "{is_f16} {:08x}", value.to_bits());
            assert_eq!(pick(value.negate()), want | 0x8000);
        }
        boundaries += 1;
    }
    assert_eq!(boundaries, 0x7c00 + 0x7f80);
    // Seeded words, specials and the largest f32 values.
    let mut rng = Rng(0x0e03_f16b);
    for _ in 0..100_000 {
        rounds(F::from_bits(rng.next() as u32));
    }
    for value in [
        F::INFINITY,
        F::NEG_INFINITY,
        F::POS_ZERO,
        F::NEG_ZERO,
        F::from_bits(0x7f7f_ffff),
    ] {
        rounds(value);
    }
    assert!(F16Bits::round(F::NAN).is_nan() && Bf16Bits::round(F::NAN).is_nan());
}

#[test]
fn issue_eng_03_integer_narrowing_traps_outside_the_range() {
    let mut stored = 0;
    for value in -1_000..=1_000_i32 {
        let fits = (-128..=127).contains(&value);
        match narrow_i8(value) {
            Ok(narrow) => {
                assert!(fits && i32::from(narrow) == value);
                stored += 1;
            }
            Err(trap) => assert!(!fits && trap == NumericTrap::IntegerOverflow),
        }
    }
    for value in 0..=1_000_u32 {
        match narrow_u8(value) {
            Ok(narrow) => {
                assert!(value <= 255 && u32::from(narrow) == value);
                stored += 1;
            }
            Err(trap) => assert!(value > 255 && trap.category() == "INTEGER_OVERFLOW"),
        }
    }
    assert_eq!(stored, 256 + 256);
    for value in [i32::MIN, -129, 128, i32::MAX] {
        assert_eq!(narrow_i8(value), Err(NumericTrap::IntegerOverflow));
    }
    for value in [256, u32::MAX] {
        assert_eq!(narrow_u8(value), Err(NumericTrap::IntegerOverflow));
    }
}

#[test]
fn issue_eng_03_a_strict_f32_fold_stays_serial_in_the_normal_form() {
    use invar::general_net::normal_form::{self, Node};
    // A strict fold is never reassociated: the parallel normal form keeps
    // refusing it with the v1 reason. Tiered folds never reach it as a float
    // sum, because the checker lowers them first (docs/numeric.md).
    let mut serial = 0;
    for attribute in ["", "#[numeric(fold = \"strict\")]"] {
        let source = format!(
            "{attribute} fn f(xs: read Array<f32>) -> f32 {{ var acc: f32 = 0.0; \
             for i in 0u32..len(xs) {{ acc = acc + xs[i]; }} return acc; }}"
        );
        let (net, proof) = support::compile(&source);
        let form = normal_form::extract(&net, &proof).expect("extract");
        assert_eq!(
            form.functions[0].loops[0].node,
            Node::Serial {
                reason: "needs f32 reassociation (strict f32 order)".to_owned()
            },
            "{source}"
        );
        serial += 1;
    }
    assert_eq!(serial, 2);
}
