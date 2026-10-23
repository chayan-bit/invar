//! ENG-03 step 6 on a device: a Tree or Exact source fold lowers to a slot
//! store and one `sum_tree` or `sum_exact` call, and forced device runs
//! reduce on the device's fsum kernels with the reference's bits.
//! Shared by the CUDA and Metal binaries; the including module supplies `BACKEND` and `run_forced`.
use super::{BACKEND, run_forced};
use invar::numeric_contract::{F32Bits as F, FoldTier};
use invar::{check, eval, front, fsum::Kind, mir};

const FUEL: u64 = 50_000_000;

/// A finite nonzero `x` as a source literal with exactly its value.
fn literal(x: F) -> String {
    let text = format!("{}", f32::from_bits(x.to_bits()));
    if text.contains('.') {
        text
    } else {
        format!("{text}.0")
    }
}

/// A fold of `n` generated terms `(i % 97) * 0.1 - 4.0`, which cancel and
/// round differently under each tier, and the oracle's result for `tier`.
fn fold(tier: FoldTier, n: u32) -> (String, F) {
    let (tenth, four) = (
        F::from_bits(0.1f32.to_bits()),
        F::from_bits(4.0f32.to_bits()),
    );
    let terms: Vec<F> = (0..n)
        .map(|i| F::from_u32(i % 97).mul(tenth).sub(four))
        .collect();
    let expected = tier.reference(F::POS_ZERO, &terms).unwrap();
    let source = format!(
        "#[numeric(fold = \"{}\")] fn fold_sum(xs: read Array<f32>) -> f32 {{ var acc: f32 = 0.0; \
         for i in 0u32..len(xs) {{ acc = acc + xs[i]; }} return acc; }} \
         fn main() -> i32 {{ var xs = filled_f32({n}u32, 0.0); \
         for i in 0u32..{n}u32 {{ xs[i] = f32_from_u32(i % 97u32) * 0.1 - 4.0; }} \
         if fold_sum(xs) == {} {{ return 1; }} return 0; }}",
        tier.name(),
        literal(expected)
    );
    (source, expected)
}

#[test]
#[ignore = "requires a GPU (CUDA with NVRTC, or Apple Metal)"]
fn issue_eng_03_tree_and_exact_folds_reduce_on_the_device_with_reference_bits() {
    let mut runs = 0;
    for tier in [FoldTier::Tree, FoldTier::Exact] {
        for n in [1u32, 4096, 100_003] {
            let (source, expected) = fold(tier, n);
            assert!(!expected.is_nan() && !expected.is_infinite() && !expected.is_zero());
            let checked = check::check(front::parse(source.as_bytes()).unwrap(), true).unwrap();
            let program = mir::lower(&checked).unwrap();
            assert_eq!(
                eval::observe(&checked, FUEL).result,
                Ok(eval::Value::I32(1))
            );
            let run = run_forced(source.as_bytes(), &program, FUEL);
            assert_eq!(
                run.observation.result,
                Ok(eval::Value::I32(1)),
                "{tier:?} n={n}"
            );
            let kind = if tier == FoldTier::Tree {
                Kind::SumTree
            } else {
                Kind::SumExact
            };
            let reduced: Vec<_> = run.reductions.iter().filter(|r| r.kind == kind).collect();
            assert_eq!(reduced.len(), 1, "{:?}", run.reductions);
            assert_eq!(
                reduced[0].backend, BACKEND,
                "{tier:?} n={n}: {:?}",
                reduced[0]
            );
            runs += 1;
        }
    }
    assert_eq!(runs, 6);
}
