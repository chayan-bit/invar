//! invar-math-cr-v1 differential against an independent build of upstream
//! CORE-MATH: every upstream worst case, special values, 10^6 random inputs
//! per function and type, and every 256th binary32 encoding of each univariate
//! function, compared bit for bit (NaN as one class) with the reference and
//! MIR engines' `invar::math` entry points.
//!
//! Heavy and opt-in:
//! `INVAR_COREMATH_UPSTREAM=<core-math clone> cargo test --release --test
//! issue_v02_math_differential -- --ignored --nocapture`.
#[path = "../support/coremath.rs"]
mod coremath;

use coremath::{build, records, same, tempdir, transcendental, worst_cases};
use invar::{
    math::{MathFn, eval_f32, eval_f64},
    numeric_contract::{F32Bits, F64Bits},
};
use std::path::Path;

const RANDOM: u64 = 1_000_000;
const STRIDE: u64 = 256;
/// The polynomial coefficient the mutation test perturbs (the cubic term of
/// sin's fast path, off by 2^-20 relative) and its mutant.
const MUTATION: (&str, &str, &str) = (
    "binary64/sin/sin.c",
    "1.0 - 0x1.55555553068fp-3 * r2",
    "1.0 - 0x1.55556553068fp-3 * r2",
);

fn cpu(f: MathFn, binary64: bool, x: u64, y: u64) -> u64 {
    if binary64 {
        let args = [F64Bits::from_bits(x), F64Bits::from_bits(y)];
        eval_f64(f, &args[..f.arity()]).to_bits()
    } else {
        let args = [F32Bits::from_bits(x as u32), F32Bits::from_bits(y as u32)];
        u64::from(eval_f32(f, &args[..f.arity()]).to_bits())
    }
}

/// (inputs compared, mismatches, first mismatch) for one function and type.
fn compare(
    driver: &Path,
    f: MathFn,
    binary64: bool,
    random: u64,
    stride: u64,
) -> (u64, u64, String) {
    let (mut total, mut wrong, mut first) = (0u64, 0u64, String::new());
    records(driver, f, binary64, random, stride, 1 << 16, |batch| {
        for &[x, y, expected] in batch {
            total += 1;
            if !same(binary64, cpu(f, binary64, x, y), expected) {
                wrong += 1;
                if first.is_empty() {
                    first = format!("x={x:#x} y={y:#x} upstream={expected:#x}");
                }
            }
        }
    });
    (total, wrong, first)
}

#[test]
#[ignore = "heavy: needs INVAR_COREMATH_UPSTREAM and minutes of CPU"]
fn v02_math_matches_upstream_core_math_on_worst_cases_and_random_inputs() {
    let directory = tempdir("differential");
    let driver = build(&directory, None);
    let jobs: Vec<_> = [false, true]
        .into_iter()
        .flat_map(|b| transcendental().map(move |f| (f, b)))
        .collect();
    let results = std::thread::scope(|scope| {
        let chunks: Vec<_> = jobs.chunks(jobs.len().div_ceil(4)).collect();
        chunks
            .into_iter()
            .map(|chunk| {
                let driver = &driver;
                scope.spawn(move || {
                    chunk
                        .iter()
                        .map(|&(f, b)| (f, b, compare(driver, f, b, RANDOM, STRIDE)))
                        .collect::<Vec<_>>()
                })
            })
            .collect::<Vec<_>>()
            .into_iter()
            .flat_map(|handle| handle.join().unwrap())
            .collect::<Vec<_>>()
    });
    let mut failures = Vec::new();
    for (f, binary64, (total, wrong, first)) in &results {
        let wc = worst_cases(*f, *binary64).is_some();
        println!(
            "{} f{} inputs={total} worst_case_file={wc} mismatches={wrong} {first}",
            f.name(),
            if *binary64 { 64 } else { 32 }
        );
        if *wrong != 0 || *total < RANDOM {
            failures.push(f.name());
        }
    }
    std::fs::remove_dir_all(directory).unwrap();
    assert!(failures.is_empty(), "{failures:?}");
}

#[test]
#[ignore = "heavy: needs INVAR_COREMATH_UPSTREAM"]
fn v02_math_differential_catches_a_mutated_polynomial_coefficient() {
    let directory = tempdir("mutation");
    let driver = build(&directory, Some(MUTATION));
    let (total, wrong, first) = compare(&driver, MathFn::Sin, true, 0, 0);
    println!("mutant sin f64: inputs={total} mismatches={wrong} {first}");
    std::fs::remove_dir_all(directory).unwrap();
    assert!(
        wrong > 0,
        "the mutated coefficient went undetected over {total} inputs"
    );
}
