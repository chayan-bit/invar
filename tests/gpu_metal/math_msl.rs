//! invar-math-cr-v1 binary32 on a real Metal GPU, bit for bit against the upstream CORE-MATH build and the strict oracles.
use invar::math::{MathFn, msl_map_source};
use v02_math_msl_common::coremath::{build, tempdir, transcendental};

#[path = "../support/v02_math_msl_common.rs"]
mod v02_math_msl_common;
use v02_math_msl_common::{MUTATION, RANDOM, against_upstream, oracle, samples};

use invar::math::CUDA_KERNEL;
use invar_metal::{BitsMap, MetalDevice};

fn load(device: &mut MetalDevice, source: &str) -> BitsMap {
    device
        .load_bits_map(source, CUDA_KERNEL)
        .unwrap_or_else(|e| {
            let path = std::env::temp_dir().join("invar-math-failed.metal");
            std::fs::write(&path, source).unwrap();
            panic!("{e:?}\nsource kept at {}", path.display())
        })
}

fn run(device: &mut MetalDevice, map: &BitsMap, x: &[u64], y: &[u64], ms: &mut f64) -> Vec<u64> {
    let mut output = Vec::with_capacity(x.len());
    for (x, y) in x.chunks(1 << 18).zip(y.chunks(1 << 18)) {
        let result = device.run_bits_map(map, x, y).unwrap();
        *ms += result.kernel_ms;
        output.extend(result.output);
    }
    output
}

#[test]
#[ignore = "requires a Metal GPU and INVAR_COREMATH_UPSTREAM"]
fn math_metal_matches_upstream_core_math_bit_for_bit() {
    let directory = tempdir("metal");
    let driver = build(&directory, None);
    let mut device = MetalDevice::open_runtime_source().expect("METAL_UNAVAILABLE");
    let mut failures = Vec::new();
    for f in transcendental() {
        let map = load(&mut device, &msl_map_source(f));
        let mut ms = 0.0;
        let (total, wrong, first) = against_upstream(&driver, f, RANDOM, |x, y| {
            run(&mut device, &map, x, y, &mut ms)
        });
        println!(
            "metal {} f32 inputs={total} mismatches={wrong} compile_ms={:.0} kernel_ms={ms:.2} {first}",
            f.name(),
            map.compile_ms
        );
        if wrong != 0 || total < RANDOM {
            failures.push(f.name());
        }
    }
    std::fs::remove_dir_all(directory).unwrap();
    assert!(failures.is_empty(), "{failures:?}");
}

#[test]
#[ignore = "requires a Metal GPU and INVAR_COREMATH_UPSTREAM"]
fn math_metal_catches_a_mutated_polynomial_coefficient() {
    let directory = tempdir("metal-mutation");
    let driver = build(&directory, None);
    let source = msl_map_source(MathFn::Sin);
    assert_eq!(source.matches(MUTATION.0).count(), 1);
    let mut device = MetalDevice::open_runtime_source().expect("METAL_UNAVAILABLE");
    let map = load(&mut device, &source.replace(MUTATION.0, MUTATION.1));
    let mut ms = 0.0;
    let (total, wrong, first) = against_upstream(&driver, MathFn::Sin, 0, |x, y| {
        run(&mut device, &map, x, y, &mut ms)
    });
    println!("mutant metal sin f32: inputs={total} mismatches={wrong} {first}");
    std::fs::remove_dir_all(directory).unwrap();
    assert!(
        wrong > 0,
        "the mutated coefficient went undetected over {total} inputs"
    );
}

#[test]
#[ignore = "requires a Metal GPU"]
fn math_metal_exact_functions_match_the_strict_oracles() {
    let mut device = MetalDevice::open_runtime_source().expect("METAL_UNAVAILABLE");
    let (x, y) = samples(RANDOM as usize);
    let mut failures = Vec::new();
    for f in MathFn::ALL
        .into_iter()
        .filter(|f| invar::math::core_math(*f).is_none())
    {
        let map = load(&mut device, &msl_map_source(f));
        let mut ms = 0.0;
        let got = run(&mut device, &map, &x, &y, &mut ms);
        let wrong = (0..x.len())
            .filter(|&i| got[i] != oracle(f, x[i], y[i]))
            .count();
        println!(
            "metal {} f32 inputs={} mismatches={wrong} kernel_ms={ms:.2}",
            f.name(),
            x.len()
        );
        if wrong != 0 {
            failures.push(f.name());
        }
    }
    assert!(failures.is_empty(), "{failures:?}");
}
