//! invar-math-cr-v1 on CUDA, bit for bit against independent oracles.
//!
//! CORE-MATH functions run the vendored C compiled by NVRTC and are compared
//! with an upstream CORE-MATH build on its worst cases, specials, 10^6 random
//! inputs per function and type and every 256th binary32 encoding. The exact
//! functions are compared with the strict oracles. A mutated polynomial
//! coefficient in the device source must be caught.
//!
//! `INVAR_COREMATH_UPSTREAM=<core-math clone> cargo test --release --features
//! cuda --test gpu_cuda math_cr_v1:: -- --ignored --nocapture --test-threads 1`.

#[path = "../support/coremath.rs"]
mod coremath;

use coremath::{build, records, same, tempdir, transcendental};
use invar::{
    math::{
        CUDA_KERNEL, MathFn, cuda_map_min_nvrtc, cuda_map_source, eval_f32, eval_f64,
        load_cuda_map, nvrtc_admits,
    },
    numeric_contract::{F32Bits, F64Bits},
};
use invar_cuda::cuda::{BitsMap, CudaDevice};
use std::path::Path;

const RANDOM: u64 = 1_000_000;
const STRIDE: u64 = 256;
const CHUNK: usize = 1 << 22;
/// The cubic coefficient of sin's fast path, off by 2^-20 relative.
/// The probe runs the same inputs as the exactness test: worst cases plus
/// 10^6 random samples.
const MUTATION: (&str, &str) = (
    "1.0 - 0x1.55555553068fp-3 * r2",
    "1.0 - 0x1.55556553068fp-3 * r2",
);

fn open() -> CudaDevice {
    CudaDevice::open()
        .expect("CUDA_UNAVAILABLE: requires CUDA GPU and NVRTC 12.4+ (12.9+ for binary64)")
}

/// Load `source`, which needs NVRTC `required`, refusing an older NVRTC first.
fn load(device: &mut CudaDevice, source: &str, required: (u32, u32)) -> BitsMap {
    nvrtc_admits(required).unwrap_or_else(|e| panic!("{e:?}"));
    device
        .load_bits_map(source, &invar_math::CUDA_OPTIONS, CUDA_KERNEL)
        .unwrap_or_else(|e| {
            let path = std::env::temp_dir().join("invar-math-failed.cu");
            std::fs::write(&path, source).unwrap();
            panic!("{e:?}\nsource kept at {}", path.display())
        })
}

/// (inputs, mismatches, first mismatch, kernel ms) of `map` on the upstream
/// driver's records.
fn against_upstream(
    device: &mut CudaDevice,
    map: &BitsMap,
    driver: &Path,
    f: MathFn,
    b: bool,
    random: u64,
) -> (u64, u64, String, f64) {
    let (mut total, mut wrong, mut first, mut ms) = (0u64, 0u64, String::new(), 0.0);
    records(driver, f, b, random, STRIDE, CHUNK, |batch| {
        let x: Vec<u64> = batch.iter().map(|r| r[0]).collect();
        let y: Vec<u64> = batch.iter().map(|r| r[1]).collect();
        let run = device.run_bits_map(map, &x, &y).unwrap();
        ms += f64::from(run.kernel_ms);
        for (record, got) in batch.iter().zip(run.output) {
            total += 1;
            if !same(b, got, record[2]) {
                wrong += 1;
                if first.is_empty() {
                    first = format!(
                        "x={:#x} y={:#x} cuda={got:#x} upstream={:#x}",
                        record[0], record[1], record[2]
                    );
                }
            }
        }
    });
    (total, wrong, first, ms)
}

#[test]
#[ignore = "requires a CUDA GPU, NVRTC 12.9+ and INVAR_COREMATH_UPSTREAM"]
fn math_cr_v1_cuda_matches_upstream_core_math_bit_for_bit() {
    let directory = tempdir("cuda");
    let driver = build(&directory, None);
    let mut device = open();
    let mut failures = Vec::new();
    for b in [false, true] {
        for f in transcendental() {
            let map = load(
                &mut device,
                &cuda_map_source(f, b),
                cuda_map_min_nvrtc(f, b),
            );
            let (total, wrong, first, ms) =
                against_upstream(&mut device, &map, &driver, f, b, RANDOM);
            println!(
                "cuda {} f{} inputs={total} mismatches={wrong} compile_ms={:.0} kernel_ms={ms:.2} {first}",
                f.name(),
                if b { 64 } else { 32 },
                map.compile_ms
            );
            if wrong != 0 || total < RANDOM {
                failures.push(format!("{}{}", f.name(), if b { 64 } else { 32 }));
            }
        }
    }
    std::fs::remove_dir_all(directory).unwrap();
    assert!(failures.is_empty(), "{failures:?}");
}

#[test]
#[ignore = "requires a CUDA GPU, NVRTC 12.9+ and INVAR_COREMATH_UPSTREAM"]
fn math_cr_v1_cuda_differential_catches_a_mutated_polynomial_coefficient() {
    let directory = tempdir("cuda-mutation");
    let driver = build(&directory, None);
    let source = cuda_map_source(MathFn::Sin, true);
    assert_eq!(source.matches(MUTATION.0).count(), 1);
    let mut device = open();
    let map = load(
        &mut device,
        &source.replace(MUTATION.0, MUTATION.1),
        cuda_map_min_nvrtc(MathFn::Sin, true),
    );
    let (total, wrong, first, _) =
        against_upstream(&mut device, &map, &driver, MathFn::Sin, true, RANDOM);
    println!("mutant cuda sin f64: inputs={total} mismatches={wrong} {first}");
    std::fs::remove_dir_all(directory).unwrap();
    assert!(
        wrong > 0,
        "the mutated coefficient went undetected over {total} inputs"
    );
}

fn samples(binary64: bool, n: u64) -> Vec<u64> {
    let specials: [f64; 12] = [
        0.0,
        -0.0,
        0.5,
        -0.5,
        1.5,
        -2.5,
        3.0,
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::NAN,
        1e-310,
        4503599627370497.0,
    ];
    let mut out: Vec<u64> = specials
        .iter()
        .map(|&v| {
            if binary64 {
                v.to_bits()
            } else {
                u64::from((v as f32).to_bits())
            }
        })
        .collect();
    let mut state = 0x9e37_79b9_7f4a_7c15u64;
    for _ in 0..n {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        out.push(if binary64 { state } else { state >> 32 });
    }
    out
}

#[test]
#[ignore = "requires a CUDA GPU and NVRTC 12.9+"]
fn math_cr_v1_cuda_exact_functions_match_the_strict_oracles() {
    let mut device = open();
    for b in [false, true] {
        let xs = samples(b, RANDOM);
        let ys: Vec<u64> = xs.iter().rev().copied().collect();
        for f in MathFn::ALL
            .into_iter()
            .filter(|f| invar::math::core_math(*f).is_none())
        {
            let map = load(
                &mut device,
                &cuda_map_source(f, b),
                cuda_map_min_nvrtc(f, b),
            );
            let run = device.run_bits_map(&map, &xs, &ys).unwrap();
            let wrong = xs
                .iter()
                .zip(&ys)
                .zip(&run.output)
                .filter(|((x, y), got)| {
                    let want = if b {
                        eval_f64(
                            f,
                            &[F64Bits::from_bits(**x), F64Bits::from_bits(**y)][..f.arity()],
                        )
                        .to_bits()
                    } else {
                        u64::from(
                            eval_f32(
                                f,
                                &[
                                    F32Bits::from_bits(**x as u32),
                                    F32Bits::from_bits(**y as u32),
                                ][..f.arity()],
                            )
                            .to_bits(),
                        )
                    };
                    **got != want
                })
                .count();
            println!(
                "cuda {} f{} inputs={} mismatches={wrong}",
                f.name(),
                if b { 64 } else { 32 },
                xs.len()
            );
            assert_eq!(wrong, 0, "{} f{}", f.name(), if b { 64 } else { 32 });
        }
    }
}

#[test]
#[ignore = "requires a CUDA GPU and NVRTC 12.9+"]
fn math_cr_v1_cuda_every_function_shares_one_program() {
    let all: Vec<_> = [false, true]
        .into_iter()
        .flat_map(|b| transcendental().map(move |f| (invar::math::core_math(f).unwrap(), b)))
        .collect();
    let mut source = invar_math::cuda_source(&all);
    source.push_str(&format!(
        "extern \"C\" __global__ void {CUDA_KERNEL}(const unsigned long long *x, const unsigned long long *y, unsigned long long *out, unsigned long long n) {{\n  \
         unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;\n  if (i >= n) return;\n  \
         double a = __longlong_as_double((long long)x[i]);\n  out[i] = (unsigned long long)__double_as_longlong({}({}(a)));\n}}\n",
        invar_math::cuda_symbol(invar_math::Function::Sin, true),
        invar_math::cuda_symbol(invar_math::Function::Log, true),
    ));
    let mut device = open();
    let map = load(&mut device, &source, invar_math::cuda_min_nvrtc(&all));
    let x = [2.0f64.to_bits()];
    let run = device.run_bits_map(&map, &x, &x).unwrap();
    let want = invar_math::f64_bits(
        invar_math::Function::Sin,
        invar_math::f64_bits(invar_math::Function::Log, x[0], 0),
        0,
    );
    println!(
        "combined program: cubin_bytes={} compile_ms={:.0}",
        map.cubin_bytes, map.compile_ms
    );
    assert_eq!(run.output[0], want);
}

#[test]
#[ignore = "requires a CUDA GPU and NVRTC 12.9+"]
fn math_cr_v1_cuda_every_function_compiles_alone() {
    let mut device = open();
    let mut failures = Vec::new();
    for b in [false, true] {
        for f in MathFn::ALL {
            match load_cuda_map(&mut device, f, b) {
                Ok(map) => println!(
                    "cuda {} f{} compiles: cubin_bytes={} compile_ms={:.0}",
                    f.name(),
                    if b { 64 } else { 32 },
                    map.cubin_bytes,
                    map.compile_ms
                ),
                Err(e) => {
                    let text = format!("{e:?}");
                    let first = text.split("\\n").take(3).collect::<Vec<_>>().join(" | ");
                    println!(
                        "cuda {} f{} FAILS: {first}",
                        f.name(),
                        if b { 64 } else { 32 }
                    );
                    failures.push(format!("{}{}", f.name(), if b { 64 } else { 32 }));
                }
            }
        }
    }
    assert!(failures.is_empty(), "{failures:?}");
}
