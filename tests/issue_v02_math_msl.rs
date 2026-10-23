//! invar-math-cr-v1 binary32 for Metal, bit for bit against independent
//! oracles.
//!
//! `invar::math::msl_map_source` turns upstream's binary32 C into MSL whose
//! binary64 intermediates run on the prelude's integer-implemented binary64
//! unit. The same text compiles as host C++, so the unit and the transformed
//! sources are checked on the CPU (here and in CI), then on a Metal GPU:
//! - the binary64 unit against the host FPU on edge-biased random encodings;
//! - every function against the CPU library, and (ignored, heavy) against an
//!   upstream CORE-MATH build on its worst cases, specials, 10^6 random
//!   inputs and every 256th encoding of each univariate function;
//! - on Metal, the same upstream comparison, the exact functions against the
//!   strict oracles, and a mutated polynomial coefficient that must be caught.
//!
//! `INVAR_COREMATH_UPSTREAM=<core-math clone> cargo test --release --features
//! metal --test issue_v02_math_msl -- --ignored --nocapture --test-threads 1`.

#[path = "support/coremath.rs"]
mod coremath;

use coremath::{build, records, tempdir, transcendental};
use invar::{
    math::{MathFn, eval_f32, msl_map_source},
    numeric_contract::F32Bits,
};
use std::{
    io::{Read, Write},
    path::{Path, PathBuf},
    process::{Command, Stdio},
};

const RANDOM: u64 = 1_000_000;
const STRIDE: u64 = 256;
const CHUNK: usize = 1 << 20;
const PRELUDE_DIR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/crates/invar-math/src");
/// The leading coefficient of sinf's `a` table, 0x1.921fb54442d17p-3, with
/// bit 40 flipped (a relative change near 2^-12).
const MUTATION: (&str, &str) = (
    "D::raw(0x3fc921fb54442d17ul)",
    "D::raw(0x3fc920fb54442d17ul)",
);

const HOST_DRIVER: &str = r#"#include <cstdio>
extern "C" void invar_math_map_host(const unsigned long *, const unsigned long *, unsigned long *, unsigned long);
int main() {
  static unsigned long in[2 << 16], x[1 << 16], y[1 << 16], out[1 << 16];
  size_t n;
  while ((n = fread(in, 16, 1 << 16, stdin)) > 0) {
    for (size_t i = 0; i < n; i++) { x[i] = in[2 * i]; y[i] = in[2 * i + 1]; }
    invar_math_map_host(x, y, out, n);
    fwrite(out, 8, n, stdout);
  }
  return 0;
}
"#;

fn cxx(args: &[&str]) {
    let output = Command::new(std::env::var("CXX").unwrap_or_else(|_| "c++".into()))
        .args(args)
        .output()
        .expect("a C++ compiler (c++ or $CXX)");
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
}

/// Build `source` as a host executable that maps stdin pairs to stdout.
fn host_build(directory: &Path, name: &str, source: &str) -> PathBuf {
    let (unit, driver) = (
        directory.join(format!("{name}.cpp")),
        // Per-function name: builds run on parallel threads in one directory.
        directory.join(format!("{name}-driver.cpp")),
    );
    std::fs::write(&unit, source).unwrap();
    std::fs::write(&driver, HOST_DRIVER).unwrap();
    let binary = directory.join(name);
    cxx(&[
        "-std=c++20",
        "-O2",
        "-ffp-contract=off",
        "-w",
        unit.to_str().unwrap(),
        driver.to_str().unwrap(),
        "-o",
        binary.to_str().unwrap(),
    ]);
    binary
}

fn host_run(binary: &Path, x: &[u64], y: &[u64]) -> Vec<u64> {
    let mut child = Command::new(binary)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .unwrap();
    let input: Vec<u8> = x
        .iter()
        .zip(y)
        .flat_map(|(a, b)| a.to_le_bytes().into_iter().chain(b.to_le_bytes()))
        .collect();
    let mut stdin = child.stdin.take().unwrap();
    let writer = std::thread::spawn(move || stdin.write_all(&input).unwrap());
    let mut bytes = Vec::new();
    child
        .stdout
        .take()
        .unwrap()
        .read_to_end(&mut bytes)
        .unwrap();
    writer.join().unwrap();
    assert!(child.wait().unwrap().success());
    bytes
        .chunks_exact(8)
        .map(|w| u64::from_le_bytes(w.try_into().unwrap()))
        .collect()
}

fn oracle(f: MathFn, x: u64, y: u64) -> u64 {
    let args = [F32Bits::from_bits(x as u32), F32Bits::from_bits(y as u32)];
    u64::from(eval_f32(f, &args[..f.arity()]).to_bits())
}

/// Binary32 specials plus `n` pseudo-random encodings, as (x, y) columns.
fn samples(n: usize) -> (Vec<u64>, Vec<u64>) {
    let specials: [f32; 14] = [
        0.0,
        -0.0,
        0.5,
        -0.5,
        1.5,
        -2.5,
        3.0,
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::NAN,
        1e-40,
        -1e-45,
        f32::MAX,
        8388609.0,
    ];
    let mut x: Vec<u64> = specials.iter().map(|v| u64::from(v.to_bits())).collect();
    let mut state = 0x9e37_79b9_7f4a_7c15u64;
    for _ in 0..n {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        x.push(state >> 32);
    }
    let y = x.iter().rev().copied().collect();
    (x, y)
}

/// (inputs, mismatches, first mismatch) of `run` against the upstream
/// driver's records for binary32 `f`.
fn against_upstream(
    driver: &Path,
    f: MathFn,
    random: u64,
    mut run: impl FnMut(&[u64], &[u64]) -> Vec<u64>,
) -> (u64, u64, String) {
    let (mut total, mut wrong, mut first) = (0u64, 0u64, String::new());
    records(driver, f, false, random, STRIDE, CHUNK, |batch| {
        let x: Vec<u64> = batch.iter().map(|r| r[0]).collect();
        let y: Vec<u64> = batch.iter().map(|r| r[1]).collect();
        for (record, got) in batch.iter().zip(run(&x, &y)) {
            total += 1;
            if !coremath::same(false, got, record[2]) {
                wrong += 1;
                if first.is_empty() {
                    first = format!(
                        "x={:#x} y={:#x} got={got:#x} upstream={:#x}",
                        record[0], record[1], record[2]
                    );
                }
            }
        }
    });
    (total, wrong, first)
}

#[test]
fn v02_math_msl_prelude_compiles_warning_free_on_the_host() {
    let directory = tempdir("msl-prelude");
    let unit = directory.join("prelude.cpp");
    std::fs::write(
        &unit,
        "#include \"msl_prelude.h\"\n#include \"msl_prelude.h\"\n\
         ulong k(ulong o) { D a = D::raw(o); return (a * 3 + 1.0f).b; }\n",
    )
    .unwrap();
    for defines in [&[][..], &["-DINVAR_CONSTANT=x", "-DINVAR_THREAD=y"][..]] {
        let mut args = vec![
            "-std=c++20",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-fsyntax-only",
            "-I",
            PRELUDE_DIR,
        ];
        args.extend(defines);
        args.push(unit.to_str().unwrap());
        cxx(&args);
    }
    std::fs::remove_dir_all(directory).unwrap();
}

#[test]
fn v02_math_msl_soft_binary64_matches_the_host_fpu() {
    let directory = tempdir("msl-softdouble");
    let binary = directory.join("softdouble_check");
    cxx(&[
        "-std=c++20",
        "-O2",
        "-ffp-contract=off",
        "-w",
        "-I",
        PRELUDE_DIR,
        concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/crates/invar-math/oracle/softdouble_check.cpp"
        ),
        "-o",
        binary.to_str().unwrap(),
        "-lm",
    ]);
    let output = Command::new(&binary).arg("1000000").output().unwrap();
    let text = String::from_utf8_lossy(&output.stdout);
    println!("{text}");
    std::fs::remove_dir_all(directory).unwrap();
    assert!(
        output.status.success() && text.contains("fails=0"),
        "{text}"
    );
}

#[test]
fn v02_math_msl_host_build_matches_the_cpu_library() {
    let directory = tempdir("msl-host");
    let (x, y) = samples(20_000);
    let failures: Vec<String> = std::thread::scope(|scope| {
        let jobs: Vec<_> = MathFn::ALL
            .into_iter()
            .map(|f| {
                let (directory, x, y) = (&directory, &x, &y);
                scope.spawn(move || {
                    let binary = host_build(directory, f.name(), &msl_map_source(f));
                    let got = host_run(&binary, x, y);
                    let wrong = (0..x.len())
                        .filter(|&i| got[i] != oracle(f, x[i], y[i]))
                        .count();
                    (wrong != 0).then(|| format!("{} mismatches={wrong}", f.name()))
                })
            })
            .collect();
        jobs.into_iter().filter_map(|j| j.join().unwrap()).collect()
    });
    std::fs::remove_dir_all(directory).unwrap();
    assert!(failures.is_empty(), "{failures:?}");
}

#[test]
fn v02_math_msl_host_build_catches_a_mutated_polynomial_coefficient() {
    let directory = tempdir("msl-host-mutant");
    let source = msl_map_source(MathFn::Sin);
    assert_eq!(source.matches(MUTATION.0).count(), 1);
    let binary = host_build(&directory, "sin", &source.replace(MUTATION.0, MUTATION.1));
    let (x, y) = samples(20_000);
    let got = host_run(&binary, &x, &y);
    std::fs::remove_dir_all(directory).unwrap();
    let wrong = (0..x.len())
        .filter(|&i| got[i] != oracle(MathFn::Sin, x[i], y[i]))
        .count();
    println!(
        "msl-host mutant sin f32: inputs={} mismatches={wrong}",
        x.len()
    );
    assert!(wrong > 0);
}

#[test]
#[ignore = "heavy: requires INVAR_COREMATH_UPSTREAM"]
fn v02_math_msl_host_build_matches_upstream_core_math() {
    let directory = tempdir("msl-host-upstream");
    let driver = build(&directory, None);
    let mut failures = Vec::new();
    for f in transcendental() {
        let binary = host_build(&directory, f.name(), &msl_map_source(f));
        let (total, wrong, first) =
            against_upstream(&driver, f, RANDOM, |x, y| host_run(&binary, x, y));
        println!(
            "msl-host {} f32 inputs={total} mismatches={wrong} {first}",
            f.name()
        );
        if wrong != 0 || total < RANDOM {
            failures.push(f.name());
        }
    }
    std::fs::remove_dir_all(directory).unwrap();
    assert!(failures.is_empty(), "{failures:?}");
}

#[cfg(all(feature = "metal", target_os = "macos"))]
mod metal {
    use super::*;
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

    fn run(
        device: &mut MetalDevice,
        map: &BitsMap,
        x: &[u64],
        y: &[u64],
        ms: &mut f64,
    ) -> Vec<u64> {
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
    fn v02_math_metal_matches_upstream_core_math_bit_for_bit() {
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
    fn v02_math_metal_catches_a_mutated_polynomial_coefficient() {
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
    fn v02_math_metal_exact_functions_match_the_strict_oracles() {
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
}
