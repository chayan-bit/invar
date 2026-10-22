//! invar-math-cr-v1 performance next to the platform libraries (ruling C13:
//! a bench harness, not `invar run`). CPU: ns per call against the host libm
//! through Rust's `f32`/`f64` methods. CUDA: kernel time over 2^24 inputs
//! against CUDA's own math (`__nv_*` from libdevice) under the same options.
//! Metal (f32): kernel time over 2^22 inputs against `metal::precise`.
//! Each run records the 1-minute load average; numbers count only below 8.
//! Both harnesses also count how often the baseline differs from the
//! correctly rounded result.
//!
//! `cargo test --release [--features cuda] --test issue_v02_math_bench --
//! --ignored --nocapture --test-threads 1`
use invar::{
    math::{MathFn, eval_f32, eval_f64},
    numeric_contract::{F32Bits, F64Bits},
};
use std::{hint::black_box, time::Instant};

const N: usize = 1 << 20;
const REPEATS: usize = 5;

fn load_average() -> String {
    if let Ok(text) = std::fs::read_to_string("/proc/loadavg") {
        return text.split_whitespace().next().unwrap_or("?").to_owned();
    }
    // macOS: `{ 1.23 1.10 0.98 }`
    std::process::Command::new("sysctl")
        .args(["-n", "vm.loadavg"])
        .output()
        .ok()
        .and_then(|o| {
            String::from_utf8_lossy(&o.stdout)
                .split_whitespace()
                .nth(1)
                .map(str::to_owned)
        })
        .unwrap_or_else(|| "unavailable".into())
}

/// Moderate-exponent inputs where every function is defined and finite.
fn inputs(n: usize, seed: u64) -> Vec<f64> {
    let mut state = seed | 1;
    (0..n)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state >> 11) as f64 / (1u64 << 53) as f64 * 0.98 + 0.01
        })
        .collect()
}

fn libm64(f: MathFn, x: f64, y: f64) -> f64 {
    match f {
        MathFn::Cbrt => x.cbrt(),
        MathFn::Hypot => x.hypot(y),
        MathFn::Sin => x.sin(),
        MathFn::Cos => x.cos(),
        MathFn::Tan => x.tan(),
        MathFn::Asin => x.asin(),
        MathFn::Acos => x.acos(),
        MathFn::Atan => x.atan(),
        MathFn::Atan2 => x.atan2(y),
        MathFn::Sinh => x.sinh(),
        MathFn::Cosh => x.cosh(),
        MathFn::Tanh => x.tanh(),
        MathFn::Exp => x.exp(),
        MathFn::Exp2 => x.exp2(),
        MathFn::Expm1 => x.exp_m1(),
        MathFn::Log => x.ln(),
        MathFn::Log2 => x.log2(),
        MathFn::Log10 => x.log10(),
        MathFn::Log1p => x.ln_1p(),
        MathFn::Pow => x.powf(y),
        _ => unreachable!(),
    }
}

fn libm32(f: MathFn, x: f32, y: f32) -> f32 {
    match f {
        MathFn::Cbrt => x.cbrt(),
        MathFn::Hypot => x.hypot(y),
        MathFn::Sin => x.sin(),
        MathFn::Cos => x.cos(),
        MathFn::Tan => x.tan(),
        MathFn::Asin => x.asin(),
        MathFn::Acos => x.acos(),
        MathFn::Atan => x.atan(),
        MathFn::Atan2 => x.atan2(y),
        MathFn::Sinh => x.sinh(),
        MathFn::Cosh => x.cosh(),
        MathFn::Tanh => x.tanh(),
        MathFn::Exp => x.exp(),
        MathFn::Exp2 => x.exp2(),
        MathFn::Expm1 => x.exp_m1(),
        MathFn::Log => x.ln(),
        MathFn::Log2 => x.log2(),
        MathFn::Log10 => x.log10(),
        MathFn::Log1p => x.ln_1p(),
        MathFn::Pow => x.powf(y),
        _ => unreachable!(),
    }
}

/// Median ns per call of `call` over `xs`/`ys`.
fn time(xs: &[f64], ys: &[f64], mut call: impl FnMut(f64, f64) -> u64) -> f64 {
    let mut samples: Vec<f64> = (0..REPEATS)
        .map(|_| {
            let start = Instant::now();
            let mut sink = 0u64;
            for (x, y) in xs.iter().zip(ys) {
                sink ^= call(black_box(*x), black_box(*y));
            }
            black_box(sink);
            start.elapsed().as_nanos() as f64 / xs.len() as f64
        })
        .collect();
    samples.sort_by(f64::total_cmp);
    samples[REPEATS / 2]
}

fn transcendental() -> impl Iterator<Item = MathFn> {
    MathFn::ALL
        .into_iter()
        .filter(|f| invar::math::core_math(*f).is_some())
}

#[test]
#[ignore = "benchmark: run in release on an idle host"]
fn v02_math_cpu_against_libm() {
    let (xs, ys) = (inputs(N, 7), inputs(N, 11));
    println!(
        "cpu bench: n={N} repeats={REPEATS} load1={}",
        load_average()
    );
    for b in [false, true] {
        for f in transcendental() {
            let ours = time(&xs, &ys, |x, y| {
                if b {
                    eval_f64(
                        f,
                        &[
                            F64Bits::from_bits(x.to_bits()),
                            F64Bits::from_bits(y.to_bits()),
                        ][..f.arity()],
                    )
                    .to_bits()
                } else {
                    let args = [
                        F32Bits::from_bits((x as f32).to_bits()),
                        F32Bits::from_bits((y as f32).to_bits()),
                    ];
                    u64::from(eval_f32(f, &args[..f.arity()]).to_bits())
                }
            });
            let libm = time(&xs, &ys, |x, y| {
                if b {
                    libm64(f, x, y).to_bits()
                } else {
                    u64::from(libm32(f, x as f32, y as f32).to_bits())
                }
            });
            let wrong = xs
                .iter()
                .zip(&ys)
                .filter(|&(&x, &y)| {
                    if b {
                        let cr = eval_f64(
                            f,
                            &[
                                F64Bits::from_bits(x.to_bits()),
                                F64Bits::from_bits(y.to_bits()),
                            ][..f.arity()],
                        );
                        cr.to_bits() != libm64(f, x, y).to_bits()
                    } else {
                        let (x, y) = (x as f32, y as f32);
                        let cr = eval_f32(
                            f,
                            &[
                                F32Bits::from_bits(x.to_bits()),
                                F32Bits::from_bits(y.to_bits()),
                            ][..f.arity()],
                        );
                        cr.to_bits() != libm32(f, x, y).to_bits()
                    }
                })
                .count();
            println!(
                "cpu {} f{} invar_ns={ours:.2} libm_ns={libm:.2} ratio={:.2} libm_not_correctly_rounded={wrong}",
                f.name(),
                if b { 64 } else { 32 },
                ours / libm
            );
        }
    }
}

#[cfg(feature = "cuda")]
#[test]
#[ignore = "benchmark: requires a CUDA GPU and an idle host"]
fn v02_math_cuda_against_cuda_math() {
    use invar::math::{CUDA_KERNEL, cuda_map_min_nvrtc, cuda_map_source, nvrtc_admits};
    use invar_cuda::cuda::CudaDevice;
    const DEVICE_N: usize = 1 << 24;
    let mut device = CudaDevice::open().expect("CUDA_UNAVAILABLE");
    let (xs, ys) = (inputs(DEVICE_N, 7), inputs(DEVICE_N, 11));
    println!(
        "cuda bench: device={} n={DEVICE_N} repeats={REPEATS} load1={}",
        device.name(),
        load_average()
    );
    for b in [false, true] {
        let encode = |v: &f64| {
            if b {
                v.to_bits()
            } else {
                u64::from((*v as f32).to_bits())
            }
        };
        let (x, y): (Vec<u64>, Vec<u64>) = (
            xs.iter().map(encode).collect(),
            ys.iter().map(encode).collect(),
        );
        for f in transcendental() {
            nvrtc_admits(cuda_map_min_nvrtc(f, b)).unwrap_or_else(|e| panic!("{e:?}"));
            let ours = cuda_map_source(f, b);
            let native = baseline_source(&ours, f, b);
            let mut timings = Vec::new();
            let mut outputs = Vec::new();
            for source in [&ours, &native] {
                let map = device
                    .load_bits_map(source, &invar_math::CUDA_OPTIONS, CUDA_KERNEL)
                    .unwrap();
                device.run_bits_map(&map, &x, &y).unwrap();
                let mut ms: Vec<f32> = (0..REPEATS)
                    .map(|_| device.run_bits_map(&map, &x, &y).unwrap().kernel_ms)
                    .collect();
                ms.sort_by(f32::total_cmp);
                timings.push(ms[REPEATS / 2]);
                outputs.push(device.run_bits_map(&map, &x, &y).unwrap().output);
            }
            let wrong = outputs[0]
                .iter()
                .zip(&outputs[1])
                .filter(|(a, b)| a != b)
                .count();
            println!(
                "cuda {} f{} invar_ms={:.3} nv_ms={:.3} ratio={:.2} nv_not_correctly_rounded={wrong}",
                f.name(),
                if b { 64 } else { 32 },
                timings[0],
                timings[1],
                timings[0] / timings[1]
            );
        }
    }
}

/// The same kernel calling CUDA's math library (libdevice `__nv_*`).
#[cfg(feature = "cuda")]
fn baseline_source(ours: &str, f: MathFn, binary64: bool) -> String {
    let kernel = &ours[ours.find("extern \"C\" __global__").unwrap()..];
    let symbol = invar_math::cuda_symbol(invar::math::core_math(f).unwrap(), binary64);
    let name = format!("{}{}", f.name(), if binary64 { "" } else { "f" });
    let call = kernel.replace(&symbol, &name);
    assert_ne!(call, kernel);
    call
}

/// `metal::precise` spelling of `f`, where Metal has one.
#[cfg(all(feature = "metal", target_os = "macos"))]
fn metal_precise(f: MathFn) -> Option<&'static str> {
    match f {
        MathFn::Sin => Some("sin(a)"),
        MathFn::Cos => Some("cos(a)"),
        MathFn::Tan => Some("tan(a)"),
        MathFn::Asin => Some("asin(a)"),
        MathFn::Acos => Some("acos(a)"),
        MathFn::Atan => Some("atan(a)"),
        MathFn::Atan2 => Some("atan2(a, b)"),
        MathFn::Sinh => Some("sinh(a)"),
        MathFn::Cosh => Some("cosh(a)"),
        MathFn::Tanh => Some("tanh(a)"),
        MathFn::Exp => Some("exp(a)"),
        MathFn::Exp2 => Some("exp2(a)"),
        MathFn::Log => Some("log(a)"),
        MathFn::Log2 => Some("log2(a)"),
        MathFn::Log10 => Some("log10(a)"),
        MathFn::Pow => Some("pow(a, b)"),
        _ => None,
    }
}

#[cfg(all(feature = "metal", target_os = "macos"))]
#[test]
#[ignore = "benchmark: requires a Metal GPU and an idle host"]
fn v02_math_metal_against_metal_precise() {
    use invar::math::{CUDA_KERNEL, msl_map_source};
    use invar_metal::{BitsMap, MetalDevice};
    const DEVICE_N: usize = 1 << 22;
    const CHUNK: usize = 1 << 18;
    let mut device = MetalDevice::open_runtime_source().expect("METAL_UNAVAILABLE");
    let (xs, ys) = (inputs(DEVICE_N, 7), inputs(DEVICE_N, 11));
    let x: Vec<u64> = xs
        .iter()
        .map(|v| u64::from((*v as f32).to_bits()))
        .collect();
    let y: Vec<u64> = ys
        .iter()
        .map(|v| u64::from((*v as f32).to_bits()))
        .collect();
    println!(
        "metal bench: device={} n={DEVICE_N} repeats={REPEATS} load1={}",
        device.name(),
        load_average()
    );
    let run = |device: &mut MetalDevice, map: &BitsMap| -> (f64, Vec<u64>) {
        let mut samples = Vec::new();
        let mut output = Vec::new();
        for repeat in 0..=REPEATS {
            let (mut ms, mut out) = (0.0, Vec::with_capacity(DEVICE_N));
            for (cx, cy) in x.chunks(CHUNK).zip(y.chunks(CHUNK)) {
                let result = device.run_bits_map(map, cx, cy).unwrap();
                ms += result.kernel_ms;
                out.extend(result.output);
            }
            if repeat > 0 {
                samples.push(ms);
            }
            output = out;
        }
        samples.sort_by(f64::total_cmp);
        (samples[REPEATS / 2], output)
    };
    for f in transcendental() {
        let ours = device
            .load_bits_map(&msl_map_source(f), CUDA_KERNEL)
            .unwrap();
        let (ours_ms, ours_out) = run(&mut device, &ours);
        let Some(call) = metal_precise(f) else {
            println!(
                "metal {} f32 invar_ms={ours_ms:.3} precise_ms=n/a (no metal::precise function)",
                f.name()
            );
            continue;
        };
        let source = format!(
            "#include <metal_stdlib>\nusing namespace metal;\n\
             kernel void {CUDA_KERNEL}(device const ulong *x [[buffer(0)]], device const ulong *y [[buffer(1)]], \
             device ulong *out [[buffer(2)]], constant uint &n [[buffer(3)]], uint i [[thread_position_in_grid]]) {{\n  \
             if (i >= n) return;\n  float a = as_type<float>((uint)x[i]), b = as_type<float>((uint)y[i]);\n  \
             float r = precise::{call};\n  out[i] = r != r ? 0x7fc00000ul : (ulong)as_type<uint>(r);\n}}\n"
        );
        let baseline = device.load_bits_map(&source, CUDA_KERNEL).unwrap();
        let (precise_ms, precise_out) = run(&mut device, &baseline);
        // Distance in binary32 encodings, ordered so it counts ulps across zero.
        let key = |bits: u64| {
            let b = bits as u32 as i64;
            if b & 0x8000_0000 != 0 {
                0x8000_0000 - b
            } else {
                b
            }
        };
        let ulps: Vec<i64> = ours_out
            .iter()
            .zip(&precise_out)
            .map(|(a, b)| (key(*a) - key(*b)).abs())
            .collect();
        let wrong = ulps.iter().filter(|u| **u != 0).count();
        println!(
            "metal {} f32 invar_ms={ours_ms:.3} precise_ms={precise_ms:.3} ratio={:.2} precise_not_correctly_rounded={wrong} precise_max_ulp={}",
            f.name(),
            ours_ms / precise_ms,
            ulps.iter().max().unwrap()
        );
    }
}
