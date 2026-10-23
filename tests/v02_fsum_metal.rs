//! v0.2 reductions on a live Metal device: kernel partials finished on the
//! host equal the CPU engines bit for bit at every size class, every leaf
//! chunk exponent and on adversarial inputs; the CLI records each placement.
#![cfg(all(feature = "metal", target_os = "macos"))]

#[path = "support/fsum_cases.rs"]
mod cases;
#[path = "support/fsum_source.rs"]
mod source;

use cases::{Rng, adversarial_vectors};
use invar::fsum;
use invar_device::fsum::Op;
use invar_metal::MetalDevice;
use std::process::Command;

const OPS: [Op; 4] = [Op::SumTree, Op::DotTree, Op::SumExact, Op::DotExact];

fn device_schema() -> serde_json::Value {
    let schema = include_str!("../schema/invar-explain-device-v5.schema.json");
    serde_json::from_str(schema).unwrap()
}

fn expected(op: Op, a: &[u32], b: &[u32]) -> u32 {
    match op {
        Op::SumTree => fsum::sum_tree(a),
        Op::DotTree => fsum::dot_tree(a, b),
        Op::SumExact => fsum::sum_exact(a),
        Op::DotExact => fsum::dot_exact(a, b),
    }
}

/// The device result, or `None` when the subnormal guard discarded it.
fn device_bits(device: &mut MetalDevice, op: Op, a: &[u32], b: &[u32], shift: u32) -> Option<u32> {
    let second = op.is_dot().then_some(b);
    let result = match device.run_fsum(op, a, second, shift) {
        Err(invar_metal::Error::SubnormalFallback) => return None,
        other => other.unwrap(),
    };
    Some(if op.is_exact() {
        fsum::finish_rows(&result.rows, a.len() as u64, op.is_dot())
    } else {
        let nodes = &result.nodes;
        fsum::canonical(fsum::fold_aligned(nodes.len(), |g| {
            f32::from_bits(nodes[g])
        }))
    })
}

#[test]
fn v02_fsum_metal_adversarial_vectors_match_cpu_bits() {
    let mut device = MetalDevice::open_runtime_source().unwrap();
    let mut published = 0;
    for (label, a, b) in adversarial_vectors()
        .into_iter()
        .filter(|(_, a, _)| !a.is_empty())
    {
        for op in OPS {
            let got = device_bits(&mut device, op, &a, &b, 2);
            if let Some(got) = got {
                assert_eq!(got, expected(op, &a, &b), "{label} {op:?}");
                published += 1;
            } else {
                assert!(!op.is_exact(), "the exact kernel is integer-only: {label}");
            }
        }
    }
    assert!(published > 100, "{published}");
}

#[test]
fn v02_fsum_metal_group_boundaries_and_leaf_shifts_are_bit_identical() {
    let mut device = MetalDevice::open_runtime_source().unwrap();
    let mut rng = Rng(0x3e7a1);
    for n in [
        1usize,
        255,
        2047,
        2048,
        2049,
        8191,
        8193,
        65_543,
        1 << 20,
        3_000_017,
    ] {
        let a = (0..n).map(|_| rng.word()).collect::<Vec<_>>();
        let b = (0..n).map(|_| rng.finite()).collect::<Vec<_>>();
        let finite = (0..n).map(|_| rng.finite()).collect::<Vec<_>>();
        for op in OPS {
            for (input, normal) in [(&a, false), (&finite, true)] {
                let want = expected(op, input, &b);
                for shift in 0..=3 {
                    match device_bits(&mut device, op, input, &b, shift) {
                        Some(got) => assert_eq!(got, want, "n={n} {op:?} shift={shift}"),
                        None => assert!(!normal && !op.is_exact(), "n={n} {op:?} guard"),
                    }
                }
            }
        }
    }
}

#[test]
fn v02_fsum_metal_subnormal_guard_and_contraction_cases() {
    let mut device = MetalDevice::open_runtime_source().unwrap();
    let subnormal = (1..=4096u32).map(|i| i % 0x7f_ffff).collect::<Vec<_>>();
    // A product that underflows from normal operands, and a sum of normals
    // whose exact result is subnormal.
    let tiny = vec![0x1f80_0000u32; 64];
    let cancel = vec![0x0080_0001u32, 0x8080_0000];
    for (a, b) in [(&subnormal, &subnormal), (&tiny, &tiny), (&cancel, &cancel)] {
        for op in OPS {
            let got = device_bits(&mut device, op, a, b, 2);
            if op.is_exact() {
                assert_eq!(got, Some(expected(op, a, b)), "{op:?}");
            } else if op == Op::DotTree || a == &subnormal || a == &cancel {
                assert_eq!(got, None, "guard must fire: {op:?} {:x}", a[0]);
            } else {
                assert_eq!(got, Some(expected(op, a, b)), "{op:?} {:x}", a[0]);
            }
        }
    }
    // `1 + 2^-23` squared needs a rounded product: a fused multiply-add
    // would keep the `2^-46` term and change the sum.
    let near = vec![0x3f80_0001u32; 4096];
    let minus = vec![0xbf80_0002u32; 4096];
    let mixed = near
        .iter()
        .zip(&minus)
        .flat_map(|(&x, &y)| [x, y])
        .collect::<Vec<_>>();
    for (a, b) in [(&near, &near), (&mixed, &mixed)] {
        for op in OPS {
            assert_eq!(
                device_bits(&mut device, op, a, b, 2),
                Some(expected(op, a, b)),
                "{op:?}"
            );
        }
    }
}

#[test]
fn v02_fsum_metal_explain_records_each_reduction_placement() {
    let path = std::env::temp_dir().join(format!("v02-fsum-metal-{}.inv", std::process::id()));
    let library = path.with_extension("metallib");
    std::fs::write(&path, include_str!("../examples/fsum.inv")).unwrap();
    let invar = env!("CARGO_BIN_EXE_invar");
    let built = Command::new(invar)
        .args([
            "build",
            path.to_str().unwrap(),
            "--target",
            "metal",
            "--output",
        ])
        .arg(&library)
        .output()
        .unwrap();
    assert!(
        built.status.success(),
        "{}",
        String::from_utf8_lossy(&built.stdout)
    );
    // The compiled-ahead library carries the fsum kernels: no runtime MSL.
    for (engine, backends) in [
        ("metal", ["metal"].as_slice()),
        ("auto", &["metal", "cpu-parallel", "cpu-serial"]),
    ] {
        let output = Command::new(invar)
            .env(invar_metal::SOURCE_TRIPWIRE, "1")
            .args([
                "explain",
                path.to_str().unwrap(),
                "--engine",
                engine,
                "--artifact",
                library.to_str().unwrap(),
                "--format",
                "json",
            ])
            .output()
            .unwrap();
        let json: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
        let device = &json["device_execution"];
        assert_eq!(device["result"], 0, "{json}");
        assert!(jsonschema::is_valid(&device_schema(), device), "{device}");
        let kinds = device["choices"]
            .as_array()
            .unwrap()
            .iter()
            .filter(|choice| {
                choice["kind"]
                    .as_str()
                    .is_some_and(|kind| kind.starts_with("fsum."))
            })
            .collect::<Vec<_>>();
        assert_eq!(kinds.len(), 2, "{json}");
        for choice in kinds {
            assert!(
                backends.contains(&choice["backend"].as_str().unwrap()),
                "{choice}"
            );
        }
        let hint = json["regions"][0]["obligations"].to_string();
        assert!(
            hint.contains("FLOAT_FOLD_HINT") && hint.contains("sum_exact"),
            "{json}"
        );
    }
    std::fs::remove_file(path).unwrap();
    std::fs::remove_file(library).unwrap();
}

/// Resident GPU medians next to the handwritten unordered Metal sum.
/// `cargo test --release --features metal --test v02_fsum_metal -- --ignored`
#[test]
#[ignore = "benchmark; run alone under INVAR_GPU_EXCLUSIVE=1"]
fn v02_fsum_metal_bench() {
    let mut device = MetalDevice::open_runtime_source().unwrap();
    let mut rng = Rng(0xbe9c);
    for log2 in [20u32, 24, 26] {
        let a = (0..1usize << log2)
            .map(|_| rng.finite())
            .collect::<Vec<_>>();
        for (op, shifts) in [(Op::SumTree, 0..=3), (Op::SumExact, 0..=0)] {
            for shift in shifts {
                // Five rounds of 30 interleaved samples; report the median round.
                let mut rounds = (0..5)
                    .map(|_| device.benchmark_fsum(op, &a, shift).unwrap())
                    .map(|bench| (bench.median_ms / bench.baseline_median_ms, bench))
                    .collect::<Vec<_>>();
                rounds.sort_by(|x, y| x.0.total_cmp(&y.0));
                let spread = (rounds[0].0, rounds[4].0);
                let (ratio, bench) = &rounds[2];
                println!(
                    "metal {op:?} n=2^{log2} shift={shift} median_ms={:.4} baseline_ms={:.4} ratio={ratio:.3} rounds=[{:.3}, {:.3}]",
                    bench.median_ms, bench.baseline_median_ms, spread.0, spread.1
                );
            }
        }
    }
}

/// End-to-end Metal (upload, kernel, host finish) against the faster of one
/// and all CPU workers, per size; prints the smallest size from which Metal wins
/// at every larger measured size (the `metal_min_terms` crossover).
/// `cargo test --release --features metal --test v02_fsum_metal v02_fsum_metal_crossover -- --ignored --nocapture`
#[test]
#[ignore = "calibration; run alone under INVAR_GPU_EXCLUSIVE=1"]
fn v02_fsum_metal_crossover() {
    use invar::{
        front::Span, mir_eval::ReduceCall, numeric_contract::F32Bits, schedule::reductions,
    };
    let mut device = MetalDevice::open_runtime_source().unwrap();
    let workers = std::thread::available_parallelism().map_or(1, |n| n.get());
    let mut rng = Rng(0xc055);
    for (op, kind) in [
        (Op::SumTree, fsum::Kind::SumTree),
        (Op::DotTree, fsum::Kind::DotTree),
        (Op::SumExact, fsum::Kind::SumExact),
        (Op::DotExact, fsum::Kind::DotExact),
    ] {
        let mut wins = Vec::new();
        for log2 in 10..=26u32 {
            let a = (0..1usize << log2)
                .map(|_| F32Bits::from_bits(rng.finite()))
                .collect::<Vec<_>>();
            let call = ReduceCall {
                function: "main",
                span: Span { start: 0, end: 0 },
                kind,
                a: &a,
                b: op.is_dot().then_some(&a[..]),
            };
            let metal = median_ms(|| {
                let words = reductions::words(call.a);
                let second = op.is_dot().then_some(&words[..]);
                let result = device
                    .run_fsum(op, &words, second, reductions::LEAF_SHIFT)
                    .unwrap();
                reductions::finish(&call, &result.nodes, &result.rows).to_bits()
            });
            // The faster of one and all CPU workers, so the crossover does not
            // depend on the CPU cost model's own coefficients.
            let one = median_ms(|| reductions::cpu_on(&call, 1).unwrap().to_bits());
            let all = median_ms(|| reductions::cpu_on(&call, workers).unwrap().to_bits());
            let cpu = one.min(all);
            println!(
                "crossover {op:?} n=2^{log2} metal_ms={metal:.4} cpu1_ms={one:.4} cpu{workers}_ms={all:.4}"
            );
            wins.push((log2, metal < cpu));
        }
        let from = wins
            .iter()
            .rev()
            .take_while(|(_, win)| *win)
            .last()
            .map(|(log2, _)| *log2);
        println!("crossover {op:?} metal_min_terms=2^{from:?}");
    }
}

fn median_ms(mut run: impl FnMut() -> u32) -> f64 {
    let mut times = (0..20)
        .map(|_| {
            let start = std::time::Instant::now();
            std::hint::black_box(run());
            start.elapsed().as_secs_f64() * 1000.0
        })
        .skip(5)
        .collect::<Vec<_>>();
    times.sort_by(f64::total_cmp);
    times[7]
}
