//! v0.2 reductions on a live Metal device: kernel partials finished on the
//! host equal the CPU engines bit for bit at every size class, every leaf
//! chunk exponent and on adversarial inputs; the CLI records each placement.
use invar_device::fsum::Op;
use invar_metal::MetalDevice;
use std::process::Command;

#[path = "../support/fsum_cases.rs"]
mod cases;
#[path = "../support/fsum_source.rs"]
mod source;

use cases::Rng;
use invar::fsum;

type Device = MetalDevice;

fn open() -> Device {
    MetalDevice::open_runtime_source().unwrap()
}

/// The device result, or `None` when the subnormal guard discarded it.
fn device_bits(device: &mut Device, op: Op, a: &[u32], b: &[u32], shift: u32) -> Option<u32> {
    let second = op.is_dot().then_some(b);
    let result = match device.run_fsum(op, a, second, shift) {
        Err(invar_metal::Error::SubnormalFallback) => return None,
        other => other.unwrap(),
    };
    Some(body::finish(op, a.len(), &result.rows, &result.nodes))
}

/// The guard discards the dispatches whose operands or exact result are subnormal:
/// the float kernels on the first three cases (a product that underflows from
/// normal operands, and a sum of normals whose exact result is subnormal).
fn guard_fires(op: Op, label: &str) -> bool {
    matches!(label, "subnormal" | "tiny" | "cancel")
        && !op.is_exact()
        && (op == Op::DotTree || label != "tiny")
}

#[path = "../support/gpu_fsum.rs"]
mod body;

fn device_schema() -> serde_json::Value {
    let schema = include_str!("../../schema/invar-explain-device-v1.schema.json");
    serde_json::from_str(schema).unwrap()
}

#[test]
fn v02_fsum_metal_explain_records_each_reduction_placement() {
    let path = std::env::temp_dir().join(format!("v02-fsum-metal-{}.inv", std::process::id()));
    let library = path.with_extension("metallib");
    std::fs::write(&path, include_str!("../../examples/fsum.inv")).unwrap();
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
/// `cargo test --release --features metal --test gpu_metal v02_fsum:: -- --ignored`
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
/// `cargo test --release --features metal --test gpu_metal v02_fsum::v02_fsum_metal_crossover -- --ignored --nocapture`
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
