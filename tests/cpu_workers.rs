use invar::{check::check, eval::Value, front::parse, mir, schedule};

#[test]
fn cpu_worker_selection_env() {
    if let Ok(expected) = std::env::var("INVAR_EXPECT_CPU_WORKERS") {
        let actual = schedule::auto_workers();
        if expected == "error" {
            assert_eq!(actual, Err(schedule::Error::InvalidWorkers));
        } else {
            assert_eq!(actual, Ok(expected.parse().unwrap()));
        }
        return;
    }
    let available = std::thread::available_parallelism().map_or(1, |count| count.get());
    let selected = schedule::auto_workers().unwrap();
    assert!((1..=available.min(schedule::MAX_WORKERS)).contains(&selected));
    println!(
        "CPU_WORKERS available={available} selected={selected} bound={}",
        schedule::MAX_WORKERS
    );
    for (value, expected) in [
        ("1", "1".into()),
        ("10", available.min(10).to_string()),
        ("64", available.min(64).to_string()),
        ("0", "error".into()),
        ("65", "error".into()),
        ("invalid", "error".into()),
    ] {
        let output = std::process::Command::new(std::env::current_exe().unwrap())
            .args(["--exact", "cpu_worker_selection_env"])
            .env("INVAR_CPU_WORKERS", value)
            .env("INVAR_EXPECT_CPU_WORKERS", expected)
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stdout)
        );
    }
}

/// Timed boundary: source scheduler + MIR, excluding parsing/lowering. Every
/// sample must publish the expected CPU region and return its checked value.
#[test]
fn f32_source_runs_on_every_admitted_worker_count() {
    // The f32 source path is reached when no integer family admits the loop.
    let source = "fn scale(a: read Array<f32>) -> Array<f32> { var out = filled_f32(len(a),0.0); for i in 0u32..len(a) { out[i] = a[i] * 2.0 + 1.0; } return out; } fn main() -> i32 { let a = filled_f32(4096u32,1.5); let out = scale(a); if out[0u32] == 4.0 and out[4095u32] == 4.0 { return 7; } return 0; }";
    let checked = check(parse(source.as_bytes()).unwrap(), true).unwrap();
    let program = mir::lower(&checked).unwrap();
    for workers in [1, 4, 5, 16, schedule::MAX_WORKERS] {
        let run = schedule::run_source(source.as_bytes(), &program, 100_000_000, workers)
            .unwrap_or_else(|error| panic!("{workers} workers: {error:?}"));
        assert_eq!(
            run.observation.result,
            Ok(Value::I32(7)),
            "{workers} workers"
        );
        assert!(
            !run.executed.is_empty(),
            "{workers} workers: no f32 region ran"
        );
        assert_eq!(run.fallback_reason, None, "{workers} workers");
        // Every requested worker received a chunk; none was capped below the request.
        assert_eq!(run.workers_used, workers);
    }
}

#[test]
#[ignore = "bounded CPU worker measurement; run through the host heavy wrapper"]
fn cpu_worker_measurement() {
    let counts = std::env::var("INVAR_WORKER_BENCH_COUNTS").unwrap_or_else(|_| "4".into());
    for (name, source, kind) in [
        (
            "map",
            "fn main() -> i32 { let xs: Array<u32> = filled_u32(1000000u32,7u32); var ys: Array<u32> = filled_u32(len(xs),0u32); for i in 0u32 .. len(xs) { ys[i] = wrapping_add(xs[i],3u32); } if ys[999999u32] == 10u32 { return 0; } return 1; }",
            schedule::PlanKind::Map,
        ),
        (
            "reduction",
            "fn main() -> i32 { let xs: Array<u32> = filled_u32(1000000u32,7u32); var acc: u32 = 0u32; for i in 0u32 .. len(xs) { acc = wrapping_add(acc,xs[i]); } if acc == 7000000u32 { return 0; } return 1; }",
            schedule::PlanKind::Reduction,
        ),
    ] {
        let checked = check(parse(source.as_bytes()).unwrap(), true).unwrap();
        let program = mir::lower(&checked).unwrap();
        for workers in counts.split(',').map(|text| text.parse::<usize>().unwrap()) {
            let mut samples = Vec::new();
            for trial in 0..10 {
                let start = std::time::Instant::now();
                let run = schedule::run_source(source.as_bytes(), &program, 100_000_000, workers)
                    .unwrap();
                let ms = start.elapsed().as_secs_f64() * 1000.0;
                assert_eq!(run.observation.result, Ok(Value::I32(0)));
                assert_eq!(run.plan, kind);
                assert_eq!(run.joined, if workers > 1 { workers } else { 0 });
                std::hint::black_box(&run);
                if trial > 0 {
                    samples.push(ms);
                }
            }
            samples.sort_by(f64::total_cmp);
            println!(
                "CPU_BENCH {name} workers={workers} elements=1000000 median_ms={:.6} min_ms={:.6} max_ms={:.6} samples_ms={samples:?}",
                samples[4], samples[0], samples[8]
            );
        }
    }
}
