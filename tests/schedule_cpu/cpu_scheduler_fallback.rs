use invar::{check::check, eval, front::parse, mir, mir_eval, schedule};
use std::process::Command;

const SOURCE: &str = "fn main() -> i32 { let xs: Array<u32> = [1u32,2u32,3u32,4u32,5u32,6u32,7u32]; var first: Array<u32> = filled_u32(len(xs),0u32); for i in 0u32 .. len(xs) { first[i] = wrapping_add(xs[i],3u32); } let stable: Array<u32> = move first; var second: Array<u32> = filled_u32(len(stable),0u32); for j in 0u32 .. len(stable) { second[j] = wrapping_mul(stable[j],7u32); } if second[6u32] == 70u32 { return 0; } return 1; }";

#[test]
fn cpu_scheduler_fallback_map_workers_agree_with_serial() {
    let checked = check(parse(SOURCE.as_bytes()).unwrap(), true).unwrap();
    let program = mir::lower(&checked).unwrap();
    let reference = eval::observe(&checked, 100_000);
    let serial = mir_eval::observe(&program, 100_000);
    assert_eq!(serial.result, reference.result);
    for workers in [1, 2, 4] {
        let run = schedule::run_source(SOURCE.as_bytes(), &program, 100_000, workers).unwrap();
        assert_eq!(run.observation.result, serial.result);
        assert_eq!(run.observation.cleanup_trace, serial.cleanup_trace);
        assert_eq!(run.plan, schedule::PlanKind::FusedMap);
    }
}

#[test]
fn cpu_scheduler_fallback_cli_engine_and_workers() {
    let path = std::env::temp_dir().join(format!("invar-m2-05-{}.inv", std::process::id()));
    std::fs::write(&path, SOURCE).unwrap();
    let reference = Command::new(env!("CARGO_BIN_EXE_invar"))
        .args([
            "run",
            path.to_str().unwrap(),
            "--engine",
            "reference",
            "--format",
            "json",
        ])
        .output()
        .unwrap();
    assert!(reference.status.success());
    for workers in ["1", "2", "4"] {
        let output = Command::new(env!("CARGO_BIN_EXE_invar"))
            .args([
                "run",
                path.to_str().unwrap(),
                "--engine",
                "cpu-parallel",
                "--workers",
                workers,
                "--format",
                "json",
            ])
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        assert_eq!(output.stdout, reference.stdout);
    }
    std::fs::remove_file(path).unwrap();
}

pub fn lower(source: &str) -> (invar::check::CheckedProgram, mir::Program) {
    let checked = check(parse(source.as_bytes()).unwrap(), true).unwrap();
    let program = mir::lower(&checked).unwrap();
    (checked, program)
}

fn literal(values: &[u32]) -> String {
    values
        .iter()
        .map(|x| format!("{x}u32"))
        .collect::<Vec<_>>()
        .join(",")
}

fn map_source(len: usize) -> String {
    let values = (0..len)
        .map(|i| (i as u32).wrapping_mul(5))
        .collect::<Vec<_>>();
    let mut source = format!(
        "fn main() -> i32 {{ let xs: Array<u32> = [{}]; var first: Array<u32> = filled_u32(len(xs),0u32); for i in 0u32 .. len(xs) {{ first[i] = wrapping_add(xs[i],3u32); }} let stable: Array<u32> = move first; var second: Array<u32> = filled_u32(len(stable),0u32); for j in 0u32 .. len(stable) {{ second[j] = wrapping_mul(stable[j],7u32); }}",
        literal(&values)
    );
    for (i, value) in values.iter().enumerate() {
        source.push_str(&format!(
            "if second[{i}u32] != {}u32 {{ return 1; }}",
            value.wrapping_add(3).wrapping_mul(7)
        ));
    }
    source.push_str("return 0; }");
    source
}

fn reduction_source(len: usize, product: bool) -> String {
    let values = (0..len)
        .map(|i| (i as u32).wrapping_mul(3).wrapping_add(1))
        .collect::<Vec<_>>();
    let (identity, op) = if product {
        (1u32, "wrapping_mul")
    } else {
        (0u32, "wrapping_add")
    };
    let expected = values.iter().copied().fold(identity, |a, b| {
        if product {
            a.wrapping_mul(b)
        } else {
            a.wrapping_add(b)
        }
    });
    format!(
        "fn main() -> i32 {{ let xs: Array<u32> = [{}]; var acc: u32 = {identity}u32; for i in 0u32 .. len(xs) {{ acc = {op}(acc,xs[i]); }} if acc == {expected}u32 {{ return 0; }} return 1; }}",
        literal(&values)
    )
}

fn scan_source(len: usize) -> String {
    let aa = (0..len)
        .map(|i| (i as u32).wrapping_mul(7).wrapping_add(2))
        .collect::<Vec<_>>();
    let bb = (0..len + 1)
        .map(|i| (i as u32).wrapping_mul(11).wrapping_add(3))
        .collect::<Vec<_>>();
    let mut source = format!(
        "fn main() -> i32 {{ let aa: Array<u32> = [{}]; let bb: Array<u32> = [{}]; var state: u32 = 17u32; var out: Array<u32> = filled_u32(len(aa),0u32); for i in 0u32 .. len(aa) {{ state = wrapping_add(wrapping_mul(aa[i],state),bb[i]); out[i] = state; }}",
        literal(&aa),
        literal(&bb)
    );
    let mut state = 17u32;
    for i in 0..len {
        state = aa[i].wrapping_mul(state).wrapping_add(bb[i]);
        source.push_str(&format!("if out[{i}u32] != {state}u32 {{ return 1; }}"));
    }
    source.push_str(&format!(
        "if state != {state}u32 {{ return 1; }} return 0; }}"
    ));
    source
}

fn prefix_scan_source(len: usize, exclusive: bool) -> String {
    let aa = (0..len)
        .map(|i| (i as u32).wrapping_mul(7).wrapping_add(2))
        .collect::<Vec<_>>();
    let bb = (0..len)
        .map(|i| (i as u32).wrapping_mul(11).wrapping_add(3))
        .collect::<Vec<_>>();
    let update = "state = wrapping_add(wrapping_mul(aa[i],state),bb[i]);";
    let body = if exclusive {
        format!("out[i] = state; {update}")
    } else {
        format!("{update} out[i] = state;")
    };
    let mut source = format!(
        "fn main() -> i32 {{ let aa: Array<u32> = [{}]; let bb: Array<u32> = [{}]; var state: u32 = 17u32; var out: Array<u32> = filled_u32(len(aa),0u32); for i in 0u32 .. len(aa) {{ {body} }}",
        literal(&aa),
        literal(&bb)
    );
    let mut state = 17u32;
    for i in 0..len {
        let before = state;
        state = aa[i].wrapping_mul(state).wrapping_add(bb[i]);
        let stored = if exclusive { before } else { state };
        source.push_str(&format!("if out[{i}u32] != {stored}u32 {{ return 1; }}"));
    }
    source.push_str(&format!(
        "if state != {state}u32 {{ return 1; }} return 0; }}"
    ));
    source
}

/// Review3 R1: a prefix-scan step publishes one receipt carrying both its
/// output array and final scalar. Inclusive and exclusive scans must publish
/// under every worker count, through the library and the CLI, exactly as the
/// reference engine evaluates them.
#[test]
fn cpu_scheduler_fallback_prefix_scans_publish_as_one_family() {
    for exclusive in [false, true] {
        for len in [3, 65] {
            let source = prefix_scan_source(len, exclusive);
            let (checked, program) = lower(&source);
            let reference = eval::observe(&checked, 1_000_000);
            assert_eq!(
                reference.result,
                Ok(eval::Value::I32(0)),
                "{exclusive} {len}"
            );
            let serial = mir_eval::observe(&program, 1_000_000);
            assert_eq!(serial.result, reference.result);
            for workers in [1, 2, 4] {
                let hooks = schedule::Hooks::default();
                let run = schedule::run_source_with_hooks(
                    source.as_bytes(),
                    &program,
                    1_000_000,
                    workers,
                    &hooks,
                )
                .unwrap();
                let at = format!("exclusive {exclusive}, len {len}, workers {workers}");
                assert_eq!(run.plan, schedule::PlanKind::Scan, "{at}");
                assert_eq!(run.observation.result, reference.result, "{at}");
                assert_eq!(run.observation.cleanup_trace, serial.cleanup_trace, "{at}");
                assert_eq!(run.observation.steps_used, serial.steps_used, "{at}");
                let threads = if len >= 64 && workers > 1 { workers } else { 0 };
                assert_eq!(
                    hooks.started.load(std::sync::atomic::Ordering::Relaxed),
                    threads,
                    "{at}"
                );
            }
            let path = std::env::temp_dir().join(format!(
                "invar-m2-05-scan-{exclusive}-{len}-{}.inv",
                std::process::id()
            ));
            std::fs::write(&path, &source).unwrap();
            let run = |engine: &[&str]| {
                let mut args = vec!["run", path.to_str().unwrap()];
                args.extend(engine);
                args.extend(["--format", "json"]);
                Command::new(env!("CARGO_BIN_EXE_invar"))
                    .args(&args)
                    .output()
                    .unwrap()
            };
            let reference = run(&["--engine", "reference"]);
            assert!(reference.status.success());
            for workers in ["1", "2", "4"] {
                let output = run(&["--engine", "cpu-parallel", "--workers", workers]);
                assert_eq!(output.status.code(), Some(0), "{exclusive} {len} {workers}");
                assert_eq!(
                    output.stdout, reference.stdout,
                    "{exclusive} {len} {workers}"
                );
            }
            std::fs::remove_file(path).unwrap();
        }
    }
}

fn zip_source(len: usize) -> String {
    let left = (0..len).map(|i| i as u32 + 1).collect::<Vec<_>>();
    let right = (0..len + 2)
        .map(|i| (i as u32).wrapping_mul(3))
        .collect::<Vec<_>>();
    let mut source = format!(
        "fn main() -> i32 {{ let left: Array<u32> = [{}]; let right: Array<u32> = [{}]; var first: Array<u32> = filled_u32(len(left),0u32); for i in 0u32 .. len(left) {{ first[i] = wrapping_add(left[i],right[i]); }} let stable: Array<u32> = move first; var second: Array<u32> = filled_u32(len(stable),0u32); for j in 0u32 .. len(stable) {{ second[j] = wrapping_mul(stable[j],7u32); }}",
        literal(&left),
        literal(&right)
    );
    for i in 0..len {
        source.push_str(&format!(
            "if second[{i}u32] != {}u32 {{ return 1; }}",
            left[i].wrapping_add(right[i]).wrapping_mul(7)
        ));
    }
    source.push_str("return 0; }");
    source
}

#[test]
fn cpu_scheduler_fallback_unequal_inputs_true_guard() {
    let source = zip_source(257);
    let (checked, program) = lower(&source);
    let reference = eval::observe(&checked, 1_000_000);
    let serial = mir_eval::observe(&program, 1_000_000);
    assert_eq!(serial.result, reference.result);
    assert_eq!(serial.result, Ok(eval::Value::I32(0)));
    for workers in [1, 2, 4] {
        let run = schedule::run_source(source.as_bytes(), &program, 1_000_000, workers).unwrap();
        assert_eq!(run.plan, schedule::PlanKind::FusedMap);
        assert_eq!(run.observation.result, serial.result);
        assert_eq!(run.observation.cleanup_trace, serial.cleanup_trace);
        assert_eq!(run.observation.steps_used, serial.steps_used);
    }
}

#[test]
fn cpu_scheduler_fallback_all_families_shapes_workers_and_cleanup() {
    for len in [0, 1, 3, 7, 65, 257] {
        for (kind, source) in [
            (schedule::PlanKind::FusedMap, map_source(len)),
            (schedule::PlanKind::Reduction, reduction_source(len, false)),
            (schedule::PlanKind::Reduction, reduction_source(len, true)),
            (schedule::PlanKind::Scan, scan_source(len)),
        ] {
            let (checked, program) = lower(&source);
            let reference = eval::observe(&checked, 1_000_000);
            let serial = mir_eval::observe(&program, 1_000_000);
            assert_eq!(reference.result, Ok(eval::Value::I32(0)), "len {len}");
            assert_eq!(serial.result, reference.result, "len {len}");
            assert_eq!(serial.cleanup_trace, reference.cleanup_trace, "len {len}");
            for workers in [1, 2, 4] {
                let hooks = schedule::Hooks::default();
                let run = schedule::run_source_with_hooks(
                    source.as_bytes(),
                    &program,
                    1_000_000,
                    workers,
                    &hooks,
                )
                .unwrap();
                assert_eq!(
                    run.observation.result, serial.result,
                    "len {len}, workers {workers}"
                );
                assert_eq!(
                    run.observation.cleanup_trace, serial.cleanup_trace,
                    "len {len}, workers {workers}"
                );
                assert_eq!(
                    run.observation.steps_used, serial.steps_used,
                    "len {len}, workers {workers}"
                );
                if len > 0 {
                    assert_eq!(run.plan, kind);
                }
                let expected_threads = if len >= 64 && workers > 1 { workers } else { 0 };
                assert_eq!(
                    hooks.started.load(std::sync::atomic::Ordering::Relaxed),
                    expected_threads
                );
                assert_eq!(run.joined, expected_threads);
                if len >= 64 {
                    assert_eq!(run.chunks.first().unwrap().start, 0);
                    assert_eq!(run.chunks.last().unwrap().end, len);
                    assert!(
                        run.chunks
                            .windows(2)
                            .all(|pair| pair[0].end == pair[1].start)
                    );
                }
            }
        }
    }
}

#[test]
fn cpu_workers_one_two_four_and_hardware_width_match_serial() {
    use std::sync::atomic::Ordering;
    let n = std::thread::available_parallelism()
        .map_or(1, |count| count.get())
        .min(schedule::MAX_WORKERS);
    // The cutoff is 64 total elements, not 64 per worker. Keep every worker
    // active while the unrolled value checks stay within MIR resource bounds.
    let len = n.max(8) * 8;
    for source in [
        map_source(len),
        reduction_source(len, false),
        scan_source(len),
        zip_source(len),
    ] {
        let (_, program) = lower(&source);
        let serial = mir_eval::observe(&program, 1_000_000);
        for workers in [1, 2, 4, n] {
            let hooks = schedule::Hooks::default();
            let run = schedule::run_source_with_hooks(
                source.as_bytes(),
                &program,
                1_000_000,
                workers,
                &hooks,
            )
            .unwrap();
            assert_eq!(run.observation.result, serial.result, "workers={workers}");
            assert_eq!(run.observation.cleanup_trace, serial.cleanup_trace);
            assert_eq!(run.observation.steps_used, serial.steps_used);
            let expected = if workers == 1 { 0 } else { workers };
            assert_eq!(hooks.started.load(Ordering::SeqCst), expected);
            assert_eq!(hooks.joined.load(Ordering::SeqCst), expected);
            assert_eq!(run.joined, expected);
            assert_eq!(run.chunks.len(), workers);
            assert_eq!(run.chunks.first().unwrap().start, 0);
            assert_eq!(run.chunks.last().unwrap().end, len);
            assert!(
                run.chunks
                    .windows(2)
                    .all(|pair| pair[0].end == pair[1].start)
            );
        }
    }
}

#[test]
fn cpu_scheduler_fallback_false_guard_is_serial_before_workers() {
    let source = "fn main() -> i32 { let left: Array<u32> = [1u32,2u32,3u32]; let right: Array<u32> = [4u32,5u32]; var first: Array<u32> = filled_u32(len(left),0u32); for i in 0u32 .. len(left) { first[i] = wrapping_add(left[i],right[i]); } let stable: Array<u32> = move first; var second: Array<u32> = filled_u32(len(stable),0u32); for j in 0u32 .. len(stable) { second[j] = wrapping_mul(stable[j],7u32); } return 0; }";
    let (checked, program) = lower(source);
    let reference = eval::observe(&checked, 100_000);
    let serial = mir_eval::observe(&program, 100_000);
    assert_eq!(serial.result, reference.result);
    let hooks = schedule::Hooks::default();
    let run =
        schedule::run_source_with_hooks(source.as_bytes(), &program, 100_000, 4, &hooks).unwrap();
    assert_eq!(run.plan, schedule::PlanKind::Serial);
    assert_eq!(run.fallback_reason, Some("UNREACHED_OR_GUARD_FALSE"));
    assert_eq!(run.observation.result, serial.result);
    assert_eq!(run.observation.cleanup_trace, serial.cleanup_trace);
    assert_eq!(hooks.started.load(std::sync::atomic::Ordering::Relaxed), 0);
}

#[test]
fn cpu_scheduler_fallback_checked_overflow_stays_serial() {
    let source = "fn main() -> i32 { let xs: Array<i32> = [2147483647, 1]; var acc: i32 = 0; for i in 0u32 .. len(xs) { acc = acc + xs[i]; } return acc; }";
    let (checked, program) = lower(source);
    let reference = eval::observe(&checked, 100_000);
    let serial = mir_eval::observe(&program, 100_000);
    assert_eq!(serial.result, reference.result);
    let hooks = schedule::Hooks {
        fault: schedule::Fault::Spawn,
        ..schedule::Hooks::default()
    };
    let run =
        schedule::run_source_with_hooks(source.as_bytes(), &program, 100_000, 4, &hooks).unwrap();
    assert_eq!(run.plan, schedule::PlanKind::Serial);
    assert_eq!(run.observation.result, serial.result);
    assert_eq!(run.observation.cleanup_trace, serial.cleanup_trace);
    assert_eq!(hooks.started.load(std::sync::atomic::Ordering::Relaxed), 0);
}

#[test]
fn cpu_scheduler_fallback_injected_failures_publish_nothing() {
    use std::sync::atomic::Ordering;
    let workers = std::thread::available_parallelism()
        .map_or(2, |count| count.get())
        .clamp(2, schedule::MAX_WORKERS);
    // Injected worker panics originate in the scheduler's worker module; every
    // other panic keeps the default report.
    let default_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        let injected = info.location().is_some_and(|location| {
            location
                .file()
                .replace('\\', "/")
                .ends_with("src/schedule/workers.rs")
        });
        if !injected {
            default_hook(info);
        }
    }));
    for source in [
        map_source(workers.max(8) * 8),
        scan_source(workers.max(8) * 8),
        zip_source(workers.max(8) * 8),
        reduction_source(workers.max(8) * 8, false),
    ] {
        let (_, program) = lower(&source);
        let serial = mir_eval::observe(&program, 1_000_000);
        assert_eq!(serial.result, Ok(eval::Value::I32(0)));
        for (fault, expected, code, running) in [
            (
                schedule::Fault::Spawn,
                schedule::Error::Spawn,
                "CPU_SPAWN",
                workers - 1,
            ),
            (
                schedule::Fault::Join,
                schedule::Error::Join,
                "CPU_JOIN",
                workers,
            ),
            (
                schedule::Fault::Panic,
                schedule::Error::WorkerPanic,
                "CPU_WORKER_PANIC",
                workers,
            ),
            (
                schedule::Fault::Allocation,
                schedule::Error::Allocation,
                "RESOURCE_LIMIT",
                0,
            ),
            (
                schedule::Fault::Overlap,
                schedule::Error::Overlap,
                "CPU_CHUNK_OVERLAP",
                0,
            ),
            (
                schedule::Fault::Timeout,
                schedule::Error::Timeout,
                "CPU_TIMEOUT",
                0,
            ),
            (
                schedule::Fault::Cancel,
                schedule::Error::Cancelled,
                "CPU_CANCELLED",
                0,
            ),
        ] {
            let hooks = schedule::Hooks {
                fault,
                ..schedule::Hooks::default()
            };
            let result = schedule::run_source_with_hooks(
                source.as_bytes(),
                &program,
                1_000_000,
                workers,
                &hooks,
            );
            let error = result.unwrap_err();
            assert_eq!(error, expected, "{fault:?}");
            assert_eq!(error.code(), code, "{fault:?}");
            // Workers that started are all joined before the typed error is returned.
            let started = hooks.started.load(Ordering::SeqCst);
            assert_eq!(started, running, "{fault:?}");
            assert_eq!(hooks.joined.load(Ordering::SeqCst), started, "{fault:?}");
            // The runtime arrays at the failing loop are exactly their pre-region contents.
            let live = hooks.live_arrays.lock().unwrap().clone().unwrap();
            assert!(!live.before.is_empty(), "{fault:?}");
            assert_eq!(live.before, live.after, "{fault:?}");
        }
    }
    drop(std::panic::take_hook());
}

#[test]
fn cpu_scheduler_fallback_cli_validation_and_scalar_serial() {
    let source = "fn main() -> i32 { return 42; }";
    let (checked, program) = lower(source);
    let reference = eval::observe(&checked, 100);
    let serial = mir_eval::observe(&program, 100);
    assert_eq!(serial.result, reference.result);
    let run = schedule::run_source(source.as_bytes(), &program, 100, 4).unwrap();
    assert_eq!(run.plan, schedule::PlanKind::Serial);
    assert_eq!(run.fallback_reason, Some("NO_ADMITTED_CANDIDATE"));
    assert_eq!(run.observation.result, serial.result);
    assert_eq!(run.workers_used, 0);

    let path = std::env::temp_dir().join(format!("invar-m2-05-scalar-{}.inv", std::process::id()));
    std::fs::write(&path, source).unwrap();
    let file = path.to_str().unwrap();
    let good = Command::new(env!("CARGO_BIN_EXE_invar"))
        .args([
            "run",
            file,
            "--engine",
            "cpu-parallel",
            "--workers",
            "4",
            "--format",
            "json",
        ])
        .output()
        .unwrap();
    let baseline = Command::new(env!("CARGO_BIN_EXE_invar"))
        .args(["run", file, "--engine", "reference", "--format", "json"])
        .output()
        .unwrap();
    let good_json: serde_json::Value = serde_json::from_slice(&good.stdout).unwrap();
    let baseline_json: serde_json::Value = serde_json::from_slice(&baseline.stdout).unwrap();
    assert_eq!(good_json["result"], baseline_json["result"]);
    assert_eq!(good_json["diagnostics"][0]["code"], "NO_ADMITTED_CANDIDATE");
    assert_eq!(good_json["diagnostics"][0]["severity"], "note");
    assert_eq!(baseline_json["diagnostics"], serde_json::json!([]));
    assert!(good.status.success());
    for extra in [
        vec!["--engine", "cpu-parallel"],
        vec!["--engine", "cpu-parallel", "--workers", "0"],
        vec!["--engine", "cpu-parallel", "--workers", "65"],
        vec!["--engine", "reference", "--workers", "4"],
    ] {
        let output = Command::new(env!("CARGO_BIN_EXE_invar"))
            .args(["run", file])
            .args(extra)
            .output()
            .unwrap();
        assert_eq!(output.status.code(), Some(2));
    }
    std::fs::remove_file(path).unwrap();
}

#[test]
fn cpu_scheduler_fallback_fuel_limit_preserves_failure_prefix() {
    let source = map_source(257);
    let (checked, program) = lower(&source);
    for fuel in [1, 100, 2_000] {
        let reference = eval::observe(&checked, fuel);
        let serial = mir_eval::observe(&program, fuel);
        assert_eq!(serial.result, reference.result);
        let run = schedule::run_source(source.as_bytes(), &program, fuel, 4).unwrap();
        assert_eq!(run.observation.result, serial.result, "fuel {fuel}");
        assert_eq!(run.observation.steps_used, serial.steps_used, "fuel {fuel}");
        assert_eq!(
            run.observation.cleanup_trace, serial.cleanup_trace,
            "fuel {fuel}"
        );
    }
}

#[test]
fn cpu_scheduler_fallback_preloop_trap_and_fuel_precede_worker_fault() {
    let source = map_source(257).replacen("let xs:", "let boom: i32 = 1 / 0; let xs:", 1);
    let (checked, program) = lower(&source);
    let reference = eval::observe(&checked, 1_000_000);
    let serial = mir_eval::observe(&program, 1_000_000);
    assert_eq!(serial.result, reference.result);
    let hooks = schedule::Hooks {
        fault: schedule::Fault::Spawn,
        ..schedule::Hooks::default()
    };
    let run =
        schedule::run_source_with_hooks(source.as_bytes(), &program, 1_000_000, 4, &hooks).unwrap();
    assert_eq!(run.plan, schedule::PlanKind::Serial);
    assert_eq!(run.observation.result, serial.result);
    assert_eq!(hooks.started.load(std::sync::atomic::Ordering::Relaxed), 0);

    let source = map_source(257);
    let (checked, program) = lower(&source);
    let reference = eval::observe(&checked, 100);
    let serial = mir_eval::observe(&program, 100);
    assert_eq!(serial.result, reference.result);
    let run = schedule::run_source_with_hooks(source.as_bytes(), &program, 100, 4, &hooks).unwrap();
    assert_eq!(run.observation.result, serial.result);
    assert_eq!(hooks.started.load(std::sync::atomic::Ordering::Relaxed), 0);
}

fn explain_kinds(source: &str) -> Vec<(usize, usize, schedule::PlanKind, String)> {
    let path = std::env::temp_dir().join(format!(
        "invar-m2-05-explain-{}-{}.inv",
        std::process::id(),
        source.len()
    ));
    std::fs::write(&path, source).unwrap();
    let output = Command::new(env!("CARGO_BIN_EXE_invar"))
        .args(["explain", path.to_str().unwrap(), "--format", "json"])
        .output()
        .unwrap();
    std::fs::remove_file(&path).unwrap();
    assert!(output.status.success());
    let document: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    document["regions"]
        .as_array()
        .unwrap()
        .iter()
        .map(|region| {
            let kind = match region["execution"]["kind"].as_str().unwrap() {
                "serial" => schedule::PlanKind::Serial,
                "map" => schedule::PlanKind::Map,
                "fused-map-producer" | "fused-map-consumer" => schedule::PlanKind::FusedMap,
                "reduction" => schedule::PlanKind::Reduction,
                "scan" => schedule::PlanKind::Scan,
                other => panic!("unknown execution kind {other}"),
            };
            assert_eq!(
                region["status"] == "admitted",
                kind != schedule::PlanKind::Serial
            );
            (
                region["span"]["start"].as_u64().unwrap() as usize,
                region["span"]["end"].as_u64().unwrap() as usize,
                kind,
                region["chosen_plan"].as_str().unwrap().to_string(),
            )
        })
        .collect()
}

/// `explain` reports exactly the plan the CPU scheduler publishes for every region.
#[test]
fn cpu_scheduler_fallback_explain_matches_executed_plans() {
    let xs = (0..70)
        .map(|i| format!("{}u32", (i as u32).wrapping_mul(2_654_435_761)))
        .collect::<Vec<_>>()
        .join(",");
    let prefix = format!(
        "fn main() -> i32 {{ let xs: Array<u32> = [{xs}]; var ys: Array<u32> = filled_u32(len(xs),0u32);"
    );
    let corpus = [
        (
            format!(
                "{prefix} for i in 0u32 .. len(xs) {{ ys[i] = wrapping_add(xs[i],1u32); }} return 0; }}"
            ),
            vec![schedule::PlanKind::Map],
        ),
        (map_source(70), vec![schedule::PlanKind::FusedMap; 2]),
        (zip_source(70), vec![schedule::PlanKind::FusedMap; 2]),
        (
            reduction_source(70, false),
            vec![schedule::PlanKind::Reduction],
        ),
        (
            reduction_source(70, true),
            vec![schedule::PlanKind::Reduction],
        ),
        (scan_source(70), vec![schedule::PlanKind::Scan]),
        (
            format!(
                "{prefix} var s: u32 = 3u32; for i in 0u32 .. len(xs) {{ ys[i] = s; s = wrapping_add(wrapping_mul(5u32,s),xs[i]); }} return 0; }}"
            ),
            vec![schedule::PlanKind::Scan],
        ),
        (
            format!("{prefix} for i in 0u32 .. len(xs) {{ ys[i] = xs[i] + 1u32; }} return 0; }}"),
            vec![schedule::PlanKind::Serial],
        ),
        (
            format!(
                "{prefix} for i in 0u32 .. len(xs) {{ ys[i] = wrapping_add(xs[0u32],1u32); }} return 0; }}"
            ),
            vec![schedule::PlanKind::Serial],
        ),
        (
            format!(
                "{prefix} var acc: u32 = 0u32; for i in 0u32 .. len(xs) {{ acc = acc + xs[i]; }} return 0; }}"
            ),
            vec![schedule::PlanKind::Serial],
        ),
        (
            format!(
                "{prefix} let c: u32 = 1u32; if c == 2u32 {{ for i in 0u32 .. len(xs) {{ ys[i] = wrapping_add(xs[i],1u32); }} }} let zs: Array<u32> = move ys; var ws: Array<u32> = filled_u32(len(zs),0u32); for j in 0u32 .. len(zs) {{ ws[j] = wrapping_mul(zs[j],3u32); }} return 0; }}"
            ),
            vec![schedule::PlanKind::Map, schedule::PlanKind::Map],
        ),
    ];
    for (source, expected) in corpus {
        let regions = explain_kinds(&source);
        assert_eq!(
            regions.iter().map(|r| r.2).collect::<Vec<_>>(),
            expected,
            "{source}"
        );
        let (checked, program) = lower(&source);
        let reference = eval::observe(&checked, 1_000_000);
        for workers in 1..=schedule::MAX_WORKERS {
            let run =
                schedule::run_source(source.as_bytes(), &program, 1_000_000, workers).unwrap();
            assert_eq!(run.observation.result, reference.result, "{source}");
            assert_eq!(run.observation.steps_used, reference.steps_used);
            // Every published step belongs to a region explain reported with that plan,
            // and every region explain admitted and the run reached was published.
            for step in &run.executed {
                let region = regions
                    .iter()
                    .find(|r| r.0 == step.span.start && r.1 == step.span.end)
                    .unwrap();
                assert_eq!(region.2, step.kind, "{} {source}", region.3);
            }
            let unreached = usize::from(source.contains("if c == 2u32"));
            assert_eq!(
                run.executed.len() + unreached,
                regions
                    .iter()
                    .filter(|r| r.2 != schedule::PlanKind::Serial)
                    .count(),
                "{source}"
            );
        }
    }
}
