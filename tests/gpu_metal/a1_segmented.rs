//! Ordinary-source segmented publication on a real Metal device; the CLI refusal test stays in `tests/issue_a1_segmented.rs`.
use invar::mir_eval;
use std::path::Path;

#[path = "../support/a1_segmented_common.rs"]
mod a1_segmented_common;
use a1_segmented_common::{cases, cli, program, source};

#[test]
#[ignore = "requires physical Metal"]
fn metal_source_equality_cost_fallback_and_completion_failures() {
    use invar::{
        plan::DeviceCostModel,
        schedule::{PlanKind, metal},
    };
    use invar_metal::{Fault, MetalDevice};
    let mut device = MetalDevice::open_runtime_source().unwrap();
    let mut passed = 0;
    for case in cases::cases(false) {
        let s = source(&case);
        let p = program(&s);
        let serial = mir_eval::observe(&p, 10_000_000);
        assert_eq!(serial.result, Ok(invar::eval::Value::I32(0)));
        let zero = match case.kind {
            cases::Kind::Compact(_) => case.input.is_empty(),
            _ => case.offsets.len() == 1,
        };
        let started = std::time::Instant::now();
        let run = metal::run_source_with_device(s.as_bytes(), &p, 10_000_000, &mut device);
        let run_ms = started.elapsed().as_secs_f64() * 1000.0;
        if zero {
            assert_eq!(run.err().unwrap().code, "METAL_UNAVAILABLE");
            continue;
        }
        let run = run.unwrap_or_else(|e| panic!("{}: {e:?}", case.name));
        assert_eq!(run.observation.result, serial.result, "{}", case.name);
        assert_eq!(
            run.observation.steps_used, serial.steps_used,
            "{}",
            case.name
        );
        assert_eq!(
            run.observation.cleanup_trace, serial.cleanup_trace,
            "{}",
            case.name
        );
        assert!(
            run.executed
                .iter()
                .any(|r| matches!(r.kind, PlanKind::Segmented | PlanKind::Compaction))
        );
        let plan = cases::plan(&case, device.capabilities());
        let started = std::time::Instant::now();
        let host = device
            .run_segmented(&plan, &case.input, &case.offsets)
            .unwrap();
        let host_ms = started.elapsed().as_secs_f64() * 1000.0;
        assert_eq!(host.output, cases::sequential_expected(&case));
        println!(
            "{} run_ms={run_ms:.3} host_ms={host_ms:.3} kernel_ms={:.3} regions={}",
            case.name,
            host.kernel_ms,
            run.executed.len()
        );
        passed += 1;
    }
    assert!(passed >= 40);
    let case = cases::cases(false)
        .into_iter()
        .find(|c| c.name == "Map/empty-interior")
        .unwrap();
    let s = source(&case);
    let p = program(&s);
    let serial = mir_eval::observe(&p, 10_000_000);
    for (cpu, gpu, fixed, chosen) in [(1e12, 1.0, 1000.0, false), (1.0, 1e12, 0.0, true)] {
        let run = metal::run_auto_source_with_fault(
            s.as_bytes(),
            &p,
            10_000_000,
            metal::Kernels::RuntimeSource,
            DeviceCostModel {
                cpu_bytes_per_ms: cpu,
                device_bytes_per_ms: gpu,
                device_fixed_ms: fixed,
            },
            Fault::None,
        )
        .unwrap();
        assert_eq!(run.observation.result, serial.result);
        assert_eq!(run.observation.steps_used, serial.steps_used);
        assert_eq!(
            run.executed.iter().any(|r| r.kind == PlanKind::Segmented),
            chosen
        );
    }
    let mut offsets = case.offsets.clone();
    offsets[1] += 1;
    let declaration = |values: &[u32]| {
        format!(
            "let offsets: Array<u32> = [{}];",
            values
                .iter()
                .map(|v| format!("{v}u32"))
                .collect::<Vec<_>>()
                .join(",")
        )
    };
    let bad_source = s.replacen(&declaration(&case.offsets), &declaration(&offsets), 1);
    assert_ne!(bad_source, s);
    let bad_program = program(&bad_source);
    let bad_serial = mir_eval::observe(&bad_program, 10_000_000);
    let model = DeviceCostModel {
        cpu_bytes_per_ms: 1.0,
        device_bytes_per_ms: 1e12,
        device_fixed_ms: 0.0,
    };
    let guard_fallback = metal::run_auto_source_with_fault(
        bad_source.as_bytes(),
        &bad_program,
        10_000_000,
        metal::Kernels::RuntimeSource,
        model,
        Fault::None,
    )
    .unwrap();
    assert_eq!(guard_fallback.observation.result, bad_serial.result);
    assert!(guard_fallback.executed.is_empty());
    let forced =
        metal::run_source_with_device(bad_source.as_bytes(), &bad_program, 10_000_000, &mut device)
            .unwrap();
    assert!(format!("{:?}", forced.observation.result).contains("METAL_UNAVAILABLE"));
    assert!(forced.executed.is_empty());
    for fault in [Fault::OutOfMemory, Fault::LaunchFailure] {
        let fallback = metal::run_auto_source_with_fault(
            s.as_bytes(),
            &p,
            10_000_000,
            metal::Kernels::RuntimeSource,
            model,
            fault,
        )
        .unwrap();
        assert_eq!(fallback.observation.result, serial.result);
        assert!(fallback.executed.is_empty());
    }
    let pending = metal::run_auto_source_with_fault(
        s.as_bytes(),
        &p,
        10_000_000,
        metal::Kernels::RuntimeSource,
        model,
        Fault::PartialCompletion,
    )
    .unwrap();
    assert!(format!("{:?}", pending.observation.result).contains("METAL_PENDING"));
    assert!(pending.executed.is_empty());
    for fault in [
        Fault::OutOfMemory,
        Fault::LaunchFailure,
        Fault::PartialCompletion,
    ] {
        let mut device = MetalDevice::open_runtime_source().unwrap();
        let failed =
            metal::run_source_with_device_fault(s.as_bytes(), &p, 10_000_000, &mut device, fault)
                .unwrap();
        assert!(failed.executed.is_empty());
        assert!(failed.observation.result.is_err());
        assert_eq!(device.is_poisoned(), fault == Fault::PartialCompletion);
    }
    let dir = std::env::temp_dir().join(format!("invar-a1-metal-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("segmented.inv");
    std::fs::write(&path, s).unwrap();
    for engine in ["metal", "auto"] {
        let result = cli(engine, &path);
        assert_eq!(result["status"], "ok", "{result}");
        println!("CLI {engine}: {result}");
    }
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
#[ignore = "requires physical Metal"]
fn metal_cli_demo() {
    use invar::{
        eval::Value,
        schedule::{PlanKind, metal},
    };
    use invar_metal::{Fault, MetalDevice};
    let source = include_str!("../../demo/a1-segmented.inv");
    let program = program(source);
    let serial = mir_eval::observe(&program, 100_000);
    assert_eq!(serial.result, Ok(Value::I32(0)));
    let mut device = MetalDevice::open_runtime_source().unwrap();
    let run =
        metal::run_source_with_device(source.as_bytes(), &program, 100_000, &mut device).unwrap();
    assert_eq!(run.observation.result, serial.result);
    assert_eq!(run.observation.steps_used, serial.steps_used);
    assert_eq!(run.observation.cleanup_trace, serial.cleanup_trace);
    assert_eq!(
        run.executed.iter().map(|r| r.kind).collect::<Vec<_>>(),
        [
            PlanKind::Segmented,
            PlanKind::Segmented,
            PlanKind::Compaction,
            PlanKind::Compaction
        ]
    );
    println!(
        "Metal completed regions: Segmented(map), Segmented(sum), Compaction(count), Compaction(copy)"
    );
    println!(
        "All output words, fuel={} and cleanup match serial MIR",
        run.observation.steps_used
    );
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("demo/a1-segmented.inv");
    for engine in ["metal", "auto"] {
        let result = cli(engine, &path);
        assert_eq!(result["status"], "ok");
        assert_eq!(result["result"]["value"], 0);
        println!("invar run demo/a1-segmented.inv --engine {engine} --format json\n{result}");
    }
    let failed = metal::run_source_with_device_fault(
        source.as_bytes(),
        &program,
        100_000,
        &mut device,
        Fault::PartialCompletion,
    )
    .unwrap();
    assert!(failed.executed.is_empty());
    assert!(format!("{:?}", failed.observation.result).contains("METAL_PENDING"));
}
