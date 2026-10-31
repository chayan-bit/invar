//! Issue #8 (M4-06).
use invar::{
    admission, check, device_cache, eval, front, mir,
    plan::DeviceCostModel,
    schedule::{self, tuning::*},
};
use invar_device::{Capabilities, Geometry};
#[path = "support/pair_reduction.rs"]
mod pair_reduction_support;

const MAP: &str = "fn main() -> i32 { let xs: Array<u32> = [0u32,4294967295u32,7u32]; var ys: Array<u32> = filled_u32(len(xs),0u32); for i in 0u32 .. len(xs) { ys[i] = wrapping_add(xs[i],5u32); } if ys[0u32] != 5u32 { return 1; } if ys[1u32] != 4u32 { return 2; } if ys[2u32] != 12u32 { return 3; } return 0; }";
fn region() -> admission::Region {
    let program = lower(MAP);
    admission::analyze(MAP.as_bytes(), &program)
        .unwrap()
        .regions()[0]
        .clone()
}
fn lower(source: &str) -> mir::Program {
    mir::lower(&check::check(front::parse(source.as_bytes()).unwrap(), true).unwrap()).unwrap()
}
fn variant() -> Schedule {
    Schedule::m3(Geometry::new(1024, 256))
}
fn candidates(n: u32) -> Vec<Candidate> {
    enumerate(&region(), n, Capabilities::cuda_12_9_sm89(), [variant()]).unwrap()
}
fn model() -> DeviceCostModel {
    DeviceCostModel {
        cpu_bytes_per_ms: 1_000_000.0,
        device_bytes_per_ms: 100_000.0,
        device_fixed_ms: 0.01,
    }
}

#[test]
fn schedule_tuning_enumeration_and_refusals() {
    let r = region();
    let caps = Capabilities::cuda_12_9_sm89();
    let empty = enumerate(&r, 0, caps, [variant()]).unwrap();
    let empty_facts = empty[0].plan().cost_facts().unwrap();
    assert_eq!(
        (
            empty_facts.input_bytes,
            empty_facts.output_bytes,
            empty_facts.temporary_bytes,
            empty_facts.launches
        ),
        (0, 0, 0, 0)
    );
    let list = enumerate(
        &r,
        3,
        caps,
        (0..32).map(|i| {
            let mut s = variant();
            s.cpu_threshold = i;
            s
        }),
    )
    .unwrap();
    assert_eq!(list.len(), 32);
    assert_eq!(
        list.iter().map(Candidate::index).collect::<Vec<_>>(),
        (0..32).collect::<Vec<_>>()
    );
    assert_eq!(
        enumerate(&r, 3, caps, std::iter::repeat(variant())).err(),
        Some(Refusal::TooManyVariants)
    );
    assert_eq!(
        enumerate(&r, 3, caps, []).err(),
        Some(Refusal::NoCandidates)
    );
    let mut bad = variant();
    bad.geometry.lanes = 3;
    assert_eq!(
        enumerate(&r, 3, caps, [bad]).err(),
        Some(Refusal::UnsupportedSchedule)
    );
    bad = variant();
    bad.geometry.blocks = 0;
    assert_eq!(
        enumerate(&r, 3, caps, [bad]).err(),
        Some(Refusal::Admission)
    );
    for axis in 0..3 {
        bad = variant();
        match axis {
            0 => bad.vector_width = 4,
            1 => bad.frontier_chunk = Some(32),
            _ => bad.fusion_cutoff = Some(4),
        }
        assert_eq!(
            enumerate(&r, 3, caps, [bad]).err(),
            Some(Refusal::UnsupportedSchedule)
        );
    }
    assert_eq!(
        measurement_gate(TuningUse::ProductionVisible),
        Err(Refusal::ProductionOutput)
    );
    assert_eq!(
        measurement_gate(TuningUse::Runtime),
        Err(Refusal::RuntimeTuning)
    );
    assert_eq!(
        measurement_gate(TuningUse::OfflineTuning),
        Err(Refusal::FreezeRequired)
    );
    let b = Budget {
        compile_ms: 100,
        tuning_ms: 1000,
        temporary_bytes: 4096,
    };
    assert!(b.check(100, 1000, 4096).is_ok());
    for (c, t, m) in [(101, 0, 0), (0, 1001, 0), (0, 0, 4097)] {
        assert_eq!(b.check(c, t, m), Err(Refusal::Budget));
    }
    assert_eq!((WARMUPS, SAMPLES, SEED), (10, 30, 0x494e_5641_5206));
}

#[test]
fn schedule_tuning_cost_residency_and_controls() {
    let c = candidates(100_000);
    let plan = c[0].plan();
    let (host, h) = select_map(plan, Residency::HOST, model(), Control::Auto, 0).unwrap();
    let (resident, d) = select_map(plan, Residency::DEVICE, model(), Control::Auto, 0).unwrap();
    assert_eq!(host.engine, Engine::Cpu);
    assert_eq!(resident.engine, Engine::Gpu);
    assert_eq!((h.transfer_bytes, d.transfer_bytes), (800_000, 0));
    assert_eq!(
        (d.shape.as_slice(), d.launches, d.temporary_bytes),
        (&[100_000][..], 1, 0)
    );
    assert_eq!(d.timings, Timings::default());
    assert_eq!(d.completion, Completion::NotSubmitted);
    assert_eq!(d.track, Track::Generated);
    assert_ne!(Track::Generated, Track::LibraryAssisted);
    assert!(d.estimate.cpu_ms >= h.estimate.cpu_ms);
    let tiny = candidates(3);
    assert_eq!(
        select_map(tiny[0].plan(), Residency::HOST, model(), Control::Auto, 0)
            .unwrap()
            .0
            .engine,
        Engine::Cpu
    );
    for (control, engine) in [
        (Control::Serial, Engine::Serial),
        (Control::Cpu, Engine::Cpu),
        (Control::Gpu, Engine::Gpu),
    ] {
        assert_eq!(
            decide(control, Some(&h), 0, 0, true, true).unwrap().engine,
            engine
        );
    }
    assert_eq!((d.elements, d.operations, d.model), (100_000, 1, model()));
    let mut stale_model = d.clone();
    stale_model.model.device_fixed_ms += 1.0;
    assert_eq!(stale_model.validate(), Err(Refusal::UnknownCost));
    let mut stale_work = d.clone();
    stale_work.operations += 1;
    assert_eq!(stale_work.validate(), Err(Refusal::UnknownCost));
    let mut bad_shape = d.clone();
    bad_shape.shape = vec![usize::MAX, 2];
    assert_eq!(bad_shape.validate(), Err(Refusal::UnknownCost));
    // A shape that disagrees with `elements` must not pass even when the
    // public estimate is recomputed consistently with `elements`.
    let mut skewed_shape = d.clone();
    skewed_shape.shape = vec![d.elements * 2];
    assert_eq!(skewed_shape.validate(), Err(Refusal::UnknownCost));
    assert_eq!(
        decide(Control::Auto, Some(&skewed_shape), 0, 0, true, true)
            .unwrap()
            .refusal,
        Some(Refusal::UnknownCost)
    );
    let mut split_shape = d.clone();
    split_shape.shape = vec![d.elements / 2, 2];
    assert_eq!(split_shape.validate(), Ok(()));
    let mut no_residency = d.clone();
    no_residency.residency = Residency::HOST;
    assert_eq!(no_residency.validate(), Err(Refusal::UnknownCost));
    let mut bad = d.clone();
    bad.estimate.cpu_ms = f64::NAN;
    assert_eq!(
        decide(Control::Auto, Some(&bad), 0, 0, true, true)
            .unwrap()
            .refusal,
        Some(Refusal::UnknownCost)
    );
    assert_eq!(
        CostRecord::from_plan(
            plan,
            Residency::HOST,
            DeviceCostModel {
                device_fixed_ms: f64::NAN,
                ..model()
            }
        )
        .err(),
        Some(Refusal::UnknownCost)
    );
    eprintln!("host={h:?}\nresident={d:?}\nchoice={host:?}->{resident:?}");
}

#[test]
fn schedule_tuning_every_failure_is_typed_and_pending_never_replays() {
    let r = CostRecord::from_plan(candidates(3)[0].plan(), Residency::HOST, model()).unwrap();
    for (guard, available, record, reason) in [
        (false, true, Some(&r), Refusal::Guard),
        (true, false, Some(&r), Refusal::MissingDevice),
        (true, true, None, Refusal::UnknownCost),
    ] {
        assert_eq!(
            decide(Control::Auto, record, 0, 0, guard, available)
                .unwrap()
                .refusal,
            Some(reason)
        );
        assert_eq!(
            decide(Control::Gpu, record, 0, 0, guard, available),
            Err(reason)
        );
    }
    for reason in [
        Refusal::Transfer,
        Refusal::Launch,
        Refusal::NumericPolicy,
        Refusal::StaleOrForgedHint,
    ] {
        for completion in [Completion::NotSubmitted, Completion::FailedQuiescent] {
            assert_eq!(
                failed(Control::Auto, reason, completion).unwrap().engine,
                Engine::Cpu
            );
            assert_eq!(failed(Control::Gpu, reason, completion), Err(reason));
        }
        assert_eq!(
            failed(Control::Auto, reason, Completion::Unknown),
            Err(Refusal::PartialCompletion)
        );
    }
    for completion in [
        Completion::NotSubmitted,
        Completion::FailedQuiescent,
        Completion::Completed,
        Completion::Unknown,
    ] {
        assert_eq!(
            failed(Control::Auto, Refusal::PartialCompletion, completion),
            Err(Refusal::PartialCompletion)
        );
    }
    let mut pending = r;
    pending.completion = Completion::Unknown;
    for control in [Control::Auto, Control::Serial, Control::Cpu, Control::Gpu] {
        assert_eq!(
            decide(control, Some(&pending), 0, 0, true, true),
            Err(Refusal::PartialCompletion)
        );
    }
}

#[test]
fn schedule_tuning_cache_identity_and_revalidation() {
    let r = region();
    let c = candidates(3);
    let hint = device_cache::expected_candidate(MAP.as_bytes(), &r, &c[0]).unwrap();
    assert!(device_cache::revalidate(MAP.as_bytes(), &r, &c[0], &hint).is_ok());
    assert_eq!(
        hint["identity"]["extraction"]["rule"],
        "typed-u32-physical-extraction-v1"
    );
    for field in [
        "source_identity",
        "typed_contract",
        "rules",
        "extraction",
        "toolchain",
        "device",
        "layout",
        "numeric",
        "abi",
        "artifact_hash",
        "schedule_version",
        "variant_index",
        "geometry",
        "options",
    ] {
        let mut forged = hint.clone();
        forged["identity"][field] = serde_json::json!("forged");
        assert!(!device_cache::validate(&forged, &hint), "{field}");
        assert_eq!(
            device_cache::revalidate(MAP.as_bytes(), &r, &c[0], &forged).err(),
            Some(Refusal::StaleOrForgedHint),
            "{field}"
        );
    }
    let mut forged = hint.clone();
    forged["key"] = serde_json::json!("0000000000000000");
    assert_eq!(
        device_cache::revalidate(MAP.as_bytes(), &r, &c[0], &forged).err(),
        Some(Refusal::StaleOrForgedHint)
    );
    forged = hint.clone();
    forged["geometry"]["lanes"] = serde_json::json!(128);
    assert!(device_cache::revalidate(MAP.as_bytes(), &r, &c[0], &forged).is_err());
    let altered = MAP.replace("wrapping_add(xs[i],5u32)", "wrapping_add(xs[i],6u32)");
    assert!(device_cache::revalidate(altered.as_bytes(), &r, &c[0], &hint).is_err());
    // A self-consistent key forged around an old contract must still fail
    // fresh source admission, even when every serialized field matches itself.
    let stale_contract = device_cache::expected_candidate(altered.as_bytes(), &r, &c[0]).unwrap();
    assert!(device_cache::revalidate(altered.as_bytes(), &r, &c[0], &stale_contract).is_err());
    assert!(device_cache::revalidate(b"fn broken", &r, &c[0], &hint).is_err());
    eprintln!("cache={hint}");
}

#[test]
fn schedule_tuning_cache_disk_miss_hit_corrupt_and_limits() {
    let root = std::env::temp_dir().join(format!("invar-m406-cache-{}", std::process::id()));
    std::fs::create_dir(&root).unwrap();
    let c = candidates(3);
    let hint = device_cache::expected_candidate(MAP.as_bytes(), &region(), &c[0]).unwrap();
    assert_eq!(
        device_cache::load_in(&root, &hint),
        device_cache::HintState::Miss
    );
    let path = root.join(format!("cuda-hint-{}.json", hint["key"].as_str().unwrap()));
    std::fs::write(&path, serde_json::to_vec(&hint).unwrap()).unwrap();
    assert_eq!(
        device_cache::load_in(&root, &hint),
        device_cache::HintState::Hit
    );
    for bytes in [b"{".to_vec(), vec![b' '; 32 * 1024 + 1]] {
        std::fs::write(&path, bytes).unwrap();
        assert_eq!(
            device_cache::load_in(&root, &hint),
            device_cache::HintState::Invalid
        );
    }
    let c2 = candidates(4);
    let new = device_cache::expected_candidate(MAP.as_bytes(), &region(), &c2[0]).unwrap();
    assert_ne!(hint["key"], new["key"]);
    assert_eq!(
        device_cache::load_in(&root, &new),
        device_cache::HintState::Miss
    );
    std::fs::remove_file(path).unwrap();
    std::fs::remove_dir(root).unwrap();
}

#[test]
fn schedule_tuning_source_semantics_lifetimes_and_filename_independence() {
    let checked = check::check(front::parse(MAP.as_bytes()).unwrap(), true).unwrap();
    let reference = eval::observe(&checked, 1_000_000);
    assert_eq!(reference.result, Ok(eval::Value::I32(0)));
    for workers in [1, 2, 4] {
        let run = schedule::run_source(MAP.as_bytes(), &lower(MAP), 1_000_000, workers).unwrap();
        assert_eq!(run.observation.result, reference.result);
        assert_eq!(run.observation.cleanup_trace, reference.cleanup_trace);
        assert_eq!(run.observation.steps_used, reference.steps_used);
    }
    // The interface has no filename/benchmark identifier; the real CLI must
    // likewise recognize arbitrary names and reject source errors in all modes.
    let root = std::env::temp_dir().join(format!("invar-m406-source-{}", std::process::id()));
    std::fs::create_dir(&root).unwrap();
    for name in ["benchmark-map.inv", "unrelated.inv"] {
        let path = root.join(name);
        std::fs::write(&path, MAP).unwrap();
        for engine in ["reference", "cpu-parallel", "auto"] {
            let mut command = std::process::Command::new(env!("CARGO_BIN_EXE_invar"));
            command
                .arg("run")
                .arg(&path)
                .args(["--engine", engine, "--format", "json"]);
            if engine == "cpu-parallel" {
                command.args(["--workers", "2"]);
            }
            let out = command.output().unwrap();
            assert!(
                out.status.success(),
                "{engine}: {}",
                String::from_utf8_lossy(&out.stdout)
            );
        }
        let explanation = std::process::Command::new(env!("CARGO_BIN_EXE_invar"))
            .arg("explain")
            .arg(&path)
            .args(["--engine", "auto", "--format", "json"])
            .output()
            .unwrap();
        assert!(explanation.status.success());
        let document: serde_json::Value = serde_json::from_slice(&explanation.stdout).unwrap();
        assert_eq!(document["device_schema"], "invar-explain-device-v5");
        let schema: serde_json::Value = serde_json::from_str(include_str!(
            "../schema/invar-explain-device-v5.schema.json"
        ))
        .unwrap();
        assert!(
            jsonschema::is_valid(&schema, &document["device_execution"]),
            "{document}"
        );
        std::fs::write(&path, "fn main() -> i32 { return missing; }").unwrap();
        for engine in [
            "reference",
            "native-serial",
            "cpu-parallel",
            "auto",
            "cuda",
            "metal",
        ] {
            let mut command = std::process::Command::new(env!("CARGO_BIN_EXE_invar"));
            command
                .arg("run")
                .arg(&path)
                .args(["--engine", engine, "--format", "json"]);
            if engine == "cpu-parallel" {
                command.args(["--workers", "2"]);
            }
            let out = command.output().unwrap();
            assert!(!out.status.success());
            assert!(!String::from_utf8_lossy(&out.stdout).contains("UNAVAILABLE"));
        }
        std::fs::remove_file(path).unwrap();
    }
    std::fs::remove_dir(root).unwrap();
}

#[cfg(any(feature = "cuda", all(feature = "metal", target_os = "macos")))]
fn hardware_case(caps: Capabilities) -> (admission::Region, Candidate, Vec<u32>, Vec<Vec<u32>>) {
    let source = MAP.replace("[0u32,4294967295u32,7u32]", "filled_u32(100000u32,1u32)");
    let program = lower(&source);
    let region = admission::analyze(source.as_bytes(), &program)
        .unwrap()
        .regions()[0]
        .clone();
    let mut data = (0..100_000u32)
        .map(|n| n.wrapping_mul(1664525).wrapping_add(1013904223))
        .collect::<Vec<_>>();
    data[0] = 0;
    data[1] = u32::MAX;
    let expected = match region.execution() {
        invar::plan::Execution::Map(map) => vec![map.plan.run_live(&[&data], data.len()).unwrap()],
        _ => panic!("independent map"),
    };
    let candidate = enumerate(&region, data.len() as u32, caps, [variant()])
        .unwrap()
        .remove(0);
    (region, candidate, data, expected)
}

#[cfg(all(feature = "metal", target_os = "macos"))]
#[test]
#[ignore = "requires real Apple7+ Metal; deterministic cost model is not calibration"]
fn metal_gpu_resident_dispatch_and_full_output() {
    use invar::schedule::tuning_resident::{self, Output};
    let mut device = invar_metal::MetalDevice::open_runtime_source()
        .expect("explicit Metal qualification requires a device");
    let (r, c, data, expected) = hardware_case(device.capabilities());
    let (_, host) = select_map(c.plan(), Residency::HOST, model(), Control::Auto, 0).unwrap();
    assert_eq!(
        decide(Control::Auto, Some(&host), 0, 0, true, true)
            .unwrap()
            .engine,
        Engine::Cpu
    );
    for control in [Control::Auto, Control::Serial, Control::Cpu, Control::Gpu] {
        let plan = enumerate(&r, data.len() as u32, device.capabilities(), [variant()])
            .unwrap()
            .remove(0)
            .into_plan();
        let map = device.prepare_resident_map(plan, &[&data]).unwrap();
        assert!(map.read().is_err(), "uncompleted output must be private");
        let run = tuning_resident::metal(&r, &mut device, map, model(), control).unwrap();
        let output = match run.output {
            Output::Resident(map) => {
                assert_eq!(run.cost.transfer_bytes, 0);
                assert_eq!(run.cost.completion, Completion::Completed);
                assert_eq!(run.decision.engine, Engine::Gpu);
                map.read().unwrap()
            }
            Output::Host(values) => {
                assert!(matches!(control, Control::Serial | Control::Cpu));
                // The record is the resident-API selection basis: no device
                // work was submitted for a host result.
                assert_eq!(run.cost.completion, Completion::NotSubmitted);
                assert_eq!(run.kernel_ms, None);
                values
            }
        };
        assert_eq!(
            output, expected,
            "{control:?}, full output including wrapping extremes"
        );
        eprintln!(
            "Metal name={} control={control:?} decision={:?} cost={}",
            device.name(),
            run.decision,
            run.cost.to_json()
        );
    }
    let empty = enumerate(&r, 0, device.capabilities(), [variant()])
        .unwrap()
        .remove(0)
        .into_plan();
    let empty_run = device.run_map(&empty, &[]).unwrap();
    assert!(empty_run.values.is_empty());
    assert_eq!(
        (
            empty_run.transfer_bytes,
            empty_run.kernel_ms,
            empty_run.host_to_host_ms
        ),
        (0, 0.0, 0.0)
    );
    assert!(!empty_run.cache_hit);
    // Every admitted geometry must preserve all words, not just an exit code.
    for lanes in [64, 128, 256] {
        let s = Schedule::m3(Geometry::new(100_000u32.div_ceil(lanes), lanes));
        let plan = enumerate(&r, 100_000, device.capabilities(), [s])
            .unwrap()
            .remove(0)
            .into_plan();
        let mut map = device.prepare_resident_map(plan, &[&data]).unwrap();
        device.run_resident_map(&mut map).unwrap();
        assert_eq!(map.read().unwrap(), expected);
    }
}

#[cfg(feature = "cuda")]
#[test]
#[ignore = "requires real CUDA GPU and compatible NVRTC; missing device fails qualification"]
fn gpu_resident_dispatch_and_full_output() {
    use invar::schedule::tuning_resident::{self, Output};
    let compiler = std::path::Path::new(env!("CARGO_BIN_EXE_invar-nvrtc"));
    let mut device = invar_cuda::cuda::CudaDevice::open()
        .expect("explicit CUDA qualification requires a device");
    let (r, c, data, expected) = hardware_case(device.capabilities());
    assert_eq!(
        select_map(c.plan(), Residency::HOST, model(), Control::Auto, 0)
            .unwrap()
            .0
            .engine,
        Engine::Cpu
    );
    let unread = enumerate(&r, data.len() as u32, device.capabilities(), [variant()])
        .unwrap()
        .remove(0)
        .into_plan();
    let unread = device
        .prepare_resident_map(unread, &[&data], compiler)
        .unwrap();
    assert!(matches!(
        device.read_resident_map(unread),
        Err(invar_cuda::Error::StaleBorrow)
    ));
    for control in [Control::Auto, Control::Serial, Control::Cpu, Control::Gpu] {
        let plan = enumerate(&r, data.len() as u32, device.capabilities(), [variant()])
            .unwrap()
            .remove(0)
            .into_plan();
        let map = device
            .prepare_resident_map(plan, &[&data], compiler)
            .unwrap();
        assert!(!map.is_completed());
        let run = tuning_resident::cuda(&r, &mut device, map, model(), control).unwrap();
        let output = match run.output {
            Output::Resident(map) => {
                assert_eq!(run.cost.transfer_bytes, 0);
                assert_eq!(run.cost.completion, Completion::Completed);
                assert_eq!(run.decision.engine, Engine::Gpu);
                device.read_resident_map(map).unwrap().1
            }
            Output::Host(values) => {
                assert!(matches!(control, Control::Serial | Control::Cpu));
                // The record is the resident-API selection basis: no device
                // work was submitted for a host result.
                assert_eq!(run.cost.completion, Completion::NotSubmitted);
                assert_eq!(run.kernel_ms, None);
                values
            }
        };
        assert_eq!(output, expected);
        eprintln!(
            "CUDA control={control:?} decision={:?} cost={}",
            run.decision,
            run.cost.to_json()
        );
    }
    for lanes in [64, 128, 256] {
        let s = Schedule::m3(Geometry::new(100_000u32.div_ceil(lanes), lanes));
        let plan = enumerate(&r, 100_000, device.capabilities(), [s])
            .unwrap()
            .remove(0)
            .into_plan();
        let map = device
            .prepare_resident_map(plan, &[&data], compiler)
            .unwrap();
        let (map, _) = device.run_resident_map(map).unwrap();
        assert_eq!(device.read_resident_map(map).unwrap().1, expected);
    }
}

#[cfg(all(feature = "metal", target_os = "macos"))]
#[test]
#[ignore = "requires Metal; injected failures must preserve source publication"]
fn metal_gpu_failure_paths_preserve_publication() {
    use invar::schedule::metal::{self, Fault};
    let (r, _, _, _) = hardware_case(
        invar_metal::MetalDevice::open_runtime_source()
            .unwrap()
            .capabilities(),
    );
    assert!(r.guard().is_some());
    let source = MAP.replace("[0u32,4294967295u32,7u32]", "filled_u32(100000u32,1u32)");
    let checked = check::check(front::parse(source.as_bytes()).unwrap(), true).unwrap();
    let reference = eval::observe(&checked, 10_000_000);
    let gpu = DeviceCostModel {
        cpu_bytes_per_ms: 1.0,
        device_bytes_per_ms: 1e12,
        device_fixed_ms: 0.0,
    };
    for fault in [
        Fault::OutOfMemory,
        Fault::LaunchFailure,
        Fault::PartialCompletion,
    ] {
        let run = metal::run_auto_source_with_fault(
            source.as_bytes(),
            &lower(&source),
            10_000_000,
            metal::Kernels::RuntimeSource,
            gpu,
            fault,
        )
        .unwrap();
        if fault == Fault::PartialCompletion {
            assert!(matches!(
                run.observation.result,
                Err(eval::EvalError::Tool {
                    code: "METAL_PENDING",
                    ..
                })
            ));
            assert!(run.executed.is_empty());
        } else {
            assert_eq!(run.observation.result, reference.result);
            assert_eq!(run.observation.steps_used, reference.steps_used);
            assert_eq!(run.observation.cleanup_trace, reference.cleanup_trace);
            assert_eq!(run.choices[0].reason, "metal-failed-before-publication");
        }
    }
}

#[cfg(feature = "cuda")]
#[test]
#[ignore = "requires CUDA; injected failures must preserve source publication"]
fn gpu_failure_paths_preserve_publication() {
    use invar::schedule::cuda::{self, Fault};
    let source = MAP.replace("[0u32,4294967295u32,7u32]", "filled_u32(100000u32,1u32)");
    let checked = check::check(front::parse(source.as_bytes()).unwrap(), true).unwrap();
    let reference = eval::observe(&checked, 10_000_000);
    let compiler = std::path::Path::new(env!("CARGO_BIN_EXE_invar-nvrtc"));
    let gpu = DeviceCostModel {
        cpu_bytes_per_ms: 1.0,
        device_bytes_per_ms: 1e12,
        device_fixed_ms: 0.0,
    };
    invar_cuda::cuda::CudaDevice::open().expect("CUDA qualification must fail without hardware");
    for fault in [
        Fault::OutOfMemory,
        Fault::LaunchFailure,
        Fault::PartialCompletion,
    ] {
        let run = cuda::run_auto_source_with_fault(
            source.as_bytes(),
            &lower(&source),
            10_000_000,
            compiler,
            gpu,
            fault,
        )
        .unwrap();
        if fault == Fault::PartialCompletion {
            assert!(matches!(
                run.observation.result,
                Err(eval::EvalError::Tool {
                    code: "CUDA_PENDING",
                    ..
                })
            ));
            assert!(run.executed.is_empty());
        } else {
            assert_eq!(run.observation.result, reference.result);
            assert_eq!(run.observation.steps_used, reference.steps_used);
            assert_eq!(run.observation.cleanup_trace, reference.cleanup_trace);
            assert_eq!(run.choices[0].reason, "cuda-failed-before-publication");
        }
    }
}

#[test]
fn schedule_tuning_live_guard_and_source_trap_are_preserved() {
    let source = MAP.replace("filled_u32(len(xs),0u32)", "filled_u32(2u32,0u32)");
    let checked = check::check(front::parse(source.as_bytes()).unwrap(), true).unwrap();
    let reference = eval::observe(&checked, 1_000_000);
    assert!(reference.result.is_err());
    let run = schedule::run_source(source.as_bytes(), &lower(&source), 1_000_000, 4).unwrap();
    assert_eq!(run.observation.result, reference.result);
    assert_eq!(run.observation.steps_used, reference.steps_used);
    assert_eq!(run.observation.cleanup_trace, reference.cleanup_trace);
    assert!(run.executed.is_empty());
}

#[test]
fn schedule_tuning_reproducible_selection_receipts_and_keys() {
    let r = region();
    let caps = Capabilities::cuda_12_9_sm89();
    let schedules = [
        variant(),
        Schedule {
            cpu_threshold: 200_000,
            ..variant()
        },
    ];
    let cs = enumerate(&r, 100_000, caps, schedules).unwrap();
    let (first, costs) =
        select(&cs, Residency::DEVICE, model(), Control::Auto, true, true).unwrap();
    for _ in 0..3 {
        assert_eq!(
            select(&cs, Residency::DEVICE, model(), Control::Auto, true, true)
                .unwrap()
                .0,
            first
        );
    }
    assert_eq!(first.variant, Some(0));
    assert_eq!(costs.len(), 2);
    let limits = Budget {
        compile_ms: 1000,
        tuning_ms: 10_000,
        temporary_bytes: 1024,
    };
    let record = receipt(&cs, limits);
    assert_eq!(record["candidates"].as_array().unwrap().len(), 2);
    assert_eq!(record["max_variants"], 32);
    let keys = cs
        .iter()
        .map(|c| device_cache::expected_candidate(MAP.as_bytes(), &r, c).unwrap())
        .collect::<Vec<_>>();
    assert_ne!(keys[0]["key"], keys[1]["key"]);
    let metal = enumerate(
        &r,
        100_000,
        Capabilities::Metal {
            apple_family: 7,
            msl_major: 3,
            msl_minor: 1,
        },
        [variant()],
    )
    .unwrap();
    assert_ne!(
        device_cache::expected_candidate(MAP.as_bytes(), &r, &metal[0]).unwrap()["key"],
        keys[0]["key"]
    );
    eprintln!("enumeration={record}\nselected={first:?}\nkeys={keys:?}");
}

#[test]
fn schedule_tuning_collective_cost_facts() {
    let pair = pair_reduction_support::source(&[2, 3, 4], &[1, 2, 3], [1, 0]);
    // The pair case crosses the 64-block Metal chunk boundary without allocating inputs.
    for (source, length, output_bytes, cuda_temporary, metal_temporary, metal_launches) in [
        (
            "fn main() -> i32 { let xs: Array<u32> = [1u32,2u32,3u32]; var s:u32=0u32; for i in 0u32 .. len(xs) { s=wrapping_add(s,xs[i]); } return 0; }",
            100_000,
            4,
            200,
            232,
            3,
        ),
        (
            "fn main() -> i32 { let xs: Array<u32> = [1u32,2u32,3u32]; var s:u32=0u32; var out: Array<u32> = filled_u32(len(xs),0u32); for i in 0u32 .. len(xs) { s=wrapping_add(wrapping_mul(3u32,s),xs[i]); out[i]=s; } return 0; }",
            100_000,
            400_004,
            400,
            432,
            3,
        ),
        (pair.as_str(), 262_145, 8, 1_040, 1_084, 6),
    ] {
        let program = lower(source);
        let report = admission::analyze(source.as_bytes(), &program).unwrap();
        for caps in [
            Capabilities::cuda_12_9_sm89(),
            Capabilities::Metal {
                apple_family: 7,
                msl_major: 3,
                msl_minor: 1,
            },
        ] {
            // Collectives have a fixed, verified hierarchical geometry. A
            // generic map geometry must refuse rather than silently retile it.
            assert_eq!(
                enumerate(&report.regions()[0], length, caps, [variant()]).err(),
                Some(Refusal::Admission)
            );
            let mut ir = invar::device_ir::from_region(&report.regions()[0], length).unwrap();
            if matches!(caps, Capabilities::Metal { .. }) {
                invar::device_ir::plan_metal(&mut ir).unwrap();
            }
            let c = enumerate(
                &report.regions()[0],
                length,
                caps,
                [Schedule::m3(ir.geometry)],
            )
            .unwrap();
            assert!(c[0].plan().ir().collective.is_some());
            let facts = c[0].plan().cost_facts().unwrap();
            let record = CostRecord::from_plan(c[0].plan(), Residency::HOST, model()).unwrap();
            assert_eq!(record.launches, facts.launches);
            assert_eq!(record.temporary_bytes, facts.temporary_bytes);
            let metal = matches!(caps, Capabilities::Metal { .. });
            assert_eq!(
                facts.temporary_bytes,
                if metal {
                    metal_temporary
                } else {
                    cuda_temporary
                }
            );
            assert_eq!(facts.launches, if metal { metal_launches } else { 3 });
            assert_eq!(
                record.transfer_bytes,
                facts.input_bytes + facts.output_bytes
            );
            assert_eq!(record.output_bytes, output_bytes);
            assert_eq!(
                select_map(c[0].plan(), Residency::DEVICE, model(), Control::Auto, 0).err(),
                Some(Refusal::UnsupportedSchedule)
            );
        }
    }
}
