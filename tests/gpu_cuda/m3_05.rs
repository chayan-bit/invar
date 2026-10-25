//! Live guarded CUDA integration: forced and auto runs, the module cache, hints, faults and the CLI.
//! The CPU graph, cache and explain tests stay in `tests/device/issue_m3_05_guarded_gpu_integration.rs`.
#[path = "../support/scratch.rs"]
mod scratch;
use invar::{admission, check, device_cache, device_ir, eval, front, mir, plan::DeviceCostModel};
use invar_device::{Capabilities, Provenance};
use std::process::Command;

#[path = "../support/m3_05_programs.rs"]
mod m3_05_programs;
use m3_05_programs::{PAIR, REVERSED_ZIP};

const MAP: &str = "fn main() -> i32 { let xs: Array<u32> = [1u32,2u32,3u32]; \
    var ys: Array<u32> = filled_u32(len(xs),0u32); \
    for i in 0u32 .. len(xs) { ys[i] = wrapping_add(xs[i],5u32); } \
    if ys[2u32] == 8u32 { return 0; } return 1; }";

const SUM: &str = "fn main() -> i32 { let xs: Array<u32> = [1u32,2u32,3u32]; \
    var acc: u32 = 7u32; for i in 0u32 .. len(xs) { acc = wrapping_add(acc,xs[i]); } \
    if acc == 13u32 { return 0; } return 1; }";

const PRODUCT: &str = "fn main() -> i32 { let xs: Array<u32> = [2u32,3u32,5u32]; \
    var acc: u32 = 3u32; for i in 0u32 .. len(xs) { acc = wrapping_mul(acc,xs[i]); } \
    if acc == 90u32 { return 0; } return 1; }";

const INCLUSIVE: &str = "fn main() -> i32 { let aa: Array<u32> = [2u32,3u32,5u32]; \
    let bb: Array<u32> = [7u32,11u32,13u32]; var state: u32 = 17u32; \
    var out: Array<u32> = filled_u32(len(aa),0u32); \
    for i in 0u32 .. len(aa) { state = wrapping_add(wrapping_mul(aa[i],state),bb[i]); out[i] = state; } \
    if out[2u32] == 683u32 { return 0; } return 1; }";

const EXCLUSIVE: &str = "fn main() -> i32 { let aa: Array<u32> = [2u32,3u32,5u32]; \
    let bb: Array<u32> = [7u32,11u32,13u32]; var state: u32 = 17u32; \
    var out: Array<u32> = filled_u32(len(aa),0u32); \
    for i in 0u32 .. len(aa) { out[i] = state; state = wrapping_add(wrapping_mul(aa[i],state),bb[i]); } \
    if out[0u32] == 17u32 { return 0; } return 1; }";

const FINAL: &str = "fn main() -> i32 { let aa: Array<u32> = [2u32,3u32,5u32]; \
    let bb: Array<u32> = [7u32,11u32,13u32]; var state: u32 = 17u32; \
    for i in 0u32 .. len(aa) { state = wrapping_add(wrapping_mul(aa[i],state),bb[i]); } \
    if state == 683u32 { return 0; } return 1; }";

const CHAIN: &str = "fn main() -> i32 { let xs: Array<u32> = [1u32,2u32,3u32]; \
    var a: Array<u32> = filled_u32(len(xs),0u32); \
    for i in 0u32 .. len(xs) { a[i] = wrapping_add(xs[i],3u32); } \
    let first: Array<u32> = move a; var b: Array<u32> = filled_u32(len(first),0u32); \
    for j in 0u32 .. len(first) { b[j] = wrapping_mul(first[j],7u32); } \
    let second: Array<u32> = move b; var c: Array<u32> = filled_u32(len(second),0u32); \
    for k in 0u32 .. len(second) { c[k] = wrapping_add(second[k],11u32); } \
    if c[0u32] != 39u32 { return 1; } if c[2u32] != 53u32 { return 1; } return 0; }";

const ZIP_DAG: &str = "fn main() -> i32 { let xs: Array<u32> = [1u32,2u32,3u32]; \
    let ys: Array<u32> = [4u32,5u32,6u32]; let zs: Array<u32> = [7u32,8u32,9u32]; \
    var a: Array<u32> = filled_u32(len(xs),0u32); \
    for i in 0u32 .. len(xs) { a[i] = wrapping_add(xs[i],ys[i]); } \
    let first: Array<u32> = move a; var b: Array<u32> = filled_u32(len(first),0u32); \
    for j in 0u32 .. len(first) { b[j] = wrapping_mul(first[j],zs[j]); } \
    if b[0u32] != 35u32 { return 1; } if b[2u32] != 81u32 { return 1; } return 0; }";

#[test]
fn issue_m3_05_guarded_gpu_integration_full_corpus_has_device_plans() {
    for source in [
        MAP,
        PAIR,
        REVERSED_ZIP,
        CHAIN,
        ZIP_DAG,
        SUM,
        PRODUCT,
        INCLUSIVE,
        EXCLUSIVE,
        FINAL,
    ] {
        let checked = check::check(front::parse(source.as_bytes()).unwrap(), true).unwrap();
        let program = mir::lower(&checked).unwrap();
        let report = admission::analyze(source.as_bytes(), &program).unwrap();
        let region = report
            .regions()
            .iter()
            .find(|region| device_ir::from_region(region, 3).is_ok())
            .unwrap();
        let ir = device_ir::from_region(region, 3).unwrap();
        let origin = Provenance {
            source_identity: ir.source_identity,
            graph_revision: ir.graph_revision,
        };
        invar_device::verify(ir, origin, Capabilities::cuda_12_9_sm89()).unwrap();
        assert_eq!(
            eval::observe(&checked, 1_000_000).result,
            Ok(eval::Value::I32(0))
        );
    }
}

#[test]
#[ignore = "requires a compatible CUDA GPU and NVRTC 12.4+"]
fn issue_m3_05_gpu_forced_corpus_and_warm_module_cache() {
    use invar::schedule;
    use invar_cuda::cuda::CudaDevice;

    let compiler = std::path::Path::new(env!("CARGO_BIN_EXE_invar-nvrtc"));
    let mut device = CudaDevice::open().expect("CUDA_UNAVAILABLE: requires a compatible CUDA GPU");
    eprintln!(
        "device={} capabilities={:?}",
        device.name(),
        device.capabilities()
    );
    for (name, source) in [
        ("map", MAP),
        ("pair", PAIR),
        ("reversed-zip", REVERSED_ZIP),
        ("chain", CHAIN),
        ("zip-dag", ZIP_DAG),
        ("sum", SUM),
        ("product", PRODUCT),
        ("inclusive", INCLUSIVE),
        ("exclusive", EXCLUSIVE),
        ("final", FINAL),
    ] {
        let checked = check::check(front::parse(source.as_bytes()).unwrap(), true).unwrap();
        let program = mir::lower(&checked).unwrap();
        let reference = eval::observe(&checked, 1_000_000);
        let cpu = schedule::run_source(source.as_bytes(), &program, 1_000_000, 4).unwrap();
        let cold = schedule::cuda::run_source_with_device(
            source.as_bytes(),
            &program,
            1_000_000,
            &mut device,
            compiler,
        )
        .unwrap();
        assert_eq!(cpu.observation.result, reference.result, "{name} CPU");
        assert_eq!(cold.observation.result, reference.result, "{name} CUDA");
        assert!(!cold.executed.is_empty(), "{name} had no CUDA publication");
        eprintln!(
            "case={name} regions={} cold_module_hits={} hint_hits={} invalid_hints={} result={:?}",
            cold.executed.len(),
            cold.cache_hits,
            cold.hint_hits,
            cold.hint_invalids,
            cold.observation.result
        );
        let warm = schedule::cuda::run_source_with_device(
            source.as_bytes(),
            &program,
            1_000_000,
            &mut device,
            compiler,
        )
        .unwrap();
        assert_eq!(
            warm.observation.result, reference.result,
            "{name} warm CUDA"
        );
        assert!(
            warm.cache_hits >= 1,
            "{name} did not reuse its compiled module"
        );
        assert!(
            warm.hint_hits >= 1,
            "{name} did not validate its persisted schedule hint"
        );
        eprintln!(
            "case={name} warm_module_hits={} warm_hint_hits={} result={:?}",
            warm.cache_hits, warm.hint_hits, warm.observation.result
        );
    }
}

#[test]
#[ignore = "requires the M2 probe corpus, a compatible CUDA GPU, and NVRTC 12.4+"]
fn issue_m3_05_gpu_full_m2_probe_corpus() {
    use invar::schedule::{self, cuda};
    use invar_cuda::cuda::CudaDevice;

    let directory = std::env::var_os("INVAR_M3_05_CORPUS_DIR")
        .expect("INVAR_M3_05_CORPUS_DIR must name the exported M2 probe corpus");
    let mut files = std::fs::read_dir(directory)
        .unwrap()
        .map(|entry| entry.unwrap().path())
        .filter(|path| {
            path.extension()
                .is_some_and(|extension| extension == "invar")
        })
        .collect::<Vec<_>>();
    files.sort();
    assert_eq!(files.len(), 252, "expected every M2 generated probe");
    let compiler = std::path::Path::new(env!("CARGO_BIN_EXE_invar-nvrtc"));
    let model = cuda::calibrate().expect("CUDA calibration must complete on real hardware");
    let mut device = CudaDevice::open().expect("CUDA_UNAVAILABLE: compatible device required");
    let mut forced = 0usize;
    let mut empty = 0usize;
    let mut ineligible = 0usize;
    for file in files {
        let source = std::fs::read(&file).unwrap();
        let checked = check::check(front::parse(&source).unwrap(), true).unwrap();
        let program = mir::lower(&checked).unwrap();
        let report = admission::analyze(&source, &program).unwrap();
        let eligible = report
            .regions()
            .iter()
            .any(|region| region.plan_kind() != invar::plan::PlanKind::Serial);
        let reference = eval::observe(&checked, 1_000_000);
        let cpu = schedule::run_source(&source, &program, 1_000_000, 4).unwrap();
        assert_eq!(cpu.observation.result, reference.result, "CPU {file:?}");
        assert_eq!(
            cpu.observation.steps_used, reference.steps_used,
            "CPU {file:?}"
        );
        assert_eq!(
            cpu.observation.cleanup_trace, reference.cleanup_trace,
            "CPU {file:?}"
        );
        let auto = cuda::run_auto_source_with_fault(
            &source,
            &program,
            1_000_000,
            compiler,
            model,
            cuda::Fault::None,
        )
        .unwrap();
        assert_eq!(auto.observation.result, reference.result, "auto {file:?}");
        assert_eq!(
            auto.observation.steps_used, reference.steps_used,
            "auto {file:?}"
        );
        assert_eq!(
            auto.observation.cleanup_trace, reference.cleanup_trace,
            "auto {file:?}"
        );
        match cuda::run_source_with_device(&source, &program, 1_000_000, &mut device, compiler) {
            Ok(run) => {
                assert!(
                    eligible && !run.executed.is_empty(),
                    "forced {file:?}: eligible={eligible} result={:?} steps={} executed={:?}",
                    run.observation.result,
                    run.observation.steps_used,
                    run.executed
                );
                assert_eq!(run.observation.result, reference.result, "CUDA {file:?}");
                assert_eq!(
                    run.observation.steps_used, reference.steps_used,
                    "CUDA {file:?}"
                );
                assert_eq!(
                    run.observation.cleanup_trace, reference.cleanup_trace,
                    "CUDA {file:?}"
                );
                forced += 1;
            }
            Err(error) if !eligible && error.code == "CUDA_UNAVAILABLE" => ineligible += 1,
            Err(error)
                if eligible
                    // Only zero-trip regions were reached, so none could publish.
                    && cpu.executed.is_empty()
                    && error.code == "CUDA_UNAVAILABLE"
                    && error.message == "no CUDA result was published" =>
            {
                empty += 1;
            }
            Err(error) => panic!("unexpected forced CUDA failure for {file:?}: {error:?}"),
        }
    }
    assert_eq!((forced, empty, ineligible), (120, 24, 108));
    eprintln!(
        "M2 corpus 252/252: CUDA={forced}, empty={empty}, ineligible={ineligible}; auto and CPU agree with reference"
    );
}

#[test]
#[ignore = "requires a compatible CUDA GPU and NVRTC 12.4+"]
fn issue_m3_05_gpu_forged_hint_cannot_replace_generated_kernel() {
    use invar_cuda::cuda::CudaDevice;

    let root = std::env::temp_dir().join(format!("invar-m3-05-forged-hint-{}", scratch::unique()));
    let cache = root.join("cache");
    std::fs::create_dir_all(&cache).unwrap();
    let source = root.join("map.inv");
    std::fs::write(&source, MAP).unwrap();
    let program =
        mir::lower(&check::check(front::parse(MAP.as_bytes()).unwrap(), true).unwrap()).unwrap();
    let report = admission::analyze(MAP.as_bytes(), &program).unwrap();
    let ir = device_ir::from_region(&report.regions()[0], 3).unwrap();
    let origin = Provenance {
        source_identity: ir.source_identity,
        graph_revision: ir.graph_revision,
    };
    let capabilities = CudaDevice::open()
        .expect("CUDA_UNAVAILABLE: compatible device required")
        .capabilities();
    let verified = invar_device::verify(ir, origin, capabilities).unwrap();
    let artifact = invar_device::emit_cuda(&verified).unwrap();
    let valid = device_cache::expected(MAP.as_bytes(), &report.regions()[0], &verified, &artifact);
    let mut forged = valid.clone();
    forged["identity"]["artifact_hash"] = "0000000000000000".into();
    forged["cuda_source"] = "#error injected CUDA body must never compile".into();
    let hint = cache.join(format!("cuda-hint-{}.json", valid["key"].as_str().unwrap()));
    std::fs::write(&hint, serde_json::to_vec(&forged).unwrap()).unwrap();
    let output = Command::new(env!("CARGO_BIN_EXE_invar"))
        .args([
            "explain",
            source.to_str().unwrap(),
            "--engine",
            "cuda",
            "--format",
            "json",
        ])
        .env("INVAR_NVRTC_HELPER", env!("CARGO_BIN_EXE_invar-nvrtc"))
        .env("INVAR_CACHE_DIR", &cache)
        .output()
        .unwrap();
    let document: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    assert!(output.status.success(), "{document}");
    assert_eq!(document["device_execution"]["result"], 0);
    assert_eq!(document["device_execution"]["schedule_hint_invalids"], 1);
    assert_eq!(
        serde_json::from_slice::<serde_json::Value>(&std::fs::read(hint).unwrap()).unwrap(),
        valid
    );
    std::fs::remove_dir_all(root).unwrap();
    eprintln!("forged hint rejected; fresh verified CUDA source compiled and repaired the hint");
}

#[test]
#[ignore = "requires a compatible CUDA GPU and NVRTC 12.4+"]
fn issue_m3_05_gpu_failure_prefix_and_quarantine() {
    use invar::{eval::EvalError, schedule::cuda};
    use invar_cuda::cuda::CudaDevice;

    let compiler = std::path::Path::new(env!("CARGO_BIN_EXE_invar-nvrtc"));
    let program =
        mir::lower(&check::check(front::parse(PAIR.as_bytes()).unwrap(), true).unwrap()).unwrap();
    let mut device = CudaDevice::open().expect("CUDA_UNAVAILABLE: requires a compatible CUDA GPU");
    for fault in [cuda::Fault::OutOfMemory, cuda::Fault::LaunchFailure] {
        let failed = cuda::run_source_with_device_fault(
            PAIR.as_bytes(),
            &program,
            1_000_000,
            &mut device,
            compiler,
            fault,
        )
        .unwrap();
        assert!(failed.executed.is_empty());
        assert!(matches!(
            failed.observation.result,
            Err(EvalError::Tool {
                code: "CUDA_UNAVAILABLE",
                ..
            })
        ));
        assert!(!device.is_poisoned());
        let recovered = cuda::run_source_with_device(
            PAIR.as_bytes(),
            &program,
            1_000_000,
            &mut device,
            compiler,
        )
        .unwrap();
        assert_eq!(recovered.observation.result, Ok(eval::Value::I32(0)));
        eprintln!("fault={fault:?} publication=none recovery=CUDA result=0");
    }
    let pending = cuda::run_source_with_device_fault(
        PAIR.as_bytes(),
        &program,
        1_000_000,
        &mut device,
        compiler,
        cuda::Fault::PartialCompletion,
    )
    .unwrap();
    assert!(pending.executed.is_empty());
    assert!(matches!(
        pending.observation.result,
        Err(EvalError::Tool {
            code: "CUDA_PENDING",
            ..
        })
    ));
    assert!(device.is_poisoned());
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
    while device.is_poisoned() {
        assert!(
            std::time::Instant::now() < deadline,
            "CUDA_PENDING: fence never completed"
        );
        device.reap();
        std::thread::yield_now();
    }
    // A fresh run after confirmed quiescence is new work, not a replay.
    let retry =
        cuda::run_source_with_device(PAIR.as_bytes(), &program, 1_000_000, &mut device, compiler)
            .unwrap();
    assert!(!retry.executed.is_empty());
    assert_eq!(retry.observation.result, Ok(eval::Value::I32(0)));
    eprintln!("fault=PartialCompletion publication=none quarantine=reaped replay=none");
}

#[test]
#[ignore = "requires a compatible CUDA GPU and NVRTC 12.4+"]
fn issue_m3_05_gpu_auto_fallback_only_before_submission() {
    use invar::{eval::EvalError, schedule::cuda};

    let compiler = std::path::Path::new(env!("CARGO_BIN_EXE_invar-nvrtc"));
    let program =
        mir::lower(&check::check(front::parse(PAIR.as_bytes()).unwrap(), true).unwrap()).unwrap();
    let choose_cuda = DeviceCostModel {
        cpu_bytes_per_ms: 1.0,
        device_bytes_per_ms: 1_000_000_000.0,
        device_fixed_ms: 0.0,
    };
    let fallback = cuda::run_auto_source_with_fault(
        PAIR.as_bytes(),
        &program,
        1_000_000,
        compiler,
        choose_cuda,
        cuda::Fault::OutOfMemory,
    )
    .unwrap();
    assert_eq!(fallback.observation.result, Ok(eval::Value::I32(0)));
    assert_eq!(fallback.choices[0].backend, "cpu-parallel");
    assert_eq!(fallback.choices[0].reason, "cuda-failed-before-publication");
    eprintln!("auto pre-submit OOM: fresh CPU output=0, no CUDA publication");
    let pending = cuda::run_auto_source_with_fault(
        PAIR.as_bytes(),
        &program,
        1_000_000,
        compiler,
        choose_cuda,
        cuda::Fault::PartialCompletion,
    )
    .unwrap();
    assert!(matches!(
        pending.observation.result,
        Err(EvalError::Tool {
            code: "CUDA_PENDING",
            ..
        })
    ));
    assert!(pending.executed.is_empty());
    eprintln!("auto partial completion: pending error, no CPU replay or publication");
}

#[test]
#[ignore = "requires a compatible CUDA GPU and NVRTC 12.4+"]
fn issue_m3_05_gpu_cli_forced_auto_and_explain_match() {
    let compiler = env!("CARGO_BIN_EXE_invar-nvrtc");
    let root = std::env::temp_dir().join(format!("invar-m3-05-cli-{}", scratch::unique()));
    std::fs::create_dir(&root).unwrap();
    let source = root.join("map.inv");
    std::fs::write(&source, MAP).unwrap();
    let bin = env!("CARGO_BIN_EXE_invar");
    let invoke = |action: &str, engine: &str, helper: &str| {
        let output = Command::new(bin)
            .args([
                action,
                source.to_str().unwrap(),
                "--engine",
                engine,
                "--format",
                "json",
            ])
            .env("INVAR_NVRTC_HELPER", helper)
            .env("INVAR_CACHE_DIR", root.join("cache"))
            .output()
            .unwrap();
        let document: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
        (output.status.code(), document)
    };
    for engine in ["cuda", "auto"] {
        let (status, run) = invoke("run", engine, compiler);
        assert_eq!(status, Some(0), "{engine}: {run}");
        let (status, explain) = invoke("explain", engine, compiler);
        assert_eq!(status, Some(0), "{engine}: {explain}");
        assert_eq!(
            explain["device_execution"]["result"],
            run["result"]["value"]
        );
        assert!(
            explain["device_execution"]["choices"]
                .as_array()
                .is_some_and(|choices| !choices.is_empty())
        );
        eprintln!("engine={engine} run={run} explain={explain}");
    }
    let (status, mismatch) = invoke("run", "cuda", "missing-nvrtc-helper.exe");
    assert_eq!(status, Some(2));
    assert_eq!(
        mismatch["diagnostics"][0]["code"],
        "DEVICE_TOOLCHAIN_MISMATCH"
    );
    std::fs::write(&source, "fn main() -> i32 { return 0; }").unwrap();
    let (status, ineligible) = invoke("run", "cuda", compiler);
    assert_eq!(status, Some(2));
    assert_eq!(ineligible["diagnostics"][0]["code"], "CUDA_UNAVAILABLE");
    std::fs::remove_dir_all(&root).unwrap();
}
