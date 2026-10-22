//! ABI12 follow-up: `device_host::cpu::CpuModel::prepare` refuses an ABI12
//! artifact explicitly, before creating a directory or invoking the compiler
//! or driver. Every other artifact keeps its prior refusal; an ABI5 control
//! shows the compiler recorder detects real invocations. One test only, so no
//! other test in this process creates model directories.
#[path = "support/executable.rs"]
mod executable;
use invar::general_net::{
    compiled,
    device_host::{Backend, cpu},
    provenance,
};
use std::path::{Path, PathBuf};

const VALID_OWNED: &str = "fn run(a: f32, n: u32) -> f32 { var xs: Array<f32> = filled_f32(n + 1u32, a); xs[0u32] = xs[0u32] * a + a; return xs[0u32]; } fn main() -> i32 { return 0; }";
const VALID_SCALAR: &str = "fn run(n: u32) -> i32 { if n < 2u32 { return 1; } return run(n - 1u32) + run(n - 2u32); } fn main() -> i32 { return run(5u32); }";

fn storage() -> compiled::OwnedLimits {
    compiled::OwnedLimits {
        elements: 64,
        allocations: 16,
        cleanup_events: 32,
    }
}

/// Model directories this process has created so far.
fn model_dirs() -> usize {
    let prefix = format!("invar-device-model-{}-", std::process::id());
    std::fs::read_dir(std::env::temp_dir())
        .map(|d| {
            d.filter_map(Result::ok)
                .filter(|e| e.file_name().to_string_lossy().starts_with(&prefix))
                .count()
        })
        .unwrap_or(0)
}

/// A compiler stand-in that records each invocation and fails.
fn recorder(dir: &Path) -> (PathBuf, PathBuf) {
    let calls = dir.join("calls");
    let script = dir.join("cxx");
    executable::write(
        &script,
        format!(
            "#!/bin/sh\necho \"$@\" >> '{}'\nexit 1\n",
            calls.to_str().unwrap().replace('\'', "'\\''")
        )
        .as_bytes(),
    )
    .unwrap();
    (script, calls)
}

fn prepare(model: &mut cpu::CpuModel, artifact: &compiled::Artifact) -> Result<(), String> {
    model.prepare(artifact, [8; 12], &[], [0; 6]).map(|_| ())
}

#[test]
fn issue_gnet_12_cpu_model_refuses_abi12_before_any_compiler_or_device_call() {
    let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR"))
        .join(format!("abi12-cpu-model-gate-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let (script, calls) = recorder(&dir);
    let l = provenance::Limits::default();

    // ABI12: explicit typed refusal; no directory and no compiler call.
    let abi12 = provenance::compile_resumable_owned_numeric(
        VALID_OWNED.as_bytes(),
        "run",
        compiled::Target::Cuda,
        l,
        storage(),
    )
    .unwrap();
    assert_eq!(abi12.artifact().abi_version(), 12);
    let before = model_dirs();
    let mut model = cpu::CpuModel::with_compiler(&script);
    // With the temporary directory absent, any directory creation would fail
    // with a different error, so the refusal precedes even that.
    let saved = std::env::var_os("TMPDIR");
    // SAFETY: this binary has a single test, so no other thread reads the
    // environment concurrently; the value is restored immediately.
    unsafe { std::env::set_var("TMPDIR", dir.join("absent")) };
    let refused = prepare(&mut model, abi12.artifact());
    match saved {
        // SAFETY: as above.
        Some(v) => unsafe { std::env::set_var("TMPDIR", v) },
        None => unsafe { std::env::remove_var("TMPDIR") },
    }
    assert_eq!(refused, Err(cpu::UNSUPPORTED_ABI12.to_owned()));
    assert_eq!(
        cpu::UNSUPPORTED_ABI12,
        "CPU device model: unsupported ABI 12 artifact"
    );
    assert_eq!(model_dirs(), before, "ABI12 created a model directory");
    assert!(!calls.exists(), "ABI12 invoked the compiler");
    // The model is not poisoned by the refusal; a poisoned one still says so.
    assert!(!model.poisoned());
    model.poison();
    assert_eq!(
        prepare(&mut model, abi12.artifact()),
        Err("poisoned CPU device model".to_owned())
    );

    // Unchanged boundaries for other artifacts, also without compiler calls.
    let mut model = cpu::CpuModel::with_compiler(&script);
    let abi3 = provenance::compile_resumable(
        VALID_SCALAR.as_bytes(),
        "run",
        compiled::Target::Cuda,
        l,
        storage(),
    )
    .unwrap();
    assert_eq!(abi3.artifact().abi_version(), 3);
    assert_eq!(
        prepare(&mut model, abi3.artifact()),
        Err("missing private layout".to_owned())
    );
    let abi11 = provenance::compile_resumable_owned_numeric(
        VALID_OWNED.as_bytes(),
        "run",
        compiled::Target::Cpu,
        l,
        storage(),
    )
    .unwrap();
    assert_eq!(abi11.artifact().abi_version(), 11);
    assert_eq!(
        prepare(&mut model, abi11.artifact()),
        Err("CPU model requires a device artifact".to_owned())
    );
    assert!(!calls.exists(), "a refused artifact invoked the compiler");
    assert_eq!(model_dirs(), before, "refused artifacts left directories");

    // Control: an ABI5 device artifact does reach the recorded compiler.
    let abi5 = provenance::compile_device_speculative(
        VALID_SCALAR.as_bytes(),
        "run",
        compiled::Target::Cuda,
        l,
        compiled::OwnedLimits {
            elements: 16,
            allocations: 8,
            cleanup_events: 16,
        },
        compiled::SpeculationLimits {
            slots: 2,
            fuel: 10000,
            transitions: 10000,
            trace_events: 1024,
            quantum: 32,
            lookahead: 128,
            scratch_bytes: 1024 * 1024,
        },
    )
    .unwrap();
    assert_eq!(abi5.artifact().abi_version(), 5);
    let refused = prepare(&mut model, abi5.artifact()).unwrap_err();
    assert!(refused.starts_with("CPU device model compile"), "{refused}");
    assert_eq!(
        std::fs::read_to_string(&calls).unwrap().lines().count(),
        1,
        "the recorder must see exactly the ABI5 control call"
    );
    // A failed compile keeps its evidence directory (prior behavior).
    if let Some(evidence) = refused.split("evidence ").nth(1) {
        let path = evidence.split(':').next().unwrap();
        let _ = std::fs::remove_dir_all(path);
    }
    let _ = std::fs::remove_dir_all(&dir);
}
