#![cfg(feature = "cuda")]

use serde_json::Value;
use std::{
    fs,
    process::Command,
    time::{SystemTime, UNIX_EPOCH},
};

fn generated_map(cold: bool) {
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let dir = std::env::temp_dir().join(format!(
        "invar-bench-cuda-map-{}-{nonce}",
        std::process::id()
    ));
    fs::create_dir(&dir).unwrap();
    let report = dir.join("report.json");
    let mut command = Command::new(env!("CARGO_BIN_EXE_invar-bench"));
    command
        .args([
            "--manifest",
            "bench/manifest-v1.json",
            "--track",
            "generated",
            "--require-gpu",
            "--generated-backend",
            "cuda",
            "--generated-case",
            "map-n33",
            "--report",
        ])
        .arg(&report);
    if cold {
        command.arg("--cold-child");
    }
    let output = command.output().unwrap();
    let value: Value = serde_json::from_slice(&fs::read(&report).unwrap()).unwrap();
    assert_eq!(
        value["status"],
        if cold { "passed" } else { "partial" },
        "{value}\n{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert_eq!(output.status.success(), cold);
    let result = &value["generated_result"];
    assert_eq!(result["full_output_correct"], true);
    assert_eq!(result["launches"], 1);
    assert_eq!(result["workspace_bytes"], 0);
    assert_eq!(result["transfer_bytes"], 396);
    assert_eq!(result["samples"], if cold { 1 } else { 30 });
    assert_eq!(result["warmups"], if cold { 0 } else { 10 });
    assert_eq!(result["measured_output_sha256"].as_str().unwrap().len(), 64);
    assert_eq!(
        result["device_resident_boundary"],
        "cuda_events_around_verified_kernel_launches_v1"
    );
    assert_eq!(
        result["host_to_host_boundary"],
        "cuda_runner_after_compile_through_host_output_copy_v1"
    );
    if !cold {
        assert_eq!(value["cold_process"]["status"], "passed");
    }
    fs::remove_dir_all(dir).unwrap();
}

// A measurement sample: Debug timings are not meaningful, and a Debug cold child
// can exceed the production 10 s CUDA child watchdog, so it builds in Release only
// (`--ignored` would otherwise still run an ignored Debug test).
#[cfg(not(debug_assertions))]
#[test]
#[ignore = "requires a qualified CUDA device and the bounded invar-nvrtc helper"]
fn generated_cuda_map_warm_and_cold_samples() {
    generated_map(false);
}

#[test]
#[ignore = "requires a qualified CUDA device and the bounded invar-nvrtc helper"]
fn generated_cuda_map_one_cold_call() {
    generated_map(true);
}
