//! Owning-device tensor dispatch tests on CUDA; the ordinary CLI tests stay in `tests/a1_tensor_dispatch.rs`.
use invar::{check, front, mir, mir_eval};
use std::process::Command;

#[path = "../support/a1_tensor_common.rs"]
mod a1_tensor_common;
use a1_tensor_common::{DIRECT, SCALAR, TC, program};

use invar::{
    plan::DeviceCostModel,
    schedule::{
        PlanKind,
        cuda::{self, Fault},
    },
};
#[test]
#[ignore = "requires an authorized CUDA host and NVRTC helper"]
fn a1_tensor_cuda_completed_forms_refusals_and_auto_cost() {
    let compiler = std::path::Path::new(env!("CARGO_BIN_EXE_invar-nvrtc"));
    let mut device = invar_cuda::cuda::CudaDevice::open().expect("CUDA hardware required");
    println!(
        "device={} capabilities={:?}",
        device.name(),
        device.capabilities()
    );
    for source in [SCALAR, DIRECT, TC] {
        let p = program(source);
        let serial = mir_eval::observe(&p, 1_000_000);
        let got =
            cuda::run_source_with_device(source.as_bytes(), &p, 1_000_000, &mut device, compiler)
                .unwrap();
        assert_eq!(got.observation.result, serial.result);
        assert_eq!(got.observation.steps_used, serial.steps_used);
        assert_eq!(got.observation.cleanup_trace, serial.cleanup_trace);
        assert_eq!(
            got.executed
                .iter()
                .filter(|e| e.kind == PlanKind::Tensor)
                .count(),
            1
        );
        for fault in [
            Fault::OutOfMemory,
            Fault::LaunchFailure,
            Fault::PartialCompletion,
        ] {
            let got = cuda::run_source_with_device_fault(
                source.as_bytes(),
                &p,
                1_000_000,
                &mut device,
                compiler,
                fault,
            )
            .unwrap();
            assert!(got.executed.is_empty());
            assert!(matches!(
                got.observation.result,
                Err(invar::eval::EvalError::Tool {
                    code: "CUDA_UNAVAILABLE",
                    ..
                })
            ));
            let fallback = cuda::run_auto_source_with_fault(
                source.as_bytes(),
                &p,
                1_000_000,
                compiler,
                DeviceCostModel {
                    cpu_bytes_per_ms: 1.0,
                    device_bytes_per_ms: 1e12,
                    device_fixed_ms: 0.0,
                },
                fault,
            )
            .unwrap();
            assert_eq!(fallback.observation.result, serial.result);
            assert_eq!(fallback.observation.steps_used, serial.steps_used);
            assert_eq!(fallback.observation.cleanup_trace, serial.cleanup_trace);
            assert!(!fallback.executed.iter().any(|e| e.kind == PlanKind::Tensor));
        }
        for device in [false, true] {
            let model = if device {
                DeviceCostModel {
                    cpu_bytes_per_ms: 1.0,
                    device_bytes_per_ms: 1e12,
                    device_fixed_ms: 0.0,
                }
            } else {
                DeviceCostModel {
                    cpu_bytes_per_ms: 1e12,
                    device_bytes_per_ms: 1.0,
                    device_fixed_ms: 10.0,
                }
            };
            let got = cuda::run_auto_source_with_fault(
                source.as_bytes(),
                &p,
                1_000_000,
                compiler,
                model,
                Fault::None,
            )
            .unwrap();
            assert_eq!(got.observation.result, serial.result);
            assert_eq!(got.observation.steps_used, serial.steps_used);
            assert_eq!(got.observation.cleanup_trace, serial.cleanup_trace);
            assert!(
                got.choices.iter().any(|c| c.function == "mat"
                    && c.backend == if device { "cuda" } else { "mir-serial" })
            );
        }
    }
}

#[test]
#[ignore = "requires CUDA; live guards must fall back before submission"]
fn a1_tensor_cuda_nonfinite_bounds_zero_trip_and_unreached() {
    let compiler = std::path::Path::new(env!("CARGO_BIN_EXE_invar-nvrtc"));
    for source in [SCALAR, DIRECT] {
        for bad in [
            source.replace("1.0,2.0", "(1.0/0.0),2.0"),
            source.replace(
                "1.0,2.0,-3.0,4.0,-5.0,6.0,7.0,8.0,-9.0,10.0,-11.0,12.0,13.0,14.0,15.0",
                "1.0",
            ),
            source.replace("3u32,5u32,7u32", "0u32,5u32,7u32"),
            source.replace(
                "let out=mat(a,b,3u32,5u32,7u32);",
                "if false{let unused=mat(a,b,3u32,5u32,7u32);}let out=filled_f32(21u32,0.0);",
            ),
        ] {
            let p = program(&bad);
            let serial = mir_eval::observe(&p, 1_000_000);
            let run = cuda::run_auto_source_with_fault(
                bad.as_bytes(),
                &p,
                1_000_000,
                compiler,
                DeviceCostModel {
                    cpu_bytes_per_ms: 1.0,
                    device_bytes_per_ms: 1e12,
                    device_fixed_ms: 0.0,
                },
                Fault::None,
            )
            .unwrap();
            assert_eq!(run.observation.result, serial.result);
            assert_eq!(run.observation.steps_used, serial.steps_used);
            assert_eq!(run.observation.cleanup_trace, serial.cleanup_trace);
            assert!(!run.executed.iter().any(|e| e.kind == PlanKind::Tensor));
        }
    }
}
#[test]
#[ignore = "requires an authorized CUDA host; executes ordinary invar run"]
fn a1_tensor_cuda_cli_routes_both_forms() {
    for path in [
        "tests/a1_tensor/scalar.inv",
        "tests/a1_tensor/direct.inv",
        "tests/a1_tensor/tc.inv",
    ] {
        for engine in ["reference", "cuda", "auto"] {
            let got = Command::new(env!("CARGO_BIN_EXE_invar"))
                .args(["run", path, "--engine", engine, "--format", "json"])
                .output()
                .unwrap();
            assert!(
                got.status.success(),
                "{}",
                String::from_utf8_lossy(&got.stdout)
            );
            let doc: serde_json::Value = serde_json::from_slice(&got.stdout).unwrap();
            assert_eq!(doc["result"]["value"], 42);
            println!(
                "run {path} --engine {engine}: {}",
                String::from_utf8_lossy(&got.stdout).trim()
            );
        }
    }
}

#[test]
#[ignore = "requires CUDA; TC enters through the family's private checker"]
fn a1_tensor_cuda_explicit_tc_completed_region() {
    let source = DIRECT.replace(
        "c[i*cols+j]+a[i*inner+k]*b[k*cols+j]",
        "tc_mul_add(a[i*inner+k],b[k*cols+j],c[i*cols+j])",
    );
    let contract =
        invar::contraction_source::Contract::from_source(source.as_bytes(), "mat").unwrap();
    assert_eq!(
        contract.policy(),
        invar::numeric_contract::NumericPolicy::TC_F16_F32
    );
    check::check(front::parse(source.as_bytes()).unwrap(), true).unwrap();
    let p = program(&source);
    let compiler = std::path::Path::new(env!("CARGO_BIN_EXE_invar-nvrtc"));
    let mut device = invar_cuda::cuda::CudaDevice::open().expect("CUDA hardware required");
    let caps = device.capabilities();
    println!("TC device={} capabilities={caps:?}", device.name());
    let emission = contract
        .emit_device(source.as_bytes(), [3, 5, 7], [15, 35], caps)
        .unwrap();
    assert!(emission.plan().is_tc());
    if let Some(dir) = std::env::var_os("INVAR_CUDA_EVIDENCE_DIR") {
        let dir = std::path::Path::new(&dir);
        let cubin = invar_cuda::cuda::compile_tc_cubin(emission.source(), caps).unwrap();
        std::fs::write(dir.join("tensor-tc.cubin"), cubin.image).unwrap();
        std::fs::write(dir.join("tensor-tc.cu"), emission.source()).unwrap();
    }
    let run = cuda::run_source_with_device(source.as_bytes(), &p, 1_000_000, &mut device, compiler)
        .unwrap();
    assert_eq!(run.observation.result, Ok(invar::eval::Value::I32(42)));
    assert_eq!(
        run.executed
            .iter()
            .filter(|e| e.kind == PlanKind::Tensor)
            .count(),
        1
    );
    let failed = cuda::run_source_with_device_fault(
        source.as_bytes(),
        &p,
        1_000_000,
        &mut device,
        compiler,
        Fault::PartialCompletion,
    )
    .unwrap();
    assert!(failed.executed.is_empty());
    assert!(matches!(
        failed.observation.result,
        Err(invar::eval::EvalError::Tool {
            code: "CUDA_UNAVAILABLE",
            ..
        })
    ));
}

#[test]
#[ignore = "requires CUDA; reports wall times without a performance gate"]
fn a1_tensor_cuda_timing_against_existing_host() {
    use std::time::Instant;
    let serial = mir_eval::observe(&program(SCALAR), 1_000_000);
    let compiler = std::path::Path::new(env!("CARGO_BIN_EXE_invar-nvrtc"));
    let mut device = invar_cuda::cuda::CudaDevice::open().unwrap();
    let old = invar::tensor::host::cuda::Cuda::open(SCALAR.as_bytes(), "mat", compiler).unwrap();
    let caps = old.capabilities();
    let mut old = invar_cuda::Runtime::new(old, 1);
    let mut adapter = Vec::new();
    let mut host = Vec::new();
    for sample in 0..4 {
        let start = Instant::now();
        let p = program(SCALAR);
        let run =
            cuda::run_source_with_device(SCALAR.as_bytes(), &p, 1_000_000, &mut device, compiler)
                .unwrap();
        let ms = start.elapsed().as_secs_f64() * 1000.0;
        assert_eq!(run.observation.result, serial.result);
        assert_eq!(run.observation.steps_used, serial.steps_used);
        assert_eq!(run.observation.cleanup_trace, serial.cleanup_trace);
        assert!(run.executed.iter().any(|e| e.kind == PlanKind::Tensor));
        if sample > 0 {
            adapter.push(ms);
        }
        let start = Instant::now();
        let run =
            invar::tensor::host::run_source(SCALAR.as_bytes(), 1_000_000, &mut old, caps).unwrap();
        let ms = start.elapsed().as_secs_f64() * 1000.0;
        assert_eq!(run.observation.result, serial.result);
        assert_eq!(run.observation.steps_used, serial.steps_used);
        assert_eq!(run.observation.cleanup_trace, serial.cleanup_trace);
        assert_eq!(run.completed, 1);
        if sample > 0 {
            host.push(ms);
        }
    }
    adapter.sort_by(f64::total_cmp);
    host.sort_by(f64::total_cmp);
    println!(
        "{}",
        serde_json::json!({"boundary":"host-to-host, already-open device, warm compiler; frontend checking, recognition and publication included", "dimensions":[3,5,7], "samples":3, "adapter_ms":adapter,"existing_host_ms":host,"adapter_median_ms":adapter[1],"existing_host_median_ms":host[1]})
    );
}
