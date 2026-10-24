//! Owning-device tensor dispatch tests on Metal; the ordinary CLI tests stay in `tests/a1_tensor_dispatch.rs`.
use invar::mir_eval;
use std::process::Command;

#[path = "../support/a1_tensor_common.rs"]
mod a1_tensor_common;
use a1_tensor_common::{DIRECT, SCALAR, TC, program};

use invar::{
    plan::DeviceCostModel,
    schedule::{
        PlanKind,
        metal::{self, Fault},
    },
};
fn model(device: bool) -> DeviceCostModel {
    if device {
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
    }
}
fn parity(a: &invar::eval::Observation, b: &invar::eval::Observation) {
    assert_eq!(a.result, b.result);
    assert_eq!(a.steps_used, b.steps_used);
    assert_eq!(a.cleanup_trace, b.cleanup_trace);
}
#[test]
#[ignore = "requires an Apple GPU and the pinned strict-f32 D1 probe"]
fn a1_tensor_metal_completed_forms_cost_and_failure_paths() {
    let mut device =
        invar_metal::MetalDevice::open_runtime_source().expect("Metal hardware required");
    for source in [SCALAR, DIRECT, TC] {
        let p = program(source);
        let serial = mir_eval::observe(&p, 1_000_000);
        let run =
            metal::run_source_with_device(source.as_bytes(), &p, 1_000_000, &mut device).unwrap();
        parity(&serial, &run.observation);
        assert_eq!(
            run.executed
                .iter()
                .filter(|e| e.kind == PlanKind::Tensor)
                .count(),
            1
        );
        for prefer_device in [false, true] {
            let run = metal::run_auto_source_with_fault(
                source.as_bytes(),
                &p,
                1_000_000,
                metal::Kernels::RuntimeSource,
                model(prefer_device),
                Fault::None,
            )
            .unwrap();
            parity(&serial, &run.observation);
            assert!(run.choices.iter().any(|c| c.function == "mat"
                && c.backend == if prefer_device { "metal" } else { "mir-serial" }));
        }
        for fault in [
            Fault::OutOfMemory,
            Fault::LaunchFailure,
            Fault::PartialCompletion,
        ] {
            let run = metal::run_source_with_device_fault(
                source.as_bytes(),
                &p,
                1_000_000,
                &mut device,
                fault,
            )
            .unwrap();
            assert!(matches!(
                run.observation.result,
                Err(invar::eval::EvalError::Tool {
                    code: "METAL_UNAVAILABLE",
                    ..
                })
            ));
            assert!(run.executed.is_empty());
            let run = metal::run_auto_source_with_fault(
                source.as_bytes(),
                &p,
                1_000_000,
                metal::Kernels::RuntimeSource,
                model(true),
                fault,
            )
            .unwrap();
            parity(&serial, &run.observation);
            assert!(!run.executed.iter().any(|e| e.kind == PlanKind::Tensor));
        }
    }
}
#[test]
#[ignore = "requires an Apple GPU and the pinned strict-f32 D1 probe"]
fn a1_tensor_metal_strict_underflow_nonfinite_bounds_and_zero_trip() {
    for source in [SCALAR, DIRECT] {
        let tiny = format!("{:.149}", f32::from_bits(1));
        for bad in [
            source.replace("1.0,2.0", &format!("{tiny},2.0")),
            source.replace("1.0,2.0", "(1.0/0.0),2.0"),
            source.replace(
                "1.0,2.0,-3.0,4.0,-5.0,6.0,7.0,8.0,-9.0,10.0,-11.0,12.0,13.0,14.0,15.0",
                "1.0",
            ),
            source.replace("3u32,5u32,7u32", "0u32,5u32,7u32"),
        ] {
            let p = program(&bad);
            let serial = mir_eval::observe(&p, 1_000_000);
            let run = metal::run_auto_source_with_fault(
                bad.as_bytes(),
                &p,
                1_000_000,
                metal::Kernels::RuntimeSource,
                model(true),
                Fault::None,
            )
            .unwrap();
            parity(&serial, &run.observation);
            assert!(!run.executed.iter().any(|e| e.kind == PlanKind::Tensor));
            let forced =
                metal::run_source(bad.as_bytes(), &p, 1_000_000, metal::Kernels::RuntimeSource);
            match forced {
                Err(e) => assert_eq!(e.code, "METAL_UNAVAILABLE"),
                Ok(r) => {
                    assert!(r.executed.is_empty());
                    assert!(r.observation.result.is_err());
                }
            }
        }
    }
}
#[test]
#[ignore = "requires an Apple GPU; executes the ordinary invar CLI"]
fn a1_tensor_metal_cli_routes_both_forms() {
    for path in [
        "tests/a1_tensor/scalar.inv",
        "tests/a1_tensor/direct.inv",
        "tests/a1_tensor/tc.inv",
    ] {
        for engine in ["reference", "metal", "auto"] {
            let got = Command::new(env!("CARGO_BIN_EXE_invar"))
                .env("INVAR_METAL_RUNTIME_SOURCE", "1")
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
#[ignore = "requires an Apple GPU; reports host-to-host wall times, no performance gate"]
fn a1_tensor_metal_timing_against_existing_host() {
    use std::time::Instant;
    let p = program(SCALAR);
    let serial = mir_eval::observe(&p, 1_000_000);
    let mut device = invar_metal::MetalDevice::open_runtime_source().unwrap();
    let old = invar::tensor::host::metal::Metal::open(
        SCALAR.as_bytes(),
        "mat",
        metal::Kernels::RuntimeSource,
    )
    .unwrap();
    let caps = old.capabilities();
    let mut old = invar_cuda::Runtime::new(old, 1);
    let mut adapter = Vec::new();
    let mut host = Vec::new();
    for sample in 0..4 {
        let start = Instant::now();
        let p = program(SCALAR);
        let run =
            metal::run_source_with_device(SCALAR.as_bytes(), &p, 1_000_000, &mut device).unwrap();
        let ms = start.elapsed().as_secs_f64() * 1000.0;
        parity(&serial, &run.observation);
        assert!(run.executed.iter().any(|e| e.kind == PlanKind::Tensor));
        if sample > 0 {
            adapter.push(ms);
        }
        let start = Instant::now();
        let run =
            invar::tensor::host::run_source(SCALAR.as_bytes(), 1_000_000, &mut old, caps).unwrap();
        let ms = start.elapsed().as_secs_f64() * 1000.0;
        parity(&serial, &run.observation);
        assert_eq!(run.completed, 1);
        if sample > 0 {
            host.push(ms);
        }
    }
    adapter.sort_by(f64::total_cmp);
    host.sort_by(f64::total_cmp);
    println!(
        "{}",
        serde_json::json!({"boundary":"host-to-host, already-open device, warm compiler; frontend checking, source recognition and publication included","dimensions":[3,5,7],"samples":3,"adapter_ms":adapter,"existing_host_ms":host,"adapter_median_ms":adapter[1],"existing_host_median_ms":host[1]})
    );
}
#[test]
#[ignore = "requires an Apple GPU; TC enters through the family's private checker"]
fn a1_tensor_metal_explicit_tc_completed_region() {
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
    let p = program(&source);
    let run = metal::run_source(
        source.as_bytes(),
        &p,
        1_000_000,
        metal::Kernels::RuntimeSource,
    )
    .unwrap();
    assert_eq!(run.observation.result, Ok(invar::eval::Value::I32(42)));
    assert_eq!(
        run.executed
            .iter()
            .filter(|e| e.kind == PlanKind::Tensor)
            .count(),
        1
    );
}
