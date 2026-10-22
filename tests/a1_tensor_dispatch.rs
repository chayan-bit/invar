//! Ordinary CLI and owning-device tests. Hardware tests fail if Metal is absent.
use invar::{check, front, mir, mir_eval};
use std::process::Command;
const SCALAR: &str = include_str!("a1_tensor/scalar.inv");
const DIRECT: &str = include_str!("a1_tensor/direct.inv");
const TC: &str = include_str!("a1_tensor/tc.inv");
#[path = "support/gnet_agree.rs"]
mod agree;
fn assert_net(checked: &check::CheckedProgram, fuel: u64, expected: &invar::eval::Observation) {
    use invar::general_net::{Limits, Net, oracle};
    let net = Net::lower(checked, Limits::default()).unwrap();
    let observed = oracle::observe(
        &net,
        &net.verify().unwrap(),
        oracle::Limits {
            fuel,
            ..oracle::Limits::default()
        },
    )
    .unwrap();
    assert_eq!(observed.failure, None, "{:?}", observed.semantic);
    agree::same(&observed.semantic, expected, "ordinary TC net oracle");
}
fn program(source: &str) -> mir::Program {
    mir::lower(&check::check(front::parse(source.as_bytes()).unwrap(), true).unwrap()).unwrap()
}
#[test]
fn a1_tensor_ordinary_tc_rounds_inputs_and_serial_fallback() {
    let source = include_str!("a1_tensor/tc.inv");
    let checked = check::check(front::parse(source.as_bytes()).unwrap(), true).unwrap();
    invar::ownership::check(&checked).unwrap();
    let p = mir::lower(&checked).unwrap();
    let reference = invar::eval::observe(&checked, 10_000);
    assert_eq!(reference.result, Ok(invar::eval::Value::I32(42)));
    for fuel in [1, reference.steps_used - 1, reference.steps_used, 10_000] {
        let expected = invar::eval::observe(&checked, fuel);
        let actual = mir_eval::observe(&p, fuel);
        assert_eq!(actual.result, expected.result);
        assert_eq!(actual.steps_used, expected.steps_used);
        assert_eq!(actual.cleanup_trace, expected.cleanup_trace);
        assert_net(&checked, fuel, &expected);
    }
    for engine in ["reference", "cpu-parallel", "auto"] {
        let mut command = Command::new(env!("CARGO_BIN_EXE_invar"));
        command.args([
            "run",
            "tests/a1_tensor/tc.inv",
            "--engine",
            engine,
            "--format",
            "json",
        ]);
        if engine == "cpu-parallel" {
            command.args(["--workers", "2"]);
        }
        let got = command.output().unwrap();
        assert!(
            got.status.success(),
            "{engine}: {} {}",
            String::from_utf8_lossy(&got.stdout),
            String::from_utf8_lossy(&got.stderr)
        );
        let doc: serde_json::Value = serde_json::from_slice(&got.stdout).unwrap();
        assert_eq!(doc["result"]["value"], 42, "{engine}: {doc}");
    }
}
#[test]
fn a1_tensor_ordinary_tc_exact_sum_range_and_demand() {
    // 65504^2 + 2^-24 - 65504^2: sequential f32 loses the middle
    // term, while D5's once-rounded exact reference returns it.
    let exact = TC
        .replace(
            "[1.00048828125]",
            "[65504.0,0.000000059604644775390625,-65504.0]",
        )
        .replace("[1.0]", "[65504.0,1.0,65504.0]")
        .replace("mat(a,b,1u32,1u32,1u32)", "mat(a,b,1u32,3u32,1u32)")
        .replace("out[0u32]==1.0", "out[0u32]==0.000000059604644775390625");
    for source in [
        exact,
        TC.replace("[1.00048828125]", "[1.00146484375]")
            .replace("out[0u32]==1.0", "out[0u32]==1.001953125"),
        TC.replace("[1.00048828125]", "[1.0,(1.0/0.0)]"),
        TC.replace("[1.0]", "[1.0,(1.0/0.0)]"),
    ] {
        let checked = check::check(front::parse(source.as_bytes()).unwrap(), true).unwrap();
        let p = mir::lower(&checked).unwrap();
        let expected = invar::eval::observe(&checked, 10_000);
        let actual = mir_eval::observe(&p, 10_000);
        assert_eq!(expected.result, Ok(invar::eval::Value::I32(42)));
        assert_eq!(actual.result, expected.result);
        assert_eq!(actual.steps_used, expected.steps_used);
        assert_eq!(actual.cleanup_trace, expected.cleanup_trace);
        assert_net(&checked, 10_000, &expected);
        #[cfg(feature = "native")]
        {
            let (status, doc) = invar::native::run_program(&p, 10_000).unwrap();
            assert_eq!(status, 0, "{doc}");
            assert_eq!(doc["result"]["value"], 42, "{doc}");
            assert_eq!(doc["steps_used"], expected.steps_used);
        }
    }
    for bad in ["65520.0", "(1.0/0.0)", "(0.0/0.0)"] {
        let source = TC.replace("1.00048828125", bad);
        let checked = check::check(front::parse(source.as_bytes()).unwrap(), true).unwrap();
        let p = mir::lower(&checked).unwrap();
        let expected = invar::eval::observe(&checked, 10_000);
        let actual = mir_eval::observe(&p, 10_000);
        assert!(matches!(
            expected.result,
            Err(invar::eval::EvalError::Tool {
                code: "TC_INPUT_RANGE",
                ..
            })
        ));
        assert_eq!(actual.result, expected.result);
        assert_eq!(actual.steps_used, expected.steps_used);
        assert_eq!(actual.cleanup_trace, expected.cleanup_trace);
        assert_net(&checked, 10_000, &expected);
        // An unreached contraction must not inspect or round its input.
        let unused = source.replace(
            "let out=mat(a,b,1u32,1u32,1u32);if out[0u32]==1.0{return 42;}return 99;",
            "if false{let out=mat(a,b,1u32,1u32,1u32);}return 42;",
        );
        assert_eq!(
            mir_eval::observe(&program(&unused), 10_000).result,
            Ok(invar::eval::Value::I32(42))
        );
        let checked = check::check(front::parse(unused.as_bytes()).unwrap(), true).unwrap();
        assert_net(&checked, 10_000, &invar::eval::observe(&checked, 10_000));
    }
    for dims in ["0u32,1u32,1u32", "1u32,0u32,1u32", "1u32,1u32,0u32"] {
        let source = format!(
            "{}fn main()->i32{{let a:Array<f32> = [1.0/0.0];let b:Array<f32> = [1.0];let out=mat(a,b,{dims});return 42;}}",
            TC.split("fn main").next().unwrap()
        );
        let checked = check::check(front::parse(source.as_bytes()).unwrap(), true).unwrap();
        let expected = invar::eval::observe(&checked, 10_000);
        assert_eq!(expected.result, Ok(invar::eval::Value::I32(42)));
        assert_net(&checked, 10_000, &expected);
    }
    let short = TC.replace("mat(a,b,1u32,1u32,1u32)", "mat(a,b,1u32,2u32,1u32)");
    let checked = check::check(front::parse(short.as_bytes()).unwrap(), true).unwrap();
    let expected = invar::eval::observe(&checked, 10_000);
    assert!(matches!(
        expected.result,
        Err(invar::eval::EvalError::Trap {
            category: invar::eval::TrapCategory::IndexOutOfBounds,
            ..
        })
    ));
    assert_net(&checked, 10_000, &expected);
}

#[test]
fn a1_tensor_ordinary_multiple_tc_functions_keep_independent_admission() {
    let source = format!(
        "{}{}",
        TC,
        TC.split("fn main")
            .next()
            .unwrap()
            .replace("fn mat(", "fn other(")
    );
    let p = program(&source);
    let serial = mir_eval::observe(&p, 10_000);
    let run = invar::schedule::run_source(source.as_bytes(), &p, 10_000, 2).unwrap();
    assert_eq!(run.observation.result, serial.result);
    assert_eq!(run.observation.steps_used, serial.steps_used);
    assert_eq!(run.observation.cleanup_trace, serial.cleanup_trace);
}

#[test]
fn a1_tensor_serial_forms_and_forced_unavailable_cli() {
    for source in [SCALAR, DIRECT, TC] {
        assert_eq!(
            mir_eval::observe(&program(source), 1_000_000).result,
            Ok(invar::eval::Value::I32(42))
        );
    }
    #[cfg(not(feature = "cuda"))]
    {
        let got = Command::new(env!("CARGO_BIN_EXE_invar"))
            .args([
                "run",
                "tests/a1_tensor/direct.inv",
                "--engine",
                "cuda",
                "--format",
                "json",
            ])
            .output()
            .unwrap();
        assert_eq!(got.status.code(), Some(2));
        let doc: serde_json::Value = serde_json::from_slice(&got.stdout).unwrap();
        assert_eq!(doc["diagnostics"][0]["code"], "CUDA_UNAVAILABLE");
    }
    #[cfg(not(all(feature = "metal", target_os = "macos")))]
    {
        let got = Command::new(env!("CARGO_BIN_EXE_invar"))
            .args([
                "run",
                "tests/a1_tensor/direct.inv",
                "--engine",
                "metal",
                "--format",
                "json",
            ])
            .output()
            .unwrap();
        assert_eq!(got.status.code(), Some(2));
        let doc: serde_json::Value = serde_json::from_slice(&got.stdout).unwrap();
        assert_eq!(doc["diagnostics"][0]["code"], "METAL_UNAVAILABLE");
    }
}

#[cfg(all(feature = "metal", target_os = "macos"))]
mod metal {
    use super::*;
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
            let run = metal::run_source_with_device(source.as_bytes(), &p, 1_000_000, &mut device)
                .unwrap();
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
            let run = metal::run_source_with_device(SCALAR.as_bytes(), &p, 1_000_000, &mut device)
                .unwrap();
            let ms = start.elapsed().as_secs_f64() * 1000.0;
            parity(&serial, &run.observation);
            assert!(run.executed.iter().any(|e| e.kind == PlanKind::Tensor));
            if sample > 0 {
                adapter.push(ms);
            }
            let start = Instant::now();
            let run = invar::tensor::host::run_source(SCALAR.as_bytes(), 1_000_000, &mut old, caps)
                .unwrap();
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
}

#[cfg(feature = "cuda")]
mod cuda {
    use super::*;
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
            let got = cuda::run_source_with_device(
                source.as_bytes(),
                &p,
                1_000_000,
                &mut device,
                compiler,
            )
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
                assert!(got.choices.iter().any(|c| c.function == "mat"
                    && c.backend == if device { "cuda" } else { "mir-serial" }));
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
        let run =
            cuda::run_source_with_device(source.as_bytes(), &p, 1_000_000, &mut device, compiler)
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
        let old =
            invar::tensor::host::cuda::Cuda::open(SCALAR.as_bytes(), "mat", compiler).unwrap();
        let caps = old.capabilities();
        let mut old = invar_cuda::Runtime::new(old, 1);
        let mut adapter = Vec::new();
        let mut host = Vec::new();
        for sample in 0..4 {
            let start = Instant::now();
            let p = program(SCALAR);
            let run = cuda::run_source_with_device(
                SCALAR.as_bytes(),
                &p,
                1_000_000,
                &mut device,
                compiler,
            )
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
            let run = invar::tensor::host::run_source(SCALAR.as_bytes(), 1_000_000, &mut old, caps)
                .unwrap();
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
}
