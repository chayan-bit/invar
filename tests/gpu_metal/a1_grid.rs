//! Ordinary run-path evidence for grid adapters on a real Metal device; the CPU boundary tests stay in `tests/a1_grid.rs`.
use invar::{check, eval, front, mir_eval};

#[path = "../support/a1_grid_common.rs"]
mod a1_grid_common;
use a1_grid_common::{FUEL, fixtures, parity, program, sources};

#[test]
fn a1_grid_metal_completed_equality_faults_cost_guards_and_cli() {
    use invar::{plan::DeviceCostModel, schedule::metal};
    use std::time::Instant;
    let cheap = DeviceCostModel {
        cpu_bytes_per_ms: 1.0,
        device_bytes_per_ms: 1e12,
        device_fixed_ms: 0.0,
    };
    let expensive = DeviceCostModel {
        cpu_bytes_per_ms: 1e12,
        device_bytes_per_ms: 1.0,
        device_fixed_ms: 10.0,
    };
    let mut device = invar_metal::MetalDevice::open_runtime_source().unwrap();
    for source in sources() {
        let program = program(&source);
        let start = Instant::now();
        let serial = mir_eval::observe(&program, FUEL);
        let serial_ms = start.elapsed().as_secs_f64() * 1000.0;
        let start = Instant::now();
        let run =
            metal::run_source_with_device(source.as_bytes(), &program, FUEL, &mut device).unwrap();
        let metal_ms = start.elapsed().as_secs_f64() * 1000.0;
        parity(&run.observation, &serial);
        assert_eq!(run.observation.result, Ok(eval::Value::I32(7)));
        assert!(!run.executed.is_empty());
        println!(
            "metal-grid kind={:?} regions={} serial_mir_ms={serial_ms:.3} ordinary_metal_ms={metal_ms:.3}",
            run.executed[0].kind,
            run.executed.len()
        );
        let checked = check::check(front::parse(source.as_bytes()).unwrap(), true).unwrap();
        let started = Instant::now();
        let direct = if source.starts_with("fn flip") {
            let contract =
                invar::transpose_source::Contract::from_checked(&checked, "flip").unwrap();
            let plan = contract
                .device_plan([17, 31], 17 * 31, device.capabilities())
                .unwrap();
            let input = (0..17u32 * 31)
                .map(|i| {
                    if contract.element() == front::Type::F32 {
                        invar::numeric_contract::F32Bits::from_decimal(&format!("{}.25", i % 97))
                            .unwrap()
                            .to_bits()
                    } else {
                        i.wrapping_mul(2654435761)
                    }
                })
                .collect::<Vec<_>>();
            let result = device.run_transpose(&plan, &input).unwrap();
            assert_eq!(
                result.output,
                (0..17usize * 31)
                    .map(|i| input[(i % 17) * 31 + i / 17])
                    .collect::<Vec<_>>()
            );
            result
        } else {
            let contract = invar::stencil_source::Contract::from_checked(&checked, "step").unwrap();
            let plan = contract
                .device_u32([3, 5], run.executed.len() as u32, device.capabilities())
                .unwrap();
            let mut expected = (1..=15).collect::<Vec<u32>>();
            let result = device.run_stencil_u32(&plan, &expected).unwrap();
            for _ in 0..run.executed.len() {
                expected = (0..15u32)
                    .map(|i| {
                        plan.reads([i / 5, i % 5])
                            .unwrap()
                            .into_iter()
                            .flatten()
                            .fold(0u32, |sum, at| sum.wrapping_add(expected[at as usize]))
                    })
                    .collect();
            }
            assert_eq!(result.output, expected);
            assert_eq!(result.phases, run.executed.len() as u32);
            assert_eq!(result.final_buffer, run.executed.len() % 2);
            result
        };
        println!(
            "existing-family-metal-host-ms={:.3} phases={} words={}",
            started.elapsed().as_secs_f64() * 1000.0,
            direct.phases,
            direct.output.len()
        );
        if source.starts_with("fn flip") {
            let mut runtime = invar_cuda::Runtime::new(invar::transpose_source::host::CpuDevice, 2);
            let start = Instant::now();
            let host = invar::transpose_source::host::run_source(
                source.as_bytes(),
                FUEL,
                &mut runtime,
                invar_device::Capabilities::cuda_12_9_sm89(),
            )
            .unwrap();
            parity(&host.observation, &serial);
            println!(
                "existing-transpose-host-cpu-ms={:.3}",
                start.elapsed().as_secs_f64() * 1000.0
            );
        }
        for (model, want_device) in [(cheap, true), (expensive, false)] {
            let auto = metal::run_auto_source_with_fault(
                source.as_bytes(),
                &program,
                FUEL,
                metal::Kernels::RuntimeSource,
                model,
                metal::Fault::None,
            )
            .unwrap();
            parity(&auto.observation, &serial);
            assert_eq!(
                auto.choices.iter().any(|c| c.backend == "metal"),
                want_device
            );
        }
        let completed_regions = run.executed.len();
        for fault in [metal::Fault::OutOfMemory, metal::Fault::LaunchFailure] {
            let run = metal::run_source_with_device_fault(
                source.as_bytes(),
                &program,
                FUEL,
                &mut device,
                fault,
            )
            .unwrap();
            assert!(run.executed.is_empty());
            assert!(matches!(
                run.observation.result,
                Err(eval::EvalError::Tool {
                    code: "METAL_UNAVAILABLE",
                    ..
                })
            ));
            assert!(!device.is_poisoned(), "test fault is confirmed quiescent");
            let auto = metal::run_auto_source_with_fault(
                source.as_bytes(),
                &program,
                FUEL,
                metal::Kernels::RuntimeSource,
                cheap,
                fault,
            )
            .unwrap();
            parity(&auto.observation, &serial);
            assert_eq!(auto.choices.first().unwrap().backend, "mir-serial");
            assert_eq!(auto.executed.len(), completed_regions - 1);
        }
    }
    for source in [
        fixtures::transpose(false, 3, 5, false, true),
        fixtures::stencil(false, false, 3, 5, 1),
    ] {
        let program = program(&source);
        let pending = metal::run_source_with_device_fault(
            source.as_bytes(),
            &program,
            FUEL,
            &mut device,
            metal::Fault::PartialCompletion,
        )
        .unwrap();
        assert!(pending.executed.is_empty());
        assert!(matches!(
            pending.observation.result,
            Err(eval::EvalError::Tool {
                code: "METAL_PENDING",
                ..
            })
        ));
        assert!(device.is_poisoned());
        let deadline = Instant::now() + std::time::Duration::from_secs(10);
        while device.is_poisoned() {
            assert!(Instant::now() < deadline);
            device.reap();
            std::thread::yield_now();
        }
        let auto = metal::run_auto_source_with_fault(
            source.as_bytes(),
            &program,
            FUEL,
            metal::Kernels::RuntimeSource,
            cheap,
            metal::Fault::PartialCompletion,
        )
        .unwrap();
        assert!(auto.executed.is_empty());
        assert!(matches!(
            auto.observation.result,
            Err(eval::EvalError::Tool {
                code: "METAL_PENDING",
                ..
            })
        ));
    }
    for source in [
        fixtures::transpose(false, 3, 5, true, true),
        fixtures::transpose(false, 3, 5, false, false),
        fixtures::transpose(false, 0, 5, false, true),
        fixtures::transpose(false, 5, 0, false, true),
        fixtures::stencil(true, true, 3, 5, 1),
    ] {
        let program = program(&source);
        let auto = metal::run_auto_source_with_fault(
            source.as_bytes(),
            &program,
            FUEL,
            metal::Kernels::RuntimeSource,
            cheap,
            metal::Fault::None,
        )
        .unwrap();
        parity(&auto.observation, &mir_eval::observe(&program, FUEL));
        assert!(auto.executed.is_empty());
        match metal::run_source_with_device(source.as_bytes(), &program, FUEL, &mut device) {
            Err(e) => assert_eq!(e.code, "METAL_UNAVAILABLE"),
            Ok(run) => {
                assert!(run.executed.is_empty());
                assert!(matches!(
                    run.observation.result,
                    Err(eval::EvalError::Tool {
                        code: "METAL_UNAVAILABLE",
                        ..
                    })
                ));
            }
        }
    }
    let path = std::path::Path::new("demo/a1-grid.inv");
    for engine in ["metal", "auto", "cuda"] {
        let output = std::process::Command::new(env!("CARGO_BIN_EXE_invar"))
            .env("INVAR_METAL_RUNTIME_SOURCE", "1")
            .current_dir(env!("CARGO_MANIFEST_DIR"))
            .arg("run")
            .arg(path)
            .args(["--engine", engine, "--format", "json"])
            .output()
            .unwrap();
        let json: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
        if engine == "cuda" {
            assert_eq!(json["diagnostics"][0]["code"], "CUDA_UNAVAILABLE");
        } else {
            assert!(output.status.success(), "{json}");
            assert_eq!(json["result"]["value"], 7);
        }
        println!("invar run demo/a1-grid.inv --engine {engine} --format json => {json}");
    }
    let output = std::process::Command::new(env!("CARGO_BIN_EXE_invar"))
        .env("INVAR_METAL_RUNTIME_SOURCE", "1")
        .current_dir(env!("CARGO_MANIFEST_DIR"))
        .arg("explain")
        .arg(path)
        .args(["--engine", "metal", "--format", "json"])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stdout)
    );
    let json: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    println!("metal routing {}", json["device_execution"]);
}
