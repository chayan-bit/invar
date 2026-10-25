//! Ordinary run-path evidence for grid adapters on a real CUDA device; the CPU boundary tests stay in `tests/device/a1_grid.rs`.
use invar::{check, eval, front, mir_eval};

#[path = "../support/a1_grid_common.rs"]
mod a1_grid_common;
use a1_grid_common::{FUEL, fixtures, parity, program, sources};

#[test]
#[ignore = "requires an exclusively locked CUDA device and INVAR_NVRTC_COMPILER"]
fn a1_grid_cuda_completed_equality() {
    use invar::{plan::DeviceCostModel, schedule::cuda};
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
    let mut device = invar_cuda::cuda::CudaDevice::open().unwrap();
    println!(
        "cuda-device name={} caps={:?}",
        device.name(),
        device.capabilities()
    );
    let compiler = std::env::var_os("INVAR_NVRTC_COMPILER").expect("INVAR_NVRTC_COMPILER");
    let compiler = std::path::Path::new(&compiler);
    for source in sources() {
        let program = program(&source);
        let started = Instant::now();
        let serial = mir_eval::observe(&program, FUEL);
        let serial_ms = started.elapsed().as_secs_f64() * 1000.0;
        let started = Instant::now();
        let run =
            cuda::run_source_with_device(source.as_bytes(), &program, FUEL, &mut device, compiler)
                .unwrap();
        let ordinary_ms = started.elapsed().as_secs_f64() * 1000.0;
        parity(&run.observation, &serial);
        assert_eq!(run.observation.result, Ok(eval::Value::I32(7)));
        assert!(!run.executed.is_empty());
        println!(
            "cuda-grid kind={:?} regions={} serial_mir_ms={serial_ms:.3} ordinary_cuda_ms={ordinary_ms:.3}",
            run.executed[0].kind,
            run.executed.len()
        );
        let checked = check::check(front::parse(source.as_bytes()).unwrap(), true).unwrap();
        let started = Instant::now();
        if source.starts_with("fn flip") {
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
            let result = device.run_transpose(&plan, &input, compiler).unwrap();
            assert_eq!(
                result.output,
                (0..17usize * 31)
                    .map(|i| input[(i % 17) * 31 + i / 17])
                    .collect::<Vec<_>>()
            );
            println!(
                "existing-family-cuda-host-ms={:.3} words={} cubin_bytes={}",
                started.elapsed().as_secs_f64() * 1000.0,
                result.output.len(),
                result.cubin_bytes
            );
            if let Some(path) = std::env::var_os("INVAR_GRID_CUDA_SOURCE") {
                std::fs::write(path, invar_device::transpose::emit_cuda(&plan).unwrap()).unwrap();
            }
        } else {
            let contract = invar::stencil_source::Contract::from_checked(&checked, "step").unwrap();
            let plan = contract
                .device_u32([3, 5], run.executed.len() as u32, device.capabilities())
                .unwrap();
            let mut expected = (1..=15).collect::<Vec<u32>>();
            let result = device.run_stencil_u32(&plan, &expected, compiler).unwrap();
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
            println!(
                "existing-family-cuda-host-ms={:.3} phases={} words={}",
                started.elapsed().as_secs_f64() * 1000.0,
                result.phases,
                result.output.len()
            );
        }
        for (model, want_device) in [(cheap, true), (expensive, false)] {
            let auto = cuda::run_auto_source_with_fault(
                source.as_bytes(),
                &program,
                FUEL,
                compiler,
                model,
                cuda::Fault::None,
            )
            .unwrap();
            parity(&auto.observation, &serial);
            assert_eq!(
                auto.choices.iter().any(|c| c.backend == "cuda"),
                want_device
            );
        }
        for fault in [cuda::Fault::OutOfMemory, cuda::Fault::LaunchFailure] {
            let failed = cuda::run_source_with_device_fault(
                source.as_bytes(),
                &program,
                FUEL,
                &mut device,
                compiler,
                fault,
            )
            .unwrap();
            assert!(failed.executed.is_empty());
            assert!(matches!(
                failed.observation.result,
                Err(eval::EvalError::Tool {
                    code: "CUDA_UNAVAILABLE",
                    ..
                })
            ));
            assert!(!device.is_poisoned());
            let auto = cuda::run_auto_source_with_fault(
                source.as_bytes(),
                &program,
                FUEL,
                compiler,
                cheap,
                fault,
            )
            .unwrap();
            parity(&auto.observation, &serial);
            assert_eq!(auto.choices.first().unwrap().backend, "mir-serial");
            assert_eq!(auto.executed.len(), run.executed.len() - 1);
        }
    }
    for source in [
        fixtures::transpose(false, 3, 5, false, true),
        fixtures::stencil(false, false, 3, 5, 1),
    ] {
        let program = program(&source);
        // A compiler failure retains the shared toolchain-mismatch diagnostic,
        // publishes nothing, and permits fallback before submission.
        let bad_compiler = std::path::Path::new("/bin/false");
        let failed = cuda::run_source_with_device(
            source.as_bytes(),
            &program,
            FUEL,
            &mut device,
            bad_compiler,
        )
        .unwrap();
        assert!(failed.executed.is_empty());
        assert!(
            matches!(
                &failed.observation.result,
                Err(eval::EvalError::Tool {
                    code: "DEVICE_TOOLCHAIN_MISMATCH",
                    ..
                })
            ),
            "{:?}",
            failed.observation.result
        );
        let auto = cuda::run_auto_source_with_fault(
            source.as_bytes(),
            &program,
            FUEL,
            bad_compiler,
            cheap,
            cuda::Fault::None,
        )
        .unwrap();
        parity(&auto.observation, &mir_eval::observe(&program, FUEL));
        assert!(auto.executed.is_empty());
        let pending = cuda::run_source_with_device_fault(
            source.as_bytes(),
            &program,
            FUEL,
            &mut device,
            compiler,
            cuda::Fault::PartialCompletion,
        )
        .unwrap();
        assert!(pending.executed.is_empty());
        assert!(matches!(
            pending.observation.result,
            Err(eval::EvalError::Tool {
                code: "CUDA_PENDING",
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
        let auto = cuda::run_auto_source_with_fault(
            source.as_bytes(),
            &program,
            FUEL,
            compiler,
            cheap,
            cuda::Fault::PartialCompletion,
        )
        .unwrap();
        assert!(auto.executed.is_empty());
        assert!(matches!(
            auto.observation.result,
            Err(eval::EvalError::Tool {
                code: "CUDA_PENDING",
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
        let auto = cuda::run_auto_source_with_fault(
            source.as_bytes(),
            &program,
            FUEL,
            compiler,
            cheap,
            cuda::Fault::None,
        )
        .unwrap();
        parity(&auto.observation, &mir_eval::observe(&program, FUEL));
        assert!(auto.executed.is_empty());
        match cuda::run_source_with_device(source.as_bytes(), &program, FUEL, &mut device, compiler)
        {
            Err(e) => assert_eq!(e.code, "CUDA_UNAVAILABLE"),
            Ok(run) => {
                assert!(run.executed.is_empty());
                assert!(matches!(
                    run.observation.result,
                    Err(eval::EvalError::Tool {
                        code: "CUDA_UNAVAILABLE",
                        ..
                    })
                ));
            }
        }
    }
    for engine in ["cuda", "auto"] {
        let output = std::process::Command::new(env!("CARGO_BIN_EXE_invar"))
            .env("INVAR_METAL_RUNTIME_SOURCE", "1")
            .current_dir(env!("CARGO_MANIFEST_DIR"))
            .args([
                "run",
                "demo/a1-grid.inv",
                "--engine",
                engine,
                "--format",
                "json",
            ])
            .output()
            .unwrap();
        let json: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
        assert!(output.status.success(), "{json}");
        assert_eq!(json["result"]["value"], 7);
        println!("invar run demo/a1-grid.inv --engine {engine} --format json => {json}");
    }
    let output = std::process::Command::new(env!("CARGO_BIN_EXE_invar"))
        .env("INVAR_METAL_RUNTIME_SOURCE", "1")
        .current_dir(env!("CARGO_MANIFEST_DIR"))
        .args([
            "explain",
            "demo/a1-grid.inv",
            "--engine",
            "cuda",
            "--format",
            "json",
        ])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stdout)
    );
    let json: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    assert!(
        json["device_execution"]["choices"]
            .as_array()
            .unwrap()
            .iter()
            .any(|r| r["backend"] == "cuda" && r["kind"] == "Transpose")
    );
    println!("cuda routing {}", json["device_execution"]);
}
