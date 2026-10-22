//! Deterministic before/after placement record over the unchanged M2 corpus.
use invar::{admission, check, eval, front, mir, schedule};
use std::{fmt::Write, path::PathBuf};

#[test]
#[ignore = "requires INVAR_M3_06_CORPUS_DIR and FAMILY_DISPATCH_LOG; Metal builds use hardware"]
fn family_dispatch_corpus_decisions() {
    let directory = std::env::var_os("INVAR_M3_06_CORPUS_DIR").expect("corpus directory");
    let mut files = std::fs::read_dir(directory)
        .unwrap()
        .map(|e| e.unwrap().path())
        .filter(|p| p.extension().is_some_and(|e| e == "inv" || e == "invar"))
        .collect::<Vec<PathBuf>>();
    files.sort();
    assert_eq!(files.len(), 252);
    #[cfg(all(feature = "metal", target_os = "macos"))]
    let measured = std::env::var_os("FAMILY_DISPATCH_MODEL").map(|path| {
        use invar::{plan::DeviceCostModel, schedule::metal};
        let path = PathBuf::from(path);
        if !path.exists() {
            let model =
                metal::calibrate(&mut || metal::open_device(&[], metal::Kernels::RuntimeSource))
                    .unwrap();
            std::fs::write(
                &path,
                serde_json::json!({
                    "cpu_bytes_per_ms": model.cpu_bytes_per_ms,
                    "device_bytes_per_ms": model.device_bytes_per_ms,
                    "device_fixed_ms": model.device_fixed_ms,
                })
                .to_string(),
            )
            .unwrap();
        }
        let data: serde_json::Value =
            serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
        DeviceCostModel {
            cpu_bytes_per_ms: data["cpu_bytes_per_ms"].as_f64().unwrap(),
            device_bytes_per_ms: data["device_bytes_per_ms"].as_f64().unwrap(),
            device_fixed_ms: data["device_fixed_ms"].as_f64().unwrap(),
        }
    });
    let mut log = String::new();
    for file in files {
        let source = std::fs::read(&file).unwrap();
        let checked = check::check(front::parse(&source).unwrap(), true).unwrap();
        let program = mir::lower(&checked).unwrap();
        let reference = eval::observe(&checked, 1_000_000);
        let report = admission::analyze(&source, &program).unwrap();
        writeln!(log, "{}", file.file_name().unwrap().to_str().unwrap()).unwrap();
        writeln!(log, "plans={}", report.to_json()).unwrap();
        let cpu = schedule::run_source(&source, &program, 1_000_000, 4).unwrap();
        assert_eq!(cpu.observation.result, reference.result);
        assert_eq!(cpu.observation.steps_used, reference.steps_used);
        assert_eq!(cpu.observation.cleanup_trace, reference.cleanup_trace);
        writeln!(
            log,
            "cpu={:?} steps={} cleanup={:?} executed={:?}",
            cpu.observation.result,
            cpu.observation.steps_used,
            cpu.observation.cleanup_trace,
            cpu.executed
        )
        .unwrap();
        #[cfg(all(feature = "metal", target_os = "macos"))]
        {
            use invar::{plan::DeviceCostModel, schedule::metal};
            let mut models = vec![
                (
                    "cpu-cost",
                    DeviceCostModel {
                        cpu_bytes_per_ms: 1e12,
                        device_bytes_per_ms: 1.0,
                        device_fixed_ms: 10.0,
                    },
                ),
                (
                    "device-cost",
                    DeviceCostModel {
                        cpu_bytes_per_ms: 1.0,
                        device_bytes_per_ms: 1e12,
                        device_fixed_ms: 0.0,
                    },
                ),
            ];
            if let Some(model) = measured {
                models.push(("measured-cost", model));
            }
            for (label, model) in models {
                let run = metal::run_auto_source_with_fault(
                    &source,
                    &program,
                    1_000_000,
                    metal::Kernels::RuntimeSource,
                    model,
                    metal::Fault::None,
                )
                .unwrap();
                assert_eq!(run.observation.result, reference.result);
                assert_eq!(run.observation.steps_used, reference.steps_used);
                assert_eq!(run.observation.cleanup_trace, reference.cleanup_trace);
                writeln!(
                    log,
                    "{label}={:?} steps={} cleanup={:?} executed={:?} choices={:?}",
                    run.observation.result,
                    run.observation.steps_used,
                    run.observation.cleanup_trace,
                    run.executed,
                    run.choices
                )
                .unwrap();
            }
            match metal::run_source(&source, &program, 1_000_000, metal::Kernels::RuntimeSource) {
                Ok(run) => {
                    assert_eq!(run.observation.result, reference.result);
                    assert_eq!(run.observation.steps_used, reference.steps_used);
                    assert_eq!(run.observation.cleanup_trace, reference.cleanup_trace);
                    writeln!(
                        log,
                        "forced={:?} steps={} executed={:?}",
                        run.observation.result, run.observation.steps_used, run.executed
                    )
                    .unwrap();
                }
                Err(error) => {
                    assert_eq!(error.code, "METAL_UNAVAILABLE");
                    assert!(cpu.executed.is_empty());
                    writeln!(log, "forced={error:?}").unwrap();
                }
            }
        }
    }
    std::fs::write(
        std::env::var_os("FAMILY_DISPATCH_LOG").expect("log destination"),
        log,
    )
    .unwrap();
}
