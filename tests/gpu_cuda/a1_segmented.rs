//! Ordinary-source segmented publication on a real CUDA device; the CLI refusal test stays in `tests/device/issue_a1_segmented.rs`.
use invar::mir_eval;
use std::path::Path;

#[path = "../support/a1_segmented_common.rs"]
mod a1_segmented_common;
use a1_segmented_common::{cases, cli, program, source};

#[test]
#[ignore = "requires physical CUDA"]
fn cuda_source_equality_and_completion_failures() {
    use invar::schedule::{PlanKind, cuda};
    use invar_cuda::cuda::CudaDevice;
    let compiler = Path::new(env!("CARGO_BIN_EXE_invar-nvrtc"));
    let mut device = CudaDevice::open().unwrap();
    for case in cases::cases(false) {
        let zero = match case.kind {
            cases::Kind::Compact(_) => case.input.is_empty(),
            _ => case.offsets.len() == 1,
        };
        if zero {
            continue;
        }
        let s = source(&case);
        let p = program(&s);
        let serial = mir_eval::observe(&p, 10_000_000);
        let run = cuda::run_source_with_device(s.as_bytes(), &p, 10_000_000, &mut device, compiler)
            .unwrap();
        assert_eq!(run.observation.result, serial.result, "{}", case.name);
        assert_eq!(
            run.observation.steps_used, serial.steps_used,
            "{}",
            case.name
        );
        assert_eq!(
            run.observation.cleanup_trace, serial.cleanup_trace,
            "{}",
            case.name
        );
        assert!(
            run.executed
                .iter()
                .any(|r| matches!(r.kind, PlanKind::Segmented | PlanKind::Compaction))
        );
    }
    let case = cases::cases(false)
        .into_iter()
        .find(|c| c.name == "Map/empty-interior")
        .unwrap();
    let s = source(&case);
    let p = program(&s);
    for fault in [
        cuda::Fault::OutOfMemory,
        cuda::Fault::LaunchFailure,
        cuda::Fault::PartialCompletion,
    ] {
        let mut device = CudaDevice::open().unwrap();
        let failed = cuda::run_source_with_device_fault(
            s.as_bytes(),
            &p,
            10_000_000,
            &mut device,
            compiler,
            fault,
        )
        .unwrap();
        assert!(failed.executed.is_empty());
        assert!(failed.observation.result.is_err());
        assert_eq!(
            device.is_poisoned(),
            fault == cuda::Fault::PartialCompletion
        );
    }
    let dir = std::env::temp_dir().join(format!("invar-a1-cuda-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("segmented.inv");
    std::fs::write(&path, s).unwrap();
    for engine in ["cuda", "auto"] {
        let result = cli(engine, &path);
        assert_eq!(result["status"], "ok", "{result}");
        assert_eq!(result["result"]["value"], 0);
        println!("CLI {engine}: {result}");
    }
    std::fs::remove_dir_all(dir).unwrap();
}
