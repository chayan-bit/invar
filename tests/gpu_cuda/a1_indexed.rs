//! Indexed source routing on a real CUDA device; the CPU routing tests stay in `tests/issue_a1_indexed.rs`.
use invar::{mir, mir_eval, schedule};

#[path = "../support/a1_indexed_common.rs"]
mod a1_indexed_common;
use a1_indexed_common::{CASES, FUEL, equal, lower, static_refusals};

#[test]
#[ignore = "requires CUDA hardware and the NVRTC helper"]
fn a1_indexed_cuda_completed_outputs_and_forced_refusals() {
    let compiler =
        std::path::PathBuf::from(std::env::var_os("INVAR_NVRTC_HELPER").expect("NVRTC helper"));
    let mut device = invar_cuda::cuda::CudaDevice::open().unwrap();
    for s in CASES {
        let p = lower(s);
        let run =
            schedule::cuda::run_source_with_device(s, &p, FUEL, &mut device, &compiler).unwrap();
        equal(&run.observation, &mir_eval::observe(&p, FUEL));
        assert_eq!(
            run.executed
                .iter()
                .filter(|r| r.kind == schedule::PlanKind::Indexed)
                .count(),
            1
        );
    }
    let bad = std::str::from_utf8(CASES[1])
        .unwrap()
        .replace("[2u32,0u32,1u32]", "[0u32,0u32,1u32]");
    let p = lower(bad.as_bytes());
    let run =
        schedule::cuda::run_source_with_device(bad.as_bytes(), &p, FUEL, &mut device, &compiler)
            .unwrap();
    assert!(run.executed.is_empty());
    assert!(matches!(
        run.observation.result,
        Err(invar::eval::EvalError::Tool {
            code: "CUDA_UNAVAILABLE",
            ..
        })
    ));
    for source in static_refusals() {
        let source = source.as_bytes();
        let p = lower(source);
        let refused =
            schedule::cuda::run_source_with_device(source, &p, FUEL, &mut device, &compiler)
                .err()
                .expect("static refusal cannot publish");
        assert_eq!(refused.code, "CUDA_UNAVAILABLE");
        let run = schedule::cuda::run_auto_source_with_fault(
            source,
            &p,
            FUEL,
            &compiler,
            invar::plan::DeviceCostModel {
                cpu_bytes_per_ms: 1.,
                device_bytes_per_ms: 1e12,
                device_fixed_ms: 0.,
            },
            schedule::cuda::Fault::PartialCompletion,
        )
        .unwrap();
        equal(&run.observation, &mir_eval::observe(&p, FUEL));
        assert!(run.executed.is_empty());
    }
    // A device failure before publication falls back to serial MIR for
    // this family and keeps the failure as the recorded reason.
    let p = lower(CASES[0]);
    for fault in [
        schedule::cuda::Fault::OutOfMemory,
        schedule::cuda::Fault::LaunchFailure,
    ] {
        let run = schedule::cuda::run_auto_source_with_fault(
            CASES[0],
            &p,
            FUEL,
            &compiler,
            invar::plan::DeviceCostModel {
                cpu_bytes_per_ms: 1.,
                device_bytes_per_ms: 1e12,
                device_fixed_ms: 0.,
            },
            fault,
        )
        .unwrap();
        equal(&run.observation, &mir_eval::observe(&p, FUEL));
        assert!(run.executed.is_empty());
        assert!(
            run.choices
                .iter()
                .any(|c| c.backend == "mir-serial" && c.reason == "cuda-failed-before-publication")
        );
    }
}
