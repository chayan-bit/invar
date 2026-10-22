//! Ordinary indexed source routing. Hardware cases must be explicitly requested.
use invar::{check, front, mir, mir_eval, schedule};
const FUEL: u64 = 1_000_000;
const CASES: [&[u8]; 6] = [
    include_bytes!("fixtures/a1-indexed/gather.inv"),
    include_bytes!("fixtures/a1-indexed/scatter.inv"),
    include_bytes!("fixtures/a1-indexed/histogram.inv"),
    include_bytes!("fixtures/a1-indexed/affine.inv"),
    include_bytes!("../bench/general-net-v01/current/frontier_graph_a.inv"),
    include_bytes!("../bench/general-net-v01/current/frontier_graph_b.inv"),
];
fn lower(s: &[u8]) -> mir::Program {
    mir::lower(&check::check(front::parse(s).unwrap(), true).unwrap()).unwrap()
}
fn equal(a: &invar::eval::Observation, b: &invar::eval::Observation) {
    assert_eq!(a.result, b.result);
    assert_eq!(a.steps_used, b.steps_used);
    assert_eq!(a.cleanup_trace, b.cleanup_trace);
}
fn static_refusals() -> Vec<String> {
    let affine = std::str::from_utf8(CASES[3]).unwrap();
    vec![
        affine.replace("for i in 0u32..", "for i in 1u32.."),
        affine.replace("2u32*i", "0u32*i"),
        affine.replace("2u32*i", "4294967295u32*i"),
        affine.replace("filled_u32(5u32", "filled_u32(4u32"),
        affine.replace("filled_u32(5u32", "filled_u32(4097u32"),
        std::str::from_utf8(CASES[2])
            .unwrap()
            .replace("wrapping_add(out[ix[i]],a[i])", "out[ix[i]] + a[i]"),
    ]
}
#[test]
fn a1_indexed_cpu_keeps_serial_source_semantics() {
    for s in CASES {
        let p = lower(s);
        let serial = mir_eval::observe(&p, FUEL);
        let run = schedule::run_source(s, &p, FUEL, 2).unwrap();
        equal(&run.observation, &serial);
        assert!(
            !run.executed
                .iter()
                .any(|r| r.kind == schedule::PlanKind::Indexed)
        );
    }
    for s in static_refusals() {
        let p = lower(s.as_bytes());
        let run = schedule::run_source(s.as_bytes(), &p, FUEL, 2).unwrap();
        equal(&run.observation, &mir_eval::observe(&p, FUEL));
        assert!(run.executed.is_empty());
    }
}

#[cfg(all(feature = "metal", target_os = "macos"))]
mod metal {
    use super::*;
    use invar::{
        plan::DeviceCostModel,
        schedule::metal::{self, Fault},
    };
    const DEVICE: DeviceCostModel = DeviceCostModel {
        cpu_bytes_per_ms: 1.,
        device_bytes_per_ms: 1e12,
        device_fixed_ms: 0.,
    };
    #[test]
    #[ignore = "requires live Metal hardware"]
    fn a1_indexed_metal_completed_outputs_and_host_timings() {
        let mut device = invar_metal::MetalDevice::open_runtime_source().unwrap();
        for (i, s) in CASES.into_iter().enumerate() {
            let p = lower(s);
            let serial = mir_eval::observe(&p, FUEL);
            let begun = std::time::Instant::now();
            let run = metal::run_source_with_device(s, &p, FUEL, &mut device).unwrap();
            let device_ms = begun.elapsed().as_secs_f64() * 1000.;
            equal(&run.observation, &serial);
            assert_eq!(
                run.executed
                    .iter()
                    .filter(|r| r.kind == schedule::PlanKind::Indexed)
                    .count(),
                1
            );
            let begun = std::time::Instant::now();
            let (host, baseline, completed) = match i {
                0 => {
                    let host = invar::indexed_source::host::run_source(s, FUEL, Default::default())
                        .unwrap();
                    (host.observation, "gather-host", host.completed)
                }
                2 => {
                    let host =
                        invar::histogram_source::host::run_source(s, FUEL, Default::default())
                            .unwrap();
                    (host.observation, "histogram-host", host.completed)
                }
                3 => {
                    let host = invar::scatter_proof::host::run_source(s, FUEL, Default::default())
                        .unwrap();
                    (host.observation, "affine-host", host.completed)
                }
                _ => (mir_eval::observe(&p, FUEL), "serial-mir", 0),
            };
            let host_ms = begun.elapsed().as_secs_f64() * 1000.;
            equal(&host, &serial);
            println!(
                "case={i} indexed=1 value={:?} steps={} metal_ms={device_ms:.3} host_ms={host_ms:.3} baseline={baseline} host_completed={completed}",
                run.observation.result, run.observation.steps_used
            );
        }
    }
    #[test]
    #[ignore = "requires live Metal hardware"]
    fn a1_indexed_metal_cost_guard_and_quiescent_fallback() {
        for s in CASES {
            let p = lower(s);
            let serial = mir_eval::observe(&p, FUEL);
            for (model, backend) in [
                (DEVICE, "metal"),
                (
                    DeviceCostModel {
                        cpu_bytes_per_ms: 1e12,
                        device_bytes_per_ms: 1.,
                        device_fixed_ms: 10.,
                    },
                    "mir-serial",
                ),
            ] {
                let run = metal::run_auto_source_with_fault(
                    s,
                    &p,
                    FUEL,
                    metal::Kernels::RuntimeSource,
                    model,
                    Fault::None,
                )
                .unwrap();
                equal(&run.observation, &serial);
                assert!(run.choices.iter().any(|c| c.backend == backend));
            }
        }
        let s = CASES[0];
        let p = lower(s);
        let serial = mir_eval::observe(&p, FUEL);
        for fault in [Fault::OutOfMemory, Fault::LaunchFailure] {
            let run = metal::run_auto_source_with_fault(
                s,
                &p,
                FUEL,
                metal::Kernels::RuntimeSource,
                DEVICE,
                fault,
            )
            .unwrap();
            equal(&run.observation, &serial);
            assert!(run.executed.is_empty());
            // The serial fallback keeps the device failure as its reason.
            assert!(run.choices.iter().any(
                |c| c.backend == "mir-serial" && c.reason == "metal-failed-before-publication"
            ));
        }
        let bad = String::from_utf8(CASES[1].to_vec())
            .unwrap()
            .replace("[2u32,0u32,1u32]", "[0u32,0u32,1u32]");
        let p = lower(bad.as_bytes());
        let run = metal::run_auto_source_with_fault(
            bad.as_bytes(),
            &p,
            FUEL,
            metal::Kernels::RuntimeSource,
            DEVICE,
            Fault::None,
        )
        .unwrap();
        equal(&run.observation, &mir_eval::observe(&p, FUEL));
        assert!(run.executed.is_empty());
        assert!(run.choices.iter().any(|c| c.reason == "live-guard-false"));
    }
    #[test]
    #[ignore = "requires live Metal hardware"]
    fn a1_indexed_metal_failures_never_publish_or_replay_pending() {
        let s = CASES[0];
        let p = lower(s);
        for fault in [
            Fault::OutOfMemory,
            Fault::LaunchFailure,
            Fault::PartialCompletion,
        ] {
            let mut device = invar_metal::MetalDevice::open_runtime_source().unwrap();
            let run = metal::run_source_with_device_fault(s, &p, FUEL, &mut device, fault).unwrap();
            assert!(run.executed.is_empty());
            let expected = if fault == Fault::PartialCompletion {
                "METAL_PENDING"
            } else {
                "METAL_UNAVAILABLE"
            };
            assert!(
                matches!(run.observation.result,Err(invar::eval::EvalError::Tool {code,..}) if code==expected)
            );
            if fault == Fault::PartialCompletion {
                assert!(device.is_poisoned());
                assert!(device.quarantine_len() > 0);
            }
        }
        let run = metal::run_auto_source_with_fault(
            s,
            &p,
            FUEL,
            metal::Kernels::RuntimeSource,
            DEVICE,
            Fault::PartialCompletion,
        )
        .unwrap();
        assert!(run.executed.is_empty());
        assert!(matches!(
            run.observation.result,
            Err(invar::eval::EvalError::Tool {
                code: "METAL_PENDING",
                ..
            })
        ));
    }
    #[test]
    #[ignore = "requires live Metal hardware"]
    fn a1_indexed_metal_unreached_zero_and_forced_guard_refusals() {
        let base = std::str::from_utf8(CASES[0]).unwrap();
        let function = base.split("fn main()").next().unwrap();
        let zero = format!(
            "{function}fn main() -> i32 {{ let a=filled_u32(0u32,0u32); let ix=filled_u32(0u32,0u32); let out=work(a,ix); return 7; }}"
        );
        let unreached = format!("{function}fn main() -> i32 {{ return 7; }}");
        for s in [
            zero,
            unreached,
            base.replace("[2u32,0u32,1u32]", "[3u32,4u32,1u32]"),
        ]
        .into_iter()
        .chain(static_refusals())
        {
            let p = lower(s.as_bytes());
            let mut device = invar_metal::MetalDevice::open_runtime_source().unwrap();
            match metal::run_source_with_device_fault(
                s.as_bytes(),
                &p,
                FUEL,
                &mut device,
                Fault::PartialCompletion,
            ) {
                Err(e) => assert_eq!(e.code, "METAL_UNAVAILABLE"),
                Ok(run) => {
                    assert!(run.executed.is_empty());
                    assert!(matches!(
                        run.observation.result,
                        Err(invar::eval::EvalError::Tool {
                            code: "METAL_UNAVAILABLE",
                            ..
                        })
                    ));
                }
            }
            let run = metal::run_auto_source_with_fault(
                s.as_bytes(),
                &p,
                FUEL,
                metal::Kernels::RuntimeSource,
                DEVICE,
                Fault::None,
            )
            .unwrap();
            equal(&run.observation, &mir_eval::observe(&p, FUEL));
            assert!(run.executed.is_empty());
            assert!(!device.is_poisoned());
        }
    }
    #[test]
    #[ignore = "requires live Metal hardware and the Metal CLI binary"]
    fn a1_indexed_metal_cli_routes_and_matches_reference() {
        let paths = [
            "tests/fixtures/a1-indexed/gather.inv",
            "tests/fixtures/a1-indexed/scatter.inv",
            "tests/fixtures/a1-indexed/histogram.inv",
            "tests/fixtures/a1-indexed/affine.inv",
            "bench/general-net-v01/current/frontier_graph_a.inv",
            "bench/general-net-v01/current/frontier_graph_b.inv",
        ];
        for path in paths {
            let mut expected = None;
            for engine in ["reference", "metal", "auto", "cuda"] {
                let output = std::process::Command::new(env!("CARGO_BIN_EXE_invar"))
                    .env("INVAR_METAL_RUNTIME_SOURCE", "1")
                    .args(["run", path, "--engine", engine, "--format", "json"])
                    .output()
                    .unwrap();
                let document: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
                println!("invar run {path} --engine {engine} --format json\n{document}");
                if engine == "cuda" {
                    assert_eq!(document["status"], "tool_error");
                    assert_eq!(document["diagnostics"][0]["code"], "CUDA_UNAVAILABLE");
                } else {
                    assert!(output.status.success(), "{document}");
                    if let Some(expected) = &expected {
                        assert_eq!(&document["result"], expected);
                    } else {
                        expected = Some(document["result"].clone());
                    }
                }
            }
            let output = std::process::Command::new(env!("CARGO_BIN_EXE_invar"))
                .env("INVAR_METAL_RUNTIME_SOURCE", "1")
                .args(["explain", path, "--engine", "metal", "--format", "json"])
                .output()
                .unwrap();
            let document: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
            assert!(output.status.success(), "{document}");
            assert!(document.to_string().contains("Indexed"), "{document}");
            println!("invar explain {path} --engine metal --format json\n{document}");
        }
    }
}

#[cfg(feature = "cuda")]
mod cuda {
    use super::*;
    #[test]
    #[ignore = "requires CUDA hardware and the NVRTC helper"]
    fn a1_indexed_cuda_completed_outputs_and_forced_refusals() {
        let compiler =
            std::path::PathBuf::from(std::env::var_os("INVAR_NVRTC_HELPER").expect("NVRTC helper"));
        let mut device = invar_cuda::cuda::CudaDevice::open().unwrap();
        for s in CASES {
            let p = lower(s);
            let run = schedule::cuda::run_source_with_device(s, &p, FUEL, &mut device, &compiler)
                .unwrap();
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
        let run = schedule::cuda::run_source_with_device(
            bad.as_bytes(),
            &p,
            FUEL,
            &mut device,
            &compiler,
        )
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
                    .any(|c| c.backend == "mir-serial"
                        && c.reason == "cuda-failed-before-publication")
            );
        }
    }
}
