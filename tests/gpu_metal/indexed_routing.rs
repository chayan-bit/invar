//! Indexed source routing on a real Metal device; the CPU routing tests stay in `tests/device/indexed_routing.rs`.
use invar::{mir_eval, schedule};

#[path = "../support/a1_indexed_common.rs"]
mod a1_indexed_common;
use a1_indexed_common::{CASES, FUEL, equal, lower, static_refusals};

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
fn indexed_routing_metal_completed_outputs_and_host_timings() {
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
                let host =
                    invar::indexed_source::host::run_source(s, FUEL, Default::default()).unwrap();
                (host.observation, "gather-host", host.completed)
            }
            2 => {
                let host =
                    invar::histogram_source::host::run_source(s, FUEL, Default::default()).unwrap();
                (host.observation, "histogram-host", host.completed)
            }
            3 => {
                let host =
                    invar::scatter_proof::host::run_source(s, FUEL, Default::default()).unwrap();
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
fn indexed_routing_metal_cost_guard_and_quiescent_fallback() {
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
fn indexed_routing_metal_failures_never_publish_or_replay_pending() {
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
fn indexed_routing_metal_unreached_zero_and_forced_guard_refusals() {
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
fn indexed_routing_metal_cli_routes_and_matches_reference() {
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
