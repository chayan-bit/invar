//! Issue label GNET-04.
#![forbid(unsafe_code)]

use std::{fs, process::Command};
#[test]
fn ordinary_source_command_uses_runtime_inputs_and_labels_cpu_model_receipts() {
    let source = std::env::temp_dir().join(format!("invar-gnet-cli-{}.inv", std::process::id()));
    fs::write(&source,b"fn leaf(x:u32)->u32{return wrapping_add(x,1u32);} fn entry(x:u32)->u32{let a=leaf(x);let b=leaf(x);return wrapping_add(a,b);}").unwrap();
    let config = r#"{"fuel":1000,"transitions":10000,"trace_events":2000,"quantum":2,"rounds":1,"capacity":2}"#;
    for (input, expected) in [("[5]", 12), ("[19]", 40)] {
        let output = Command::new(env!("CARGO_BIN_EXE_invar"))
            .args([
                "gnet-device",
                source.to_str().unwrap(),
                "entry",
                "cpu-model-cuda",
                input,
                config,
            ])
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stdout)
        );
        let report: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
        assert_eq!(report["result"], expected);
        assert_eq!(report["status"], 0);
        assert_eq!(report["execution"]["evidence_backend"], "CpuGroupModel");
        assert_eq!(
            report["execution"]["scope"],
            "adapter_observed_fenced_device_checkpoint"
        );
        assert_eq!(report["execution"]["buffers"].as_array().unwrap().len(), 12);
        assert_eq!(report["submitted"], report["completed"]);
        assert!(report["submitted"].as_u64().unwrap() > 1);
    }
    let bad = Command::new(env!("CARGO_BIN_EXE_invar"))
        .args([
            "gnet-device",
            source.to_str().unwrap(),
            "entry",
            "cpu-model-cuda",
            "[-1]",
            config,
        ])
        .output()
        .unwrap();
    assert!(!bad.status.success());
    let report: serde_json::Value = serde_json::from_slice(&bad.stdout).unwrap();
    assert!(
        report["error"]
            .as_str()
            .unwrap()
            .contains("typed argument/layout mismatch")
    );
    #[cfg(not(feature = "cuda"))]
    {
        let forced = Command::new(env!("CARGO_BIN_EXE_invar"))
            .args([
                "gnet-device",
                source.to_str().unwrap(),
                "entry",
                "cuda",
                "[5]",
                config,
            ])
            .output()
            .unwrap();
        assert!(!forced.status.success());
        let report: serde_json::Value = serde_json::from_slice(&forced.stdout).unwrap();
        assert!(
            report["error"]
                .as_str()
                .unwrap()
                .contains("forced CUDA requires")
        );
        assert!(report.get("result").is_none());
    }
    fs::remove_file(source).unwrap();
}

/// `gnet-device` compiles with the certified limits of the device suites
/// (`Limits::certified()`, eight speculative slots), so a runtime capacity
/// up to eight is accepted and a wider one is refused.
#[test]
fn gnet_device_takes_the_certified_speculation_limits() {
    let source =
        std::env::temp_dir().join(format!("invar-gnet-cli-slots-{}.inv", std::process::id()));
    fs::write(&source, b"fn leaf(x:u32)->u32{return wrapping_add(x,1u32);} fn entry(x:u32)->u32{let a=leaf(x);let b=leaf(x);let c=leaf(x);let d=leaf(x);return wrapping_add(wrapping_add(a,b),wrapping_add(c,d));}").unwrap();
    let run = |capacity: usize| {
        let config = format!(
            r#"{{"fuel":1000,"transitions":10000,"trace_events":2000,"quantum":2,"rounds":1,"capacity":{capacity}}}"#
        );
        let output = Command::new(env!("CARGO_BIN_EXE_invar"))
            .args([
                "gnet-device",
                source.to_str().unwrap(),
                "entry",
                "cpu-model-cuda",
                "[5]",
                &config,
            ])
            .output()
            .unwrap();
        let report: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
        (output.status.success(), report)
    };
    assert_eq!(
        invar::general_net::device_host::Limits::certified()
            .private
            .slots,
        8
    );
    for capacity in [2, 8] {
        let (ok, report) = run(capacity);
        assert!(ok, "capacity {capacity}: {report}");
        assert_eq!(report["result"], 24, "capacity {capacity}");
        assert_eq!(report["status"], 0);
    }
    let (ok, report) = run(9);
    assert!(!ok, "{report}");
    assert!(report.get("result").is_none_or(|r| r != 24), "{report}");
    fs::remove_file(source).unwrap();
}
