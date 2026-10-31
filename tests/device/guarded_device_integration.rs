//! Issue label M3-05.
use invar::{
    admission, check, device_cache, device_ir, eval, front, mir,
    plan::{DeviceCostModel, Execution},
};
use invar_device::{Capabilities, Provenance};
use std::process::Command;

#[path = "../support/m3_05_programs.rs"]
mod m3_05_programs;
use m3_05_programs::{PAIR, REVERSED_ZIP};

#[path = "../support/scratch.rs"]
mod scratch;

#[test]
fn guarded_device_integration_pair_uses_verified_graph_plan() {
    let checked = check::check(front::parse(PAIR.as_bytes()).unwrap(), true).unwrap();
    let program = mir::lower(&checked).unwrap();
    let report = admission::analyze(PAIR.as_bytes(), &program).unwrap();
    let region = &report.regions()[0];
    assert!(matches!(region.execution(), Execution::FusedProducer(_)));
    let ir = device_ir::from_region(region, 3).unwrap();
    let dag = ir.dag.as_ref().unwrap();
    assert_eq!(dag.inputs.len(), 1);
    assert_eq!(dag.outputs.len(), 2);
    let provenance = Provenance {
        source_identity: ir.source_identity,
        graph_revision: ir.graph_revision,
    };
    let verified = invar_device::verify(ir, provenance, Capabilities::cuda_12_9_sm89()).unwrap();
    assert!(
        invar_device::emit_cuda(&verified)
            .unwrap()
            .contains("invar_map_v1")
    );
    assert_eq!(
        eval::observe(&checked, 100_000).result,
        Ok(eval::Value::I32(0))
    );
}

#[test]
fn guarded_device_integration_reversed_zip_slots_lower_to_device_ir() {
    let checked = check::check(front::parse(REVERSED_ZIP.as_bytes()).unwrap(), true).unwrap();
    let program = mir::lower(&checked).unwrap();
    let report = admission::analyze(REVERSED_ZIP.as_bytes(), &program).unwrap();
    let region = &report.regions()[0];
    assert!(matches!(region.execution(), Execution::FusedProducer(_)));
    let ir = device_ir::from_region(region, 3).unwrap();
    let dag = ir.dag.as_ref().unwrap();
    assert_eq!(dag.inputs.len(), 2);
    assert!(matches!(dag.nodes[0].op, invar_device::DagOp::Input(1)));
    let provenance = Provenance {
        source_identity: ir.source_identity,
        graph_revision: ir.graph_revision,
    };
    invar_device::verify(ir, provenance, Capabilities::cuda_12_9_sm89()).unwrap();
    assert_eq!(
        eval::observe(&checked, 100_000).result,
        Ok(eval::Value::I32(0))
    );
}

#[test]
fn guarded_device_integration_cache_rejects_forged_artifacts_and_policy() {
    let program =
        mir::lower(&check::check(front::parse(PAIR.as_bytes()).unwrap(), true).unwrap()).unwrap();
    let report = admission::analyze(PAIR.as_bytes(), &program).unwrap();
    let region = &report.regions()[0];
    let ir = device_ir::from_region(region, 3).unwrap();
    let origin = Provenance {
        source_identity: ir.source_identity,
        graph_revision: ir.graph_revision,
    };
    let verified = invar_device::verify(ir, origin, Capabilities::cuda_12_9_sm89()).unwrap();
    let artifact = invar_device::emit_cuda(&verified).unwrap();
    let valid = device_cache::expected(PAIR.as_bytes(), region, &verified, &artifact);
    assert!(device_cache::validate(&valid, &valid));
    let mut forged = valid.clone();
    forged["identity"]["artifact_hash"] = "deadbeef".into();
    forged["key"] = "0000000000000000".into();
    assert!(!device_cache::validate(&forged, &valid));
    let mut injected = valid.clone();
    injected["cuda_source"] = "extern \"C\" __global__ void injected() {}".into();
    assert!(!device_cache::validate(&injected, &valid));
    let mut stale = valid.clone();
    stale["geometry"]["lanes"] = 128.into();
    assert!(!device_cache::validate(&stale, &valid));
    let mut version = valid.clone();
    version["schema_version"] = 2.into();
    assert!(!device_cache::validate(&version, &valid));
}

#[test]
fn guarded_device_integration_measured_cost_requires_clear_win() {
    let model = DeviceCostModel {
        cpu_bytes_per_ms: 1_000_000.0,
        device_bytes_per_ms: 10_000_000.0,
        device_fixed_ms: 0.5,
    };
    assert!(!model.estimate(1024, 128, 1, 1).choose_device);
    assert!(model.estimate(64_000_000, 8_000_000, 16, 1).choose_device);
    assert!(
        !DeviceCostModel {
            cpu_bytes_per_ms: f64::NAN,
            ..model
        }
        .estimate(64_000_000, 8_000_000, 16, 1)
        .choose_device
    );
}

#[test]
fn guarded_device_integration_auto_explain_reports_executed_result() {
    let path = std::env::temp_dir().join(format!("invar-m3-05-auto-{}.inv", scratch::unique()));
    std::fs::write(&path, PAIR).unwrap();
    let explain = Command::new(env!("CARGO_BIN_EXE_invar"))
        .args([
            "explain",
            path.to_str().unwrap(),
            "--engine",
            "auto",
            "--format",
            "json",
        ])
        .output()
        .unwrap();
    let run = Command::new(env!("CARGO_BIN_EXE_invar"))
        .args([
            "run",
            path.to_str().unwrap(),
            "--engine",
            "auto",
            "--format",
            "json",
        ])
        .output()
        .unwrap();
    std::fs::remove_file(&path).unwrap();
    assert!(
        explain.status.success(),
        "{}",
        String::from_utf8_lossy(&explain.stderr)
    );
    assert!(
        run.status.success(),
        "{}",
        String::from_utf8_lossy(&run.stderr)
    );
    let explanation: serde_json::Value = serde_json::from_slice(&explain.stdout).unwrap();
    let outcome: serde_json::Value = serde_json::from_slice(&run.stdout).unwrap();
    assert_eq!(
        explanation["device_execution"]["result"],
        outcome["result"]["value"]
    );
    assert!(
        explanation["device_execution"]["choices"]
            .as_array()
            .is_some_and(|choices| !choices.is_empty())
    );
}

#[cfg(not(feature = "cuda"))]
#[test]
fn guarded_device_integration_forced_cuda_never_uses_cpu_without_feature() {
    let path = std::env::temp_dir().join(format!("invar-m3-05-{}.inv", scratch::unique()));
    std::fs::write(&path, PAIR).unwrap();
    let output = Command::new(env!("CARGO_BIN_EXE_invar"))
        .args([
            "run",
            path.to_str().unwrap(),
            "--engine",
            "cuda",
            "--format",
            "json",
        ])
        .output()
        .unwrap();
    std::fs::remove_file(&path).unwrap();
    assert_eq!(output.status.code(), Some(2));
    let document: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(document["status"], "tool_error");
    assert_eq!(document["diagnostics"][0]["code"], "CUDA_UNAVAILABLE");
}
