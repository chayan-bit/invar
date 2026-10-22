#![forbid(unsafe_code)]
use std::{fs, process::Command};
const TRANSPOSE: &str = "fn transpose(a:read Array<u32>,rows:u32,cols:u32)->Array<u32>{var b=filled_u32(rows*cols,0u32);for r in 0u32..rows{for c in 0u32..cols{b[c*rows+r]=a[r*cols+c];}}return b;} fn main()->i32{let a:Array<u32> = [3u32,7u32,11u32,13u32,17u32,19u32];let b=transpose(a,2u32,3u32);if b[0u32]==3u32 and b[1u32]==13u32 and b[2u32]==7u32 and b[3u32]==17u32 and b[4u32]==11u32 and b[5u32]==19u32{return 7;}return 0;}";
const TENSOR: &str = "fn mat(a:read Array<f32>,b:read Array<f32>,rows:u32,inner:u32,cols:u32)->Array<f32>{var c=filled_f32(rows*cols,0.0);for i in 0u32..rows{for j in 0u32..cols{var sum:f32=0.0;for k in 0u32..inner{sum=sum+a[i*inner+k]*b[k*cols+j];}c[i*cols+j]=sum;}}return c;} fn main()->i32{let a:Array<f32> = [1.0,2.0];let b:Array<f32> = [3.0,4.0];let c=mat(a,b,1u32,2u32,1u32);if c[0u32]==11.0{return 7;}return 0;}";
fn invoke(
    source: &str,
    family: &str,
    entry: &str,
    backend: &str,
    fuel: &str,
    suffix: &str,
) -> (bool, serde_json::Value) {
    let path = std::env::temp_dir().join(format!(
        "invar-source-device-cli-{}-{suffix}.inv",
        std::process::id()
    ));
    fs::write(&path, source).unwrap();
    let output = Command::new(env!("CARGO_BIN_EXE_invar"))
        .args([
            "source-device",
            path.to_str().unwrap(),
            family,
            entry,
            backend,
            fuel,
        ])
        .output()
        .unwrap();
    fs::remove_file(path).unwrap();
    (
        output.status.success(),
        serde_json::from_slice(&output.stdout).unwrap(),
    )
}
#[test]
fn source_device_command_runs_real_source_and_labels_model_completion() {
    for (family, entry, source) in [
        ("transpose", "transpose", TRANSPOSE),
        ("tensor", "mat", TENSOR),
    ] {
        let (ok, report) = invoke(source, family, entry, "cpu-model", "100000", family);
        assert!(ok, "{report}");
        assert_eq!(report["result"], 7);
        assert_eq!(report["backend"], "cpu-model");
        assert_eq!(report["completed_regions"], 1);
        assert_eq!(report["requested_entry"], entry);
        assert_eq!(report["source_sha256"].as_str().unwrap().len(), 64);
        assert!(report.get("gpu_execution_claim").is_none());
        let (ok, report) = invoke(source, family, entry, "cpu-model", "1", family);
        assert!(!ok);
        assert_eq!(report["completed_regions"], 0);
        assert!(
            report["error"]
                .as_str()
                .unwrap()
                .contains("EvaluationLimit")
        );
    }
}
#[test]
fn source_device_invalid_source_entry_protocol_fail_before_gpu_open() {
    for (source, family, entry, fuel) in [
        ("fn main()->i32{return nope;}", "tensor", "mat", "1000"),
        (TENSOR, "tensor", "unknown", "1000"),
        (TRANSPOSE, "wrong", "transpose", "1000"),
        (TRANSPOSE, "transpose", "transpose", "0"),
    ] {
        let (ok, report) = invoke(source, family, entry, "cuda", fuel, "invalid");
        assert!(!ok, "{report}");
        assert!(report["error"].is_string());
        assert!(report.get("result").is_none());
        assert!(!report["error"].as_str().unwrap().contains("forced CUDA:"));
    }
}
