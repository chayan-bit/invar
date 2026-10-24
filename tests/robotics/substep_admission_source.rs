//! Source/checker and CLI refusal only; no native physics or ROS execution.
use std::{
    fs,
    path::PathBuf,
    process::{Command, Stdio},
};

struct Files {
    input: PathBuf,
    output: PathBuf,
}
impl Drop for Files {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.input);
        let _ = fs::remove_dir_all(&self.output);
    }
}
fn files(arm: usize, target: &str) -> Files {
    let original = if arm == 7 {
        include_str!("../../examples/cells/reach_place/cell.inv")
    } else {
        include_str!("../../examples/cells/reach_place/cell-ur5e.inv")
    };
    let start = original.find("component mujoco_sim {").unwrap();
    let end = start + original[start..].find("\n}").unwrap() + 2;
    let clause = include_str!("../fixtures/substep_admission/policy-clause.inv")
        .replace("invar::mujoco_substep_admission_v1", target);
    let component = original[start..end].replace("    impl invar run;", clause.trim_end());
    let source = format!("{}{}{}", &original[..start], component, &original[end..]);
    let nonce = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let name = format!("substep-admission-{}-{arm}-{nonce}", std::process::id());
    let input = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("examples/cells/reach_place")
        .join(format!("{name}.inv"));
    let output = std::env::temp_dir().join(name);
    fs::write(&input, source).unwrap();
    Files { input, output }
}

#[test]
fn ordinary_source_check_accepts_exact_policy_and_ros_refuses_before_output() {
    for arm in [6, 7] {
        let paths = files(arm, "invar::mujoco_substep_admission_v1");
        let check = Command::new(env!("CARGO_BIN_EXE_invar"))
            .args(["system", "check"])
            .arg(&paths.input)
            .args(["--format", "json"])
            .output()
            .unwrap();
        assert!(
            check.status.success(),
            "{}",
            String::from_utf8_lossy(&check.stderr)
        );
        let generation = Command::new(env!("CARGO_BIN_EXE_invar"))
            .args(["system", "gen", "ros2"])
            .arg(&paths.input)
            .arg("--out")
            .arg(&paths.output)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .unwrap();
        let temporary_prefix = format!(".invar-ros2-{}-", generation.id());
        let generation = generation.wait_with_output().unwrap();
        assert_eq!(generation.status.code(), Some(1));
        let document: serde_json::Value = serde_json::from_slice(&generation.stdout).unwrap();
        assert_eq!(document["diagnostics"][0]["code"], "S1504");
        let error = document["diagnostics"][0]["message"].as_str().unwrap();
        assert!(
            error.contains("native host adapter unavailable; policy is not active"),
            "{error}"
        );
        assert!(!paths.output.exists());
        let parent = paths.output.parent().unwrap();
        assert!(!fs::read_dir(parent).unwrap().any(|entry| {
            entry
                .unwrap()
                .file_name()
                .to_string_lossy()
                .starts_with(&temporary_prefix)
        }));
    }
}

#[test]
fn unknown_external_target_retains_manifest_resolution_error() {
    for target in [
        "invar::mujoco_substep_admission_v2",
        "invar::mujoco_substep_admission_v1_extra",
    ] {
        let paths = files(7, target);
        let result = Command::new(env!("CARGO_BIN_EXE_invar"))
            .args(["system", "check"])
            .arg(&paths.input)
            .args(["--format", "json"])
            .output()
            .unwrap();
        assert_eq!(result.status.code(), Some(1));
        let document: serde_json::Value = serde_json::from_slice(&result.stdout).unwrap();
        assert_eq!(document["diagnostics"][0]["code"], "S1001");
        let error = document["diagnostics"][0]["message"].as_str().unwrap();
        assert!(error.contains("extern component manifest"), "{error}");
        assert!(!paths.output.exists());
    }
}

/// P1: `system gen mjcf` refuses a policy source with S1504 and writes neither
/// the output directory nor any temporary; `system gen urdf` keeps S1607.
#[test]
fn system_gen_mjcf_and_urdf_refuse_policy_sources_without_output() {
    for arm in [6, 7] {
        for (kind, code) in [("mjcf", "S1504"), ("urdf", "S1607")] {
            let paths = files(arm, "invar::mujoco_substep_admission_v1");
            let generation = Command::new(env!("CARGO_BIN_EXE_invar"))
                .args(["system", "gen", kind])
                .arg(&paths.input)
                .arg("--out")
                .arg(&paths.output)
                .stdout(Stdio::piped())
                .stderr(Stdio::piped())
                .spawn()
                .unwrap();
            let temporary_prefix = format!(".invar-{kind}-{}-", generation.id());
            let generation = generation.wait_with_output().unwrap();
            assert_eq!(generation.status.code(), Some(1), "{kind} arm {arm}");
            let document: serde_json::Value = serde_json::from_slice(&generation.stdout).unwrap();
            assert_eq!(document["diagnostics"][0]["code"], code, "{kind} arm {arm}");
            if kind == "mjcf" {
                let error = document["diagnostics"][0]["message"].as_str().unwrap();
                assert!(
                    error.contains("native host adapter unavailable; policy is not active"),
                    "{error}"
                );
            }
            assert!(!paths.output.exists(), "{kind} arm {arm}");
            let parent = paths.output.parent().unwrap();
            assert!(!fs::read_dir(parent).unwrap().any(|entry| {
                entry
                    .unwrap()
                    .file_name()
                    .to_string_lossy()
                    .starts_with(&temporary_prefix)
            }));
        }
    }
}
