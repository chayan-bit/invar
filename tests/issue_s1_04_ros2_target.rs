use std::{
    collections::BTreeMap,
    path::{Path, PathBuf},
    process::Command,
};

use sha2::{Digest, Sha256};

fn fixture() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/s1_03/cell-12.inv")
}

fn generated_files(root: &Path) -> Vec<(PathBuf, Vec<u8>)> {
    fn visit(base: &Path, path: &Path, files: &mut Vec<(PathBuf, Vec<u8>)>) {
        for entry in std::fs::read_dir(path).unwrap() {
            let entry = entry.unwrap();
            let path = entry.path();
            if path.is_dir() {
                visit(base, &path, files);
            } else {
                files.push((
                    path.strip_prefix(base).unwrap().to_owned(),
                    std::fs::read(path).unwrap(),
                ));
            }
        }
    }
    let mut files = Vec::new();
    visit(root, root, &mut files);
    files.sort_by(|a, b| a.0.cmp(&b.0));
    files
}

#[test]
fn issue_s1_04_ros2_target_public_generation_is_deterministic() {
    let nonce = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let root = std::env::temp_dir().join(format!("invar-s1-04-{}-{nonce}", std::process::id()));
    std::fs::create_dir(&root).unwrap();
    let mut results = Vec::new();
    for name in ["first", "second"] {
        let output = root.join(name);
        let run = Command::new(env!("CARGO_BIN_EXE_invar"))
            .args(["system", "gen", "ros2"])
            .arg(fixture())
            .arg("--out")
            .arg(&output)
            .output()
            .unwrap();
        assert!(
            run.status.success(),
            "{}",
            String::from_utf8_lossy(&run.stderr)
        );
        results.push(generated_files(&output));
    }
    assert_eq!(results[0], results[1]);
    let actual: BTreeMap<String, String> = results[0]
        .iter()
        .map(|(path, bytes)| {
            (
                path.to_str().unwrap().to_owned(),
                Sha256::digest(bytes)
                    .iter()
                    .map(|byte| format!("{byte:02x}"))
                    .collect(),
            )
        })
        .collect();
    let golden: BTreeMap<String, String> =
        serde_json::from_str(include_str!("fixtures/s1_04/cell-12-golden-sha256.json")).unwrap();
    assert_eq!(actual, golden, "generated ROS workspace drifted");
    assert!(
        results[0]
            .iter()
            .any(|(path, _)| path.ends_with("QuantityI32M.msg"))
    );
    assert!(results[0].iter().all(|(_, bytes)| {
        !bytes
            .windows(root.to_string_lossy().len())
            .any(|part| part == root.to_string_lossy().as_bytes())
    }));
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn issue_s1_04_ros2_target_parameter_range_uses_versioned_ir() {
    let source = std::fs::read_to_string(
        Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/s1_04/range-valid.inv"),
    )
    .unwrap();
    let program = invar::front::parse_system(source.as_bytes()).unwrap();
    let ir = invar::system::compile(program).unwrap();
    assert_eq!(ir.checked_document().unwrap()["schema"], "invar-system-v12");
    let exposure = &ir.checked_document().unwrap()["components"][0]["params"]["exposure"];
    assert_eq!(exposure["value"], "3");
    assert_eq!(exposure["range"]["min"], "1");
    assert_eq!(exposure["range"]["max"], "5");
    let files = invar::system::ros2::generate(&ir).unwrap();
    let yaml = &files["src/invar_cell_bringup/config/params/sensor.yaml"];
    assert!(yaml.contains("read_only: true"));
    assert!(yaml.contains("bounds<>: [1, 5]"));
}

#[test]
fn issue_s1_04_ros2_target_separate_process_has_own_node() {
    let source = std::fs::read(
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/s1_04/runtime/cell-separate.inv"),
    )
    .unwrap();
    let ir = invar::system::compile(invar::front::parse_system(&source).unwrap()).unwrap();
    let files = invar::system::ros2::generate(&ir).unwrap();
    assert!(files["src/invar_planar2_node/CMakeLists.txt"].contains("add_executable(planar2_node"));
    assert!(
        files["src/invar_cell_bringup/launch/cell.launch.py"]
            .contains("Node(package='invar_planar2_node', executable='planar2_node'")
    );
    assert!(
        !files["src/invar_cell_bringup/launch/cell.launch.py"]
            .contains("actions.append(ComposableNodeContainer")
    );
}

#[test]
fn issue_s1_04_ros2_target_public_errors_leave_no_output() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/s1_04/negative");
    for entry in std::fs::read_dir(root).unwrap() {
        let path = entry.unwrap().path();
        let code = path.file_name().unwrap().to_str().unwrap()[..5].to_owned();
        let output_dir = std::env::temp_dir().join(format!(
            "invar-s1-04-negative-{code}-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        ));
        assert!(
            !output_dir.exists(),
            "stale output: {}",
            output_dir.display()
        );
        let output = Command::new(env!("CARGO_BIN_EXE_invar"))
            .args(["system", "gen", "ros2"])
            .arg(&path)
            .arg("--out")
            .arg(&output_dir)
            .output()
            .unwrap();
        assert_eq!(output.status.code(), Some(1), "{}", path.display());
        let document: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
        assert_eq!(
            document["diagnostics"][0]["code"],
            code,
            "{}",
            path.display()
        );
        assert!(!output_dir.exists(), "partial output: {}", path.display());
    }
}
