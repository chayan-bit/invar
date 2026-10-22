//! CPU preparation plus an explicit, separately selected pinned ROS build gate.
#[path = "support/planner_ros_build.rs"]
mod support;

use serde_json::{Value, json};
use std::{fs, path::Path, process::Command};

#[test]
fn issue_s1_06_planner_ros_build_preparation_and_hash_rejection() {
    let root = std::env::temp_dir().join(format!("invar-ros-build-prep-{}", std::process::id()));
    fs::create_dir(&root).unwrap();
    for n in [6, 7] {
        let files = support::workspace(n);
        assert_eq!(files, support::workspace(n));
        assert!(files.contains_key("src/invar_planner_node/src/node.cpp"));
        assert!(files.contains_key("src/invar_controller_bridge/src/main.cpp"));
        support::write_workspace(&root.join(n.to_string()), &files);
        for expected in support::expected(&files, n).values() {
            let mut built = json!({"type_hashes": [
                {"type_name":"unrelated/msg/Type", "hash_string":"wrong"}, expected
            ]});
            support::compare(expected, &built).unwrap();
            built["type_hashes"][1]["hash_string"] = json!("RIHS01_wrong");
            assert!(support::compare(expected, &built).is_err());
            built["type_hashes"] = json!([expected, expected]);
            assert!(support::compare(expected, &built).is_err());
            built["type_hashes"] = json!([]);
            assert!(support::compare(expected, &built).is_err());
            assert!(support::compare(expected, &json!({})).is_err());
            let mut wrong_name = expected.clone();
            wrong_name["type_name"] = json!("wrong/msg/Type");
            assert!(support::compare(expected, &json!({"type_hashes":[wrong_name]})).is_err());
        }
    }
    fs::remove_dir_all(root).unwrap();
}

#[test]
#[ignore = "requires an explicitly pinned, pre-provisioned Linux ROS 2 Lyrical image; no GPU"]
fn issue_s1_06_planner_ros_build_env_both_counts_and_hash_parity() {
    let image = std::env::var("INVAR_S1_06_ROS_IMAGE_ID").expect("set immutable local image ID");
    let pin = image
        .strip_prefix("sha256:")
        .expect("image ID must be sha256:...");
    assert!(pin.len() == 64 && pin.bytes().all(|b| b.is_ascii_hexdigit()));
    let root =
        std::env::var_os("INVAR_S1_06_ROS_BUILD_DIR").expect("set new absolute evidence directory");
    let root = Path::new(&root);
    assert!(root.is_absolute());
    fs::create_dir(root).unwrap();
    let mut expected = Vec::new();
    for n in [6, 7] {
        let files = support::workspace(n);
        support::write_workspace(&root.join(n.to_string()), &files);
        expected.push((n, support::expected(&files, n), files));
    }
    let script =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/s1_06/planner_ros_build.sh");
    let status = Command::new("bash")
        .arg(script)
        .arg(root)
        .arg(&image)
        .status()
        .unwrap();
    assert!(
        status.success(),
        "pinned ROS build failed; inspect evidence directory"
    );
    let mut records = Vec::new();
    for (n, messages, files) in expected {
        for (relative, content) in files {
            assert_eq!(
                fs::read(root.join(n.to_string()).join(&relative)).unwrap(),
                content.as_bytes(),
                "generated input changed during build: {relative}"
            );
        }
        for (name, hash) in messages {
            let path = root.join(format!("{n}/build/invar_cell_interfaces/rosidl_generator_type_description/invar_cell_interfaces/msg/{name}.json"));
            let bytes = fs::read(&path).unwrap();
            let built: Value = serde_json::from_slice(&bytes).unwrap();
            support::compare(&hash, &built).unwrap_or_else(|error| panic!("{name}: {error}"));
            records.push(json!({"type_name": hash["type_name"], "hash_string":hash["hash_string"], "ros_description_sha256":support::sha256(&bytes)}));
        }
    }
    assert_eq!(records.len(), 6);
    fs::write(
        root.join("hash-parity.json"),
        serde_json::to_vec_pretty(&json!({
            "status":"pass", "scope":"pinned_ros_build_and_hash_parity_only",
            "image_id":image, "host_os":std::env::consts::OS, "host_arch":std::env::consts::ARCH,
            "messages":records, "planner_execution":"not_run", "s1_06_acceptance":"pending"
        }))
        .unwrap(),
    )
    .unwrap();
}
