use std::{path::Path, process::Command};

#[test]
#[ignore = "requires the pinned ARM64 ROS 2 Lyrical image"]
fn issue_s1_04_ros2_target_env_build_hash_launch_and_smoke() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let image = std::env::var("INVAR_S1_04_ROS_IMAGE").expect("set pinned ROS image tag");
    let image_id = std::env::var("INVAR_S1_04_ROS_IMAGE_ID").expect("set image ID");
    let run_dir = std::env::var("INVAR_S1_04_RUN_DIR").expect("set a new evidence directory");
    let output = Command::new("bash")
        .arg(root.join("tests/fixtures/s1_04/runtime/env.sh"))
        .arg(env!("CARGO_BIN_EXE_invar"))
        .arg(run_dir)
        .arg(image)
        .arg(image_id)
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "stdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
}
