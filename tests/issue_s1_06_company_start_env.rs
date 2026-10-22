use std::{
    path::Path,
    process::Command,
    sync::atomic::{AtomicU64, Ordering},
};

use invar::{front::parse_system, system};

#[test]
#[ignore = "requires source-bound MCAP metrics and task cells; tolerance repeat acceptance gate"]
fn issue_s1_06_company_start_env_recorded_task_acceptance_both_arms() {
    let cells = std::env::var("INVAR_S1_06_TASK_CELL_ROOT").expect("set task-wired cell directory");
    let evidence =
        std::env::var("INVAR_S1_06_TASK_EVIDENCE_ROOT").expect("set archived metric directory");
    let mut failed = Vec::new();
    for arm in ["panda", "ur5e"] {
        let path = Path::new(&cells).join(format!("cell-task-{arm}.inv"));
        let source = std::fs::read(&path).unwrap();
        let ir =
            system::compile_at(parse_system(&source).unwrap(), path.parent().unwrap()).unwrap();
        let metrics = std::fs::read(Path::new(&evidence).join(format!("{arm}-runs.json"))).unwrap();
        let (report, accepted) = ir.report_with_run_evidence(&metrics).unwrap();
        println!("{report}");
        if !accepted {
            failed.push(arm);
        }
    }
    assert!(
        failed.is_empty(),
        "S1-06 recorded acceptance failed: {}",
        failed.join(", ")
    );
}

#[test]
#[ignore = "requires the pinned S1-06 MuJoCo environment"]
fn issue_s1_06_company_start_env_cell_physics_loads_and_steps() {
    cell_physics_smoke(false);
}

#[test]
#[ignore = "requires pinned MuJoCo environment; strict S1-06 home-hold prerequisite"]
fn issue_s1_06_company_start_env_home_hold_obeys_all_joint_and_contact_limits() {
    cell_physics_smoke(true);
}

#[test]
#[ignore = "requires pinned MuJoCo environment; post-step contact regression"]
fn issue_s1_06_company_start_env_contacts_match_final_integrated_state() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let result = Command::new("python3")
        .arg(root.join("tests/fixtures/s1_06/cell_contacts_regression.py"))
        .output()
        .unwrap();
    assert!(
        result.status.success(),
        "stdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&result.stdout),
        String::from_utf8_lossy(&result.stderr)
    );
    println!("{}", String::from_utf8_lossy(&result.stdout));
}

/// The pinned image (default) or, with `INVAR_S1_06_RUNNER=userspace` and
/// `INVAR_S1_06_ENV_ROOT`, the Route 1 user-space environment (`runner.sh`).
fn ros_environment() -> (String, String) {
    if std::env::var("INVAR_S1_06_RUNNER").as_deref() == Ok("userspace") {
        std::env::var("INVAR_S1_06_ENV_ROOT").expect("set the Route 1 environment root");
        return (String::new(), String::new());
    }
    (
        std::env::var("INVAR_S1_06_ROS_IMAGE").expect("set pinned ROS image tag"),
        std::env::var("INVAR_S1_06_ROS_IMAGE_ID").expect("set image ID"),
    )
}

fn cell_physics_smoke(strict_hold: bool) {
    static NEXT_RUN: AtomicU64 = AtomicU64::new(0);
    let run_id = NEXT_RUN.fetch_add(1, Ordering::Relaxed);
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let mut failures = Vec::new();
    for arm in ["panda", "ur5e"] {
        let name = if arm == "panda" {
            "cell.inv"
        } else {
            "cell-ur5e.inv"
        };
        let path = root.join("examples/cells/reach_place").join(name);
        let source = std::fs::read(&path).unwrap();
        let ir =
            system::compile_at(parse_system(&source).unwrap(), path.parent().unwrap()).unwrap();
        let files = system::generate_mjcf(&ir).unwrap();
        let output =
            std::env::temp_dir().join(format!("invar-s106-{arm}-{}-{run_id}", std::process::id()));
        std::fs::create_dir(&output).unwrap();
        for (relative, bytes) in files {
            let target = output.join(relative);
            std::fs::create_dir_all(target.parent().unwrap()).unwrap();
            std::fs::write(target, bytes).unwrap();
        }
        let document = output.join("cell.json");
        std::fs::write(&document, ir.canonical_json()).unwrap();
        let result = Command::new("python3")
            .arg(root.join("tests/fixtures/s1_06/cell_physics_smoke.py"))
            .arg(output.join("scene.xml"))
            .arg(document)
            .arg(if strict_hold {
                "strict-hold"
            } else {
                "physics"
            })
            .output()
            .unwrap();
        if !result.status.success() {
            failures.push(format!(
                "{arm}: stdout:\n{}\nstderr:\n{}",
                String::from_utf8_lossy(&result.stdout),
                String::from_utf8_lossy(&result.stderr)
            ));
        }
        println!("{arm}: {}", String::from_utf8_lossy(&result.stdout));
        std::fs::remove_dir_all(output).unwrap();
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

/// The S1-04 runtime smoke on this cell with `mujoco_ros2_control`, both arms.
#[test]
#[ignore = "requires the pinned S1-06 ROS 2 Lyrical environment (s104 image or Route 1)"]
fn issue_s1_06_company_start_env_s1_04_mujoco_cell_smoke_both_arms() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let (image, image_id) = ros_environment();
    let run_dir = std::env::var("INVAR_S1_06_SMOKE_DIR").expect("set new evidence directory");
    let result = Command::new("bash")
        .arg(root.join("tests/fixtures/s1_06/cell_smoke.sh"))
        .arg(env!("CARGO_BIN_EXE_invar"))
        .arg(run_dir)
        .arg(image)
        .arg(image_id)
        .output()
        .unwrap();
    println!("{}", String::from_utf8_lossy(&result.stdout));
    assert!(
        result.status.success(),
        "stdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&result.stdout),
        String::from_utf8_lossy(&result.stderr)
    );
}

#[test]
#[ignore = "requires the pinned S1-06 ROS 2 Lyrical environment (s104 image or Route 1)"]
fn issue_s1_06_company_start_env_ur5e_controller_motion() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let (image, image_id) = ros_environment();
    let run_dir = std::env::var("INVAR_S1_06_RUN_DIR").expect("set new evidence directory");
    let result = Command::new("bash")
        .arg(root.join("tests/fixtures/s1_06/ros_smoke.sh"))
        .arg(env!("CARGO_BIN_EXE_invar"))
        .arg(run_dir)
        .arg(image)
        .arg(image_id)
        .output()
        .unwrap();
    assert!(
        result.status.success(),
        "stdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&result.stdout),
        String::from_utf8_lossy(&result.stderr)
    );
}
