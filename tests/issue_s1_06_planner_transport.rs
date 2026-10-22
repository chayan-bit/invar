//! Real INVP transport with an explicit CPU-only child double; no cuRobo evidence.
#[path = "support/executable.rs"]
mod executable;
use std::{fs, path::Path, process::Command};

use invar_foreign::component::planner::{Config, Planner, PlannerState, Request};
use invar_foreign::component::{ComponentError, Pose};

fn pose() -> Pose {
    Pose {
        translation_m: [0.0; 3],
        quaternion_xyzw: [0.0, 0.0, 0.0, 1.0],
    }
}

fn exercise<const N: usize>() {
    let repo = Path::new(env!("CARGO_MANIFEST_DIR"));
    let root = std::env::temp_dir().join(format!(
        "invar-planner-transport-{}-{N}",
        std::process::id()
    ));
    fs::create_dir(&root).unwrap();
    let source = include_str!("fixtures/s1_06/planner-manifest.inv")
        .replace("f64,7", &format!("f64,{N}"))
        .replace("request7_", &format!("request{N}_"))
        .replace("trajectory7_", &format!("trajectory{N}_"));
    let manifest =
        invar::system::compile_manifest(invar::front::parse_system(source.as_bytes()).unwrap())
            .unwrap();
    let header = invar::system::foreign::header(&manifest).unwrap();
    fs::write(root.join("invar_curobo_bridge_v1.h"), header).unwrap();
    let component = repo.join("examples/cells/reach_place/components/curobo_planner");
    let library = root.join(if cfg!(target_os = "macos") {
        "bridge.dylib"
    } else {
        "bridge.so"
    });
    let compiled = Command::new("cc")
        .args([
            "-std=c11",
            "-O2",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-pedantic",
            "-fPIC",
            "-pthread",
        ])
        .arg(if cfg!(target_os = "macos") {
            "-dynamiclib"
        } else {
            "-shared"
        })
        .arg("-I")
        .arg(&root)
        .arg(component.join("bridge.c"))
        .arg(component.join("planner_reference.c"))
        .arg("-lm")
        .arg("-o")
        .arg(&library)
        .output()
        .unwrap();
    assert!(
        compiled.status.success(),
        "{}",
        String::from_utf8_lossy(&compiled.stderr)
    );
    let yaml = root.join("fixture.yml");
    fs::write(&yaml, "{}").unwrap();
    let names: Vec<_> = (0..N).map(|i| format!("joint{i}")).collect();
    let names: Vec<_> = names.iter().map(String::as_str).collect();
    let limits = [[-2.0, 2.0]; N];
    let speeds = [2.0; N];
    let config = Config {
        model_sha256: [0; 32],
        robot_config_sha256: [0; 32],
        scene_sha256: [0; 32],
        joint_limits_rad: &limits,
        velocity_limits_rad_s: &speeds,
        table_pose: pose(),
        table_half_extents_m: [0.5, 0.5, 0.05],
        cube_edge_m: 0.05,
        base_frame: "base",
        tool_frame: "tool",
        joint_names: &names,
        robot_config_path: &yaml,
        scene_config_path: &yaml,
    };
    let request = Request {
        start_q_rad: [0.0; N],
        goal_q_rad: [0.0; N],
        target_base_from_tool: pose(),
        cube_base_pose: pose(),
        tool_from_cube: pose(),
        phase: 0,
    };
    let child_double = repo.join("tests/fixtures/foreign-components/curobo_planner/fake_server.py");
    let child_double = serde_json::to_string(child_double.to_str().unwrap()).unwrap();
    for (mode, expected_status) in [
        ("normal", None),
        ("reply_start_edge", None),
        ("reply_bad_start", Some(6)),
        ("crash", Some(4)),
        ("bad_reply", Some(6)),
        ("hang", Some(5)),
    ] {
        // Per-child environment only; never mutate the multithreaded test process's environment.
        let launcher = root.join(format!("child-{mode}.py"));
        executable::write(&launcher, format!("#!/usr/bin/env python3\nimport os, runpy\nos.environ['INVAR_PLANNER_FAKE_MODE'] = {mode:?}\nrunpy.run_path({child_double}, run_name='__main__')\n").as_bytes()).unwrap();
        let mut planner = Planner::<N>::open(&library, "invar_curobo_bridge_v1").unwrap();
        planner.init(&launcher).unwrap();
        planner.configure(&config).unwrap();
        let result = planner.plan(&request);
        match expected_status {
            None => {
                let trajectory = result.unwrap();
                assert_eq!(trajectory.waypoints().len(), 2);
                assert_eq!(trajectory.waypoints()[1].q_rad, [0.0; N]);
                planner.shutdown().unwrap();
            }
            Some(expected) => {
                assert!(
                    matches!(result, Err(ComponentError::Status(actual)) if actual == expected),
                    "mode {mode}"
                );
                assert_eq!(planner.state(), PlannerState::Failed);
                assert!(matches!(
                    planner.plan(&request),
                    Err(ComponentError::Lifecycle)
                ));
                assert!(matches!(planner.shutdown(), Err(ComponentError::Status(4))));
            }
        }
        assert_eq!(planner.state(), PlannerState::Shutdown);
    }
    fs::remove_dir_all(root).unwrap();
}

#[test]
fn issue_s1_06_planner_generated_bridge_exercises_real_invp_transport() {
    exercise::<6>();
    exercise::<7>();
}
