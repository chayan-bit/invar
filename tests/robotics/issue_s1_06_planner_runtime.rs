use std::{fs, path::PathBuf, process::Command};

use invar_foreign::component::planner::{Config, Planner, PlannerState as State, Request};
use invar_foreign::component::{Component, ComponentError, Pose};

fn pose() -> Pose {
    Pose {
        translation_m: [0.0; 3],
        quaternion_xyzw: [0.0, 0.0, 0.0, 1.0],
    }
}

fn library<const N: usize>() -> (PathBuf, PathBuf) {
    let root =
        std::env::temp_dir().join(format!("invar-planner-runtime-{}-{N}", std::process::id()));
    fs::create_dir(&root).unwrap();
    fs::write(
        root.join("wire.h"),
        invar::robotics::planner::header(N as u32).unwrap(),
    )
    .unwrap();
    fs::write(
        root.join("fixture.c"),
        format!(
            r#"
#include "wire.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>
const char *probe_version(void) {{ return "invar-planner-component-v2/{N}/fixture"; }}
int probe_init(void **context, const unsigned char *child, unsigned int size) {{
    (void)child; (void)size; *context = calloc(1, sizeof(int)); return *context ? 0 : 3;
}}
int probe_configure(void *context, const void *config) {{
    if (!context || *(int*)context) return 2;
    const inv_planner_config_wire_v1 *cfg = config;
    if (!cfg || cfg->robot_kind != {N} || cfg->joint_count != {N}
        || cfg->model_sha256[0] != 1 || cfg->robot_config_sha256[0] != 2 || cfg->scene_sha256[0] != 3
        || cfg->table_pose.quaternion_xyzw[3] != 1.0 || cfg->table_half_extents_m[0] != 0.5
        || cfg->cube_edge_m != 0.05 || cfg->base_frame.length != 4
        || memcmp(cfg->base_frame.bytes, "base", 4) || cfg->tool_frame.length != 4
        || cfg->joint_names[0].length != 6 || memcmp(cfg->joint_names[0].bytes, "joint0", 6)
        || !cfg->robot_config_path.length || !cfg->scene_config_path.length
        || cfg->joint_limits_rad[{N}-1][0] != -2.0 || cfg->joint_limits_rad[{N}-1][1] != 2.0) return 1;
    *(int*)context = 1; return 0;
}}
int probe_step(void *context, const void *input, void *output) {{
    if (!context || *(int*)context != 1) return 2;
    const inv_planner_request{N}_f64_v2 *request = input;
    inv_planner_trajectory{N}_f64_v1 *result = output;
    memset(result, 0, sizeof(*result));
    result->dt_s = 0.1; result->count = 2; result->waypoints[1].time_from_start_s = 0.1;
    double fault = request->start_q_rad[0];
    if (fault == 0.5) result->count = 257;
    if (fault == 0.6) result->waypoints[1].q_rad[0] = NAN;
    if (fault == 0.7) result->waypoints[1].time_from_start_s = 0.0;
    if (fault == 0.8) result->waypoints[1].dq_rad_s[0] = 100.0;
    if (fault == 0.9) return 3; /* Deliberate partial write followed by failure. */
    if (fault == 1.3) return 3;
    if (fault == 1.4) return 4;
    if (fault == 1.5) return 5;
    if (fault == 1.6) return 6;
    if (fault == 1.7) return 7;
    return 0;
}}
int probe_shutdown(void **context) {{ free(*context); *context = NULL; return 0; }}
"#
        ),
    )
    .unwrap();
    let path = root.join(if cfg!(target_os = "macos") {
        "libprobe.dylib"
    } else {
        "libprobe.so"
    });
    let output = Command::new("cc")
        .args([
            "-std=c11",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-pedantic",
            "-fPIC",
        ])
        .arg(if cfg!(target_os = "macos") {
            "-dynamiclib"
        } else {
            "-shared"
        })
        .arg(root.join("fixture.c"))
        .arg("-o")
        .arg(&path)
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    (root, path)
}

fn exercise<const N: usize>() {
    let (root, path) = library::<N>();
    assert!(matches!(
        Component::open(&path, "probe", N),
        Err(ComponentError::Version)
    ));
    assert!(matches!(
        Planner::<5>::open(&path, "probe"),
        Err(ComponentError::Input)
    ));
    if N == 6 {
        assert!(matches!(
            Planner::<7>::open(&path, "probe"),
            Err(ComponentError::Version)
        ));
    } else {
        assert!(matches!(
            Planner::<6>::open(&path, "probe"),
            Err(ComponentError::Version)
        ));
    }
    let child = root.join("child");
    let yaml = root.join("model.yml");
    fs::write(&child, "fixture-not-executed").unwrap();
    fs::write(&yaml, "{}").unwrap();
    let names: Vec<_> = (0..N).map(|i| format!("joint{i}")).collect();
    let names: Vec<_> = names.iter().map(String::as_str).collect();
    let limits = [[-2.0, 2.0]; N];
    let speeds = [2.0; N];
    let mut config = Config {
        model_sha256: [1; 32],
        robot_config_sha256: [2; 32],
        scene_sha256: [3; 32],
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
    let mut request = Request {
        start_q_rad: [0.0; N],
        goal_q_rad: [0.0; N],
        target_base_from_tool: pose(),
        cube_base_pose: pose(),
        tool_from_cube: pose(),
        phase: 0,
    };
    let mut planner = Planner::<N>::open(&path, "probe").unwrap();
    assert!(matches!(
        planner.plan(&request),
        Err(ComponentError::Lifecycle)
    ));
    planner.init(&child).unwrap();
    config.cube_edge_m = f64::NAN;
    assert_eq!(planner.configure(&config), Err(ComponentError::Input));
    assert_eq!(planner.state(), State::Initialized);
    config.cube_edge_m = 0.05;
    planner.configure(&config).unwrap();
    let output = planner.plan(&request).unwrap();
    assert_eq!(output.waypoints().len(), 2);
    assert_eq!(output.dt_s(), 0.1);
    request.start_q_rad[N - 1] = 1e-6;
    request.goal_q_rad = request.start_q_rad;
    assert!(
        planner.plan(&request).is_ok(),
        "inclusive start continuity boundary"
    );
    request.start_q_rad[N - 1] = 0.0;
    // Only a home request may carry a joint goal other than its start.
    assert!(matches!(planner.plan(&request), Err(ComponentError::Input)));
    request.phase = 4;
    assert!(planner.plan(&request).is_ok(), "home joint goal");
    request.goal_q_rad[0] = 2.5;
    assert!(matches!(planner.plan(&request), Err(ComponentError::Input)));
    request.goal_q_rad = request.start_q_rad;
    request.phase = 5;
    assert!(matches!(planner.plan(&request), Err(ComponentError::Input)));
    request.phase = 0;
    request.goal_q_rad[0] = -0.0;
    assert!(matches!(planner.plan(&request), Err(ComponentError::Input)));
    request.goal_q_rad[0] = 0.0;
    request.start_q_rad[0] = f64::NAN;
    request.goal_q_rad[0] = f64::NAN;
    assert!(matches!(planner.plan(&request), Err(ComponentError::Input)));
    planner.shutdown().unwrap();
    assert_eq!(planner.state(), State::Shutdown);
    drop(planner);
    for (value, status) in [
        (f64::from_bits(1e-6_f64.to_bits() + 1), None),
        (0.5, None),
        (0.6, None),
        (0.7, None),
        (0.8, None),
        (0.9, Some(3)),
        (1.3, Some(3)),
        (1.4, Some(4)),
        (1.5, Some(5)),
        (1.6, Some(6)),
        (1.7, Some(7)),
    ] {
        let mut planner = Planner::<N>::open(&path, "probe").unwrap();
        planner.init(&child).unwrap();
        planner.configure(&config).unwrap();
        request.start_q_rad[0] = value;
        request.goal_q_rad[0] = value;
        let result = planner.plan(&request);
        if let Some(expected) = status {
            assert!(matches!(result, Err(ComponentError::Status(actual)) if actual == expected));
        } else {
            assert!(
                matches!(result, Err(ComponentError::Output)),
                "fault {value}"
            );
        }
        assert_eq!(planner.state(), State::Failed);
        request.start_q_rad[0] = 0.0;
        request.goal_q_rad[0] = 0.0;
        assert!(matches!(
            planner.plan(&request),
            Err(ComponentError::Lifecycle)
        ));
        planner.shutdown().unwrap();
        assert_eq!(planner.state(), State::Shutdown);
    }
    assert_eq!(output.waypoints()[0].q_rad[0], 0.0);
    fs::remove_dir_all(root).unwrap();
}

#[test]
fn issue_s1_06_planner_runtime_six_joint_lifecycle_and_untrusted_output() {
    exercise::<6>();
}

#[test]
fn issue_s1_06_planner_runtime_seven_joint_lifecycle_and_untrusted_output() {
    exercise::<7>();
}
