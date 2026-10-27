use std::{path::Path, process::Command};

use invar::{front::parse_system, system};
use serde_json::Value;

#[test]
#[ignore = "requires the pinned MuJoCo Python environment"]
fn issue_s1_05_robot_cell_mujoco_env_generated_models_step_and_match_declarations() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    for (name, bodies, joints, actuators) in [
        ("panda", 12, 9, 8),
        ("ur5e", 8, 6, 6),
        ("planar", 5, 1, 1),
        ("urdf", 3, 1, 1),
    ] {
        let fixture = if name == "urdf" {
            root.join("tests/fixtures/s1_05/urdf/robot.urdf")
        } else if name == "planar" {
            root.join("tests/fixtures/s1_05/planar-cell.inv")
        } else {
            root.join(format!("examples/cells/menagerie-{name}.inv"))
        };
        let source = if name == "urdf" {
            system::import_urdf(&fixture).unwrap().0.into_bytes()
        } else {
            std::fs::read(&fixture).unwrap()
        };
        let ir =
            system::compile_at(parse_system(&source).unwrap(), fixture.parent().unwrap()).unwrap();
        let document: Value = serde_json::from_str(&ir.canonical_json()).unwrap();
        let files = system::generate_mjcf(&ir).unwrap();
        let output = std::env::temp_dir().join(format!("invar-s105-{name}-{}", std::process::id()));
        std::fs::create_dir(&output).unwrap();
        for (relative, bytes) in &files {
            let path = output.join(relative);
            std::fs::create_dir_all(path.parent().unwrap()).unwrap();
            std::fs::write(path, bytes).unwrap();
        }
        let script = r#"
import json, math, mujoco, sys, warnings
with warnings.catch_warnings(record=True) as emitted:
    warnings.simplefilter('always')
    m = mujoco.MjModel.from_xml_path(sys.argv[1])
    d = mujoco.MjData(m)
    if m.nkey:
        d.qpos[:] = m.key_qpos[0]
        d.ctrl[:] = m.key_ctrl[0]
    for _ in range(1000):
        mujoco.mj_step(m, d)
        assert all(math.isfinite(x) for x in d.qpos)
        assert all(math.isfinite(x) for x in d.qvel)
    assert not emitted, [str(item.message) for item in emitted]
    assert all(item.number == 0 for item in d.warning)
    print(json.dumps({'bodies': [[m.body(i).name, float(m.body(i).mass[0])] for i in range(1,m.nbody)], 'joints': [[m.joint(i).name, list(map(float,m.joint(i).range))] for i in range(m.njnt)], 'nbody':m.nbody, 'njnt':m.njnt, 'nu':m.nu, 'steps':1000}))
"#;
        let result = Command::new("python3")
            .args(["-c", script, output.join("scene.xml").to_str().unwrap()])
            .output()
            .unwrap();
        assert!(
            result.status.success(),
            "{name}: {}",
            String::from_utf8_lossy(&result.stderr)
        );
        let observed: Value = serde_json::from_slice(&result.stdout).unwrap();
        assert_eq!(
            (
                observed["nbody"].as_u64().unwrap(),
                observed["njnt"].as_u64().unwrap(),
                observed["nu"].as_u64().unwrap()
            ),
            (bodies, joints, actuators)
        );
        assert_eq!(observed["steps"], 1000);
        let robot = &document["robots"][0];
        for link in robot["links"].as_array().unwrap() {
            let actual = observed["bodies"]
                .as_array()
                .unwrap()
                .iter()
                .find(|body| body[0] == link["name"])
                .unwrap()[1]
                .as_f64()
                .unwrap();
            let expected: f64 = link["mass"]["value"].as_str().unwrap().parse().unwrap();
            assert_eq!(
                actual.to_bits(),
                expected.to_bits(),
                "{name}: {} mass",
                link["name"]
            );
        }
        for joint in robot["joints"].as_array().unwrap() {
            let actual = observed["joints"]
                .as_array()
                .unwrap()
                .iter()
                .find(|item| item[0] == joint["name"])
                .unwrap();
            for (i, bound) in ["lower", "upper"].iter().enumerate() {
                let expected: f64 = joint["position"][bound]["value"]
                    .as_str()
                    .unwrap()
                    .parse()
                    .unwrap();
                assert_eq!(
                    actual[1][i].as_f64().unwrap().to_bits(),
                    expected.to_bits(),
                    "{name}: {} {bound}",
                    joint["name"]
                );
            }
        }
        std::fs::remove_dir_all(output).unwrap();
    }
}

#[test]
#[ignore = "requires the pinned MuJoCo Python environment"]
fn issue_s1_05_robot_cell_mujoco_env_force_torque_has_six_axes() {
    let source = include_str!("../fixtures/s1_05/planar-cell.inv").replacen(
        "kind joint-position;",
        "kind force-torque;",
        1,
    );
    let ir = system::compile(parse_system(source.as_bytes()).unwrap()).unwrap();
    let files = system::generate_mjcf(&ir).unwrap();
    let path = std::env::temp_dir().join(format!("invar-s105-ft-{}.xml", std::process::id()));
    std::fs::write(&path, &files["scene.xml"]).unwrap();
    let result = Command::new("python3")
        .args(["-c", "import mujoco,sys; m=mujoco.MjModel.from_xml_path(sys.argv[1]); d=mujoco.MjData(m); mujoco.mj_step(m,d); assert m.nsensor == 2; assert list(m.sensor_dim) == [3,3]", path.to_str().unwrap()])
        .output().unwrap();
    std::fs::remove_file(path).unwrap();
    crate::child_deadline::assert_success(&result);
}

#[test]
#[ignore = "requires the pinned MuJoCo Python environment"]
fn issue_s1_05_robot_cell_mujoco_env_camera_loads() {
    let source = include_str!("../fixtures/s1_05/planar-cell.inv").replacen(
        "    sensor encoder {",
        "    sensor eye {\n        kind camera;\n        source arm;\n        intrinsics (width 640 px, height 480 px, fx 500.25 px, fy 501 px, cx 321.25 px, cy 239.5 px, distortion none);\n    }\n    sensor encoder {",
        1,
    );
    let ir = system::compile(parse_system(source.as_bytes()).unwrap()).unwrap();
    let files = system::generate_mjcf(&ir).unwrap();
    let path = std::env::temp_dir().join(format!("invar-s105-camera-{}.xml", std::process::id()));
    std::fs::write(&path, &files["scene.xml"]).unwrap();
    let result = Command::new("python3")
        .args(["-c", "import mujoco,sys; m=mujoco.MjModel.from_xml_path(sys.argv[1]); d=mujoco.MjData(m); mujoco.mj_step(m,d); assert list(m.cam_resolution[0]) == [640,480]; assert list(m.cam_intrinsic[0]) == [500.25,501,1.25,-0.5]", path.to_str().unwrap()])
        .output().unwrap();
    std::fs::remove_file(path).unwrap();
    crate::child_deadline::assert_success(&result);
}
