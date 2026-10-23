use std::{path::Path, process::Command};

use invar::{front::parse_system, system};

const PANDA: &str = include_str!("../examples/cells/reach_place/cell.inv");
const UR5E: &str = include_str!("../examples/cells/reach_place/cell-ur5e.inv");
const MANIFEST_IMPORT: &str = include_str!("../examples/cells/reach_place/manifest-import.inv");
const MANIFEST_SELECT: &str = include_str!("../examples/cells/reach_place/manifest-select.inv");
const ROBOT_DOF: &str = include_str!("../examples/cells/reach_place/robot-dof.inv");

fn compile(source: &str) -> system::SystemIr {
    let directory = Path::new(env!("CARGO_MANIFEST_DIR")).join("examples/cells/reach_place");
    system::compile_at(parse_system(source.as_bytes()).unwrap(), &directory).unwrap()
}

#[test]
fn issue_s1_06_integration_start_one_robot_reference_switches_arms() {
    let differences = PANDA
        .lines()
        .zip(UR5E.lines())
        .filter(|(left, right)| left != right)
        .collect::<Vec<_>>();
    assert_eq!(PANDA.lines().count(), UR5E.lines().count());
    assert_eq!(
        differences,
        vec![(
            "    robot menagerie::franka_emika_panda;",
            "    robot menagerie::universal_robots_ur5e;"
        )]
    );
    let panda = compile(PANDA);
    let ur5e = compile(UR5E);
    system::SystemIr::from_json(panda.canonical_json().as_bytes()).unwrap();
    system::SystemIr::from_json(ur5e.canonical_json().as_bytes()).unwrap();
    assert!(panda.graph_issues().is_empty());
    assert!(ur5e.graph_issues().is_empty());
    let panda: serde_json::Value = serde_json::from_str(&panda.canonical_json()).unwrap();
    let ur5e: serde_json::Value = serde_json::from_str(&ur5e.canonical_json()).unwrap();
    assert_eq!(panda["cells"], ur5e["cells"]);
    assert_eq!(panda["schema"], "invar-system-v12");
    assert_eq!(ur5e["schema"], "invar-system-v12");
    assert_eq!(
        panda["cells"][0]["objects"][0]["dynamic"],
        serde_json::Value::Null
    );
    assert!(
        panda["cells"][0]["objects"]
            .as_array()
            .unwrap()
            .iter()
            .any(|object| object["name"] == "cube" && object["dynamic"]["mass"]["value"] == "0.1")
    );
    let objects = panda["cells"][0]["objects"].as_array().unwrap();
    let cube = objects
        .iter()
        .find(|object| object["name"] == "cube")
        .unwrap();
    let target = objects
        .iter()
        .find(|object| object["name"] == "place_target")
        .unwrap();
    assert_eq!(target["pose"]["xyz"][2], cube["pose"]["xyz"][2]);
    assert_eq!(
        panda["products"][0]["robot"],
        "menagerie::franka_emika_panda"
    );
    assert_eq!(
        ur5e["products"][0]["robot"],
        "menagerie::universal_robots_ur5e"
    );
}

#[test]
fn issue_s1_06_integration_start_imports_current_component_evidence() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let directory = root.join("examples/cells/reach_place");
    let ir = system::compile_at(
        parse_system(MANIFEST_IMPORT.as_bytes()).unwrap(),
        &directory,
    )
    .unwrap();
    assert!(ir.graph_issues().is_empty(), "{:?}", ir.graph_issues());
    let document: serde_json::Value = serde_json::from_str(&ir.canonical_json()).unwrap();
    system::SystemIr::from_json(ir.canonical_json().as_bytes()).unwrap();
    assert_eq!(document["extern_components"][0]["name"], "pinocchio_fk");
    let path = directory.join("manifest-import.inv");
    let output = Command::new(env!("CARGO_BIN_EXE_invar"))
        .args([
            "system",
            "check",
            path.to_str().unwrap(),
            "--format",
            "json",
        ])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stdout)
    );
    let escaped = MANIFEST_IMPORT.replacen(
        "components/pinocchio_fk/manifest.inv",
        "../pinocchio_fk/manifest.inv",
        1,
    );
    assert_eq!(
        system::compile_at(parse_system(escaped.as_bytes()).unwrap(), &directory)
            .unwrap_err()
            .code,
        "S1400"
    );
    let unrecorded = directory.join("manifest-import-unrecorded.inv");
    let output = Command::new(env!("CARGO_BIN_EXE_invar"))
        .args(["check", unrecorded.to_str().unwrap(), "--format", "json"])
        .output()
        .unwrap();
    assert!(!output.status.success());
    let diagnostic: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(diagnostic["diagnostics"][0]["code"], "S1405");
}

#[test]
fn issue_s1_06_integration_start_selects_one_robot_manifest() {
    let directory = Path::new(env!("CARGO_MANIFEST_DIR")).join("examples/cells/reach_place");
    let ir = system::compile_at(
        parse_system(MANIFEST_SELECT.as_bytes()).unwrap(),
        &directory,
    )
    .unwrap();
    assert!(ir.graph_issues().is_empty());
    let document: serde_json::Value = serde_json::from_str(&ir.canonical_json()).unwrap();
    assert_eq!(document["extern_components"].as_array().unwrap().len(), 1);
    let output = Command::new(env!("CARGO_BIN_EXE_invar"))
        .args([
            "system",
            "check",
            directory.join("manifest-select.inv").to_str().unwrap(),
            "--format",
            "json",
        ])
        .output()
        .unwrap();
    assert!(output.status.success());
    let ur5e = MANIFEST_SELECT.replacen(
        "    robot menagerie::franka_emika_panda;",
        "    robot menagerie::universal_robots_ur5e;",
        1,
    );
    let ur5e_ir = system::compile_at(parse_system(ur5e.as_bytes()).unwrap(), &directory).unwrap();
    assert!(ur5e_ir.graph_issues().is_empty());
    let ur5e_document: serde_json::Value = serde_json::from_str(&ur5e_ir.canonical_json()).unwrap();
    let ports = ur5e_document["extern_components"][0]["ports"]
        .as_array()
        .unwrap();
    assert_eq!(
        ports.iter().find(|port| port["name"] == "q").unwrap()["type"],
        "std::robotics::JointPositions<f64,6>"
    );
    let unrecorded = ur5e.replacen("manifest-ur5e.inv", "manifest-unrecorded.inv", 1);
    assert_eq!(
        system::compile_at(parse_system(unrecorded.as_bytes()).unwrap(), &directory)
            .unwrap_err()
            .code,
        "S1405"
    );
    let zero = MANIFEST_SELECT.replacen(
        "for robot menagerie::franka_emika_panda;",
        "for robot menagerie::unknown_arm;",
        1,
    );
    assert_eq!(
        system::compile_at(parse_system(zero.as_bytes()).unwrap(), &directory)
            .unwrap_err()
            .code,
        "S1001"
    );
    let duplicate = MANIFEST_SELECT.replacen(
        "for robot menagerie::franka_emika_panda;",
        "for robot menagerie::universal_robots_ur5e;",
        1,
    );
    assert_eq!(
        system::compile_at(parse_system(duplicate.as_bytes()).unwrap(), &directory)
            .unwrap_err()
            .code,
        "S1000"
    );
}

#[test]
fn issue_s1_06_integration_start_resolves_product_robot_dof() {
    let directory = Path::new(env!("CARGO_MANIFEST_DIR")).join("examples/cells/reach_place");
    let source_path = directory.join("robot-dof.inv");
    let checked = Command::new(env!("CARGO_BIN_EXE_invar"))
        .args([
            "system",
            "check",
            source_path.to_str().unwrap(),
            "--format",
            "json",
        ])
        .output()
        .unwrap();
    assert!(
        checked.status.success(),
        "{}",
        String::from_utf8_lossy(&checked.stderr)
    );
    for (robot, count) in [
        ("menagerie::franka_emika_panda", 7),
        ("menagerie::universal_robots_ur5e", 6),
    ] {
        let source = ROBOT_DOF.replacen("menagerie::franka_emika_panda", robot, 1);
        let ir = compile(&source);
        assert!(ir.graph_issues().is_empty(), "{:?}", ir.graph_issues());
        let json = ir.canonical_json();
        assert_eq!(
            json.matches(&format!("JointPositions<f64,{count}>"))
                .count(),
            2
        );
        assert!(!json.contains("JointPositions<f64,robot_dof>"));
    }
    let unsupported = ROBOT_DOF.replacen("menagerie::franka_emika_panda", "menagerie::unknown", 1);
    assert_eq!(
        system::compile_at(parse_system(unsupported.as_bytes()).unwrap(), &directory)
            .unwrap_err()
            .code,
        "S1001"
    );
}

#[test]
fn issue_s1_06_integration_start_generated_fk_uses_selected_joint_names() {
    for (source, first_joint, count) in [(PANDA, "joint1", 7), (UR5E, "shoulder_pan_joint", 6)] {
        let ir = compile(source);
        assert!(ir.graph_issues().is_empty(), "{:?}", ir.graph_issues());
        let files = system::ros2::generate(&ir).unwrap();
        let node = &files["src/invar_fk_node/src/node.cpp"];
        assert!(node.contains(&format!("\"{first_joint}\"")));
        assert!(node.contains(&format!("\"model_kind\", {count}")));
        assert!(
            files["src/invar_cell_bringup/launch/cell.launch.py"]
                .contains("('q', '/joint_states')")
        );
    }
}

#[test]
fn issue_s1_06_integration_start_rejects_unsafe_attachment_before_generation() {
    let directory = Path::new(env!("CARGO_MANIFEST_DIR")).join("examples/cells/reach_place");
    for (source, code) in [
        (
            PANDA.replacen("tool hand pose", "tool missing_link pose", 1),
            "S1601",
        ),
        (
            PANDA.replacen(
                "dynamic mass 0.1 kg attachable;",
                "dynamic mass 0 kg attachable;",
                1,
            ),
            "S1602",
        ),
        (
            PANDA.replacen("object cube {", "fixture cube {", 1),
            "S1602",
        ),
        (
            PANDA.replacen("        collision box (0.025, 0.025, 0.025) m;", "", 1),
            "S1602",
        ),
    ] {
        let failure =
            system::compile_at(parse_system(source.as_bytes()).unwrap(), &directory).unwrap_err();
        assert_eq!(failure.code, code);
    }
}

#[test]
fn issue_s1_06_integration_start_cell_generates_deterministically() {
    for (name, source) in [("panda", PANDA), ("ur5e", UR5E)] {
        let ir = compile(source);
        let first = system::generate_mjcf(&ir).unwrap();
        assert_eq!(system::generate_urdf(&ir).unwrap_err().code, "S1607");
        assert_eq!(first, system::generate_mjcf(&ir).unwrap());
        let scene = std::str::from_utf8(&first["scene.xml"]).unwrap();
        for object in ["table", "cube", "place_target"] {
            assert!(
                scene.contains(&format!("name=\"{object}\"")),
                "{name}: {object}"
            );
        }
        assert!(scene.contains("name=\"invar_tool\""), "{name}: tool site");
        assert!(
            scene.contains("name=\"cube_free\""),
            "{name}: cube freejoint"
        );
        assert!(
            scene.contains("name=\"cube_attach\""),
            "{name}: disabled weld"
        );
        assert!(scene.contains("active=\"false\""), "{name}: disabled weld");
        // The generated hardware selects home. Omitted free-joint key entries
        // reset the cube to zero instead of its source pose, even though qpos0
        // itself is correct. Preserve the vendor arm/finger prefix explicitly.
        let home_key = scene
            .split("<key ")
            .find(|part| part.split('>').next().unwrap().contains("name=\"home\""))
            .unwrap();
        let qpos = home_key
            .split("qpos=\"")
            .nth(1)
            .unwrap()
            .split('"')
            .next()
            .unwrap()
            .split_whitespace()
            .collect::<Vec<_>>();
        let expected_prefix: &[&str] = if name == "panda" {
            &[
                "0", "0", "0", "-1.57079", "0", "1.57079", "-0.7853", "0.04", "0.04",
            ]
        } else {
            &["-1.5708", "-1.5708", "1.5708", "-1.5708", "-1.5708", "0"]
        };
        assert_eq!(
            &qpos[..expected_prefix.len()],
            expected_prefix,
            "{name}: unchanged robot home"
        );
        assert_eq!(
            &qpos[expected_prefix.len()..],
            ["0.3", "0", "0.025", "1", "0", "0", "0"],
            "{name}: source cube home pose"
        );
        let ros = system::ros2::generate(&ir).unwrap();
        assert_eq!(ros, system::ros2::generate(&ir).unwrap());
        assert!(!ros.is_empty());
    }
}

#[test]
fn issue_s1_06_integration_start_public_cli_emits_complete_ros_workspace() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    for (name, file, first_joint) in [
        ("panda", "cell.inv", "joint1"),
        ("ur5e", "cell-ur5e.inv", "shoulder_pan_joint"),
    ] {
        let source = root.join("examples/cells/reach_place").join(file);
        let output =
            std::env::temp_dir().join(format!("invar-s106-ros-{name}-{}", std::process::id()));
        let result = Command::new(env!("CARGO_BIN_EXE_invar"))
            .args([
                "system",
                "gen",
                "ros2",
                source.to_str().unwrap(),
                "--out",
                output.to_str().unwrap(),
            ])
            .output()
            .unwrap();
        assert!(
            result.status.success(),
            "{name}: {}",
            String::from_utf8_lossy(&result.stdout)
        );
        let config = output.join("src/invar_cell_bringup/config");
        assert!(config.join("mjcf/scene.xml").is_file());
        // Only the Panda keeps mesh assets (collision meshes); the UR5e is all primitives.
        assert_eq!(
            config.join("mjcf/assets").is_dir(),
            name == "panda",
            "{name}"
        );
        let yaml = std::fs::read_to_string(config.join("controllers.yaml")).unwrap();
        assert!(yaml.contains(first_joint), "{name}: selected joint");
        let urdf = std::fs::read_to_string(config.join("ros2_control.urdf")).unwrap();
        assert!(urdf.contains(first_joint), "{name}: selected URDF joint");
        assert!(urdf.contains("@INVAR_MJCF_PATH@"));
        let parsed = roxmltree::Document::parse(&urdf).unwrap();
        let links = parsed
            .descendants()
            .filter(|node| node.has_tag_name("link"))
            .count();
        let children = parsed
            .descendants()
            .filter(|node| {
                node.has_tag_name("joint")
                    && node
                        .parent()
                        .is_some_and(|parent| parent.has_tag_name("robot"))
            })
            .filter_map(|joint| joint.children().find(|node| node.has_tag_name("child")))
            .filter_map(|child| child.attribute("link"))
            .collect::<std::collections::BTreeSet<_>>();
        assert_eq!(children.len() + 1, links, "{name}: connected URDF tree");
        let launch =
            std::fs::read_to_string(output.join("src/invar_cell_bringup/launch/cell.launch.py"))
                .unwrap();
        assert!(launch.contains("package='mujoco_ros2_control'"));
        assert!(launch.contains("attachment_plugin.yaml"));
        let plugin = std::fs::read_to_string(
            output.join("src/invar_cell_bringup/src/attachment_plugin.cpp"),
        )
        .unwrap();
        assert!(plugin.contains("mjOBJ_EQUALITY"));
        assert!(plugin.contains("/invar/set_attachment"));
        std::fs::remove_dir_all(output).unwrap();
    }
}

#[test]
fn issue_s1_06_integration_start_ros_tendon_projection_preserves_physics() {
    let ir = compile(PANDA);
    let original = system::generate_mjcf(&ir).unwrap();
    let scene = std::str::from_utf8(&original["scene.xml"]).unwrap();
    assert_eq!(scene.matches("name=\"actuator8\"").count(), 1);
    let output = std::env::temp_dir().join(format!("invar-s106-tendon-{}", std::process::id()));
    let source = Path::new(env!("CARGO_MANIFEST_DIR")).join("examples/cells/reach_place/cell.inv");
    let result = Command::new(env!("CARGO_BIN_EXE_invar"))
        .args(["system", "gen", "ros2"])
        .arg(source)
        .arg("--out")
        .arg(&output)
        .output()
        .unwrap();
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stdout)
    );
    let projected =
        std::fs::read_to_string(output.join("src/invar_cell_bringup/config/mjcf/scene.xml"))
            .unwrap();
    std::fs::remove_dir_all(output).unwrap();
    assert_eq!(
        projected,
        scene.replacen("name=\"actuator8\"", "name=\"finger_joint1\"", 1),
        "ROS projection may rename the tendon actuator, never its transmission or physics"
    );
    assert_eq!(
        system::generate_mjcf(&ir).unwrap(),
        original,
        "ROS generation must not mutate checked IR or canonical MJCF assets"
    );
}

#[test]
fn issue_s1_06_integration_start_ros_rejects_unbounded_joint_limits() {
    let ir = compile(&PANDA.replacen("velocity 2.175 rad/s;", "velocity unbounded;", 1));
    assert_eq!(system::ros2::generate(&ir).unwrap_err().code, "S1504");
}

#[test]
fn issue_s1_06_integration_start_ros_rejects_multiple_attachments() {
    let source = PANDA.replacen(
        "    io attach: digital-out;",
        "    object second_cube { pose xyz (0.6 m, 0 m, 0.025 m) quat (1, 0, 0, 0); dynamic mass 0.1 kg attachable; collision box (0.025, 0.025, 0.025) m; }\n    io attach: digital-out;",
        1,
    );
    assert_eq!(
        system::ros2::generate(&compile(&source)).unwrap_err().code,
        "S1504"
    );
}

#[test]
fn issue_s1_06_integration_start_rejects_policy_and_budget_failures_before_launch() {
    let directory = Path::new(env!("CARGO_MANIFEST_DIR")).join("examples/cells/reach_place");
    for (source, code) in [
        (
            PANDA.replacen(
                "license \"Apache-2.0\";\n    isolation separate-process;",
                "license \"GPL-3.0-only\" link static;\n    isolation in-process;",
                1,
            ),
            "S1300",
        ),
        (
            PANDA.replacen("license \"Apache-2.0\";", "license \"All rights reserved\";", 1),
            "S1606",
        ),
        (
            PANDA.replacen("capacity cpu_cores = 6", "capacity cpu_cores = 1", 1),
            "S1250",
        ),
        (
            PANDA.replacen(
                "connect controller.cmd -> hardware.cmd policy sample-and-hold;",
                "connect controller.cmd -> hardware.cmd policy sample-and-hold;\nflow task: controller.cmd -> hardware.cmd within 1 ms;",
                1,
            ),
            "S1102",
        ),
    ] {
        let found = match system::compile_at(parse_system(source.as_bytes()).unwrap(), &directory) {
            Err(failure) => failure.code,
            Ok(ir) => ir.graph_issues().first().map(|issue| issue.0).unwrap_or("none"),
        };
        assert_eq!(found, code);
    }
}

#[test]
fn issue_s1_06_integration_start_rejects_unreachable_target_frame() {
    let invalid = PANDA.replacen(
        "frame place_target: world;",
        "frame place_target: missing;",
        1,
    );
    assert_ne!(invalid, PANDA);
    let directory = Path::new(env!("CARGO_MANIFEST_DIR")).join("examples/cells/reach_place");
    let failure =
        system::compile_at(parse_system(invalid.as_bytes()).unwrap(), &directory).unwrap_err();
    assert_eq!(failure.code, "S1200");
}

/// The Panda gripper home hold is source-declared (decision 2026-10-13): it
/// replaces the vendored home key's tendon command, stays inside the control
/// range, and a hold outside that range fails with a registered code.
#[test]
fn issue_s1_06_integration_start_declares_the_gripper_home_hold() {
    let ir = compile(PANDA);
    let document: serde_json::Value = serde_json::from_str(&ir.canonical_json()).unwrap();
    let panda = document["robots"]
        .as_array()
        .unwrap()
        .iter()
        .find(|robot| robot["name"] == "franka_emika_panda")
        .unwrap();
    let held = panda["actuators"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|actuator| !actuator["hold"].is_null())
        .collect::<Vec<_>>();
    assert_eq!(held.len(), 1);
    assert_eq!(held[0]["name"], "actuator8");
    assert_eq!(
        held[0]["hold"],
        serde_json::json!({"value": "253", "unit": "one"})
    );
    let files = system::generate_mjcf(&ir).unwrap();
    let scene = String::from_utf8(files["scene.xml"].clone()).unwrap();
    let model = roxmltree::Document::parse(&scene).unwrap();
    let home = model
        .descendants()
        .find(|n| n.has_tag_name("key") && n.attribute("name") == Some("home"))
        .unwrap();
    assert_eq!(
        home.attribute("ctrl"),
        Some("0 0 0 -1.57079 0 1.57079 -0.7853 253")
    );
    let directory = Path::new(env!("CARGO_MANIFEST_DIR")).join("examples/cells/reach_place");
    for (hold, code, message) in [
        (
            "255.5 one",
            "S1603",
            "actuator hold command lies outside its control range",
        ),
        (
            "-1 one",
            "S1603",
            "actuator hold command lies outside its control range",
        ),
        ("253 rad", "S1600", "expected one unit"),
    ] {
        let source = PANDA.replace("hold 253 one;", &format!("hold {hold};"));
        let error =
            system::compile_at(parse_system(source.as_bytes()).unwrap(), &directory).unwrap_err();
        assert_eq!(
            (error.code, error.message.as_str()),
            (code, message),
            "{hold}"
        );
    }
    // The mount relation (v12) is stripped to isolate the hold versioning.
    let unheld = compile(
        &PANDA
            .replace("            hold 253 one;\n", "")
            .replace(" on table\n        exclude (menagerie::universal_robots_ur5e::base, menagerie::universal_robots_ur5e::shoulder_link)", ""),
    );
    let unheld: serde_json::Value = serde_json::from_str(&unheld.canonical_json()).unwrap();
    assert_eq!(unheld["schema"], "invar-system-v10");
}

#[test]
fn issue_s1_06_run_reports_bind_source_and_enforce_repeat_tolerance() {
    use serde_json::json;
    use sha2::{Digest, Sha256};

    let digest = |bytes: &[u8]| {
        Sha256::digest(bytes)
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect::<String>()
    };
    let ir = compile(PANDA);
    // Synthetic measurements exercise the reporter, never runtime qualification.
    let state = json!({
        "q_rad": vec![0.0; 7], "dq_rad_s": vec![0.0; 7],
        "robot_q_si": vec![0.0; 9], "robot_dq_si_per_s": vec![0.0; 9],
        "world_from_tool": vec![0.0; 7], "world_from_cube": vec![0.0; 7], "attached": [0.0]
    });
    let state_hash = digest(&vec![0_u8; (7 + 7 + 9 + 9 + 7 + 7 + 1) * 8]);
    let moves = [
        "pregrasp",
        "descend",
        "lift",
        "transfer",
        "place_descend",
        "retreat",
        "home",
    ];
    let waypoints = moves
        .iter()
        .enumerate()
        .map(|(i, name)| {
            json!({
                "move": name, "simulation_ns": i + 1,
                "tool_error_m_rad": [0.001, 0.001],
                "pinocchio_vs_recorded_m_rad": [1e-15, 1e-15],
                "pinocchio_vs_mujoco_m_rad": [1e-15, 1e-15]
            })
        })
        .collect::<Vec<_>>();
    let runs = (0..3)
        .map(|i| {
            json!({
                "run": format!("synthetic-{i}"), "seed": 123, "launch_exit": 0,
                "snapshots": 7, "topic_counts": {"/invar/cell_snapshot": 7,
                    "/sequencer/request": 7, "/sequencer/approved": 7, "/sequencer/phase": 24},
                "launch_log_sha256": "a".repeat(64),
                "mcap_sha256": format!("{i:064x}"), "mcap_bytes": 100,
                "complete_sim_ns": 8, "waypoints": waypoints,
                "final_cube_error_m_rad": [0.001, 0.001], "home_error_rad": 1e-8,
                "forbidden_robot_contacts": 0, "max_joint_or_velocity_limit_excess": -1e-8,
                "final_attached": false, "final_recorded_state": state, "final_recorded_state_bits": vec![0_u64; 47],
                "final_recorded_state_sha256": state_hash
            })
        })
        .collect::<Vec<_>>();
    let mut evidence = json!({
        "schema": "invar-s106-run-evidence-v1", "arm": "panda",
        "source_ir_sha256": digest(ir.canonical_json().as_bytes()),
        "binding_sha256": "a".repeat(64), "scene_sha256": "a".repeat(64),
        "model_sha256": "a".repeat(64), "build_log_sha256": "a".repeat(64),
        "build_packages_finished": 6,
        "versions": {"pinocchio": "4.1.0", "mujoco": "3.12.0", "numpy": "2.5.3"},
        "runs": runs
    });
    let report = |e: &serde_json::Value| ir.report_with_run_evidence(e.to_string().as_bytes());
    let (text, accepted) = report(&evidence).unwrap();
    assert!(accepted);
    assert!(text.contains(
        "Recorded task/FK/limit/contact checks (tool <2 mm / <1 degree; home <=1e-6 rad): pass"
    ));
    assert!(text.contains("bit-exact for one seed"));
    assert!(text.contains("Full MuJoCo integration state is absent"));
    assert!(text.contains("Deadlines and WCET values are declared budgets"));
    let evidence_path =
        std::env::temp_dir().join(format!("invar-s106-report-{}.json", std::process::id()));
    std::fs::write(&evidence_path, evidence.to_string()).unwrap();
    let output = Command::new(env!("CARGO_BIN_EXE_invar"))
        .args(["system", "report"])
        .arg(Path::new(env!("CARGO_MANIFEST_DIR")).join("examples/cells/reach_place/cell.inv"))
        .arg("--run-evidence")
        .arg(&evidence_path)
        .output()
        .unwrap();
    std::fs::remove_file(evidence_path).unwrap();
    assert_eq!(output.status.code(), Some(0));
    assert_eq!(String::from_utf8(output.stdout).unwrap(), text);
    let mut full = evidence.clone();
    full["mujoco_state_spec"] = json!("mjSTATE_INTEGRATION");
    full["mujoco_integration_state_size"] = json!(47);
    let mut full_bits = vec![0_u64; 47];
    full_bits[0] = (8e-9_f64).to_bits();
    for run in full["runs"].as_array_mut().unwrap() {
        run["mujoco_integration_state_bits"] = json!(full_bits);
    }
    assert!(report(&full).unwrap().1);
    full["runs"][1]["mujoco_integration_state_bits"][1] = json!(1_u64);
    assert!(report(&full).unwrap().1);
    let mut varied = evidence.clone();
    varied["runs"][1]["final_recorded_state_bits"][0] = json!(1_u64);
    let mut varied_bytes = vec![0_u8; 47 * 8];
    varied_bytes[0] = 1;
    varied["runs"][1]["final_recorded_state_sha256"] = json!(digest(&varied_bytes));
    let (text, accepted) = report(&varied).unwrap();
    assert!(accepted);
    assert!(text.contains("NOT bit-exact for one seed"));
    let mut joint_edge = evidence.clone();
    joint_edge["runs"][0]["home_error_rad"] = json!(1e-6);
    assert!(report(&joint_edge).unwrap().1);
    for (position, angle, home) in [
        (0.002, 0.001, 1e-8),
        (0.001, std::f64::consts::PI / 180.0, 1e-8),
        (0.001, 0.001, 1.000001e-6),
    ] {
        let mut boundary = evidence.clone();
        boundary["runs"][0]["waypoints"][0]["tool_error_m_rad"] = json!([position, angle]);
        boundary["runs"][0]["home_error_rad"] = json!(home);
        assert!(!report(&boundary).unwrap().1);
    }
    let mut other_seed = evidence.clone();
    other_seed["runs"][1]["seed"] = json!(124);
    assert!(!report(&other_seed).unwrap().1);
    evidence["runs"][0]["waypoints"][0]["tool_error_m_rad"] = json!([0.003, 0.001]);
    assert!(report(&evidence).unwrap().0.contains(
        "Recorded task/FK/limit/contact checks (tool <2 mm / <1 degree; home <=1e-6 rad): fail"
    ));
    evidence["runs"][0]["final_recorded_state_bits"][0] = json!(0.1_f64.to_bits());
    assert_eq!(
        report(&evidence).unwrap_err(),
        "final state digest mismatch"
    );
    evidence["source_ir_sha256"] = json!("b".repeat(64));
    assert_eq!(
        report(&evidence).unwrap_err(),
        "run evidence does not match checked system IR"
    );
}
