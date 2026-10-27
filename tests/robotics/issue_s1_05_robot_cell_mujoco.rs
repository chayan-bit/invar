use invar::{
    front::{SystemItemKind, parse_system},
    system::{self, SystemIr},
};
use std::{path::Path, process::Command};

#[path = "../support/scratch.rs"]
mod scratch;
const PLANAR: &[u8] = include_bytes!("../fixtures/s1_05/planar-cell.inv");

#[test]
fn issue_s1_05_robot_cell_mujoco_parses_typed_planar_cell() {
    let program = parse_system(PLANAR).unwrap();
    assert!(program.system_items.iter().any(|item| matches!(
        &item.kind,
        SystemItemKind::RobotModel(robot)
            if robot.name == "Planar" && robot.links.len() == 2 && robot.joints.len() == 1
    )));
    assert!(program.system_items.iter().any(|item| matches!(
        &item.kind,
        SystemItemKind::CellModel(cell)
            if cell.name == "Bench" && cell.objects.len() == 2 && cell.io.len() == 1
    )));
}

#[test]
fn issue_s1_05_robot_cell_mujoco_checks_v5_tree_and_schema() {
    let ir = system::compile(parse_system(PLANAR).unwrap()).unwrap();
    let bytes = ir.canonical_json();
    let document: serde_json::Value = serde_json::from_str(&bytes).unwrap();
    assert_eq!(document["schema"], "invar-system-v12");
    assert_eq!(document["version"], 12);
    assert_eq!(document["robots"][0]["links"].as_array().unwrap().len(), 2);
    assert_eq!(document["cells"][0]["objects"].as_array().unwrap().len(), 2);
    assert_eq!(document["robots"][0]["links"][1]["mass"]["value"], "0.5");
    assert!(ir.graph_issues().is_empty(), "{:?}", ir.graph_issues());
    assert!(SystemIr::from_json(bytes.as_bytes()).is_ok());
}

#[test]
fn issue_s1_05_robot_cell_mujoco_camera_intrinsics_are_exact() {
    let source = std::str::from_utf8(PLANAR).unwrap().replacen(
        "    sensor encoder {",
        "    sensor eye {\n        kind camera;\n        source arm;\n        intrinsics (width 640 px, height 480 px, fx 500.25 px, fy 501 px, cx 321.25 px, cy 239.5 px, distortion none);\n    }\n    sensor encoder {",
        1,
    );
    let ir = system::compile(parse_system(source.as_bytes()).unwrap()).unwrap();
    let document: serde_json::Value = serde_json::from_str(&ir.canonical_json()).unwrap();
    assert_eq!(
        document["robots"][0]["sensors"][0]["intrinsics"]["fx_px"],
        "500.25"
    );
    let scene = system::generate_mjcf(&ir).unwrap();
    let xml = std::str::from_utf8(&scene["scene.xml"]).unwrap();
    assert!(xml.contains("focalpixel=\"500.25 501\""));
    assert!(xml.contains("principalpixel=\"1.25 -0.5\""));
    assert!(xml.contains("resolution=\"640 480\""));
    assert!(xml.contains("sensorsize=\"640 480\""));
    for (old, new, code) in [
        ("width 640 px", "width 0 px", "S1602"),
        ("height 480 px", "height 480 m", "S1600"),
        ("distortion none", "distortion plumb_bob", "S1607"),
    ] {
        let changed = source.replacen(old, new, 1);
        let error = match parse_system(changed.as_bytes()) {
            Ok(program) => system::compile(program).unwrap_err(),
            Err(error) => error,
        };
        assert_eq!(error.code, code);
    }
}

#[test]
fn issue_s1_05_robot_cell_mujoco_rejects_invalid_robot_models() {
    let source = std::str::from_utf8(PLANAR).unwrap();
    for (old, new, code) in [
        (
            "parent base;\n        child arm;",
            "parent missing;\n        child arm;",
            "S1601",
        ),
        ("mass 0.5 kg;", "mass -0.5 kg;", "S1602"),
        (
            "inertia (0.001, 0.001, 0.001, 0, 0, 0)",
            "inertia (0, 0.001, 0.001, 0, 0, 0)",
            "S1602",
        ),
        (
            "position -1 rad .. 1 rad;",
            "position 2 rad .. 1 rad;",
            "S1603",
        ),
        (
            "link base {\n        parent world;",
            "link base {\n        parent arm;",
            "S1601",
        ),
        (
            "robot Planar;\n    capacity",
            "robot Missing;\n    capacity",
            "S1601",
        ),
    ] {
        let mutated = source.replacen(old, new, 1);
        assert_ne!(mutated, source);
        let error = system::compile(parse_system(mutated.as_bytes()).unwrap()).unwrap_err();
        assert_eq!(error.code, code, "{old} => {new}: {error:?}");
    }
}

#[test]
fn issue_s1_05_robot_cell_mujoco_public_diagnostics() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    for code in [
        "S1600", "S1601", "S1602", "S1603", "S1605", "S1606", "S1607", "S1608",
    ] {
        let path = std::fs::read_dir(root.join("tests/fixtures/s1_05/negative"))
            .unwrap()
            .map(|entry| entry.unwrap().path())
            .find(|path| {
                path.file_name()
                    .unwrap()
                    .to_string_lossy()
                    .starts_with(code)
            })
            .unwrap();
        let output = Command::new(env!("CARGO_BIN_EXE_invar"))
            .args(["check", path.to_str().unwrap(), "--format", "json"])
            .output()
            .unwrap();
        assert_eq!(output.status.code(), Some(1), "{}", path.display());
        assert!(output.stderr.is_empty(), "{}", path.display());
        let diagnostic: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
        assert_eq!(
            diagnostic["diagnostics"][0]["code"],
            code,
            "{}",
            path.display()
        );
    }
}

#[test]
fn issue_s1_05_robot_cell_mujoco_menagerie_round_trip() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    for (name, directory) in [
        ("panda", "franka_emika_panda"),
        ("ur5e", "universal_robots_ur5e"),
    ] {
        let model = root.join(format!(
            "examples/cells/reach_place/assets/menagerie/{directory}/scene.xml"
        ));
        let imported = Command::new(env!("CARGO_BIN_EXE_invar"))
            .args(["import", "mjcf", model.to_str().unwrap()])
            .current_dir(root)
            .output()
            .unwrap();
        assert!(
            imported.status.success(),
            "{name}: {}",
            String::from_utf8_lossy(&imported.stderr)
        );
        let imported_ir =
            system::compile_at(parse_system(&imported.stdout).unwrap(), root).unwrap();
        let imported_files = system::generate_mjcf(&imported_ir).unwrap();
        let fixture = root.join(format!("examples/cells/menagerie-{name}.inv"));
        let source = std::fs::read(&fixture).unwrap();
        let ir =
            system::compile_at(parse_system(&source).unwrap(), fixture.parent().unwrap()).unwrap();
        let files = system::generate_mjcf(&ir).unwrap();
        assert!(files.len() > 1, "{name}: missing checked asset file set");
        let expected = system::canonical_mjcf(&root.join(format!(
            "examples/cells/reach_place/assets/menagerie/{directory}/scene.xml"
        )))
        .unwrap();
        assert_eq!(
            std::str::from_utf8(&imported_files["scene.xml"]).unwrap(),
            expected,
            "{name} CLI import"
        );
        assert_eq!(
            std::str::from_utf8(&files["scene.xml"]).unwrap(),
            expected,
            "{name}"
        );
    }
}

#[test]
fn issue_s1_05_robot_cell_mujoco_typed_edits_change_generated_scene() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let source = std::str::from_utf8(PLANAR).unwrap();
    let changed = source.replacen("mass 0.5 kg;", "mass 0.6 kg;", 1).replacen(
        "mount xyz (0 m, 0 m, 0 m)",
        "mount xyz (1 m, 0 m, 0 m)",
        1,
    );
    let scene =
        system::generate_mjcf(&system::compile(parse_system(changed.as_bytes()).unwrap()).unwrap())
            .unwrap();
    let xml = std::str::from_utf8(&scene["scene.xml"]).unwrap();
    assert!(xml.contains("mass=\"0.6\""));
    assert!(xml.contains("name=\"Bench_mount\""));
    assert!(xml.contains("name=\"table\""));

    let fixture = root.join("examples/cells/menagerie-ur5e.inv");
    let source = std::fs::read_to_string(&fixture).unwrap();
    let changed = source.replacen("mass 4 kg;", "mass 4.1 kg;", 1).replacen(
        "collision capsule (0.06, 0.06) m;",
        "collision capsule (0.07, 0.06) m;",
        1,
    );
    assert_ne!(changed, source);
    let ir = system::compile_at(
        parse_system(changed.as_bytes()).unwrap(),
        fixture.parent().unwrap(),
    )
    .unwrap();
    let scene = system::generate_mjcf(&ir).unwrap();
    let xml = std::str::from_utf8(&scene["scene.xml"]).unwrap();
    assert!(xml.contains("mass=\"4.1\""));
    assert!(xml.contains("size=\"0.07 0.06\""));
}

#[test]
fn issue_s1_05_robot_cell_mujoco_generates_standard_urdf() {
    let source = std::str::from_utf8(PLANAR)
        .unwrap()
        .replace("capsule (0.02, 0.1)", "cylinder (0.02, 0.1)")
        .replace(
            "    sensor encoder {\n        kind joint-position;\n        source shoulder;\n    }\n",
            "",
        );
    let ir = system::compile(parse_system(source.as_bytes()).unwrap()).unwrap();
    let files = system::generate_urdf(&ir).unwrap();
    let document =
        roxmltree::Document::parse(std::str::from_utf8(&files["robot.urdf"]).unwrap()).unwrap();
    let root = document.root_element();
    assert_eq!(root.tag_name().name(), "robot");
    assert_eq!(root.attribute("name"), Some("Planar"));
    assert_eq!(
        root.children()
            .filter(|child| child.has_tag_name("link"))
            .count(),
        2
    );
    assert_eq!(
        root.children()
            .filter(|child| child.has_tag_name("joint"))
            .count(),
        1
    );
    assert!(
        root.children()
            .any(|child| child.has_tag_name("ros2_control"))
    );
    let directory = std::env::temp_dir().join(format!("invar-s1-05-urdf-{}", scratch::unique()));
    std::fs::create_dir(&directory).unwrap();
    let urdf = directory.join("robot.urdf");
    std::fs::write(&urdf, &files["robot.urdf"]).unwrap();
    std::fs::write(
        directory.join("LICENSE"),
        "Apache License\nVersion 2.0, January 2004\n",
    )
    .unwrap();
    let (imported, warnings) = system::import_urdf(&urdf).unwrap();
    assert!(warnings.is_empty());
    let imported =
        system::compile_at(parse_system(imported.as_bytes()).unwrap(), &directory).unwrap();
    let regenerated = system::generate_urdf(&imported).unwrap();
    assert_eq!(regenerated["robot.urdf"], files["robot.urdf"]);
    let canonical = system::canonical_urdf(&urdf).unwrap();
    assert_eq!(canonical.as_bytes(), &files["robot.urdf"]);
    let malformed = std::str::from_utf8(&files["robot.urdf"]).unwrap().replacen(
        "<mass value=\"1\"/>",
        "<mass extra=\"1\" value=\"1\"/>",
        1,
    );
    assert_ne!(malformed.as_bytes(), &files["robot.urdf"]);
    std::fs::write(&urdf, malformed).unwrap();
    assert_eq!(system::import_urdf(&urdf).unwrap_err().code, "S1607");
    std::fs::remove_dir_all(&directory).unwrap();
}

#[test]
fn issue_s1_05_robot_cell_mujoco_urdf_reports_loss_boundary() {
    let source = std::str::from_utf8(PLANAR)
        .unwrap()
        .replace("capsule (0.02, 0.1)", "cylinder (0.02, 0.1)");
    let ir = system::compile(parse_system(source.as_bytes()).unwrap()).unwrap();
    let files = system::generate_urdf(&ir).unwrap();
    assert!(
        std::str::from_utf8(&files["urdf-warnings.txt"])
            .unwrap()
            .contains("S1604 warning: robot sensors")
    );
    assert!(files.contains_key("scene.xml"));

    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    for name in ["panda", "ur5e"] {
        let source = root.join(format!("examples/cells/menagerie-{name}.inv"));
        let ir = system::compile_at(
            parse_system(&std::fs::read(&source).unwrap()).unwrap(),
            source.parent().unwrap(),
        )
        .unwrap();
        let error = system::generate_urdf(&ir).unwrap_err();
        assert_eq!(error.code, "S1607");
        assert!(
            error.message.contains("tendon, equality, or keyframe"),
            "{name}: {error:?}"
        );
    }
}

#[test]
fn issue_s1_05_robot_cell_mujoco_product_robot_switch_keeps_cell() {
    let planar = std::str::from_utf8(PLANAR).unwrap();
    let (robot, rest) = planar.split_once("\ncell Bench {").unwrap();
    let alternate = robot
        .replacen("robot Planar {", "robot Alternate {", 1)
        .replacen("mass 0.5 kg;", "mass 0.7 kg;", 1);
    let source = format!("{robot}\n{alternate}\ncell Bench {{{rest}");
    let switched = source.replacen(
        "robot Planar;\n    capacity",
        "robot Alternate;\n    capacity",
        1,
    );
    assert_ne!(switched, source);
    let generated = |source: &str| {
        let ir = system::compile(parse_system(source.as_bytes()).unwrap()).unwrap();
        String::from_utf8(system::generate_mjcf(&ir).unwrap()["scene.xml"].clone()).unwrap()
    };
    let before = generated(&source);
    let after = generated(&switched);
    assert_ne!(before, after);
    let without_arm = |scene: &str| {
        let document = roxmltree::Document::parse(scene).unwrap();
        let root = document
            .descendants()
            .find(|node| node.has_tag_name("body") && node.attribute("name") == Some("base"))
            .unwrap();
        let range = root.range();
        format!(
            "{}<selected-robot/>{}",
            &scene[..range.start],
            &scene[range.end..]
        )
    };
    assert_eq!(without_arm(&before), without_arm(&after));
    assert!(before.contains("name=\"table\""));
    assert!(after.contains("name=\"table\""));
}

#[test]
fn issue_s1_05_robot_cell_mujoco_handwritten_urdf_round_trip() {
    let source = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/s1_05/urdf/robot.urdf");
    let canonical = system::canonical_urdf(&source).unwrap();
    let (imported, warnings) = system::import_urdf(&source).unwrap();
    assert!(warnings.is_empty());
    let command = Command::new(env!("CARGO_BIN_EXE_invar"))
        .args(["import", "urdf", source.to_str().unwrap()])
        .output()
        .unwrap();
    assert!(
        command.status.success(),
        "{}",
        String::from_utf8_lossy(&command.stderr)
    );
    assert_eq!(command.stdout, imported.as_bytes());
    let checked = system::compile_at(
        parse_system(imported.as_bytes()).unwrap(),
        source.parent().unwrap(),
    )
    .unwrap();
    let model: serde_json::Value = serde_json::from_str(&checked.canonical_json()).unwrap();
    assert_eq!(model["robots"][0]["actuators"].as_array().unwrap().len(), 1);
    let regenerated = system::generate_urdf(&checked).unwrap();
    assert_eq!(
        std::str::from_utf8(&regenerated["robot.urdf"]).unwrap(),
        canonical
    );
    let directory =
        std::env::temp_dir().join(format!("invar-s1-05-urdf-cli-{}", scratch::unique()));
    std::fs::create_dir(&directory).unwrap();
    let program = directory.join("imported.inv");
    std::fs::write(&program, imported).unwrap();
    let output = directory.join("generated");
    let command = Command::new(env!("CARGO_BIN_EXE_invar"))
        .args([
            "system",
            "gen",
            "urdf",
            program.to_str().unwrap(),
            "--out",
            output.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert!(
        command.status.success(),
        "{}",
        String::from_utf8_lossy(&command.stderr)
    );
    assert_eq!(
        std::fs::read_to_string(output.join("robot.urdf")).unwrap(),
        canonical
    );
    std::fs::remove_dir_all(&directory).unwrap();
}

#[test]
fn issue_s1_05_robot_cell_mujoco_force_torque_is_six_axis_site_sensor() {
    let source = std::str::from_utf8(PLANAR).unwrap().replacen(
        "kind joint-position;",
        "kind force-torque;",
        1,
    );
    let ir = system::compile(parse_system(source.as_bytes()).unwrap()).unwrap();
    let scene = system::generate_mjcf(&ir).unwrap();
    let xml = std::str::from_utf8(&scene["scene.xml"]).unwrap();
    assert!(xml.contains("name=\"encoder_site\""));
    assert!(xml.contains("<force name=\"encoder_force\" site=\"encoder_site\"/>"));
    assert!(xml.contains("<torque name=\"encoder_torque\" site=\"encoder_site\"/>"));
    assert!(!xml.contains("jointactuatorfrc"));
}

#[test]
fn issue_s1_05_robot_cell_mujoco_reports_non_dynamic_import_warning() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let source = root.join("tests/fixtures/s1_05/negative/S1604-visual-warning.inv");
    let result = Command::new(env!("CARGO_BIN_EXE_invar"))
        .args(["check", source.to_str().unwrap(), "--format", "json"])
        .output()
        .unwrap();
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stdout)
    );
    let diagnostic: serde_json::Value = serde_json::from_slice(&result.stdout).unwrap();
    assert_eq!(diagnostic["diagnostics"][0]["code"], "S1604");
    assert_eq!(diagnostic["diagnostics"][0]["severity"], "warning");
    let report = Command::new(env!("CARGO_BIN_EXE_invar"))
        .args(["system", "report", source.to_str().unwrap()])
        .output()
        .unwrap();
    assert!(report.status.success());
    assert!(String::from_utf8_lossy(&report.stdout).contains("S1604 warning"));
}

#[test]
fn issue_s1_05_robot_cell_mujoco_rejects_asset_hash_license_and_dynamics() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    for (name, code) in [
        ("S1605-mesh-hash-mismatch.inv", "S1605"),
        ("S1606-asset-license-rejected.inv", "S1606"),
        ("S1607-unsupported-dynamics.inv", "S1607"),
    ] {
        let source = root.join("tests/fixtures/s1_05/negative").join(name);
        let result = Command::new(env!("CARGO_BIN_EXE_invar"))
            .args(["check", source.to_str().unwrap(), "--format", "json"])
            .output()
            .unwrap();
        assert_eq!(result.status.code(), Some(1), "{name}");
        let diagnostic: serde_json::Value = serde_json::from_slice(&result.stdout).unwrap();
        assert_eq!(diagnostic["diagnostics"][0]["code"], code, "{name}");
    }
}

#[test]
fn issue_s1_05_vendored_menagerie_files_match_their_pinned_manifest() {
    use sha2::{Digest, Sha256};
    use std::fmt::Write;
    let root =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("examples/cells/reach_place/assets/menagerie");
    let manifest = std::fs::read_to_string(root.join("MANIFEST.tsv")).unwrap();
    for pin in [
        "# source https://github.com/google-deepmind/mujoco_menagerie",
        "# tag v2026.9.2",
        "# commit c96a32d28fb5da84da38c1da4d749e7a13212855",
    ] {
        assert!(
            manifest.lines().any(|line| line == pin),
            "manifest lacks {pin}"
        );
    }
    let mut rows = manifest.lines().filter(|line| !line.starts_with('#'));
    assert_eq!(rows.next(), Some("path\tbytes\tsha256\tlicense"));
    let mut listed = std::collections::BTreeMap::new();
    for row in rows {
        let [path, bytes, sha256, license] = row.split('\t').collect::<Vec<_>>()[..] else {
            panic!("malformed manifest row {row:?}");
        };
        let data = std::fs::read(root.join(path)).unwrap();
        assert_eq!(data.len().to_string(), bytes, "{path} byte count");
        let mut digest = String::new();
        for byte in Sha256::digest(&data) {
            write!(digest, "{byte:02x}").unwrap();
        }
        assert_eq!(digest, sha256, "{path} sha256");
        assert!(
            listed.insert(path.to_owned(), license.to_owned()).is_none(),
            "{path} listed twice"
        );
    }
    assert_eq!(listed["franka_emika_panda/LICENSE"], "Apache-2.0");
    assert_eq!(listed["universal_robots_ur5e/LICENSE"], "BSD-3-Clause");
    for (path, license) in &listed {
        let family = path.split('/').next().unwrap();
        assert_eq!(
            license,
            &listed[&format!("{family}/LICENSE")],
            "{path} license"
        );
    }
    let mut stack = vec![root.clone()];
    let mut present = Vec::new();
    while let Some(directory) = stack.pop() {
        for entry in std::fs::read_dir(directory).unwrap() {
            let path = entry.unwrap().path();
            if path.is_dir() {
                stack.push(path);
            } else {
                present.push(
                    path.strip_prefix(&root)
                        .unwrap()
                        .to_str()
                        .unwrap()
                        .replace('\\', "/"),
                );
            }
        }
    }
    present.retain(|path| path != "MANIFEST.tsv");
    present.sort();
    assert_eq!(
        present,
        listed.keys().cloned().collect::<Vec<_>>(),
        "unlisted or missing vendored files"
    );
}
