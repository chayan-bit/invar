//! Checked explicit grasp semantics; generation/CPU evidence, not live acceptance.
#[path = "../support/sequencer_binding.rs"]
#[allow(dead_code)]
mod support;
use invar::{front, system};
use serde_json::Value;
use std::{fs, path::Path};

const GRASP: &str = "        grasp pose xyz(0 m, 0 m, 0.025 m) quat(0, 1, 0, 0);\n";
/// The source-declared gripper hold raises the IR to v11; this test isolates
/// the grasp versioning, so it compiles the cells without the hold.
const HOLD: &str = "            hold 253 one;\n";
/// The mount relation raises the IR to v12; it is stripped for the same reason.
const MOUNT: &str = " on table\n        exclude (menagerie::universal_robots_ur5e::base, menagerie::universal_robots_ur5e::shoulder_link)";
fn compile(source: &str) -> Result<system::SystemIr, front::FrontError> {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("examples/cells/reach_place");
    system::compile_at(front::parse_system(source.as_bytes())?, &root)
}

#[test]
fn explicit_grasp_is_versioned_and_legacy_weld_is_preserved() {
    for name in ["cell.inv", "cell-ur5e.inv"] {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("examples/cells/reach_place");
        let source = fs::read_to_string(root.join(name))
            .unwrap()
            .replace(HOLD, "")
            .replace(MOUNT, "");
        for explicit in [false, true] {
            let source = if explicit {
                source.clone()
            } else {
                source.replace(GRASP, "")
            };
            let ir = compile(&source).unwrap();
            let doc = ir.checked_document().unwrap();
            assert_eq!(doc["version"], 12);
            assert_eq!(
                system::SystemIr::from_json(ir.canonical_json().as_bytes())
                    .unwrap()
                    .canonical_json(),
                ir.canonical_json()
            );
            let files = system::generate_mjcf(&ir).unwrap();
            let xml = roxmltree::Document::parse(std::str::from_utf8(&files["scene.xml"]).unwrap())
                .unwrap();
            let site = xml
                .descendants()
                .find(|n| n.has_tag_name("site") && n.attribute("name") == Some("cube_grasp"))
                .unwrap();
            assert_eq!(
                site.attribute("pos").unwrap_or("0 0 0"),
                if explicit { "0 0 0.025" } else { "0 0 0" }
            );
            assert_eq!(
                site.attribute("quat").unwrap_or("1 0 0 0"),
                if explicit { "0 1 0 0" } else { "1 0 0 0" }
            );
            let cube = xml
                .descendants()
                .find(|n| n.has_tag_name("body") && n.attribute("name") == Some("cube"))
                .unwrap();
            assert_eq!(cube.attribute("pos"), Some("0.3 0 0.025"));
            let weld = xml
                .descendants()
                .find(|n| n.has_tag_name("weld") && n.attribute("name") == Some("cube_attach"))
                .unwrap();
            assert_eq!(weld.attribute("site2"), Some("cube_grasp"));
            assert_eq!(weld.attribute("active"), Some("false"));
            if explicit {
                let mut old = doc.clone();
                old["schema"] = "invar-system-v11".into();
                old["version"] = 11.into();
                assert!(system::SystemIr::from_json(&serde_json::to_vec(&old).unwrap()).is_err());
            }
        }
    }
}

#[test]
fn malformed_or_inapplicable_grasps_fail_before_generation() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("examples/cells/reach_place");
    let source = fs::read_to_string(root.join("cell.inv")).unwrap();
    for bad in [
        source.replace(GRASP, &format!("{GRASP}{GRASP}")),
        source.replace("quat(0, 1, 0, 0);", "quat(0, 0, 0, 0);"),
        source.replace(
            "grasp pose xyz(0 m, 0 m, 0.025 m)",
            "grasp pose xyz(0 s, 0 m, 0.025 m)",
        ),
        source.replace("        dynamic mass 0.1 kg attachable;\n", ""),
        source.replace("object cube {", "fixture cube {"),
    ] {
        assert!(compile(&bad).is_err());
    }
}

#[test]
fn checked_binding_uses_source_grasp_and_preserves_cube_target() {
    for n in [6, 7] {
        let source = support::source(n).replace(
            "        dynamic mass 0.1 kg attachable;\n",
            &format!("        dynamic mass 0.1 kg attachable;\n{GRASP}"),
        );
        let ir = compile(&source).unwrap();
        let workspace = system::ros2::generate(&ir).unwrap();
        let binding: Value = serde_json::from_str(
            &workspace["src/invar_cell_bringup/config/sequencer-binding-v1.json"],
        )
        .unwrap();
        assert_eq!(binding["schema"], "invar-sequencer-binding-v2");
        assert_eq!(
            binding["tool_projection"]["schema"],
            "invar-tool-projection-v2"
        );
        for (key, expected) in [
            ("cube_from_grasp", [0., 0., 0.025, 1., 0., 0., 0.]),
            ("grasp_tool_base", [0.3, 0., 0.05, 1., 0., 0., 0.]),
            ("pregrasp_tool_base", [0.3, 0., 0.15, 1., 0., 0., 0.]),
            ("final_cube_base", [0.5, 0.2, 0.025, 0., 0., 0., 1.]),
        ] {
            for (actual, expected) in binding[key].as_array().unwrap().iter().zip(expected) {
                assert!((actual.as_f64().unwrap() - expected).abs() < 1e-12);
            }
        }
    }
}
