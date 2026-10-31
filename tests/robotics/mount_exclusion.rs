//! Cell mount relation (IR v12): the UR5e base and shoulder are mounted on the table
//! and their table contact is excluded; anything else is a typed refusal.
//! Issue #29 (S1-06).
use invar::{front, system};
use serde_json::json;
use std::{fs, path::Path};

const SHOULDER: &str = "menagerie::universal_robots_ur5e::shoulder_link";

fn root() -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("examples/cells/reach_place")
}

fn compile(source: &str) -> Result<system::SystemIr, front::FrontError> {
    system::compile_at(front::parse_system(source.as_bytes())?, &root())
}

fn source(name: &str) -> String {
    fs::read_to_string(root().join(name)).unwrap()
}

/// Rendered table exclusions; robot-internal exclusions (Panda link0/link1) are vendor model.
fn excludes(ir: &system::SystemIr) -> Vec<(String, String)> {
    let files = system::generate_mjcf(ir).unwrap();
    let text = std::str::from_utf8(&files["scene.xml"]).unwrap().to_owned();
    let xml = roxmltree::Document::parse(&text).unwrap();
    xml.descendants()
        .filter(|n| n.has_tag_name("exclude") && n.attribute("body1") == Some("table"))
        .map(|n| {
            let side = |key| n.attribute(key).unwrap().to_owned();
            (side("body1"), side("body2"))
        })
        .collect()
}

#[test]
fn ur5e_mount_exclusion_is_checked_versioned_and_rendered() {
    let ir = compile(&source("cell-ur5e.inv")).unwrap();
    let doc = ir.checked_document().unwrap();
    assert_eq!(doc["schema"], "invar-system-v12");
    let robot = "menagerie::universal_robots_ur5e";
    assert_eq!(
        doc["cells"][0]["mount_surface"],
        json!({"fixture":"table","exclude":[{"robot":robot,"link":"base"},{"robot":robot,"link":"shoulder_link"}]})
    );
    let round_trip = system::SystemIr::from_json(ir.canonical_json().as_bytes()).unwrap();
    assert_eq!(round_trip.canonical_json(), ir.canonical_json());
    let mut old = doc.clone();
    old["schema"] = "invar-system-v11".into();
    old["version"] = 11.into();
    assert!(system::SystemIr::from_json(&serde_json::to_vec(&old).unwrap()).is_err());
    let pairs = |links: &[&str]| -> Vec<(String, String)> {
        links
            .iter()
            .map(|l| ("table".to_owned(), (*l).to_owned()))
            .collect()
    };
    assert_eq!(excludes(&ir), pairs(&["base", "shoulder_link"]));
    // The Panda product shares the cell source; UR5e exclusions do not apply to it.
    assert_eq!(excludes(&compile(&source("cell.inv")).unwrap()), pairs(&[]));
}

#[test]
fn mount_relation_refusals_are_typed_before_generation() {
    let ur5e = source("cell-ur5e.inv");
    assert!(ur5e.contains(SHOULDER));
    let mount = "mount xyz (0 m, 0 m, 0 m)";
    assert!(ur5e.contains(mount));
    for (changed, code, message) in [
        (
            ur5e.replace(SHOULDER, "menagerie::universal_robots_ur5e::upper_arm_link"),
            "S1608",
            "'menagerie::universal_robots_ur5e::upper_arm_link' is not mounted to fixture 'table'",
        ),
        (
            ur5e.replace(SHOULDER, "menagerie::universal_robots_ur5e::missing_link"),
            "S1601",
            "does not resolve",
        ),
        (
            ur5e.replace(SHOULDER, "menagerie::unknown_arm::shoulder_link"),
            "S1601",
            "does not resolve",
        ),
        (
            ur5e.replace(mount, "mount xyz (0 m, 0 m, 0.01 m)"),
            "S1608",
            "not on the top face of fixture 'table'",
        ),
        (
            ur5e.replace("on table\n", "on place_target\n"),
            "S1608",
            "needs exactly one collision box",
        ),
    ] {
        let error = compile(&changed).expect_err(message);
        assert_eq!(error.code, code, "{}", error.message);
        assert!(error.message.contains(message), "{}", error.message);
    }
}
