//! Build-only planner/controller fixtures; never production planner evidence.
use std::{collections::BTreeMap, fs, path::Path};

use invar::{front, system};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};

pub fn workspace(n: u32) -> BTreeMap<String, String> {
    assert!(matches!(n, 6 | 7));
    let directory = Path::new(env!("CARGO_MANIFEST_DIR")).join("examples/cells/reach_place");
    let file = if n == 6 { "cell-ur5e.inv" } else { "cell.inv" };
    let manifest = include_str!("../fixtures/s1_06/planner-manifest.inv")
        .lines()
        .filter(|line| !line.starts_with("frame "))
        .collect::<Vec<_>>()
        .join("\n")
        .replace("f64,7", &format!("f64,{n}"))
        .replace("request7_", &format!("request{n}_"))
        .replace("trajectory7_", &format!("trajectory{n}_"));
    let probe = format!(
        r#"
{manifest}
// Build-only source: the stub emits no requests and establishes no task evidence.
component SequencerBuildProbe {{
 param robot_mount: Transform<robot_base,world> = identity;
 // Build-only typed object-pose publisher; emits no telemetry.
 port out measured_cube: std::robotics::Pose<f64,world,cube> @ 0.2 Hz;
 port out request: std::robotics::PlannerRequest<f64,{n},base,tool,cube> @ 0.2 Hz;
 resource cpu_cores = 1, memory = 1 MiB;
 license "Apache-2.0"; isolation in-process; wcet 1 ms; deadline 5 s; impl invar run;
}}
"#
    );
    // Keep measured FK alongside the request consumer. This build-only probe
    // establishes generation/type-hash behavior, not object telemetry or a task.
    let source = fs::read_to_string(directory.join(file)).unwrap()
        .replace("component joint_trajectory_controller {", &format!("{probe}\ncomponent joint_trajectory_controller {{\n port in trajectory: std::robotics::PlannerTrajectory<f64,{n}> @ 0.2 Hz;"))
        .replace("connect controller.cmd", "connect sequencer.request -> planner.request policy queue(1);\nconnect planner.trajectory -> controller.trajectory policy queue(1);\nconnect controller.cmd")
        .replace("    controller: joint_trajectory_controller;", "    sequencer: SequencerBuildProbe;\n    planner: curobo_planner;\n    controller: joint_trajectory_controller;");
    let ir =
        system::compile_at(front::parse_system(source.as_bytes()).unwrap(), &directory).unwrap();
    assert!(ir.graph_issues().is_empty(), "{:?}", ir.graph_issues());
    system::ros2::generate(&ir).unwrap()
}

pub fn sha256(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

pub fn expected(files: &BTreeMap<String, String>, n: u32) -> BTreeMap<String, Value> {
    // The request is V2 since the per-phase planner world (decision D2).
    [("Request", 2), ("Waypoint", 1), ("Trajectory", 1)]
        .into_iter()
        .map(|(kind, version)| {
            let name = format!("Planner{kind}{n}V{version}");
            let path = format!("src/invar_cell_interfaces/hashes/{name}.json");
            (name, serde_json::from_str(&files[&path]).unwrap())
        })
        .collect()
}

/// Match by full type name, never by the first array entry. Missing/duplicate
/// identities and mismatched hashes fail even when another entry has the hash.
pub fn compare(expected: &Value, built: &Value) -> Result<(), &'static str> {
    let name = expected["type_name"]
        .as_str()
        .ok_or("missing expected type")?;
    let hash = expected["hash_string"]
        .as_str()
        .ok_or("missing expected hash")?;
    if !hash
        .strip_prefix("RIHS01_")
        .is_some_and(|hex| hex.len() == 64 && hex.bytes().all(|byte| byte.is_ascii_hexdigit()))
    {
        return Err("invalid expected hash");
    }
    let entries = built["type_hashes"]
        .as_array()
        .ok_or("missing ROS hashes")?;
    let matches: Vec<_> = entries
        .iter()
        .filter(|entry| entry["type_name"] == name)
        .collect();
    if matches.len() != 1 || matches[0]["hash_string"] != hash {
        return Err("ROS type identity/hash mismatch");
    }
    Ok(())
}

pub fn write_workspace(root: &Path, files: &BTreeMap<String, String>) {
    fs::create_dir(root).unwrap();
    for (relative, contents) in files {
        let path = root.join(relative);
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        fs::write(path, contents).unwrap();
    }
    let hashes: BTreeMap<_, _> = files
        .iter()
        .map(|(path, content)| (path, sha256(content.as_bytes())))
        .collect();
    fs::write(
        root.join("source-hashes.json"),
        serde_json::to_vec_pretty(&json!({
            "scope": "build_only_no_planner_or_task_execution",
            "files": hashes,
        }))
        .unwrap(),
    )
    .unwrap();
}
