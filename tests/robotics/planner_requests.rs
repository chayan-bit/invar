//! Generator for the pinned v3 planner requests and runtime scene fields.
//!
//! The fixed requests come from the Pinocchio goldens (start joints, goal tool poses
//! projected into the cell tool frame) and from the checked cell (cube pose, grasp, table
//! and mount exclusion). The test fails when a committed file differs from the generated
//! value. Regenerate with `INVAR_REGENERATE_PLANNER_REQUESTS=DIR cargo test --test
//! robotics planner_requests::`, which writes every generated file into DIR (never into
//! the repository) for review and copying.
//! Issue #29 (S1-06).
use super::file_hash::hash;
use invar::system::{planner_scene, tool_projection::Pose};
use serde_json::{Value, json};
use std::{fs, path::Path};

const COMPONENT: &str = "examples/cells/reach_place/components/curobo_planner";
const GOLDENS: &str = "tests/fixtures/foreign-components/pinocchio_goldens.json";
/// Planner world phases (planner-component-v2), by wire value.
const PHASES: [&str; 5] = ["free", "approach", "held", "held-contact", "home"];

fn golden_pose(vector: &Value) -> Pose {
    let t = &vector["translation_m"];
    let q = &vector["quaternion_wxyz"];
    let f = |v: &Value| v.as_f64().unwrap();
    Pose {
        translation_m: [f(&t[0]), f(&t[1]), f(&t[2])],
        quaternion_xyzw: [f(&q[1]), f(&q[2]), f(&q[3]), f(&q[0])],
    }
}

/// The production child is pinned by content in every runtime binding.
fn child_hash() -> String {
    hash(
        &Path::new(env!("CARGO_MANIFEST_DIR"))
            .join(COMPONENT)
            .join("planner_server.py"),
    )
}

fn checked_ir(cell_path: &Path) -> invar::system::SystemIr {
    invar::system::compile_at(
        invar::front::parse_system(&fs::read(cell_path).unwrap()).unwrap(),
        cell_path.parent().unwrap(),
    )
    .unwrap()
}

/// Generated (endpoint, runtime) for one arm; `pair` is the (start, goal) golden pair.
fn generate(arm: &str, n: usize, cell: &str, pair: (usize, usize)) -> (Value, Value) {
    let repo = Path::new(env!("CARGO_MANIFEST_DIR"));
    let cell_path = repo.join("examples/cells/reach_place").join(cell);
    let ir = checked_ir(&cell_path);
    let projection = invar::system::tool_projection::selected(&ir).unwrap();
    let scene = planner_scene::selected(&ir).unwrap();
    let source: Value = serde_json::from_slice(&fs::read(repo.join(GOLDENS)).unwrap()).unwrap();
    let reference = source["arms"]
        .as_array()
        .unwrap()
        .iter()
        .find(|a| a["arm"] == arm)
        .unwrap();
    let component = repo.join(COMPONENT);
    let runtime_name = format!("runtime-source-{arm}-v3.json");
    let mut runtime: Value =
        serde_json::from_slice(&fs::read(component.join(&runtime_name)).unwrap()).unwrap();
    runtime["tool_projection"] = projection.value();
    runtime["cell_source"]["sha256"] = json!(hash(&cell_path));
    runtime["artifacts"]["child"]["sha256"] = json!(child_hash());
    runtime["config"]["table_pose"] = json!(planner_scene::flat(scene.table_pose));
    runtime["config"]["table_half_extents_m"] = json!(scene.table_half_extents_m);
    runtime["config"]["mount_exclusion"] = scene.mount_exclusion.clone();
    let names = runtime["config"]["joint_names"].as_array().unwrap();
    let cell_tool = |index: usize| {
        projection
            .fk_to_source(golden_pose(&reference["vectors"][index]))
            .unwrap()
    };
    // Every planner world phase in both directions. Free, approach and home: the cube
    // rests at its cell pose. Held and held-contact: it is held at the declared grasp,
    // so it starts at the start tool composed with that grasp. Home carries the goal
    // golden's joints as its joint goal; the target is that golden's tool.
    let tool_from_cube = scene.tool_from_cube;
    let mut vectors = Vec::new();
    for (start, goal) in [pair, (pair.1, pair.0)] {
        let held = cell_tool(start).compose(tool_from_cube).unwrap();
        for (phase, name) in PHASES.iter().enumerate() {
            let cube = if matches!(phase, 2 | 3) {
                held
            } else {
                scene.cube_base_pose
            };
            let mut vector = json!({
                "id": format!("{arm}-{start}-to-{goal}-{name}"),
                "source_start_vector": start,
                "source_goal_vector": goal,
                "start_q_rad": reference["vectors"][start]["q_rad"],
                "target_base_from_tool": planner_scene::flat(cell_tool(goal)),
                "cube_base_pose": planner_scene::flat(cube),
                "tool_from_cube": planner_scene::flat(tool_from_cube),
                "phase": phase,
            });
            if phase == 4 {
                vector["goal_q_rad"] = reference["vectors"][goal]["q_rad"].clone();
            }
            vectors.push(vector);
        }
    }
    let endpoint = json!({
        "schema": "invar-planner-endpoint-goldens-v3",
        "joint_count": n,
        "seed": 123,
        "qualification": "unexecuted_fixed_requests_not_task_acceptance",
        "source_reference": {
            "path": format!("../../../../../{GOLDENS}"),
            "sha256": hash(&repo.join(GOLDENS)),
            "arm": arm,
        },
        "joint_mapping": names
            .iter()
            .zip(reference["joint_names"].as_array().unwrap())
            .map(|(a, b)| json!([a, b]))
            .collect::<Vec<_>>(),
        "metrics": {
            "translation": "euclidean-m",
            "orientation": "sign-invariant-geodesic-rad",
            "translation_max_m": 0.002,
            "orientation_max_rad": std::f64::consts::PI / 180.0,
            "start_max_rad": 1e-6,
        },
        "tool_projection": projection.value(),
        "vectors": vectors,
    });
    (endpoint, runtime)
}

#[test]
fn planner_source_requests_are_generated_from_goldens_and_cell() {
    let component = Path::new(env!("CARGO_MANIFEST_DIR")).join(COMPONENT);
    let output =
        std::env::var_os("INVAR_REGENERATE_PLANNER_REQUESTS").map(std::path::PathBuf::from);
    let mut stale = Vec::new();
    // UR5e uses goldens 0 and 2: golden 1's tool is 0.508 m below the base, inside the table.
    for (arm, n, cell, pair) in [
        ("panda", 7, "cell.inv", (0, 1)),
        ("ur5e", 6, "cell-ur5e.inv", (0, 2)),
    ] {
        let (endpoint, runtime) = generate(arm, n, cell, pair);
        for (name, value) in [
            (format!("endpoint-source-{arm}-v3.json"), endpoint),
            (format!("runtime-source-{arm}-v3.json"), runtime),
        ] {
            if let Some(directory) = &output {
                fs::create_dir_all(directory).unwrap();
                let text = serde_json::to_string_pretty(&value).unwrap() + "\n";
                fs::write(directory.join(&name), text).unwrap();
            }
            let committed: Value =
                serde_json::from_slice(&fs::read(component.join(&name)).unwrap()).unwrap();
            if committed != value {
                stale.push(name);
            }
        }
    }
    assert!(
        stale.is_empty(),
        "differ from their generated values: {stale:?}"
    );
}
