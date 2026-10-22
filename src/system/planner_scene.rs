//! Planner scene of the selected cell, derived only from the checked source.
//!
//! Frames: the *source base* is the robot mount frame (the parent of the robot root
//! body); planner requests carry cube poses in it. The planner world (cuRobo base) is
//! the *reference base*, the robot root body. `source_base_from_reference_base` comes
//! from the checked tool projection, so a robot whose root is rotated in its source
//! model (the UR5e base is turned by pi about z) gets its table in the right place.
use serde_json::{Value, json};

use super::tool_projection::{self, Pose};

#[derive(Clone, Debug, PartialEq)]
pub struct PlannerScene {
    /// Mount-surface fixture pose in the planner (reference base) frame.
    pub table_pose: Pose,
    pub table_half_extents_m: [f64; 3],
    /// The single attachable object's pose in the source base frame.
    pub cube_base_pose: Pose,
    /// The held object in the cell tool frame: the inverse of its declared grasp pose
    /// (the tool frame in the object frame, which the generated MJCF weld aligns).
    pub tool_from_cube: Pose,
    /// `{"fixture": NAME, "links": [...]}`: the selected robot's checked mount exclusions.
    pub mount_exclusion: Value,
}

fn number(value: &Value) -> Result<f64, &'static str> {
    value
        .as_str()
        .and_then(|s| s.parse::<f64>().ok())
        .filter(|n| n.is_finite())
        .ok_or("finite cell number")
}

/// Pose of a checked IR pose value (`xyz` metres, `quat` wxyz, normalized here).
pub fn ir_pose(value: &Value) -> Result<Pose, &'static str> {
    let xyz = value["xyz"]
        .as_array()
        .filter(|v| v.len() == 3)
        .ok_or("cell pose position")?;
    let quat = value["quat"]
        .as_array()
        .filter(|v| v.len() == 4)
        .ok_or("cell pose rotation")?;
    let p = [
        number(&xyz[0]["value"])?,
        number(&xyz[1]["value"])?,
        number(&xyz[2]["value"])?,
    ];
    let q = [
        number(&quat[0])?,
        number(&quat[1])?,
        number(&quat[2])?,
        number(&quat[3])?,
    ];
    tool_projection::source_pose(p, q)
}

/// Pose as the planner wire array `[x, y, z, qx, qy, qz, qw]` (negative zero cleared).
pub fn flat(pose: Pose) -> [f64; 7] {
    let [x, y, z] = pose.translation_m;
    let [qx, qy, qz, qw] = pose.quaternion_xyzw;
    [x, y, z, qx, qy, qz, qw].map(|v| v + 0.0)
}

pub fn selected(ir: &super::SystemIr) -> Result<PlannerScene, &'static str> {
    let document = ir.checked_document()?;
    let [cell] = document["cells"]
        .as_array()
        .map(Vec::as_slice)
        .unwrap_or(&[])
    else {
        return Err("planner scene needs exactly one cell");
    };
    let fixture = cell["mount_surface"]["fixture"]
        .as_str()
        .ok_or("planner scene needs a declared cell mount surface")?;
    let objects = cell["objects"].as_array().ok_or("cell objects")?;
    let table = objects
        .iter()
        .find(|o| o["name"] == fixture)
        .ok_or("mount surface fixture")?;
    let dims = table["geometries"]
        .as_array()
        .and_then(|g| {
            g.iter()
                .find(|g| g["role"] == "collision" && g["kind"] == "box")
        })
        .and_then(|g| g["dimensions"].as_array())
        .filter(|d| d.len() == 3)
        .ok_or("mount surface collision box")?;
    let attachable: Vec<_> = objects
        .iter()
        .filter(|o| o["dynamic"]["attachable"] == true)
        .collect();
    let [cube] = attachable.as_slice() else {
        return Err("planner scene needs exactly one attachable object");
    };
    let projection = tool_projection::selected(ir)?;
    let base_from_world = ir_pose(&cell["mount"])?.inverse()?;
    let reference_from_world = projection
        .source_base_from_reference_base
        .inverse()?
        .compose(base_from_world)?;
    let product = document["products"]
        .as_array()
        .and_then(|p| p.iter().find(|p| p["id"] == document["system"]["product"]))
        .ok_or("selected product")?;
    let robot = product["robot"].as_str().ok_or("selected robot")?;
    Ok(PlannerScene {
        table_pose: reference_from_world.compose(ir_pose(&table["pose"])?)?,
        table_half_extents_m: [number(&dims[0])?, number(&dims[1])?, number(&dims[2])?],
        cube_base_pose: base_from_world.compose(ir_pose(&cube["pose"])?)?,
        tool_from_cube: ir_pose(&cube["grasp"])
            .map_err(|_| "planner scene needs a declared grasp on the attachable object")?
            .inverse()?,
        mount_exclusion: json!({"fixture":fixture,"links":super::mount::excluded_links(cell, robot)}),
    })
}
