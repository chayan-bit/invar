//! Versioned planner wire layouts, independent of the private INVP transport.

use serde_json::{Value, json};

pub fn port_layout(ty: &str) -> Option<String> {
    let n = super::planner_joint_count(ty)?;
    let (head, _) = super::generic_parts(ty)?;
    match head {
        "std::robotics::PlannerRequest" => Some(format!("inv_planner_request{n}_f64_v2")),
        "std::robotics::PlannerTrajectory" => Some(format!("inv_planner_trajectory{n}_f64_v1")),
        _ => None,
    }
}

pub fn manifest_joints(ports: &[Value]) -> Result<u32, &'static str> {
    if ports.len() != 2 {
        return Err("planner requires one request and one trajectory port");
    }
    let input = ports
        .iter()
        .find(|port| port["direction"] == "in")
        .ok_or("missing planner request")?;
    let output = ports
        .iter()
        .find(|port| port["direction"] == "out")
        .ok_or("missing planner trajectory")?;
    let input_ty = input["type"].as_str().ok_or("missing planner input type")?;
    let output_ty = output["type"]
        .as_str()
        .ok_or("missing planner output type")?;
    let n = super::planner_joint_count(input_ty).ok_or("unsupported planner joint count")?;
    if !input_ty.starts_with("std::robotics::PlannerRequest<")
        || !output_ty.starts_with("std::robotics::PlannerTrajectory<")
        || super::planner_joint_count(output_ty) != Some(n)
        || port_layout(input_ty).as_deref() != input["layout"].as_str()
        || port_layout(output_ty).as_deref() != output["layout"].as_str()
    {
        return Err("planner port types, counts and C layouts must agree");
    }
    Ok(n)
}

/// Exact LP64 descriptor shared with the checked generated header and runtime.
pub fn config_layout() -> Value {
    json!({"layout":"inv_planner_config_v1", "size":480, "align":8, "fields":[
        {"name":"robot_kind","type":"u32","offset":0},
        {"name":"joint_count","type":"u32","offset":4},
        {"name":"model_sha256","type":"u8[32]","offset":8},
        {"name":"robot_config_sha256","type":"u8[32]","offset":40},
        {"name":"scene_sha256","type":"u8[32]","offset":72},
        {"name":"joint_limits_rad","type":"f64[7][2]","offset":104},
        {"name":"table_pose","type":"pose_f64","offset":216},
        {"name":"table_half_extents_m","type":"f64[3]","offset":272},
        {"name":"cube_edge_m","type":"f64","offset":296},
        {"name":"base_frame","type":"borrowed_text_v1","offset":304,"borrow":"configure-call"},
        {"name":"tool_frame","type":"borrowed_text_v1","offset":320,"borrow":"configure-call"},
        {"name":"joint_names","type":"borrowed_text_v1[7]","offset":336,"borrow":"configure-call"},
        {"name":"robot_config_path","type":"borrowed_text_v1","offset":448,"borrow":"configure-call"},
        {"name":"scene_config_path","type":"borrowed_text_v1","offset":464,"borrow":"configure-call"}
    ]})
}

/// Emit deterministic six/seven-joint C layouts without changing the existing
/// 64-waypoint JointTrajectory ABI. No library is loaded by header generation.
pub fn header(joints: u32) -> Result<String, &'static str> {
    if !matches!(joints, 6 | 7) {
        return Err("planner-v1 requires six or seven joints");
    }
    let request_size = 176 + 16 * joints;
    let waypoint_size = 8 + 16 * joints;
    let trajectory_size = 16 + 256 * waypoint_size;
    Ok(format!(
        r#"#ifndef INVAR_PLANNER{joints}_WIRE_V1_H
#define INVAR_PLANNER{joints}_WIRE_V1_H
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
#define INVAR_PLANNER_ASSERT static_assert
#define INVAR_PLANNER_ALIGNOF alignof
#else
#define INVAR_PLANNER_ASSERT _Static_assert
#define INVAR_PLANNER_ALIGNOF _Alignof
#endif
#ifndef INVAR_PLANNER_POSE_WIRE_V1_DEFINED
#define INVAR_PLANNER_POSE_WIRE_V1_DEFINED
typedef struct {{ double xyz_m[3]; double quaternion_xyzw[4]; }} inv_planner_pose_wire_v1;
typedef struct {{ const uint8_t *bytes; uint16_t length; }} inv_planner_text_wire_v1;
typedef struct {{
    uint32_t robot_kind, joint_count;
    uint8_t model_sha256[32], robot_config_sha256[32], scene_sha256[32];
    double joint_limits_rad[7][2];
    inv_planner_pose_wire_v1 table_pose;
    double table_half_extents_m[3], cube_edge_m;
    inv_planner_text_wire_v1 base_frame, tool_frame, joint_names[7];
    inv_planner_text_wire_v1 robot_config_path, scene_config_path;
}} inv_planner_config_wire_v1;
#endif
typedef struct {{
    double start_q_rad[{joints}];
    double goal_q_rad[{joints}];
    inv_planner_pose_wire_v1 target_base_from_tool;
    inv_planner_pose_wire_v1 cube_base_pose;
    inv_planner_pose_wire_v1 tool_from_cube;
    uint32_t phase;
}} inv_planner_request{joints}_f64_v2;
typedef struct {{
    double time_from_start_s;
    double q_rad[{joints}];
    double dq_rad_s[{joints}];
}} inv_planner_waypoint{joints}_f64_v1;
typedef struct {{
    double dt_s;
    uint32_t count;
    inv_planner_waypoint{joints}_f64_v1 waypoints[256];
}} inv_planner_trajectory{joints}_f64_v1;
INVAR_PLANNER_ASSERT(sizeof(inv_planner_pose_wire_v1) == 56, "planner pose size");
INVAR_PLANNER_ASSERT(sizeof(inv_planner_text_wire_v1) == 16, "planner text size");
INVAR_PLANNER_ASSERT(sizeof(inv_planner_config_wire_v1) == 480, "planner config size");
INVAR_PLANNER_ASSERT(offsetof(inv_planner_config_wire_v1, joint_limits_rad) == 104, "planner limit offset");
INVAR_PLANNER_ASSERT(offsetof(inv_planner_config_wire_v1, table_pose) == 216, "planner table offset");
INVAR_PLANNER_ASSERT(offsetof(inv_planner_config_wire_v1, joint_names) == 336, "planner names offset");
INVAR_PLANNER_ASSERT(offsetof(inv_planner_config_wire_v1, robot_config_path) == 448, "planner path offset");
INVAR_PLANNER_ASSERT(sizeof(inv_planner_request{joints}_f64_v2) == {request_size}, "planner request size");
INVAR_PLANNER_ASSERT(INVAR_PLANNER_ALIGNOF(inv_planner_request{joints}_f64_v2) == 8, "planner request alignment");
INVAR_PLANNER_ASSERT(sizeof(inv_planner_waypoint{joints}_f64_v1) == {waypoint_size}, "planner waypoint size");
INVAR_PLANNER_ASSERT(sizeof(inv_planner_trajectory{joints}_f64_v1) == {trajectory_size}, "planner trajectory size");
INVAR_PLANNER_ASSERT(offsetof(inv_planner_trajectory{joints}_f64_v1, count) == 8, "planner count offset");
INVAR_PLANNER_ASSERT(offsetof(inv_planner_trajectory{joints}_f64_v1, waypoints) == 16, "planner waypoints offset");
/* Finite f64 only. phase is 0 free, 1 approach (cube obstacle off), 2 held,
 * 3 held with only the cube's table-contact slab uncovered, 4 home (joint goal
 * goal_q_rad); every other phase repeats start_q_rad in goal_q_rad.
 * Successful trajectories have 2..256
 * waypoints, finite positive dt, monotonic times, valid position/speed limits.
 * Failure never publishes partially written output. The source segments[4]
 * of 64 waypoints flatten in index order; the legacy trajectory is unchanged.
 */
#undef INVAR_PLANNER_ASSERT
#undef INVAR_PLANNER_ALIGNOF
#endif
"#
    ))
}
