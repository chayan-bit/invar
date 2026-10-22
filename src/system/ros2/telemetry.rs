//! Versioned coherent physics-step snapshot; no asynchronous topic join.
use super::{GenerateError, invalid, required};
use serde_json::{Value, json};
use std::collections::BTreeMap;

pub(super) fn joints(robot: &Value) -> Result<Vec<String>, GenerateError> {
    let names = robot["actuators"]
        .as_array()
        .ok_or_else(|| invalid("S1504", "missing telemetry actuators"))?
        .iter()
        .filter(|a| a["mode"] == "position" && a["target_kind"] == "joint")
        .map(|a| required(a, "target").map(str::to_owned))
        .collect::<Result<Vec<_>, _>>()?;
    if !matches!(names.len(), 6 | 7) {
        return Err(invalid("S1504", "unsupported telemetry joint count"));
    }
    Ok(names)
}
pub(super) fn add_message(messages: &mut BTreeMap<String, String>, n: usize) {
    messages.insert(format!("CellSnapshot{n}V1"),format!(
        "# Forwarded copy of one completed physics step; world poses, source-bound joint order.\nuint64 simulation_ns\nuint64 step_index\nuint8[32] source_ir_sha256\nuint8[32] joint_order_sha256\nfloat64[{n}] q_rad\nfloat64[{n}] dq_rad_s\nfloat64[7] world_from_tool\nfloat64[7] world_from_cube\nuint32 attached\nuint8[<=4096] geom_classes\nint32[<=2048] contact_geom_ids\nuint8[32] robot_joint_order_sha256\nfloat64[<=64] robot_q_si\nfloat64[<=64] robot_dq_si_per_s\n"));
}
pub(super) fn descriptor(robot: &Value, hash: &str) -> Result<Value, GenerateError> {
    let names = joints(robot)?;
    let root = robot["links"]
        .as_array()
        .and_then(|links| links.iter().find(|l| l["parent"] == "world"))
        .and_then(|l| l["name"].as_str())
        .ok_or_else(|| invalid("S1504", "telemetry robot root missing"))?;
    let order =
        serde_json::to_string(&names).map_err(|_| invalid("S1500", "joint order serialization"))?;
    let mut full = Vec::new();
    for joint in robot["joints"]
        .as_array()
        .ok_or_else(|| invalid("S1504", "missing robot joints"))?
    {
        let unit = match joint["kind"].as_str() {
            Some("fixed") => continue,
            Some("revolute") => "rad",
            Some("prismatic") => "m",
            _ => return Err(invalid("S1504", "unsupported full robot joint kind")),
        };
        let number = |v: &Value| {
            required(v, "value")?
                .parse::<f64>()
                .ok()
                .filter(|n| n.is_finite())
                .ok_or_else(|| invalid("S1504", "finite robot joint limit required"))
        };
        let lo = number(&joint["position"]["lower"])?;
        let hi = number(&joint["position"]["upper"])?;
        let speed = number(&joint["velocity"])?;
        if lo >= hi || speed <= 0.0 {
            return Err(invalid("S1504", "invalid full robot joint limits"));
        }
        full.push(json!({"name":joint["name"],"kind":joint["kind"],"position_unit":unit,"position_limits_si":[lo,hi],"velocity_limit_si_per_s":speed}));
    }
    if full.is_empty() || full.len() > 64 {
        return Err(invalid("S1504", "full robot joint capacity"));
    }
    let all_names = full.iter().map(|j| j["name"].clone()).collect::<Vec<_>>();
    let all_order = serde_json::to_string(&all_names)
        .map_err(|_| invalid("S1500", "full joint order serialization"))?;
    Ok(
        json!({"schema":"invar-cell-snapshot-v1","source_ir_sha256":hash,"selected_robot":robot["id"],
        "pinned_robot_asset_sha256":robot["asset"]["sha256"],"joint_names":names,"joint_order_sha256":super::sha256_hex(order.as_bytes()),
        "robot_joints":full,"robot_joint_order_sha256":super::sha256_hex(all_order.as_bytes()),
        "robot_root":root,"pose_frame":"world","topic":"/invar/cell_snapshot","joint_count":names.len(),
        "max_geoms":4096,"max_contact_pairs":1024,"geom_classes":{"environment":0,"robot":1,"table":2,"object":3},
        "provenance":"checked source binding; not a compiled-model byte attestation"}),
    )
}
pub(super) fn bytes_literal(hex: &str) -> Result<String, GenerateError> {
    if hex.len() != 64 || !hex.bytes().all(|b| b.is_ascii_hexdigit()) {
        return Err(invalid("S1504", "invalid telemetry binding digest"));
    }
    Ok((0..32)
        .map(|i| format!("0x{}", &hex[2 * i..2 * i + 2]))
        .collect::<Vec<_>>()
        .join(","))
}
