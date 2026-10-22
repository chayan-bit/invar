//! Count-specific bounded planner messages and RIHS01 descriptions.
use std::collections::{BTreeMap, BTreeSet};

pub(super) fn for_port(ty: &str, messages: &mut BTreeMap<String, String>) -> Option<String> {
    let n = crate::robotics::planner_joint_count(ty)?;
    let (head, _) = crate::robotics::generic_parts(ty)?;
    match head {
        "std::robotics::PlannerRequest" => {
            let name = format!("PlannerRequest{n}V2");
            messages.insert(name.clone(), format!(
                "# Poses are xyz metres followed by quaternion xyzw. Frames are checked in source.\n# phase: 0 free, 1 approach, 2 held, 3 held_contact, 4 home (joint goal goal_q_rad).\nfloat64[{n}] start_q_rad\nfloat64[{n}] goal_q_rad\nfloat64[7] target_base_from_tool\nfloat64[7] cube_base_pose\nfloat64[7] tool_from_cube\nuint32 phase\n"
            ));
            Some(name)
        }
        "std::robotics::PlannerTrajectory" => {
            let waypoint = format!("PlannerWaypoint{n}V1");
            messages.insert(
                waypoint.clone(),
                format!("float64 time_from_start_s\nfloat64[{n}] q_rad\nfloat64[{n}] dq_rad_s\n"),
            );
            let name = format!("PlannerTrajectory{n}V1");
            // ROS sequence length is the wire count; converters must require 2..=256.
            messages.insert(name.clone(), format!(
                "# Successful output requires 2..256 finite, checked waypoints.\nfloat64 dt_s\n{waypoint}[<=256] waypoints\n"
            ));
            Some(name)
        }
        _ => None,
    }
}

// RIHS01 uses insertion-ordered JSON and Python's ", "/": " separators.
// Field IDs and ordering follow rosidl_generator_type_description and
// type_description_interfaces/FieldType.msg; actual ROS build parity is a gate.
pub(super) fn type_hash(
    name: &str,
    messages: &BTreeMap<String, String>,
) -> Result<String, &'static str> {
    fn description(
        name: &str,
        messages: &BTreeMap<String, String>,
        references: &mut BTreeSet<String>,
    ) -> Result<String, &'static str> {
        let body = messages.get(name).ok_or("missing planner message")?;
        let mut fields = Vec::new();
        for line in body.lines().filter(|line| !line.starts_with('#')) {
            let (ty, field) = line.split_once(' ').ok_or("invalid planner field")?;
            let (id, capacity, nested) = if ty == "float64" {
                (11, 0, String::new())
            } else if ty == "uint32" {
                (7, 0, String::new())
            } else if ty == "uint64" {
                (9, 0, String::new())
            } else if let Some(size) = ty
                .strip_prefix("uint8[")
                .and_then(|s| s.strip_suffix(']'))
                .filter(|s| !s.starts_with("<="))
            {
                (
                    51,
                    size.parse::<u32>().map_err(|_| "invalid byte array")?,
                    String::new(),
                )
            } else if ty == "uint8[<=4096]" {
                (99, 4096, String::new())
            } else if ty == "int32[<=2048]" {
                (102, 2048, String::new())
            } else if ty == "float64[<=64]" {
                (107, 64, String::new())
            } else if let Some(size) = ty
                .strip_prefix("float64[")
                .and_then(|v| v.strip_suffix(']'))
            {
                (
                    59,
                    size.parse::<u32>().map_err(|_| "invalid planner array")?,
                    String::new(),
                )
            } else if let Some(inner) = ty.strip_suffix("[<=256]") {
                if !messages.contains_key(inner) {
                    return Err("missing nested planner message");
                }
                references.insert(inner.to_owned());
                (97, 256, format!("invar_cell_interfaces/msg/{inner}"))
            } else {
                return Err("unsupported planner message field");
            };
            fields.push(format!(
                "{{\"name\": \"{field}\", \"type\": {{\"type_id\": {id}, \"capacity\": {capacity}, \"string_capacity\": 0, \"nested_type_name\": \"{nested}\"}}}}"
            ));
        }
        Ok(format!(
            "{{\"type_name\": \"invar_cell_interfaces/msg/{name}\", \"fields\": [{}]}}",
            fields.join(", ")
        ))
    }
    let mut references = BTreeSet::new();
    let main = description(name, messages, &mut references)?;
    let mut descriptions = BTreeMap::new();
    while let Some(reference) = references.pop_first() {
        if descriptions.contains_key(&reference) {
            continue;
        }
        if descriptions.len() >= 3 {
            return Err("planner message reference bound exceeded");
        }
        let value = description(&reference, messages, &mut references)?;
        descriptions.insert(reference, value);
    }
    let input = format!(
        "{{\"type_description\": {main}, \"referenced_type_descriptions\": [{}]}}",
        descriptions.into_values().collect::<Vec<_>>().join(", ")
    );
    Ok(format!("RIHS01_{}", super::sha256_hex(input.as_bytes())))
}
