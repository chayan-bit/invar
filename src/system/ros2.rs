//! Deterministic ROS 2 Lyrical workspace files from checked system IR.

use crate::hex::sha256_hex;
use std::collections::{BTreeMap, BTreeSet};

use serde_json::{Value, json};

use super::SystemIr;

mod controller_bridge;
mod planner_messages;
mod planner_node;
mod sequencer_node;
mod telemetry;

const MAX_FILES: usize = 256;
const MAX_BYTES: usize = 4 * 1024 * 1024;

#[derive(Debug)]
pub struct GenerateError {
    pub code: &'static str,
    pub message: String,
}

fn invalid(code: &'static str, message: impl Into<String>) -> GenerateError {
    GenerateError {
        code,
        message: message.into(),
    }
}

fn required<'a>(value: &'a Value, key: &str) -> Result<&'a str, GenerateError> {
    value[key]
        .as_str()
        .ok_or_else(|| invalid("S1500", format!("checked IR lacks {key}")))
}

fn ros_name(name: &str) -> Result<String, GenerateError> {
    if name.len() > 128 || name.is_empty() || !name.is_ascii() {
        return Err(invalid("S1502", "ROS name must be bounded ASCII"));
    }
    let mut result = String::new();
    for (index, character) in name.chars().enumerate() {
        if character.is_ascii_uppercase() {
            if index > 0 && !result.ends_with('_') {
                result.push('_');
            }
            result.push(character.to_ascii_lowercase());
        } else if character.is_ascii_lowercase() || character.is_ascii_digit() || character == '_' {
            result.push(character);
        } else {
            return Err(invalid(
                "S1502",
                format!("'{name}' has no ROS name mapping"),
            ));
        }
    }
    if !result.starts_with(|c: char| c.is_ascii_lowercase())
        || result.ends_with('_')
        || result.contains("__")
    {
        return Err(invalid(
            "S1502",
            format!("'{name}' has no ROS name mapping"),
        ));
    }
    Ok(result)
}

fn unit_message(unit: &str) -> Result<String, GenerateError> {
    if unit.is_empty() || unit.len() > 32 || !unit.bytes().all(|byte| byte.is_ascii_alphanumeric())
    {
        return Err(invalid("S1501", "unit has no ROS message mapping"));
    }
    let mut chars = unit.chars();
    let first = chars
        .next()
        .ok_or_else(|| invalid("S1501", "unit missing"))?;
    Ok(format!("{}{}", first.to_ascii_uppercase(), chars.as_str()))
}

fn port_message(
    ty: &str,
    generated: &mut BTreeMap<String, (&'static str, u8)>,
    planner: &mut BTreeMap<String, String>,
) -> Result<String, GenerateError> {
    if let Some(name) = planner_messages::for_port(ty, planner) {
        return Ok(format!("invar_cell_interfaces/msg/{name}"));
    }
    if let Some(inner) = ty
        .strip_prefix("Quantity<")
        .and_then(|value| value.strip_suffix('>'))
    {
        let (scalar, unit) = inner
            .split_once(',')
            .ok_or_else(|| invalid("S1501", "invalid quantity port"))?;
        let (ros_scalar, type_id, title) = match scalar {
            "i32" => ("int32", 6, "I32"),
            "u32" => ("uint32", 7, "U32"),
            "f32" => ("float32", 10, "F32"),
            "f64" => ("float64", 11, "F64"),
            _ => {
                return Err(invalid(
                    "S1501",
                    format!("'{ty}' has no ROS representation"),
                ));
            }
        };
        let name = format!("Quantity{title}{}", unit_message(unit)?);
        generated.insert(name.clone(), (ros_scalar, type_id));
        return Ok(format!("invar_cell_interfaces/msg/{name}"));
    }
    if ty.starts_with("Transform<") && ty.ends_with('>') {
        return Ok("geometry_msgs/msg/TransformStamped".to_owned());
    }
    if let Some(record) = ty.strip_prefix("std::robotics::") {
        let stem = record.split_once('<').map_or(record, |(stem, _)| stem);
        let mappings = crate::robotics::ros_mapping()
            .map_err(|_| invalid("S1500", "invalid checked ROS mapping table"))?;
        if let Some(ros) = mappings["records"]
            .as_array()
            .and_then(|rows| rows.iter().find(|row| row["invar"] == stem))
            .and_then(|row| row["ros"].as_str())
        {
            return Ok(ros.to_owned());
        }
    }
    Err(invalid(
        "S1501",
        format!("'{ty}' has no ROS representation"),
    ))
}

fn type_hash(package: &str, name: &str, type_id: u8) -> String {
    let description = format!(
        "{{\"type_description\": {{\"type_name\": \"{package}/msg/{name}\", \"fields\": [{{\"name\": \"value\", \"type\": {{\"type_id\": {type_id}, \"capacity\": 0, \"string_capacity\": 0, \"nested_type_name\": \"\"}}}}]}}, \"referenced_type_descriptions\": []}}"
    );
    format!("RIHS01_{}", sha256_hex(description.as_bytes()))
}

fn header(path: &str) -> &'static str {
    if path.ends_with(".xml") || path.ends_with(".urdf") {
        "xml"
    } else if path.ends_with(".cpp") || path.ends_with(".hpp") || path.ends_with(".h") {
        "cpp"
    } else {
        "line"
    }
}

fn insert(
    files: &mut BTreeMap<String, String>,
    path: impl Into<String>,
    body: String,
    hash: &str,
) -> Result<(), GenerateError> {
    let path = path.into();
    if files.len() >= MAX_FILES
        || path.starts_with('/')
        || path
            .split('/')
            .any(|part| part.is_empty() || part == "." || part == "..")
    {
        return Err(invalid(
            "S1500",
            "generated file-set limit or path violation",
        ));
    }
    let stamp = format!("Invar ROS 2 generator v1; IR sha256 {hash}; generated, do not edit");
    let content = match header(&path) {
        "xml" if body.starts_with("<?xml") => {
            let (declaration, rest) = body
                .split_once('\n')
                .ok_or_else(|| invalid("S1500", "invalid generated XML"))?;
            format!("{declaration}\n<!-- {stamp} -->\n{rest}")
        }
        "xml" => format!("<!-- {stamp} -->\n{body}"),
        "cpp" => format!("// {stamp}\n{body}"),
        _ if path.ends_with(".json5") => format!("// {stamp}\n{body}"),
        _ if path.ends_with(".json") => body,
        _ => format!("# {stamp}\n{body}"),
    };
    if files.insert(path, content).is_some() {
        return Err(invalid("S1503", "two generated files have the same name"));
    }
    if files.values().map(String::len).sum::<usize>() > MAX_BYTES {
        return Err(invalid("S1500", "generated file-set size limit exceeded"));
    }
    Ok(())
}

fn parameter_yaml(component: &Value) -> Result<String, GenerateError> {
    let name = ros_name(required(component, "name")?)?;
    let parameters = component["params"]
        .as_object()
        .ok_or_else(|| invalid("S1500", "invalid checked parameters"))?;
    let mut yaml = format!("{name}:\n");
    if parameters.is_empty() {
        yaml.push_str("  {}\n");
    }
    for (key, value) in parameters {
        let key = ros_name(key)?;
        let (kind, default, description) = if value["kind"] == "identity-transform" {
            (
                "double_array",
                "[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]".to_owned(),
                format!(
                    "Identity transform {} to {}",
                    required(value, "from")?,
                    required(value, "to")?
                ),
            )
        } else {
            let unit = required(value, "unit")?;
            (
                "double",
                required(value, "value")?.to_owned(),
                format!("Invar unit: {unit}"),
            )
        };
        yaml.push_str(&format!("  {key}:\n    type: {kind}\n    default_value: {default}\n    description: {description:?}\n    read_only: true\n"));
        if let Some(range) = value.get("range") {
            yaml.push_str(&format!(
                "    validation:\n      bounds<>: [{}, {}]\n",
                required(range, "min")?,
                required(range, "max")?
            ));
        }
    }
    Ok(yaml)
}

fn qos_depth(policy: &str) -> Result<u16, GenerateError> {
    if policy == "sample-and-hold" {
        return Ok(1);
    }
    let depth = policy
        .strip_prefix("queue(")
        .and_then(|value| value.strip_suffix(')'))
        .and_then(|value| value.parse::<u16>().ok());
    match depth {
        Some(value @ 1..=256) => Ok(value),
        _ => Err(invalid(
            "S1505",
            format!("'{policy}' has no ROS QoS mapping"),
        )),
    }
}

fn control_files(
    files: &mut BTreeMap<String, String>,
    hash: &str,
    controller: &Value,
    robot: Option<&Value>,
) -> Result<(), GenerateError> {
    let rate = controller["ports"]
        .as_array()
        .and_then(|ports| ports.first())
        .and_then(|port| port["rate"]["value"].as_str())
        .and_then(|value| value.parse::<u16>().ok())
        .filter(|rate| (1..=1000).contains(rate))
        .ok_or_else(|| invalid("S1504", "controller needs an integral 1..1000 Hz port rate"))?;
    if let Some(robot) = robot {
        return cell_control_files(files, hash, rate, robot);
    }
    let xml = "<ros2_control name=\"InvarHardware\" type=\"system\">\n  <hardware><plugin>mujoco_ros2_control/MujocoSystemInterface</plugin></hardware>\n  <joint name=\"joint1\">\n    <command_interface name=\"position\"/>\n    <state_interface name=\"position\"/>\n    <state_interface name=\"velocity\"/>\n  </joint>\n</ros2_control>\n";
    insert(
        files,
        "src/invar_cell_bringup/config/ros2_control.fragment.urdf",
        xml.to_owned(),
        hash,
    )?;
    let robot = format!(
        "<?xml version=\"1.0\"?>\n<robot name=\"invar_cell\">\n  <link name=\"base_link\"/>\n  <link name=\"joint_link\"/>\n  <joint name=\"joint1\" type=\"revolute\">\n    <parent link=\"base_link\"/><child link=\"joint_link\"/>\n    <origin xyz=\"0 0 0\" rpy=\"0 0 0\"/><axis xyz=\"0 0 1\"/>\n    <limit lower=\"-3.14\" upper=\"3.14\" effort=\"10\" velocity=\"1\"/>\n  </joint>\n{xml}</robot>\n"
    );
    insert(
        files,
        "src/invar_cell_bringup/config/ros2_control.urdf",
        robot,
        hash,
    )?;
    let yaml = format!(
        "controller_manager:\n  ros__parameters:\n    update_rate: {rate}\n    joint_state_broadcaster:\n      type: joint_state_broadcaster/JointStateBroadcaster\n    joint_trajectory_controller:\n      type: joint_trajectory_controller/JointTrajectoryController\njoint_trajectory_controller:\n  ros__parameters:\n    joints: [joint1]\n    command_interfaces: [position]\n    state_interfaces: [position, velocity]\n"
    );
    insert(
        files,
        "src/invar_cell_bringup/config/controllers.yaml",
        yaml,
        hash,
    )
}

fn cell_control_files(
    files: &mut BTreeMap<String, String>,
    hash: &str,
    rate: u16,
    robot: &Value,
) -> Result<(), GenerateError> {
    fn pose_origin(pose: &Value) -> Result<String, GenerateError> {
        let xyz = pose["xyz"]
            .as_array()
            .ok_or_else(|| invalid("S1504", "robot pose is missing"))?
            .iter()
            .map(|part| required(part, "value"))
            .collect::<Result<Vec<_>, _>>()?
            .join(" ");
        let q = pose["quat"]
            .as_array()
            .ok_or_else(|| invalid("S1504", "robot rotation is missing"))?
            .iter()
            .map(|part| part.as_str().and_then(|part| part.parse::<f64>().ok()))
            .collect::<Option<Vec<_>>>()
            .ok_or_else(|| invalid("S1504", "robot rotation is invalid"))?;
        if q.len() != 4 || q.iter().any(|part| !part.is_finite()) {
            return Err(invalid("S1504", "robot rotation is invalid"));
        }
        let norm = q.iter().map(|part| part * part).sum::<f64>().sqrt();
        if norm < 1e-12 || !norm.is_finite() {
            return Err(invalid("S1504", "robot rotation has no finite orientation"));
        }
        let (w, x, y, z) = (q[0] / norm, q[1] / norm, q[2] / norm, q[3] / norm);
        let roll = (2.0 * (w * x + y * z)).atan2(1.0 - 2.0 * (x * x + y * y));
        let pitch = (2.0 * (w * y - z * x)).clamp(-1.0, 1.0).asin();
        let yaw = (2.0 * (w * z + x * y)).atan2(1.0 - 2.0 * (y * y + z * z));
        Ok(format!(
            "<origin xyz=\"{xyz}\" rpy=\"{roll} {pitch} {yaw}\"/>"
        ))
    }
    let links = robot["links"]
        .as_array()
        .ok_or_else(|| invalid("S1504", "robot links are missing"))?;
    let joints = robot["joints"]
        .as_array()
        .ok_or_else(|| invalid("S1504", "robot joints are missing"))?;
    let actuators = robot["actuators"]
        .as_array()
        .ok_or_else(|| invalid("S1504", "robot actuators are missing"))?;
    let active = actuators
        .iter()
        .filter(|actuator| actuator["mode"] == "position" && actuator["target_kind"] == "joint")
        .map(|actuator| required(actuator, "target"))
        .collect::<Result<Vec<_>, _>>()?;
    if !(6..=7).contains(&active.len())
        || active.iter().collect::<BTreeSet<_>>().len() != active.len()
        || active
            .iter()
            .any(|name| !joints.iter().any(|joint| joint["name"] == **name))
    {
        return Err(invalid(
            "S1504",
            "cell controller needs six or seven distinct position joints",
        ));
    }
    let root = links
        .iter()
        .find(|link| link["parent"] == "world")
        .ok_or_else(|| invalid("S1504", "robot root link is missing"))?;
    let mut description = format!(
        "<?xml version=\"1.0\"?>\n<robot name=\"{}\">\n<link name=\"world\"/>\n",
        required(robot, "name")?
    );
    for link in links {
        description.push_str(&format!("<link name=\"{}\"/>\n", required(link, "name")?));
    }
    description.push_str(&format!("<joint name=\"world_to_{}\" type=\"fixed\"><parent link=\"world\"/><child link=\"{}\"/>{}</joint>\n", required(root, "name")?, required(root, "name")?, pose_origin(&root["origin"])?));
    for joint in joints {
        let name = required(joint, "name")?;
        let child = required(joint, "child")?;
        let link = links
            .iter()
            .find(|link| link["name"] == child)
            .ok_or_else(|| invalid("S1504", "joint child link is missing"))?;
        let kind = required(joint, "kind")?;
        let kind = if kind == "revolute" && joint["position"].is_null() {
            "continuous"
        } else {
            kind
        };
        description.push_str(&format!("<joint name=\"{name}\" type=\"{kind}\"><parent link=\"{}\"/><child link=\"{child}\"/>{}", required(joint, "parent")?, pose_origin(&link["origin"])?));
        if kind != "fixed" {
            let axis = joint["axis"]
                .as_array()
                .ok_or_else(|| invalid("S1504", "joint axis is missing"))?
                .iter()
                .map(|part| part.as_str())
                .collect::<Option<Vec<_>>>()
                .ok_or_else(|| invalid("S1504", "joint axis is invalid"))?
                .join(" ");
            description.push_str(&format!("<axis xyz=\"{axis}\"/>"));
            let lower = joint["position"]["lower"]["value"].as_str();
            let upper = joint["position"]["upper"]["value"].as_str();
            let effort = joint["effort"]["value"]
                .as_str()
                .ok_or_else(|| invalid("S1504", "ROS joint needs a finite effort limit"))?;
            let velocity = joint["velocity"]["value"]
                .as_str()
                .ok_or_else(|| invalid("S1504", "ROS joint needs a finite velocity limit"))?;
            description.push_str(&format!(
                "<limit effort=\"{effort}\" velocity=\"{velocity}\"{}{} />",
                lower.map_or(String::new(), |value| format!(" lower=\"{value}\"")),
                upper.map_or(String::new(), |value| format!(" upper=\"{value}\""))
            ));
        }
        description.push_str("</joint>\n");
    }
    for link in links.iter().filter(|link| {
        link["parent"] != "world" && !joints.iter().any(|joint| joint["child"] == link["name"])
    }) {
        let name = required(link, "name")?;
        description.push_str(&format!("<joint name=\"invar_fixed_{name}\" type=\"fixed\"><parent link=\"{}\"/><child link=\"{name}\"/>{}</joint>\n", required(link, "parent")?, pose_origin(&link["origin"])?));
    }
    if !robot["tool"].is_null() {
        description.push_str(&format!("<link name=\"invar_tool\"/><joint name=\"invar_tool_joint\" type=\"fixed\"><parent link=\"{}\"/><child link=\"invar_tool\"/>{}</joint>\n", required(&robot["tool"], "link")?, pose_origin(&robot["tool"]["pose"])?));
    }
    let mut control = String::from(
        "<ros2_control name=\"InvarHardware\" type=\"system\"><hardware><plugin>mujoco_ros2_control/MujocoSystemInterface</plugin><param name=\"mujoco_model\">@INVAR_MJCF_PATH@</param><param name=\"headless\">true</param><param name=\"initial_keyframe\">home</param></hardware>",
    );
    for name in &active {
        control.push_str(&format!("<joint name=\"{name}\"><command_interface name=\"position\"/><state_interface name=\"position\"/><state_interface name=\"velocity\"/></joint>"));
    }
    control.push_str("</ros2_control>\n");
    description.push_str(&control);
    description.push_str("</robot>\n");
    insert(
        files,
        "src/invar_cell_bringup/config/ros2_control.fragment.urdf",
        control,
        hash,
    )?;
    insert(
        files,
        "src/invar_cell_bringup/config/ros2_control.urdf",
        description,
        hash,
    )?;
    let yaml = format!(
        "controller_manager:\n  ros__parameters:\n    update_rate: {rate}\n    joint_state_broadcaster:\n      type: joint_state_broadcaster/JointStateBroadcaster\n    joint_trajectory_controller:\n      type: joint_trajectory_controller/JointTrajectoryController\njoint_trajectory_controller:\n  ros__parameters:\n    joints: [{}]\n    command_interfaces: [position]\n    state_interfaces: [position, velocity]\n",
        active.join(", ")
    );
    insert(
        files,
        "src/invar_cell_bringup/config/controllers.yaml",
        yaml,
        hash,
    )
}

fn attachment_plugin_files(
    files: &mut BTreeMap<String, String>,
    hash: &str,
    object: &str,
    robot: &Value,
) -> Result<(), GenerateError> {
    if object.is_empty()
        || object.len() > 128
        || !object
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || byte == b'_')
    {
        return Err(invalid(
            "S1504",
            "attachment object has no safe plugin name",
        ));
    }
    let descriptor = telemetry::descriptor(robot, hash)?;
    let n = descriptor["joint_count"]
        .as_u64()
        .ok_or_else(|| invalid("S1504", "missing telemetry count"))?;
    let names = telemetry::joints(robot)?
        .iter()
        .map(|s| serde_json::to_string(s).map_err(|_| invalid("S1500", "joint literal")))
        .collect::<Result<Vec<_>, _>>()?
        .join(",");
    let source = include_str!("ros2_templates/attachment_plugin.cpp")
        .replace("@OBJECT@", object)
        .replace("@JOINTS@", &n.to_string())
        .replace("@JOINT_NAMES@", &names)
        .replace(
            "@ROBOT_JOINTS@",
            &descriptor["robot_joints"]
                .as_array()
                .ok_or_else(|| invalid("S1504", "full joint descriptors"))?
                .len()
                .to_string(),
        )
        .replace(
            "@ROBOT_JOINT_NAMES@",
            &descriptor["robot_joints"]
                .as_array()
                .ok_or_else(|| invalid("S1504", "full joint descriptors"))?
                .iter()
                .map(|j| {
                    serde_json::to_string(&j["name"])
                        .map_err(|_| invalid("S1500", "full joint literal"))
                })
                .collect::<Result<Vec<_>, _>>()?
                .join(","),
        )
        .replace(
            "@ROBOT_JOINT_TYPES@",
            &descriptor["robot_joints"]
                .as_array()
                .ok_or_else(|| invalid("S1504", "full joint descriptors"))?
                .iter()
                .map(|j| {
                    if j["kind"] == "prismatic" {
                        "mjJNT_SLIDE"
                    } else {
                        "mjJNT_HINGE"
                    }
                })
                .collect::<Vec<_>>()
                .join(","),
        )
        .replace(
            "@ROBOT_ORDER_BYTES@",
            &telemetry::bytes_literal(required(&descriptor, "robot_joint_order_sha256")?)?,
        )
        .replace("@ROBOT_ROOT@", required(&descriptor, "robot_root")?)
        .replace("@IR_BYTES@", &telemetry::bytes_literal(hash)?)
        .replace(
            "@ORDER_BYTES@",
            &telemetry::bytes_literal(required(&descriptor, "joint_order_sha256")?)?,
        );
    insert(
        files,
        "src/invar_cell_bringup/config/cell-snapshot-v1.json",
        format!("{descriptor}\n"),
        hash,
    )?;
    insert(
        files,
        "src/invar_cell_bringup/src/attachment_plugin.cpp",
        source,
        hash,
    )?;
    insert(files, "src/invar_cell_bringup/attachment_plugin.xml", "<library path=\"invar_attachment_plugin\"><class name=\"invar_cell_bringup/AttachmentPlugin\" type=\"invar_cell_bringup::AttachmentPlugin\" base_class_type=\"mujoco_ros2_control_plugins::MuJoCoROS2ControlPluginBase\"/></library>\n".to_owned(), hash)?;
    insert(files, "src/invar_cell_bringup/config/attachment_plugin.yaml", "/**:\n  ros__parameters:\n    mujoco_plugins:\n      invar_attachment:\n        type: invar_cell_bringup/AttachmentPlugin\n        update_rate: 100.0\n".to_owned(), hash)
}

fn foreign_node(
    files: &mut BTreeMap<String, String>,
    hash: &str,
    name: &str,
    component: &Value,
    connections: &[Value],
    robot: Option<&Value>,
) -> Result<(String, bool), GenerateError> {
    let package = format!("invar_{name}_node");
    let namespace = format!("invar_generated_{name}");
    let prefix = required(&component["abi"], "prefix")?;
    let isolation = required(component, "isolation")?;
    let separate = isolation == "separate-process";
    if !separate && isolation != "in-process" {
        return Err(invalid(
            "S1500",
            "foreign isolation has no ROS node mapping",
        ));
    }
    let ports = component["ports"]
        .as_array()
        .ok_or_else(|| invalid("S1500", "missing foreign ports"))?;
    let input = ports
        .iter()
        .find(|port| port["direction"] == "in")
        .ok_or_else(|| invalid("S1500", "foreign input missing"))?;
    let output = ports
        .iter()
        .find(|port| port["direction"] == "out")
        .ok_or_else(|| invalid("S1500", "foreign output missing"))?;
    let input_name = required(input, "name")?;
    let output_name = required(output, "name")?;
    if ros_name(input_name)? != input_name || ros_name(output_name)? != output_name {
        return Err(invalid(
            "S1502",
            "foreign port name must be a ROS-compatible C identifier",
        ));
    }
    if component["config"]["layout"] == "inv_planner_config_v1" {
        return planner_node::generate(files, hash, name, component, connections, robot);
    }
    let joints = match required(input, "layout")? {
        "inv_joint2_f64" => 2,
        "inv_joint6_f64" => 6,
        "inv_joint7_f64" => 7,
        _ => {
            return Err(invalid(
                "S1501",
                "foreign input layout has no ROS conversion",
            ));
        }
    };
    if required(output, "layout")? != "inv_pose_f64" {
        return Err(invalid(
            "S1501",
            "foreign output layout has no ROS conversion",
        ));
    }
    let input_path = format!("{name}.{input_name}");
    let output_path = format!("{name}.{output_name}");
    let input_depth = connections
        .iter()
        .find(|edge| edge["target"] == input_path)
        .map(|edge| required(edge, "policy").and_then(qos_depth))
        .transpose()?
        .unwrap_or(1);
    let output_depth = connections
        .iter()
        .find(|edge| edge["source"] == output_path)
        .map(|edge| required(edge, "policy").and_then(qos_depth))
        .transpose()?
        .unwrap_or(1);
    let joint_names = if component["config"]["layout"] == "inv_fk_config_v1" {
        robot
            .and_then(|robot| robot["actuators"].as_array())
            .map(|actuators| {
                actuators
                    .iter()
                    .filter(|actuator| {
                        actuator["mode"] == "position" && actuator["target_kind"] == "joint"
                    })
                    .map(|actuator| required(actuator, "target").map(str::to_owned))
                    .collect::<Result<Vec<_>, _>>()
            })
            .transpose()?
    } else {
        None
    };
    if joint_names
        .as_ref()
        .is_some_and(|names| names.len() != joints)
    {
        return Err(invalid(
            "S1504",
            "foreign joint layout differs from selected robot",
        ));
    }
    let joint_names =
        joint_names.unwrap_or_else(|| (1..=joints).map(|index| format!("joint{index}")).collect());
    let joint_names = joint_names
        .iter()
        .map(|name| format!("\"{name}\""))
        .collect::<Vec<_>>()
        .join(", ");
    let header = super::foreign::header(component).map_err(|message| invalid("S1500", message))?;
    let projection = if component["config"]["layout"] == "inv_fk_config_v1" {
        robot
            .map(super::tool_projection::derive)
            .transpose()
            .map_err(|m| invalid("S1504", m))?
    } else {
        None
    };
    insert(
        files,
        format!("src/{package}/include/{package}/tool_projection.hpp"),
        super::tool_projection::header(projection.as_ref()),
        hash,
    )?;
    if let Some(p) = &projection {
        insert(
            files,
            format!("src/{package}/tool-projection.json"),
            super::tool_projection::metadata(p),
            hash,
        )?;
    }
    insert(
        files,
        format!("src/{package}/include/{package}/{prefix}.h"),
        header,
        hash,
    )?;
    let templates = [
        ("node.hpp", include_str!("ros2_templates/node.hpp.in")),
        (
            "conversions.hpp",
            include_str!("ros2_templates/conversions.hpp.in"),
        ),
        ("node.cpp", include_str!("ros2_templates/node.cpp.in")),
        ("main.cpp", include_str!("ros2_templates/main.cpp.in")),
    ];
    let config = match required(&component["config"], "layout")? {
        "inv_planar2_config_v1" => format!(
            "{prefix}_config config{{declare_parameter<double>(\"proximal_m\", 1.0), declare_parameter<double>(\"distal_m\", 1.0)}};"
        ),
        "inv_fk_config_v1" => format!(
            "const auto model_path = declare_parameter<std::string>(\"model_xml_path\", \"\");\n  const auto model_kind = declare_parameter<int>(\"model_kind\", {joints});\n  if (!std::filesystem::path(model_path).is_absolute() || model_path.size() > UINT32_MAX || model_kind != {joints})\n    throw std::runtime_error(\"invalid foreign model path or kind\");\n  @REFERENCE_MODEL_CHECK@\n  {prefix}_config config{{reinterpret_cast<const uint8_t *>(model_path.data()), static_cast<uint32_t>(model_path.size()), static_cast<uint32_t>(model_kind)}};"
        ),
        _ => {
            return Err(invalid(
                "S1500",
                "foreign config layout has no ROS node binding",
            ));
        }
    };
    let init_path = if separate {
        "const auto child_path = declare_parameter<std::string>(\"child_path\", \"\");\n  if (!std::filesystem::path(child_path).is_absolute() || child_path.size() > UINT32_MAX)\n    throw std::runtime_error(\"foreign child_path must be bounded and absolute\");\n  const auto * child_data = reinterpret_cast<const uint8_t *>(child_path.data());\n  const auto child_size = static_cast<uint32_t>(child_path.size());"
    } else {
        "const uint8_t * child_data = nullptr;\n  const uint32_t child_size = 0;"
    };
    for (file, source) in templates {
        let content = source
            .replace("@NAMESPACE@", &namespace)
            .replace("@NODE@", name)
            .replace("@PREFIX@", prefix)
            .replace("@JOINTS@", &joints.to_string())
            .replace("@JOINT_NAMES@", &joint_names)
            .replace("@INPUT@", input_name)
            .replace("@OUTPUT@", output_name)
            .replace("@INPUT_DEPTH@", &input_depth.to_string())
            .replace("@OUTPUT_DEPTH@", &output_depth.to_string())
            .replace("@CONFIG@", &config)
            .replace("@REFERENCE_MODEL_CHECK@", if projection.is_some() {"if (!invar_tool_projection_v1::reference_model_matches(model_path)) throw std::runtime_error(\"tool projection reference model mismatch\");"} else {""})
            .replace("@PROJECTION_VERSION_CHECK@", if projection.is_some() {"if (std::strcmp(version_text, \"invar-component-v1/invar-pinocchio-reference-abi-1/pinocchio-4.1.0\") != 0) throw std::runtime_error(\"tool projection requires pinned reference ABI\");"} else {""})
            .replace("@INIT_PATH@", init_path);
        let folder = if file.ends_with(".hpp") || file.ends_with(".h") {
            format!("include/{package}")
        } else {
            "src".to_owned()
        };
        insert(
            files,
            format!("src/{package}/{folder}/{file}"),
            content,
            hash,
        )?;
    }
    let mut cmake = format!(
        "cmake_minimum_required(VERSION 3.16)\nproject({package})\nfind_package(ament_cmake REQUIRED)\nfind_package(rclcpp REQUIRED)\nfind_package(rclcpp_components REQUIRED)\nfind_package(sensor_msgs REQUIRED)\nfind_package(geometry_msgs REQUIRED)\nadd_library(${{PROJECT_NAME}}_component SHARED src/node.cpp)\ntarget_compile_features(${{PROJECT_NAME}}_component PUBLIC cxx_std_17)\ntarget_include_directories(${{PROJECT_NAME}}_component PRIVATE include/${{PROJECT_NAME}})\ntarget_link_libraries(${{PROJECT_NAME}}_component rclcpp::rclcpp rclcpp_components::component sensor_msgs::sensor_msgs geometry_msgs::geometry_msgs dl)\nrclcpp_components_register_nodes(${{PROJECT_NAME}}_component \"{namespace}::ComponentNode\")\ninstall(TARGETS ${{PROJECT_NAME}}_component DESTINATION lib)\n"
    );
    if separate {
        cmake.push_str(&format!("add_executable({name}_node src/main.cpp)\ntarget_compile_features({name}_node PUBLIC cxx_std_17)\ntarget_include_directories({name}_node PRIVATE include/${{PROJECT_NAME}})\ntarget_link_libraries({name}_node ${{PROJECT_NAME}}_component rclcpp::rclcpp)\ninstall(TARGETS {name}_node DESTINATION lib/${{PROJECT_NAME}})\n"));
    }
    cmake.push_str("ament_package()\n");
    insert(files, format!("src/{package}/CMakeLists.txt"), cmake, hash)?;
    let package_xml = format!(
        "<?xml version=\"1.0\"?>\n<package format=\"3\"><name>{package}</name><version>0.1.0</version><description>Invar generated foreign component node</description><maintainer email=\"invar@example.invalid\">Invar</maintainer><license>Apache-2.0</license><buildtool_depend>ament_cmake</buildtool_depend><depend>rclcpp</depend><depend>rclcpp_components</depend><depend>sensor_msgs</depend><depend>geometry_msgs</depend><export><build_type>ament_cmake</build_type></export></package>\n"
    );
    insert(
        files,
        format!("src/{package}/package.xml"),
        package_xml,
        hash,
    )?;
    Ok((package, separate))
}

/// Build a bounded file-set without reading source, clock, environment or network.
pub fn generate(ir: &SystemIr) -> Result<BTreeMap<String, String>, GenerateError> {
    let document = ir
        .checked_document()
        .map_err(|message| invalid("S1500", message))?;
    super::substep_admission::ensure_export_supported(document)
        .map_err(|message| invalid("S1504", message))?;
    let hash = sha256_hex(ir.canonical_json().as_bytes());
    let system = required(&document["system"], "name")?;
    let _system_ros = ros_name(system)?;
    let components = document["components"]
        .as_array()
        .ok_or_else(|| invalid("S1500", "missing checked components"))?;
    let external = document["extern_components"]
        .as_array()
        .ok_or_else(|| invalid("S1500", "missing checked extern components"))?;
    let mut by_id = BTreeMap::new();
    for component in components.iter().chain(external) {
        by_id.insert(required(component, "id")?, component);
    }
    let instances = document["system"]["instances"]
        .as_array()
        .ok_or_else(|| invalid("S1500", "missing checked instances"))?;
    let cell_robot = if document["schema"] == "invar-system-v6"
        || (matches!(
            document["schema"].as_str(),
            Some(
                "invar-system-v7"
                    | "invar-system-v8"
                    | "invar-system-v9"
                    | "invar-system-v10"
                    | "invar-system-v11"
                    | "invar-system-v12"
            )
        ) && document["robots"].is_array())
    {
        let product_id = required(&document["system"], "product")?;
        let product = document["products"]
            .as_array()
            .and_then(|products| products.iter().find(|product| product["id"] == product_id))
            .ok_or_else(|| invalid("S1504", "selected product is missing"))?;
        let robot_id = required(product, "robot")?;
        Some(
            document["robots"]
                .as_array()
                .and_then(|robots| robots.iter().find(|robot| robot["id"] == robot_id))
                .ok_or_else(|| invalid("S1504", "selected cell robot is missing"))?,
        )
    } else {
        None
    };
    let connections = document["connections"]
        .as_array()
        .ok_or_else(|| invalid("S1500", "missing checked connections"))?;
    let mut names = BTreeSet::new();
    let mut messages = BTreeMap::new();
    let mut planner_messages = BTreeMap::new();
    if let Some(robot) = cell_robot
        && document["cells"].as_array().is_some_and(|cells| {
            cells.iter().any(|c| {
                c["objects"]
                    .as_array()
                    .is_some_and(|objects| objects.iter().any(|o| !o["dynamic"].is_null()))
            })
        })
    {
        telemetry::add_message(&mut planner_messages, telemetry::joints(robot)?.len());
    }
    let mut files = BTreeMap::new();
    let mut notices = String::new();
    let mut launch_args = String::new();
    let mut composables = Vec::new();
    let mut separate_nodes = Vec::new();
    let mut parameter_targets = String::new();
    for instance in instances {
        let name = ros_name(required(instance, "name")?)?;
        if !names.insert(name.clone()) {
            return Err(invalid(
                "S1503",
                "two components map to the same ROS node name",
            ));
        }
        let component = by_id
            .get(required(instance, "component")?)
            .ok_or_else(|| invalid("S1500", "missing checked component"))?;
        if component["params"]
            .as_object()
            .is_some_and(|params| !params.is_empty())
        {
            parameter_targets.push_str(&format!(
                "generate_parameter_library(invar_params_{name} config/params/{name}.yaml)\n"
            ));
        }
        for port in component["ports"]
            .as_array()
            .ok_or_else(|| invalid("S1500", "missing checked ports"))?
        {
            port_message(
                required(port, "type")?,
                &mut messages,
                &mut planner_messages,
            )?;
        }
        insert(
            &mut files,
            format!("src/invar_cell_bringup/config/params/{name}.yaml"),
            parameter_yaml(component)?,
            &hash,
        )?;
        notices.push_str(&format!(
            "{name}: {}\n",
            required(&component["license"], "expression")?
        ));
        if component["external"] == true {
            let (package, separate) =
                foreign_node(&mut files, &hash, &name, component, connections, cell_robot)?;
            launch_args.push_str(&format!(
                "    actions.append(DeclareLaunchArgument('{name}_library', default_value=''))\n"
            ));
            let mut parameter_entries = vec![format!(
                "'library_path': LaunchConfiguration('{name}_library')"
            )];
            if separate {
                launch_args.push_str(&format!(
                    "    actions.append(DeclareLaunchArgument('{name}_child', default_value=''))\n"
                ));
                parameter_entries
                    .push(format!("'child_path': LaunchConfiguration('{name}_child')"));
            }
            if component["config"]["layout"] == "inv_fk_config_v1" {
                launch_args.push_str(&format!(
                    "    actions.append(DeclareLaunchArgument('{name}_model', default_value=''))\n"
                ));
                parameter_entries.push(format!(
                    "'model_xml_path': LaunchConfiguration('{name}_model')"
                ));
            } else if component["config"]["layout"] == "inv_planar2_config_v1" {
                for side in ["proximal", "distal"] {
                    launch_args.push_str(&format!("    actions.append(DeclareLaunchArgument('{name}_{side}_m', default_value='1.0'))\n"));
                    parameter_entries.push(format!("'{side}_m': ParameterValue(LaunchConfiguration('{name}_{side}_m'), value_type=float)"));
                }
            }
            let parameters = format!("{{{}}}", parameter_entries.join(", "));
            let parameters = if component["config"]["layout"] == "inv_planner_config_v1" {
                launch_args.push_str(&format!(
                    "    actions.append(DeclareLaunchArgument('{name}_config', description='Required planner ROS parameter YAML'))\n"
                ));
                format!("LaunchConfiguration('{name}_config'), {parameters}")
            } else {
                parameters
            };
            let mut remaps = Vec::new();
            for connection in connections {
                let target = required(connection, "target")?;
                let Some((target_instance, target_port)) = target.split_once('.') else {
                    return Err(invalid("S1500", "invalid checked connection"));
                };
                if target_instance != required(instance, "name")? {
                    continue;
                }
                let source = required(connection, "source")?;
                let Some((source_instance, source_port)) = source.split_once('.') else {
                    return Err(invalid("S1500", "invalid checked connection"));
                };
                let source_topic = if source_instance == "joint_states"
                    && source_port == "q"
                    && instances
                        .iter()
                        .find(|candidate| candidate["name"] == source_instance)
                        .and_then(|candidate| candidate["component"].as_str())
                        .and_then(|id| by_id.get(id))
                        .is_some_and(|component| component["name"] == "joint_state_broadcaster")
                {
                    "/joint_states".to_owned()
                } else {
                    format!("/{}/{}", ros_name(source_instance)?, ros_name(source_port)?)
                };
                remaps.push(format!(
                    "('{}', '{}')",
                    ros_name(target_port)?,
                    source_topic
                ));
            }
            let remaps = format!("[{}]", remaps.join(", "));
            if separate {
                separate_nodes.push(format!("        Node(package='{package}', executable='{name}_node', name='{name}', namespace='{name}', parameters=[{parameters}], remappings={remaps}, output='screen'),\n"));
            } else {
                composables.push(format!("            ComposableNode(package='{package}', plugin='invar_generated_{name}::ComponentNode', name='{name}', namespace='{name}', parameters=[{parameters}], remappings={remaps}),\n"));
            }
        }
    }
    sequencer_node::generate(
        &mut files,
        &hash,
        document,
        cell_robot,
        &by_id,
        &mut launch_args,
    )?;
    let package = "invar_cell_interfaces";
    let mut cmake = format!(
        "cmake_minimum_required(VERSION 3.16)\nproject({package})\nfind_package(ament_cmake REQUIRED)\nfind_package(rosidl_default_generators REQUIRED)\n"
    );
    if messages.is_empty() && planner_messages.is_empty() {
        cmake.push_str("ament_package()\n");
    } else {
        cmake.push_str("rosidl_generate_interfaces(${PROJECT_NAME}\n");
        for (name, (scalar, type_id)) in &messages {
            insert(
                &mut files,
                format!("src/{package}/msg/{name}.msg"),
                format!("{scalar} value\n"),
                &hash,
            )?;
            cmake.push_str(&format!("  \"msg/{name}.msg\"\n"));
            let type_name = format!("{package}/msg/{name}");
            let record = json!({"generator":"invar-ros2-v1","ir_sha256":hash,"generated_do_not_edit":true,"type_name":type_name,"hash_string":type_hash(package,name,*type_id)});
            insert(
                &mut files,
                format!("src/{package}/hashes/{name}.json"),
                format!("{record}\n"),
                &hash,
            )?;
        }
        for (name, body) in &planner_messages {
            insert(
                &mut files,
                format!("src/{package}/msg/{name}.msg"),
                body.clone(),
                &hash,
            )?;
            cmake.push_str(&format!("  \"msg/{name}.msg\"\n"));
            let type_hash = planner_messages::type_hash(name, &planner_messages)
                .map_err(|message| invalid("S1500", message))?;
            let record = json!({"generator":"invar-ros2-v1","ir_sha256":hash,"generated_do_not_edit":true,"type_name":format!("{package}/msg/{name}"),"hash_string":type_hash});
            insert(
                &mut files,
                format!("src/{package}/hashes/{name}.json"),
                format!("{record}\n"),
                &hash,
            )?;
        }
        cmake.push_str(")\nament_export_dependencies(rosidl_default_runtime)\n");
        if planner_messages
            .keys()
            .any(|name| name.starts_with("Planner"))
        {
            for n in [6, 7] {
                if !planner_messages.contains_key(&format!("PlannerRequest{n}V2"))
                    && !planner_messages.contains_key(&format!("PlannerTrajectory{n}V1"))
                {
                    continue;
                }
                let wire = crate::robotics::planner::header(n)
                    .map_err(|message| invalid("S1500", message))?;
                insert(
                    &mut files,
                    format!("src/{package}/include/{package}/planner{n}_wire.h"),
                    wire,
                    &hash,
                )?;
                let conversions = include_str!("ros2_templates/planner_conversions.hpp.in")
                    .replace("@JOINTS@", &n.to_string());
                insert(
                    &mut files,
                    format!("src/{package}/include/{package}/planner{n}_conversions.hpp"),
                    conversions,
                    &hash,
                )?;
            }
            cmake.push_str("install(DIRECTORY include/ DESTINATION include)\nament_export_include_directories(include)\n");
        }
        cmake.push_str("ament_package()\n");
    }
    insert(
        &mut files,
        format!("src/{package}/CMakeLists.txt"),
        cmake,
        &hash,
    )?;
    insert(
        &mut files,
        format!("src/{package}/package.xml"),
        format!(
            "<?xml version=\"1.0\"?>\n<package format=\"3\"><name>{package}</name><version>0.1.0</version><description>Invar generated interfaces</description><maintainer email=\"invar@example.invalid\">Invar</maintainer><license>Apache-2.0</license><buildtool_depend>ament_cmake</buildtool_depend><build_depend>rosidl_default_generators</build_depend><exec_depend>rosidl_default_runtime</exec_depend><member_of_group>rosidl_interface_packages</member_of_group><export><build_type>ament_cmake</build_type></export></package>\n"
        ),
        &hash,
    )?;
    insert(&mut files, "NOTICE.generated.txt", notices, &hash)?;
    insert(
        &mut files,
        "src/invar_cell_bringup/config/rmw_zenoh/session.json5",
        "{mode: \"peer\", connect: {endpoints: [\"tcp/127.0.0.1:7447\"]}, listen: {endpoints: [\"tcp/127.0.0.1:0\"]}, timestamping: {enabled: true}, scouting: {multicast: {enabled: false}}}\n".to_owned(),
        &hash,
    )?;
    insert(
        &mut files,
        "src/invar_cell_bringup/config/rmw_zenoh/router.json5",
        "{mode: \"router\", listen: {endpoints: [\"tcp/127.0.0.1:7447\"]}, timestamping: {enabled: true}, scouting: {multicast: {enabled: false}}}\n".to_owned(),
        &hash,
    )?;
    let mut qos = Vec::new();
    for connection in connections {
        qos.push(json!({"source":required(connection,"source")?,"target":required(connection,"target")?,"history":"KEEP_LAST","depth":qos_depth(required(connection,"policy")?)?,"reliability":"RELIABLE"}));
    }
    let graph = json!({"generator":"invar-ros2-v1","ir_sha256":hash,"generated_do_not_edit":true,"connections":qos});
    insert(
        &mut files,
        "src/invar_cell_bringup/config/graph.json",
        format!("{graph}\n"),
        &hash,
    )?;
    let controller = instances
        .iter()
        .find(|instance| instance["name"] == "controller")
        .and_then(|instance| instance["component"].as_str())
        .and_then(|id| by_id.get(id).copied());
    let hardware = instances
        .iter()
        .any(|instance| instance["name"] == "hardware");
    let attachments = document["cells"]
        .as_array()
        .into_iter()
        .flatten()
        .flat_map(|cell| cell["objects"].as_array().into_iter().flatten())
        .filter(|object| !object["dynamic"].is_null())
        .map(|object| required(object, "name"))
        .collect::<Result<Vec<_>, _>>()?;
    if attachments.len() > 1 {
        return Err(invalid("S1504", "ROS cell supports one attachable object"));
    }
    let attachment = attachments.first().copied();
    if let Some(controller) = controller {
        if !hardware {
            return Err(invalid(
                "S1504",
                "controller component needs hardware interface",
            ));
        }
        control_files(&mut files, &hash, controller, cell_robot)?;
        if let Some(node) =
            controller_bridge::generate(&mut files, &hash, controller, connections, cell_robot)?
        {
            separate_nodes.push(node);
        }
    }
    if let Some(object) = attachment {
        if controller.is_none() || cell_robot.is_none() {
            return Err(invalid(
                "S1504",
                "attachable cell needs simulated controller",
            ));
        }
        attachment_plugin_files(
            &mut files,
            &hash,
            object,
            cell_robot.ok_or_else(|| invalid("S1504", "missing telemetry robot"))?,
        )?;
    }
    let mut launch = String::from(
        "from launch import LaunchDescription\nfrom launch.actions import DeclareLaunchArgument, SetEnvironmentVariable\nfrom launch.substitutions import LaunchConfiguration\nfrom launch_ros.actions import Node, ComposableNodeContainer\nfrom launch_ros.descriptions import ComposableNode\nfrom launch_ros.parameter_descriptions import ParameterValue\nfrom ament_index_python.packages import get_package_share_directory\nfrom pathlib import Path\n\ndef generate_launch_description():\n    share = Path(get_package_share_directory('invar_cell_bringup'))\n    actions = [\n        SetEnvironmentVariable('RMW_IMPLEMENTATION', 'rmw_zenoh_cpp'),\n        SetEnvironmentVariable('ZENOH_ROUTER_CHECK_ATTEMPTS', '10'),\n        SetEnvironmentVariable('ZENOH_SESSION_CONFIG_URI', str(share / 'config/rmw_zenoh/session.json5')),\n        SetEnvironmentVariable('ZENOH_ROUTER_CONFIG_URI', str(share / 'config/rmw_zenoh/router.json5')),\n        Node(package='rmw_zenoh_cpp', executable='rmw_zenohd', output='screen'),\n    ]\n",
    );
    launch.push_str(&launch_args);
    if launch_args.contains("recording_action =") {
        launch=launch.replace("from launch.actions import DeclareLaunchArgument, SetEnvironmentVariable", "from launch.actions import DeclareLaunchArgument, SetEnvironmentVariable, ExecuteProcess, RegisterEventHandler, EmitEvent\nfrom launch.event_handlers import OnProcessExit\nfrom launch.events import Shutdown");
    }
    if !composables.is_empty() {
        launch.push_str("    actions.append(ComposableNodeContainer(name='invar_components', namespace='', package='rclcpp_components', executable='component_container', composable_node_descriptions=[\n");
        launch.push_str(&composables.concat());
        launch.push_str("    ], output='screen'))\n");
    }
    if !separate_nodes.is_empty() {
        launch.push_str("    actions.extend([\n");
        launch.push_str(&separate_nodes.concat());
        launch.push_str("    ])\n");
    }
    if controller.is_some() {
        if cell_robot.is_some() {
            launch.push_str("    robot = (share / 'config/ros2_control.urdf').read_text().replace('@INVAR_MJCF_PATH@', str(share / 'config/mjcf/scene.xml'))\n    actions.extend([\n        Node(package='robot_state_publisher', executable='robot_state_publisher', name='robot_state_publisher', parameters=[{'robot_description': robot, 'use_sim_time': True}], output='screen'),\n        Node(package='mujoco_ros2_control', executable='ros2_control_node', name='controller_manager', parameters=[{'robot_description': robot, 'use_sim_time': True}, str(share / 'config/controllers.yaml')");
            if attachment.is_some() {
                launch.push_str(", str(share / 'config/attachment_plugin.yaml')");
            }
            launch.push_str("], output='screen'),\n        Node(package='controller_manager', executable='spawner', arguments=['joint_state_broadcaster', '--controller-manager', '/controller_manager'], output='screen'),\n        Node(package='controller_manager', executable='spawner', arguments=['joint_trajectory_controller', '--controller-manager', '/controller_manager', '--param-file', str(share / 'config/controllers.yaml')], output='screen'),\n    ])\n");
        } else {
            launch.push_str("    robot = (share / 'config/ros2_control.urdf').read_text()\n    actions.extend([\n        Node(package='robot_state_publisher', executable='robot_state_publisher', name='robot_state_publisher', parameters=[{'robot_description': robot}], output='screen'),\n        Node(package='controller_manager', executable='ros2_control_node', name='controller_manager', parameters=[str(share / 'config/controllers.yaml')], output='screen'),\n        Node(package='controller_manager', executable='spawner', arguments=['joint_state_broadcaster', '--controller-manager', '/controller_manager'], output='screen'),\n        Node(package='controller_manager', executable='spawner', arguments=['joint_trajectory_controller', '--controller-manager', '/controller_manager', '--param-file', str(share / 'config/controllers.yaml')], output='screen'),\n    ])\n");
        }
    }
    launch.push_str("    return LaunchDescription(actions)\n");
    insert(
        &mut files,
        "src/invar_cell_bringup/launch/cell.launch.py",
        launch,
        &hash,
    )?;
    let attachment_cmake = if attachment.is_some() {
        "if(POLICY CMP0144)\n  cmake_policy(SET CMP0144 NEW)\nendif()\nfind_package(rclcpp REQUIRED)\nfind_package(geometry_msgs REQUIRED)\nfind_package(std_msgs REQUIRED)\nfind_package(std_srvs REQUIRED)\nfind_package(invar_cell_interfaces REQUIRED)\nfind_package(pluginlib REQUIRED)\nfind_package(mujoco_ros2_control_plugins REQUIRED)\nfind_package(mujoco_vendor REQUIRED)\nadd_library(invar_attachment_plugin SHARED src/attachment_plugin.cpp)\ntarget_compile_features(invar_attachment_plugin PUBLIC cxx_std_17)\ntarget_include_directories(invar_attachment_plugin PRIVATE ${invar_cell_interfaces_INCLUDE_DIRS})\ntarget_link_libraries(invar_attachment_plugin rclcpp::rclcpp geometry_msgs::geometry_msgs std_msgs::std_msgs std_srvs::std_srvs ${invar_cell_interfaces_TARGETS} pluginlib::pluginlib mujoco_ros2_control_plugins::mujoco_ros2_control_plugins mujoco::mujoco)\npluginlib_export_plugin_description_file(mujoco_ros2_control_plugins attachment_plugin.xml)\ninstall(TARGETS invar_attachment_plugin DESTINATION lib)\n"
    } else {
        ""
    };
    insert(
        &mut files,
        "src/invar_cell_bringup/CMakeLists.txt",
        format!(
            "cmake_minimum_required(VERSION 3.16)\nproject(invar_cell_bringup)\nfind_package(ament_cmake REQUIRED)\nfind_package(generate_parameter_library REQUIRED)\n{parameter_targets}{attachment_cmake}install(DIRECTORY launch config DESTINATION share/${{PROJECT_NAME}})\nament_package()\n"
        ),
        &hash,
    )?;
    let attachment_deps = if attachment.is_some() {
        "<depend>rclcpp</depend><depend>geometry_msgs</depend><depend>std_msgs</depend><depend>std_srvs</depend><depend>invar_cell_interfaces</depend><depend>pluginlib</depend><depend>mujoco_ros2_control_plugins</depend><depend>mujoco_vendor</depend>"
    } else {
        ""
    };
    insert(
        &mut files,
        "src/invar_cell_bringup/package.xml",
        format!(
            "<?xml version=\"1.0\"?>\n<package format=\"3\"><name>invar_cell_bringup</name><version>0.1.0</version><description>Invar generated bringup</description><maintainer email=\"invar@example.invalid\">Invar</maintainer><license>Apache-2.0</license><buildtool_depend>ament_cmake</buildtool_depend><build_depend>generate_parameter_library</build_depend>{attachment_deps}<exec_depend>launch_ros</exec_depend><exec_depend>controller_manager</exec_depend><exec_depend>robot_state_publisher</exec_depend><exec_depend>rmw_zenoh_cpp</exec_depend><export><build_type>ament_cmake</build_type></export></package>\n"
        ),
        &hash,
    )?;
    Ok(files)
}
