use std::collections::BTreeMap;

use serde_json::Value;

use super::{GenerateError, insert, invalid, qos_depth, required, ros_name};

fn number(value: &Value, unit: &str) -> Result<(f64, String), GenerateError> {
    if value["unit"] != unit {
        return Err(invalid("S1504", "controller joint limit unit mismatch"));
    }
    let value = required(value, "value")?
        .parse::<f64>()
        .map_err(|_| invalid("S1504", "invalid controller joint limit"))?;
    if !value.is_finite() {
        return Err(invalid("S1504", "nonfinite controller joint limit"));
    }
    Ok((value, format!("{value:.17e}")))
}

pub(super) fn generate(
    files: &mut BTreeMap<String, String>,
    hash: &str,
    controller: &Value,
    connections: &[Value],
    robot: Option<&Value>,
) -> Result<Option<String>, GenerateError> {
    let ports = controller["ports"]
        .as_array()
        .ok_or_else(|| invalid("S1500", "missing controller ports"))?;
    let inputs: Vec<_> = ports
        .iter()
        .filter(|port| {
            port["direction"] == "in"
                && port["type"]
                    .as_str()
                    .is_some_and(|ty| ty.starts_with("std::robotics::PlannerTrajectory<"))
        })
        .collect();
    if inputs.is_empty() {
        return Ok(None);
    }
    if inputs.len() != 1 {
        return Err(invalid(
            "S1504",
            "controller needs one planner trajectory input",
        ));
    }
    let input = inputs[0];
    let n = crate::robotics::planner_joint_count(required(input, "type")?)
        .ok_or_else(|| invalid("S1504", "unsupported controller planner count"))?;
    let robot = robot
        .ok_or_else(|| invalid("S1504", "planner controller bridge requires checked robot"))?;
    let actuators = robot["actuators"]
        .as_array()
        .ok_or_else(|| invalid("S1504", "missing robot actuators"))?;
    let joints = robot["joints"]
        .as_array()
        .ok_or_else(|| invalid("S1504", "missing robot joints"))?;
    let mut names = Vec::new();
    let mut limits = Vec::new();
    let mut speeds = Vec::new();
    for actuator in actuators
        .iter()
        .filter(|actuator| actuator["mode"] == "position" && actuator["target_kind"] == "joint")
    {
        let name = required(actuator, "target")?;
        let joint = joints
            .iter()
            .find(|joint| joint["name"] == name)
            .ok_or_else(|| invalid("S1504", "actuator joint missing"))?;
        if joint["kind"] != "revolute" {
            return Err(invalid(
                "S1504",
                "planner controller requires revolute joints",
            ));
        }
        let (low, low_text) = number(&joint["position"]["lower"], "rad")?;
        let (high, high_text) = number(&joint["position"]["upper"], "rad")?;
        let (speed, speed_text) = number(&joint["velocity"], "rad/s")?;
        if low >= high || speed <= 0.0 {
            return Err(invalid(
                "S1504",
                "controller limits must be nonempty and positive",
            ));
        }
        names.push(
            serde_json::to_string(name)
                .map_err(|_| invalid("S1500", "joint name serialization failed"))?,
        );
        limits.push(format!("{{{low_text}, {high_text}}}"));
        speeds.push(speed_text);
    }
    if names.len() != n as usize {
        return Err(invalid(
            "S1504",
            "controller planner count differs from robot",
        ));
    }
    let target = format!("controller.{}", required(input, "name")?);
    let edge = connections
        .iter()
        .find(|edge| edge["target"] == target)
        .ok_or_else(|| invalid("S1504", "controller planner input has no source edge"))?;
    let source = required(edge, "source")?;
    let (instance, port) = source
        .split_once('.')
        .ok_or_else(|| invalid("S1500", "invalid planner source endpoint"))?;
    let topic = format!("/{}/{}", ros_name(instance)?, ros_name(port)?);
    let depth = qos_depth(required(edge, "policy")?)?;
    let body = include_str!("../ros2_templates/controller_bridge.cpp.in")
        .replace("@JOINTS@", &n.to_string())
        .replace("@NAMES@", &names.join(", "))
        .replace("@LIMITS@", &limits.join(", "))
        .replace("@SPEEDS@", &speeds.join(", "))
        .replace("@DEPTH@", &depth.to_string());
    let package = "invar_controller_bridge";
    insert(files, format!("src/{package}/src/main.cpp"), body, hash)?;
    insert(
        files,
        format!("src/{package}/CMakeLists.txt"),
        format!(
            "cmake_minimum_required(VERSION 3.16)\nproject({package})\nfind_package(ament_cmake REQUIRED)\nfind_package(rclcpp REQUIRED)\nfind_package(invar_cell_interfaces REQUIRED)\nfind_package(trajectory_msgs REQUIRED)\nadd_executable(controller_bridge src/main.cpp)\ntarget_compile_features(controller_bridge PUBLIC cxx_std_17)\ntarget_include_directories(controller_bridge PRIVATE ${{invar_cell_interfaces_INCLUDE_DIRS}})\ntarget_link_libraries(controller_bridge rclcpp::rclcpp ${{invar_cell_interfaces_TARGETS}} ${{trajectory_msgs_TARGETS}})\ninstall(TARGETS controller_bridge DESTINATION lib/${{PROJECT_NAME}})\nament_package()\n"
        ),
        hash,
    )?;
    insert(
        files,
        format!("src/{package}/package.xml"),
        format!(
            "<?xml version=\"1.0\"?>\n<package format=\"3\"><name>{package}</name><version>0.1.0</version><description>Invar checked controller projection</description><maintainer email=\"invar@example.invalid\">Invar</maintainer><license>Apache-2.0</license><buildtool_depend>ament_cmake</buildtool_depend><depend>rclcpp</depend><depend>invar_cell_interfaces</depend><depend>trajectory_msgs</depend><export><build_type>ament_cmake</build_type></export></package>\n"
        ),
        hash,
    )?;
    Ok(Some(format!(
        "        Node(package='{package}', executable='controller_bridge', name='invar_controller_bridge', parameters=[{{'use_sim_time': True}}], remappings=[('trajectory', '{topic}')], output='screen'),\n"
    )))
}
