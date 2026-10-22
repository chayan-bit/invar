use std::collections::BTreeMap;

use serde_json::Value;

use super::{GenerateError, insert, invalid, qos_depth, required};

pub(super) fn generate(
    files: &mut BTreeMap<String, String>,
    hash: &str,
    name: &str,
    component: &Value,
    connections: &[Value],
    robot: Option<&Value>,
) -> Result<(String, bool), GenerateError> {
    let ports = component["ports"]
        .as_array()
        .ok_or_else(|| invalid("S1500", "missing planner ports"))?;
    let n = crate::robotics::planner::manifest_joints(ports)
        .map_err(|message| invalid("S1500", message))?;
    if component["isolation"] != "separate-process" {
        return Err(invalid("S1500", "planner node requires process isolation"));
    }
    if let Some(actuators) = robot.and_then(|robot| robot["actuators"].as_array()) {
        let count = actuators
            .iter()
            .filter(|actuator| actuator["mode"] == "position" && actuator["target_kind"] == "joint")
            .count();
        if count != n as usize {
            return Err(invalid(
                "S1504",
                "planner joint count differs from selected robot",
            ));
        }
    }
    let input = ports
        .iter()
        .find(|port| port["direction"] == "in")
        .ok_or_else(|| invalid("S1500", "missing planner input"))?;
    let output = ports
        .iter()
        .find(|port| port["direction"] == "out")
        .ok_or_else(|| invalid("S1500", "missing planner output"))?;
    let input = required(input, "name")?;
    let output = required(output, "name")?;
    let depth = |port: &str, endpoint: &str| -> Result<u16, GenerateError> {
        let path = format!("{name}.{port}");
        connections
            .iter()
            .find(|edge| edge[endpoint] == path)
            .map(|edge| required(edge, "policy").and_then(qos_depth))
            .transpose()
            .map(|value| value.unwrap_or(1))
    };
    let prefix = required(&component["abi"], "prefix")?;
    let package = format!("invar_{name}_node");
    let namespace = format!("invar_generated_{name}");
    let projection = robot
        .map(crate::system::tool_projection::derive)
        .transpose()
        .map_err(|m| invalid("S1504", m))?;
    insert(
        files,
        format!("src/{package}/include/{package}/tool_projection.hpp"),
        crate::system::tool_projection::header(projection.as_ref()),
        hash,
    )?;
    if let Some(p) = &projection {
        insert(
            files,
            format!("src/{package}/tool-projection.json"),
            crate::system::tool_projection::metadata(p),
            hash,
        )?;
    }
    let header =
        crate::system::foreign::header(component).map_err(|message| invalid("S1500", message))?;
    insert(
        files,
        format!("src/{package}/include/{package}/{prefix}.h"),
        header,
        hash,
    )?;
    for (file, template) in [
        (
            "node.hpp",
            include_str!("../ros2_templates/planner_node.hpp.in"),
        ),
        (
            "planner_config.hpp",
            include_str!("../ros2_templates/planner_config.hpp.in"),
        ),
        (
            "node.cpp",
            include_str!("../ros2_templates/planner_node.cpp.in"),
        ),
        (
            "main.cpp",
            include_str!("../ros2_templates/planner_main.cpp.in"),
        ),
    ] {
        let body = template
            .replace("@NAMESPACE@", &namespace)
            .replace("@NODE@", name)
            .replace("@PREFIX@", prefix)
            .replace("@JOINTS@", &n.to_string())
            .replace("@REFERENCE_FRAME_CHECK@", &projection.as_ref().map_or(String::new(),|p|format!("if (values.tool_frame != \"{}\") failure(1, \"planner reference tool frame mismatch\");",p.planner_reference_frame)))
            .replace("@INPUT@", input)
            .replace("@OUTPUT@", output)
            .replace("@INPUT_DEPTH@", &depth(input, "target")?.to_string())
            .replace("@OUTPUT_DEPTH@", &depth(output, "source")?.to_string());
        let folder = if file.ends_with(".hpp") {
            format!("include/{package}")
        } else {
            "src".to_owned()
        };
        insert(files, format!("src/{package}/{folder}/{file}"), body, hash)?;
    }
    let cmake = format!(
        "cmake_minimum_required(VERSION 3.16)\nproject({package})\nfind_package(ament_cmake REQUIRED)\nfind_package(rclcpp REQUIRED)\nfind_package(rclcpp_components REQUIRED)\nfind_package(invar_cell_interfaces REQUIRED)\nfind_package(rcl_interfaces REQUIRED)\nadd_library(${{PROJECT_NAME}}_component SHARED src/node.cpp)\ntarget_compile_features(${{PROJECT_NAME}}_component PUBLIC cxx_std_17)\ntarget_include_directories(${{PROJECT_NAME}}_component PRIVATE include/${{PROJECT_NAME}} ${{invar_cell_interfaces_INCLUDE_DIRS}})\ntarget_link_libraries(${{PROJECT_NAME}}_component rclcpp::rclcpp rclcpp_components::component ${{invar_cell_interfaces_TARGETS}} ${{rcl_interfaces_TARGETS}} dl)\nrclcpp_components_register_nodes(${{PROJECT_NAME}}_component \"{namespace}::ComponentNode\")\nadd_executable({name}_node src/main.cpp)\ntarget_compile_features({name}_node PUBLIC cxx_std_17)\ntarget_include_directories({name}_node PRIVATE include/${{PROJECT_NAME}})\ntarget_link_libraries({name}_node ${{PROJECT_NAME}}_component rclcpp::rclcpp)\ninstall(TARGETS ${{PROJECT_NAME}}_component DESTINATION lib)\ninstall(TARGETS {name}_node DESTINATION lib/${{PROJECT_NAME}})\nament_package()\n"
    );
    insert(files, format!("src/{package}/CMakeLists.txt"), cmake, hash)?;
    let xml = format!(
        "<?xml version=\"1.0\"?>\n<package format=\"3\"><name>{package}</name><version>0.1.0</version><description>Invar generated checked planner node</description><maintainer email=\"invar@example.invalid\">Invar</maintainer><license>Apache-2.0</license><buildtool_depend>ament_cmake</buildtool_depend><depend>rclcpp</depend><depend>rclcpp_components</depend><depend>invar_cell_interfaces</depend><depend>rcl_interfaces</depend><export><build_type>ament_cmake</build_type></export></package>\n"
    );
    insert(files, format!("src/{package}/package.xml"), xml, hash)?;
    Ok((package, true))
}
