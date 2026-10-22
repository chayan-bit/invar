//! Explicit versioned task binding. Names alone never enable task execution.
use super::{GenerateError, insert, invalid, required, ros_name, telemetry};
use serde_json::{Value, json};
use std::collections::BTreeMap;

const TARGET: &str = "invar::s1_06_reach_place_v1";
fn source_pose(v: &Value) -> Result<crate::system::tool_projection::Pose, GenerateError> {
    let xyz = v["xyz"]
        .as_array()
        .filter(|x| x.len() == 3)
        .ok_or_else(|| invalid("S1504", "source pose position"))?;
    let q = v["quat"]
        .as_array()
        .filter(|x| x.len() == 4)
        .ok_or_else(|| invalid("S1504", "source pose rotation"))?;
    let number = |v: &Value| {
        v.as_str()
            .and_then(|s| s.parse::<f64>().ok())
            .filter(|n| n.is_finite())
            .ok_or_else(|| invalid("S1504", "finite source pose"))
    };
    let p = [
        number(&xyz[0]["value"])?,
        number(&xyz[1]["value"])?,
        number(&xyz[2]["value"])?,
    ];
    let q = [
        number(&q[0])?,
        number(&q[1])?,
        number(&q[2])?,
        number(&q[3])?,
    ];
    let norm = q.iter().map(|x| x * x).sum::<f64>().sqrt();
    let pose = crate::system::tool_projection::Pose {
        translation_m: p,
        quaternion_xyzw: [q[1] / norm, q[2] / norm, q[3] / norm, q[0] / norm],
    };
    if !pose.valid() {
        return Err(invalid("S1504", "invalid source pose"));
    }
    Ok(pose)
}
fn flat(p: crate::system::tool_projection::Pose) -> Vec<f64> {
    p.translation_m
        .into_iter()
        .chain(p.quaternion_xyzw)
        .collect()
}
fn home(robot: &Value, names: &[String]) -> Result<Vec<f64>, GenerateError> {
    fn visit(v: &Value, joints: &mut Vec<String>, keys: &mut Vec<String>) {
        if v["tag"] == "joint"
            && let Some(name) = v["attrs"]["name"].as_str()
        {
            joints.push(name.into());
        }
        if v["tag"] == "key"
            && v["attrs"]["name"] == "home"
            && let Some(q) = v["attrs"]["qpos"].as_str()
        {
            keys.push(q.into());
        }
        for c in v["children"].as_array().into_iter().flatten() {
            visit(c, joints, keys);
        }
    }
    let mut joints = Vec::new();
    let mut keys = Vec::new();
    visit(&robot["source_tree"], &mut joints, &mut keys);
    if keys.len() != 1 {
        return Err(invalid("S1504", "task needs one pinned home keyframe"));
    }
    let q = keys[0]
        .split_whitespace()
        .map(str::parse::<f64>)
        .collect::<Result<Vec<_>, _>>()
        .map_err(|_| invalid("S1504", "invalid home keyframe"))?;
    if q.len() != joints.len() || q.iter().any(|n| !n.is_finite()) {
        return Err(invalid("S1504", "unsupported home joint layout"));
    }
    names
        .iter()
        .map(|name| {
            joints
                .iter()
                .position(|j| j == name)
                .and_then(|i| q.get(i))
                .copied()
                .ok_or_else(|| invalid("S1504", "home joint missing"))
        })
        .collect()
}
pub(super) fn generate(
    files: &mut BTreeMap<String, String>,
    hash: &str,
    document: &Value,
    robot: Option<&Value>,
    components: &BTreeMap<&str, &Value>,
    launch_args: &mut String,
) -> Result<(), GenerateError> {
    let mut selected = None;
    for instance in document["system"]["instances"]
        .as_array()
        .ok_or_else(|| invalid("S1500", "missing instances"))?
    {
        let c = components
            .get(required(instance, "component")?)
            .ok_or_else(|| invalid("S1500", "missing task component"))?;
        let target = c["implementation"]["target"].as_str().unwrap_or("");
        if c["implementation"]["kind"] == "extern"
            && target.starts_with("invar::s1_06_reach_place_")
        {
            if target != TARGET || selected.is_some() {
                return Err(invalid("S1504", "unknown/duplicate sequencer binding"));
            }
            selected = Some((instance, *c));
        }
    }
    let Some((instance, c)) = selected else {
        return Ok(());
    };
    if c["isolation"] != "separate-process" {
        return Err(invalid(
            "S1504",
            "Rust task binding requires process isolation",
        ));
    }
    let memory = &c["resources"]["memory"];
    let bytes = required(memory, "value")?
        .parse::<f64>()
        .ok()
        .zip(match memory["unit"].as_str() {
            Some("MiB") => Some(1048576.0),
            Some("GiB") => Some(1073741824.0),
            Some("B") => Some(1.0),
            _ => None,
        })
        .map(|(n, m)| n * m);
    if bytes.is_none_or(|b| !b.is_finite() || b < 128.0 * 1048576.0) {
        return Err(invalid("S1504", "task requires declared memory >=128 MiB"));
    }
    let robot = robot.ok_or_else(|| invalid("S1504", "task requires checked robot"))?;
    let names = telemetry::joints(robot)?;
    let n = names.len();
    let name = ros_name(required(instance, "name")?)?;
    let ports = c["ports"]
        .as_array()
        .ok_or_else(|| invalid("S1504", "missing task ports"))?;
    for (port, direction, ty, hz) in [
        (
            "request",
            "out",
            format!("std::robotics::PlannerRequest<f64,{n},base,tool,cube>"),
            0.2,
        ),
        (
            "planned",
            "in",
            format!("std::robotics::PlannerTrajectory<f64,{n}>"),
            0.2,
        ),
        (
            "approved",
            "out",
            format!("std::robotics::PlannerTrajectory<f64,{n}>"),
            0.2,
        ),
        (
            "measured_cube",
            "out",
            "std::robotics::Pose<f64,world,cube>".into(),
            100.0,
        ),
        (
            "tool",
            "in",
            "std::robotics::Pose<f64,base,tool>".into(),
            100.0,
        ),
    ] {
        if !ports.iter().any(|p| {
            p["name"] == port
                && p["direction"] == direction
                && p["type"] == ty
                && p["rate"]["unit"] == "Hz"
                && p["rate"]["value"]
                    .as_str()
                    .and_then(|s| s.parse::<f64>().ok())
                    == Some(hz)
        }) {
            return Err(invalid(
                "S1504",
                format!("missing count/frame-specific task port {port}"),
            ));
        }
    }
    if ports.len() != 5 {
        return Err(invalid("S1504", "unsupported extra task port"));
    }
    let edges = document["connections"]
        .as_array()
        .ok_or_else(|| invalid("S1504", "missing task edges"))?;
    let planned = format!("{name}.planned");
    let approved = format!("{name}.approved");
    let request = format!("{name}.request");
    let tool = format!("{name}.tool");
    let edge = |key: &str, path: &str| -> Result<&Value, GenerateError> {
        let values = edges.iter().filter(|e| e[key] == path).collect::<Vec<_>>();
        if values.len() != 1 {
            return Err(invalid("S1504", "task needs unique checked source edge"));
        }
        Ok(values[0])
    };
    let planned_source = required(edge("target", &planned)?, "source")?;
    let (planner, planner_port) = planned_source
        .split_once('.')
        .ok_or_else(|| invalid("S1504", "planner endpoint"))?;
    let planner_instance = document["system"]["instances"]
        .as_array()
        .into_iter()
        .flatten()
        .find(|i| i["name"] == planner)
        .ok_or_else(|| invalid("S1504", "planner instance"))?;
    let planner_component = components
        .get(required(planner_instance, "component")?)
        .ok_or_else(|| invalid("S1504", "planner component"))?;
    let req_target = required(edge("source", &request)?, "target")?;
    if planner_component["config"]["layout"] != "inv_planner_config_v1"
        || planner_port != "trajectory"
        || req_target != format!("{planner}.request")
        || required(edge("source", &approved)?, "target")? != "controller.trajectory"
    {
        return Err(invalid("S1504", "task planner/controller graph mismatch"));
    }
    let fk_source = required(edge("target", &tool)?, "source")?;
    let (fk_name, _) = fk_source
        .split_once('.')
        .ok_or_else(|| invalid("S1504", "FK endpoint"))?;
    let fk_instance = document["system"]["instances"]
        .as_array()
        .into_iter()
        .flatten()
        .find(|i| i["name"] == fk_name)
        .ok_or_else(|| invalid("S1504", "FK instance"))?;
    let fk = components
        .get(required(fk_instance, "component")?)
        .ok_or_else(|| invalid("S1504", "FK component"))?;
    if fk["config"]["layout"] != "inv_fk_config_v1" || fk["isolation"] != "in-process" {
        return Err(invalid(
            "S1504",
            "task requires independent checked FK manifest",
        ));
    }
    if !components.values().any(|component| {
        let mount = &component["params"]["base_mount"];
        mount["kind"] == "identity-transform"
            && mount["from"] == "base"
            && mount["to"] == "robot_base"
    }) {
        return Err(invalid(
            "S1504",
            "task requires checked identity base/robot_base mount",
        ));
    }
    let cells = document["cells"]
        .as_array()
        .filter(|c| c.len() == 1)
        .ok_or_else(|| invalid("S1504", "task requires one checked cell"))?;
    let cell = &cells[0];
    let objects = cell["objects"]
        .as_array()
        .ok_or_else(|| invalid("S1504", "task objects missing"))?;
    let cube = objects
        .iter()
        .find(|o| o["name"] == "cube" && o["dynamic"]["attachable"] == true)
        .ok_or_else(|| invalid("S1504", "attachable cube missing"))?;
    let target = objects
        .iter()
        .find(|o| o["name"] == "place_target")
        .ok_or_else(|| invalid("S1504", "place target missing"))?;
    let mount = source_pose(&cell["mount"])?;
    if mount != crate::system::tool_projection::Pose::IDENTITY {
        return Err(invalid(
            "S1504",
            "task cell mount must agree with declared identity robot/world binding",
        ));
    }
    let final_cube = mount
        .inverse()
        .and_then(|p| p.compose(source_pose(&target["pose"]).map_err(|_| "target pose")?))
        .map_err(|m| invalid("S1504", m))?;
    let mut limits = Vec::new();
    let mut speeds = Vec::new();
    let joints = robot["joints"]
        .as_array()
        .ok_or_else(|| invalid("S1504", "robot joints missing"))?;
    for name in &names {
        let j = joints
            .iter()
            .find(|j| j["name"] == *name)
            .ok_or_else(|| invalid("S1504", "joint missing"))?;
        let number = |v: &Value| {
            required(v, "value")?
                .parse::<f64>()
                .map_err(|_| invalid("S1504", "joint limit number"))
        };
        limits.push([
            number(&j["position"]["lower"])?,
            number(&j["position"]["upper"])?,
        ]);
        speeds.push(number(&j["velocity"])?);
    }
    let projection =
        crate::system::tool_projection::derive(robot).map_err(|m| invalid("S1504", m))?;
    let descriptor = telemetry::descriptor(robot, hash)?;
    let full_names = descriptor["robot_joints"]
        .as_array()
        .ok_or_else(|| invalid("S1504", "full joint descriptors"))?
        .iter()
        .map(|j| required(j, "name").map(str::to_owned))
        .collect::<Result<Vec<_>, _>>()?;
    let mut config = json!({"schema":"invar-sequencer-binding-v1","source_ir_sha256":hash,"joint_order_sha256":descriptor["joint_order_sha256"],"joint_count":n,"joint_names":names,
        "home_q_rad":home(robot,&names)?,"world_from_base":flat(mount),"final_cube_base":flat(final_cube),"joint_limits_rad":limits,"velocity_limits_rad_s":speeds,
        "robot_joints":descriptor["robot_joints"],"robot_joint_order_sha256":descriptor["robot_joint_order_sha256"],"home_robot_q_si":home(robot,&full_names)?,
        "fk_prefix":fk["abi"]["prefix"],"tool_projection":projection.value(),"deadlines":"deadlines are declared, not measured"});
    if let Some(grasp) = cube.get("grasp") {
        let cube_from_grasp = source_pose(grasp)?;
        let world_goal = source_pose(&cube["pose"])?
            .compose(cube_from_grasp)
            .map_err(|m| invalid("S1504", m))?;
        let mut world_pregrasp = world_goal;
        world_pregrasp.translation_m[2] += 0.1;
        let inverse = mount.inverse().map_err(|m| invalid("S1504", m))?;
        config["schema"] = json!("invar-sequencer-binding-v2");
        config["cube_from_grasp"] = json!(flat(cube_from_grasp));
        config["grasp_tool_base"] = json!(flat(
            inverse
                .compose(world_goal)
                .map_err(|m| invalid("S1504", m))?
        ));
        config["pregrasp_tool_base"] = json!(flat(
            inverse
                .compose(world_pregrasp)
                .map_err(|m| invalid("S1504", m))?
        ));
    }
    let config = format!("{config}\n");
    let config_hash = super::sha256_hex(config.as_bytes());
    insert(
        files,
        "src/invar_cell_bringup/config/sequencer-binding-v1.json",
        config,
        hash,
    )?;
    let body = include_str!("../ros2_templates/sequencer_node.cpp.in")
        .replace("@JOINTS@", &n.to_string())
        .replace("@NODE@", &name)
        .replace(
            "@TOOL_TOPIC@",
            &format!(
                "/{}/{}",
                ros_name(fk_name)?,
                ros_name(
                    fk_source
                        .split_once('.')
                        .ok_or_else(|| invalid("S1504", "FK endpoint"))?
                        .1
                )?
            ),
        )
        .replace("@BINDING_SHA@", &config_hash)
        .replace("@PLANNER@", &ros_name(planner)?)
        .replace(
            "@PLANNED_TOPIC@",
            &format!("/{}/{}", ros_name(planner)?, ros_name(planner_port)?),
        );
    let package = format!("invar_{name}_task");
    insert(files, format!("src/{package}/src/main.cpp"), body, hash)?;
    insert(
        files,
        format!("src/{package}/include/process.hpp"),
        include_str!("../ros2_templates/sequencer_process.hpp.in").into(),
        hash,
    )?;
    insert(
        files,
        format!("src/{package}/CMakeLists.txt"),
        format!(
            "cmake_minimum_required(VERSION 3.16)\nproject({package})\nfind_package(ament_cmake REQUIRED)\nfind_package(rclcpp REQUIRED)\nfind_package(rcl_interfaces REQUIRED)\nfind_package(invar_cell_interfaces REQUIRED)\nfind_package(std_srvs REQUIRED)\nfind_package(std_msgs REQUIRED)\nfind_package(geometry_msgs REQUIRED)\nadd_executable({name}_task src/main.cpp)\ntarget_compile_features({name}_task PUBLIC cxx_std_17)\ntarget_include_directories({name}_task PRIVATE include ${{invar_cell_interfaces_INCLUDE_DIRS}})\ntarget_link_libraries({name}_task rclcpp::rclcpp ${{invar_cell_interfaces_TARGETS}} ${{std_srvs_TARGETS}} ${{std_msgs_TARGETS}} ${{geometry_msgs_TARGETS}} ${{rcl_interfaces_TARGETS}})\ninstall(TARGETS {name}_task DESTINATION lib/${{PROJECT_NAME}})\nament_package()\n"
        ),
        hash,
    )?;
    insert(
        files,
        format!("src/{package}/package.xml"),
        format!(
            "<?xml version=\"1.0\"?>\n<package format=\"3\"><name>{package}</name><version>0.1.0</version><description>Explicit Rust reach-place task boundary</description><maintainer email=\"invar@example.invalid\">Invar</maintainer><license>Apache-2.0</license><buildtool_depend>ament_cmake</buildtool_depend><depend>rclcpp</depend><depend>rcl_interfaces</depend><depend>invar_cell_interfaces</depend><depend>std_srvs</depend><depend>std_msgs</depend><depend>geometry_msgs</depend><export><build_type>ament_cmake</build_type></export></package>\n"
        ),
        hash,
    )?;
    for arg in ["runtime", "task", "task_sha256", "recording_path"] {
        launch_args.push_str(&format!("    actions.append(DeclareLaunchArgument('{name}_{arg}', description='Required bounded task runtime input'))\n"));
    }
    launch_args.push_str(&format!("    recording_action = ExecuteProcess(cmd=['ros2', 'bag', 'record', '-s', 'mcap', '-o', LaunchConfiguration('{name}_recording_path'), '-a'], output='screen', sigterm_timeout='10.0', sigkill_timeout='5.0')\n    task_action = Node(package='{package}', executable='{name}_task', name='{name}', namespace='{name}', parameters=[{{'use_sim_time': True, 'runtime_path': LaunchConfiguration('{name}_runtime'), 'binding_path': str(share / 'config/sequencer-binding-v1.json'), 'task_path': LaunchConfiguration('{name}_task'), 'task_sha256': LaunchConfiguration('{name}_task_sha256')}}], output='screen')\n    def required_exit(event, context):\n        if context.is_shutdown:\n            return []\n        if event.action is task_action and event.returncode == 0:\n            return [EmitEvent(event=Shutdown(reason='Invar task completed; finalize MCAP'))]\n        raise RuntimeError('S1406: required Invar task/recorder exited: ' + str(event.returncode))\n    actions.extend([RegisterEventHandler(OnProcessExit(target_action=task_action, on_exit=required_exit)), RegisterEventHandler(OnProcessExit(target_action=recording_action, on_exit=required_exit)), recording_action, task_action])\n"));
    Ok(())
}
