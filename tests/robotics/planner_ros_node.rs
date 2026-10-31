//! Issue #29 (S1-06).
use crate::child_deadline;
use invar::{front, system};

#[test]
#[cfg_attr(
    not(unix),
    ignore = "the generated ROS 2 node is a Linux program (dlfcn.h)"
)]
fn planner_ros_node_is_separate_and_uses_checked_conversions() {
    for n in [6, 7] {
        let manifest = include_str!("../fixtures/s1_06/planner-manifest.inv")
            .replace("f64,7", &format!("f64,{n}"))
            .replace("request7_", &format!("request{n}_"))
            .replace("trajectory7_", &format!("trajectory{n}_"));
        let source = format!(
            r#"{manifest}
fn run() -> i32 {{ return 0; }}
component Sequencer {{
 param base_mount: Transform<base,world> = identity;
 // Generation-only measured-port stand-ins; not runtime telemetry.
 port out measured_cube: std::robotics::Pose<f64,world,cube> @ 0.2 Hz;
 port out measured_tool: std::robotics::Pose<f64,base,tool> @ 0.2 Hz;
 port out request: std::robotics::PlannerRequest<f64,{n},base,tool,cube> @ 0.2 Hz;
 port in trajectory: std::robotics::PlannerTrajectory<f64,{n}> @ 0.2 Hz;
 resource cpu_cores = 1, memory = 1 MiB;
 wcet 1 ms; deadline 5 s; license "Apache-2.0"; isolation in-process; impl invar run;
}}
connect sequencer.request -> planner.request policy queue(1);
connect planner.trajectory -> sequencer.trajectory policy queue(1);
product P {{ distribution internal; commercial false; robot menagerie::franka_emika_panda;
 capacity cpu_cores = 2, memory = 2 GiB, gpu_memory = 2 GiB; }}
system planner_probe for P {{ sequencer: Sequencer; planner: curobo_planner; }}
"#
        );
        let ir = system::compile(front::parse_system(source.as_bytes()).unwrap()).unwrap();
        assert!(ir.graph_issues().is_empty(), "{:?}", ir.graph_issues());
        let files = system::ros2::generate(&ir).expect("generated planner node");
        assert_eq!(files, system::ros2::generate(&ir).unwrap());
        let node = &files["src/invar_planner_node/src/node.cpp"];
        assert!(node.contains(&format!("invar-planner-component-v2/{n}/")));
        assert!(node.contains("request_from_ros"));
        assert!(node.contains("trajectory_to_ros"));
        assert!(node.contains("planner configuration rejected"));
        assert!(node.contains("descriptor.read_only = true;"));
        for parameter in node
            .lines()
            .filter(|line| line.contains("declare_parameter<"))
        {
            assert!(
                parameter.contains(", descriptor)"),
                "mutable config: {parameter}"
            );
        }
        let launch = &files["src/invar_cell_bringup/launch/cell.launch.py"];
        assert!(launch.contains("DeclareLaunchArgument('planner_config'"));
        assert!(launch.contains("executable='planner_node'"));
        assert!(launch.contains("('request', '/sequencer/request')"));
        assert!(files["src/invar_planner_node/CMakeLists.txt"].contains("invar_cell_interfaces"));
        // Generation is not a claim that this fixture's Invar stub runs a task sequencer.
        check_cpp(n, &files);
    }
}

fn check_cpp(n: u32, files: &std::collections::BTreeMap<String, String>) {
    use std::{fs, process::Command};
    let root = std::env::temp_dir().join(format!("invar-planner-node-{}-{n}", std::process::id()));
    fs::create_dir(&root).unwrap();
    for (name, content) in files.iter().filter(|(name, _)| {
        name.ends_with(".h") || name.ends_with(".hpp") || name.ends_with(".cpp")
    }) {
        let path = root.join(name);
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        fs::write(path, content).unwrap();
    }
    let interfaces = root.join("src/invar_cell_interfaces/include");
    let includes = root.join("src/invar_planner_node/include/invar_planner_node");
    let source = root.join("config.cpp");
    fs::write(
        &source,
        include_str!("../fixtures/s1_06/planner_ros_config.cpp").replace("@N@", &n.to_string()),
    )
    .unwrap();
    let executable = root.join("config");
    let (i1, i2) = (
        format!("-I{}", interfaces.display()),
        format!("-I{}", includes.display()),
    );
    let flags = [
        "-std=c++17",
        "-Wall",
        "-Wextra",
        "-Werror",
        "-pedantic",
        &i1,
        &i2,
    ];
    child_deadline::cxx(&flags, &source, &executable);
    let result = Command::new(&executable).arg(&source).output().unwrap();
    child_deadline::assert_success(&result);
    // Minimal API stand-ins check C++ syntax only; actual pinned ROS build is still required.
    let stubs = root.join("stubs");
    fs::create_dir_all(stubs.join("rclcpp")).unwrap();
    fs::create_dir_all(stubs.join("rcl_interfaces/msg")).unwrap();
    fs::write(stubs.join("rcl_interfaces/msg/parameter_descriptor.hpp"), "#pragma once\nnamespace rcl_interfaces::msg { struct ParameterDescriptor { bool read_only = false; }; }\n").unwrap();
    fs::create_dir_all(stubs.join("rclcpp_components")).unwrap();
    fs::create_dir_all(stubs.join("invar_cell_interfaces/msg")).unwrap();
    fs::write(stubs.join("rclcpp/rclcpp.hpp"), r#"
#pragma once
#include <memory>
#include <cstddef>
#include <rcl_interfaces/msg/parameter_descriptor.hpp>
namespace rclcpp {
struct NodeOptions {};
struct KeepLast { explicit KeepLast(std::size_t) {} };
struct QoS { explicit QoS(KeepLast) {} QoS & reliable() { return *this; } };
template<class T> struct Subscription { using SharedPtr = std::shared_ptr<Subscription>; };
template<class T> struct Publisher { using SharedPtr = std::shared_ptr<Publisher>; void publish(const T &) {} };
class Node { public:
 Node(const char *, const NodeOptions &) {}
 virtual ~Node() = default;
 template<class T> T declare_parameter(const char *, T value, const rcl_interfaces::msg::ParameterDescriptor &) { return value; }
 template<class T> typename Publisher<T>::SharedPtr create_publisher(const char *, const QoS &) { return {}; }
 template<class T, class F> typename Subscription<T>::SharedPtr create_subscription(const char *, const QoS &, F) { return {}; }
};
inline void init(int, char **) {} inline void shutdown() {}
template<class T> void spin(std::shared_ptr<T>) {}
}
"#).unwrap();
    fs::write(
        stubs.join("rclcpp_components/register_node_macro.hpp"),
        "#define RCLCPP_COMPONENTS_REGISTER_NODE(...)\n",
    )
    .unwrap();
    fs::write(
        stubs.join(format!(
            "invar_cell_interfaces/msg/planner_request{n}_v2.hpp"
        )),
        format!(
            r#"
#pragma once
#include <array>
#include <memory>
namespace invar_cell_interfaces::msg {{ struct PlannerRequest{n}V2 {{
using ConstSharedPtr = std::shared_ptr<const PlannerRequest{n}V2>;
std::array<double,{n}> start_q_rad, goal_q_rad;
std::array<double,7> target_base_from_tool, cube_base_pose, tool_from_cube;
unsigned phase;
}}; }}
"#
        ),
    )
    .unwrap();
    fs::write(
        stubs.join(format!(
            "invar_cell_interfaces/msg/planner_trajectory{n}_v1.hpp"
        )),
        format!(
            r#"
#pragma once
#include <array>
#include <vector>
namespace invar_cell_interfaces::msg {{
struct PlannerWaypoint{n}V1 {{ double time_from_start_s; std::array<double,{n}> q_rad, dq_rad_s; }};
struct PlannerTrajectory{n}V1 {{ double dt_s; std::vector<PlannerWaypoint{n}V1> waypoints; }};
}}
"#
        ),
    )
    .unwrap();
    let compiled = Command::new("c++")
        .args([
            "-std=c++17",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-pedantic",
            "-fsyntax-only",
        ])
        .arg("-I")
        .arg(&stubs)
        .arg("-I")
        .arg(&interfaces)
        .arg("-I")
        .arg(&includes)
        .arg(root.join("src/invar_planner_node/src/node.cpp"))
        .arg(root.join("src/invar_planner_node/src/main.cpp"))
        .output()
        .unwrap();
    child_deadline::assert_success(&compiled);
    fs::remove_dir_all(root).unwrap();
}
