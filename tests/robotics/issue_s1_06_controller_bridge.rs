use crate::child_deadline;
use std::path::Path;

use invar::{front, system};

#[test]
fn issue_s1_06_controller_bridge_uses_selected_robot_limits_and_source_edge() {
    let directory = Path::new(env!("CARGO_MANIFEST_DIR")).join("examples/cells/reach_place");
    for (file, n, first_joint) in [
        ("cell.inv", 7, "joint1"),
        ("cell-ur5e.inv", 6, "shoulder_pan_joint"),
    ] {
        let source = std::fs::read_to_string(directory.join(file)).unwrap();
        let source = source.replace("component joint_trajectory_controller {", r#"
// Generation-only producer; this does not execute a planner or establish motion evidence.
component PlannerWireProbe {
 port out trajectory: std::robotics::PlannerTrajectory<f64,robot_dof> @ 0.2 Hz;
 resource cpu_cores = 1, memory = 1 MiB;
 license "Apache-2.0"; isolation in-process; wcet 1 ms; deadline 5 s; impl invar run;
}
component joint_trajectory_controller {
 port in trajectory: std::robotics::PlannerTrajectory<f64,robot_dof> @ 0.2 Hz;
"#)
            .replace("connect controller.cmd", "connect planner.trajectory -> controller.trajectory policy queue(1);\nconnect controller.cmd")
            .replace("    controller: joint_trajectory_controller;", "    planner: PlannerWireProbe;\n    controller: joint_trajectory_controller;");
        let ir = system::compile_at(front::parse_system(source.as_bytes()).unwrap(), &directory)
            .unwrap();
        assert!(ir.graph_issues().is_empty(), "{:?}", ir.graph_issues());
        let files = system::ros2::generate(&ir).unwrap();
        let bridge = &files["src/invar_controller_bridge/src/main.cpp"];
        assert!(bridge.contains(&format!("PlannerTrajectory{n}V1")));
        assert!(bridge.contains(first_joint));
        assert!(bridge.contains("controller_trajectory_from_ros"));
        assert!(bridge.contains("/joint_trajectory_controller/joint_trajectory"));
        assert!(
            files["src/invar_cell_bringup/launch/cell.launch.py"].contains("/planner/trajectory")
        );
        assert_eq!(files, system::ros2::generate(&ir).unwrap());
        check_cpp_runtime(n, first_joint, &files);
        let wrong_count = if n == 6 { 7 } else { 6 };
        let wrong = source.replace(
            "PlannerTrajectory<f64,robot_dof>",
            &format!("PlannerTrajectory<f64,{wrong_count}>"),
        );
        let wrong =
            system::compile_at(front::parse_system(wrong.as_bytes()).unwrap(), &directory).unwrap();
        assert_eq!(system::ros2::generate(&wrong).unwrap_err().code, "S1504");
    }
}

fn check_cpp_runtime(
    n: u32,
    first_joint: &str,
    files: &std::collections::BTreeMap<String, String>,
) {
    use std::{fs, process::Command};
    let root = std::env::temp_dir().join(format!(
        "invar-controller-bridge-{}-{n}",
        std::process::id()
    ));
    fs::create_dir(&root).unwrap();
    let includes = root.join("include");
    for (path, content) in files {
        if let Some(relative) = path.strip_prefix("src/invar_cell_interfaces/include/") {
            let output = includes.join(relative);
            fs::create_dir_all(output.parent().unwrap()).unwrap();
            fs::write(output, content).unwrap();
        }
    }
    for directory in ["rclcpp", "invar_cell_interfaces/msg", "trajectory_msgs/msg"] {
        fs::create_dir_all(includes.join(directory)).unwrap();
    }
    fs::write(includes.join("rclcpp/rclcpp.hpp"), r#"
#pragma once
#include <memory>
#include <cstddef>
#include <functional>
#include <utility>
namespace rclcpp {
// CPU stand-ins capture calls made by the real generated callback. They do not
// establish executor, middleware or live-controller qualification.
inline std::function<void(std::shared_ptr<const void>)> receive;
inline std::function<void()> spin_hook;
inline unsigned publications = 0, shutdowns = 0;
struct KeepLast { explicit KeepLast(std::size_t) {} };
struct QoS { explicit QoS(KeepLast) {} QoS & reliable() { return *this; } };
template<class T> struct Subscription { using SharedPtr = std::shared_ptr<Subscription>; };
template<class T> struct Publisher {
 using SharedPtr = std::shared_ptr<Publisher>;
 inline static T last_message{};
 void publish(const T & message) { last_message = message; ++publications; }
};
class Node { public:
 explicit Node(const char *) {} virtual ~Node() = default;
 template<class T> typename Publisher<T>::SharedPtr create_publisher(const char *, const QoS &) {
  return std::make_shared<Publisher<T>>();
 }
 template<class T, class F> typename Subscription<T>::SharedPtr create_subscription(const char *, const QoS &, F callback) {
  receive = [callback = std::move(callback)](std::shared_ptr<const void> message) {
   callback(std::static_pointer_cast<const T>(message));
  };
  return std::make_shared<Subscription<T>>();
 }
};
inline void init(int, char **) {}
inline void shutdown() { ++shutdowns; receive = {}; }
template<class T> void spin(std::shared_ptr<T>) { spin_hook(); }
template<class T> void send(const T & message) { receive(std::make_shared<const T>(message)); }
}
"#).unwrap();
    fs::write(
        includes.join(format!(
            "invar_cell_interfaces/msg/planner_trajectory{n}_v1.hpp"
        )),
        format!(
            r#"
#pragma once
#include <array>
#include <memory>
#include <vector>
namespace invar_cell_interfaces::msg {{
struct Point {{ double time_from_start_s; std::array<double,{n}> q_rad, dq_rad_s; }};
struct PlannerTrajectory{n}V1 {{
 using ConstSharedPtr = std::shared_ptr<const PlannerTrajectory{n}V1>;
 double dt_s; std::vector<Point> waypoints;
}};
}}
"#
        ),
    )
    .unwrap();
    fs::write(
        includes.join("trajectory_msgs/msg/joint_trajectory.hpp"),
        r#"
#pragma once
#include <cstdint>
#include <string>
#include <vector>
namespace trajectory_msgs::msg {
struct Point { std::vector<double> positions, velocities;
 struct Duration { std::int32_t sec; std::uint32_t nanosec; } time_from_start; };
struct JointTrajectory { std::vector<std::string> joint_names; std::vector<Point> points; };
}
"#,
    )
    .unwrap();
    let source = root.join("bridge.cpp");
    fs::write(&source, &files["src/invar_controller_bridge/src/main.cpp"]).unwrap();
    let harness = root.join("runtime.cpp");
    let fixture = include_str!("../fixtures/s1_06/controller_bridge_runtime.cpp")
        .replace("@N@", &n.to_string())
        .replace("@FIRST_JOINT@", first_joint);
    fs::write(&harness, fixture).unwrap();
    let executable = root.join("runtime");
    let include = format!("-I{}", includes.display());
    let flags = [
        "-std=c++17",
        "-Wall",
        "-Wextra",
        "-Werror",
        "-pedantic",
        "-O2",
        &include,
    ];
    child_deadline::cxx(&flags, &harness, &executable);
    let result = Command::new(&executable).output().unwrap();
    assert!(
        result.status.success(),
        "generated {n}-joint callback failed:\n{}\n{}",
        String::from_utf8_lossy(&result.stdout),
        String::from_utf8_lossy(&result.stderr),
    );
    assert_eq!(
        String::from_utf8_lossy(&result.stdout).trim(),
        format!("controller_callback_cases=7 joint_count={n}")
    );
    assert_eq!(
        String::from_utf8_lossy(&result.stderr)
            .matches("\"code\":\"S1404\"")
            .count(),
        6,
    );
    fs::remove_dir_all(root).unwrap();
}
