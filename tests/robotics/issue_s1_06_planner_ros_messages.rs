use crate::child_deadline;
use invar::{front, system};

#[test]
fn issue_s1_06_planner_ros_messages_preserve_bounds_and_joint_counts() {
    for n in [6, 7] {
        let source = include_str!("../fixtures/s1_06/planner-wire.inv")
            .replace("robot_dof", &n.to_string())
            .replace("f64,6", &format!("f64,{n}"));
        let ir = system::compile(front::parse_system(source.as_bytes()).unwrap()).unwrap();
        let files = system::ros2::generate(&ir).expect("planner wire messages must generate");
        assert_eq!(files, system::ros2::generate(&ir).unwrap());
        let root = "src/invar_cell_interfaces";
        let request = &files[&format!("{root}/msg/PlannerRequest{n}V2.msg")];
        assert!(request.contains(&format!("float64[{n}] start_q_rad")));
        assert!(request.contains(&format!("float64[{n}] goal_q_rad")));
        assert!(request.contains("float64[7] target_base_from_tool"));
        assert!(request.contains("float64[7] cube_base_pose"));
        assert!(request.contains("float64[7] tool_from_cube"));
        assert!(request.contains("uint32 phase"));
        let waypoint = &files[&format!("{root}/msg/PlannerWaypoint{n}V1.msg")];
        assert!(waypoint.contains(&format!("float64[{n}] q_rad")));
        assert!(waypoint.contains(&format!("float64[{n}] dq_rad_s")));
        let trajectory = &files[&format!("{root}/msg/PlannerTrajectory{n}V1.msg")];
        assert!(trajectory.contains(&format!("PlannerWaypoint{n}V1[<=256] waypoints")));
        assert!(trajectory.contains("float64 dt_s"));
        let cmake = &files[&format!("{root}/CMakeLists.txt")];
        for (kind, version) in [("Request", 2), ("Waypoint", 1), ("Trajectory", 1)] {
            assert!(cmake.contains(&format!("msg/Planner{kind}{n}V{version}.msg")));
            // Independent Python json.dumps/SHA256 reference, not ROS build evidence.
            let expected: serde_json::Value =
                serde_json::from_str(include_str!("../fixtures/s1_06/planner-ros-hashes.json"))
                    .unwrap();
            let name = format!("Planner{kind}{n}V{version}");
            let record: serde_json::Value =
                serde_json::from_str(&files[&format!("{root}/hashes/{name}.json")]).unwrap();
            assert_eq!(record["hash_string"], expected[&name]);
        }
        // This checks deterministic generation only, not a live ROS build or planner run.
        check_conversions(n, &files);
    }
}

fn check_conversions(n: u32, files: &std::collections::BTreeMap<String, String>) {
    use std::{fs, process::Command};
    let root = std::env::temp_dir().join(format!("invar-planner-ros-{}-{n}", std::process::id()));
    fs::create_dir(&root).unwrap();
    for name in [
        format!("planner{n}_wire.h"),
        format!("planner{n}_conversions.hpp"),
    ] {
        fs::write(
            root.join(&name),
            &files[&format!("src/invar_cell_interfaces/include/invar_cell_interfaces/{name}")],
        )
        .unwrap();
    }
    let source = include_str!("../fixtures/s1_06/planner_ros_conversions.cpp")
        .replace("@N@", &n.to_string());
    fs::write(root.join("probe.cpp"), source).unwrap();
    let executable = root.join("probe");
    child_deadline::cxx(
        &["-std=c++17", "-Wall", "-Wextra", "-Werror", "-pedantic"],
        &root.join("probe.cpp"),
        &executable,
    );
    let result = Command::new(&executable).output().unwrap();
    child_deadline::assert_success(&result);
    fs::remove_dir_all(root).unwrap();
}
