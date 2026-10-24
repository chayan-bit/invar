//! Actual generator fixture with labelled planner/FK test doubles; not qualification.
use invar::{front, system};
use std::{collections::BTreeMap, fs, path::Path};
/// The example cells' explicit cube grasp clause.
pub const GRASP: &str = "        grasp pose xyz(0 m, 0 m, 0.025 m) quat(0, 1, 0, 0);\n";
pub fn source(n: usize) -> String {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("examples/cells/reach_place");
    let original =
        fs::read_to_string(root.join(if n == 6 { "cell-ur5e.inv" } else { "cell.inv" })).unwrap();
    let planner = include_str!("../fixtures/s1_06/planner-manifest.inv")
        .lines()
        .filter(|s| !s.starts_with("frame "))
        .collect::<Vec<_>>()
        .join("\n")
        .replace("f64,7", &format!("f64,{n}"))
        .replace("request7_", &format!("request{n}_"))
        .replace("trajectory7_", &format!("trajectory{n}_"));
    let task = format!(
        r#"
{planner}
component TaskBindingProbe {{
 param robot_mount: Transform<robot_base,world> = identity;
 port out measured_cube: std::robotics::Pose<f64,world,cube> @ 100 Hz;
 port in tool: std::robotics::Pose<f64,base,tool> @ 100 Hz;
 port out request: std::robotics::PlannerRequest<f64,{n},base,tool,cube> @ 0.2 Hz;
 port in planned: std::robotics::PlannerTrajectory<f64,{n}> @ 0.2 Hz;
 port out approved: std::robotics::PlannerTrajectory<f64,{n}> @ 0.2 Hz;
 resource cpu_cores = 1, memory = 256 MiB;
 license "Apache-2.0";isolation separate-process;wcet 1 ms;deadline 10 ms;
 impl extern invar::s1_06_reach_place_v1;
}}
"#
    );
    original.replace("component joint_trajectory_controller {",&format!("{task}\ncomponent joint_trajectory_controller {{\n port in trajectory: std::robotics::PlannerTrajectory<f64,{n}> @ 0.2 Hz;"))
        .replace("connect controller.cmd","connect sequencer.request -> planner.request policy queue(1);\nconnect planner.trajectory -> sequencer.planned policy queue(1);\nconnect sequencer.approved -> controller.trajectory policy queue(1);\nconnect fk.pose -> sequencer.tool policy sample-and-hold;\nconnect controller.cmd")
        .replace("    controller: joint_trajectory_controller;","    sequencer: TaskBindingProbe;\n    planner: curobo_planner;\n    controller: joint_trajectory_controller;")
}
pub fn generate(source: &str) -> Result<BTreeMap<String, String>, system::ros2::GenerateError> {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("examples/cells/reach_place");
    let parsed =
        front::parse_system(source.as_bytes()).map_err(|e| system::ros2::GenerateError {
            code: e.code,
            message: e.message,
        })?;
    let ir = system::compile_at(parsed, &root).map_err(|e| system::ros2::GenerateError {
        code: e.code,
        message: e.message,
    })?;
    system::ros2::generate(&ir)
}

pub fn scene(n: usize) -> BTreeMap<String, Vec<u8>> {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("examples/cells/reach_place");
    let parsed = front::parse_system(source(n).as_bytes()).unwrap();
    let ir = system::compile_at(parsed, &root).unwrap();
    system::generate_ros2_mjcf(&ir).unwrap()
}
