// Shared fixtures, each loaded once and used by several modules below.
mod file_hash;
mod issue_s1_01_system_dialect;
mod issue_s1_02_foreign_manifests;
mod issue_s1_03_graph_checks;
mod issue_s1_04_ros2_target;
mod issue_s1_04_ros2_target_env;
mod issue_s1_05_robot_cell_mujoco;
mod issue_s1_05_robot_cell_mujoco_env;
mod issue_s1_06_controller_bridge;
mod issue_s1_06_grasp_v10;
mod issue_s1_06_integration_start;
mod issue_s1_06_integration_start_env;
mod issue_s1_06_mount_exclusion;
mod issue_s1_06_panda_inertia;
mod issue_s1_06_planner_boundary;
mod issue_s1_06_planner_contract;
mod issue_s1_06_planner_requests;
mod issue_s1_06_planner_ros_build;
mod issue_s1_06_planner_ros_messages;
mod issue_s1_06_planner_ros_node;
mod issue_s1_06_planner_runtime;
mod issue_s1_06_planner_transport;
mod issue_s1_06_sequencer_binding;
mod issue_s1_06_sequencer_preparation;
mod issue_s1_06_tool_projection;
mod issue_s1_11_robotics_vocabulary;
#[allow(dead_code)]
#[path = "../support/planner_ros_build.rs"]
mod planner_ros_build_support;
#[allow(dead_code)]
#[path = "../support/sequencer_binding.rs"]
mod sequencer_binding_support;
mod substep_admission_source;
