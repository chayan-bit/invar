use invar::{check, front, robotics};

const PLANNER_WIRE: &str = include_str!("../fixtures/s1_06/planner-wire.inv");

#[test]
fn issue_s1_06_planner_config_requires_additive_v9_and_process_isolation() {
    let manifest = include_str!("../fixtures/s1_06/planner-manifest.inv");
    let source = format!(
        "{manifest}\nfn run() -> i32 {{ return 0; }}
component Sequencer {{
port out request: std::robotics::PlannerRequest<f64,7,base,tool,cube> @ 0.2 Hz;
port in trajectory: std::robotics::PlannerTrajectory<f64,7> @ 0.2 Hz;
resource cpu_cores = 1, memory = 1 MiB;
wcet 1 ms; deadline 5 s; license \"Apache-2.0\"; isolation in-process; impl invar run;
}}
connect sequencer.request -> planner.request policy queue(1);
connect planner.trajectory -> sequencer.trajectory policy queue(1);
product P {{ distribution internal; commercial false; robot menagerie::franka_emika_panda; capacity cpu_cores = 2, memory = 2 GiB, gpu_memory = 2 GiB; }}
system planner_config_probe for P {{ sequencer: Sequencer; planner: curobo_planner; }}"
    );
    let ir = invar::system::compile(front::parse_system(source.as_bytes()).unwrap()).unwrap();
    let json = ir.canonical_json();
    let mut document: serde_json::Value = serde_json::from_str(&json).unwrap();
    assert_eq!(document["schema"], "invar-system-v12");
    assert_eq!(document["version"], 12);
    assert_eq!(document["planner_frame_policy"], "request-consumer-v1");
    invar::system::SystemIr::from_json(json.as_bytes()).unwrap();
    document["schema"] = serde_json::json!("invar-system-v11");
    document["version"] = serde_json::json!(11);
    assert!(invar::system::SystemIr::from_json(document.to_string().as_bytes()).is_err());
    for bad in [
        manifest.replace("separate-process", "in-process"),
        manifest.replace("PlannerTrajectory<f64,7>", "PlannerTrajectory<f64,6>"),
    ] {
        assert!(
            invar::system::compile_manifest(front::parse_system(bad.as_bytes()).unwrap()).is_err()
        );
    }
}

#[test]
fn issue_s1_06_planner_foreign_header_is_checked_and_count_specific() {
    let source = include_str!("../fixtures/s1_06/planner-manifest.inv");
    for n in [6, 7] {
        let source = source
            .replace("f64,7", &format!("f64,{n}"))
            .replace("request7_", &format!("request{n}_"))
            .replace("trajectory7_", &format!("trajectory{n}_"));
        let manifest =
            invar::system::compile_manifest(front::parse_system(source.as_bytes()).unwrap())
                .unwrap();
        let header = invar::system::foreign::header(&manifest).unwrap();
        assert!(header.contains(&format!("invar-planner-component-v2/{n}/")));
        assert!(header.contains("port_request"));
        assert!(header.contains("port_trajectory"));
        assert_eq!(manifest["config"]["size"], 480);
        let mut wrong = manifest.clone();
        wrong["config"]["size"] = serde_json::json!(479);
        assert!(invar::system::foreign::header(&wrong).is_err());
        let mut wrong = manifest;
        wrong["ports"][0]["layout"] = serde_json::json!("inv_pose_f64");
        assert!(invar::system::foreign::header(&wrong).is_err());
    }
}

fn port(ty: &str) -> Result<(), front::FrontError> {
    let source = format!(
        "frame world; frame base: world; frame tool: base; frame object: world; fn port(value: read {ty}) -> i32 {{ return 0; }}"
    );
    check::check_port_types(front::parse(source.as_bytes())?)
}

#[test]
fn issue_s1_06_planner_wire_types_are_distinct_and_bounded() {
    for (n, request_bytes, trajectory_bytes, legacy_bytes) in
        [(6, 272, 26640, 6664), (7, 288, 30736, 7688)]
    {
        let request = format!("std::robotics::PlannerRequest<f64,{n},base,tool,object>");
        let trajectory = format!("std::robotics::PlannerTrajectory<f64,{n}>");
        port(&request).expect("planner request is an admitted wire type");
        port(&trajectory).expect("planner trajectory is an admitted wire type");
        assert_eq!(robotics::c_layout_size(&request), Some(request_bytes));
        assert_eq!(robotics::c_layout_size(&trajectory), Some(trajectory_bytes));
        assert_eq!(
            robotics::c_layout_size(&format!("std::robotics::JointTrajectory<f64,{n}>")),
            Some(legacy_bytes)
        );
    }
}

#[test]
fn issue_s1_06_planner_wire_types_reject_wrong_counts_scalars_and_frames() {
    for ty in [
        "std::robotics::PlannerRequest<f64,5,base,tool,object>",
        "std::robotics::PlannerRequest<f32,6,base,tool,object>",
        "std::robotics::PlannerRequest<f64,6,missing,tool,object>",
        "std::robotics::PlannerTrajectory<f64,8>",
        "std::robotics::PlannerTrajectory<i32,6>",
    ] {
        assert!(
            port(ty).is_err(),
            "invalid planner wire type accepted: {ty}"
        );
    }
}

#[test]
fn issue_s1_06_planner_does_not_expand_general_fixed_array_limit() {
    let source =
        b"struct Wide { values: FixedArray<f64,256> } fn port(x: read Wide) -> i32 { return 0; }";
    assert!(check::check_port_types(front::parse(source).unwrap()).is_err());
}

#[test]
fn issue_s1_06_planner_generated_c_layouts_match_independent_offsets() {
    use std::{fs, process::Command};
    let path = std::env::temp_dir().join(format!("invar-planner-layout-{}", std::process::id()));
    fs::create_dir(&path).unwrap();
    for n in [6, 7] {
        let header = robotics::planner::header(n).unwrap();
        assert_eq!(header, robotics::planner::header(n).unwrap());
        fs::write(path.join(format!("planner{n}.h")), header).unwrap();
    }
    fs::write(
        path.join("layout.c"),
        r#"
#include "planner6.h"
#include "planner7.h"
_Static_assert(offsetof(inv_planner_request6_f64_v2, goal_q_rad) == 48, "six goal");
_Static_assert(offsetof(inv_planner_request6_f64_v2, target_base_from_tool) == 96, "six target");
_Static_assert(offsetof(inv_planner_request6_f64_v2, cube_base_pose) == 152, "six cube");
_Static_assert(offsetof(inv_planner_request6_f64_v2, tool_from_cube) == 208, "six attachment");
_Static_assert(offsetof(inv_planner_request6_f64_v2, phase) == 264, "six phase");
_Static_assert(sizeof(inv_planner_request6_f64_v2) == 272, "six size");
_Static_assert(offsetof(inv_planner_request7_f64_v2, goal_q_rad) == 56, "seven goal");
_Static_assert(offsetof(inv_planner_request7_f64_v2, target_base_from_tool) == 112, "seven target");
_Static_assert(offsetof(inv_planner_request7_f64_v2, phase) == 280, "seven phase");
_Static_assert(sizeof(inv_planner_request7_f64_v2) == 288, "seven size");
_Static_assert(offsetof(inv_planner_waypoint6_f64_v1, dq_rad_s) == 56, "six speed");
_Static_assert(offsetof(inv_planner_waypoint7_f64_v1, dq_rad_s) == 64, "seven speed");
int main(void) { return 0; }
"#,
    )
    .unwrap();
    let result = Command::new("cc")
        .args(["-std=c11", "-Wall", "-Wextra", "-Werror", "-pedantic"])
        .arg(path.join("layout.c"))
        .arg("-o")
        .arg(path.join("layout"))
        .output()
        .unwrap();
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    assert!(
        Command::new(path.join("layout"))
            .status()
            .unwrap()
            .success()
    );
    fs::remove_dir_all(path).unwrap();
    assert!(robotics::planner::header(5).is_err());
    assert!(robotics::planner::header(8).is_err());
}

#[test]
fn issue_s1_06_planner_system_ir_is_additive_and_specializes_robot_dof() {
    let ir = invar::system::compile(front::parse_system(PLANNER_WIRE.as_bytes()).unwrap())
        .expect("versioned planner wire graph");
    assert!(ir.graph_issues().is_empty(), "{:?}", ir.graph_issues());
    let json = ir.canonical_json();
    let document: serde_json::Value = serde_json::from_str(&json).unwrap();
    assert_eq!(document["schema"], "invar-system-v12");
    assert_eq!(document["version"], 12);
    assert_eq!(document["planner_frame_policy"], "request-consumer-v1");
    assert_eq!(document["planner_contract"], "invar-planner-wire-v1");
    assert!(!json.contains("robot_dof"));
    invar::system::SystemIr::from_json(json.as_bytes()).unwrap();
    let mut malformed = document;
    malformed["planner_contract"] = serde_json::json!("unknown");
    assert!(invar::system::SystemIr::from_json(malformed.to_string().as_bytes()).is_err());
}

#[test]
fn issue_s1_06_request_consumers_require_measured_sources_and_preserve_legacy() {
    use invar::system::{SystemIr, graph};
    use serde_json::{Value, json};
    let ir = invar::system::compile(front::parse_system(PLANNER_WIRE.as_bytes()).unwrap()).unwrap();
    let document: Value = serde_json::from_str(&ir.canonical_json()).unwrap();
    assert!(graph::check(&document).is_empty());
    for port_name in ["tool_pose", "object_pose"] {
        let mut missing = document.clone();
        let ports = missing["components"]
            .as_array_mut()
            .unwrap()
            .iter_mut()
            .find(|c| c["name"] == "MeasuredFrameProbe")
            .unwrap()["ports"]
            .as_array_mut()
            .unwrap();
        ports.retain(|port| port["name"] != port_name);
        assert!(graph::check(&missing).iter().any(|i| i.code == "S1201"));
    }
    let mut duplicate = document.clone();
    let instance = duplicate["system"]["instances"]
        .as_array()
        .unwrap()
        .iter()
        .find(|i| i["component"] == "MeasuredFrameProbe")
        .unwrap()
        .clone();
    let mut second = instance;
    second["name"] = json!("duplicate");
    second["id"] = json!("duplicate");
    duplicate["system"]["instances"]
        .as_array_mut()
        .unwrap()
        .push(second);
    assert!(graph::check(&duplicate).iter().any(|i| i.code == "S1202"));
    for policy in [None, Some("unknown")] {
        let mut wrong = document.clone();
        match policy {
            None => {
                wrong
                    .as_object_mut()
                    .unwrap()
                    .remove("planner_frame_policy");
            }
            Some(p) => wrong["planner_frame_policy"] = json!(p),
        }
        assert!(SystemIr::from_json(wrong.to_string().as_bytes()).is_err());
        assert!(graph::check(&wrong).iter().any(|i| i.code == "S1200"));
    }
    let mut undeclared = document.clone();
    undeclared["frames"]
        .as_array_mut()
        .unwrap()
        .retain(|f| f["name"] != "object");
    assert!(graph::check(&undeclared).iter().any(|i| i.code == "S1200"));
    for (bad, code) in [
        (PLANNER_WIRE.replace("frame object: base;", ""), "S1001"),
        (
            PLANNER_WIRE.replace("frame object: base;", "frame object;"),
            "S1200",
        ),
    ] {
        match invar::system::compile(front::parse_system(bad.as_bytes()).unwrap()) {
            Ok(ir) => assert!(ir.graph_issues().iter().any(|i| i.0 == code)),
            Err(e) => assert_eq!(e.code, code),
        }
    }
}
