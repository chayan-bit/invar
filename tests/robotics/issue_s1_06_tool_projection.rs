//! CPU adapter execution, not ROS/cuRobo or complete task qualification.
use invar::{
    front,
    system::{
        self,
        tool_projection::{self, Pose},
    },
};
use serde_json::{Value, json};
use std::{fs, path::Path, process::Command};

// Shared fixture also supplies hash/build helpers used by other tests.
#[allow(dead_code)]
#[path = "../support/planner_ros_build.rs"]
mod planner_ros_build;

fn ir(arm: &str) -> system::SystemIr {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("examples/cells/reach_place");
    let name = if arm == "panda" {
        "cell.inv"
    } else {
        "cell-ur5e.inv"
    };
    system::compile_at(
        front::parse_system(&fs::read(dir.join(name)).unwrap()).unwrap(),
        &dir,
    )
    .unwrap()
}
fn near(a: Pose, b: Pose) {
    assert!(
        a.translation_m
            .iter()
            .zip(b.translation_m)
            .all(|(x, y)| (x - y).abs() < 1e-12)
    );
    let same = a
        .quaternion_xyzw
        .iter()
        .zip(b.quaternion_xyzw)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f64::max);
    let opposite = a
        .quaternion_xyzw
        .iter()
        .zip(b.quaternion_xyzw)
        .map(|(x, y)| (x + y).abs())
        .fold(0.0, f64::max);
    assert!(same.min(opposite) < 1e-12);
}
fn cpp_pose(p: Pose) -> String {
    format!(
        "P{{{{{}}},{{{}}}}}",
        p.translation_m
            .iter()
            .map(|x| format!("{x:.17e}"))
            .collect::<Vec<_>>()
            .join(","),
        p.quaternion_xyzw
            .iter()
            .map(|x| format!("{x:.17e}"))
            .collect::<Vec<_>>()
            .join(",")
    )
}

#[test]
fn issue_s1_06_source_tool_projection_executes_rust_and_generated_cpp_both_arms() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let reference: Value = serde_json::from_slice(
        &fs::read(root.join("tests/fixtures/foreign-components/pinocchio_goldens.json")).unwrap(),
    )
    .unwrap();
    let audit: Value = serde_json::from_slice(
        &fs::read(root.join(
            "tests/fixtures/archived-evidence/s1-06/ur5e-tool-frame-20261014/tool-frame-audit.json",
        ))
        .unwrap(),
    )
    .unwrap();
    for (arm, n) in [("panda", 7), ("ur5e", 6)] {
        let checked = ir(arm);
        let projection = tool_projection::selected(&checked).unwrap();
        let original = checked.canonical_json();
        let expected = if n == 7 {
            Pose {
                translation_m: [0.0, 0.0, 0.1034],
                quaternion_xyzw: [0.0, 0.0, 0.0, 1.0],
            }
        } else {
            // Decision 2026-10-14: the UR5e cell tool is declared at attachment_site.
            Pose::IDENTITY
        };
        near(projection.reference_from_cell, expected);
        // An identity transform is its own inverse; direction cases need a non-identity one.
        let r = projection.reference_from_cell;
        let directed = r.translation_m.iter().any(|v| v.abs() > 1e-9)
            || r.quaternion_xyzw[3].abs() < 1.0 - 1e-12;
        tool_projection::verify_value(&projection.value()).unwrap();
        assert_eq!(projection.schema, "invar-tool-projection-v2");
        for case in 0..6 {
            let mut corrupt = projection.value();
            match case {
                0 => corrupt["reference_model_sha256"] = json!("0".repeat(64)),
                1 => corrupt["reference_implementation_sha256"] = json!("0".repeat(64)),
                2 => {
                    corrupt["reference_from_cell"]["quaternion_xyzw"] = json!([0.0, 0.0, 0.0, 2.0])
                }
                3 if !directed => continue,
                3 => {
                    let inverse = projection.reference_from_cell.inverse().unwrap();
                    corrupt["reference_from_cell"] = json!({"translation_m":inverse.translation_m,"quaternion_xyzw":inverse.quaternion_xyzw});
                }
                4 => {
                    corrupt["source_base_from_reference_base"]["translation_m"] =
                        json!([0.1, 0.0, 0.0])
                }
                _ => corrupt["reference_base_frame"] = json!("unrelated"),
            }
            assert!(tool_projection::verify_value(&corrupt).is_err());
        }
        let files = planner_ros_build::workspace(n);
        let metadata: Value = serde_json::from_str(
            files["src/invar_planner_node/tool-projection.json"]
                .lines()
                .filter(|line| !line.starts_with("//"))
                .collect::<Vec<_>>()
                .join("\n")
                .as_str(),
        )
        .unwrap();
        assert_eq!(metadata, projection.value());
        assert_eq!(
            files["src/invar_fk_node/include/invar_fk_node/tool_projection.hpp"],
            files["src/invar_planner_node/include/invar_planner_node/tool_projection.hpp"]
        );
        assert!(
            files["src/invar_fk_node/src/node.cpp"]
                .contains("fk_to_cell(output.pose.translation_m")
        );
        assert!(
            files["src/invar_planner_node/src/node.cpp"]
                .contains("request_to_reference(input.port_request)")
        );
        assert_eq!(checked.canonical_json(), original);
        let temp =
            std::env::temp_dir().join(format!("invar-tool-projection-{}-{n}", std::process::id()));
        fs::create_dir(&temp).unwrap();
        fs::write(
            temp.join("tool_projection.hpp"),
            &files["src/invar_planner_node/include/invar_planner_node/tool_projection.hpp"],
        )
        .unwrap();
        let arm_ref = reference["arms"]
            .as_array()
            .unwrap()
            .iter()
            .find(|a| a["arm"] == arm)
            .unwrap();
        let mut program = String::from(
            "#include \"tool_projection.hpp\"\n#include <cassert>\n#include <limits>\n#include <cstring>\nusing namespace invar_tool_projection_v1; using P=Pose;\nstruct Wire {double xyz_m[3];double quaternion_xyzw[4];}; struct Request {Wire target_base_from_tool,tool_from_cube,cube_base_pose; double start_q_rad[7];unsigned attached;};\nvoid near(P a,P b) { for(int i=0;i<3;++i) assert(std::abs(a.p[i]-b.p[i])<1e-12); double same=0,opposite=0;for(int i=0;i<4;++i) {same=std::fmax(same,std::abs(a.q[i]-b.q[i]));opposite=std::fmax(opposite,std::abs(a.q[i]+b.q[i]));} assert(std::fmin(same,opposite)<1e-12);}\nint main(){\n",
        );
        for (index, v) in arm_ref["vectors"].as_array().unwrap().iter().enumerate() {
            let arr = |v: &Value, len: usize| {
                let a = v.as_array().unwrap();
                assert_eq!(a.len(), len);
                a.iter().map(|x| x.as_f64().unwrap()).collect::<Vec<_>>()
            };
            let p = arr(&v["translation_m"], 3);
            let q = arr(&v["quaternion_wxyz"], 4);
            let raw = Pose {
                translation_m: p.try_into().unwrap(),
                quaternion_xyzw: [q[1], q[2], q[3], q[0]],
            };
            let actual = projection.fk_to_source(raw).unwrap();
            let row = audit["rows"]
                .as_array()
                .unwrap()
                .iter()
                .find(|r| r["arm"] == arm && r["source_vector"] == index)
                .unwrap();
            let p = arr(&row["generated_site_base_translation_m"], 3);
            let q = arr(&row["generated_site_base_quaternion_wxyz"], 4);
            let source = Pose {
                translation_m: p.try_into().unwrap(),
                quaternion_xyzw: [q[1], q[2], q[3], q[0]],
            };
            let source = projection
                .source_base_from_reference_base
                .compose(source)
                .unwrap();
            near(actual, source);
            near(
                projection
                    .source_base_from_reference_base
                    .inverse()
                    .unwrap()
                    .compose(actual)
                    .unwrap()
                    .compose(projection.reference_from_cell.inverse().unwrap())
                    .unwrap(),
                raw,
            );
            // Wrong direction must fail unchanged 2mm/1degree gates at every state.
            let wrong = raw
                .compose(projection.reference_from_cell.inverse().unwrap())
                .unwrap();
            let position = wrong
                .translation_m
                .iter()
                .zip(source.translation_m)
                .map(|(a, b)| (a - b).powi(2))
                .sum::<f64>()
                .sqrt();
            let dot = wrong
                .quaternion_xyzw
                .iter()
                .zip(source.quaternion_xyzw)
                .map(|(a, b)| a * b)
                .sum::<f64>()
                .abs()
                .min(1.0);
            assert!(
                !directed || position > 0.002 || 2.0 * dot.acos() > std::f64::consts::PI / 180.0
            );
            program += &format!(
                "{{P raw={},expected={}; P actual=raw;assert(fk_to_cell(actual.p,actual.q));near(actual,expected); Request request{{}}; for(int i=0;i<3;++i)request.target_base_from_tool.xyz_m[i]=actual.p[i];for(int i=0;i<4;++i)request.target_base_from_tool.quaternion_xyzw[i]=actual.q[i];request.tool_from_cube.quaternion_xyzw[3]=1;request.cube_base_pose.xyz_m[0]=0.3;request.cube_base_pose.quaternion_xyzw[3]=1;request.start_q_rad[0]=0.1; request.attached=1;auto original=request;assert(request_to_reference(request));near(P{{{{request.target_base_from_tool.xyz_m[0],request.target_base_from_tool.xyz_m[1],request.target_base_from_tool.xyz_m[2]}},{{request.target_base_from_tool.quaternion_xyzw[0],request.target_base_from_tool.quaternion_xyzw[1],request.target_base_from_tool.quaternion_xyzw[2],request.target_base_from_tool.quaternion_xyzw[3]}}}},raw); near(P{{{{request.tool_from_cube.xyz_m[0],request.tool_from_cube.xyz_m[1],request.tool_from_cube.xyz_m[2]}},{{request.tool_from_cube.quaternion_xyzw[0],request.tool_from_cube.quaternion_xyzw[1],request.tool_from_cube.quaternion_xyzw[2],request.tool_from_cube.quaternion_xyzw[3]}}}},reference_from_cell);P cube{{{{original.cube_base_pose.xyz_m[0],original.cube_base_pose.xyz_m[1],original.cube_base_pose.xyz_m[2]}},{{0,0,0,1}}}};assert(apply(cube.p,cube.q,inverse(source_base_from_reference_base),identity));near(P{{{{request.cube_base_pose.xyz_m[0],request.cube_base_pose.xyz_m[1],request.cube_base_pose.xyz_m[2]}},{{request.cube_base_pose.quaternion_xyzw[0],request.cube_base_pose.quaternion_xyzw[1],request.cube_base_pose.quaternion_xyzw[2],request.cube_base_pose.quaternion_xyzw[3]}}}},cube);assert(request.start_q_rad[0]==0.1 && request.attached==1);}}\n",
                cpp_pose(raw),
                cpp_pose(source)
            );
        }
        program += "Request invalid{};invalid.target_base_from_tool.quaternion_xyzw[3]=1;invalid.cube_base_pose.quaternion_xyzw[3]=1;invalid.tool_from_cube.quaternion_xyzw[3]=2;auto original=invalid;assert(!request_to_reference(invalid));assert(std::memcmp(&invalid,&original,sizeof(Request))==0); Request bad_cube{};bad_cube.target_base_from_tool.quaternion_xyzw[3]=1;bad_cube.tool_from_cube.quaternion_xyzw[3]=1;bad_cube.cube_base_pose.quaternion_xyzw[3]=2;auto saved_cube=bad_cube;assert(!request_to_reference(bad_cube));assert(std::memcmp(&bad_cube,&saved_cube,sizeof(Request))==0); P nan=identity;nan.p[0]=std::numeric_limits<double>::quiet_NaN();assert(!fk_to_cell(nan.p,nan.q));return 0;}\n";
        fs::write(
            temp.join("model.xml"),
            reference["arms"]
                .as_array()
                .unwrap()
                .iter()
                .find(|a| a["arm"] == arm)
                .map(|a| {
                    fs::read(
                        root.join("examples/cells/reach_place/assets/menagerie")
                            .join(a["model_file"].as_str().unwrap()),
                    )
                    .unwrap()
                })
                .unwrap(),
        )
        .unwrap();
        let mut extra = fs::read(temp.join("model.xml")).unwrap();
        extra.push(b' ');
        fs::write(temp.join("bad.xml"), extra).unwrap();
        // Execute the exact-byte model guard, including trailing-byte refusal.
        program=program.replace("int main(){",&format!("int main(){{assert(reference_model_matches(\"{}\"));assert(!reference_model_matches(\"{}\"));",temp.join("model.xml").display(),temp.join("bad.xml").display()));
        fs::write(temp.join("probe.cpp"), program).unwrap();
        let result = Command::new("c++")
            .args(["-std=c++17", "-Wall", "-Wextra", "-Werror", "-pedantic"])
            .arg(temp.join("probe.cpp"))
            .arg("-o")
            .arg(temp.join("probe"))
            .output()
            .unwrap();
        assert!(
            result.status.success(),
            "{}",
            String::from_utf8_lossy(&result.stderr)
        );
        assert!(Command::new(temp.join("probe")).status().unwrap().success());
        fs::remove_dir_all(temp).unwrap();
    }
}

#[test]
fn issue_s1_06_projection_rejects_unknown_models_wrong_anchor_and_corrupt_site() {
    for arm in ["panda", "ur5e"] {
        let checked = ir(arm);
        let d = checked.checked_document().unwrap();
        let id = if arm == "panda" {
            "menagerie::franka_emika_panda"
        } else {
            "menagerie::universal_robots_ur5e"
        };
        let robot = d["robots"]
            .as_array()
            .unwrap()
            .iter()
            .find(|r| r["id"] == id)
            .unwrap();
        for case in 0..4 {
            let mut corrupt = robot.clone();
            match case {
                0 => corrupt["asset"]["sha256"] = json!("0".repeat(64)),
                1 => corrupt["tool"]["link"] = json!("unrelated"),
                2 => corrupt["tool"]["pose"]["quat"] = json!(["0", "0", "0", "0"]),
                _ => corrupt["source_tree"] = json!({"tag":"mujoco","attrs":{},"children":[]}),
            }
            assert!(tool_projection::derive(&corrupt).is_err());
        }
        // The cell declaration drives the offset; it is not a fitted constant.
        let mut moved = robot.clone();
        moved["tool"]["pose"]["xyz"][2]["value"] = json!("0.2");
        assert_ne!(
            tool_projection::derive(&moved).unwrap().reference_from_cell,
            tool_projection::derive(robot).unwrap().reference_from_cell
        );
    }
    assert!(
        Pose {
            translation_m: [f64::INFINITY, 0.0, 0.0],
            ..Pose::IDENTITY
        }
        .inverse()
        .is_err()
    );
    assert!(
        Pose {
            quaternion_xyzw: [0.0, 0.0, 0.0, 2.0],
            ..Pose::IDENTITY
        }
        .compose(Pose::IDENTITY)
        .is_err()
    );
}

fn z_axis(p: Pose) -> [f64; 3] {
    let [x, y, z, w] = p.quaternion_xyzw;
    [
        2.0 * (x * z + w * y),
        2.0 * (y * z - w * x),
        1.0 - 2.0 * (x * x + y * y),
    ]
}

/// The declared cell tool's +z is the flange normal: at every golden FK state it equals
/// the +z of the FK reference frame (Panda hand, UR5e attachment_site/tool0), both arms.
/// The UR5e tool declared before decision 2026-10-14 was 90 degrees off this axis.
#[test]
fn issue_s1_06_declared_tool_z_is_the_fk_flange_normal_both_arms() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let goldens: Value = serde_json::from_slice(
        &fs::read(root.join("tests/fixtures/foreign-components/pinocchio_goldens.json")).unwrap(),
    )
    .unwrap();
    for arm in ["panda", "ur5e"] {
        let projection = tool_projection::selected(&ir(arm)).unwrap();
        let base = projection
            .source_base_from_reference_base
            .inverse()
            .unwrap();
        let vectors = goldens["arms"]
            .as_array()
            .unwrap()
            .iter()
            .find(|a| a["arm"] == arm)
            .unwrap()["vectors"]
            .as_array()
            .unwrap();
        assert_eq!(vectors.len(), 4);
        for vector in vectors {
            let f = |v: &Value, i: usize| v[i].as_f64().unwrap();
            let (t, q) = (&vector["translation_m"], &vector["quaternion_wxyz"]);
            let fk = Pose {
                translation_m: [f(t, 0), f(t, 1), f(t, 2)],
                quaternion_xyzw: [f(q, 1), f(q, 2), f(q, 3), f(q, 0)],
            };
            // The cell tool in the FK reference base frame.
            let tool = base.compose(projection.fk_to_source(fk).unwrap()).unwrap();
            let (a, b) = (z_axis(tool), z_axis(fk));
            let error = (0..3).map(|i| (a[i] - b[i]).abs()).fold(0.0, f64::max);
            assert!(
                error < 1e-12,
                "{arm}: tool z {a:?} is not the flange normal {b:?}"
            );
        }
    }
}
