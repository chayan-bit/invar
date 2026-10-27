//! Public CLI with explicit CPU doubles. These tests produce no planner pass.
use super::file_hash::hash;
use serde_json::{Value, json};
use std::{
    fs,
    path::{Path, PathBuf},
    process::Command,
};

fn write(path: &Path, value: &Value) {
    fs::write(path, serde_json::to_vec_pretty(value).unwrap()).unwrap();
}
fn compile(root: &Path, file: &str, library: &str) -> PathBuf {
    let path = root.join(format!("{library}.{}", std::env::consts::DLL_EXTENSION));
    let result = Command::new("cc")
        .args([
            "-std=c11",
            "-O2",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-pedantic",
            "-fPIC",
        ])
        .arg(if cfg!(target_os = "macos") {
            "-dynamiclib"
        } else {
            "-shared"
        })
        .arg(root.join(file))
        .arg("-o")
        .arg(&path)
        .output()
        .unwrap();
    crate::child_deadline::assert_success(&result);
    path
}
fn cli(args: &[&Path]) -> std::process::Output {
    Command::new(env!("CARGO_BIN_EXE_invar"))
        .args(args)
        .output()
        .unwrap()
}
fn exercise(n: usize) {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("target")
        .join(format!("invar-planner-contract-{}-{n}", std::process::id()));
    // CARGO_TARGET_DIR may place Cargo outputs outside the repository.
    // Preserve a unique fixture root while creating its missing parent.
    fs::create_dir_all(root.parent().unwrap()).unwrap();
    fs::create_dir(&root).unwrap();
    for file in ["model.xml", "model.urdf", "robot.yml", "scene.yml", "child"] {
        fs::write(root.join(file), "CPU fixture; never hardware\n").unwrap();
    }
    let repo = Path::new(env!("CARGO_MANIFEST_DIR"));
    let cell_name = if n == 7 { "cell.inv" } else { "cell-ur5e.inv" };
    let model_name = if n == 7 {
        "franka_emika_panda"
    } else {
        "universal_robots_ur5e"
    };
    let cell_path = repo.join("examples/cells/reach_place").join(cell_name);
    let projection = {
        fs::copy(
            repo.join("examples/cells/reach_place/assets/menagerie")
                .join(model_name)
                .join(if n == 7 { "panda.xml" } else { "ur5e.xml" }),
            root.join("model.xml"),
        )
        .unwrap();
        let ir = invar::system::compile_at(
            invar::front::parse_system(&fs::read(&cell_path).unwrap()).unwrap(),
            cell_path.parent().unwrap(),
        )
        .unwrap();
        invar::system::tool_projection::selected(&ir).unwrap()
    };
    // v3 runtimes and requests must carry the cell-derived table, cube and exclusions.
    let scene = {
        let ir = invar::system::compile_at(
            invar::front::parse_system(&fs::read(&cell_path).unwrap()).unwrap(),
            cell_path.parent().unwrap(),
        )
        .unwrap();
        invar::system::planner_scene::selected(&ir).unwrap()
    };
    fs::write(
        root.join("wire.h"),
        invar::robotics::planner::header(n as u32).unwrap(),
    )
    .unwrap();
    fs::write(root.join("fk.c"),r#"
#include "fixture_fk.h"
#include <stdlib.h>
struct fixture_fk_context { int configured; };
const char *fixture_fk_version(void) { return "invar-component-v1/fk-cpu-fixture"; }
int fixture_fk_init(fixture_fk_context **context, const uint8_t *child, uint32_t size) { (void)child; (void)size; *context=calloc(1,sizeof(**context)); return *context?0:3; }
int fixture_fk_configure(fixture_fk_context *context, const fixture_fk_config *config) { (void)config; return context?0:2; }
int fixture_fk_step(fixture_fk_context *context,const fixture_fk_inputs *input,fixture_fk_outputs *output) {
 if(!context) return 2;
 output->pose=(fixture_fk_pose_f64){{input->q.values[0],0,0},{0,0,0,1}}; return 0;
}
int fixture_fk_shutdown(fixture_fk_context **context) { free(*context);*context=NULL;return 0; }
"#).unwrap();
    let names: Vec<_> = (0..n).map(|i| format!("joint{i}")).collect();
    let mut second = vec![0.0; n];
    second[0] = 0.1;
    let fk_golden = json!({"joint_names":names,"model_sha256":hash(&root.join("model.xml")),"vectors":[
        {"q_rad":vec![0.0;n],"translation_m":[0.0,0.0,0.0],"quaternion_xyzw":[0.0,0.0,0.0,1.0]},
        {"q_rad":second,"translation_m":[0.1,0.0,0.0],"quaternion_xyzw":[0.0,0.0,0.0,1.0]}]});
    write(&root.join("fk-golden.json"), &fk_golden);
    fs::write(
        root.join("fk.inv"),
        format!(
            r#"
frame world;frame base:world;frame tool:base;
extern component fixture_fk {{
 abi c "fixture_fk";lifecycle init, configure, step, shutdown;config c "inv_fk_config_v1";
 port in q:std::robotics::JointPositions<f64,{n}> @ 10 Hz layout c "inv_joint{n}_f64";
 port out pose:std::robotics::Pose<f64,base,tool> @ 10 Hz layout c "inv_pose_f64";
 resource cpu_cores=1,memory=1 MiB;wcet 1 ms;deadline 100 ms;
 license "BSD-2-Clause" link dynamic;isolation in-process;
 golden "fk-golden.json" tolerance abs 0.000000000001 m,abs 0.000000000001 rad;
 property "pose.rotation is unit norm within 0.000000000001";shim "fk.c";
}}
"#
        ),
    )
    .unwrap();
    let fk_manifest = invar::system::compile_manifest(
        invar::front::parse_system(&fs::read(root.join("fk.inv")).unwrap()).unwrap(),
    )
    .unwrap();
    fs::write(
        root.join("fixture_fk.h"),
        invar::system::foreign::header(&fk_manifest).unwrap(),
    )
    .unwrap();
    let fk_library = compile(&root, "fk.c", "fk");
    let fk_output = root.join("fk-evidence");
    let result = cli(&[
        Path::new("component"),
        Path::new("test"),
        &root.join("fk.inv"),
        Path::new("--library"),
        &fk_library,
        Path::new("--model"),
        &root.join("model.xml"),
        Path::new("--output"),
        &fk_output,
    ]);
    crate::child_deadline::assert_success(&result);
    let fk_evidence = PathBuf::from(
        serde_json::from_slice::<Value>(&result.stdout).unwrap()["evidence"]
            .as_str()
            .unwrap(),
    );
    // Dependency header is virtual/hash-bound from the declared FK manifest.
    // The planner snapshot must not require a mutable shared header on disk.
    fs::remove_file(root.join("fixture_fk.h")).unwrap();
    fs::write(root.join("planner.inv"),format!(r#"
frame world;frame base:world;frame tool:base;frame cube:world;
extern component fixture_planner {{
 abi c "fixture_planner";lifecycle init, configure, step, shutdown;config c "inv_planner_config_v1";
 port in request:std::robotics::PlannerRequest<f64,{n},base,tool,cube> @ 0.2 Hz layout c "inv_planner_request{n}_f64_v2";
 port out trajectory:std::robotics::PlannerTrajectory<f64,{n}> @ 0.2 Hz layout c "inv_planner_trajectory{n}_f64_v1";
 resource cpu_cores=1,memory=1 MiB,gpu_memory=1 MiB;wcet 5 s;deadline 5 s;
 license "Apache-2.0";isolation separate-process;
 golden "endpoint.json" tolerance abs 0.002 m,abs 0.017453292519943295 rad;
 property "trajectory is finite";shim "planner.c";input "wire.h";input "runtime.json";
 input "planner.{extension}";input "fk.{extension}";input "child";
 input "model.urdf";input "model.xml";input "robot.yml";input "scene.yml";
 input "fk.inv";input "fk.c";input "fk-golden.json";input "{fk_record}";
}}
"#,
extension=std::env::consts::DLL_EXTENSION,
fk_record=fk_evidence.strip_prefix(&root).unwrap().to_str().unwrap().replace('\\', "/")
)).unwrap();
    {
        let p = root.join("planner.inv");
        let source = fs::read_to_string(&p).unwrap();
        fs::write(&p,source.replace("input \"wire.h\";",&format!("input \"../../examples/cells/reach_place/{cell_name}\"; input \"../../examples/cells/reach_place/assets/menagerie/{model_name}/scene.xml\"; input \"wire.h\";"))).unwrap();
    }
    let pose = json!([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]);
    let mut golden = json!({"schema":"invar-planner-endpoint-goldens-v3","joint_count":n,"seed":123,
        "metrics":{"translation":"euclidean-m","orientation":"sign-invariant-geodesic-rad","translation_max_m":0.002,"orientation_max_rad":std::f64::consts::PI/180.0,"start_max_rad":1e-6},
        "joint_mapping":names.iter().map(|s|json!([s,s])).collect::<Vec<_>>(),
        // Every planner world phase; vector 1 is held. The fixture moves q0 to the
        // target x, so the home joint goal is the start with q0 = 0.1.
        "vectors":([0, 2, 1, 3, 4].iter().map(|phase| {
            let mut v = json!({"id":format!("phase-{phase}"),"phase":phase,"start_q_rad":vec![0.0;n],"target_base_from_tool":[0.1,0.0,0.0,0.0,0.0,0.0,1.0],"cube_base_pose":pose,"tool_from_cube":pose});
            if *phase == 4 {
                let mut goal = vec![0.0; n];
                goal[0] = 0.1;
                v["goal_q_rad"] = json!(goal);
            }
            v
        }).collect::<Vec<_>>())});
    {
        let p = &projection;
        golden["tool_projection"] = p.value();
        let raw_target = invar::system::tool_projection::Pose {
            translation_m: [0.1, 0.0, 0.0],
            quaternion_xyzw: [0.0, 0.0, 0.0, 1.0],
        };
        let target = p.fk_to_source(raw_target).unwrap();
        let flat = invar::system::planner_scene::flat;
        // The fixture FK puts the tool at (q0, 0, 0); both vectors start at q = 0.
        let start_tool = p
            .fk_to_source(invar::system::tool_projection::Pose {
                translation_m: [0.0; 3],
                quaternion_xyzw: [0.0, 0.0, 0.0, 1.0],
            })
            .unwrap();
        let held = start_tool.compose(scene.tool_from_cube).unwrap();
        for v in golden["vectors"].as_array_mut().unwrap() {
            let cube = if v["phase"] == 2 || v["phase"] == 3 {
                held
            } else {
                scene.cube_base_pose
            };
            v["cube_base_pose"] = json!(flat(cube));
            v["tool_from_cube"] = json!(flat(scene.tool_from_cube));
            v["target_base_from_tool"] = json!([
                target.translation_m[0],
                target.translation_m[1],
                target.translation_m[2],
                target.quaternion_xyzw[0],
                target.quaternion_xyzw[1],
                target.quaternion_xyzw[2],
                target.quaternion_xyzw[3]
            ]);
        }
    }
    write(&root.join("endpoint.json"), &golden);
    let output = root.join("output");
    let mut first_diagnostic = None;
    for fault in 0..5 {
        fs::write(root.join("planner.c"),format!(r#"
#include "wire.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>
const char *fixture_planner_version(void) {{ return "invar-planner-component-v2/{n}/cpu-fixture"; }}
int fixture_planner_init(void **context,const uint8_t *child,uint32_t size) {{ (void)child;(void)size;*context=calloc(1,1);return *context?0:3; }}
int fixture_planner_configure(void *context,const void *config) {{ (void)config;return context?0:2; }}
int fixture_planner_step(void *context,const void *input,void *output) {{
 if(!context) return 2;
 const inv_planner_request{n}_f64_v2 *r=input;inv_planner_trajectory{n}_f64_v1 *t=output;
 memset(t,0,sizeof(*t));t->dt_s=0.1;t->count=2;t->waypoints[1].time_from_start_s=0.1;
 for(unsigned i=0;i<{n};++i) t->waypoints[0].q_rad[i]=t->waypoints[1].q_rad[i]=r->start_q_rad[i];
 t->waypoints[1].q_rad[0]=r->target_base_from_tool.xyz_m[0];
 if({fault}==1) t->waypoints[1].q_rad[0]+=0.003;
 if({fault}==2) t->waypoints[0].q_rad[{n}-1]+=0.000002;
 if({fault}==3) t->waypoints[1].dq_rad_s[{n}-1]=NAN;
 if({fault}==4) t->waypoints[1].time_from_start_s=0;
 return 0;
}}
int fixture_planner_shutdown(void **context) {{ free(*context);*context=NULL;return 0; }}
"#)).unwrap();
        let library = compile(&root, "planner.c", "planner");
        let bindings = [
            ("library", library.clone()),
            ("child", root.join("child")),
            ("model", root.join("model.urdf")),
            ("robot_config", root.join("robot.yml")),
            ("scene", root.join("scene.yml")),
            ("fk_library", fk_library.clone()),
            ("fk_model", root.join("model.xml")),
            ("fk_manifest", root.join("fk.inv")),
            ("fk_evidence", fk_evidence.clone()),
        ];
        let mut runtime = json!({"schema":"invar-planner-runtime-v3","qualification":"cpu-diagnostic",
            "artifacts":bindings.iter().map(|(r,p)|(*r,json!({"path":p,"sha256":hash(p)}))).collect::<std::collections::BTreeMap<_,_>>(),
            "config":{"joint_names":names,"joint_limits_rad":vec![[-2.0,2.0];n],"velocity_limits_rad_s":vec![2.0;n],"table_pose":pose,"table_half_extents_m":[0.5,0.5,0.05],"cube_edge_m":0.05,"base_frame":"base","tool_frame":"tool"}});
        {
            let p = &projection;
            runtime["tool_projection"] = p.value();
            runtime["cell_source"] = json!({"path":cell_path,"sha256":hash(&cell_path)});
            runtime["config"]["tool_frame"] = json!(p.planner_reference_frame);
            runtime["config"]["table_pose"] =
                json!(invar::system::planner_scene::flat(scene.table_pose));
            runtime["config"]["table_half_extents_m"] = json!(scene.table_half_extents_m);
            runtime["config"]["mount_exclusion"] = scene.mount_exclusion.clone();
        }
        write(&root.join("runtime.json"), &runtime);
        let args = [
            Path::new("component"),
            Path::new("test"),
            &root.join("planner.inv"),
            Path::new("--library"),
            &library,
            Path::new("--model"),
            &root.join("model.urdf"),
            Path::new("--child"),
            &root.join("child"),
            Path::new("--planner-runtime"),
            &root.join("runtime.json"),
            Path::new("--output"),
            &output,
        ];
        let result = cli(&args);
        if fault == 0 {
            crate::child_deadline::assert_success(&result);
            let report: Value = serde_json::from_slice(&result.stdout).unwrap();
            assert_eq!(report["status"], "diagnostic");
            let path = PathBuf::from(report["evidence"].as_str().unwrap());
            let record: Value = serde_json::from_slice(&fs::read(&path).unwrap()).unwrap();
            assert_eq!(record["schema"], "invar-planner-diagnostic-v3");
            let mut relabelled = record.clone();
            relabelled["schema"] = json!("invar-planner-contract-evidence-v3");
            relabelled["status"] = json!("pass");
            let forgery = root.join("relabelled-diagnostic.json");
            write(&forgery, &relabelled);
            assert!(
                invar::system::foreign::verify_evidence(
                    &forgery,
                    report["manifest_hash"].as_str().unwrap()
                )
                .is_err()
            );
            assert!(
                invar::system::foreign::verify_evidence(
                    &path,
                    report["manifest_hash"].as_str().unwrap()
                )
                .is_err()
            );
            assert_eq!(record["results"]["vectors"].as_array().unwrap().len(), 5);
            assert_eq!(record["results"]["vectors"][4]["goal_error_rad"], 0.0);
            assert_eq!(record["results"]["vectors"][0]["final_q_rad"][0], 0.1);
            first_diagnostic = Some((path.clone(), hash(&path)));
            {
                // The held cube must follow FK and the declared grasp, not a free pose.
                let pristine = golden.clone();
                let x = golden["vectors"][1]["cube_base_pose"][0].as_f64().unwrap();
                golden["vectors"][1]["cube_base_pose"][0] = json!(x + 1e-3);
                write(&root.join("endpoint.json"), &golden);
                let loose = cli(&args);
                assert!(!loose.status.success());
                assert!(
                    String::from_utf8_lossy(&loose.stderr)
                        .contains("attached cube is not held at the declared grasp")
                );
                golden = pristine.clone();
                golden["vectors"][0]["tool_from_cube"][0] = json!(1.0);
                write(&root.join("endpoint.json"), &golden);
                let regrasped = cli(&args);
                assert!(!regrasped.status.success());
                assert!(
                    String::from_utf8_lossy(&regrasped.stderr)
                        .contains("request cube differs from checked cell source")
                );
                golden = pristine;
                write(&root.join("endpoint.json"), &golden);
            }
            runtime["qualification"] = json!("production");
            write(&root.join("runtime.json"), &runtime);
            let forbidden = cli(&args);
            assert!(!forbidden.status.success());
            assert!(
                String::from_utf8_lossy(&forbidden.stderr).contains("substituted planner child")
            );
            runtime["qualification"] = json!("cpu-diagnostic");
            write(&root.join("runtime.json"), &runtime);
            runtime["artifacts"]["fk_library"]["sha256"] = json!("0".repeat(64));
            write(&root.join("runtime.json"), &runtime);
            let stale = cli(&args);
            assert!(!stale.status.success());
            assert!(String::from_utf8_lossy(&stale.stderr).contains("S1405"));
            runtime["artifacts"]["fk_library"]["sha256"] = json!(hash(&fk_library));
            write(&root.join("runtime.json"), &runtime);
            let half = (std::f64::consts::PI / 180.0) * 1.01 / 2.0;
            golden["vectors"][0]["target_base_from_tool"] =
                json!([0.1, 0.0, 0.0, 0.0, 0.0, half.sin(), half.cos()]);
            write(&root.join("endpoint.json"), &golden);
            golden["joint_mapping"][0] = json!(["joint0", "wrong_fk_joint"]);
            write(&root.join("endpoint.json"), &golden);
            let mapping = cli(&args);
            assert!(!mapping.status.success());
            assert!(String::from_utf8_lossy(&mapping.stderr).contains("joint mapping"));
            golden["joint_mapping"][0] = json!(["joint0", "joint0"]);
            write(&root.join("endpoint.json"), &golden);
            let original_manifest = fs::read_to_string(root.join("planner.inv")).unwrap();
            fs::write(
                root.join("planner.inv"),
                original_manifest.replace("input \"fk.c\";", ""),
            )
            .unwrap();
            let transitive = cli(&args);
            assert!(!transitive.status.success());
            assert!(
                String::from_utf8_lossy(&transitive.stderr).contains("complete FK input snapshot")
            );
            fs::write(root.join("planner.inv"), &original_manifest).unwrap();
            fs::write(
                root.join("fk-clash.inv"),
                fs::read_to_string(root.join("fk.inv"))
                    .unwrap()
                    .replace(
                        &format!("f64,{n}>"),
                        &format!("f64,{}>", if n == 6 { 7 } else { 6 }),
                    )
                    .replace(
                        &format!("inv_joint{n}_f64"),
                        &format!("inv_joint{}_f64", if n == 6 { 7 } else { 6 }),
                    ),
            )
            .unwrap();
            fs::write(
                root.join("planner.inv"),
                original_manifest.replace(
                    "input \"fk.inv\";",
                    "input \"fk.inv\";input \"fk-clash.inv\";",
                ),
            )
            .unwrap();
            let conflict = cli(&args);
            assert!(!conflict.status.success());
            assert!(String::from_utf8_lossy(&conflict.stderr).contains("ABI header conflict"));
            fs::write(root.join("planner.inv"), &original_manifest).unwrap();
            let rotation = cli(&args);
            assert!(!rotation.status.success());
            assert!(
                String::from_utf8_lossy(&rotation.stderr).contains("independent endpoint error")
            );
            golden["vectors"][0]["target_base_from_tool"] =
                json!([0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]);
            write(&root.join("endpoint.json"), &golden);
        } else {
            assert!(!result.status.success());
            assert!(String::from_utf8_lossy(&result.stderr).contains("S1403"));
        }
    }
    let (saved, digest) = first_diagnostic.unwrap();
    assert_eq!(hash(&saved), digest);
    let mut failures = 0;
    let mut endpoint_failures = 0;
    for entry in fs::read_dir(&output).unwrap() {
        let p = entry.unwrap().path();
        let record: Value = serde_json::from_slice(&fs::read(p).unwrap()).unwrap();
        assert_ne!(record["status"], "pass");
        if record["status"] == "error" {
            failures += 1;
            assert_eq!(record["schema"], "invar-planner-contract-attempt-v1");
            for key in ["entry_manifest_sha256", "entry_runtime_sha256"] {
                assert_eq!(record[key].as_str().unwrap().len(), 64);
            }
            if record["message"]
                .as_str()
                .unwrap()
                .contains("independent endpoint error")
            {
                endpoint_failures += 1;
                assert_eq!(
                    record["observations"]["final_q_rad"]
                        .as_array()
                        .unwrap()
                        .len(),
                    n
                );
                assert_eq!(
                    record["observations"]["actual_translation_m"]
                        .as_array()
                        .unwrap()
                        .len(),
                    3
                );
                assert_eq!(
                    record["observations"]["actual_quaternion_xyzw"]
                        .as_array()
                        .unwrap()
                        .len(),
                    4
                );
            }
        }
    }
    assert_eq!(failures, 12);
    assert_eq!(endpoint_failures, 2);
    fs::remove_dir_all(root).unwrap();
}

#[test]
fn issue_s1_06_planner_contract_cpu_six_joint_faults_and_no_pass() {
    exercise(6);
}
#[test]
fn issue_s1_06_planner_contract_cpu_seven_joint_faults_and_no_pass() {
    exercise(7);
}

#[test]
fn issue_s1_06_planner_contract_preparation_binds_source_requests_and_refuses_unbound_runtime() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let directory = root.join("examples/cells/reach_place/components/curobo_planner");
    let source_path = root.join("tests/fixtures/foreign-components/pinocchio_goldens.json");
    let source: Value = serde_json::from_slice(&fs::read(&source_path).unwrap()).unwrap();
    for (arm, n, manifest_name) in [
        ("panda", 7, "manifest-source-panda-v3.inv"),
        ("ur5e", 6, "manifest-source-ur5e-v3.inv"),
    ] {
        let manifest = invar::system::compile_manifest(
            invar::front::parse_system(&fs::read(directory.join(manifest_name)).unwrap()).unwrap(),
        )
        .unwrap();
        let golden: Value = serde_json::from_slice(
            &fs::read(directory.join(format!("endpoint-source-{arm}-v3.json"))).unwrap(),
        )
        .unwrap();
        let runtime: Value = serde_json::from_slice(
            &fs::read(directory.join(format!("runtime-source-{arm}-v3.json"))).unwrap(),
        )
        .unwrap();
        let reference = source["arms"]
            .as_array()
            .unwrap()
            .iter()
            .find(|a| a["arm"] == arm)
            .unwrap();
        assert_eq!(golden["source_reference"]["sha256"], hash(&source_path));
        assert_eq!(golden["joint_count"], n);
        assert_eq!(golden["seed"], 123);
        assert_eq!(
            golden["qualification"],
            "unexecuted_fixed_requests_not_task_acceptance"
        );
        assert_eq!(manifest["golden"][0]["tolerances"][0]["value"], "0.002");
        assert_eq!(
            manifest["golden"][0]["tolerances"][1]["value"],
            "0.017453292519943295"
        );
        assert_eq!(
            runtime["artifacts"]["child"]["sha256"],
            hash(&directory.join("planner_server.py"))
        );
        let fk_model = directory.join(runtime["artifacts"]["fk_model"]["path"].as_str().unwrap());
        let fk_manifest = directory.join(
            runtime["artifacts"]["fk_manifest"]["path"]
                .as_str()
                .unwrap(),
        );
        assert_eq!(runtime["artifacts"]["fk_model"]["sha256"], hash(&fk_model));
        assert_eq!(
            runtime["artifacts"]["fk_manifest"]["sha256"],
            hash(&fk_manifest)
        );
        assert_eq!(
            runtime["artifacts"]["fk_model"]["sha256"],
            reference["model_sha256"]
        );
        for role in [
            "library",
            "fk_library",
            "robot_config",
            "scene",
            "fk_evidence",
        ] {
            assert!(runtime["artifacts"][role]["sha256"].is_null());
            assert!(
                !directory
                    .join(runtime["artifacts"][role]["path"].as_str().unwrap())
                    .exists()
            );
        }
        assert_eq!(
            runtime["config"]["joint_names"].as_array().unwrap().len(),
            n
        );
        assert_eq!(golden["joint_mapping"].as_array().unwrap().len(), n);
        let output = std::env::temp_dir().join(format!(
            "invar-unbound-runtime-{}-{arm}",
            std::process::id()
        ));
        let rejected = cli(&[
            Path::new("component"),
            Path::new("test"),
            &directory.join(manifest_name),
            Path::new("--planner-runtime"),
            &directory.join(format!("runtime-source-{arm}-v3.json")),
            Path::new("--output"),
            &output,
        ]);
        assert!(!rejected.status.success());
        let entries: Vec<_> = fs::read_dir(&output)
            .unwrap()
            .map(|e| e.unwrap().path())
            .collect();
        assert_eq!(entries.len(), 1);
        let attempt: Value = serde_json::from_slice(&fs::read(&entries[0]).unwrap()).unwrap();
        assert_eq!(attempt["schema"], "invar-planner-contract-attempt-v1");
        assert_eq!(attempt["status"], "error");
        fs::remove_dir_all(output).unwrap();
        for i in 0..n {
            assert_eq!(
                golden["joint_mapping"][i],
                json!([
                    runtime["config"]["joint_names"][i],
                    reference["joint_names"][i]
                ])
            );
        }
        let cell_path = root.join("examples/cells/reach_place").join(if n == 7 {
            "cell.inv"
        } else {
            "cell-ur5e.inv"
        });
        let ir = invar::system::compile_at(
            invar::front::parse_system(&fs::read(&cell_path).unwrap()).unwrap(),
            cell_path.parent().unwrap(),
        )
        .unwrap();
        // The vectors themselves are regenerated in issue_s1_06_planner_requests.
        let projection = invar::system::tool_projection::selected(&ir).unwrap();
        assert_eq!(runtime["tool_projection"], projection.value());
        assert_eq!(golden["tool_projection"], projection.value());
        assert_eq!(golden["vectors"].as_array().unwrap().len(), 10);
    }
}
