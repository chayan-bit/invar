//! Actual generated binding / private Rust process. CPU doubles, no ROS/cuRobo claim.
use super::sequencer_binding_support as support;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{
    fs,
    io::{Read, Write},
    path::{Path, PathBuf},
    process::{Child, Command, Stdio},
};
fn hash(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}
fn pose(v: [f64; 3], q: [f64; 4]) -> Vec<f64> {
    v.into_iter().chain(q).collect()
}
fn transform(v: &Value) -> invar::system::tool_projection::Pose {
    invar::system::tool_projection::Pose {
        translation_m: std::array::from_fn(|i| v["translation_m"][i].as_f64().unwrap()),
        quaternion_xyzw: std::array::from_fn(|i| v["quaternion_xyzw"][i].as_f64().unwrap()),
    }
}
/// The FK stand-in's fixed raw orientation: the one that projects onto the binding's grasp.
fn fk_rotation(binding: &Value) -> [f64; 4] {
    let p = transform(&binding["tool_projection"]["reference_from_cell"]);
    let b = transform(&binding["tool_projection"]["source_base_from_reference_base"]);
    let goal = &binding["grasp_tool_base"];
    let goal = invar::system::tool_projection::Pose {
        translation_m: std::array::from_fn(|i| goal[i].as_f64().unwrap()),
        quaternion_xyzw: std::array::from_fn(|i| goal[3 + i].as_f64().unwrap()),
    };
    let raw = b.inverse().unwrap().compose(goal).unwrap();
    raw.compose(p.inverse().unwrap()).unwrap().quaternion_xyzw
}
fn projected(binding: &Value, q: &[f64]) -> Vec<f64> {
    let p = transform(&binding["tool_projection"]["reference_from_cell"]);
    let b = transform(&binding["tool_projection"]["source_base_from_reference_base"]);
    let raw = invar::system::tool_projection::Pose {
        translation_m: [q[0], q[1], q[2]],
        quaternion_xyzw: fk_rotation(binding),
    };
    let result = b.compose(raw).unwrap().compose(p).unwrap();
    pose(result.translation_m, result.quaternion_xyzw)
}
fn u32(b: &mut Vec<u8>, v: u32) {
    b.extend(v.to_le_bytes());
}
fn u64(b: &mut Vec<u8>, v: u64) {
    b.extend(v.to_le_bytes());
}
fn f64(b: &mut Vec<u8>, v: f64) {
    b.extend(v.to_le_bytes());
}
fn digest(hex: &str) -> Vec<u8> {
    (0..32)
        .map(|i| u8::from_str_radix(&hex[2 * i..2 * i + 2], 16).unwrap())
        .collect()
}
struct Session {
    child: Child,
    id: u32,
}
impl Session {
    fn call(&mut self, kind: u16, b: &[u8]) -> (u16, Vec<u8>) {
        self.id += 1;
        let mut h = b"IVSQ".to_vec();
        h.extend(2u16.to_le_bytes());
        h.extend(kind.to_le_bytes());
        u32(&mut h, b.len() as u32);
        u32(&mut h, self.id);
        let input = self.child.stdin.as_mut().unwrap();
        input.write_all(&h).unwrap();
        input.write_all(b).unwrap();
        input.flush().unwrap();
        let mut header = [0; 16];
        self.child
            .stdout
            .as_mut()
            .unwrap()
            .read_exact(&mut header)
            .unwrap();
        assert_eq!(&header[..6], b"IVSQ\x02\x00");
        assert_eq!(
            u32::from_le_bytes(header[12..16].try_into().unwrap()),
            self.id
        );
        let size = u32::from_le_bytes(header[8..12].try_into().unwrap()) as usize;
        assert!(size <= 65536);
        let mut data = vec![0; size];
        self.child
            .stdout
            .as_mut()
            .unwrap()
            .read_exact(&mut data)
            .unwrap();
        (u16::from_le_bytes(header[6..8].try_into().unwrap()), data)
    }
}
impl Drop for Session {
    fn drop(&mut self) {
        if self.child.try_wait().unwrap().is_none() {
            self.child.kill().unwrap();
            self.child.wait().unwrap();
        }
    }
}
struct Fixture {
    root: PathBuf,
    binding: Value,
    task: Value,
    files: std::collections::BTreeMap<String, String>,
}
impl Fixture {
    fn generated_callbacks(&self, n: usize) {
        self.launch_exit_callbacks();
        fs::write(
            self.root.join("stub.hpp"),
            include_str!("../fixtures/s1_06/sequencer_ros_stubs.hpp")
                .replace("@JOINTS@", &n.to_string()),
        )
        .unwrap();
        for name in [
            "rclcpp/rclcpp.hpp".into(),
            "rcl_interfaces/msg/parameter_descriptor.hpp".into(),
            "geometry_msgs/msg/pose.hpp".into(),
            "std_srvs/srv/set_bool.hpp".into(),
            "std_msgs/msg/string.hpp".into(),
            format!("invar_cell_interfaces/msg/cell_snapshot{n}_v1.hpp"),
            format!("invar_cell_interfaces/msg/planner_request{n}_v2.hpp"),
            format!("invar_cell_interfaces/msg/planner_trajectory{n}_v1.hpp"),
        ] {
            let path = self.root.join(name);
            fs::create_dir_all(path.parent().unwrap()).unwrap();
            fs::write(path, "#include \"stub.hpp\"\n").unwrap();
        }
        fs::write(
            self.root.join("node.cpp"),
            &self.files["src/invar_sequencer_task/src/main.cpp"],
        )
        .unwrap();
        fs::write(
            self.root.join("process.hpp"),
            &self.files["src/invar_sequencer_task/include/process.hpp"],
        )
        .unwrap();
        let values = |v: &Value| {
            v.as_array()
                .unwrap()
                .iter()
                .map(|x| format!("{:.17e}", x.as_f64().unwrap()))
                .collect::<Vec<_>>()
                .join(",")
        };
        let bytes = |key| {
            digest(self.binding[key].as_str().unwrap())
                .iter()
                .map(u8::to_string)
                .collect::<Vec<_>>()
                .join(",")
        };
        fs::write(
            self.root.join("projection.hpp"),
            &self.files["src/invar_fk_node/include/invar_fk_node/tool_projection.hpp"],
        )
        .unwrap();
        let projection = &self.binding["tool_projection"]["reference_from_cell"];
        let data = format!(
            "#pragma once\n#include <array>\n#include <cstdint>\ninline const std::array<double,{n}> home{{{}}};\ninline const std::array<double,3> offset{{{}}};\ninline const std::array<double,4> rotation{{{}}};\ninline const std::array<double,7> final_cube{{{}}};\ninline const std::array<std::uint8_t,32> source_hash{{{}}},order_hash{{{}}};\ninline std::uint64_t step_index=0;\n",
            values(&self.binding["home_q_rad"]),
            values(&projection["translation_m"]),
            values(&projection["quaternion_xyzw"]),
            values(&self.binding["final_cube_base"]),
            bytes("source_ir_sha256"),
            bytes("joint_order_sha256")
        );
        let rotation = fk_rotation(&self.binding)
            .map(|x| format!("{x:.17e}"))
            .join(",");
        let data = format!("{data}inline const std::array<double,4> fk_rotation{{{rotation}}};\n");
        fs::write(self.root.join("probe_data.hpp"), data).unwrap();
        let names = self.binding["robot_joints"].as_array().unwrap();
        let indices = self.binding["joint_names"]
            .as_array()
            .unwrap()
            .iter()
            .map(|name| {
                names
                    .iter()
                    .position(|j| j["name"] == *name)
                    .unwrap()
                    .to_string()
            })
            .collect::<Vec<_>>()
            .join(",");
        use std::io::Write as _;
        writeln!(fs::OpenOptions::new().append(true).open(self.root.join("probe_data.hpp")).unwrap(),"inline const std::array<double,{}> robot_home{{{}}};\ninline const std::array<std::uint8_t,32> robot_order_hash{{{}}};\ninline const std::array<std::size_t,{n}> arm_indices{{{indices}}};",names.len(),values(&self.binding["home_robot_q_si"]),bytes("robot_joint_order_sha256")).unwrap();
        fs::write(
            self.root.join("probe.cpp"),
            include_str!("../fixtures/s1_06/sequencer_node_probe.cpp.in")
                .replace("@JOINTS@", &n.to_string()),
        )
        .unwrap();
        let output = Command::new("c++")
            .args([
                "-std=c++17",
                "-Wall",
                "-Wextra",
                "-Werror",
                "-pedantic",
                "-pthread",
                "-I",
            ])
            .arg(&self.root)
            .arg(self.root.join("probe.cpp"))
            .arg("-o")
            .arg(self.root.join("probe"))
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "generated callback compile: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            for (mode, content) in [
                ("stalled_process", "#!/bin/sh\nexec sleep 20\n"),
                ("partial_process", "#!/bin/sh\nprintf 'IVSQ'\n"),
            ] {
                let path = self.root.join(format!("binding.json.{mode}"));
                fs::write(&path, content).unwrap();
                fs::set_permissions(path, fs::Permissions::from_mode(0o700)).unwrap();
            }
        }
        // These execute the generated callbacks and private process transport,
        // using CPU ROS/FK stand-ins, including every point of 256-point plans.
        for mode in [
            "valid",
            "endpoint",
            "late_invalid",
            "late_velocity",
            "nonfinite",
            "start",
            "collapsed_time",
            "capacity",
            "stale_source",
            "stale_order",
            "same_state",
            "robot_limit",
            "robot_velocity",
            "robot_conflict",
            "dead_process",
            "stalled_process",
            "partial_process",
            "main_failure",
            "main_shutdown",
            "contact",
            "service",
            "not_ready",
        ] {
            let output = Command::new(self.root.join("probe"))
                .arg(env!("CARGO_BIN_EXE_invar-sequencer"))
                .arg(self.root.join("binding.json"))
                .arg(self.root.join("task.json"))
                .arg(hash(self.task.to_string().as_bytes()))
                .arg(mode)
                .output()
                .unwrap();
            assert_eq!(
                output.status.code(),
                Some(if matches!(mode, "valid" | "not_ready") {
                    0
                } else {
                    1
                }),
                "{n} {mode}: {}",
                String::from_utf8_lossy(&output.stderr)
            );
            let stdout = String::from_utf8_lossy(&output.stdout);
            assert_eq!(
                stdout.contains("\"task_complete\":true"),
                matches!(mode, "valid" | "not_ready"),
                "{stdout}"
            );
        }
    }
    fn launch_exit_callbacks(&self) {
        let launch = self.root.join("launch.py");
        fs::write(
            &launch,
            &self.files["src/invar_cell_bringup/launch/cell.launch.py"],
        )
        .unwrap();
        let script = r#"
import ast,sys,types
tree=ast.parse(open(sys.argv[1]).read())
callback=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='required_exit')
task=object();recorder=object()
namespace={'task_action':task,'EmitEvent':lambda **kw:kw,'Shutdown':lambda **kw:kw}
exec(compile(ast.Module(body=[callback],type_ignores=[]),sys.argv[1],'exec'),namespace)
call=namespace['required_exit']
context=types.SimpleNamespace(is_shutdown=False)
assert len(call(types.SimpleNamespace(action=task,returncode=0),context))==1
for action,code in [(task,1),(task,-9),(recorder,0),(recorder,1)]:
    try: call(types.SimpleNamespace(action=action,returncode=code),context)
    except RuntimeError as e: assert 'S1406' in str(e)
    else: raise AssertionError('required process failure was accepted')
context.is_shutdown=True
for action in [task,recorder]: assert call(types.SimpleNamespace(action=action,returncode=-2),context)==[]
"#;
        let output = Command::new("python3")
            .args(["-c", script])
            .arg(launch)
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
    }
    fn new(n: usize, label: &str) -> Self {
        Self::with_source(n, label, &support::source(n))
    }
    fn with_source(n: usize, label: &str, source: &str) -> Self {
        let files = support::generate(source).unwrap();
        let binding: Value =
            serde_json::from_str(&files["src/invar_cell_bringup/config/sequencer-binding.json"])
                .unwrap();
        let root = std::env::temp_dir().join(format!(
            "invar-bound-task-{}-{n}-{label}",
            std::process::id()
        ));
        fs::create_dir(&root).unwrap();
        fs::write(
            root.join("binding.json"),
            &files["src/invar_cell_bringup/config/sequencer-binding.json"],
        )
        .unwrap();
        let rotation = fk_rotation(&binding).map(|x| format!("{x:.17e}"));
        fs::write(root.join("fk.c"),r#"
/* CPU protocol fixture; not Pinocchio or physical FK qualification. */
#include <stdlib.h>
const char *invar_pinocchio_bridge_v1_version(void){return "invar-component-v1/invar-pinocchio-reference-abi-1/pinocchio-4.1.0";}
int invar_pinocchio_bridge_v1_init(void **c,const unsigned char *p,unsigned n){(void)p;(void)n;*c=calloc(1,1);return *c?0:3;}
int invar_pinocchio_bridge_v1_configure(void *c,const void *v){(void)v;return c?0:2;}
int invar_pinocchio_bridge_v1_step(void *c,const void *v,void *out){if(!c)return 2;const double *q=v;double *p=out;p[0]=q[0];p[1]=q[1];p[2]=q[2];p[3]=@QX@;p[4]=@QY@;p[5]=@QZ@;p[6]=@QW@;return 0;}
int invar_pinocchio_bridge_v1_shutdown(void **c){free(*c);*c=0;return 0;}
"#.replace("@QX@", &rotation[0]).replace("@QY@", &rotation[1]).replace("@QZ@", &rotation[2]).replace("@QW@", &rotation[3])).unwrap();
        let lib = root.join(format!("fk.{}", std::env::consts::DLL_EXTENSION));
        let output = Command::new("cc")
            .args([
                "-std=c11",
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
            .arg(root.join("fk.c"))
            .arg("-o")
            .arg(&lib)
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        let model = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("examples/cells/reach_place/assets/menagerie")
            .join(if n == 7 {
                "franka_emika_panda/panda.xml"
            } else {
                "universal_robots_ur5e/ur5e.xml"
            });
        let task = json!({"schema":"invar-sequencer-task-v1","source_ir_sha256":binding["source_ir_sha256"],"fk_model_path":model,"fk_library_path":lib,"fk_library_sha256":hash(&fs::read(&lib).unwrap()),"qualification":"CPU_double_only"});
        fs::write(root.join("task.json"), task.to_string()).unwrap();
        Self {
            root,
            binding,
            task,
            files,
        }
    }
    fn session(&self) -> Session {
        let child = Command::new(env!("CARGO_BIN_EXE_invar-sequencer"))
            .args(["--binding"])
            .arg(self.root.join("binding.json"))
            .args([
                "--binding-sha256",
                &hash(&fs::read(self.root.join("binding.json")).unwrap()),
                "--task",
            ])
            .arg(self.root.join("task.json"))
            .args([
                "--task-sha256",
                &hash(&fs::read(self.root.join("task.json")).unwrap()),
            ])
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .unwrap();
        Session { child, id: 0 }
    }
    fn telemetry(
        &self,
        ns: u64,
        step: u64,
        q: &[f64],
        cube: &[f64],
        attached: bool,
        contacts: &[i32],
    ) -> Vec<u8> {
        let mut b = Vec::new();
        u64(&mut b, ns);
        u64(&mut b, step);
        b.extend(digest(self.binding["source_ir_sha256"].as_str().unwrap()));
        b.extend(digest(self.binding["joint_order_sha256"].as_str().unwrap()));
        for x in q {
            f64(&mut b, *x);
        }
        for _ in q {
            f64(&mut b, 0.0);
        }
        for x in projected(&self.binding, q) {
            f64(&mut b, x);
        }
        for x in cube {
            f64(&mut b, *x);
        }
        u32(&mut b, u32::from(attached));
        u32(&mut b, 4);
        b.extend([0, 1, 2, 3]);
        u32(&mut b, contacts.len() as u32);
        for id in contacts {
            b.extend(id.to_le_bytes());
        }
        b.extend(digest(
            self.binding["robot_joint_order_sha256"].as_str().unwrap(),
        ));
        let names = self.binding["robot_joints"].as_array().unwrap();
        u32(&mut b, names.len() as u32);
        for (i, joint) in names.iter().enumerate() {
            let value = self.binding["joint_names"]
                .as_array()
                .unwrap()
                .iter()
                .position(|name| *name == joint["name"])
                .map_or(
                    self.binding["home_robot_q_si"][i].as_f64().unwrap(),
                    |arm| q[arm],
                );
            f64(&mut b, value);
        }
        for _ in names {
            f64(&mut b, 0.0);
        }
        b
    }
    fn failed(&self, kind: u16, body: &[u8]) {
        let mut s = self.session();
        s.id = 1;
        let mut h = b"IVSQ\x02\x00".to_vec();
        h.extend(kind.to_le_bytes());
        u32(&mut h, body.len() as u32);
        u32(&mut h, 1);
        s.child
            .stdin
            .take()
            .unwrap()
            .write_all(&[h, body.to_vec()].concat())
            .unwrap();
        let status = s.child.wait().unwrap();
        let mut stdout = Vec::new();
        s.child
            .stdout
            .take()
            .unwrap()
            .read_to_end(&mut stdout)
            .unwrap();
        let mut stderr = Vec::new();
        s.child
            .stderr
            .take()
            .unwrap()
            .read_to_end(&mut stderr)
            .unwrap();
        assert!(!status.success());
        assert!(stdout.is_empty(), "no partial reply");
        let text = String::from_utf8_lossy(&stderr);
        assert!(text.contains("\"success\":false"), "{text}");
    }
}
impl Drop for Fixture {
    fn drop(&mut self) {
        fs::remove_dir_all(&self.root).unwrap();
    }
}

#[test]
fn issue_s1_06_binding_unannotated_cube_targets_its_centre() {
    for n in [6, 7] {
        let source = support::source(n).replace(support::GRASP, "");
        let fixture = Fixture::with_source(n, "centre-grasp", &source);
        let binding = &fixture.binding;
        let q = binding["home_q_rad"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap())
            .collect::<Vec<_>>();
        let cube = [0.3, 0.0, 0.025, 0.0, 0.0, 0.0, 1.0];
        let mut session = fixture.session();
        let (kind, payload) = session.call(1, &fixture.telemetry(0, 1, &q, &cube, false, &[]));
        assert_eq!(kind, 11);
        let (_, bytes) = split_label(&payload);
        for i in 0..7 {
            let value = f64::from_le_bytes(
                bytes[16 * n + i * 8..16 * n + (i + 1) * 8]
                    .try_into()
                    .unwrap(),
            );
            assert!((value - binding["pregrasp_tool_base"][i].as_f64().unwrap()).abs() < 1e-12);
        }
    }
}

/// IVSQ/2 snapshot replies start with the task phase label (u8 length, ASCII).
fn split_label(payload: &[u8]) -> (String, &[u8]) {
    let n = usize::from(payload[0]);
    (
        String::from_utf8(payload[1..1 + n].to_vec()).unwrap(),
        &payload[1 + n..],
    )
}

fn flow(n: usize) {
    let f = Fixture::new(n, "flow");
    f.generated_callbacks(n);
    let mut s = f.session();
    let home = f.binding["home_q_rad"]
        .as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_f64().unwrap())
        .collect::<Vec<_>>();
    let mut q = home.clone();
    let mut cube = vec![0.3, 0.0, 0.025, 0.0, 0.0, 0.0, 1.0];
    let final_cube = f.binding["final_cube_base"]
        .as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_f64().unwrap())
        .collect::<Vec<_>>();
    let mut step = 0u64;
    let mut snapshot = |s: &mut Session, ns: u64, q: &[f64], cube: &[f64], attached: bool| {
        step += 1;
        let (kind, payload) = s.call(1, &f.telemetry(ns, step, q, cube, attached, &[]));
        let (label, body) = split_label(&payload);
        (kind, label, body.to_vec())
    };
    const S: u64 = 1_000_000_000;
    // (move, planner world phase, request time, held while planning)
    for (stage, phase, ns, held) in [
        ("pregrasp", 0, 0, false),
        ("descend", 1, 5 * S, false),
        ("lift", 3, 10 * S, true),
        ("transfer", 2, 15 * S, true),
        ("place_descend", 3, 20 * S, true),
        ("retreat", 1, 25 * S, false),
        ("home", 4, 30 * S, false),
    ] {
        let (kind, label, r) = snapshot(&mut s, ns, &q, &cube, held);
        assert_eq!(kind, 11, "{stage}");
        assert!(label.starts_with(&format!("{stage}/planning/")), "{label}");
        assert_eq!(r.len(), 16 * n + 172);
        let value = |i: usize| f64::from_le_bytes(r[8 * i..8 * i + 8].try_into().unwrap());
        let goal_q = (n..2 * n).map(value).collect::<Vec<_>>();
        assert_eq!(
            u32::from_le_bytes(r[16 * n + 168..].try_into().unwrap()),
            phase
        );
        let target = (0..7).map(|i| value(2 * n + i)).collect::<Vec<_>>();
        let target = invar::system::tool_projection::Pose {
            translation_m: target[..3].try_into().unwrap(),
            quaternion_xyzw: target[3..].try_into().unwrap(),
        };
        let raw = transform(&f.binding["tool_projection"]["source_base_from_reference_base"])
            .inverse()
            .unwrap()
            .compose(target)
            .unwrap()
            .compose(
                transform(&f.binding["tool_projection"]["reference_from_cell"])
                    .inverse()
                    .unwrap(),
            )
            .unwrap();
        let mut end = q.clone();
        if phase == 4 {
            // Home is a joint goal: the exact home joints.
            assert_eq!(goal_q, home);
            end = goal_q;
        } else {
            assert_eq!(goal_q, q);
            end[..3].copy_from_slice(&raw.translation_m);
        }
        let mut plan = Vec::new();
        u64(&mut plan, ns);
        f64(&mut plan, 1.0);
        u32(&mut plan, 2);
        for (time, state) in [(0.0, &q), (1.0, &end)] {
            f64(&mut plan, time);
            for x in state {
                f64(&mut plan, *x);
            }
            for _ in state {
                f64(&mut plan, 0.0);
            }
        }
        let (kind, accepted) = s.call(2, &plan);
        assert_eq!(kind, 21);
        assert_eq!(accepted, plan);
        q = end;
        if stage == "place_descend" {
            cube = final_cube.clone();
        }
        let arrived = ns + S;
        let (kind, label, payload) = snapshot(&mut s, arrived, &q, &cube, held);
        match stage {
            "descend" | "place_descend" => {
                let attach = stage == "descend";
                assert_eq!(kind, 12);
                assert_eq!(label, if attach { "attach" } else { "release" });
                assert_eq!(payload, [u8::from(attach)]);
                let mut ack = Vec::new();
                u64(&mut ack, arrived);
                u32(&mut ack, 1);
                assert_eq!(s.call(3, &ack), (0, vec![]));
                let (kind, label, payload) =
                    snapshot(&mut s, arrived + 10_000_000, &q, &cube, attach);
                assert_eq!((kind, payload.len()), (0, 0));
                if attach {
                    // The held cube must hold still for the settle window first.
                    assert_eq!(label, "settle");
                    let (kind, label, _) = snapshot(&mut s, arrived + 210_000_000, &q, &cube, true);
                    assert_eq!(
                        (kind, label.as_str()),
                        (0, "lift/need_plan/world=held_contact")
                    );
                } else {
                    assert_eq!(label, "retreat/need_plan/world=approach");
                }
            }
            "home" => {
                assert_eq!((kind, label.as_str()), (13, "complete"));
                assert!(payload.is_empty());
            }
            _ => {
                assert_eq!(kind, 0);
                assert!(label.contains("/need_plan/"), "{label}");
                assert!(payload.is_empty());
            }
        }
    }
    assert_eq!(q, home);
    s.child.stdin.take();
    assert!(s.child.wait().unwrap().success());
}
#[test]
fn issue_s1_06_binding_six_joint_real_process_ordering() {
    flow(6);
}
#[test]
fn issue_s1_06_binding_seven_joint_real_process_ordering() {
    flow(7);
}
#[test]
fn issue_s1_06_binding_source_graph_and_protocol_faults() {
    for n in [6, 7] {
        let f = Fixture::new(n, "faults");
        let home = f.binding["home_q_rad"]
            .as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_f64().unwrap())
            .collect::<Vec<_>>();
        let cube = [0.3, 0.0, 0.025, 0.0, 0.0, 0.0, 1.0];
        let good = f.telemetry(0, 1, &home, &cube, false, &[]);
        let mut stale = good.clone();
        stale[16] ^= 1;
        f.failed(1, &stale);
        f.failed(1, &f.telemetry(0, 1, &home, &cube, false, &[1, 2]));
        f.failed(1, &f.telemetry(0, 1, &home, &cube, false, &[-1, 2]));
        let mut bad = good.clone();
        bad.extend([0]);
        f.failed(1, &bad);
        f.failed(99, &[]);
        f.failed(2, &[]);
        for src in [
            support::source(n).replace("s1_06_reach_place_v1", "s1_06_reach_place_v2"),
            support::source(n).replace("memory = 256 MiB", "memory = 1 MiB"),
            support::source(n).replace("port out approved:", "port out bypass:"),
            support::source(n).replace(
                "measured_cube: std::robotics::Pose<f64,world,cube> @ 100 Hz",
                "measured_cube: std::robotics::Pose<f64,world,cube> @ 10 Hz",
            ),
            support::source(n)
                .replace("-> sequencer.planned", "-> controller.trajectory")
                .replace(
                    "connect sequencer.approved -> controller.trajectory policy queue(1);",
                    "",
                ),
        ] {
            assert!(support::generate(&src).is_err());
        }
        let launch = &f.files["src/invar_cell_bringup/launch/cell.launch.py"];
        assert!(launch.contains("recording_action"));
        // launch's ExecuteProcess takes substitutions, not floats: a float fails launch loading.
        assert!(launch.contains("sigterm_timeout='10.0', sigkill_timeout='5.0'"));
        assert!(
            f.files[&format!("src/invar_cell_interfaces/msg/CellSnapshot{n}V1.msg")]
                .contains("int32[<=2048]")
        );
    }
}

#[test]
#[ignore = "requires provisioned MuJoCo 3.12.0 CPU C headers/library; no ROS qualification"]
fn issue_s1_06_snapshot_mujoco_cpu_env() {
    let include =
        std::env::var("INVAR_MUJOCO_INCLUDE").expect("set exact MuJoCo include directory");
    let library = std::env::var("INVAR_MUJOCO_LIBRARY").expect("set exact MuJoCo library path");
    for n in [6, 7] {
        let f = Fixture::new(n, "mujoco");
        f.generated_callbacks(n);
        for (path, bytes) in support::scene(n) {
            let path = f.root.join(path);
            fs::create_dir_all(path.parent().unwrap()).unwrap();
            fs::write(path, bytes).unwrap();
        }
        for name in [
            "geometry_msgs/msg/pose_stamped.hpp",
            "std_msgs/msg/bool.hpp",
            "std_msgs/msg/int32_multi_array.hpp",
        ] {
            let path = f.root.join(name);
            fs::create_dir_all(path.parent().unwrap()).unwrap();
            fs::write(path, "#include \"stub.hpp\"\n").unwrap();
        }
        let base = f
            .root
            .join("mujoco_ros2_control_plugins/mujoco_ros2_control_plugins_base.hpp");
        fs::create_dir_all(base.parent().unwrap()).unwrap();
        fs::write(base,"#pragma once\n#include \"stub.hpp\"\n#include <mujoco/mujoco.h>\nnamespace mujoco_ros2_control_plugins { struct MuJoCoROS2ControlPluginBase { virtual ~MuJoCoROS2ControlPluginBase()=default; virtual bool init(rclcpp::Node::SharedPtr,const mjModel*,mjData*)=0; virtual void update(const mjModel*,mjData*)=0; virtual void pre_step(mjData*)=0; virtual void on_reset(mjData*)=0; virtual void cleanup()=0; }; }\n").unwrap();
        fs::create_dir_all(f.root.join("pluginlib")).unwrap();
        fs::write(
            f.root.join("pluginlib/class_list_macros.hpp"),
            "#pragma once\n#define PLUGINLIB_EXPORT_CLASS(A,B)\n",
        )
        .unwrap();
        fs::write(
            f.root.join("plugin.cpp"),
            &f.files["src/invar_cell_bringup/src/attachment_plugin.cpp"],
        )
        .unwrap();
        let descriptor: Value =
            serde_json::from_str(&f.files["src/invar_cell_bringup/config/cell-snapshot-v1.json"])
                .unwrap();
        let names = f.binding["joint_names"]
            .as_array()
            .unwrap()
            .iter()
            .map(Value::to_string)
            .collect::<Vec<_>>()
            .join(",");
        let probe = include_str!("../fixtures/s1_06/snapshot_plugin_probe.cpp.in")
            .replace("@JOINTS@", &n.to_string())
            .replace("@JOINT_NAMES@", &names)
            .replace(
                "@ROBOT_JOINTS@",
                &descriptor["robot_joints"]
                    .as_array()
                    .unwrap()
                    .len()
                    .to_string(),
            )
            .replace(
                "@ROBOT_JOINT_NAMES@",
                &descriptor["robot_joints"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|j| j["name"].to_string())
                    .collect::<Vec<_>>()
                    .join(","),
            )
            .replace("@ROBOT_ROOT@", descriptor["robot_root"].as_str().unwrap());
        fs::write(f.root.join("snapshot_probe.cpp"), probe).unwrap();
        let output = Command::new("c++")
            .args([
                "-std=c++17",
                "-Wall",
                "-Wextra",
                "-Werror",
                "-pedantic",
                "-pthread",
                "-I",
            ])
            .arg(&f.root)
            .arg("-I")
            .arg(&include)
            .arg(f.root.join("snapshot_probe.cpp"))
            .arg(&library)
            .arg(format!(
                "-Wl,-rpath,{}",
                Path::new(&library).parent().unwrap().display()
            ))
            .arg("-o")
            .arg(f.root.join("snapshot_probe"))
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        let output = Command::new(f.root.join("snapshot_probe"))
            .arg(f.root.join("scene.xml"))
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        println!("{}", String::from_utf8_lossy(&output.stdout));
    }
}
