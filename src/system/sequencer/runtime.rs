//! Private IVSQ/2 process boundary for the explicitly selected task binding.
//! IVSQ/2: every snapshot (kind 1) reply body starts with the task phase label
//! (u8 length, ASCII) so the node can log and publish each phase change.
//! Little-endian bounded frames; one exchange at a time, no shell or AI runtime.
use super::{Action, Config, Sequencer, Snapshot};
use crate::system::tool_projection::{self, Pose};
use invar_foreign::component::{
    Component,
    planner::{Trajectory, Waypoint},
};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::{
    ffi::OsString,
    fs,
    io::{Read, Write},
    path::Path,
};

pub const MAX_FRAME: usize = 65_536;
type Result<T> = std::result::Result<T, String>;
fn hash(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}
fn read_file(path: &Path, expected: &str, limit: u64) -> Result<Vec<u8>> {
    if !path.is_absolute() || !path.is_file() || expected.len() != 64 {
        return Err("invalid bound file".into());
    }
    let mut data = Vec::new();
    fs::File::open(path)
        .map_err(|_| "file open")?
        .take(limit + 1)
        .read_to_end(&mut data)
        .map_err(|_| "file read")?;
    if data.len() as u64 > limit || hash(&data) != expected {
        return Err("bound file hash/size mismatch".into());
    }
    Ok(data)
}
fn verify_file(path: &Path, expected: &str, limit: u64) -> Result<()> {
    if !path.is_absolute() || !path.is_file() || expected.len() != 64 {
        return Err("invalid bound file".into());
    }
    let mut file = fs::File::open(path)
        .map_err(|_| "file open")?
        .take(limit + 1);
    let mut hasher = Sha256::new();
    let mut size = 0u64;
    let mut buffer = [0u8; 16384];
    loop {
        let n = file.read(&mut buffer).map_err(|_| "file read")?;
        if n == 0 {
            break;
        }
        size += n as u64;
        if size > limit {
            return Err("bound file size mismatch".into());
        }
        hasher.update(&buffer[..n]);
    }
    let actual = hasher
        .finalize()
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect::<String>();
    if actual != expected {
        return Err("bound file hash mismatch".into());
    }
    Ok(())
}
fn number(v: &Value) -> Result<f64> {
    v.as_f64()
        .filter(|n| n.is_finite())
        .ok_or_else(|| "invalid finite number".into())
}
fn array<const N: usize>(v: &Value) -> Result<[f64; N]> {
    let v = v
        .as_array()
        .filter(|v| v.len() == N)
        .ok_or("invalid fixed array")?;
    let mut out = [0.0; N];
    for (o, v) in out.iter_mut().zip(v) {
        *o = number(v)?;
    }
    Ok(out)
}
fn pose(v: &Value) -> Result<Pose> {
    let p = array::<7>(v)?;
    let p = Pose {
        translation_m: [p[0], p[1], p[2]],
        quaternion_xyzw: [p[3], p[4], p[5], p[6]],
    };
    if !p.valid() {
        return Err("invalid pose".into());
    }
    Ok(p)
}
fn text<'a>(v: &'a Value, key: &str) -> Result<&'a str> {
    v[key].as_str().ok_or_else(|| format!("missing {key}"))
}
fn digest(s: &str) -> Result<[u8; 32]> {
    if s.len() != 64 || !s.bytes().all(|b| b.is_ascii_hexdigit()) {
        return Err("invalid digest".into());
    }
    let mut out = [0; 32];
    for (i, b) in out.iter_mut().enumerate() {
        *b = u8::from_str_radix(&s[2 * i..2 * i + 2], 16).map_err(|_| "invalid digest")?;
    }
    Ok(out)
}
struct Cursor<'a> {
    data: &'a [u8],
    index: usize,
}
impl<'a> Cursor<'a> {
    fn take<const N: usize>(&mut self) -> Result<[u8; N]> {
        let end = self.index.checked_add(N).ok_or("cursor overflow")?;
        let out = self
            .data
            .get(self.index..end)
            .ok_or("truncated frame")?
            .try_into()
            .map_err(|_| "field width")?;
        self.index = end;
        Ok(out)
    }
    fn u32(&mut self) -> Result<u32> {
        Ok(u32::from_le_bytes(self.take()?))
    }
    fn u64(&mut self) -> Result<u64> {
        Ok(u64::from_le_bytes(self.take()?))
    }
    fn f64(&mut self) -> Result<f64> {
        let n = f64::from_le_bytes(self.take()?);
        if !n.is_finite() {
            return Err("nonfinite wire number".into());
        }
        Ok(n)
    }
    fn values<const N: usize>(&mut self) -> Result<[f64; N]> {
        let mut a = [0.0; N];
        for v in &mut a {
            *v = self.f64()?;
        }
        Ok(a)
    }
    fn pose(&mut self) -> Result<Pose> {
        let p = self.values::<7>()?;
        let out = Pose {
            translation_m: [p[0], p[1], p[2]],
            quaternion_xyzw: [p[3], p[4], p[5], p[6]],
        };
        if !out.valid() {
            return Err("invalid wire pose".into());
        }
        Ok(out)
    }
    fn end(&self) -> Result<()> {
        if self.index != self.data.len() {
            return Err("trailing frame bytes".into());
        }
        Ok(())
    }
}
fn frame(reader: &mut impl Read) -> Result<(u16, u32, Vec<u8>)> {
    let mut header = [0; 16];
    reader
        .read_exact(&mut header)
        .map_err(|_| "truncated/closed sequencer stream")?;
    if &header[..4] != b"IVSQ" || header[4..6] != IVSQ_VERSION.to_le_bytes() {
        return Err("unknown IVSQ protocol".into());
    }
    let mut c = Cursor {
        data: &header[6..],
        index: 0,
    };
    let kind = u16::from_le_bytes(c.take()?);
    let len = c.u32()? as usize;
    let id = c.u32()?;
    if len > MAX_FRAME {
        return Err("IVSQ frame bound".into());
    }
    let mut body = vec![0; len];
    reader
        .read_exact(&mut body)
        .map_err(|_| "truncated frame body")?;
    Ok((kind, id, body))
}
fn reply(writer: &mut impl Write, kind: u16, id: u32, body: &[u8]) -> Result<()> {
    if body.len() > MAX_FRAME {
        return Err("reply bound".into());
    }
    writer
        .write_all(b"IVSQ")
        .and_then(|_| writer.write_all(&IVSQ_VERSION.to_le_bytes()))
        .and_then(|_| writer.write_all(&kind.to_le_bytes()))
        .and_then(|_| writer.write_all(&(body.len() as u32).to_le_bytes()))
        .and_then(|_| writer.write_all(&id.to_le_bytes()))
        .and_then(|_| writer.write_all(body))
        .and_then(|_| writer.flush())
        .map_err(|_| "sequencer reply write".into())
}
const IVSQ_VERSION: u16 = 2;
fn put_pose(v: &mut Vec<u8>, p: invar_foreign::component::Pose) {
    for x in p.translation_m.into_iter().chain(p.quaternion_xyzw) {
        v.extend(x.to_le_bytes());
    }
}

pub fn command(args: Vec<OsString>) -> Result<()> {
    if args.len() != 8
        || args[0] != "--binding"
        || args[2] != "--binding-sha256"
        || args[4] != "--task"
        || args[6] != "--task-sha256"
    {
        return Err("expected hash-bound binding and task paths".into());
    }
    let binding: Value = serde_json::from_slice(&read_file(
        Path::new(&args[1]),
        args[3].to_str().ok_or("binding digest encoding")?,
        MAX_FRAME as u64,
    )?)
    .map_err(|_| "binding JSON")?;
    let task: Value = serde_json::from_slice(&read_file(
        Path::new(&args[5]),
        args[7].to_str().ok_or("task digest encoding")?,
        MAX_FRAME as u64,
    )?)
    .map_err(|_| "task JSON")?;
    if binding["schema"] != "invar-sequencer-binding-v2"
        || task["schema"] != "invar-sequencer-task-v1"
        || task["source_ir_sha256"] != binding["source_ir_sha256"]
    {
        return Err("binding/task schema/source mismatch".into());
    }
    match binding["joint_count"].as_u64() {
        Some(6) => run::<6>(&binding, &task),
        Some(7) => run::<7>(&binding, &task),
        _ => Err("unsupported task joint count".into()),
    }
}
fn run<const N: usize>(binding: &Value, task: &Value) -> Result<()> {
    tool_projection::verify_value(&binding["tool_projection"]).map_err(str::to_owned)?;
    let projection_value = &binding["tool_projection"]["reference_from_cell"];
    let projection = Pose {
        translation_m: array(&projection_value["translation_m"])?,
        quaternion_xyzw: array(&projection_value["quaternion_xyzw"])?,
    };
    let base_value = &binding["tool_projection"]["source_base_from_reference_base"];
    let source_base = Pose {
        translation_m: array(&base_value["translation_m"])?,
        quaternion_xyzw: array(&base_value["quaternion_xyzw"])?,
    };
    let base_inverse = pose(&binding["world_from_base"])?
        .inverse()
        .map_err(str::to_owned)?;
    let model = Path::new(text(task, "fk_model_path")?);
    verify_file(
        model,
        text(&binding["tool_projection"], "reference_model_sha256")?,
        2 * 1024 * 1024,
    )?;
    let library = Path::new(text(task, "fk_library_path")?);
    verify_file(library, text(task, "fk_library_sha256")?, 64 * 1024 * 1024)?;
    let mut fk = Component::open(library, text(binding, "fk_prefix")?, N)
        .map_err(|e| format!("FK load {e:?}"))?;
    if fk.version() != "invar-component-v1/invar-pinocchio-reference-abi-1/pinocchio-4.1.0" {
        return Err("FK reference ABI version".into());
    }
    fk.init(None).map_err(|e| format!("FK init {e:?}"))?;
    fk.configure_fk(model, N as u32)
        .map_err(|e| format!("FK configure {e:?}"))?;
    let project = |p: invar_foreign::component::Pose| {
        source_base
            .compose(Pose {
                translation_m: p.translation_m,
                quaternion_xyzw: p.quaternion_xyzw,
            })
            .map_err(str::to_owned)?
            .compose(projection)
            .map_err(str::to_owned)
    };
    let home_q = array::<N>(&binding["home_q_rad"])?;
    let home_tool = project(fk.step(&home_q).map_err(|e| format!("home FK {e:?}"))?)?;
    let limit_rows = binding["joint_limits_rad"]
        .as_array()
        .filter(|v| v.len() == N)
        .ok_or("joint limit count")?;
    let mut limits = [[0.0; 2]; N];
    for (o, v) in limits.iter_mut().zip(limit_rows) {
        *o = array(v)?;
    }
    let speeds = array::<N>(&binding["velocity_limits_rad_s"])?;
    let mut core = Sequencer::new(Config {
        home_q_rad: home_q,
        home_tool,
        pregrasp_tool: pose(&binding["pregrasp_tool_base"])?,
        grasp_tool: pose(&binding["grasp_tool_base"])?,
        final_cube: pose(&binding["final_cube_base"])?,
        joint_limits_rad: limits,
        velocity_limits_rad_s: speeds,
    })
    .map_err(|e| format!("task config {e:?}"))?;
    let source = digest(text(binding, "source_ir_sha256")?)?;
    let order = digest(text(binding, "joint_order_sha256")?)?;
    let robot_order = digest(text(binding, "robot_joint_order_sha256")?)?;
    let robot_joints = binding["robot_joints"]
        .as_array()
        .filter(|v| !v.is_empty() && v.len() <= 64)
        .ok_or("full robot joint capacity")?;
    let mut robot_limits = Vec::new();
    let mut robot_speeds = Vec::new();
    let mut robot_names = Vec::new();
    for joint in robot_joints {
        if !matches!(
            (joint["kind"].as_str(), joint["position_unit"].as_str()),
            (Some("revolute"), Some("rad")) | (Some("prismatic"), Some("m"))
        ) {
            return Err("full robot joint kind/unit mismatch".into());
        }
        let [lo, hi] = array::<2>(&joint["position_limits_si"])?;
        let speed = number(&joint["velocity_limit_si_per_s"])?;
        if lo >= hi || speed <= 0.0 {
            return Err("invalid full robot joint limits".into());
        }
        robot_limits.push([lo, hi]);
        robot_speeds.push(speed);
        robot_names.push(text(joint, "name")?);
    }
    if robot_names
        .iter()
        .copied()
        .collect::<std::collections::BTreeSet<_>>()
        .len()
        != robot_names.len()
        || digest(&hash(
            serde_json::to_vec(&robot_names)
                .map_err(|_| "joint identity encoding")?
                .as_slice(),
        ))? != robot_order
    {
        return Err("full robot joint identity mismatch".into());
    }
    let arm_names = binding["joint_names"]
        .as_array()
        .filter(|v| v.len() == N)
        .ok_or("arm joint names")?;
    let mut arm_indices = Vec::new();
    for name in arm_names {
        arm_indices.push(
            robot_names
                .iter()
                .position(|n| Some(*n) == name.as_str())
                .ok_or("arm missing from full robot joints")?,
        );
    }
    let mut geom_classes = None;
    let mut last_step = None;
    let mut pending = None;
    let mut exchange = 0u32;
    let stdin = std::io::stdin();
    let stdout = std::io::stdout();
    let mut input = stdin.lock();
    let mut output = stdout.lock();
    for _ in 0..100_000 {
        let (kind, id, body) = frame(&mut input)?;
        exchange = exchange.checked_add(1).ok_or("exchange overflow")?;
        if id != exchange {
            return Err("stale/duplicate exchange".into());
        }
        let mut c = Cursor {
            data: &body,
            index: 0,
        };
        let (response, payload) = match kind {
            1 => {
                let ns = c.u64()?;
                let step = c.u64()?;
                if c.take::<32>()? != source
                    || c.take::<32>()? != order
                    || last_step.is_some_and(|old: u64| old.checked_add(1) != Some(step))
                {
                    return Err("snapshot source/order/step mismatch".into());
                }
                let q = c.values::<N>()?;
                let dq = c.values::<N>()?;
                let tool = c.pose()?;
                let cube = c.pose()?;
                let attached = c.u32()?;
                let count = c.u32()? as usize;
                if !(1..=4096).contains(&count) {
                    return Err("snapshot geometry bound".into());
                }
                let mut classes = Vec::with_capacity(count);
                for _ in 0..count {
                    let b = c.take::<1>()?[0];
                    if b > 3 {
                        return Err("unknown geometry class".into());
                    }
                    classes.push(b);
                }
                if geom_classes.as_ref().is_some_and(|old| old != &classes) {
                    return Err("geometry identity changed".into());
                }
                geom_classes = Some(classes.clone());
                let count = c.u32()? as usize;
                if count > 2048 || !count.is_multiple_of(2) {
                    return Err("snapshot contact bound".into());
                }
                let mut forbidden = 0;
                for _ in 0..count / 2 {
                    let a = i32::from_le_bytes(c.take()?);
                    let b = i32::from_le_bytes(c.take()?);
                    let (Some(a), Some(b)) = (
                        usize::try_from(a).ok().and_then(|i| classes.get(i)),
                        usize::try_from(b).ok().and_then(|i| classes.get(i)),
                    ) else {
                        return Err("unknown contact geom ID".into());
                    };
                    if (*a == 1 && (*b == 1 || *b == 2)) || (*b == 1 && *a == 2) {
                        forbidden += 1;
                    }
                }
                if c.take::<32>()? != robot_order || c.u32()? as usize != robot_joints.len() {
                    return Err("full robot joint identity/count".into());
                }
                let mut robot_q = Vec::with_capacity(robot_joints.len());
                for [lo, hi] in &robot_limits {
                    let value = c.f64()?;
                    if value < *lo || value > *hi {
                        return Err("full robot joint position limit".into());
                    }
                    robot_q.push(value);
                }
                let mut robot_dq = Vec::with_capacity(robot_joints.len());
                for speed in &robot_speeds {
                    let value = c.f64()?;
                    if value.abs() > *speed {
                        return Err("full robot joint velocity limit".into());
                    }
                    robot_dq.push(value);
                }
                for (i, full) in arm_indices.iter().enumerate() {
                    if robot_q[*full] != q[i] || robot_dq[*full] != dq[i] {
                        return Err("arm/full robot joint state conflict".into());
                    }
                }
                c.end()?;
                if attached > 1
                    || q.iter().zip(limits).any(|(q, [lo, hi])| q < &lo || q > &hi)
                    || dq.iter().zip(speeds).any(|(v, m)| v.abs() > m)
                {
                    return Err("snapshot measured joint/attachment contract".into());
                }
                let tool = base_inverse.compose(tool).map_err(str::to_owned)?;
                let cube = base_inverse.compose(cube).map_err(str::to_owned)?;
                let independent =
                    project(fk.step(&q).map_err(|e| format!("same-state FK {e:?}"))?)?;
                let distance = independent
                    .translation_m
                    .iter()
                    .zip(tool.translation_m)
                    .map(|(a, b)| (a - b).powi(2))
                    .sum::<f64>()
                    .sqrt();
                if !distance.is_finite() || distance > 1e-6 {
                    return Err("same-state FK/physics exceeds 1e-6 m".into());
                }
                last_step = Some(step);
                let action = core
                    .observe(Snapshot {
                        simulation_ns: ns,
                        q_rad: q,
                        dq_rad_s: dq,
                        tool_base: tool,
                        cube_base: cube,
                        attached: attached != 0,
                        forbidden_contacts: forbidden,
                    })
                    .map_err(|e| format!("telemetry {e:?}"))?;
                let label = core.phase().label();
                let mut bytes = vec![u8::try_from(label.len()).map_err(|_| "phase label")?];
                bytes.extend(label.as_bytes());
                match action {
                    None => (0, bytes),
                    Some(Action::Plan(r)) => {
                        if pending.is_some() {
                            return Err("duplicate outstanding plan".into());
                        }
                        pending = Some(r);
                        for v in r.start_q_rad.into_iter().chain(r.goal_q_rad) {
                            bytes.extend(v.to_le_bytes());
                        }
                        put_pose(&mut bytes, r.target_base_from_tool);
                        put_pose(&mut bytes, r.cube_base_pose);
                        put_pose(&mut bytes, r.tool_from_cube);
                        bytes.extend(r.phase.to_le_bytes());
                        (11, bytes)
                    }
                    Some(Action::SetAttachment(state)) => {
                        bytes.push(u8::from(state));
                        (12, bytes)
                    }
                    Some(Action::Complete) => (13, bytes),
                }
            }
            2 => {
                let ns = c.u64()?;
                let dt = c.f64()?;
                let count = c.u32()? as usize;
                if !(2..=256).contains(&count) {
                    return Err("trajectory capacity".into());
                }
                let mut points = Vec::with_capacity(count);
                for _ in 0..count {
                    points.push(Waypoint {
                        time_from_start_s: c.f64()?,
                        q_rad: c.values()?,
                        dq_rad_s: c.values()?,
                    });
                }
                c.end()?;
                let request = pending.take().ok_or("unsolicited trajectory")?;
                let trajectory =
                    Trajectory::checked(dt, &points, &request.start_q_rad, &limits, &speeds)
                        .map_err(|e| format!("trajectory {e:?}"))?;
                let q = points.last().ok_or("missing endpoint")?.q_rad;
                let endpoint = project(fk.step(&q).map_err(|e| format!("endpoint FK {e:?}"))?)?;
                core.planned(&trajectory, endpoint, ns)
                    .map_err(|e| format!("plan gate {e:?}"))?;
                // Return the exact accepted input bytes. Glue must byte-compare
                // them before publishing the corresponding retained ROS message.
                (21, body.clone())
            }
            3 => {
                let ns = c.u64()?;
                let success = c.u32()?;
                c.end()?;
                if success > 1 {
                    return Err("attachment Boolean".into());
                }
                core.attachment_response(success == 1, ns)
                    .map_err(|e| format!("attachment {e:?}"))?;
                (0, Vec::new())
            }
            4 => {
                c.end()?;
                core.planner_failed()
                    .map_err(|e| format!("planner {e:?}"))?;
                return Err("planner failed".into());
            }
            _ => return Err("unknown task event".into()),
        };
        reply(&mut output, response, id, &payload)?;
        if response == 13 {
            fk.shutdown().map_err(|e| format!("FK shutdown {e:?}"))?;
            return Ok(());
        }
    }
    Err("task exchange count bound".into())
}
