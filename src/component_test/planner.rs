//! Hash-bound planner endpoint contract. Diagnostic fixtures never write passes.
use std::{
    collections::BTreeMap,
    path::{Path, PathBuf},
};

use invar::{front, robotics, system};
use invar_foreign::component::planner::{Config, Planner, Request, phase};
use invar_foreign::component::{Component, ComponentError, Pose};
use serde_json::{Value, json};
use sha2::Digest;

use super::{Failure, fail, file_hash, foreign, numbers, string};

const METRES: f64 = 0.002;
const RADIANS: f64 = std::f64::consts::PI / 180.0;
const START_RAD: f64 = 1e-6;
const ROLES: [&str; 9] = [
    "library",
    "child",
    "model",
    "robot_config",
    "scene",
    "fk_library",
    "fk_model",
    "fk_manifest",
    "fk_evidence",
];

fn load(path: &Path) -> Result<Value, Failure> {
    use std::io::Read;
    let mut bytes = Vec::new();
    std::fs::File::open(path)
        .map_err(|e| fail(e.to_string()))?
        .take(64 * 1024 + 1)
        .read_to_end(&mut bytes)
        .map_err(|e| fail(e.to_string()))?;
    if bytes.len() > 64 * 1024 {
        return Err(fail("planner JSON exceeds 64 KiB"));
    }
    serde_json::from_slice(&bytes).map_err(|e| fail(e.to_string()))
}

fn pose(value: &Value, key: &str) -> Result<Pose, Failure> {
    let a = numbers::<7>(value, key)?;
    let result = Pose {
        translation_m: [a[0], a[1], a[2]],
        quaternion_xyzw: [a[3], a[4], a[5], a[6]],
    };
    if !valid_pose(result) {
        return Err(fail("pose requires finite unit xyzw quaternion"));
    }
    Ok(result)
}

fn project(pose: Pose, offset: system::tool_projection::Pose, left: bool) -> Result<Pose, Failure> {
    let p = system::tool_projection::Pose {
        translation_m: pose.translation_m,
        quaternion_xyzw: pose.quaternion_xyzw,
    };
    let result = if left {
        offset.compose(p)
    } else {
        p.compose(offset)
    }
    .map_err(fail)?;
    Ok(Pose {
        translation_m: result.translation_m,
        quaternion_xyzw: result.quaternion_xyzw,
    })
}

/// Wire poses agree componentwise within `tolerance` (either quaternion sign).
fn within(actual: &[f64], expected: [f64; 7], tolerance: f64) -> bool {
    let close = |sign: f64| (3..7).all(|i| (actual[i] - sign * expected[i]).abs() <= tolerance);
    actual.len() == 7
        && (0..3).all(|i| (actual[i] - expected[i]).abs() <= tolerance)
        && (close(1.0) || close(-1.0))
}

/// Planner wire pose equals a checked cell-derived pose (either quaternion sign).
fn same_pose(value: &Value, expected: system::tool_projection::Pose) -> bool {
    value
        .as_array()
        .and_then(|a| a.iter().map(Value::as_f64).collect::<Option<Vec<_>>>())
        .is_some_and(|actual| within(&actual, system::planner_scene::flat(expected), 1e-12))
}

fn checked(pose: Pose) -> system::tool_projection::Pose {
    system::tool_projection::Pose {
        translation_m: pose.translation_m,
        quaternion_xyzw: pose.quaternion_xyzw,
    }
}

fn valid_pose(pose: Pose) -> bool {
    pose.translation_m
        .iter()
        .chain(&pose.quaternion_xyzw)
        .all(|x| x.is_finite())
        && (pose
            .quaternion_xyzw
            .iter()
            .map(|x| x * x)
            .sum::<f64>()
            .sqrt()
            - 1.0)
            .abs()
            <= 1e-12
}

fn endpoint(actual: Pose, target: Pose) -> Result<(f64, f64), Failure> {
    if !valid_pose(actual) || !valid_pose(target) {
        return Err(Failure::new("S1403", "invalid independent FK/target pose"));
    }
    let distance = actual
        .translation_m
        .iter()
        .zip(target.translation_m)
        .map(|(a, b)| (a - b).powi(2))
        .sum::<f64>()
        .sqrt();
    let same = actual
        .quaternion_xyzw
        .iter()
        .zip(target.quaternion_xyzw)
        .map(|(a, b)| (a - b).powi(2))
        .sum::<f64>()
        .sqrt();
    let opposite = actual
        .quaternion_xyzw
        .iter()
        .zip(target.quaternion_xyzw)
        .map(|(a, b)| (a + b).powi(2))
        .sum::<f64>()
        .sqrt();
    let angle = 4.0 * same.min(opposite).atan2(same.max(opposite));
    if !distance.is_finite() || !angle.is_finite() || distance > METRES || angle > RADIANS {
        return Err(Failure::new(
            "S1403",
            format!("independent endpoint error: {distance} m, {angle} rad"),
        ));
    }
    Ok((distance, angle))
}

struct Artifact {
    path: PathBuf,
    hash: String,
}

fn bind(runtime: &Value, directory: &Path) -> Result<BTreeMap<&'static str, Artifact>, Failure> {
    if runtime["schema"] != "invar-planner-runtime-v3"
        || runtime["artifacts"].as_object().map(|a| a.len()) != Some(ROLES.len())
    {
        return Err(fail("invalid planner runtime/artifact roles"));
    }
    let mut result = BTreeMap::new();
    for role in ROLES {
        let declared = &runtime["artifacts"][role];
        let path = directory
            .join(string(declared, "path")?)
            .canonicalize()
            .map_err(|e| fail(e.to_string()))?;
        let metadata = path.metadata().map_err(|e| fail(e.to_string()))?;
        if !metadata.is_file() || metadata.len() > 64 * 1024 * 1024 {
            return Err(fail("runtime artifact exceeds 64 MiB or is not a file"));
        }
        let hash = file_hash(&path)?;
        if declared["sha256"] != hash {
            return Err(Failure::new(
                "S1405",
                format!("{role} artifact hash mismatch"),
            ));
        }
        result.insert(role, Artifact { path, hash });
    }
    if result["library"].path == result["fk_library"].path
        || result["library"].hash == result["fk_library"].hash
    {
        return Err(fail("planner and independent FK libraries must differ"));
    }
    Ok(result)
}

fn bytes_hash(hash: &str) -> Result<[u8; 32], Failure> {
    if hash.len() != 64 || !hash.bytes().all(|b| b.is_ascii_hexdigit()) {
        return Err(fail("invalid SHA256"));
    }
    let mut out = [0; 32];
    for (i, byte) in out.iter_mut().enumerate() {
        *byte =
            u8::from_str_radix(&hash[2 * i..2 * i + 2], 16).map_err(|_| fail("invalid SHA256"))?;
    }
    Ok(out)
}

fn immutable(path: &Path, record: &Value) -> Result<(), Failure> {
    use std::io::Write;
    let parent = path
        .parent()
        .ok_or_else(|| fail("missing evidence directory"))?;
    std::fs::create_dir_all(parent).map_err(|e| fail(e.to_string()))?;
    let bytes = serde_json::to_vec_pretty(record).map_err(|e| fail(e.to_string()))?;
    if bytes.len() > 64 * 1024 {
        return Err(fail("planner evidence exceeds 64 KiB"));
    }
    let temporary = parent.join(format!(".planner-{}.{}.tmp", std::process::id(), nonce()?));
    let result = (|| {
        let mut file = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&temporary)?;
        file.write_all(&bytes)?;
        file.sync_all()?;
        std::fs::hard_link(&temporary, path)
    })();
    let _ = std::fs::remove_file(temporary);
    let same_record = || -> Result<bool, Failure> {
        use std::io::Read;
        let mut existing = Vec::new();
        std::fs::File::open(path)
            .map_err(|e| fail(e.to_string()))?
            .take(64 * 1024 + 1)
            .read_to_end(&mut existing)
            .map_err(|e| fail(e.to_string()))?;
        Ok(existing == bytes)
    };
    match result {
        Ok(()) => Ok(()),
        Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists && same_record()? => Ok(()),
        Err(e) => Err(fail(e.to_string())),
    }
}

fn nonce() -> Result<u128, Failure> {
    Ok(std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_err(|e| fail(e.to_string()))?
        .as_nanos())
}

fn host() -> Result<Value, Failure> {
    use std::{
        process::{Command, Stdio},
        time::{Duration, Instant},
    };
    if std::env::consts::OS != "linux" {
        return Err(fail("production planner contract requires Linux"));
    }
    let path = std::env::temp_dir().join(format!(
        "invar-planner-host-{}-{}",
        std::process::id(),
        nonce()?
    ));
    let result = (|| {
        let file = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&path)
            .map_err(|e| fail(e.to_string()))?;
        let stderr = file.try_clone().map_err(|e| fail(e.to_string()))?;
        let mut child = Command::new("nvidia-smi")
            .args(["--query-gpu=driver_version,name", "--format=csv,noheader"])
            .stdin(Stdio::null())
            .stdout(file)
            .stderr(stderr)
            .spawn()
            .map_err(|e| fail(e.to_string()))?;
        let deadline = Instant::now() + Duration::from_secs(5);
        loop {
            match child.try_wait() {
                Ok(Some(status)) if status.success() => break,
                Ok(Some(_)) => return Err(fail("NVIDIA host query failed")),
                Err(e) => {
                    let _ = child.kill();
                    let _ = child.wait();
                    return Err(fail(e.to_string()));
                }
                Ok(None) => {}
            }
            if Instant::now() >= deadline || path.metadata().map_or(true, |m| m.len() > 65536) {
                let _ = child.kill();
                let _ = child.wait();
                return Err(fail("NVIDIA host query bound exceeded"));
            }
            std::thread::sleep(Duration::from_millis(20));
        }
        use std::io::Read;
        let mut bytes = Vec::new();
        std::fs::File::open(&path)
            .map_err(|e| fail(e.to_string()))?
            .take(65537)
            .read_to_end(&mut bytes)
            .map_err(|e| fail(e.to_string()))?;
        let text = String::from_utf8(bytes).map_err(|e| fail(e.to_string()))?;
        if text.len() > 65536 || text.trim().is_empty() {
            return Err(fail("invalid NVIDIA host query"));
        }
        for line in text.lines() {
            let (driver, device) = line
                .split_once(',')
                .ok_or_else(|| fail("invalid NVIDIA host query"))?;
            if device.trim().is_empty() {
                return Err(fail("invalid NVIDIA host query"));
            }
            let driver = driver.trim();
            let version = driver
                .split('.')
                .map(str::parse::<u32>)
                .collect::<Result<Vec<_>, _>>()
                .map_err(|_| fail("invalid NVIDIA driver version"))?;
            if version.len() != 3 || version.as_slice() < [580, 65, 6].as_slice() {
                return Err(fail("NVIDIA driver must be >=580.65.06"));
            }
        }
        Ok(json!({"os":"linux", "arch":std::env::consts::ARCH, "nvidia_smi":text}))
    })();
    let _ = std::fs::remove_file(path);
    result
}

fn execute<const N: usize>(
    path: &Path,
    runtime_path: &Path,
    library: Option<&Path>,
    model: Option<&Path>,
    child: Option<&Path>,
) -> Result<Value, Failure> {
    let path = path.canonicalize().map_err(|e| fail(e.to_string()))?;
    let directory = path
        .parent()
        .ok_or_else(|| fail("missing manifest parent"))?;
    let root = path
        .ancestors()
        .find(|p| p.join(".git").exists())
        .unwrap_or(directory);
    let source = std::fs::read(&path).map_err(|e| fail(e.to_string()))?;
    let manifest = front::parse_system(&source)
        .and_then(system::compile_manifest)
        .map_err(|e| fail(e.message))?;
    if manifest["config"]["layout"] != "inv_planner_config_v1"
        || manifest["isolation"] != "separate-process"
    {
        return Err(fail("requires a separate-process planner manifest"));
    }
    let count = robotics::planner::manifest_joints(
        manifest["ports"]
            .as_array()
            .ok_or_else(|| fail("missing ports"))?,
    )
    .map_err(fail)?;
    if count as usize != N || super::tolerances(&manifest)? != (METRES, RADIANS) {
        return Err(fail(
            "requires the explicit approved endpoint manifest tolerances",
        ));
    }
    let header = system::foreign::header(&manifest).map_err(fail)?;
    let (digest, inputs) =
        system::foreign::manifest_hash(root, &path, &manifest, &header).map_err(fail)?;
    let runtime_path = runtime_path
        .canonicalize()
        .map_err(|e| fail(e.to_string()))?;
    if !manifest["inputs"]
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(Value::as_str)
        .any(|p| {
            directory
                .join(p)
                .canonicalize()
                .is_ok_and(|p| p == runtime_path)
        })
    {
        return Err(fail("runtime must be a declared manifest input"));
    }
    let runtime_hash = file_hash(&runtime_path)?;
    let runtime = load(&runtime_path)?;
    let diagnostic = match runtime["qualification"].as_str() {
        Some("cpu-diagnostic") => true,
        Some("production") => false,
        _ => return Err(fail("invalid runtime qualification")),
    };
    let artifacts = bind(
        &runtime,
        runtime_path
            .parent()
            .ok_or_else(|| fail("missing runtime parent"))?,
    )?;
    let declared: std::collections::BTreeSet<_> = manifest["inputs"]
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(Value::as_str)
        .map(|p| {
            directory
                .join(p)
                .canonicalize()
                .map_err(|e| fail(e.to_string()))
        })
        .collect::<Result<_, _>>()?;
    let (projection, cell_cube) = {
        let cell = &runtime["cell_source"];
        let cell_path = runtime_path
            .parent()
            .ok_or_else(|| fail("missing runtime parent"))?
            .join(string(cell, "path")?)
            .canonicalize()
            .map_err(|e| fail(e.to_string()))?;
        if !declared.contains(&cell_path) || cell["sha256"] != file_hash(&cell_path)? {
            return Err(fail(
                "cell projection source must be hash-bound manifest input",
            ));
        }
        if cell_path.metadata().map_err(|e| fail(e.to_string()))?.len() > 1024 * 1024 {
            return Err(fail("cell source exceeds 1 MiB"));
        }
        let bytes = std::fs::read(&cell_path).map_err(|e| fail(e.to_string()))?;
        let ir = front::parse_system(&bytes)
            .and_then(|p| system::compile_at(p, cell_path.parent().unwrap()))
            .map_err(|e| fail(e.message))?;
        let p = system::tool_projection::selected(&ir).map_err(fail)?;
        let scene = system::planner_scene::selected(&ir).map_err(fail)?;
        let config = &runtime["config"];
        if config["mount_exclusion"] != scene.mount_exclusion
            || !same_pose(&config["table_pose"], scene.table_pose)
            || config["table_half_extents_m"] != json!(scene.table_half_extents_m)
        {
            return Err(fail(
                "runtime planner scene differs from checked cell source",
            ));
        }
        if runtime["tool_projection"] != p.value()
            || !system::tool_projection::same_kinematics(
                &p.reference_model_sha256,
                &artifacts["fk_model"].hash,
            )
        {
            return Err(fail("projection/model differs from checked pinned source"));
        }
        let scene_path = cell_path
            .parent()
            .unwrap()
            .join(
                ir.checked_document().map_err(fail)?["robots"]
                    .as_array()
                    .and_then(|r| {
                        r.iter().find(|r| {
                            r["tool"] == p.cell_tool
                                && r["asset"]["sha256"] == p.source_scene_sha256
                        })
                    })
                    .and_then(|r| r["asset"]["path"].as_str())
                    .ok_or_else(|| fail("missing projection source scene"))?,
            )
            .canonicalize()
            .map_err(|e| fail(e.to_string()))?;
        if !declared.contains(&scene_path) || file_hash(&scene_path)? != p.source_scene_sha256 {
            return Err(fail(
                "projection source scene must be hash-bound manifest input",
            ));
        }
        (p, (scene.cube_base_pose, scene.tool_from_cube))
    };
    if artifacts.values().any(|a| !declared.contains(&a.path)) {
        return Err(fail(
            "every selected runtime artifact must be a declared manifest input",
        ));
    }
    for (role, selected) in [("library", library), ("model", model), ("child", child)] {
        if !selected.is_some_and(|p| p.canonicalize().is_ok_and(|p| p == artifacts[role].path)) {
            return Err(fail(format!(
                "explicit --{role} must match runtime binding"
            )));
        }
    }
    if !diagnostic
        && artifacts["child"].hash
            != super::hex(&sha2::Sha256::digest(include_bytes!(
                "../../examples/cells/reach_place/components/curobo_planner/planner_server.py"
            )))
    {
        return Err(fail(
            "production contract forbids a substituted planner child",
        ));
    }
    let golden = load(&directory.join(string(&manifest["golden"][0], "path")?))?;
    if golden["schema"] != "invar-planner-endpoint-goldens-v3"
        || golden["joint_count"] != N
        || golden["seed"] != 123
        || golden["metrics"]
            != json!({"translation":"euclidean-m","orientation":"sign-invariant-geodesic-rad","translation_max_m":METRES,"orientation_max_rad":RADIANS,"start_max_rad":START_RAD})
    {
        return Err(fail("invalid endpoint metric/seed contract"));
    }
    if golden["tool_projection"] != runtime["tool_projection"] {
        return Err(fail("goal and runtime tool projections disagree"));
    }
    let vectors = golden["vectors"]
        .as_array()
        .ok_or_else(|| fail("missing endpoint vectors"))?;
    if !(5..=16).contains(&vectors.len())
        || !(phase::FREE..=phase::HOME).all(|p| vectors.iter().any(|v| v["phase"] == p))
    {
        return Err(fail(
            "requires 5..16 fixed vectors covering every planner world phase",
        ));
    }
    let config = &runtime["config"];
    let names = config["joint_names"]
        .as_array()
        .ok_or_else(|| fail("missing joint names"))?
        .iter()
        .map(|n| n.as_str().ok_or_else(|| fail("invalid joint name")))
        .collect::<Result<Vec<_>, _>>()?;
    let limits = config["joint_limits_rad"]
        .as_array()
        .ok_or_else(|| fail("missing joint limits"))?
        .iter()
        .map(|a| numbers::<2>(&json!({"limit":a}), "limit"))
        .collect::<Result<Vec<_>, _>>()?;
    let speeds = numbers::<N>(config, "velocity_limits_rad_s")?;
    if names.len() != N
        || limits.len() != N
        || names
            .iter()
            .copied()
            .collect::<std::collections::BTreeSet<_>>()
            .len()
            != N
        || names
            .iter()
            .any(|s| s.is_empty() || s.len() > 64 || s.contains('\0'))
        || limits.iter().any(|[lo, hi]| lo >= hi)
        || speeds.iter().any(|v| *v <= 0.0)
    {
        return Err(fail("invalid ordered joint limits/names/speeds"));
    }
    let fk_source =
        std::fs::read(&artifacts["fk_manifest"].path).map_err(|e| fail(e.to_string()))?;
    let fk_manifest = front::parse_system(&fk_source)
        .and_then(system::compile_manifest)
        .map_err(|e| fail(e.message))?;
    let fk_header = system::foreign::header(&fk_manifest).map_err(fail)?;
    let (fk_digest, fk_inputs) = system::foreign::manifest_hash(
        root,
        &artifacts["fk_manifest"].path,
        &fk_manifest,
        &fk_header,
    )
    .map_err(fail)?;
    if fk_inputs
        .iter()
        .any(|(role, p, _)| role != "generated-header" && !declared.contains(&root.join(p)))
    {
        return Err(fail(
            "planner manifest must declare the complete FK input snapshot",
        ));
    }
    system::foreign::verify_evidence(&artifacts["fk_evidence"].path, &fk_digest)
        .map_err(|e| Failure::new("S1405", e))?;
    let fk_record = load(&artifacts["fk_evidence"].path)?;
    if fk_manifest["config"]["layout"] != "inv_fk_config_v1"
        || fk_manifest["isolation"] != "in-process"
        || fk_record["library_sha256"] != artifacts["fk_library"].hash
        || fk_record["model_sha256"] != artifacts["fk_model"].hash
    {
        return Err(Failure::new(
            "S1405",
            "FK evidence does not bind selected library/model",
        ));
    }
    let fk_golden = load(
        &artifacts["fk_manifest"]
            .path
            .parent()
            .ok_or_else(|| fail("missing FK parent"))?
            .join(string(&fk_manifest["golden"][0], "path")?),
    )?;
    let arm = if let Some(arms) = fk_golden["arms"].as_array() {
        arms.iter()
            .find(|a| a["model_sha256"] == artifacts["fk_model"].hash)
            .ok_or_else(|| Failure::new("S1405", "FK model does not match golden source"))?
    } else {
        &fk_golden
    };
    let mapping = golden["joint_mapping"]
        .as_array()
        .ok_or_else(|| fail("missing ordered joint mapping"))?;
    let fk_names = arm["joint_names"]
        .as_array()
        .ok_or_else(|| fail("missing golden FK joint order"))?;
    if arm["model_sha256"] != artifacts["fk_model"].hash
        || fk_names.len() != N
        || mapping.len() != N
        || (0..N).any(|i| mapping[i] != json!([names[i], fk_names[i]]))
    {
        return Err(fail(
            "planner/FK joint mapping is not source-aligned in order",
        ));
    }
    let request_type = manifest["ports"]
        .as_array()
        .and_then(|p| p.iter().find(|p| p["direction"] == "in"))
        .and_then(|p| p["type"].as_str())
        .ok_or_else(|| fail("missing request type"))?;
    // The checked planner record has exactly five flat scalar/count/frame args.
    let request_args: Vec<_> = request_type
        .strip_prefix("std::robotics::PlannerRequest<")
        .and_then(|s| s.strip_suffix('>'))
        .ok_or_else(|| fail("invalid request type"))?
        .split(',')
        .collect();
    if request_args.len() != 5 {
        return Err(fail("invalid request type"));
    }
    let expected_pose = format!(
        "std::robotics::Pose<f64,{},{}>",
        request_args[2], request_args[3]
    );
    if !fk_manifest["ports"]
        .as_array()
        .into_iter()
        .flatten()
        .any(|p| p["direction"] == "out" && p["type"] == expected_pose)
        || !fk_manifest["ports"]
            .as_array()
            .into_iter()
            .flatten()
            .any(|p| p["direction"] == "in" && p["layout"] == format!("inv_joint{N}_f64"))
    {
        return Err(fail("FK joint count or nominal base/tool frames disagree"));
    }
    let mut planner = Planner::<N>::open(
        &artifacts["library"].path,
        string(&manifest["abi"], "prefix")?,
    )
    .map_err(foreign)?;
    let mut fk = Component::open(
        &artifacts["fk_library"].path,
        string(&fk_manifest["abi"], "prefix")?,
        N,
    )
    .map_err(foreign)?;
    if fk_record["library_version"] != fk.version()
        || (!diagnostic
            && (planner.version() != format!("invar-planner-component-v2/{N}/INVP-v3")
                || fk.version()
                    != "invar-component-v1/invar-pinocchio-reference-abi-1/pinocchio-4.1.0"))
    {
        return Err(fail(
            "fixture/unbound library version cannot qualify production",
        ));
    }
    let configured = Config {
        model_sha256: bytes_hash(&artifacts["model"].hash)?,
        robot_config_sha256: bytes_hash(&artifacts["robot_config"].hash)?,
        scene_sha256: bytes_hash(&artifacts["scene"].hash)?,
        joint_limits_rad: &limits,
        velocity_limits_rad_s: &speeds,
        table_pose: pose(config, "table_pose")?,
        table_half_extents_m: numbers::<3>(config, "table_half_extents_m")?,
        cube_edge_m: config["cube_edge_m"]
            .as_f64()
            .ok_or_else(|| fail("missing cube edge"))?,
        base_frame: string(config, "base_frame")?,
        tool_frame: string(config, "tool_frame")?,
        joint_names: &names,
        robot_config_path: &artifacts["robot_config"].path,
        scene_config_path: &artifacts["scene"].path,
    };
    if configured.cube_edge_m != 0.05
        || configured.table_half_extents_m.iter().any(|v| *v <= 0.0)
        || [configured.base_frame, configured.tool_frame]
            .iter()
            .any(|s| s.is_empty() || s.len() > 64 || s.contains('\0'))
    {
        return Err(fail("invalid fixed scene/frame configuration"));
    }
    if projection.planner_reference_frame != configured.tool_frame {
        return Err(fail(
            "planner frame differs from checked reference convention",
        ));
    }
    let mut ids = std::collections::BTreeSet::new();
    let mut requests = Vec::new();
    for vector in vectors {
        let id = string(vector, "id")?;
        if id.is_empty() || id.len() > 64 || !ids.insert(id) {
            return Err(fail("invalid/duplicate vector id"));
        }
        let world = vector["phase"]
            .as_u64()
            .filter(|n| *n <= u64::from(phase::HOME))
            .ok_or_else(|| fail("invalid planner world phase"))? as u32;
        let start_q_rad = numbers::<N>(vector, "start_q_rad")?;
        // Only home vectors carry a joint goal; every other phase repeats the start.
        let goal_q_rad = match (world == phase::HOME, vector.get("goal_q_rad")) {
            (true, Some(_)) => numbers::<N>(vector, "goal_q_rad")?,
            (false, None) => start_q_rad,
            _ => return Err(fail("joint goal only and always on home vectors")),
        };
        let request = Request {
            start_q_rad,
            goal_q_rad,
            target_base_from_tool: pose(vector, "target_base_from_tool")?,
            cube_base_pose: pose(vector, "cube_base_pose")?,
            tool_from_cube: pose(vector, "tool_from_cube")?,
            phase: world,
        };
        // A resting cube is at its cell pose; a held one is checked against FK below.
        let (cube, grasp) = cell_cube;
        if !same_pose(&vector["tool_from_cube"], grasp)
            || (!phase::held(request.phase) && !same_pose(&vector["cube_base_pose"], cube))
        {
            return Err(fail("request cube differs from checked cell source"));
        }
        let bounds: [[f64; 2]; N] = limits
            .as_slice()
            .try_into()
            .map_err(|_| fail("joint limit count"))?;
        if !request.valid(&bounds) {
            return Err(Failure::new(
                "S1404",
                "start/goal outside declared joint limits or invalid request",
            ));
        }
        requests.push((id, request));
    }
    let host = if diagnostic {
        json!({"os":std::env::consts::OS,"arch":std::env::consts::ARCH,"qualification":"cpu-diagnostic"})
    } else {
        host()?
    };
    let mut observations = Vec::new();
    let result = (|| {
        planner.init(&artifacts["child"].path).map_err(foreign)?;
        fk.init(None).map_err(foreign)?;
        planner.configure(&configured).map_err(foreign)?;
        fk.configure_fk(&artifacts["fk_model"].path, N as u32)
            .map_err(foreign)?;
        for (id, request) in &requests {
            let p = &projection;
            let mut native = *request;
            native.target_base_from_tool = project(
                request.target_base_from_tool,
                p.reference_from_cell.inverse().map_err(fail)?,
                false,
            )?;
            native.tool_from_cube = project(request.tool_from_cube, p.reference_from_cell, true)?;
            native.target_base_from_tool = project(
                native.target_base_from_tool,
                p.source_base_from_reference_base.inverse().map_err(fail)?,
                true,
            )?;
            native.cube_base_pose = project(
                request.cube_base_pose,
                p.source_base_from_reference_base.inverse().map_err(fail)?,
                true,
            )?;
            if phase::held(request.phase) {
                let start = fk.step(&request.start_q_rad).map_err(foreign)?;
                let tool = project(
                    project(start, p.reference_from_cell, false)?,
                    p.source_base_from_reference_base,
                    true,
                )?;
                let held = project(request.tool_from_cube, checked(tool), true)?;
                let cube = system::planner_scene::flat(checked(request.cube_base_pose));
                if !within(&system::planner_scene::flat(checked(held)), cube, 1e-9) {
                    return Err(fail("attached cube is not held at the declared grasp"));
                }
            }
            let trajectory = planner.plan(&native).map_err(|e| {
                let mut failure=foreign(e);
                failure.details=Some(json!({"vector":id,"start_q_rad":request.start_q_rad.as_slice(),"completed_vectors":observations}));
                failure
            })?;
            let first = trajectory
                .waypoints()
                .first()
                .ok_or_else(|| fail("missing first waypoint"))?;
            let last = trajectory
                .waypoints()
                .last()
                .ok_or_else(|| fail("missing final waypoint"))?;
            let start_error = first
                .q_rad
                .iter()
                .zip(request.start_q_rad)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0_f64, f64::max);
            if start_error > START_RAD {
                return Err(Failure::new("S1403", "start continuity exceeded"));
            }
            let raw = fk.step(&last.q_rad).map_err(foreign)?;
            let actual = project(
                project(raw, p.reference_from_cell, false)?,
                p.source_base_from_reference_base,
                true,
            )?;
            let (translation, orientation) = endpoint(actual, request.target_base_from_tool).map_err(|mut failure| {
                failure.details=Some(json!({"vector":id,"final_q_rad":last.q_rad.as_slice(),"actual_translation_m":actual.translation_m,"actual_quaternion_xyzw":actual.quaternion_xyzw,"target_translation_m":request.target_base_from_tool.translation_m,"target_quaternion_xyzw":request.target_base_from_tool.quaternion_xyzw,"completed_vectors":observations}));
                failure
            })?;
            let mut observation = json!({"id":id,"phase":request.phase,"waypoints":trajectory.waypoints().len(),"dt_s":trajectory.dt_s(),"start_error_rad":start_error,"final_q_rad":last.q_rad.as_slice(),"final_fk_translation_m":actual.translation_m,"final_fk_quaternion_xyzw":actual.quaternion_xyzw,"translation_error_m":translation,"orientation_error_rad":orientation});
            if request.phase == phase::HOME {
                // The home move must end on the exact joint goal, not another IK branch.
                let goal_error = last
                    .q_rad
                    .iter()
                    .zip(request.goal_q_rad)
                    .map(|(a, b)| (a - b).abs())
                    .fold(0.0_f64, f64::max);
                if goal_error > START_RAD {
                    return Err(Failure::new("S1403", "home joint goal exceeded"));
                }
                observation["goal_error_rad"] = json!(goal_error);
            }
            observations.push(observation);
        }
        Ok(())
    })();
    let planner_shutdown = planner.shutdown().map_err(foreign);
    let fk_shutdown = fk.shutdown().map_err(foreign);
    result?;
    planner_shutdown?;
    fk_shutdown?;
    let empty = Pose {
        translation_m: [0.0; 3],
        quaternion_xyzw: [0.0, 0.0, 0.0, 1.0],
    };
    if !matches!(
        planner.plan(&Request {
            start_q_rad: [0.0; N],
            goal_q_rad: [0.0; N],
            target_base_from_tool: empty,
            cube_base_pose: empty,
            tool_from_cube: empty,
            phase: phase::FREE
        }),
        Err(ComponentError::Lifecycle)
    ) || fk.step(&[0.0; N]) != Err(ComponentError::Lifecycle)
    {
        return Err(Failure::new("S1402", "step after shutdown ran"));
    }
    if source != std::fs::read(&path).map_err(|e| fail(e.to_string()))?
        || runtime_hash != file_hash(&runtime_path)?
        || fk_source
            != std::fs::read(&artifacts["fk_manifest"].path).map_err(|e| fail(e.to_string()))?
        || system::foreign::manifest_hash(root, &path, &manifest, &header)
            .map_err(fail)?
            .0
            != digest
        || system::foreign::manifest_hash(
            root,
            &artifacts["fk_manifest"].path,
            &fk_manifest,
            &fk_header,
        )
        .map_err(fail)?
        .0 != fk_digest
    {
        return Err(Failure::new(
            "S1405",
            "contract inputs changed during execution",
        ));
    }
    for (role, artifact) in &artifacts {
        if file_hash(&artifact.path)? != artifact.hash {
            return Err(Failure::new(
                "S1405",
                format!("{role} changed during execution"),
            ));
        }
    }
    Ok(
        json!({"schema":if diagnostic {"invar-planner-diagnostic-v3"} else {"invar-planner-contract-evidence-v3"},"tool_projection":runtime["tool_projection"],"cell_source":runtime["cell_source"],"status":if diagnostic {"diagnostic"} else {"pass"},"component":manifest["name"],"manifest_hash":digest,"runtime_sha256":runtime_hash,"library_version":planner.version(),"library_sha256":artifacts["library"].hash,"host":host,"fk_manifest_hash":fk_digest,"fk_library_version":fk.version(),"seed":123,"metrics":golden["metrics"],"inputs":inputs.iter().map(|(role,path,sha)| json!({"role":role,"path":path,"sha256":sha})).collect::<Vec<_>>(),"artifacts":artifacts.iter().map(|(role,a)| (*role,json!({"path":a.path,"sha256":a.hash}))).collect::<BTreeMap<_,_>>(),"results":{"layout":"pass","lifecycle":"pass","invalid_inputs":"checked-boundary","vectors":observations},"scope":"endpoint_and_start_only; collision/live_S1_acceptance_remains_separate"}),
    )
}

pub(super) fn run(
    path: &Path,
    runtime: &Path,
    library: Option<&Path>,
    output: &Path,
    model: Option<&Path>,
    child: Option<&Path>,
) -> Result<Value, Failure> {
    let entry_manifest_sha256 = file_hash(path).ok();
    let entry_runtime_sha256 = file_hash(runtime).ok();
    let result = (|| {
        let count = load(runtime)?["config"]["joint_names"]
            .as_array()
            .map(|n| n.len())
            .ok_or_else(|| fail("missing runtime joint names"))?;
        match count {
            6 => execute::<6>(path, runtime, library, model, child),
            7 => execute::<7>(path, runtime, library, model, child),
            _ => Err(fail("planner requires six/seven joints")),
        }
    })();
    match result {
        Ok(record) => {
            let diagnostic = record["status"] == "diagnostic";
            let name = string(&record, "component")?;
            let digest = string(&record, "manifest_hash")?;
            let destination = if diagnostic {
                output.join(format!("diagnostic-{name}-{digest}-{}.json", nonce()?))
            } else {
                output.join(format!("{name}-{digest}.json"))
            };
            immutable(&destination, &record)?;
            Ok(
                json!({"status":if diagnostic {"diagnostic"} else {"ok"},"evidence":destination,"manifest_hash":digest,"vectors":record["results"]["vectors"].as_array().map(Vec::len)}),
            )
        }
        Err(mut error) => {
            let destination = output.join(format!(
                "failed-planner-{}-{}.json",
                std::process::id(),
                nonce()?
            ));
            immutable(
                &destination,
                &json!({"schema":"invar-planner-contract-attempt-v1","status":"error","code":error.code,"message":error.message,"manifest":path,"runtime":runtime,"entry_manifest_sha256":entry_manifest_sha256,"entry_runtime_sha256":entry_runtime_sha256,"observations":error.details}),
            )?;
            error.message = format!(
                "{}; failed attempt: {}",
                error.message,
                destination.display()
            );
            Err(error)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn identity() -> Pose {
        Pose {
            translation_m: [0.0; 3],
            quaternion_xyzw: [0.0, 0.0, 0.0, 1.0],
        }
    }
    #[test]
    fn endpoint_uses_euclidean_distance_and_sign_invariant_geodesic() {
        let target = identity();
        let mut actual = target;
        actual.translation_m[0] = METRES;
        assert!(endpoint(actual, target).is_ok());
        actual.translation_m[0] = f64::from_bits(METRES.to_bits() + 1);
        assert!(endpoint(actual, target).is_err());
        actual.translation_m = [0.0015, 0.0015, 0.0];
        assert!(endpoint(actual, target).is_err());
        actual = target;
        actual.quaternion_xyzw[3] = -1.0;
        assert_eq!(endpoint(actual, target).unwrap(), (0.0, 0.0));
        for (scale, pass) in [(0.99999, true), (1.00001, false)] {
            let half = RADIANS * scale / 2.0;
            actual.quaternion_xyzw = [0.0, 0.0, half.sin(), half.cos()];
            assert_eq!(endpoint(actual, target).is_ok(), pass);
        }
        actual.quaternion_xyzw = [0.0; 4];
        assert!(endpoint(actual, target).is_err());
        actual = target;
        actual.translation_m[0] = f64::NAN;
        assert!(endpoint(actual, target).is_err());
    }

    #[test]
    fn evidence_publication_preserves_existing_bytes_and_removes_temporary_files() {
        let directory = std::env::temp_dir().join(format!(
            "invar-planner-immutable-{}-{}",
            std::process::id(),
            nonce().unwrap()
        ));
        let path = directory.join("attempt.json");
        let record = json!({"status":"error","message":"retained failure"});
        immutable(&path, &record).unwrap();
        let original = std::fs::read(&path).unwrap();
        immutable(&path, &record).unwrap();
        assert!(immutable(&path, &json!({"status":"pass"})).is_err());
        assert_eq!(std::fs::read(&path).unwrap(), original);
        assert_eq!(std::fs::read_dir(&directory).unwrap().count(), 1);
        assert!(immutable(&directory.join("too-large.json"), &json!("x".repeat(65536))).is_err());
        assert_eq!(std::fs::read_dir(&directory).unwrap().count(), 1);
        std::fs::remove_dir_all(directory).unwrap();
    }
}
