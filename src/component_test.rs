//! The public component-v1 contract harness and immutable passing evidence writer.

use invar::hex::hex;
use std::ffi::OsString;
use std::path::{Path, PathBuf};

use invar::{front, system};
use invar_foreign::component::{Component, ComponentError, Pose, State};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};

mod planner;

#[derive(Debug)]
struct Failure {
    code: &'static str,
    message: String,
    details: Option<Value>,
}

impl Failure {
    fn new(code: &'static str, message: impl Into<String>) -> Self {
        Self {
            code,
            message: message.into(),
            details: None,
        }
    }
}

fn fail(message: impl Into<String>) -> Failure {
    Failure::new("S1400", message)
}

fn foreign(error: ComponentError) -> Failure {
    let code = match error {
        ComponentError::Library | ComponentError::Symbol | ComponentError::Version => "S1401",
        ComponentError::Lifecycle | ComponentError::Status(2) => "S1402",
        ComponentError::Input | ComponentError::Status(1) => "S1404",
        ComponentError::Status(4) | ComponentError::Status(5) => "S1406",
        ComponentError::Status(6) | ComponentError::Protocol => "S1407",
        ComponentError::Output => "S1403",
        ComponentError::Status(_) => "S1401",
    };
    Failure::new(code, format!("foreign component: {error:?}"))
}

fn string<'a>(value: &'a Value, key: &str) -> Result<&'a str, Failure> {
    value[key]
        .as_str()
        .ok_or_else(|| fail(format!("missing {key}")))
}

fn numbers<const N: usize>(value: &Value, key: &str) -> Result<[f64; N], Failure> {
    let array = value[key]
        .as_array()
        .ok_or_else(|| fail(format!("missing {key}")))?;
    if array.len() != N {
        return Err(fail(format!("{key} must contain {N} values")));
    }
    let mut result = [0.0; N];
    for (i, slot) in result.iter_mut().enumerate() {
        let number = array[i]
            .as_f64()
            .ok_or_else(|| fail(format!("invalid {key}[{i}]")))?;
        if !number.is_finite() {
            return Err(fail(format!("non-finite {key}[{i}]")));
        }
        *slot = number;
    }
    Ok(result)
}

fn file_hash(path: &Path) -> Result<String, Failure> {
    use std::io::Read;
    const LIMIT: u64 = 64 * 1024 * 1024;
    let metadata = path.metadata().map_err(|error| fail(error.to_string()))?;
    if !metadata.is_file() || metadata.len() > LIMIT {
        return Err(fail("runtime file exceeds 64 MiB"));
    }
    let mut file = std::fs::File::open(path)
        .map_err(|error| fail(error.to_string()))?
        .take(LIMIT + 1);
    let mut hasher = Sha256::new();
    let mut buffer = [0; 8192];
    let mut count = 0;
    loop {
        let size = file
            .read(&mut buffer)
            .map_err(|error| fail(error.to_string()))?;
        if size == 0 {
            break;
        }
        count += size as u64;
        if count > LIMIT {
            return Err(fail("runtime file exceeds 64 MiB"));
        }
        hasher.update(&buffer[..size]);
    }
    Ok(hex(&hasher.finalize()))
}

fn expected_pose(vector: &Value, planar: bool) -> Result<Pose, Failure> {
    if planar {
        let yaw = vector["yaw_rad"]
            .as_f64()
            .ok_or_else(|| fail("missing yaw_rad"))?;
        let x = vector["x_m"].as_f64().ok_or_else(|| fail("missing x_m"))?;
        let y = vector["y_m"].as_f64().ok_or_else(|| fail("missing y_m"))?;
        if !yaw.is_finite() || !x.is_finite() || !y.is_finite() {
            return Err(fail("non-finite golden pose"));
        }
        Ok(Pose {
            translation_m: [x, y, 0.0],
            quaternion_xyzw: [0.0, 0.0, (yaw / 2.0).sin(), (yaw / 2.0).cos()],
        })
    } else if vector.get("quaternion_wxyz").is_some() {
        let translation_m = numbers::<3>(vector, "translation_m")?;
        let q = numbers::<4>(vector, "quaternion_wxyz")?;
        Ok(Pose {
            translation_m,
            quaternion_xyzw: [q[1], q[2], q[3], q[0]],
        })
    } else {
        Ok(Pose {
            translation_m: numbers::<3>(vector, "translation_m")?,
            quaternion_xyzw: numbers::<4>(vector, "quaternion_xyzw")?,
        })
    }
}

fn check_pose(actual: Pose, expected: Pose, metres: f64, radians: f64) -> Result<(), Failure> {
    for i in 0..3 {
        if (actual.translation_m[i] - expected.translation_m[i]).abs() > metres {
            return Err(Failure::new(
                "S1403",
                format!("translation[{i}] exceeds {metres} m"),
            ));
        }
    }
    let difference: f64 = actual
        .quaternion_xyzw
        .iter()
        .zip(expected.quaternion_xyzw)
        .map(|(a, b)| (a - b) * (a - b))
        .sum::<f64>()
        .sqrt();
    let opposite: f64 = actual
        .quaternion_xyzw
        .iter()
        .zip(expected.quaternion_xyzw)
        .map(|(a, b)| (a + b) * (a + b))
        .sum::<f64>()
        .sqrt();
    let angular_error = 4.0 * difference.min(opposite).atan2(difference.max(opposite));
    if !angular_error.is_finite() || angular_error > radians {
        return Err(Failure::new(
            "S1403",
            format!("rotation exceeds {radians} rad"),
        ));
    }
    Ok(())
}

fn tolerances(manifest: &Value) -> Result<(f64, f64), Failure> {
    let entries = manifest["golden"][0]["tolerances"]
        .as_array()
        .ok_or_else(|| fail("missing tolerances"))?;
    let mut metres = None;
    let mut radians = None;
    for entry in entries {
        let value = entry["value"]
            .as_str()
            .ok_or_else(|| fail("invalid tolerance"))?
            .parse::<f64>()
            .map_err(|_| fail("invalid tolerance"))?;
        if !value.is_finite() || value <= 0.0 {
            return Err(fail("invalid tolerance"));
        }
        match string(entry, "unit")? {
            "m" if metres.replace(value).is_none() => {}
            "rad" if radians.replace(value).is_none() => {}
            _ => return Err(fail("duplicate or unsupported tolerance unit")),
        }
    }
    Ok((
        metres.ok_or_else(|| fail("missing m tolerance"))?,
        radians.ok_or_else(|| fail("missing rad tolerance"))?,
    ))
}

fn fixture_vectors<'a>(
    golden: &'a Value,
    planar: bool,
    arm: &str,
) -> Result<&'a Vec<Value>, Failure> {
    if planar {
        return golden["vectors"]
            .as_array()
            .ok_or_else(|| fail("missing golden vectors"));
    }
    if let Some(arms) = golden["arms"].as_array() {
        return arms
            .iter()
            .find(|entry| entry["arm"] == arm)
            .ok_or_else(|| fail(format!("{arm} goldens missing")))?["vectors"]
            .as_array()
            .ok_or_else(|| fail(format!("missing {arm} vectors")));
    }
    golden["vectors"]
        .as_array()
        .ok_or_else(|| fail("missing golden vectors"))
}

fn run(
    path: &Path,
    library: Option<&Path>,
    output: &Path,
    model: Option<&Path>,
    child: Option<&Path>,
) -> Result<Value, Failure> {
    let path = path
        .canonicalize()
        .map_err(|error| fail(error.to_string()))?;
    let source = std::fs::read(&path).map_err(|error| fail(error.to_string()))?;
    let manifest = front::parse_system(&source)
        .and_then(system::compile_manifest)
        .map_err(|error| fail(error.message))?;
    let header = system::foreign::header(&manifest).map_err(fail)?;
    let prefix = string(&manifest["abi"], "prefix")?;
    let default_library = path
        .parent()
        .ok_or_else(|| fail("manifest has no parent"))?
        .join(format!("lib{prefix}.{}", std::env::consts::DLL_EXTENSION));
    let library = library
        .unwrap_or(&default_library)
        .canonicalize()
        .map_err(|error| Failure::new("S1401", error.to_string()))?;
    let root = path
        .ancestors()
        .find(|ancestor| ancestor.join(".git").exists())
        .or_else(|| path.parent())
        .ok_or_else(|| fail("manifest has no source root"))?;
    let (digest, inputs) =
        system::foreign::manifest_hash(root, &path, &manifest, &header).map_err(fail)?;
    if std::fs::read(&path).map_err(|error| fail(error.to_string()))? != source {
        return Err(Failure::new(
            "S1405",
            "manifest changed before contract test",
        ));
    }
    let library_hash = file_hash(&library)?;
    let model_hash = model.map(file_hash).transpose()?;
    let child_hash = child.map(file_hash).transpose()?;
    let name = string(&manifest, "name")?;
    let planar = manifest["config"]["layout"] == "inv_planar2_config_v1";
    let input_layout = manifest["ports"]
        .as_array()
        .and_then(|ports| ports.iter().find(|port| port["direction"] == "in"))
        .and_then(|port| port["layout"].as_str())
        .ok_or_else(|| fail("missing joint input layout"))?;
    let joint_count = match (planar, input_layout) {
        (true, "inv_joint2_f64") => 2,
        (false, "inv_joint6_f64") => 6,
        (false, "inv_joint7_f64") => 7,
        _ => return Err(fail("joint layout disagrees with config")),
    };
    let arm = if joint_count == 6 { "ur5e" } else { "panda" };
    let process = manifest["isolation"] == "separate-process";
    if process != child.is_some() {
        return Err(fail(
            "separate-process requires --child; in-process forbids it",
        ));
    }
    let (metres, radians) = tolerances(&manifest)?;
    let golden_path = path
        .parent()
        .ok_or_else(|| fail("missing manifest parent"))?
        .join(string(&manifest["golden"][0], "path")?);
    let golden: Value = serde_json::from_slice(
        &std::fs::read(golden_path).map_err(|error| fail(error.to_string()))?,
    )
    .map_err(|error| fail(error.to_string()))?;
    if let Some(expected_model) = golden["arms"]
        .as_array()
        .and_then(|arms| arms.iter().find(|entry| entry["arm"] == arm))
        .and_then(|arm| arm["model_sha256"].as_str())
        && model_hash.as_deref() != Some(expected_model)
    {
        return Err(Failure::new(
            "S1405",
            "model bytes disagree with golden provenance",
        ));
    }
    let vectors = fixture_vectors(&golden, planar, arm)?;
    if vectors.is_empty() || vectors.len() > 4096 {
        return Err(fail("golden vector count must be 1..4096"));
    }
    let mut component = Component::open(&library, prefix, joint_count).map_err(foreign)?;
    if component.step(&vec![0.0; joint_count]) != Err(ComponentError::Lifecycle) {
        return Err(Failure::new("S1402", "step before init ran"));
    }
    component.init(child).map_err(foreign)?;
    if component.step(&vec![0.0; joint_count]) != Err(ComponentError::Lifecycle) {
        return Err(Failure::new("S1402", "step before configure ran"));
    }
    if planar {
        let lengths = numbers::<2>(&golden, "lengths_m")?;
        component
            .configure_planar2(lengths[0], lengths[1])
            .map_err(foreign)?;
    } else {
        let model = model.ok_or_else(|| fail("FK component requires --model"))?;
        component
            .configure_fk(model, joint_count as u32)
            .map_err(foreign)?;
    }
    if component.state() != State::Configured {
        return Err(Failure::new(
            "S1402",
            "configure did not enter configured state",
        ));
    }
    if component.step(&vec![0.0; joint_count - 1]) != Err(ComponentError::Input)
        || component.step(&vec![f64::NAN; joint_count]) != Err(ComponentError::Input)
    {
        return Err(Failure::new(
            "S1404",
            "out-of-contract joint input was accepted",
        ));
    }
    for (index, vector) in vectors.iter().enumerate() {
        let q = vector["q_rad"]
            .as_array()
            .ok_or_else(|| fail("missing q_rad"))?
            .iter()
            .map(|v| v.as_f64().ok_or_else(|| fail("invalid q_rad")))
            .collect::<Result<Vec<_>, _>>()?;
        let actual = component.step(&q).map_err(foreign)?;
        check_pose(actual, expected_pose(vector, planar)?, metres, radians).map_err(
            |mut failure| {
                failure.message = format!("golden vector {index}: {}", failure.message);
                failure
            },
        )?;
    }
    // Fixed-seed property samples stay within declared joint limits when available.
    let limits = golden["arms"]
        .as_array()
        .and_then(|arms| arms.iter().find(|entry| entry["arm"] == arm))
        .and_then(|arm| arm["joint_limits_rad"].as_array());
    let mut rng = 0x494e_5641_5253_3132_u64;
    for _ in 0..16 {
        let mut q = Vec::with_capacity(joint_count);
        for i in 0..joint_count {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            let unit = (rng >> 11) as f64 / ((1_u64 << 53) as f64);
            let (low, high) = limits
                .and_then(|items| items.get(i))
                .and_then(|item| item.as_array())
                .and_then(|pair| pair.first().zip(pair.get(1)))
                .and_then(|(a, b)| a.as_f64().zip(b.as_f64()))
                .unwrap_or((-1.0, 1.0));
            q.push(low + unit * (high - low));
        }
        let pose = component.step(&q).map_err(foreign)?;
        let norm: f64 = pose.quaternion_xyzw.iter().map(|v| v * v).sum();
        if (norm.sqrt() - 1.0).abs() > radians {
            return Err(Failure::new(
                "S1403",
                "rotation unit norm property violated",
            ));
        }
    }
    component.shutdown().map_err(foreign)?;
    if component.step(&vec![0.0; joint_count]) != Err(ComponentError::Lifecycle) {
        return Err(Failure::new("S1402", "step after shutdown ran"));
    }
    let (after, _) =
        system::foreign::manifest_hash(root, &path, &manifest, &header).map_err(fail)?;
    if after != digest {
        return Err(Failure::new(
            "S1405",
            "manifest inputs changed during contract test",
        ));
    }
    if std::fs::read(&path).map_err(|error| fail(error.to_string()))? != source {
        return Err(Failure::new(
            "S1405",
            "manifest changed during contract test",
        ));
    }
    if library_hash != file_hash(&library)?
        || model_hash != model.map(file_hash).transpose()?
        || child_hash != child.map(file_hash).transpose()?
    {
        return Err(Failure::new(
            "S1405",
            "runtime input changed during contract test",
        ));
    }
    let record = json!({
        "schema": "invar-component-evidence-v1",
        "component": name, "manifest_hash": digest, "library_version": component.version(),
        "library_sha256": library_hash, "model_sha256": model_hash, "child_sha256": child_hash,
        "host": {"os": std::env::consts::OS, "arch": std::env::consts::ARCH},
        "inputs": inputs.iter().map(|(role,path,sha)| json!({"role":role,"path":path,"sha256":sha})).collect::<Vec<_>>(),
        "results": {"layout": "pass", "lifecycle": "pass", "golden_vectors": vectors.len(), "property_samples": 16, "invalid_inputs": "pass", "process": if process {"pass"} else {"not_applicable"}},
        "status": "pass"
    });
    std::fs::create_dir_all(output).map_err(|error| fail(error.to_string()))?;
    let evidence = output.join(format!("{name}-{digest}.json"));
    let bytes = serde_json::to_vec_pretty(&record).map_err(|error| fail(error.to_string()))?;
    let temporary = output.join(format!(".{name}-{digest}.{}.tmp", std::process::id()));
    let write = (|| {
        let mut file = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&temporary)?;
        std::io::Write::write_all(&mut file, &bytes)?;
        file.sync_all()?;
        std::fs::hard_link(&temporary, &evidence)
    })();
    let _ = std::fs::remove_file(&temporary);
    match write {
        Ok(()) => {}
        Err(error)
            if error.kind() == std::io::ErrorKind::AlreadyExists
                && std::fs::read(&evidence).map_err(|e| fail(e.to_string()))? == bytes => {}
        Err(error) => return Err(fail(error.to_string())),
    }
    Ok(
        json!({"status":"ok", "evidence":evidence, "manifest_hash":digest, "golden_vectors":vectors.len()}),
    )
}

pub fn command(args: &[OsString]) -> i32 {
    let mut library: Option<PathBuf> = None;
    let mut output: Option<PathBuf> = None;
    let mut model: Option<PathBuf> = None;
    let mut child: Option<PathBuf> = None;
    let mut planner_runtime: Option<PathBuf> = None;
    let Some(manifest) = args.first() else {
        return super::usage();
    };
    let mut options = args[1..].chunks_exact(2);
    if !options.remainder().is_empty() {
        return super::usage();
    }
    for pair in &mut options {
        let value = PathBuf::from(&pair[1]);
        match pair[0].to_str() {
            Some("--library") if library.is_none() => library = Some(value),
            Some("--output") if output.is_none() => output = Some(value),
            Some("--model") if model.is_none() => model = Some(value),
            Some("--child") if child.is_none() => child = Some(value),
            Some("--planner-runtime") if planner_runtime.is_none() => planner_runtime = Some(value),
            _ => return super::usage(),
        }
    }
    let path = Path::new(manifest);
    let Some(parent) = path.parent() else {
        return super::usage();
    };
    let output = output.unwrap_or_else(|| parent.join("evidence"));
    let result = if let Some(runtime) = planner_runtime {
        planner::run(
            path,
            &runtime,
            library.as_deref(),
            &output,
            model.as_deref(),
            child.as_deref(),
        )
    } else {
        run(
            path,
            library.as_deref(),
            &output,
            model.as_deref(),
            child.as_deref(),
        )
    };
    match result {
        Ok(result) => {
            println!("{result}");
            0
        }
        Err(error) => {
            eprintln!(
                "{}",
                json!({"schema":"invar-component-test-v1","status":"error","code":error.code,"message":error.message})
            );
            2
        }
    }
}
