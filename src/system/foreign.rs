//! Pure, bounded C-header generation from a checked foreign manifest IR.

use crate::hex::hex;
use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write;
use std::path::Path;

use serde_json::Value;
use sha2::{Digest, Sha256};

use super::{c_identifier, foreign_config_layout, manifest_dependency_path};

mod planner;

const MAX_PORTS: usize = 16;
const MAX_HEADER_BYTES: usize = 64 * 1024;
const MAX_INPUT_BYTES: usize = 16 * 1024 * 1024;
const MAX_TOTAL_INPUT_BYTES: usize = 64 * 1024 * 1024;

pub type ManifestSnapshot = (String, Vec<(String, String, String)>);

fn port_c_type(layout: &str) -> Option<(&'static str, usize)> {
    match layout {
        "inv_joint2_f64" => Some(("joint2_f64", 16)),
        "inv_joint6_f64" => Some(("joint6_f64", 48)),
        "inv_joint7_f64" => Some(("joint7_f64", 56)),
        "inv_pose_f64" => Some(("pose_f64", 56)),
        _ => None,
    }
}

fn valid_port_type(ty: &str, layout: &str) -> bool {
    matches!(
        (ty, layout),
        ("std::robotics::JointPositions<f64,2>", "inv_joint2_f64")
            | ("std::robotics::JointPositions<f64,6>", "inv_joint6_f64")
            | ("std::robotics::JointPositions<f64,7>", "inv_joint7_f64")
    ) || (ty.starts_with("std::robotics::Pose<f64,")
        && ty.ends_with('>')
        && layout == "inv_pose_f64")
}

/// Generate the public, versioned adapter header. The adapter deliberately has
/// a distinct ABI from hand-authored Pinocchio and process reference shims.
pub fn header(manifest: &Value) -> Result<String, &'static str> {
    let prefix = manifest["abi"]["prefix"]
        .as_str()
        .ok_or("missing C ABI prefix")?;
    if !c_identifier(prefix) || manifest["abi"]["kind"] != "c" {
        return Err("invalid C ABI prefix");
    }
    let config = &manifest["config"];
    let config_name = config["layout"].as_str().ok_or("missing typed config")?;
    if foreign_config_layout(config_name).as_ref() != Some(config) {
        return Err("unverified C config layout");
    }
    if config_name == "inv_planner_config_v1" {
        return planner::header(manifest);
    }
    let ports = manifest["ports"].as_array().ok_or("missing ports")?;
    if ports.is_empty() || ports.len() > MAX_PORTS {
        return Err("foreign port count limit");
    }
    let mut inputs = Vec::new();
    let mut outputs = Vec::new();
    let mut layouts = BTreeSet::new();
    let mut names = BTreeSet::new();
    for port in ports {
        let name = port["name"].as_str().ok_or("port name missing")?;
        let direction = port["direction"].as_str().ok_or("port direction missing")?;
        let ty = port["type"].as_str().ok_or("port type missing")?;
        let layout = port["layout"].as_str().ok_or("port layout missing")?;
        if !c_identifier(name) || !names.insert(name) || !valid_port_type(ty, layout) {
            return Err("foreign port C layout mismatch");
        }
        layouts.insert(layout);
        match direction {
            "in" => inputs.push((name, layout)),
            "out" => outputs.push((name, layout)),
            _ => return Err("invalid foreign port direction"),
        }
    }
    if inputs.len() != 1 || outputs.len() != 1 || outputs[0].1 != "inv_pose_f64" {
        return Err("component-v1 FK adapter needs one joint input and one pose output");
    }
    let mut out = String::new();
    let guard = format!("INVAR_{}_COMPONENT_V1_H", prefix.to_ascii_uppercase());
    writeln!(
        out,
        "#ifndef {guard}\n#define {guard}\n#include <stddef.h>\n#include <stdint.h>"
    )
    .map_err(|_| "header write failed")?;
    out.push_str("#if UINTPTR_MAX != UINT64_MAX\n#error \"Invar component v1 needs a 64-bit C target\"\n#endif\n");
    out.push_str("#ifdef __cplusplus\n#define INVAR_COMPONENT_ASSERT static_assert\n#define INVAR_COMPONENT_ALIGNOF alignof\nextern \"C\" {\n#else\n#define INVAR_COMPONENT_ASSERT _Static_assert\n#define INVAR_COMPONENT_ALIGNOF _Alignof\n#endif\n");
    writeln!(out, "typedef struct {prefix}_context {prefix}_context;")
        .map_err(|_| "header write failed")?;
    for layout in layouts {
        let (suffix, size) = port_c_type(layout).ok_or("unknown foreign layout")?;
        if layout == "inv_pose_f64" {
            writeln!(out, "typedef struct {{ double translation_m[3]; double quaternion_xyzw[4]; }} {prefix}_{suffix};")
                .map_err(|_| "header write failed")?;
            writeln!(out, "INVAR_COMPONENT_ASSERT(offsetof({prefix}_{suffix}, quaternion_xyzw) == 24, \"pose quaternion offset\");")
                .map_err(|_| "header write failed")?;
        } else {
            let joints = size / 8;
            writeln!(
                out,
                "typedef struct {{ double values[{joints}]; }} {prefix}_{suffix};"
            )
            .map_err(|_| "header write failed")?;
        }
        writeln!(out, "INVAR_COMPONENT_ASSERT(sizeof({prefix}_{suffix}) == {size} && INVAR_COMPONENT_ALIGNOF({prefix}_{suffix}) == 8, \"{layout} layout\");")
            .map_err(|_| "header write failed")?;
    }
    match config_name {
        "inv_fk_config_v1" => {
            writeln!(out, "typedef struct {{ const uint8_t *model_xml_path; uint32_t model_xml_path_len; uint32_t model_kind; }} {prefix}_config;")
                .map_err(|_| "header write failed")?;
            writeln!(out, "INVAR_COMPONENT_ASSERT(sizeof({prefix}_config) == 16 && INVAR_COMPONENT_ALIGNOF({prefix}_config) == 8 && offsetof({prefix}_config, model_xml_path_len) == 8 && offsetof({prefix}_config, model_kind) == 12, \"config layout\");")
                .map_err(|_| "header write failed")?;
        }
        "inv_planar2_config_v1" => {
            writeln!(
                out,
                "typedef struct {{ double proximal_m; double distal_m; }} {prefix}_config;"
            )
            .map_err(|_| "header write failed")?;
            writeln!(out, "INVAR_COMPONENT_ASSERT(sizeof({prefix}_config) == 16 && INVAR_COMPONENT_ALIGNOF({prefix}_config) == 8 && offsetof({prefix}_config, distal_m) == 8, \"config layout\");")
                .map_err(|_| "header write failed")?;
        }
        _ => return Err("unknown C config layout"),
    }
    for (direction, fields) in [("inputs", &inputs), ("outputs", &outputs)] {
        writeln!(out, "typedef struct {{").map_err(|_| "header write failed")?;
        for (name, layout) in fields {
            let (suffix, _) = port_c_type(layout).ok_or("unknown port C layout")?;
            writeln!(out, "    {prefix}_{suffix} {name};").map_err(|_| "header write failed")?;
        }
        writeln!(out, "}} {prefix}_{direction};").map_err(|_| "header write failed")?;
    }
    let status_prefix = prefix.to_ascii_uppercase();
    writeln!(out, "enum {{ {status_prefix}_OK = 0, {status_prefix}_INVALID_INPUT = 1, {status_prefix}_LIFECYCLE = 2, {status_prefix}_FAILURE = 3, {status_prefix}_CHILD_EXIT = 4, {status_prefix}_TIMEOUT = 5, {status_prefix}_MALFORMED = 6 }};")
        .map_err(|_| "header write failed")?;
    out.push_str("/* version returns UTF-8 `invar-component-v1/<library-version>`.\n   init starts with *out_context == NULL and leaves it NULL on failure.\n   init takes an absolute child path only for separate-process isolation; in-process uses NULL,0.\n   configure borrows config pointer fields for this call only.\n   step leaves output bytes unchanged on every nonzero status.\n   shutdown clears *context and releases exactly that owner. */\n");
    writeln!(out, "const char *{prefix}_version(void);").map_err(|_| "header write failed")?;
    writeln!(out, "int32_t {prefix}_init({prefix}_context **out_context, const uint8_t *launch_path, uint32_t launch_path_len);")
        .map_err(|_| "header write failed")?;
    writeln!(
        out,
        "int32_t {prefix}_configure({prefix}_context *context, const {prefix}_config *config);"
    )
    .map_err(|_| "header write failed")?;
    writeln!(out, "int32_t {prefix}_step({prefix}_context *context, const {prefix}_inputs *input, {prefix}_outputs *output);")
        .map_err(|_| "header write failed")?;
    writeln!(
        out,
        "int32_t {prefix}_shutdown({prefix}_context **context);"
    )
    .map_err(|_| "header write failed")?;
    out.push_str("#ifdef __cplusplus\n}\n#endif\n#undef INVAR_COMPONENT_ASSERT\n#undef INVAR_COMPONENT_ALIGNOF\n");
    writeln!(out, "#endif /* {guard} */").map_err(|_| "header write failed")?;
    if out.len() > MAX_HEADER_BYTES {
        return Err("generated header byte limit");
    }
    Ok(out)
}

/// Hash the declared contract-input snapshot with role/path/length framing.
/// Callers must verify the same digest after execution before recording success.
pub fn manifest_hash(
    allowed_root: &Path,
    path: &Path,
    manifest: &Value,
    generated_header: &str,
) -> Result<ManifestSnapshot, String> {
    let allowed_root = allowed_root
        .canonicalize()
        .map_err(|error| error.to_string())?;
    if !allowed_root.is_dir() {
        return Err("allowed source root is not a directory".into());
    }
    let path = path.canonicalize().map_err(|error| error.to_string())?;
    if !path.starts_with(&allowed_root) || !path.is_file() {
        return Err("manifest escapes allowed source root".into());
    }
    let directory = path.parent().ok_or("manifest directory missing")?;
    let mut files = Vec::new();
    let shim = manifest["shim"]
        .as_str()
        .ok_or("manifest shim path missing")?;
    files.push(("shim", shim));
    let golden = manifest["golden"]
        .as_array()
        .ok_or("manifest golden list missing")?;
    for entry in golden {
        let name = entry["path"]
            .as_str()
            .ok_or("manifest golden path missing")?;
        files.push(("golden", name));
    }
    let declared = manifest["inputs"]
        .as_array()
        .ok_or("manifest additional-input list missing")?;
    for entry in declared {
        files.push((
            "input",
            entry.as_str().ok_or("manifest input path missing")?,
        ));
    }
    if files.len() > 64 {
        return Err("manifest input file limit".into());
    }
    let mut inputs = BTreeMap::new();
    let mut total = 0usize;
    for (role, name) in files {
        if !manifest_dependency_path(name) {
            return Err("manifest input path must be relative".into());
        }
        let resolved = directory
            .join(name)
            .canonicalize()
            .map_err(|error| error.to_string())?;
        if !resolved.starts_with(&allowed_root)
            || !resolved.is_file()
            || resolved == path
            || inputs.contains_key(&resolved)
        {
            return Err(
                "manifest input escapes root, duplicates another input, or is not a file".into(),
            );
        }
        let metadata = resolved.metadata().map_err(|error| error.to_string())?;
        if metadata.len() > MAX_INPUT_BYTES as u64 {
            return Err("manifest input exceeds byte limit".into());
        }
        let bytes = std::fs::read(&resolved).map_err(|error| error.to_string())?;
        if bytes.len() > MAX_INPUT_BYTES {
            return Err("manifest input exceeds byte limit".into());
        }
        total = total.saturating_add(bytes.len());
        if total > MAX_TOTAL_INPUT_BYTES {
            return Err("manifest inputs exceed total byte limit".into());
        }
        let normalized = resolved
            .strip_prefix(&allowed_root)
            .map_err(|error| error.to_string())?
            .to_str()
            .ok_or("non-UTF-8 manifest input path")?
            .replace('\\', "/");
        inputs.insert(resolved, (role.to_owned(), normalized, bytes));
    }
    let generated_name = format!(
        "{}.h",
        manifest["abi"]["prefix"]
            .as_str()
            .ok_or("missing ABI prefix")?
    );
    // A planner contract binds an independently declared FK manifest. Its
    // generated ABI header is a virtual dependency, just like this component's
    // own header; capture those bytes instead of requiring a mutable shared
    // six/seven-joint header beside the FK source. Legacy non-planner snapshots
    // retain their original byte framing and include rules.
    let mut dependency_headers = BTreeMap::new();
    if manifest["config"]["layout"] == "inv_planner_config_v1" {
        for (file, (_, _, bytes)) in &inputs {
            if file.extension().and_then(|p| p.to_str()) != Some("inv") {
                continue;
            }
            if let Ok(dependency) =
                crate::front::parse_system(bytes).and_then(crate::system::compile_manifest)
            {
                let prefix = dependency["abi"]["prefix"]
                    .as_str()
                    .ok_or("dependency ABI prefix missing")?;
                let name = format!("{prefix}.h");
                let bytes = header(&dependency).map_err(str::to_owned)?;
                if (name == generated_name && bytes != generated_header)
                    || dependency_headers
                        .get(&name)
                        .is_some_and(|previous| previous != &bytes)
                {
                    return Err("declared dependency ABI header conflict".into());
                }
                dependency_headers.insert(name, bytes);
            }
        }
        if inputs.len() + dependency_headers.len() > 64 {
            return Err("manifest input/header file limit".into());
        }
    }
    for (file, (_, _, bytes)) in &inputs {
        if !matches!(
            file.extension().and_then(|part| part.to_str()),
            Some("c" | "cpp" | "h" | "hpp")
        ) {
            continue;
        }
        let source = std::str::from_utf8(bytes).map_err(|_| "C source is not UTF-8")?;
        for line in source.lines() {
            let Some(rest) = line.trim_start().strip_prefix("#include") else {
                continue;
            };
            let Some(quoted) = rest.trim_start().strip_prefix('"') else {
                continue;
            };
            let name = quoted.split('"').next().ok_or("malformed local include")?;
            if name == generated_name || dependency_headers.contains_key(name) {
                continue;
            }
            let dependency = file
                .parent()
                .ok_or("C source has no parent")?
                .join(name)
                .canonicalize()
                .map_err(|_| "local C include is missing")?;
            if !dependency.starts_with(&allowed_root) || !inputs.contains_key(&dependency) {
                return Err(format!("local C include is not declared: {name}"));
            }
        }
    }
    let source = std::fs::read(&path).map_err(|error| error.to_string())?;
    if source.len() > MAX_INPUT_BYTES || generated_header.len() > MAX_HEADER_BYTES {
        return Err("manifest or generated header exceeds byte limit".into());
    }
    let mut hasher = Sha256::new();
    hasher.update(b"invar-component-manifest-v1\0");
    let mut records = Vec::new();
    let source_name = path
        .strip_prefix(&allowed_root)
        .map_err(|error| error.to_string())?
        .to_str()
        .ok_or("non-UTF-8 manifest path")?
        .replace('\\', "/");
    let mut parts = vec![
        ("manifest".to_owned(), source_name, source),
        (
            "generated-header".to_owned(),
            "<generated>".to_owned(),
            generated_header.as_bytes().to_vec(),
        ),
    ];
    parts.extend(inputs.into_values());
    parts.extend(dependency_headers.into_iter().map(|(name, bytes)| {
        (
            "generated-header".to_owned(),
            format!("<generated:{name}>"),
            bytes.into_bytes(),
        )
    }));
    parts.sort_by(|left, right| (&left.0, &left.1).cmp(&(&right.0, &right.1)));
    for (role, name, bytes) in parts {
        hasher.update((role.len() as u64).to_le_bytes());
        hasher.update(role.as_bytes());
        hasher.update((name.len() as u64).to_le_bytes());
        hasher.update(name.as_bytes());
        hasher.update((bytes.len() as u64).to_le_bytes());
        hasher.update(&bytes);
        records.push((role, name, hex(&Sha256::digest(&bytes))));
    }
    Ok((hex(&hasher.finalize()), records))
}

/// Accept only a passing record for the exact current manifest snapshot.
pub fn verify_evidence(path: &Path, expected_hash: &str) -> Result<(), &'static str> {
    if expected_hash.len() != 64 || !expected_hash.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Err("invalid manifest hash");
    }
    let metadata = path.metadata().map_err(|_| "passing evidence is absent")?;
    if !metadata.is_file() || metadata.len() > 64 * 1024 {
        return Err("evidence is not a bounded JSON file");
    }
    let record: Value =
        serde_json::from_slice(&std::fs::read(path).map_err(|_| "cannot read evidence")?)
            .map_err(|_| "malformed evidence JSON")?;
    if record["schema"] == "invar-planner-contract-evidence-v3" {
        return planner::verify_evidence(&record, expected_hash);
    }
    if record["schema"] != "invar-component-evidence-v1"
        || record["status"] != "pass"
        || record["manifest_hash"] != expected_hash
        || !record["component"].as_str().is_some_and(c_identifier)
        || !record["library_version"].as_str().is_some_and(|version| {
            version.starts_with("invar-component-v1/") && version.len() <= 256
        })
        || !record["library_sha256"].as_str().is_some_and(|hash| {
            hash.len() == 64 && hash.bytes().all(|byte| byte.is_ascii_hexdigit())
        })
        || record["host"]["os"]
            .as_str()
            .is_none_or(|name| name.is_empty())
        || record["host"]["arch"]
            .as_str()
            .is_none_or(|name| name.is_empty())
        || !record["inputs"].as_array().is_some_and(|inputs| {
            !inputs.is_empty()
                && inputs.len() <= 66
                && inputs.iter().all(|input| {
                    input["role"].as_str().is_some_and(|role| {
                        matches!(
                            role,
                            "manifest" | "generated-header" | "shim" | "golden" | "input"
                        )
                    }) && input["path"].as_str().is_some_and(|name| !name.is_empty())
                        && input["sha256"].as_str().is_some_and(|hash| {
                            hash.len() == 64 && hash.bytes().all(|byte| byte.is_ascii_hexdigit())
                        })
                })
        })
        || record["results"]["layout"] != "pass"
        || record["results"]["lifecycle"] != "pass"
        || record["results"]["invalid_inputs"] != "pass"
        || record["results"]["golden_vectors"].as_u64().unwrap_or(0) == 0
        || record["results"]["property_samples"].as_u64().unwrap_or(0) == 0
        || !matches!(
            record["results"]["process"].as_str(),
            Some("pass" | "not_applicable")
        )
    {
        return Err("evidence is stale or incomplete");
    }
    Ok(())
}
