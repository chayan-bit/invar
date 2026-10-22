//! Untrusted, bounded schedule hints. Executable device artifacts never enter here.

use std::{
    fs,
    io::{Read, Write},
    path::PathBuf,
};

use crate::schedule::tuning::{Candidate, MAX_VARIANTS, Refusal, SCHEDULE_VERSION, Schedule};
use invar_device::{Capabilities, Verified, source_hash};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};

use crate::admission::Region;

const SCHEMA: u32 = 1;
const MAX_HINT_BYTES: usize = 32 * 1024;
const MAX_HINT_FILES: usize = 64;
const ABI: &str = "invar-device-ir-v1-u32-owned-completed";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HintState {
    Hit,
    Miss,
    Invalid,
}

fn sha256(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|n| format!("{n:02x}"))
        .collect()
}

/// Build the only accepted hint from fresh source, admitted graph and verified IR.
/// Debug encoding is scoped to pinned Rust 1.95.0; the cache contains no code.
pub fn expected(source: &[u8], region: &Region, plan: &Verified, artifact: &str) -> Value {
    expected_variant(
        source,
        region,
        plan,
        artifact,
        0,
        Schedule::m3(plan.ir().geometry),
    )
}

fn expected_variant(
    source: &[u8],
    region: &Region,
    plan: &Verified,
    artifact: &str,
    index: usize,
    schedule: Schedule,
) -> Value {
    let ir = plan.ir();
    let mut contract = ir.clone();
    // Process-local graph digests are not cache identity. Bind exact source
    // bytes and the structural contract with SHA-256 instead.
    contract.source_identity = 0;
    contract.graph_revision = 0;
    let (backend, toolchain) = match plan.capabilities() {
        Capabilities::Cuda { .. } => ("cuda", "rustc-1.95.0+nvrtc-c++17"),
        Capabilities::Metal { .. } => ("metal", "rustc-1.95.0+msl-3.1"),
    };
    let identity = json!({
        "source_identity": format!("sha256:{}", sha256(source)),
        "typed_contract": format!("{:?}", contract),
        "rules": region.execution().rules(),
        "extraction": {"rule": crate::device_ir::EXTRACTION_RULE,
            "typed_schema": crate::typed_net::SCHEMA_VERSION,
            "physical_ir": invar_device::IR_VERSION},
        "toolchain": toolchain,
        "device": format!("{:?}", plan.capabilities()),
        "layout": format!("{:?}", ir.input.layout),
        "numeric": format!("{:?}", ir.numeric),
        "device_ir_version": plan.device_ir().ir().version,
        "numeric_policy": format!("{:?}", plan.device_ir().ir().numeric),
        "physical_layouts": format!("{:?}", plan.device_ir().ir().buffers),
        "abi": ABI,
        "schedule_version": SCHEDULE_VERSION,
        "variant_index": index,
        "geometry": {"blocks": plan.ir().geometry.blocks, "lanes": plan.ir().geometry.lanes},
        "options": {"tile": schedule.tile, "vector_width": schedule.vector_width,
            "fusion_cutoff": schedule.fusion_cutoff, "frontier_chunk": schedule.frontier_chunk,
            "cpu_threshold": schedule.cpu_threshold,
            "seed": crate::schedule::tuning::SEED,
            "warmups": crate::schedule::tuning::WARMUPS,
            "samples": crate::schedule::tuning::SAMPLES,
            "max_variants": MAX_VARIANTS},
        "artifact_hash": format!("{:016x}", source_hash(artifact)),
    });
    let key = sha256(&serde_json::to_vec(&identity).unwrap_or_default());
    json!({
        "schema_version": SCHEMA,
        "key": key,
        "identity": identity,
        "geometry": {"blocks": ir.geometry.blocks, "lanes": ir.geometry.lanes},
        "backend": backend,
    })
}

/// Regenerate every identity field and the selected physical capability before
/// a hint can be used. A hash, matching key or old Verified is never admission.
pub fn revalidate(
    source: &[u8],
    region: &Region,
    candidate: &Candidate,
    hint: &Value,
) -> Result<Verified, Refusal> {
    let old = candidate.plan();
    let checked = crate::front::parse(source)
        .ok()
        .and_then(|parsed| crate::check::check(parsed, true).ok())
        .ok_or(Refusal::Admission)?;
    let program = crate::mir::lower(&checked).map_err(|_| Refusal::Admission)?;
    let report = crate::admission::analyze(source, &program).map_err(|_| Refusal::Admission)?;
    let region = report
        .regions()
        .iter()
        .find(|fresh| fresh.function() == region.function() && fresh.span() == region.span())
        .ok_or(Refusal::StaleOrForgedHint)?;
    let fresh = crate::schedule::tuning::enumerate(
        region,
        old.ir().length,
        old.capabilities(),
        [candidate.schedule()],
    )?
    .remove(0)
    .into_plan();
    if old.ir() != fresh.ir() || candidate.index() >= MAX_VARIANTS {
        return Err(Refusal::StaleOrForgedHint);
    }
    let artifact = match fresh.capabilities() {
        Capabilities::Cuda { .. } => invar_device::emit_cuda(&fresh),
        Capabilities::Metal { .. } => invar_device::emit_msl(&fresh),
    }
    .map_err(|_| Refusal::Admission)?;
    let expected = expected_variant(
        source,
        region,
        &fresh,
        &artifact,
        candidate.index(),
        candidate.schedule(),
    );
    if !validate(hint, &expected) {
        return Err(Refusal::StaleOrForgedHint);
    }
    Ok(fresh)
}

pub fn expected_candidate(
    source: &[u8],
    region: &Region,
    candidate: &Candidate,
) -> Result<Value, Refusal> {
    let artifact = match candidate.plan().capabilities() {
        Capabilities::Cuda { .. } => invar_device::emit_cuda(candidate.plan()),
        Capabilities::Metal { .. } => invar_device::emit_msl(candidate.plan()),
    }
    .map_err(|_| Refusal::Admission)?;
    Ok(expected_variant(
        source,
        region,
        candidate.plan(),
        &artifact,
        candidate.index(),
        candidate.schedule(),
    ))
}

pub fn validate(value: &Value, expected: &Value) -> bool {
    let Ok(bytes) = serde_json::to_vec(value) else {
        return false;
    };
    bytes.len() <= MAX_HINT_BYTES
        && value.get("schema_version") == Some(&json!(SCHEMA))
        && value == expected
}

fn directory() -> Option<PathBuf> {
    if let Some(path) = std::env::var_os("INVAR_CACHE_DIR") {
        return Some(PathBuf::from(path));
    }
    if let Some(path) = std::env::var_os("LOCALAPPDATA") {
        return Some(PathBuf::from(path).join("Invar").join("cache"));
    }
    if let Some(path) = std::env::var_os("XDG_CACHE_HOME") {
        return Some(PathBuf::from(path).join("invar"));
    }
    std::env::var_os("HOME").map(|home| PathBuf::from(home).join(".cache").join("invar"))
}

fn path(expected: &Value) -> Option<PathBuf> {
    path_in(&directory()?, expected)
}

fn path_in(directory: &std::path::Path, expected: &Value) -> Option<PathBuf> {
    let key = expected.get("key")?.as_str()?;
    if key.len() != 64 || !key.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return None;
    }
    let backend = expected.get("backend")?.as_str()?;
    if !matches!(backend, "cuda" | "metal") {
        return None;
    }
    Some(directory.join(format!("{backend}-hint-{key}.json")))
}

pub fn load(expected: &Value) -> HintState {
    let Some(path) = path(expected) else {
        return HintState::Miss;
    };
    load_path(path, expected)
}

pub fn load_in(directory: &std::path::Path, expected: &Value) -> HintState {
    match path_in(directory, expected) {
        Some(path) => load_path(path, expected),
        None => HintState::Invalid,
    }
}

fn load_path(path: PathBuf, expected: &Value) -> HintState {
    let file = match fs::File::open(path) {
        Ok(file) => file,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return HintState::Miss,
        Err(_) => return HintState::Invalid,
    };
    let mut bytes = Vec::new();
    if file
        .take((MAX_HINT_BYTES + 1) as u64)
        .read_to_end(&mut bytes)
        .is_err()
    {
        return HintState::Invalid;
    }
    if bytes.len() > MAX_HINT_BYTES {
        return HintState::Invalid;
    }
    match serde_json::from_slice::<Value>(&bytes) {
        Ok(value) if validate(&value, expected) => HintState::Hit,
        _ => HintState::Invalid,
    }
}

/// Replace only this derived hint path. An invalid hint is discarded and rebuilt.
pub fn save(expected: &Value) -> std::io::Result<()> {
    let path = path(expected).ok_or_else(|| std::io::Error::other("cache path unavailable"))?;
    let bytes = serde_json::to_vec(expected)?;
    if bytes.len() > MAX_HINT_BYTES {
        return Err(std::io::Error::other("schedule hint too large"));
    }
    let parent = path
        .parent()
        .ok_or_else(|| std::io::Error::other("cache parent unavailable"))?;
    fs::create_dir_all(parent)?;
    if !path.exists() {
        let entries = fs::read_dir(parent)?
            .take(MAX_HINT_FILES * 4)
            .collect::<std::io::Result<Vec<_>>>()?;
        if entries.len() == MAX_HINT_FILES * 4 {
            return Err(std::io::Error::other(
                "schedule hint directory is over quota",
            ));
        }
        let backend = expected
            .get("backend")
            .and_then(Value::as_str)
            .unwrap_or("cuda");
        let prefix = format!("{backend}-hint-");
        let mut owned = entries
            .into_iter()
            .filter(|entry| {
                let name = entry.file_name();
                let name = name.to_string_lossy();
                name.starts_with(&prefix)
                    && name.ends_with(".json")
                    && entry.file_type().is_ok_and(|kind| kind.is_file())
            })
            .map(|entry| entry.path())
            .collect::<Vec<_>>();
        // ponytail: deterministic eviction may churn hot hints; use LRU if measured misses matter.
        owned.sort();
        let excess = owned.len().saturating_sub(MAX_HINT_FILES - 1);
        for stale in owned.into_iter().take(excess) {
            fs::remove_file(stale)?;
        }
    }
    let temporary = path.with_extension(format!("{}.tmp", std::process::id()));
    let mut file = fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&temporary)?;
    file.write_all(&bytes)?;
    file.sync_all()?;
    drop(file);
    if path.exists() {
        fs::remove_file(&path)?;
    }
    fs::rename(temporary, path)
}
