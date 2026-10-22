//! ABI12 stage 2c execution receipt (`invar.gnet.abi12.execution-receipt` v2)
//! for the subprocess CPU group model only. The builder requires the session
//! seal, so only a published `Session<CpuGroupModel>` can create one; there is
//! no constructor from JSON. Every authority and evidence label is written
//! here, never supplied by a caller: CPU modelling is not device execution,
//! `module_receipt_id` is null, and resume and device qualification are false.
//! Qualification remains a separate evidence document.
#![allow(
    dead_code,
    reason = "ABI12 stage 2c S2: read by the session tests now and by the later public host"
)]
use super::{Error, ProvenancedArtifact, compiled, digest, document_digest};
use crate::general_net::cuda_owned_numeric_host::ReceiptSeal;
use crate::hex::hex;
use serde_json::{Value, json};
use std::path::{Path, PathBuf};

/// One hashed toolchain program with the shared libraries its interpreter maps.
pub(crate) struct ProgramIdentity<'a> {
    pub role: &'a str,
    pub path: &'a Path,
    pub sha256: [u8; 32],
    pub libraries: &'a [(PathBuf, [u8; 32])],
}

/// Compile-stage bounds by class (review R1/R2). Only the `hard_*` fields are
/// enforced (rlimits inherited by every compile-stage process, process-group
/// kill and reap on deadline). `modelled_max_processes` describes gcc's
/// sequential pipeline and is NOT enforced. `post_compile_scratch_bytes` is a
/// check after compiling, NOT a peak cap.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct CompileStage {
    pub hard_address_space_bytes: u64,
    pub hard_file_bytes: u64,
    pub hard_probe_deadline_ms: u64,
    pub hard_compile_deadline_ms: u64,
    pub modelled_max_processes: u64,
    pub post_compile_scratch_bytes: u64,
}

/// Toolchain roles, each required exactly once.
const ROLES: [&str; 6] = ["shell", "driver", "cc1plus", "collect2", "as", "ld"];

/// Model identity (D6), as recorded by the backend from bytes it wrote/ran.
pub(crate) struct ModelIdentity<'a> {
    pub compiler_path: &'a Path,
    pub compiler_sha256: [u8; 32],
    pub compiler_version_sha256: [u8; 32],
    pub host_triple: &'a str,
    pub flags: &'a [&'a str],
    pub shim_sha256: [u8; 32],
    pub driver_sha256: [u8; 32],
    pub model_source_sha256: [u8; 32],
    pub executable_sha256: [u8; 32],
    pub kernel_sha256: [u8; 32],
    pub stack_bytes: usize,
    pub child_address_space_bytes: usize,
    pub lanes: usize,
    pub words: [usize; 7],
    pub toolchain: Vec<ProgramIdentity<'a>>,
    pub runtime_static: bool,
    pub runtime_libraries: &'a [(PathBuf, [u8; 32])],
    pub compile_stage: CompileStage,
    /// Steady-state session disk after compiling (post-compile scratch check
    /// plus three hard per-file caps) and its cap; NOT a peak bound (R2).
    pub steady_state_disk_bytes: u64,
    pub steady_state_disk_cap: u64,
    /// Every recorded toolchain file was re-hashed after the compile (R3).
    pub toolchain_reverified: bool,
}

/// Facts of one published run.
pub(crate) struct Run<'a> {
    pub compile: &'a ProvenancedArtifact,
    /// Process-local plan nonce (not a reproducible identity).
    pub plan_nonce: u64,
    /// roots, fuel, transitions, trace events, quantum, max submissions.
    pub config: [u64; 6],
    pub input_sha256: [u8; 32],
    pub initial_sha256: [u8; 32],
    /// sha256 of each completed epoch's device-state file, in order.
    pub epochs: &'a [[u8; 32]],
    /// (status word, sha256 of the 10 terminal packet words) per root.
    pub terminals: &'a [(u64, [u8; 32])],
    pub work_reserved: u64,
    pub work_used: u64,
    pub host_bytes: usize,
    pub model: ModelIdentity<'a>,
}

#[derive(Debug)]
pub(crate) struct ExecutionReceipt {
    id: String,
    manifest: Value,
}
impl ExecutionReceipt {
    pub(crate) fn id(&self) -> &str {
        &self.id
    }
    pub(crate) fn manifest(&self) -> &Value {
        &self.manifest
    }
    /// Exact equality with the receipt built from the live session.
    pub(crate) fn verify_manifest(&self, manifest: &Value) -> Result<(), Error> {
        if self.manifest == *manifest {
            Ok(())
        } else {
            Err(Error::ManifestMismatch)
        }
    }
}

/// Build the receipt of a published CPU-group-model run.
pub(crate) fn cpu_group_model(_seal: ReceiptSeal, run: Run<'_>) -> Result<ExecutionReceipt, Error> {
    let a = run.compile.artifact();
    let block = &run.compile.manifest()["cuda_owned_numeric_continuation"];
    if a.abi_version() != 12
        || a.target() != compiled::Target::Cuda
        || block["schema"] != "invar.gnet.abi12.compile-receipt"
        || block["authority"]
            != json!({"execution": false, "module_load": false, "resume": false,
                "device_qualification": false})
    {
        return Err(Error::ExecutionMismatch);
    }
    // The model compiled exactly the receipt-bound emitted text.
    if hex(&run.model.kernel_sha256) != digest(a.source().as_bytes()) {
        return Err(Error::ArtifactMismatch);
    }
    // F1: the full toolchain is bound, and the runtime claim agrees with the
    // flags (a static executable lists no runtime libraries).
    let m = &run.model;
    let mut roles: Vec<&str> = m.toolchain.iter().map(|p| p.role).collect();
    roles.sort_unstable();
    let mut want = ROLES;
    want.sort_unstable();
    if roles != want
        || m.runtime_static != m.flags.contains(&"-static")
        || (m.runtime_static && !m.runtime_libraries.is_empty())
        || [
            m.compile_stage.hard_address_space_bytes,
            m.compile_stage.hard_file_bytes,
            m.compile_stage.hard_probe_deadline_ms,
            m.compile_stage.hard_compile_deadline_ms,
            m.compile_stage.modelled_max_processes,
            m.compile_stage.post_compile_scratch_bytes,
        ]
        .contains(&0)
        || m.steady_state_disk_bytes == 0
        || m.steady_state_disk_bytes > m.steady_state_disk_cap
        || !m.toolchain_reverified
    {
        return Err(Error::ExecutionMismatch);
    }
    let roots = usize::try_from(run.config[0]).map_err(|_| Error::Limit)?;
    if run.epochs.is_empty()
        || run.epochs.len() as u64 > run.config[5]
        || run.terminals.len() != roots
        || run.work_used > run.work_reserved
    {
        return Err(Error::ExecutionMismatch);
    }
    let libraries = |l: &[(PathBuf, [u8; 32])]| {
        l.iter()
            .map(|(p, h)| json!({"path": p.to_string_lossy(), "sha256": hex(h)}))
            .collect::<Vec<_>>()
    };
    let cs = &m.compile_stage;
    let aggregate = cs
        .modelled_max_processes
        .checked_mul(cs.hard_address_space_bytes)
        .ok_or(Error::Limit)?;
    let manifest = json!({
        "schema": "invar.gnet.abi12.execution-receipt",
        "schema_version": 2,
        "scope": "fenced_cuda_owned_numeric_execution",
        "compile_receipt": {"id": run.compile.id(),
            "schema": "invar.gnet.abi12.compile-receipt", "schema_version": 1,
            "artifact_sha256": digest(a.source().as_bytes())},
        "plan_nonce": run.plan_nonce,
        "evidence_backend": "CpuGroupModel",
        "module_receipt_id": null,
        "resume_authorized": false,
        "device_qualification": false,
        "execution": {"model": "subprocess-cpu-group-model", "loader": "subprocess",
            "library": null, "workers": 32, "dispatcher": 1, "barrier_parties": 33,
            "stack_bytes": m.stack_bytes,
            "child_address_space_bytes": m.child_address_space_bytes,
            "compile_stage": {
                "hard": {"address_space_bytes_per_process": cs.hard_address_space_bytes,
                    "file_bytes_per_file": cs.hard_file_bytes,
                    "probe_deadline_ms": cs.hard_probe_deadline_ms,
                    "compile_deadline_ms": cs.hard_compile_deadline_ms,
                    "enforcement": "rlimits inherited by every compile-stage process; process-group kill and reap on deadline"},
                "modelled": {"max_concurrent_processes": cs.modelled_max_processes,
                    "aggregate_address_space_bytes": aggregate,
                    "basis": "gcc driver runs its pipeline sequentially (no -pipe)",
                    "enforced": false},
                "post_compile_check": {"scratch_bytes": cs.post_compile_scratch_bytes,
                    "peak_bounded": false},
                "in_runtime_host_cap": false}},
        "model": {"compiler_path": m.compiler_path.to_string_lossy(),
            "compiler_sha256": hex(&m.compiler_sha256),
            "compiler_version_sha256": hex(&m.compiler_version_sha256),
            "host_triple": m.host_triple, "flags": m.flags,
            "shim_sha256": hex(&m.shim_sha256), "driver_sha256": hex(&m.driver_sha256),
            "model_source_sha256": hex(&m.model_source_sha256),
            "executable_sha256": hex(&m.executable_sha256),
            "kernel_sha256": hex(&m.kernel_sha256), "lanes": m.lanes, "words": m.words,
            "toolchain": m.toolchain.iter().map(|p| json!({"role": p.role,
                "path": p.path.to_string_lossy(), "sha256": hex(&p.sha256),
                "libraries": libraries(p.libraries)})).collect::<Vec<_>>(),
            "runtime": {"static": m.runtime_static, "libraries": libraries(m.runtime_libraries)},
            "identity_scope": {
                "bound": ["toolchain programs (shell, driver, cc1plus, collect2, as, ld) and their interpreter-mapped libraries",
                    "the model executable bytes",
                    "the executable's runtime libraries (none when static)"],
                "excluded": ["kernel", "vDSO", "system headers",
                    "static archives (libc.a, libstdc++.a, libgcc.a, libgcc_eh.a)",
                    "crt objects", "gcc spec files"],
                "runtime_code": "bound by the executable bytes (static product profile)",
                "reproducible_build": false,
                "toolchain_hash_window": "hashed before the compile and re-verified after it; a change made and restored between the two checks is not detected; no atomic pinning"}},
        "config": {"roots": run.config[0], "fuel": run.config[1],
            "transitions": run.config[2], "trace_events": run.config[3],
            "quantum": run.config[4], "max_submissions": run.config[5]},
        "input_sha256": hex(&run.input_sha256),
        "initial_state_sha256": hex(&run.initial_sha256),
        "epochs": run.epochs.iter().enumerate().map(|(k, d)| json!({
            "epoch": k + 1, "state_sha256": hex(d)})).collect::<Vec<_>>(),
        "terminals": run.terminals.iter().enumerate().map(|(r, (status, d))| json!({
            "root": r, "status": status, "packet_sha256": hex(d)})).collect::<Vec<_>>(),
        "ledger": {"work_reserved": run.work_reserved, "work_used": run.work_used,
            "work_model": "static-conservative-bound", "instrumented": false,
            "host_bytes": run.host_bytes, "runtime_host_cap": 268_435_456,
            "disk": {"steady_state_session_bytes": m.steady_state_disk_bytes,
                "steady_state_cap": m.steady_state_disk_cap,
                "class": "steady state after compile: post-compile scratch check plus three hard per-file caps; not a whole-session peak bound"}}
    });
    let id = document_digest(&manifest);
    Ok(ExecutionReceipt { id, manifest })
}
