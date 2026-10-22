//! Content provenance for the checked-source -> net -> emitted-source boundary.
//!
//! A digest identifies bytes/configuration, never proves semantics or execution.
//! Construction runs the real frontend, ownership/net checks and emitter. There
//! is no filename, benchmark ID or digest-driven computation dispatch here.

use super::{Net, Proof, compiled, oracle};
use crate::{check, eval, front};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::sync::{
    Arc,
    atomic::{AtomicU64, Ordering},
};

mod numeric_resume;
mod owned_numeric_cuda;
pub(crate) mod owned_numeric_execution;
pub use numeric_resume::{NumericContinuationComponents, NumericContinuationIdentity};

const VERSION: u32 = 3;
const MAX_SOURCE_BYTES: usize = 1024 * 1024;
const MAX_ENTRY_BYTES: usize = 4096;
const MAX_CAPTURE_BYTES: usize = 256 * 1024 * 1024;

#[derive(Clone, Copy, Debug, Default)]
pub struct Limits {
    pub net: super::Limits,
    pub compiler: compiled::Limits,
}

#[derive(Debug, PartialEq, Eq)]
pub enum Error {
    Limit,
    Source(front::FrontError),
    Net(super::Error),
    Compiler(compiled::Error),
    SourceMismatch,
    NetMismatch,
    ArtifactMismatch,
    ContinuationMismatch,
    ManifestMismatch,
    ExecutionMismatch,
    Fence(compiled::FenceError),
}

fn digest(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

fn document_digest(document: &Value) -> String {
    // serde_json::Value has ordered object keys in the locked dependency build.
    digest(document.to_string().as_bytes())
}

/// Private compiler-bound snapshot, NOT a serialized MIR/net format. Debug
/// spellings are scoped by the real compiler build digest and snapshot version.
/// Sort MIR's only unordered collection (locals), retaining all typed operations,
/// declarations, spans, connections and checked operational annotations.
/// source_identity is the foundation's randomized in-process stale-proof token;
/// it is deliberately not promoted into reproducible cryptographic provenance.
fn net_bytes(net: &Net, proof: &Proof) -> Result<Vec<u8>, Error> {
    let mir = net.compiled_mir(proof).map_err(Error::Net)?;
    let mut functions = Vec::new();
    for function in &mir.functions {
        let locals: BTreeMap<_, _> = function.locals.iter().collect();
        functions.push(format!(
            "{:?}",
            (
                &function.name,
                &function.foreign_oracle,
                &function.params,
                &function.result,
                &function.return_from,
                function.span,
                function.entry,
                &function.blocks,
                locals,
                &function.declaration_order,
            )
        ));
    }
    Ok(format!(
        "{:?}",
        (
            VERSION,
            net.contract_version,
            net.graph.schema_version,
            net.graph.revision,
            net.limits,
            (
                mir.schema_version,
                mir.numeric_policy,
                mir.layout_version,
                mir.tensor_contracts
            ),
            functions,
            mir.records,
            mir.variants,
            &net.graph.wires,
            &net.agents,
            &net.control,
        )
    )
    .into_bytes())
}

/// Immutable source-derived artifact with a verified net and exact byte inputs.
/// The manifest is an integrity receipt, not coverage/hardware acceptance.
#[derive(Debug)]
pub struct ProvenancedArtifact {
    source: Vec<u8>,
    entry: String,
    net: Net,
    proof: Proof,
    net_bytes: Vec<u8>,
    artifact: compiled::Artifact,
    manifest: Value,
    id: String,
}

/// Check source and emit a program-specific artifact in one trusted path.
/// No caller-provided checked tree, hash, compiler identity or artifact is trusted.
pub fn compile(
    source: &[u8],
    entry: &str,
    target: compiled::Target,
    limits: Limits,
) -> Result<ProvenancedArtifact, Error> {
    compile_mode(source, entry, target, limits, Mode::Scalar)
}

/// Source-derived CPU ABI8 strict-f32-v1 scalar artifact. This does not widen
/// the public scalar export ABI or admit numeric owners/devices/continuations.
pub fn compile_numeric(
    source: &[u8],
    entry: &str,
    target: compiled::Target,
    limits: Limits,
) -> Result<ProvenancedArtifact, Error> {
    compile_mode(source, entry, target, limits, Mode::Numeric)
}

/// Source-derived CPU ABI9 scalar numeric continuation. The immutable receipt
/// binds its actual task layout; it does not deserialize or authorize a task.
pub fn compile_resumable_numeric(
    source: &[u8],
    entry: &str,
    target: compiled::Target,
    limits: Limits,
) -> Result<ProvenancedArtifact, Error> {
    compile_mode(source, entry, target, limits, Mode::ResumableNumeric)
}

/// Source-derived ABI2 owned-array artifact; storage limits are compiler-owned
/// backend bounds, distinct from source semantics and oracle tool limits.
pub fn compile_owned(
    source: &[u8],
    entry: &str,
    target: compiled::Target,
    limits: Limits,
    storage: compiled::OwnedLimits,
) -> Result<ProvenancedArtifact, Error> {
    compile_mode(source, entry, target, limits, Mode::Owned(storage))
}

/// Source-derived CPU ABI10 direct numeric owners. This private profile binds
/// its heap/cleanup layout without widening scalar exports or admitting owners
/// as external parameters, nominal numeric fields, devices or continuations.
pub fn compile_owned_numeric(
    source: &[u8],
    entry: &str,
    target: compiled::Target,
    limits: Limits,
    storage: compiled::OwnedLimits,
) -> Result<ProvenancedArtifact, Error> {
    compile_mode(source, entry, target, limits, Mode::OwnedNumeric(storage))
}

/// Source-derived direct numeric owner continuation: ABI11 on CPU, ABI12
/// (`cuda-owned-numeric-continuation-v1`) on CUDA, refused on Metal. The
/// receipt binds storage and task layouts; it never authorizes task restore.
/// An ABI12 receipt is a compile receipt only: no load, execution, resume or
/// device-qualification authority. Source/net errors keep their priority and
/// a CUDA emission refusal keeps the pre-ABI12 `NumericTarget`.
pub fn compile_resumable_owned_numeric(
    source: &[u8],
    entry: &str,
    target: compiled::Target,
    limits: Limits,
    storage: compiled::OwnedLimits,
) -> Result<ProvenancedArtifact, Error> {
    compile_mode(
        source,
        entry,
        target,
        limits,
        Mode::ResumableOwnedNumeric(storage),
    )
}

/// Source-derived ABI3 artifact. Continuation layout/identity is bound at compile
/// time; runtime quantum/submission budgets belong to separate observations.
pub fn compile_resumable(
    source: &[u8],
    entry: &str,
    target: compiled::Target,
    limits: Limits,
    storage: compiled::OwnedLimits,
) -> Result<ProvenancedArtifact, Error> {
    compile_mode(source, entry, target, limits, Mode::Resumable(storage))
}

/// Source-derived ABI4 CPU child orchestration, with exact private bounds.
pub fn compile_speculative(
    source: &[u8],
    entry: &str,
    target: compiled::Target,
    limits: Limits,
    storage: compiled::OwnedLimits,
    speculation: compiled::SpeculationLimits,
) -> Result<ProvenancedArtifact, Error> {
    compile_mode(
        source,
        entry,
        target,
        limits,
        Mode::Speculative(storage, speculation, false),
    )
}

/// Test-only ABI5 compilation with a racing device fault hook; its manifest
/// records the fault, so it never verifies as a production artifact.
#[cfg(feature = "gnet-fault-injection")]
pub fn compile_device_speculative_with_fault(
    source: &[u8],
    entry: &str,
    target: compiled::Target,
    limits: Limits,
    storage: compiled::OwnedLimits,
    speculation: compiled::SpeculationLimits,
    fault: compiled::Fault,
) -> Result<ProvenancedArtifact, Error> {
    compile_mode(
        source,
        entry,
        target,
        limits,
        Mode::FaultedDevice(storage, speculation, fault),
    )
}

/// Source-derived ABI5 device children. Compilation creates no fence receipt.
pub fn compile_device_speculative(
    source: &[u8],
    entry: &str,
    target: compiled::Target,
    limits: Limits,
    storage: compiled::OwnedLimits,
    speculation: compiled::SpeculationLimits,
) -> Result<ProvenancedArtifact, Error> {
    compile_mode(
        source,
        entry,
        target,
        limits,
        Mode::Speculative(storage, speculation, true),
    )
}

enum Mode {
    #[cfg(feature = "gnet-fault-injection")]
    FaultedDevice(
        compiled::OwnedLimits,
        compiled::SpeculationLimits,
        compiled::Fault,
    ),
    Scalar,
    Numeric,
    ResumableNumeric,
    Owned(compiled::OwnedLimits),
    OwnedNumeric(compiled::OwnedLimits),
    ResumableOwnedNumeric(compiled::OwnedLimits),
    Resumable(compiled::OwnedLimits),
    Speculative(compiled::OwnedLimits, compiled::SpeculationLimits, bool),
}

fn layouts(artifact: &compiled::Artifact) -> Result<Value, Error> {
    let owned = artifact.owned_layout().map(|layout| {
        json!({"elements": layout.elements(), "allocations": layout.allocations(),
            "cleanup_events": layout.cleanup_events(), "heap_words": layout.heap_words(),
            "cleanup_words": layout.cleanup_words(), "storage_bytes": layout.storage_bytes(),
            "output_words": layout.output_words(), "kernel_name": layout.kernel_name()})
    });
    let continuation = artifact.continuation_layout().map(|layout| {
        json!({"task_bytes": layout.task_bytes(), "kernel_name": layout.kernel_name(),
            "config_words": layout.config_words(), "max_roots": layout.max_roots(),
            "max_slices": layout.max_slices()})
    });
    let speculation = artifact.speculation_layout().map(|layout| {
        let l = layout.limits();
        json!({"slots": layout.slots(), "slot_bytes": layout.slot_bytes(),
            "heap_words": layout.heap_words(), "trace_words": layout.trace_words(),
            "cleanup_words": layout.cleanup_words(), "scratch_bytes": layout.scratch_bytes(),
            "entry_point": layout.entry_point(),
            "limits": {"slots": l.slots, "fuel": l.fuel, "transitions": l.transitions,
                "trace_events": l.trace_events, "quantum": l.quantum,
                "lookahead": l.lookahead, "scratch_bytes": l.scratch_bytes}})
    });
    let device = artifact
        .device_layout()
        .map(|layout| {
            let minimum = compiled::DeviceConfig {
                fuel: 1,
                transitions: 0,
                trace_events: 0,
                quantum: 1,
                rounds: 1,
                capacity: 0,
            };
            layout.buffer_bytes(minimum).map(|bytes| {
                json!({
            "kernel_name": layout.kernel_name(), "group_threads": layout.group_threads(),
            "groups": layout.groups(), "max_rounds": layout.max_rounds(),
            "state_words": layout.state_words(), "minimum_config_words": minimum.words(),
            "minimum_buffer_bytes": bytes, "buffers": bytes.len()})
            })
        })
        .transpose()
        .map_err(Error::Compiler)?;
    let site = |s: compiled::CallSite| json!({"function":s.function,"block":s.block,"position":s.position});
    let fork = artifact.fork_plan().map(|plan| {
        json!({
            "candidates": plan.candidates.iter().map(|c| json!({"source":site(c.source),
                "target":site(c.target), "callee":c.callee, "projection":c.projection,
                "scalar_values":c.scalar_values,"scalar_places":c.scalar_places,
                "replay_owners":c.replay_owners})).collect::<Vec<_>>(),
            "serialized": plan.serialized.iter().map(|s| json!({"source":site(s.source),
                "target":site(s.target),"reason":format!("{:?}",s.reason)})).collect::<Vec<_>>()
        })
    });
    Ok(json!({"owned": owned, "continuation": continuation,
        "speculation":speculation, "device":device, "fork_plan":fork}))
}

/// The 18 continuation task counters after the frames, shared by the ABI11
/// and ABI12 receipt blocks: name, offset and representation.
fn task_fields(frame_bytes: usize) -> Vec<Value> {
    [
        "started",
        "done",
        "depth",
        "max_depth",
        "steps",
        "transitions",
        "used",
        "next_frame",
        "allocated",
        "live",
        "cleanup_used",
        "fuel",
        "transition_cap",
        "trace_cap",
        "status",
        "result",
        "start",
        "end",
    ]
    .iter()
    .enumerate()
    .map(|(index, name)| {
        json!({"name": name, "offset": frame_bytes + index * 8,
            "representation": if index < 14 { "u64" } else { "i64" }})
    })
    .collect()
}

fn compile_mode(
    source: &[u8],
    entry: &str,
    target: compiled::Target,
    limits: Limits,
    mode: Mode,
) -> Result<ProvenancedArtifact, Error> {
    if source.len() > MAX_SOURCE_BYTES || entry.is_empty() || entry.len() > MAX_ENTRY_BYTES {
        return Err(Error::Limit);
    }
    let parsed = front::parse(source).map_err(Error::Source)?;
    let checked = check::check(parsed, false).map_err(Error::Source)?;
    let net = Net::lower(&checked, limits.net).map_err(Error::Net)?;
    let proof = net.verify().map_err(Error::Net)?;
    let snapshot = net_bytes(&net, &proof)?;
    // ABI12 stage 2b: size/entry limits, source parse/check and net errors keep
    // their pre-ABI12 priority. Metal is not an owned-numeric continuation
    // target. On CUDA only a successful ABI12 emission is admitted; every CUDA
    // emission refusal keeps the pre-ABI12 outcome `NumericTarget`, so only
    // valid admitted programs change outcome.
    #[cfg(feature = "gnet-fault-injection")]
    let fault = match &mode {
        Mode::FaultedDevice(_, _, fault) => Some(*fault),
        _ => None,
    };
    let device_continuation =
        matches!(mode, Mode::ResumableOwnedNumeric(_)) && target != compiled::Target::Cpu;
    if device_continuation && target != compiled::Target::Cuda {
        return Err(Error::Compiler(compiled::Error::NumericTarget));
    }
    let artifact = match mode {
        Mode::Scalar => compiled::emit(&net, &proof, entry, target, limits.compiler),
        Mode::Numeric => compiled::emit_numeric(&net, &proof, entry, target, limits.compiler),
        Mode::ResumableNumeric => {
            compiled::emit_resumable_numeric(&net, &proof, entry, target, limits.compiler)
        }
        Mode::Owned(storage) => {
            compiled::emit_owned(&net, &proof, entry, target, limits.compiler, storage)
        }
        Mode::OwnedNumeric(storage) => {
            compiled::emit_owned_numeric(&net, &proof, entry, target, limits.compiler, storage)
        }
        Mode::ResumableOwnedNumeric(storage) => compiled::emit_resumable_owned_numeric(
            &net,
            &proof,
            entry,
            target,
            limits.compiler,
            storage,
        ),
        Mode::Resumable(storage) => {
            compiled::emit_resumable(&net, &proof, entry, target, limits.compiler, storage)
        }
        Mode::Speculative(storage, speculation, false) => compiled::emit_speculative(
            &net,
            &proof,
            entry,
            target,
            limits.compiler,
            storage,
            speculation,
        ),
        Mode::Speculative(storage, speculation, true) => compiled::emit_device_speculative(
            &net,
            &proof,
            entry,
            target,
            limits.compiler,
            storage,
            speculation,
        ),
        #[cfg(feature = "gnet-fault-injection")]
        Mode::FaultedDevice(storage, speculation, fault) => {
            compiled::emit_device_speculative_with_fault(
                &net,
                &proof,
                entry,
                target,
                limits.compiler,
                storage,
                speculation,
                fault,
            )
        }
    }
    .map_err(|error| {
        Error::Compiler(if device_continuation {
            compiled::Error::NumericTarget
        } else {
            error
        })
    })?;
    let output_words = artifact
        .owned_layout()
        .map_or(compiled::OUTPUT_WORDS, |l| l.output_words());
    let mut manifest = json!({
        "version": VERSION,
        "scope": "checked_source_net_emitted_source",
        "compiler_build": env!("INVAR_BUILD_DIGEST"),
        "net_contract": super::CONTRACT_VERSION,
        "source": {"sha256": digest(source), "bytes": source.len()},
        "net": {"sha256": digest(&snapshot), "bytes": snapshot.len()},
        "entry": entry,
        "target": format!("{:?}", artifact.target()),
        "abi_version": artifact.abi_version(),
        "layouts": layouts(&artifact)?,
        "params": artifact.params().iter().map(|t| format!("{t:?}")).collect::<Vec<_>>(),
        "result": format!("{:?}", artifact.result()),
        "artifact": {"sha256": digest(artifact.source().as_bytes()), "bytes": artifact.source().len()},
        "storage": {"states": artifact.states(), "frame_bytes": artifact.frame_bytes(),
            "max_frames": compiled::MAX_FRAMES, "output_words": output_words,
            "event_words": compiled::EVENT_WORDS},
        "limits": {
            "net": {"functions": limits.net.functions, "blocks": limits.net.blocks,
                "instructions": limits.net.instructions, "wires": limits.net.wires,
                "literal_bytes": limits.net.literal_bytes},
            "compiler": {"values": limits.compiler.values, "places": limits.compiler.places,
                "states": limits.compiler.states, "frame_bytes": limits.compiler.frame_bytes,
                "source_bytes": limits.compiler.source_bytes}
        }
    });
    // Additive numeric fields: existing ABI1-7 receipt shapes/version remain.
    if let Some(policy) = artifact.numeric_policy() {
        let mir = net.compiled_mir(&proof).map_err(Error::Net)?;
        manifest["numeric"] = json!({
            "policy": policy.to_json(), "semantic_id": policy.id(),
            "layout_version": mir.layout_version,
            "f32": {"encoding":"ieee754-binary32", "bits":32,
                "carrier":"nonnegative-i64-u32-bits", "maximum":u32::MAX,
                "identity_equality":"exact-bits", "engine_nan_equality":"unobservable-class"},
            "word_bytes":8, "output_words":output_words,
            "event_words":compiled::EVENT_WORDS
        });
        if matches!(artifact.abi_version(), 10..=12) {
            manifest["numeric"]["array_f32"] = json!({
                "kind":258, "element_word_bytes":8, "element_bits":32,
                "encoding":"nonnegative-i64-u32-bits", "external_owner_inputs":false
            });
        }
    }
    if let Some(entry) = artifact.entry_layout() {
        manifest["entry_packet"] = json!({"version":1,"input_words":entry.input_words(),"modes":entry.modes().iter().map(|m|format!("{m:?}")).collect::<Vec<_>>(),"element_stride_bytes":8,"host_element_bytes":4,"numeric":"integer-software-binary32","native_float_admission":false,"subnormal_path":"exact-integer-software","publication":"successful-source-return-after-confirmed-completion"});
    }
    if artifact.abi_version() == 11 {
        let frame_bytes = artifact.frame_bytes();
        manifest["numeric_owned_continuation"] = json!({
            "profile": "cpu-owned-numeric-continuation-v1",
            "contract_version": 1,
            "word_bytes": 8,
            "frame_count": compiled::MAX_FRAMES,
            "frame_stride_bytes": frame_bytes / compiled::MAX_FRAMES,
            "task_fields": task_fields(frame_bytes),
            "output_words": 10,
            "quantum_unit": "mir-transitions",
            "snapshot_scope": "evidence-only-no-task-restore",
            "terminal_packet": "abi10-compatible"
        });
    }
    // Exclusive ABI12 CUDA compile-receipt block; it grants no authority.
    if artifact.abi_version() == 12 {
        manifest[owned_numeric_cuda::KEY] = owned_numeric_cuda::block(&artifact)?;
    }
    #[cfg(feature = "gnet-fault-injection")]
    if let Some(fault) = fault {
        manifest["test_only_fault_injection"] = json!(format!("{fault:?}"));
    }
    let id = document_digest(&manifest);
    Ok(ProvenancedArtifact {
        source: source.to_vec(),
        entry: entry.to_owned(),
        net,
        proof,
        net_bytes: snapshot,
        artifact,
        manifest,
        id,
    })
}

impl ProvenancedArtifact {
    pub fn id(&self) -> &str {
        &self.id
    }
    pub fn manifest(&self) -> &Value {
        &self.manifest
    }
    pub fn artifact(&self) -> &compiled::Artifact {
        &self.artifact
    }
    pub fn source(&self) -> &[u8] {
        &self.source
    }
    pub fn checked_net(&self) -> (&Net, &Proof) {
        (&self.net, &self.proof)
    }
    /// Exact private snapshot bytes hashed in the receipt. This is not an input
    /// format; there is no deserializer or imported-proof acceptance path.
    pub fn net_snapshot(&self) -> &[u8] {
        &self.net_bytes
    }
    pub fn verify_source(&self, bytes: &[u8]) -> Result<(), Error> {
        if self.source == bytes {
            Ok(())
        } else {
            Err(Error::SourceMismatch)
        }
    }
    /// Check an actual current net/proof, including valid but changed graphs.
    pub fn verify_net(&self, net: &Net, proof: &Proof) -> Result<(), Error> {
        if net_bytes(net, proof)? == self.net_bytes {
            Ok(())
        } else {
            Err(Error::NetMismatch)
        }
    }
    pub fn verify_artifact_bytes(&self, bytes: &[u8]) -> Result<(), Error> {
        if self.artifact.source().as_bytes() == bytes {
            Ok(())
        } else {
            Err(Error::ArtifactMismatch)
        }
    }
    /// Compare an untrusted receipt to the manifest constructed from real inputs.
    /// A self-consistent hash on a forged receipt is insufficient.
    pub fn verify_manifest(&self, manifest: &Value) -> Result<(), Error> {
        if self.manifest == *manifest {
            Ok(())
        } else {
            Err(Error::ManifestMismatch)
        }
    }

    /// Bind caller-supplied CPU ABI8 components, including actual executable
    /// bytes and exact f32 encodings. This is an integrity identity, not an
    /// observation, completed-source certificate or permission to publish.
    /// Root must observe the process and validate its semantic output itself.
    pub fn numeric_cpu_identity(
        &self,
        executable: &[u8],
        args: &[eval::Value],
        limits: oracle::Limits,
        output: &[i64],
        trace: &[u64],
    ) -> Result<NumericCpuIdentity, Error> {
        if self.artifact.abi_version() != 8
            || self.artifact.target() != compiled::Target::Cpu
            || executable.is_empty()
            || output.len() != compiled::OUTPUT_WORDS
        {
            return Err(Error::ExecutionMismatch);
        }
        let input = numeric_words(self.artifact.params(), args)?;
        let events = u64::try_from(output[7]).map_err(|_| Error::ExecutionMismatch)?;
        let trace_words = events
            .checked_mul(compiled::EVENT_WORDS as u64)
            .and_then(|n| usize::try_from(n).ok())
            .ok_or(Error::Limit)?;
        if events > limits.trace_events as u64 || trace.len() != trace_words {
            return Err(Error::ExecutionMismatch);
        }
        let bytes = input
            .len()
            .checked_add(output.len())
            .and_then(|n| n.checked_add(trace.len()))
            .and_then(|n| n.checked_mul(8))
            .and_then(|n| n.checked_add(executable.len()))
            .ok_or(Error::Limit)?;
        if bytes > MAX_CAPTURE_BYTES {
            return Err(Error::Limit);
        }
        let output_bytes = signed_bytes(output);
        let trace_bytes: Vec<u8> = trace.iter().flat_map(|n| n.to_le_bytes()).collect();
        let manifest = json!({"version":VERSION,
            "scope":"caller_supplied_abi8_cpu_components", "artifact_id":self.id,
            "backend":"cpu_components_only", "abi_version":8,
            "execution_observed":false, "source_completion_certified":false,
            "resume_authorized":false, "numeric":self.manifest["numeric"],
            "layouts":self.manifest["layouts"], "storage":self.manifest["storage"],
            "params":self.manifest["params"], "result":self.manifest["result"],
            "executable":{"sha256":digest(executable),"bytes":executable.len()},
            "input":{"words":input,"byte_order":"little-endian",
                "sha256":digest(&signed_bytes(&input))},
            "limits":oracle_limits(limits),
            "output":{"sha256":digest(&output_bytes),"bytes":output_bytes.len()},
            "trace":{"sha256":digest(&trace_bytes),"bytes":trace_bytes.len(),"events":events}
        });
        Ok(NumericCpuIdentity {
            id: document_digest(&manifest),
            manifest,
            artifact_id: self.id.clone(),
            emission: self.artifact.source().as_bytes().to_vec(),
            executable: executable.to_vec(),
            input,
            limits,
            output: output.to_vec(),
            trace: trace.to_vec(),
        })
    }

    /// Capture the byte identity of one ABI3 task component under this artifact.
    /// The bytes remain caller-supplied: this does not certify a valid task,
    /// observed execution, fence completion or its separate heap/trace/cleanup
    /// workset. It is not a task deserializer or permission to resume it.
    pub fn continuation_identity(&self, task: &[u8]) -> Result<ContinuationIdentity, Error> {
        // ABI12 task state spans a per-lane seven-buffer workset; a single-task
        // byte identity would misdescribe it.
        if self.artifact.abi_version() == 12 {
            return Err(Error::ContinuationMismatch);
        }
        let layout = self
            .artifact
            .continuation_layout()
            .ok_or(Error::ContinuationMismatch)?;
        if task.len() != layout.task_bytes() {
            return Err(Error::ContinuationMismatch);
        }
        let manifest = json!({"version": VERSION, "scope": "caller_supplied_task_component",
            "artifact_id": self.id, "abi_version": self.artifact.abi_version(),
            "layout": layouts(&self.artifact)?["continuation"],
            "task": {"bytes": task.len(), "sha256": digest(task)}});
        let id = document_digest(&manifest);
        Ok(ContinuationIdentity {
            artifact_id: self.id.clone(),
            task: task.to_vec(),
            manifest,
            id,
        })
    }

    /// Reject cross-artifact tasks and byte mutations against a prior capture.
    /// Hash equality is not used to route computation or validate task semantics.
    pub fn verify_continuation(
        &self,
        identity: &ContinuationIdentity,
        task: &[u8],
    ) -> Result<(), Error> {
        if identity.artifact_id == self.id && identity.task == task {
            Ok(())
        } else {
            Err(Error::ContinuationMismatch)
        }
    }

    /// Bind a backend's owning workset to this ABI5 artifact. Backend-issued
    /// allocation IDs, honest lengths, matching module, live input leases and
    /// truthful launch/fence callbacks remain the trusted host boundary.
    pub fn fenced_execution<R>(
        &self,
        workset: compiled::Workset,
        resources: R,
        max_submissions: u64,
        backend: FenceBackend,
    ) -> Result<DeviceExecution<R>, Error> {
        // ABI12 is a compile receipt only: refused explicitly, before any
        // workset, budget or capture check. Older ABIs keep their prior checks.
        if self.artifact.abi_version() == 12 {
            return Err(Error::ExecutionMismatch);
        }
        let layout = self
            .artifact
            .device_layout()
            .ok_or(Error::ExecutionMismatch)?;
        if workset.target() != self.artifact.target() || !backend.admits(workset.target()) {
            return Err(Error::ExecutionMismatch);
        }
        let expected = layout
            .workset(workset.config(), *workset.ranges())
            .map_err(Error::Compiler)?;
        if expected.buffer_bytes() != workset.buffer_bytes() {
            return Err(Error::ExecutionMismatch);
        }
        let bytes = workset
            .ranges()
            .iter()
            .try_fold(0usize, |n, r| n.checked_add(r.bytes))
            .ok_or(Error::Limit)?;
        if bytes
            > if self.artifact.abi_version() == 13 {
                crate::array_limits::MAX_DEVICE_BYTES
            } else {
                MAX_CAPTURE_BYTES
            }
        {
            return Err(Error::Limit);
        }
        let gate = compiled::FencedExecution::new(workset, resources, max_submissions)
            .map_err(Error::Fence)?;
        static NEXT: AtomicU64 = AtomicU64::new(1);
        let instance = NEXT
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |n| n.checked_add(1))
            .map_err(|_| Error::Limit)?;
        Ok(DeviceExecution {
            gate,
            artifact_id: self.id.clone(),
            abi_version: self.artifact.abi_version(),
            max_capture_bytes: if self.artifact.abi_version() == 13 {
                crate::array_limits::MAX_DEVICE_BYTES
            } else {
                MAX_CAPTURE_BYTES
            },
            layouts: layouts(&self.artifact)?,
            max_submissions,
            backend,
            instance,
            token: Arc::new(()),
            successful_submissions: 0,
            successful_completions: 0,
        })
    }

    /// Execute the independent CPU net oracle, including for a device-targeted
    /// artifact. This records CPU oracle execution ONLY, never device execution
    /// or execution of the emitted source. Runtime inputs/budgets/results are
    /// separate from the compile receipt and cannot change its identity.
    pub fn observe_oracle(
        &self,
        args: &[eval::Value],
        limits: oracle::Limits,
    ) -> Result<OracleExecution, Error> {
        // Capture mutable entry input before execution; never echo large host
        // arrays into a receipt. Debug encoding is scoped by compiler_build.
        let input = if self.artifact.abi_version() == 13 {
            entry_input_identity(args)?
        } else {
            json!(format!("{args:?}"))
        };
        let observation = oracle::observe_call(&self.net, &self.proof, &self.entry, args, limits)
            .map_err(Error::Net)?;
        let result = FrozenResult::capture(&observation.semantic.result)?;
        let mut manifest = json!({
            "version": VERSION,
            "backend": "cpu_net_oracle",
            "execution_model": "sequential_net_oracle",
            "continuation_execution": false,
            "artifact_id": self.id,
            "artifact_target": format!("{:?}", self.artifact.target()),
            "abi_version": self.artifact.abi_version(),
            "layouts": layouts(&self.artifact)?,
            "input": input,
            "limits": {"fuel": limits.fuel, "transitions": limits.transitions,
                "trace_events": limits.trace_events, "array_elements": limits.array_elements},
            "observed": {
                "result": format!("{:?}", observation.semantic.result),
                "source_steps": observation.semantic.steps_used,
                "cleanup": format!("{:?}", observation.semantic.cleanup_trace),
                "failure": format!("{:?}", observation.failure),
                "transitions": observation.transitions,
                "max_frames": observation.max_frames,
                "event_count": observation.events.len(),
                "events_sha256": digest(format!("{:?}", observation.events).as_bytes())
            }
        });
        if self.artifact.numeric_policy().is_some() {
            manifest["numeric"] = self.manifest["numeric"].clone();
            if self.artifact.abi_version() != 13 {
                manifest["input_words"] = json!(numeric_words(self.artifact.params(), args)?);
            }
            manifest["result_bits"] = match &observation.semantic.result {
                Ok(eval::Value::F32(bits)) => json!(bits.to_bits()),
                _ => Value::Null,
            };
            if self.artifact.abi_version() == 10 {
                manifest["result_element_bits"] = match &observation.semantic.result {
                    Ok(eval::Value::ArrayF32(bits)) => {
                        json!(
                            bits.borrow()
                                .iter()
                                .map(|n| n.to_bits())
                                .collect::<Vec<_>>()
                        )
                    }
                    _ => Value::Null,
                };
            }
        }
        let id = document_digest(&manifest);
        Ok(OracleExecution {
            observation,
            result,
            manifest,
            id,
        })
    }
}

fn oracle_limits(limits: oracle::Limits) -> Value {
    json!({"fuel":limits.fuel,"transitions":limits.transitions,
        "trace_events":limits.trace_events,"array_elements":limits.array_elements})
}
fn signed_bytes(words: &[i64]) -> Vec<u8> {
    words.iter().flat_map(|n| n.to_le_bytes()).collect()
}
fn entry_input_identity(args: &[eval::Value]) -> Result<Value, Error> {
    if args.len() > crate::eval::host_read::MAX_PARAMETERS {
        return Err(Error::Limit);
    }
    for arg in args {
        use eval::Value as V;
        let length = match arg {
            V::I32(_) | V::U32(_) | V::F32(_) | V::Bool(_) | V::Unit => 0,
            V::ArrayI32(v) => v.try_borrow().map_err(|_| Error::ExecutionMismatch)?.len(),
            V::ArrayU32(v) => v.try_borrow().map_err(|_| Error::ExecutionMismatch)?.len(),
            V::ArrayF32(v) => v.try_borrow().map_err(|_| Error::ExecutionMismatch)?.len(),
            V::Record(v) => {
                let fields = v.try_borrow().map_err(|_| Error::ExecutionMismatch)?;
                if fields.len() > compiled::MAX_RECORD_FIELDS
                    || fields
                        .iter()
                        .any(|v| !matches!(v, V::I32(_) | V::U32(_) | V::F32(_) | V::Bool(_)))
                {
                    return Err(Error::ExecutionMismatch);
                }
                fields.len()
            }
            _ => return Err(Error::ExecutionMismatch),
        };
        if length > crate::array_limits::MAX_ARRAY_LEN {
            return Err(Error::Limit);
        }
    }
    struct Sink(Sha256);
    impl std::fmt::Write for Sink {
        fn write_str(&mut self, text: &str) -> std::fmt::Result {
            self.0.update(text.as_bytes());
            Ok(())
        }
    }
    let mut sink = Sink(Sha256::new());
    std::fmt::write(&mut sink, format_args!("{args:?}")).map_err(|_| Error::ExecutionMismatch)?;
    let sha256 = sink
        .0
        .finalize()
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect::<String>();
    let groups = args
        .iter()
        .enumerate()
        .map(|(i, a)| {
            (0..i)
                .find(|j| oracle::external_alias(&args[*j], a))
                .unwrap_or(i)
        })
        .collect::<Vec<_>>();
    Ok(json!({"encoding":"compiler-scoped-rust-debug-v1","sha256":sha256,"alias_groups":groups}))
}

fn numeric_words(types: &[front::Type], args: &[eval::Value]) -> Result<Vec<i64>, Error> {
    if types.len() != args.len() {
        return Err(Error::ExecutionMismatch);
    }
    types
        .iter()
        .zip(args)
        .map(|(ty, arg)| match (ty, arg) {
            (front::Type::I32, eval::Value::I32(n)) => Ok(i64::from(*n)),
            (front::Type::U32, eval::Value::U32(n)) => Ok(i64::from(*n)),
            (front::Type::Bool, eval::Value::Bool(n)) => Ok(i64::from(*n)),
            (front::Type::F32, eval::Value::F32(n)) => Ok(i64::from(n.to_bits())),
            _ => Err(Error::ExecutionMismatch),
        })
        .collect()
}

/// Immutable identities of supplied components. No imported receipt can become
/// an observed CPU run or a GPU/fence receipt through this type.
#[derive(Debug)]
pub struct NumericCpuIdentity {
    id: String,
    manifest: Value,
    artifact_id: String,
    emission: Vec<u8>,
    executable: Vec<u8>,
    input: Vec<i64>,
    limits: oracle::Limits,
    output: Vec<i64>,
    trace: Vec<u64>,
}
impl NumericCpuIdentity {
    pub fn id(&self) -> &str {
        &self.id
    }
    pub fn manifest(&self) -> &Value {
        &self.manifest
    }
    pub fn verify_manifest(&self, manifest: &Value) -> Result<(), Error> {
        if &self.manifest == manifest {
            Ok(())
        } else {
            Err(Error::ManifestMismatch)
        }
    }
    /// Require this actual source-derived artifact, including exact emission.
    pub fn verify_artifact(&self, artifact: &ProvenancedArtifact) -> Result<(), Error> {
        if artifact.id == self.artifact_id && artifact.artifact.source().as_bytes() == self.emission
        {
            Ok(())
        } else {
            Err(Error::ArtifactMismatch)
        }
    }
    /// Compare all exact supplied bytes/carriers/budgets without trusting hashes.
    pub fn verify_components(
        &self,
        executable: &[u8],
        input: &[i64],
        limits: oracle::Limits,
        output: &[i64],
        trace: &[u64],
    ) -> Result<(), Error> {
        if executable == self.executable
            && input == self.input
            && output == self.output
            && trace == self.trace
            && oracle_limits(limits) == oracle_limits(self.limits)
        {
            Ok(())
        } else {
            Err(Error::ExecutionMismatch)
        }
    }
}

/// The trusted adapter identifies its evidence honestly. CPU group-model
/// completion is useful executable evidence but is never GPU qualification.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FenceBackend {
    CudaDriver,
    MetalCommand,
    CpuGroupModel,
}
impl FenceBackend {
    fn admits(self, target: compiled::Target) -> bool {
        matches!(
            (self, target),
            (Self::CudaDriver, compiled::Target::Cuda)
                | (Self::MetalCommand, compiled::Target::Metal)
                | (
                    Self::CpuGroupModel,
                    compiled::Target::Cuda | compiled::Target::Metal
                )
        )
    }
}

/// All twelve complete backend-issued ranges, including allocated padding.
/// Order is core ABI5 buffer order; bytes remain opaque, not deserialized tasks.
pub type DeviceBuffers = [Vec<u8>; 12];

/// Receipt creation is possible only through the existing owning fence gate.
/// No constructor accepts an imported fence flag, manifest or result hash.
pub struct DeviceExecution<R> {
    abi_version: u32,
    max_capture_bytes: usize,
    gate: compiled::FencedExecution<R>,
    artifact_id: String,
    layouts: Value,
    max_submissions: u64,
    backend: FenceBackend,
    instance: u64,
    token: Arc<()>,
    successful_submissions: u64,
    successful_completions: u64,
}
impl<R> DeviceExecution<R> {
    pub fn state(&self) -> compiled::ExecutionState {
        self.gate.state()
    }
    pub fn submitted(&self) -> u64 {
        self.gate.submitted()
    }
    pub fn completed(&self) -> u64 {
        self.gate.completed()
    }
    pub fn workset(&self) -> &compiled::Workset {
        self.gate.workset()
    }
    pub fn submit(
        &mut self,
        launch: impl FnOnce(&mut R) -> compiled::Submission,
    ) -> Result<(), Error> {
        self.gate.submit(launch).map_err(Error::Fence)?;
        // The core gate caps attempts at 1024. These success counts cannot wrap.
        self.successful_submissions += 1;
        Ok(())
    }
    pub fn poll(
        &mut self,
        query: impl FnOnce(&mut R) -> compiled::FenceObservation,
    ) -> Result<(), Error> {
        self.gate.poll(query).map_err(Error::Fence)?;
        self.successful_completions += 1;
        Ok(())
    }
    pub fn cancel(&mut self) {
        self.gate.cancel();
    }
    pub fn reclaim(self) -> Result<R, Error> {
        self.gate.reclaim().map_err(Error::Fence)
    }

    fn check_buffers(&self, buffers: &DeviceBuffers) -> Result<(), Error> {
        if buffers
            .iter()
            .zip(self.workset().ranges())
            .any(|(b, r)| b.len() != r.bytes)
        {
            return Err(Error::ExecutionMismatch);
        }
        for (bytes, word) in buffers[3][..48]
            .chunks_exact(8)
            .zip(self.workset().config().words())
        {
            if bytes != word.to_le_bytes() {
                return Err(Error::ExecutionMismatch);
            }
        }
        Ok(())
    }

    /// Readback is called only after actual adapter-reported successful fence
    /// observation. Root status may still be unfinished/failed: this captures
    /// a checkpoint and certifies neither source completion nor resume legality.
    pub fn capture(
        &self,
        read: impl FnOnce(&R) -> DeviceBuffers,
    ) -> Result<DeviceCheckpoint, Error> {
        let buffers = self.gate.observe(read).map_err(Error::Fence)?;
        self.check_buffers(&buffers)?;
        let workset = self.workset();
        let manifest = json!({"version":VERSION, "scope":"adapter_observed_fenced_device_checkpoint",
            "artifact_id":self.artifact_id, "abi_version":self.abi_version, "layouts":self.layouts,
            "artifact_target":format!("{:?}",workset.target()),
            "evidence_backend":format!("{:?}",self.backend),
            "fence_evidence":"core_FencedExecution_observe_after_Complete",
            "source_completion_certified":false, "resume_authorized":false,
            "execution_instance":self.instance, "instance_scope":"in_process_only",
            "config_words":workset.config().words(), "required_buffer_bytes":workset.buffer_bytes(),
            "config_encoding":"u64_little_endian",
            "ranges":workset.ranges().iter().map(|r|json!({"device":r.device,
                "allocation":r.allocation,"offset":r.offset,"bytes":r.bytes})).collect::<Vec<_>>(),
            "host_budget":{"max_submissions":self.max_submissions,"max_capture_bytes":self.max_capture_bytes},
            "fence":{"state":format!("{:?}",self.state()),"submitted":self.submitted(),"completed":self.completed(),
                "counter_semantics":"core_host_attempt_ordinals", "successful_submissions":self.successful_submissions,
                "successful_completions":self.successful_completions},
            "buffers":buffers.iter().map(|b|json!({"bytes":b.len(),"sha256":digest(b)})).collect::<Vec<_>>()});
        let id = document_digest(&manifest);
        Ok(DeviceCheckpoint {
            buffers,
            manifest,
            id,
            token: self.token.clone(),
            submitted: self.submitted(),
        })
    }

    /// Exact byte comparison under this same completed epoch. New submission,
    /// another owning gate or changed primary/private storage makes it stale.
    /// Receipt equality grants no kernel dispatch or continuation deserialization.
    pub fn verify_checkpoint(
        &self,
        receipt: &DeviceCheckpoint,
        read: impl FnOnce(&R) -> DeviceBuffers,
    ) -> Result<(), Error> {
        let buffers = self.gate.observe(read).map_err(Error::Fence)?;
        self.check_buffers(&buffers)?;
        if Arc::ptr_eq(&self.token, &receipt.token)
            && receipt.submitted == self.submitted()
            && self.completed() == self.submitted()
            && receipt.buffers == buffers
        {
            Ok(())
        } else {
            Err(Error::ExecutionMismatch)
        }
    }
}

/// Immutable fenced checkpoint of the whole ABI5 workset. Private children,
/// generations, task/config, input, heap, traces, cleanup and output are all
/// captured as exact bytes; a digest itself proves none of their semantics.
#[derive(Debug)]
pub struct DeviceCheckpoint {
    buffers: DeviceBuffers,
    manifest: Value,
    id: String,
    token: Arc<()>,
    submitted: u64,
}
impl DeviceCheckpoint {
    pub fn buffers(&self) -> &DeviceBuffers {
        &self.buffers
    }
    pub fn manifest(&self) -> &Value {
        &self.manifest
    }
    pub fn id(&self) -> &str {
        &self.id
    }
    pub fn verify_manifest(&self, manifest: &Value) -> Result<(), Error> {
        if self.manifest == *manifest {
            Ok(())
        } else {
            Err(Error::ManifestMismatch)
        }
    }
}

/// Opaque identity for captured task bytes; no execution claim or resume method.
#[derive(Debug)]
pub struct ContinuationIdentity {
    artifact_id: String,
    task: Vec<u8>,
    manifest: Value,
    id: String,
}
impl ContinuationIdentity {
    pub fn id(&self) -> &str {
        &self.id
    }
    pub fn manifest(&self) -> &Value {
        &self.manifest
    }
    pub fn verify_manifest(&self, manifest: &Value) -> Result<(), Error> {
        if self.manifest == *manifest {
            Ok(())
        } else {
            Err(Error::ManifestMismatch)
        }
    }
}

/// Receipt created only by an observed oracle call; compile() creates none.
#[derive(Debug)]
pub struct OracleExecution {
    observation: oracle::Observation,
    result: FrozenResult,
    manifest: Value,
    id: String,
}
impl OracleExecution {
    /// Defensive snapshot: returned array storage never aliases this receipt.
    /// Copies the bounded trace, cleanup, owned arrays and flat record result.
    pub fn observation(&self) -> oracle::Observation {
        oracle::Observation {
            semantic: eval::Observation {
                result: self.result.materialize(),
                steps_used: self.observation.semantic.steps_used,
                cleanup_trace: self
                    .observation
                    .semantic
                    .cleanup_trace
                    .iter()
                    .map(|e| eval::CleanupEvent {
                        name: e.name.clone(),
                        length: e.length,
                    })
                    .collect(),
            },
            failure: self.observation.failure,
            events: self.observation.events.clone(),
            transitions: self.observation.transitions,
            max_frames: self.observation.max_frames,
        }
    }
    pub fn manifest(&self) -> &Value {
        &self.manifest
    }
    pub fn id(&self) -> &str {
        &self.id
    }
    pub fn verify_manifest(&self, manifest: &Value) -> Result<(), Error> {
        if self.manifest == *manifest {
            Ok(())
        } else {
            Err(Error::ManifestMismatch)
        }
    }
}

#[derive(Debug)]
enum FrozenResult {
    I32(i32),
    U32(u32),
    Bool(bool),
    F32(crate::numeric_contract::F32Bits),
    Unit,
    ArrayI32(Vec<i32>),
    ArrayU32(Vec<u32>),
    ArrayF32(Vec<crate::numeric_contract::F32Bits>),
    Record(Vec<FrozenResult>),
    Trap(eval::TrapCategory, front::Span),
    Limit(u64, u64),
    Tool(&'static str, String),
}
impl FrozenResult {
    fn capture(result: &Result<eval::Value, eval::EvalError>) -> Result<Self, Error> {
        Ok(match result {
            Ok(eval::Value::I32(n)) => Self::I32(*n),
            Ok(eval::Value::U32(n)) => Self::U32(*n),
            Ok(eval::Value::Bool(n)) => Self::Bool(*n),
            Ok(eval::Value::F32(n)) => Self::F32(*n),
            Ok(eval::Value::Unit) => Self::Unit,
            Ok(eval::Value::ArrayI32(xs)) => Self::ArrayI32(xs.borrow().clone()),
            Ok(eval::Value::ArrayU32(xs)) => Self::ArrayU32(xs.borrow().clone()),
            Ok(eval::Value::ArrayF32(xs)) => {
                let values = xs.borrow();
                let mut frozen = Vec::new();
                frozen
                    .try_reserve_exact(values.len())
                    .map_err(|_| Error::Limit)?;
                frozen.extend_from_slice(&values);
                Self::ArrayF32(frozen)
            }
            Ok(eval::Value::Record(fields)) => {
                let fields = fields.try_borrow().map_err(|_| Error::ExecutionMismatch)?;
                if fields.len() > compiled::MAX_RECORD_FIELDS {
                    return Err(Error::Limit);
                }
                let mut frozen = Vec::new();
                frozen
                    .try_reserve_exact(fields.len())
                    .map_err(|_| Error::Limit)?;
                for value in fields.iter() {
                    frozen.push(match value {
                        eval::Value::I32(v) => Self::I32(*v),
                        eval::Value::U32(v) => Self::U32(*v),
                        eval::Value::F32(v) => Self::F32(*v),
                        eval::Value::Bool(v) => Self::Bool(*v),
                        _ => return Err(Error::Compiler(compiled::Error::UnsupportedType)),
                    });
                }
                Self::Record(frozen)
            }
            // Nested/owning aggregates and borrowed views remain unsupported.
            Ok(_) => return Err(Error::Compiler(compiled::Error::UnsupportedType)),
            Err(eval::EvalError::Trap { category, span }) => Self::Trap(*category, *span),
            Err(eval::EvalError::EvaluationLimit {
                fuel_limit,
                steps_used,
            }) => Self::Limit(*fuel_limit, *steps_used),
            Err(eval::EvalError::Tool { code, message }) => Self::Tool(code, message.clone()),
        })
    }
    fn materialize(&self) -> Result<eval::Value, eval::EvalError> {
        use std::{cell::RefCell, rc::Rc};
        Ok(match self {
            Self::I32(n) => eval::Value::I32(*n),
            Self::U32(n) => eval::Value::U32(*n),
            Self::Bool(n) => eval::Value::Bool(*n),
            Self::F32(n) => eval::Value::F32(*n),
            Self::Unit => eval::Value::Unit,
            Self::ArrayI32(xs) => eval::Value::ArrayI32(Rc::new(RefCell::new(xs.clone()))),
            Self::ArrayU32(xs) => eval::Value::ArrayU32(Rc::new(RefCell::new(xs.clone()))),
            Self::ArrayF32(xs) => {
                let mut values = Vec::new();
                values
                    .try_reserve_exact(xs.len())
                    .map_err(|_| eval::EvalError::Tool {
                        code: "RESOURCE_LIMIT",
                        message: "numeric owner observation allocation failed".into(),
                    })?;
                values.extend_from_slice(xs);
                eval::Value::ArrayF32(Rc::new(RefCell::new(values)))
            }
            Self::Record(fields) => eval::Value::Record(Rc::new(RefCell::new(
                fields
                    .iter()
                    .map(Self::materialize)
                    .collect::<Result<Vec<_>, _>>()?,
            ))),
            Self::Trap(category, span) => {
                return Err(eval::EvalError::Trap {
                    category: *category,
                    span: *span,
                });
            }
            Self::Limit(fuel_limit, steps_used) => {
                return Err(eval::EvalError::EvaluationLimit {
                    fuel_limit: *fuel_limit,
                    steps_used: *steps_used,
                });
            }
            Self::Tool(code, message) => {
                return Err(eval::EvalError::Tool {
                    code,
                    message: message.clone(),
                });
            }
        })
    }
}
