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

mod artifact;
mod compile_mode;
mod device;
mod execution;
mod owned_numeric_cuda;
pub(crate) mod owned_numeric_execution;
use compile_mode::{Mode, compile_mode, task_fields};
pub use device::{DeviceBuffers, DeviceCheckpoint, DeviceExecution, FenceBackend};
use execution::FrozenResult;
pub use execution::{ContinuationIdentity, OracleExecution};

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
