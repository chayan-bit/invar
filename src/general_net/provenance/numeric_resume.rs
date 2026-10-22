//! Exact identities of supplied ABI9 components. No receipt is a task decoder,
//! execution observation, semantic validator or authorization to resume.
use super::{
    Error, MAX_CAPTURE_BYTES, ProvenancedArtifact, VERSION, compiled, digest, document_digest,
    eval, numeric_words, oracle, oracle_limits, signed_bytes,
};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};

/// A caller's bounded CPU continuation components. None of these bytes are
/// executed or interpreted as task state when constructing an identity.
pub struct NumericContinuationComponents<'a> {
    pub executable: &'a [u8],
    pub args: &'a [eval::Value],
    pub limits: oracle::Limits,
    pub quantum: u64,
    pub task: &'a [u8],
    /// Present only when the caller supplies an eight-word output component.
    /// Presence does not assert task termination or valid source completion.
    pub output: Option<&'a [i64]>,
    pub trace: &'a [u64],
}

#[derive(Debug)]
pub struct NumericContinuationIdentity {
    id: String,
    manifest: Value,
    artifact_id: String,
    params: Vec<super::front::Type>,
    emission: Vec<u8>,
    executable: Vec<u8>,
    input: Vec<i64>,
    limits: oracle::Limits,
    quantum: u64,
    task: Vec<u8>,
    output: Option<Vec<i64>>,
    trace: Vec<u64>,
}

fn copy<T: Copy>(values: &[T]) -> Result<Vec<T>, Error> {
    let mut result = Vec::new();
    result
        .try_reserve_exact(values.len())
        .map_err(|_| Error::Limit)?;
    result.extend_from_slice(values);
    Ok(result)
}

fn trace_digest(words: &[u64]) -> String {
    let mut hash = Sha256::new();
    for word in words {
        hash.update(word.to_le_bytes());
    }
    hash.finalize().iter().map(|b| format!("{b:02x}")).collect()
}

impl ProvenancedArtifact {
    /// Bind the complete supplied scalar CPU workset to this actual ABI9
    /// emission. Validate framing and exact scalar types, then copy the bytes.
    /// Invalid runtime budgets remain recordable as failure evidence; this
    /// identity never admits those budgets or certifies task fields/outputs.
    pub fn numeric_continuation_identity(
        &self,
        components: NumericContinuationComponents<'_>,
    ) -> Result<NumericContinuationIdentity, Error> {
        let c = components;
        if self.artifact.abi_version() != 9 || self.artifact.target() != compiled::Target::Cpu {
            return Err(Error::ExecutionMismatch);
        }
        let layout = self
            .artifact
            .continuation_layout()
            .ok_or(Error::ContinuationMismatch)?;
        if c.task.len() != layout.task_bytes() {
            return Err(Error::ContinuationMismatch);
        }
        if c.executable.is_empty()
            || !c.trace.len().is_multiple_of(compiled::EVENT_WORDS)
            || c.trace.len() / compiled::EVENT_WORDS > c.limits.trace_events
            || c.output.is_some_and(|o| o.len() != compiled::OUTPUT_WORDS)
        {
            return Err(Error::ExecutionMismatch);
        }
        if let Some(output) = c.output {
            let events = usize::try_from(output[7]).map_err(|_| Error::ExecutionMismatch)?;
            if events != c.trace.len() / compiled::EVENT_WORDS {
                return Err(Error::ExecutionMismatch);
            }
        }
        let input = numeric_words(self.artifact.params(), c.args)?;
        let size = input
            .len()
            .checked_add(c.trace.len())
            .and_then(|n| n.checked_add(c.output.map_or(0, <[i64]>::len)))
            .and_then(|n| n.checked_mul(8))
            .and_then(|n| n.checked_add(c.executable.len()))
            .and_then(|n| n.checked_add(c.task.len()))
            .ok_or(Error::Limit)?;
        if size > MAX_CAPTURE_BYTES {
            return Err(Error::Limit);
        }
        let trace_sha = trace_digest(c.trace);
        let output = c.output.map(signed_bytes);
        let manifest = json!({
            "version":VERSION, "scope":"caller_supplied_abi9_cpu_components",
            "backend":"cpu_components_only", "abi_version":9, "artifact_id":self.id,
            "execution_observed":false, "task_semantics_validated":false,
            "source_completion_certified":false, "resume_authorized":false,
            "numeric":self.manifest["numeric"], "layouts":self.manifest["layouts"],
            "storage":self.manifest["storage"], "params":self.manifest["params"],
            "result":self.manifest["result"], "limits":oracle_limits(c.limits),
            "quantum":c.quantum,
            "executable":{"bytes":c.executable.len(),"sha256":digest(c.executable)},
            "input":{"words":input,"byte_order":"little-endian",
                "sha256":digest(&signed_bytes(&input))},
            "task":{"bytes":c.task.len(),"sha256":digest(c.task),
                "encoding":"opaque-caller-task-bytes"},
            "output":output.as_ref().map(|b|json!({"bytes":b.len(),"sha256":digest(b)})),
            "trace":{"bytes":c.trace.len()*8,"sha256":trace_sha,
                "events":c.trace.len()/compiled::EVENT_WORDS,"byte_order":"little-endian"}
        });
        Ok(NumericContinuationIdentity {
            id: document_digest(&manifest),
            manifest,
            artifact_id: self.id.clone(),
            emission: copy(self.artifact.source().as_bytes())?,
            params: self.artifact.params().to_vec(),
            executable: copy(c.executable)?,
            input,
            limits: c.limits,
            quantum: c.quantum,
            task: copy(c.task)?,
            output: c.output.map(copy).transpose()?,
            trace: copy(c.trace)?,
        })
    }
}

impl NumericContinuationIdentity {
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
    pub fn verify_artifact(&self, artifact: &ProvenancedArtifact) -> Result<(), Error> {
        if self.artifact_id == artifact.id && self.emission == artifact.artifact.source().as_bytes()
        {
            Ok(())
        } else {
            Err(Error::ArtifactMismatch)
        }
    }
    /// Compare exact current components with the captured immutable copy. This
    /// also compares typed initial inputs, independently of captured task bits.
    pub fn verify_components(
        &self,
        components: NumericContinuationComponents<'_>,
    ) -> Result<(), Error> {
        let c = components;
        // Types come from the real emitted artifact, never serialized metadata.
        let input = numeric_words(&self.params, c.args)?;
        if self.executable == c.executable
            && self.input == input
            && oracle_limits(self.limits) == oracle_limits(c.limits)
            && self.quantum == c.quantum
            && self.task == c.task
            && self.output.as_deref() == c.output
            && self.trace == c.trace
        {
            Ok(())
        } else {
            Err(Error::ExecutionMismatch)
        }
    }
}
