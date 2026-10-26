//! Verification, identities, fenced execution and oracle observation of a provenanced artifact.

use super::compile_mode::layouts;
use super::*;

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
            if self.artifact.abi_version() == 11 {
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
