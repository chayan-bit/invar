//! Device fences: backend admission, buffer ranges, executions and checkpoints.

use super::*;

/// The trusted adapter identifies its evidence honestly. CPU group-model
/// completion is useful executable evidence but is never GPU qualification.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FenceBackend {
    CudaDriver,
    MetalCommand,
    CpuGroupModel,
}
impl FenceBackend {
    pub(super) fn admits(self, target: compiled::Target) -> bool {
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
    pub(super) abi_version: u32,
    pub(super) max_capture_bytes: usize,
    pub(super) gate: compiled::FencedExecution<R>,
    pub(super) artifact_id: String,
    pub(super) layouts: Value,
    pub(super) max_submissions: u64,
    pub(super) backend: FenceBackend,
    pub(super) instance: u64,
    pub(super) token: Arc<()>,
    pub(super) successful_submissions: u64,
    pub(super) successful_completions: u64,
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

    pub(super) fn check_buffers(&self, buffers: &DeviceBuffers) -> Result<(), Error> {
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
    pub(super) buffers: DeviceBuffers,
    pub(super) manifest: Value,
    pub(super) id: String,
    pub(super) token: Arc<()>,
    pub(super) submitted: u64,
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
