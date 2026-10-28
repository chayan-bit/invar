use std::{
    collections::VecDeque,
    ptr,
    time::{Duration, Instant},
};

use invar_device::{
    Capabilities, CollectiveKind, PrefixMode, Quarantine, Verified, emit_msl, source_hash,
};
use objc2::{rc::Retained, runtime::ProtocolObject};
use objc2_foundation::NSString;
use objc2_metal::{
    MTLBlitCommandEncoder, MTLBuffer, MTLCommandBuffer, MTLCommandBufferStatus, MTLCommandEncoder,
    MTLCommandQueue, MTLCompileOptions, MTLComputeCommandEncoder, MTLComputePipelineState,
    MTLCreateSystemDefaultDevice, MTLDevice, MTLGPUFamily, MTLLanguageVersion, MTLLibrary,
    MTLResourceOptions, MTLSize,
};

use crate::{Error, Fault, WorkState};

mod archive;
mod bench;
mod bits_map;
mod floating;
mod fsum;
mod generated_bench;
pub mod gnet;
mod grid;
mod indexed;
#[cfg(test)]
mod indexed_ordering;
mod law_collective;
mod library;
mod pathsum;
mod resident;
mod run;
mod summary;
pub use summary::{SummaryBench, SummaryResult};
mod segmented;
pub use bits_map::{BitsMap, BitsMapRun};
mod sorting;
mod spatial;
pub mod task;
pub use archive::{Refusal as ArchiveRefusal, Report as ArchiveReport, State as ArchiveState};
pub use floating::{Diagnostic, FloatingResult};
pub use fsum::{FsumBench, FsumResult};
pub use generated_bench::GeneratedBenchResult;
pub use grid::GridResult;
pub use law_collective::{HostPages, WarmMutation};
pub use library::{CacheReport, MAX_LIBRARIES, MAX_PIPELINES, PipelineMeasurement};
pub use pathsum::{PATHSUM_GROUP, PathSum};
pub use resident::ResidentMap;
pub use segmented::SegmentedResult;
pub use sorting::SortResult;
pub use spatial::SpatialResult;

// objc2-metal's device types link CoreGraphics; this block imports no symbols.
// SAFETY: an empty extern block declares nothing callable; it only adds the
// framework to the link line, so no foreign signature can be misdeclared.
#[link(name = "CoreGraphics", kind = "framework")]
unsafe extern "C" {}

const DEADLINE: Duration = Duration::from_secs(10);
/// Completion poll interval. Short enough that dispatch latency stays near the
/// device's; a busy spin instead slowed resident scans by about 25% on M4.
const POLL: Duration = Duration::from_micros(50);
const MAX_QUARANTINE: usize = 8;

type BufferRef = Retained<ProtocolObject<dyn MTLBuffer>>;
type PipelineRef = Retained<ProtocolObject<dyn MTLComputePipelineState>>;

/// A new command buffer on `queue`.
fn command_buffer(
    queue: &ProtocolObject<dyn MTLCommandQueue>,
) -> Result<Retained<ProtocolObject<dyn MTLCommandBuffer>>, Error> {
    queue
        .commandBuffer()
        .ok_or_else(|| Error::Unavailable("Metal command buffer unavailable".into()))
}

/// A new compute encoder on `command`.
fn compute_encoder(
    command: &ProtocolObject<dyn MTLCommandBuffer>,
) -> Result<Retained<ProtocolObject<dyn MTLComputeCommandEncoder>>, Error> {
    command
        .computeCommandEncoder()
        .ok_or_else(|| Error::Unavailable("Metal compute encoder unavailable".into()))
}

struct MetalWork {
    state: WorkState,
    command: Retained<ProtocolObject<dyn MTLCommandBuffer>>,
    _buffers: Vec<BufferRef>,
    _pipelines: Vec<PipelineRef>,
    /// Host pages a no-copy buffer of this command reads; kept while it may run.
    _host: Option<HostPages>,
}

impl MetalWork {
    fn prepared(
        command: Retained<ProtocolObject<dyn MTLCommandBuffer>>,
        buffers: Vec<BufferRef>,
        pipelines: Vec<PipelineRef>,
    ) -> Self {
        Self {
            state: WorkState::Prepared,
            command,
            _buffers: buffers,
            _pipelines: pipelines,
            _host: None,
        }
    }

    fn submit(&mut self) -> Result<(), Error> {
        let submitted = self
            .state
            .submitted()
            .ok_or_else(|| Error::Device("Metal work is not prepared".into()))?;
        self.command.commit();
        self.state = submitted;
        Ok(())
    }

    fn observe(&mut self) -> WorkState {
        let completed = match self.command.status() {
            MTLCommandBufferStatus::Completed => Some(true),
            MTLCommandBufferStatus::Error => Some(false),
            _ => None,
        };
        self.state = self.state.observed(completed);
        self.state
    }

    fn mark_unknown(&mut self) {
        self.state = self.state.uncertain();
    }
}

pub struct MetalDevice {
    device: Retained<ProtocolObject<dyn MTLDevice>>,
    queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
    caps: Capabilities,
    quarantine: Quarantine<MetalWork>,
    libraries: library::Libraries,
    /// D1 strict binary32 probe verdict, run at most once per opened device.
    strict_probe: Option<invar_device::f32_probe::Report>,
}

impl Drop for MetalDevice {
    fn drop(&mut self) {
        // Release confirmed work; the quarantine then leaks, never frees, any
        // command that may still access its buffers and pipelines.
        self.reap();
    }
}

impl MetalDevice {
    pub fn open() -> Result<Self, Error> {
        let device = MTLCreateSystemDefaultDevice().ok_or_else(|| {
            Error::Unavailable("No Metal device; use an Apple7 or newer GPU".into())
        })?;
        if !device.supportsFamily(MTLGPUFamily::Apple7) {
            return Err(Error::Toolchain(
                "Metal GPU family Apple7 or newer is required".into(),
            ));
        }
        let apple_family = [
            (10, MTLGPUFamily::Apple10),
            (9, MTLGPUFamily::Apple9),
            (8, MTLGPUFamily::Apple8),
            (7, MTLGPUFamily::Apple7),
        ]
        .into_iter()
        .find(|(_, family)| device.supportsFamily(*family))
        .map(|(number, _)| number)
        .unwrap_or(7);
        let queue = device
            .newCommandQueue()
            .ok_or_else(|| Error::Unavailable("Metal command queue unavailable".into()))?;
        Ok(Self {
            device,
            queue,
            caps: Capabilities::Metal {
                apple_family,
                msl_major: 3,
                msl_minor: 1,
            },
            quarantine: Quarantine::new(MAX_QUARANTINE),
            libraries: library::Libraries::new(false),
            strict_probe: None,
        })
    }

    pub fn capabilities(&self) -> Capabilities {
        self.caps
    }
    pub fn name(&self) -> String {
        self.device.name().to_string()
    }
    /// Whether dispatch is refused: unreaped work remains, or quarantine
    /// overflow leaked work whose completion can never be confirmed.
    pub fn is_poisoned(&self) -> bool {
        self.quarantine.blocks_dispatch()
    }

    pub fn quarantine_len(&self) -> usize {
        self.quarantine.len()
    }

    /// Release, exactly once, quarantined commands that reached a terminal state.
    pub fn reap(&mut self) {
        self.quarantine.reap(|work| {
            matches!(
                work.observe(),
                WorkState::Completed | WorkState::FailedQuiescent
            )
        });
    }

    /// Reap confirmed work, then refuse dispatch while any remains.
    fn admit(&mut self) -> Result<(), Error> {
        self.reap();
        if self.is_poisoned() {
            return Err(Error::Pending);
        }
        Ok(())
    }

    /// Wait for a terminal command state until `DEADLINE` after `begun`.
    /// A missed deadline quarantines the work unreleased.
    fn await_terminal(&mut self, mut work: MetalWork, begun: Instant) -> Result<MetalWork, Error> {
        while !matches!(
            work.observe(),
            WorkState::Completed | WorkState::FailedQuiescent
        ) {
            if begun.elapsed() > DEADLINE {
                self.quarantine(work);
                return Err(Error::Pending);
            }
            std::thread::sleep(POLL);
        }
        Ok(work)
    }

    fn quarantine(&mut self, mut work: MetalWork) {
        work.mark_unknown();
        self.quarantine.push(work);
    }

    fn encode(
        &self,
        command: &ProtocolObject<dyn MTLCommandBuffer>,
        pipeline: &ProtocolObject<dyn MTLComputePipelineState>,
        buffers: &[&ProtocolObject<dyn MTLBuffer>],
        blocks: u32,
        lanes: u32,
    ) -> Result<(), Error> {
        if lanes as usize > pipeline.maxTotalThreadsPerThreadgroup() {
            return Err(Error::Toolchain(
                "Metal pipeline threadgroup limit is below verified geometry".into(),
            ));
        }
        let encoder = compute_encoder(command)?;
        encoder.setComputePipelineState(pipeline);
        for (index, buffer) in buffers.iter().enumerate() {
            // SAFETY: verified emitter signatures bind these buffers to
            // consecutive slots at offset zero; each retained buffer covers its
            // declared data and outlives the command through its work entry.
            unsafe { encoder.setBuffer_offset_atIndex(Some(*buffer), 0, index) };
        }
        encoder.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize {
                width: blocks as usize,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: lanes as usize,
                height: 1,
                depth: 1,
            },
        );
        encoder.endEncoding();
        Ok(())
    }

    /// Advisory budget. Submission rechecks the live working set before allocation.
    pub fn available_capacity(&self) -> usize {
        usize::try_from(
            self.device
                .recommendedMaxWorkingSetSize()
                .saturating_sub(self.device.currentAllocatedSize() as u64),
        )
        .unwrap_or(0)
    }

    fn check_submission(&self, largest: usize, total: usize) -> Result<(), Error> {
        crate::check_capacity(
            largest,
            total,
            self.device.maxBufferLength() as u64,
            self.device.recommendedMaxWorkingSetSize(),
            self.device.currentAllocatedSize() as u64,
        )
    }

    fn buffer(&self, bytes: usize) -> Result<Retained<ProtocolObject<dyn MTLBuffer>>, Error> {
        self.check_submission(bytes, bytes)?;
        self.device
            .newBufferWithLength_options(bytes, MTLResourceOptions::StorageModeShared)
            .ok_or(Error::ResourceLimit("Metal buffer allocation failed"))
    }
}
