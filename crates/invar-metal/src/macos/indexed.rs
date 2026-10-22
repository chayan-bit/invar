//! Fresh owned irregular buffers. Ordered phases share one completed command;
//! unknown completion retains all buffers and pipelines in the device quarantine.
use super::*;
use invar_device::indexed::{self as ir, Bound};

impl MetalDevice {
    pub fn run_indexed(&mut self, bound: &Bound) -> Result<ir::Receipt, Error> {
        self.run_indexed_with_fault(bound, Fault::None)
    }
    #[doc(hidden)]
    pub fn run_indexed_with_fault(
        &mut self,
        bound: &Bound,
        fault: Fault,
    ) -> Result<ir::Receipt, Error> {
        self.admit()?;
        if bound.plan().capabilities() != self.caps {
            return Err(Error::Limit);
        }
        if fault == Fault::OutOfMemory {
            return Err(Error::OutOfMemory);
        }
        let bytes = bound.buffer_bytes();
        if bytes
            .iter()
            .any(|&n| n > crate::MAX_DEVICE_BYTES || n > self.device.maxBufferLength())
        {
            return Err(Error::ResourceLimit(
                "indexed buffer exceeds Metal maxBufferLength or shared aggregate limit",
            ));
        }
        let allocated = bytes
            .iter()
            .try_fold(self.device.currentAllocatedSize(), |a, b| a.checked_add(*b))
            .and_then(|n| n.checked_add(ir::schedule(bound.plan()).len() * 4))
            .ok_or(Error::ResourceLimit("indexed allocation byte sum overflow"))?;
        if allocated as u64 > self.device.recommendedMaxWorkingSetSize() {
            return Err(Error::ResourceLimit(
                "indexed buffers exceed Metal recommended working set",
            ));
        }
        let source = ir::emit_msl(bound.plan())
            .map_err(|e| Error::Toolchain(format!("indexed IR: {e:?}")))?;
        let names = ir::kernel_names(bound.plan());
        let pipelines = names
            .iter()
            .map(|name| self.compile(&source, name).map(|(p, _)| p))
            .collect::<Result<Vec<_>, _>>()?;
        let (_, lanes) = ir::launch_dimensions(bound.plan());
        if pipelines
            .iter()
            .any(|p| lanes as usize > p.maxTotalThreadsPerThreadgroup())
        {
            return Err(Error::ResourceLimit(
                "indexed launch exceeds Metal pipeline threadgroup limit",
            ));
        }
        let initial = ir::initial_buffers(bound).map_err(|_| Error::OutOfMemory)?;
        let mut buffers = Vec::new();
        // Release each host staging allocation immediately after its copy.
        for words in initial {
            let buffer = self.buffer(words.len().max(1) * 4)?;
            // SAFETY: fresh shared buffer owns at least words.len() initialized
            // u32 slots, is disjoint from host words and has no submitted users.
            unsafe {
                std::ptr::copy_nonoverlapping(
                    words.as_ptr(),
                    buffer.contents().as_ptr().cast::<u32>(),
                    words.len(),
                );
            }
            buffers.push(buffer);
        }
        if fault == Fault::LaunchFailure {
            return Err(Error::Device(
                "indexed injected launch failure before submission".into(),
            ));
        }
        let schedule = ir::schedule(bound.plan());
        let phases = schedule.len() as u32;
        if !schedule.is_empty() {
            let command = self
                .queue
                .commandBuffer()
                .ok_or_else(|| Error::Unavailable("indexed command buffer unavailable".into()))?;
            let mut owned = buffers.clone();
            for (round, kernel) in schedule {
                let round_buffer = self.buffer(4)?;
                // SAFETY: fresh unsubmitted shared allocation owns one aligned
                // u32. Retained unchanged until this whole command is terminal.
                unsafe {
                    round_buffer.contents().as_ptr().cast::<u32>().write(round);
                }
                let mut references: Vec<_> = buffers.iter().map(|b| &**b).collect();
                references.push(&round_buffer);
                // Each encoder ends before the next begins. Tracked buffer
                // hazards order expansion/compaction and all subsequent rounds.
                let (phase_blocks, phase_lanes) = ir::phase_launch_dimensions(bound.plan(), kernel);
                self.encode(
                    &command,
                    &pipelines[kernel],
                    &references,
                    phase_blocks,
                    phase_lanes,
                )?;
                owned.push(round_buffer);
            }
            let mut work = MetalWork::prepared(command, owned, pipelines);
            let begun = Instant::now();
            work.submit()?;
            if fault == Fault::PartialCompletion {
                self.quarantine(work);
                return Err(Error::Pending);
            }
            let work = self.await_terminal(work, begun)?;
            if work.state != WorkState::Completed {
                return Err(Error::Device(
                    "indexed command failed quiescent; no publication".into(),
                ));
            }
        }
        let output_len = bound.plan().ir().output.elements as usize;
        // SAFETY: all commands completed; output and control shared buffers are
        // retained, initialized and aligned for exactly these validated lengths.
        let (words, status) = unsafe {
            (
                ir::copy(std::slice::from_raw_parts(
                    buffers[2].contents().as_ptr().cast::<u32>(),
                    output_len,
                ))
                .map_err(|_| Error::OutOfMemory)?,
                std::slice::from_raw_parts(buffers[6].contents().as_ptr().cast::<u32>(), 2)
                    .to_vec(),
            )
        };
        let output = ir::decode(bound.plan(), words, &status)
            .map_err(|e| Error::Device(format!("indexed output refusal: {e:?}")))?;
        Ok(ir::Receipt {
            target: "metal",
            reason: "verified-u32-completed",
            output,
            source_hash: source_hash(&source),
            transfer_bytes: bound.transfer_bytes(),
            transfer_mode: "shared-memory-host-copy",
            completed_phases: phases,
        })
    }
}
