//! M4-04 segmented map, segmented reduction and stable compaction on Metal,
//! from the same verified IR as CUDA. Inputs are admitted before upload; only
//! a Completed command buffer publishes output.

use super::*;
use invar_device::segmented as seg;

pub struct SegmentedResult {
    /// Map: one word per element. Reduce: one per segment. Compact: exactly
    /// the retained words, in source order.
    pub output: Vec<u32>,
    pub source_hash: u64,
    pub compile_ms: f64,
    pub kernel_ms: f64,
    pub transfer_bytes: usize,
}

impl MetalDevice {
    pub fn run_segmented(
        &mut self,
        plan: &seg::Verified,
        input: &[u32],
        offsets: &[u32],
    ) -> Result<SegmentedResult, Error> {
        self.run_segmented_with_fault(plan, input, offsets, Fault::None)
    }

    pub fn run_segmented_with_fault(
        &mut self,
        plan: &seg::Verified,
        input: &[u32],
        offsets: &[u32],
        fault: Fault,
    ) -> Result<SegmentedResult, Error> {
        self.admit()?;
        match fault {
            Fault::OutOfMemory => return Err(Error::OutOfMemory),
            Fault::LaunchFailure => {
                return Err(Error::Device(
                    "injected launch rejection before submission".into(),
                ));
            }
            _ => {}
        }
        if plan.capabilities() != self.caps {
            return Err(Error::Limit);
        }
        let launches = plan.launches(input, offsets).map_err(|_| Error::Limit)?;
        let words = plan.buffer_words();
        let transfer_bytes = words
            .iter()
            .try_fold(0usize, |total, n| {
                total.checked_add(n.max(&1).checked_mul(4)?)
            })
            .ok_or(Error::Limit)?;
        crate::check_aggregate_limit(transfer_bytes)?;
        let largest = words.iter().map(|n| n.max(&1) * 4).max().unwrap_or(0);
        self.check_submission(largest, transfer_bytes)?;
        if self.device.maxThreadgroupMemoryLength() < seg::LANES as usize * 8 {
            return Err(Error::ResourceLimit(
                "segmented scratch exceeds Metal threadgroup memory",
            ));
        }
        let source =
            seg::emit_msl(plan).map_err(|e| Error::Toolchain(format!("MSL emission: {e:?}")))?;
        let started = Instant::now();
        let pipelines = launches
            .iter()
            .map(|launch| {
                self.compile(&source, plan.kernels()[launch.kernel])
                    .map(|(p, _)| p)
            })
            .collect::<Result<Vec<_>, _>>()?;
        if pipelines
            .iter()
            .any(|pipeline| pipeline.maxTotalThreadsPerThreadgroup() < seg::LANES as usize)
        {
            return Err(Error::ResourceLimit(
                "segmented geometry exceeds Metal pipeline threadgroup limit",
            ));
        }
        let compile_ms = started.elapsed().as_secs_f64() * 1000.0;
        let buffers = words
            .iter()
            .map(|n| self.buffer(n.max(&1) * 4))
            .collect::<Result<Vec<_>, _>>()?;
        for (values, buffer) in [input, offsets].into_iter().zip(&buffers) {
            // SAFETY: each retained shared buffer holds at least `values.len()`
            // u32 slots (its verified word count), is disjoint from the source
            // slice, and no command references it yet.
            unsafe {
                ptr::copy_nonoverlapping(
                    values.as_ptr(),
                    buffer.contents().as_ptr().cast(),
                    values.len(),
                );
            }
        }
        // Compaction has no offsets: its slot 1 is the output. Outputs and
        // scratch start zeroed (Metal zero-fills new buffers); every read word
        // is written by the kernels first.
        let compact = matches!(plan.ir().family, seg::Family::Compact { .. });
        let command = command_buffer(&self.queue)?;
        for (launch, pipeline) in launches.iter().zip(&pipelines) {
            let bound = launch
                .buffers
                .iter()
                .map(|&slot| &*buffers[slot])
                .collect::<Vec<_>>();
            self.encode(&command, pipeline, &bound, launch.blocks, seg::LANES)?;
        }
        let mut work = MetalWork::prepared(command.clone(), buffers.clone(), pipelines);
        work.submit()?;
        if fault == Fault::PartialCompletion {
            self.quarantine(work);
            return Err(Error::Pending);
        }
        let work = self.await_terminal(work, Instant::now())?;
        if work.state != WorkState::Completed {
            return Err(Error::Device(format!(
                "Metal segmented command failed: {:?}",
                command.error()
            )));
        }
        let kernel_ms = (command.GPUEndTime() - command.GPUStartTime()) * 1000.0;
        let output_slot = if compact { 1 } else { 2 };
        let length = if compact {
            // SAFETY: Completed is terminal; slot 4 is a retained one-word buffer.
            let total = unsafe { ptr::read(buffers[4].contents().as_ptr().cast::<u32>()) };
            if total as usize > words[output_slot] {
                return Err(Error::Device("compaction total exceeds its input".into()));
            }
            total as usize
        } else {
            words[output_slot]
        };
        let mut output = Vec::new();
        output
            .try_reserve_exact(length)
            .map_err(|_| Error::OutOfMemory)?;
        output.resize(length, 0u32);
        // SAFETY: Completed is terminal; the retained output buffer holds at
        // least `length` u32 slots (`length` is bounded by its word count).
        unsafe {
            ptr::copy_nonoverlapping(
                buffers[output_slot].contents().as_ptr().cast(),
                output.as_mut_ptr(),
                length,
            );
        }
        Ok(SegmentedResult {
            output,
            source_hash: source_hash(&source),
            compile_ms,
            kernel_ms,
            transfer_bytes,
        })
    }
}

#[cfg(test)]
#[path = "segmented_boundaries.rs"]
mod boundaries;
