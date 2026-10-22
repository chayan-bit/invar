//! P4a path-sum launcher: ordered stages of one source in one command buffer,
//! with `(args, part, out, stops)` bound at buffers 0..3 and 256-thread groups.
//! Compiler-generated and handwritten (#140) kernels share it, so both sides of
//! a gate comparison pay the same encode, submit, wait and read.
use super::*;

pub const PATHSUM_GROUP: usize = 256;
const ARG_BYTES: usize = 64;

/// `stops` bytes for a launcher of `groups` groups: one word per group, since
/// kernels may write `stops[GROUP]`, and never fewer than one group's reduce width.
fn stop_bytes(groups: usize) -> Option<usize> {
    groups.max(PATHSUM_GROUP).checked_mul(4)
}

pub struct PathSum {
    queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
    pipelines: Vec<PipelineRef>,
    args: BufferRef,
    part: BufferRef,
    out: BufferRef,
    stops: BufferRef,
    groups: usize,
    out_bytes: usize,
}

impl MetalDevice {
    /// Compile `kernels` of `source` for at most `groups` groups per stage,
    /// each publishing one `element_bytes` partial; `out` holds one element.
    pub fn pathsum(
        &mut self,
        source: &str,
        kernels: &[&str],
        groups: usize,
        element_bytes: usize,
    ) -> Result<PathSum, Error> {
        self.admit()?;
        let part_bytes = groups.checked_mul(element_bytes).ok_or(Error::Limit)?;
        if source.len() > invar_device::MAX_GNET_SOURCE_BYTES
            || part_bytes == 0
            || !element_bytes.is_multiple_of(4)
        {
            return Err(Error::Limit);
        }
        let out_bytes = element_bytes;
        let pipelines = kernels
            .iter()
            .map(|name| self.compile(source, name).map(|(pipeline, _)| pipeline))
            .collect::<Result<Vec<_>, _>>()?;
        if pipelines
            .iter()
            .any(|p| p.maxTotalThreadsPerThreadgroup() < PATHSUM_GROUP)
        {
            return Err(Error::Limit);
        }
        let buffer = |bytes: usize| -> Result<BufferRef, Error> {
            let buffer = self.buffer(bytes)?;
            // SAFETY: a new shared allocation owns `bytes` writable bytes and
            // no command can reference it yet.
            unsafe { ptr::write_bytes(buffer.contents().as_ptr().cast::<u8>(), 0, bytes) };
            Ok(buffer)
        };
        Ok(PathSum {
            queue: self.queue.clone(),
            pipelines,
            args: buffer(ARG_BYTES)?,
            part: buffer(part_bytes)?,
            out: buffer(out_bytes)?,
            stops: buffer(stop_bytes(groups).ok_or(Error::Limit)?)?,
            groups,
            out_bytes,
        })
    }
}

impl PathSum {
    pub fn write_args(&mut self, args: &[u8]) -> Result<(), Error> {
        if args.len() > ARG_BYTES {
            return Err(Error::Limit);
        }
        let block = self.args.contents().as_ptr().cast::<u8>();
        // SAFETY: no command is in flight (dispatch waits for completion) and
        // the shared allocation covers `ARG_BYTES >= args.len()` bytes. The whole
        // block is cleared first, so no tail word of an earlier write survives.
        unsafe {
            ptr::write_bytes(block, 0, ARG_BYTES);
            ptr::copy_nonoverlapping(args.as_ptr(), block, args.len());
        };
        Ok(())
    }

    /// Encode `(kernel index, groups)` stages in order, wait, and return the
    /// GPU execution interval in milliseconds.
    pub fn dispatch(&mut self, stages: &[(usize, usize)]) -> Result<f64, Error> {
        let command = self
            .queue
            .commandBuffer()
            .ok_or_else(|| Error::Unavailable("Metal command buffer unavailable".into()))?;
        for &(kernel, groups) in stages {
            let pipeline = self.pipelines.get(kernel).ok_or(Error::Limit)?;
            if groups == 0 || groups > self.groups {
                return Err(Error::Limit);
            }
            let encoder = command
                .computeCommandEncoder()
                .ok_or_else(|| Error::Unavailable("Metal compute encoder unavailable".into()))?;
            encoder.setComputePipelineState(pipeline);
            for (i, buffer) in [&self.args, &self.part, &self.out, &self.stops]
                .into_iter()
                .enumerate()
            {
                // SAFETY: every path-sum source declares at most these four
                // buffers at slots 0..3; each is retained by `self`, which
                // outlives the command (dispatch waits for completion).
                unsafe { encoder.setBuffer_offset_atIndex(Some(buffer), 0, i) };
            }
            encoder.dispatchThreadgroups_threadsPerThreadgroup(
                MTLSize {
                    width: groups,
                    height: 1,
                    depth: 1,
                },
                MTLSize {
                    width: PATHSUM_GROUP,
                    height: 1,
                    depth: 1,
                },
            );
            encoder.endEncoding();
        }
        command.commit();
        command.waitUntilCompleted();
        if command.status() != MTLCommandBufferStatus::Completed {
            return Err(Error::Device(format!(
                "Metal path-sum command failed: {:?}",
                command.error()
            )));
        }
        let ms = (command.GPUEndTime() - command.GPUStartTime()) * 1000.0;
        if !ms.is_finite() || ms <= 0.0 {
            return Err(Error::Device("Metal GPU clock returned no interval".into()));
        }
        Ok(ms)
    }

    /// The first `words` u32 words of the output buffer.
    pub fn read_out(&self, words: usize) -> Result<Vec<u32>, Error> {
        if words * 4 > self.out_bytes {
            return Err(Error::Limit);
        }
        let mut out = vec![0u32; words];
        // SAFETY: dispatch observed completion; the shared allocation covers
        // `out_bytes >= 4 * words` initialized bytes.
        unsafe {
            ptr::copy_nonoverlapping(
                self.out.contents().as_ptr().cast::<u32>(),
                out.as_mut_ptr(),
                words,
            )
        };
        Ok(out)
    }
}
