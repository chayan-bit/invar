//! M4-02 tiled word transpose and ordered, out-of-place u32 stencil phases.
use super::*;
use invar_device::{stencil_u32, transpose};

pub struct GridResult {
    pub output: Vec<u32>,
    pub target: &'static str,
    pub source_hash: u64,
    pub compile_ms: f64,
    pub host_to_host_ms: f64,
    pub transfer_bytes: usize,
    pub phases: u32,
    pub final_buffer: usize,
}

impl MetalDevice {
    pub fn run_transpose(
        &mut self,
        plan: &transpose::Verified,
        input: &[u32],
    ) -> Result<GridResult, Error> {
        self.run_transpose_with_fault(plan, input, Fault::None)
    }
    pub fn run_transpose_with_fault(
        &mut self,
        plan: &transpose::Verified,
        input: &[u32],
        fault: Fault,
    ) -> Result<GridResult, Error> {
        self.admit()?;
        if plan.capabilities() != self.caps {
            return Err(Error::Limit);
        }
        let p = plan.ir();
        let source = transpose::emit_msl(plan).map_err(|e| Error::Toolchain(format!("{e:?}")))?;
        let logical = u64::from(p.dimensions[0]) * u64::from(p.dimensions[1]);
        let output = self.run_grid(
            &source,
            transpose::KERNEL_NAME,
            input,
            [p.input.elements, p.output.elements],
            [p.input.alignment, p.output.alignment],
            if logical == 0 { 0 } else { 1 },
            false,
            p.dimensions,
            fault,
        )?;
        let start = usize::try_from(p.output.offset).map_err(|_| Error::Limit)?;
        let end = start
            .checked_add(usize::try_from(logical).map_err(|_| Error::Limit)?)
            .ok_or(Error::Limit)?;
        if output.output[..start]
            .iter()
            .chain(output.output[end..].iter())
            .any(|&x| x != 0)
        {
            return Err(Error::Device("transpose padding modified".into()));
        }
        Ok(output)
    }
    pub fn run_stencil_u32(
        &mut self,
        plan: &stencil_u32::Verified,
        input: &[u32],
    ) -> Result<GridResult, Error> {
        self.run_stencil_u32_with_fault(plan, input, Fault::None)
    }
    pub fn run_stencil_u32_with_fault(
        &mut self,
        plan: &stencil_u32::Verified,
        input: &[u32],
        fault: Fault,
    ) -> Result<GridResult, Error> {
        self.admit()?;
        if plan.capabilities() != self.caps {
            return Err(Error::Limit);
        }
        let p = plan.ir();
        let source = stencil_u32::emit_msl(plan).map_err(|e| Error::Toolchain(format!("{e:?}")))?;
        let steps = if p.dimensions.contains(&0) {
            0
        } else {
            p.steps
        };
        let output = self.run_grid(
            &source,
            stencil_u32::KERNEL_NAME,
            input,
            [p.input.elements, p.output.elements],
            [p.input.alignment, p.output.alignment],
            steps,
            true,
            p.dimensions,
            fault,
        )?;
        let start = usize::try_from(p.output.offset).map_err(|_| Error::Limit)?;
        let end = start
            .checked_add((u64::from(p.dimensions[0]) * u64::from(p.dimensions[1])) as usize)
            .ok_or(Error::Limit)?;
        for (i, &word) in output.output.iter().enumerate() {
            if (i < start || i >= end) && word != if steps.is_multiple_of(2) { input[i] } else { 0 }
            {
                return Err(Error::Device("stencil padding modified".into()));
            }
        }
        Ok(output)
    }
    #[allow(clippy::too_many_arguments)] // One private submission boundary for both fixed two-buffer kernels.
    fn run_grid(
        &mut self,
        source: &str,
        name: &str,
        input: &[u32],
        lengths: [u64; 2],
        alignments: [u64; 2],
        steps: u32,
        ping_pong: bool,
        dimensions: [u32; 2],
        fault: Fault,
    ) -> Result<GridResult, Error> {
        let lengths = [
            usize::try_from(lengths[0]).map_err(|_| Error::Limit)?,
            usize::try_from(lengths[1]).map_err(|_| Error::Limit)?,
        ];
        let bytes = [
            lengths[0].checked_mul(4).ok_or(Error::Limit)?,
            lengths[1].checked_mul(4).ok_or(Error::Limit)?,
        ];
        if lengths[0] != input.len() {
            return Err(Error::Limit);
        }
        if bytes[0]
            .checked_add(bytes[1])
            .and_then(|n| n.checked_mul(2))
            .and_then(|n| n.checked_add(bytes[1]))
            .is_none_or(|n| n as u64 > transpose::MAX_OWNED_BYTES)
        {
            return Err(Error::ResourceLimit("M4-02 owned storage exceeds 5 GiB"));
        }
        // Query before reserving host output or allocating shared GPU storage.
        if bytes.iter().any(|&n| n > self.device.maxBufferLength()) {
            return Err(Error::ResourceLimit(
                "M4-02 buffer exceeds Metal maxBufferLength",
            ));
        }
        match fault {
            Fault::OutOfMemory => return Err(Error::OutOfMemory),
            Fault::LaunchFailure => {
                return Err(Error::Device("injected pre-submission rejection".into()));
            }
            _ => {}
        }
        let mut output = Vec::new();
        output
            .try_reserve_exact(lengths[1])
            .map_err(|_| Error::OutOfMemory)?;
        output.resize(lengths[1], 0);
        let hash = source_hash(source);
        if steps == 0 {
            if ping_pong {
                output.copy_from_slice(input);
            }
            return Ok(GridResult {
                output,
                target: "metal",
                source_hash: hash,
                compile_ms: 0.0,
                host_to_host_ms: 0.0,
                transfer_bytes: 0,
                phases: 0,
                final_buffer: 0,
            });
        }
        let begun = Instant::now();
        let (pipeline, _) = self.compile(source, name)?;
        let compile_ms = begun.elapsed().as_secs_f64() * 1000.0;
        if pipeline.maxTotalThreadsPerThreadgroup() < 256
            || pipeline.staticThreadgroupMemoryLength() > self.device.maxThreadgroupMemoryLength()
        {
            return Err(Error::Limit);
        }
        let begun = Instant::now();
        let buffers = [self.buffer(bytes[0])?, self.buffer(bytes[1])?];
        for (buffer, alignment) in buffers.iter().zip(alignments) {
            if !(buffer.contents().as_ptr() as u64).is_multiple_of(alignment) {
                return Err(Error::Limit);
            }
        }
        // SAFETY: fresh disjoint shared buffers cover the exact byte counts;
        // synchronous initialization occurs before any encoder is committed.
        unsafe {
            ptr::copy_nonoverlapping(
                input.as_ptr().cast::<u8>(),
                buffers[0].contents().as_ptr().cast::<u8>(),
                bytes[0],
            );
            ptr::write_bytes(buffers[1].contents().as_ptr().cast::<u8>(), 0, bytes[1]);
        }
        let command = self
            .queue
            .commandBuffer()
            .ok_or_else(|| Error::Unavailable("Metal command unavailable".into()))?;
        let blocks = (u64::from(dimensions[0]).div_ceil(16) * u64::from(dimensions[1]).div_ceil(16))
            .clamp(1, 65535) as usize;
        for step in 0..steps {
            let read = if ping_pong { (step % 2) as usize } else { 0 };
            let encoder = command
                .computeCommandEncoder()
                .ok_or_else(|| Error::Unavailable("Metal encoder unavailable".into()))?;
            encoder.setComputePipelineState(&pipeline);
            for (slot, b) in [&buffers[read], &buffers[1 - read]].into_iter().enumerate() {
                // SAFETY: sealed sources bind two fully initialized uint buffers
                // at slots 0/1, offset zero, retained through terminal completion.
                unsafe {
                    encoder.setBuffer_offset_atIndex(Some(b), 0, slot);
                }
            }
            encoder.dispatchThreadgroups_threadsPerThreadgroup(
                MTLSize {
                    width: blocks,
                    height: 1,
                    depth: 1,
                },
                MTLSize {
                    width: 16,
                    height: 16,
                    depth: 1,
                },
            );
            encoder.endEncoding();
            // Separate ordered encoders establish each complete time slice.
        }
        let mut work = MetalWork::prepared(command, buffers.to_vec(), vec![pipeline]);
        work.submit()?;
        if fault == Fault::PartialCompletion {
            self.quarantine(work);
            return Err(Error::Pending);
        }
        let work = self.await_terminal(work, begun)?;
        if work.state != WorkState::Completed {
            return Err(Error::Device("Metal grid failed quiescent".into()));
        }
        let final_buffer = if ping_pong { (steps % 2) as usize } else { 1 };
        // SAFETY: actual terminal completion precedes read. The retained shared
        // buffer covers output length and no asynchronous access remains.
        unsafe {
            ptr::copy_nonoverlapping(
                work._buffers[final_buffer]
                    .contents()
                    .as_ptr()
                    .cast::<u32>(),
                output.as_mut_ptr(),
                output.len(),
            );
        }
        Ok(GridResult {
            output,
            target: "metal",
            source_hash: hash,
            compile_ms,
            host_to_host_ms: begun.elapsed().as_secs_f64() * 1000.0,
            transfer_bytes: bytes[0] + 2 * bytes[1],
            phases: steps,
            final_buffer,
        })
    }
}
