//! Owned map, DAG and collective submissions; only Completed work publishes.

use super::*;

pub struct MapResult {
    pub values: Vec<u32>,
    pub source_hash: u64,
    pub compile_ms: f64,
    pub kernel_ms: f64,
    pub host_to_host_ms: f64,
    /// Host copies into/out of the shared buffers, excluding allocation.
    pub transfer_ms: f64,
    pub transfer_bytes: usize,
    pub cache_hit: bool,
}

pub struct FusedResult {
    pub outputs: Vec<Vec<u32>>,
    pub source_hash: u64,
    pub compile_ms: f64,
    pub kernel_ms: f64,
    pub host_to_host_ms: f64,
    /// Host copies into/out of the shared buffers, excluding allocation.
    pub transfer_ms: f64,
    pub transfer_bytes: usize,
    pub cache_hit: bool,
}

pub struct CollectiveResult {
    pub final_state: u32,
    pub pair: Option<[u32; 2]>,
    pub prefixes: Vec<u32>,
    pub source_hash: u64,
    pub compile_ms: f64,
    pub kernel_ms: f64,
    pub host_to_host_ms: f64,
    pub transfer_bytes: usize,
    pub cache_hit: bool,
}

/// Every retained buffer one collective command touches.
pub(super) struct CollectiveBuffers {
    pub(super) inputs: Vec<BufferRef>,
    pub(super) summary: BufferRef,
    pub(super) prefix: BufferRef,
    pub(super) output: BufferRef,
    pub(super) seed: BufferRef,
    pub(super) n: BufferRef,
    pub(super) blocks: BufferRef,
    /// Scan only: first block index of each chunk, one u32 per chunk.
    pub(super) bases: BufferRef,
    /// Scan only: chunks + 1 carry pairs; slot 0 holds the identity.
    pub(super) carries: BufferRef,
}

impl CollectiveBuffers {
    pub(super) fn all(&self) -> Vec<BufferRef> {
        let mut buffers = self.inputs.clone();
        buffers.extend([
            self.summary.clone(),
            self.prefix.clone(),
            self.output.clone(),
            self.seed.clone(),
            self.n.clone(),
            self.blocks.clone(),
            self.bases.clone(),
            self.carries.clone(),
        ]);
        buffers
    }
}

/// Scan chunks for `blocks` block summaries.
pub(super) fn scan_chunks(blocks: u32) -> usize {
    blocks
        .div_ceil(invar_device::METAL_SCAN_CHUNK_BLOCKS)
        .max(1) as usize
}

/// Device bytes a collective allocates: inputs, two summary arrays, output,
/// three u32 scalars, chunk bases and carries.
pub(super) fn collective_bytes(
    ir: &invar_device::Ir,
    input_count: usize,
    scan: bool,
) -> Option<usize> {
    let collective = ir.collective.as_ref()?;
    if input_count != collective.inputs.len()
        || scan
            != matches!(
                collective.kind,
                CollectiveKind::AffineScan { .. } | CollectiveKind::PairReduction { .. }
            )
    {
        return None;
    }
    (ir.length as usize)
        .checked_mul(input_count)?
        .checked_mul(4)?
        .checked_add((collective.output.elements as usize).checked_mul(4)?)?
        .checked_add(invar_device::collective_workspace(ir, true)?)
}

impl MetalDevice {
    pub fn run_map(&mut self, plan: &Verified, input: &[u32]) -> Result<MapResult, Error> {
        self.run_map_with_fault(plan, input, Fault::None)
    }

    pub fn run_map_with_fault(
        &mut self,
        plan: &Verified,
        input: &[u32],
        fault: Fault,
    ) -> Result<MapResult, Error> {
        if plan.ir().dag.is_some() || plan.ir().collective.is_some() {
            return Err(Error::Limit);
        }
        let result = self.run_fused_map_with_fault(plan, &[input], fault)?;
        let values = result.outputs.into_iter().next().ok_or(Error::Limit)?;
        Ok(MapResult {
            values,
            source_hash: result.source_hash,
            compile_ms: result.compile_ms,
            kernel_ms: result.kernel_ms,
            host_to_host_ms: result.host_to_host_ms,
            transfer_ms: result.transfer_ms,
            transfer_bytes: result.transfer_bytes,
            cache_hit: result.cache_hit,
        })
    }

    pub fn run_fused_map(
        &mut self,
        plan: &Verified,
        inputs: &[&[u32]],
    ) -> Result<FusedResult, Error> {
        self.run_fused_map_with_fault(plan, inputs, Fault::None)
    }

    pub fn run_fused_map_with_fault(
        &mut self,
        plan: &Verified,
        inputs: &[&[u32]],
        fault: Fault,
    ) -> Result<FusedResult, Error> {
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
        let ir = plan.ir();
        let input_count = ir.dag.as_ref().map_or(1, |dag| dag.inputs.len());
        let output_count = ir.dag.as_ref().map_or(1, |dag| dag.outputs.len());
        if plan.capabilities() != self.caps
            || ir.collective.is_some()
            || inputs.len() != input_count
            || inputs.iter().any(|input| input.len() != ir.length as usize)
        {
            return Err(Error::Limit);
        }
        let bytes = (ir.length as usize).checked_mul(4).ok_or(Error::Limit)?;
        let total = input_count.checked_add(output_count).ok_or(Error::Limit)?;
        let device_bytes = bytes
            .max(4)
            .checked_mul(total)
            .and_then(|n| n.checked_add(4))
            .ok_or(Error::ResourceLimit("map byte count overflow"))?;
        if (bytes as u64) * (total as u64) > invar_device::MAX_DAG_TRANSFER_BYTES {
            return Err(Error::ResourceLimit("map exceeds 8 GiB transfer budget"));
        }
        self.check_submission(bytes.max(4), device_bytes)?;
        let source =
            emit_msl(plan).map_err(|e| Error::Toolchain(format!("MSL emission: {e:?}")))?;
        if ir.length == 0 {
            return Ok(FusedResult {
                outputs: vec![Vec::new(); output_count],
                source_hash: source_hash(&source),
                compile_ms: 0.0,
                kernel_ms: 0.0,
                host_to_host_ms: 0.0,
                transfer_ms: 0.0,
                transfer_bytes: 0,
                cache_hit: false,
            });
        }
        let started = Instant::now();
        let (pipeline, cache_hit) = self.compile(&source, "invar_map_v1")?;
        let compile_ms = started.elapsed().as_secs_f64() * 1000.0;
        let host_started = Instant::now();
        let input_buffers = (0..input_count)
            .map(|_| self.buffer(bytes.max(4)))
            .collect::<Result<Vec<_>, _>>()?;
        let output_buffers = (0..output_count)
            .map(|_| self.buffer(bytes.max(4)))
            .collect::<Result<Vec<_>, _>>()?;
        let n_buffer = self.buffer(4)?;
        let copying = Instant::now();
        // SAFETY: shared Metal storage is at least `bytes` long, has u32 alignment,
        // and no command has been submitted yet. The hardware test compares
        // every copied word after a terminal command-buffer state.
        for (input, buffer) in inputs.iter().zip(&input_buffers) {
            // SAFETY: each disjoint retained shared buffer has `bytes` writable
            // bytes and has not been submitted to the queue.
            unsafe {
                ptr::copy_nonoverlapping(
                    input.as_ptr(),
                    buffer.contents().as_ptr().cast(),
                    input.len(),
                );
            }
        }
        let mut transfer_ms = copying.elapsed().as_secs_f64() * 1000.0;
        // SAFETY: n_buffer owns one u32 and no command has been submitted.
        unsafe {
            ptr::write(n_buffer.contents().as_ptr().cast::<u32>(), ir.length);
        }
        let command = self
            .queue
            .commandBuffer()
            .ok_or_else(|| Error::Unavailable("Metal command buffer unavailable".into()))?;
        let encoder = command
            .computeCommandEncoder()
            .ok_or_else(|| Error::Unavailable("Metal compute encoder unavailable".into()))?;
        encoder.setComputePipelineState(&pipeline);
        for (index, buffer) in input_buffers.iter().chain(&output_buffers).enumerate() {
            // SAFETY: the verified map/DAG ABI assigns consecutive input then
            // output slots at offset zero; each buffer holds `bytes` and stays
            // retained by MetalWork until a terminal command-buffer state.
            unsafe { encoder.setBuffer_offset_atIndex(Some(buffer), 0, index) };
        }
        // SAFETY: `n` follows the last output slot in the same verified ABI;
        // n_buffer holds one u32 and is retained by MetalWork like the others.
        unsafe { encoder.setBuffer_offset_atIndex(Some(&n_buffer), 0, total) };
        let lanes = ir.geometry.lanes as usize;
        if lanes > pipeline.maxTotalThreadsPerThreadgroup() {
            return Err(Error::Toolchain(
                "Metal pipeline threadgroup limit is below verified geometry".into(),
            ));
        }
        encoder.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize {
                width: ir.geometry.blocks as usize,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: lanes,
                height: 1,
                depth: 1,
            },
        );
        encoder.endEncoding();
        let mut buffers = input_buffers
            .iter()
            .chain(&output_buffers)
            .cloned()
            .collect::<Vec<_>>();
        buffers.push(n_buffer.clone());
        let mut work = MetalWork::prepared(command.clone(), buffers, vec![pipeline.clone()]);
        work.submit()?;
        if fault == Fault::PartialCompletion {
            // Submission succeeded, but completion was intentionally withheld.
            self.quarantine(work);
            return Err(Error::Pending);
        }
        let work = self.await_terminal(work, Instant::now())?;
        if work.state != WorkState::Completed {
            return Err(Error::Device(format!(
                "Metal command failed: {:?}",
                command.error()
            )));
        }
        let kernel_ms = (command.GPUEndTime() - command.GPUStartTime()) * 1000.0;
        let mut outputs = Vec::with_capacity(output_count);
        for buffer in &output_buffers {
            let mut values = Vec::new();
            values
                .try_reserve_exact(ir.length as usize)
                .map_err(|_| Error::ResourceLimit("host output allocation failed"))?;
            values.resize(ir.length as usize, 0u32);
            let copying = Instant::now();
            // SAFETY: Completed is terminal; each retained shared output buffer
            // has at least `values.len()` u32 slots before this copy.
            unsafe {
                ptr::copy_nonoverlapping(
                    buffer.contents().as_ptr().cast(),
                    values.as_mut_ptr(),
                    values.len(),
                );
            }
            transfer_ms += copying.elapsed().as_secs_f64() * 1000.0;
            outputs.push(values);
        }
        Ok(FusedResult {
            outputs,
            source_hash: source_hash(&source),
            compile_ms,
            kernel_ms,
            host_to_host_ms: host_started.elapsed().as_secs_f64() * 1000.0,
            transfer_ms,
            transfer_bytes: bytes * total,
            cache_hit,
        })
    }

    pub fn run_collective(
        &mut self,
        plan: &Verified,
        inputs: &[&[u32]],
        seed: u32,
    ) -> Result<CollectiveResult, Error> {
        self.run_collective_with_fault(plan, inputs, seed, Fault::None)
    }

    pub fn run_collective_with_fault(
        &mut self,
        plan: &Verified,
        inputs: &[&[u32]],
        seed: u32,
        fault: Fault,
    ) -> Result<CollectiveResult, Error> {
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
        let ir = plan.ir();
        let collective = ir.collective.as_ref().ok_or(Error::Limit)?;
        if plan.capabilities() != self.caps
            || inputs.len() != collective.inputs.len()
            || inputs.iter().any(|input| input.len() != ir.length as usize)
        {
            return Err(Error::Limit);
        }
        let prefix_mode = match collective.kind {
            CollectiveKind::Reduction { .. } | CollectiveKind::PairReduction { .. } => {
                PrefixMode::None
            }
            CollectiveKind::AffineScan { prefix, .. } => prefix,
        };
        let scan = matches!(
            collective.kind,
            CollectiveKind::AffineScan { .. } | CollectiveKind::PairReduction { .. }
        );
        let source =
            emit_msl(plan).map_err(|e| Error::Toolchain(format!("MSL emission: {e:?}")))?;
        let hash = source_hash(&source);
        if ir.length == 0 {
            return Ok(CollectiveResult {
                final_state: seed,
                pair: matches!(collective.kind, CollectiveKind::PairReduction { .. })
                    .then_some([1, 0]),
                prefixes: Vec::new(),
                source_hash: hash,
                compile_ms: 0.0,
                kernel_ms: 0.0,
                host_to_host_ms: 0.0,
                transfer_bytes: 0,
                cache_hit: false,
            });
        }
        let started = Instant::now();
        let compiled = ["invar_block_v1", "invar_propagate_v1", "invar_writeback_v1"]
            .map(|name| self.compile(&source, name))
            .into_iter()
            .collect::<Result<Vec<_>, _>>()?;
        let cache_hit = compiled.iter().all(|(_, hit)| *hit);
        let pipelines = compiled
            .into_iter()
            .map(|(pipeline, _)| pipeline)
            .collect::<Vec<_>>();
        let compile_ms = started.elapsed().as_secs_f64() * 1000.0;
        let host_started = Instant::now();
        let length = ir.length as usize;
        let bytes = length.checked_mul(4).ok_or(Error::Limit)?;
        let output_bytes = (collective.output.elements as usize)
            .checked_mul(4)
            .ok_or(Error::Limit)?;
        let owned = self.collective_buffers(ir, inputs.len(), scan, seed)?;
        for (values, buffer) in inputs.iter().zip(&owned.inputs) {
            // SAFETY: exact-length initialized source copied before submission
            // into a disjoint retained shared allocation of `bytes` bytes.
            unsafe {
                ptr::copy_nonoverlapping(
                    values.as_ptr(),
                    buffer.contents().as_ptr().cast(),
                    length,
                );
            }
        }
        let command = self
            .queue
            .commandBuffer()
            .ok_or_else(|| Error::Unavailable("Metal command buffer unavailable".into()))?;
        self.encode_collective(&command, ir, &pipelines, &owned, scan)?;
        let output = owned.output.clone();
        let buffers = owned.all();
        let mut work = MetalWork::prepared(command.clone(), buffers, pipelines.clone());
        work.submit()?;
        if fault == Fault::PartialCompletion {
            self.quarantine(work);
            return Err(Error::Pending);
        }
        let work = self.await_terminal(work, Instant::now())?;
        if work.state != WorkState::Completed {
            return Err(Error::Device(format!(
                "Metal collective failed: {:?}",
                command.error()
            )));
        }
        let kernel_ms = (command.GPUEndTime() - command.GPUStartTime()) * 1000.0;
        let mut values = Vec::new();
        values
            .try_reserve_exact(collective.output.elements as usize)
            .map_err(|_| Error::ResourceLimit("host output allocation failed"))?;
        values.resize(collective.output.elements as usize, 0u32);
        // SAFETY: Completed confirms all three ordered encoders finished; the
        // output buffer remains retained and covers every copied u32.
        unsafe {
            ptr::copy_nonoverlapping(
                output.contents().as_ptr().cast(),
                values.as_mut_ptr(),
                values.len(),
            );
        }
        let pair = if matches!(collective.kind, CollectiveKind::PairReduction { .. }) {
            Some([
                *values.first().ok_or(Error::Limit)?,
                *values.get(1).ok_or(Error::Limit)?,
            ])
        } else {
            None
        };
        let final_state = values.pop().ok_or(Error::Limit)?;
        let prefixes = if scan && prefix_mode != PrefixMode::None {
            values
        } else {
            Vec::new()
        };
        Ok(CollectiveResult {
            final_state,
            pair,
            prefixes,
            source_hash: hash,
            compile_ms,
            kernel_ms,
            host_to_host_ms: host_started.elapsed().as_secs_f64() * 1000.0,
            transfer_bytes: bytes * inputs.len() + output_bytes,
            cache_hit,
        })
    }

    /// Allocate one collective's retained storage and write its scalars,
    /// chunk bases and identity carry. Inputs are left for the caller.
    pub(super) fn collective_buffers(
        &self,
        ir: &invar_device::Ir,
        input_count: usize,
        scan: bool,
        seed: u32,
    ) -> Result<CollectiveBuffers, Error> {
        let collective = ir.collective.as_ref().ok_or(Error::Limit)?;
        let bytes = (ir.length as usize).checked_mul(4).ok_or(Error::Limit)?;
        let output_bytes =
            (collective.output.elements as usize)
                .checked_mul(4)
                .ok_or(Error::ResourceLimit(
                    "collective output byte count overflow",
                ))?;
        let device_bytes = collective_bytes(ir, input_count, scan)
            .ok_or(Error::ResourceLimit("collective byte count overflow"))?;
        self.check_submission(bytes.max(output_bytes).max(4), device_bytes)?;
        let summary_bytes = (ir.geometry.blocks as usize)
            .checked_mul(if scan { 8 } else { 4 })
            .ok_or(Error::Limit)?;
        let chunks = scan_chunks(ir.geometry.blocks);
        let buffers = CollectiveBuffers {
            inputs: (0..input_count)
                .map(|_| self.buffer(bytes.max(4)))
                .collect::<Result<Vec<_>, _>>()?,
            summary: self.buffer(summary_bytes)?,
            prefix: self.buffer(summary_bytes)?,
            output: self.buffer(output_bytes)?,
            seed: self.buffer(4)?,
            n: self.buffer(4)?,
            blocks: self.buffer(4)?,
            bases: self.buffer(chunks * 4)?,
            carries: self.buffer((chunks + 1) * 8)?,
        };
        let step = invar_device::METAL_SCAN_CHUNK_BLOCKS;
        // SAFETY: each retained shared buffer was just allocated with room for
        // the typed values written here, and no command references it yet.
        unsafe {
            ptr::write(buffers.seed.contents().as_ptr().cast::<u32>(), seed);
            ptr::write(buffers.n.contents().as_ptr().cast::<u32>(), ir.length);
            ptr::write(
                buffers.blocks.contents().as_ptr().cast::<u32>(),
                ir.geometry.blocks,
            );
            let bases = std::slice::from_raw_parts_mut(
                buffers.bases.contents().as_ptr().cast::<u32>(),
                chunks,
            );
            for (chunk, base) in bases.iter_mut().enumerate() {
                *base = chunk as u32 * step;
            }
            // Carry slot 0 is the affine identity (a = 1, b = 0).
            ptr::write(
                buffers.carries.contents().as_ptr().cast::<[u32; 2]>(),
                [1, 0],
            );
        }
        Ok(buffers)
    }

    /// Encode the three ordered collective passes. A reduction uses one
    /// encoder per pass. A scan runs block, propagate and writeback per chunk
    /// in one serial encoder, so each chunk's second input read is cache-hot.
    pub(super) fn encode_collective(
        &self,
        command: &ProtocolObject<dyn MTLCommandBuffer>,
        ir: &invar_device::Ir,
        pipelines: &[PipelineRef],
        buffers: &CollectiveBuffers,
        scan: bool,
    ) -> Result<(), Error> {
        let [block, propagate, writeback] = pipelines else {
            return Err(Error::Limit);
        };
        let inputs = buffers.inputs.iter().map(|b| &**b).collect::<Vec<_>>();
        if !scan {
            let mut first = inputs.clone();
            first.extend([&*buffers.summary, &*buffers.n]);
            self.encode(
                command,
                block,
                &first,
                ir.geometry.blocks,
                ir.geometry.lanes,
            )?;
            let second = [&*buffers.summary, &*buffers.prefix, &*buffers.blocks];
            self.encode(command, propagate, &second, 1, ir.geometry.lanes)?;
            let third = [&*buffers.prefix, &*buffers.output, &*buffers.seed];
            return self.encode(command, writeback, &third, 1, 1);
        }
        for (pipeline, lanes) in [
            (block, ir.geometry.lanes),
            (propagate, 1024),
            (writeback, ir.geometry.lanes),
        ] {
            if lanes as usize > pipeline.maxTotalThreadsPerThreadgroup() {
                return Err(Error::Toolchain(
                    "Metal pipeline threadgroup limit is below verified geometry".into(),
                ));
            }
        }
        let encoder = command
            .computeCommandEncoder()
            .ok_or_else(|| Error::Unavailable("Metal compute encoder unavailable".into()))?;
        let slots = inputs.len();
        let step = invar_device::METAL_SCAN_CHUNK_BLOCKS;
        let size = |width: usize| MTLSize {
            width,
            height: 1,
            depth: 1,
        };
        for chunk in 0..scan_chunks(ir.geometry.blocks) {
            let groups = (ir.geometry.blocks - chunk as u32 * step).min(step) as usize;
            let bind = |buffer: &ProtocolObject<dyn MTLBuffer>, offset: usize, index: usize| {
                // SAFETY: every buffer is retained by the submitted work until a
                // terminal state; `offset` addresses this chunk's u32 base (4 B
                // stride) or carry pair (8 B stride) inside the allocation made
                // by `collective_buffers`, whose MSL signature fixes `index`.
                unsafe { encoder.setBuffer_offset_atIndex(Some(buffer), offset, index) };
            };
            encoder.setComputePipelineState(block);
            for (index, input) in inputs.iter().enumerate() {
                bind(input, 0, index);
            }
            bind(&buffers.summary, 0, slots);
            bind(&buffers.n, 0, slots + 1);
            bind(&buffers.bases, chunk * 4, slots + 2);
            encoder.dispatchThreadgroups_threadsPerThreadgroup(
                size(groups),
                size(ir.geometry.lanes as usize),
            );
            encoder.setComputePipelineState(propagate);
            bind(&buffers.summary, 0, 0);
            bind(&buffers.prefix, 0, 1);
            bind(&buffers.blocks, 0, 2);
            bind(&buffers.bases, chunk * 4, 3);
            bind(&buffers.carries, chunk * 8, 4);
            encoder.dispatchThreadgroups_threadsPerThreadgroup(size(1), size(1024));
            encoder.setComputePipelineState(writeback);
            for (index, input) in inputs.iter().enumerate() {
                bind(input, 0, index);
            }
            bind(&buffers.summary, 0, slots);
            bind(&buffers.prefix, 0, slots + 1);
            bind(&buffers.output, 0, slots + 2);
            bind(&buffers.seed, 0, slots + 3);
            bind(&buffers.n, 0, slots + 4);
            bind(&buffers.blocks, 0, slots + 5);
            bind(&buffers.bases, chunk * 4, slots + 6);
            encoder.dispatchThreadgroups_threadsPerThreadgroup(
                size(groups),
                size(ir.geometry.lanes as usize),
            );
        }
        encoder.endEncoding();
        Ok(())
    }
}
