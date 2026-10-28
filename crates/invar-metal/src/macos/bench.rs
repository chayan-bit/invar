//! Resident bandwidth benchmarks and host round-trip calibration.

use super::*;
use run::{CollectiveBuffers, collective_bytes};

#[derive(Debug)]
pub struct BenchResult {
    pub copy_median_ms: f64,
    pub generated_median_ms: f64,
    pub copy_gbps: f64,
    pub generated_gbps: f64,
    pub bandwidth_fraction: f64,
    pub source_hash: u64,
    pub length: usize,
    pub warmups: usize,
    pub samples: usize,
    pub outputs_correct: bool,
    /// Bytes the fraction is charged: inputs read once plus outputs written once.
    pub required_bytes: usize,
    /// Bytes the kernels request, including re-reads and temporaries.
    pub logical_bytes: usize,
}

pub(super) fn median(mut values: Vec<f64>) -> Result<f64, Error> {
    if values.len() != 30
        || values
            .iter()
            .any(|value| !value.is_finite() || *value <= 0.0)
    {
        return Err(Error::Device("invalid Metal GPU timing".into()));
    }
    values.sort_by(f64::total_cmp);
    Ok((values[14] + values[15]) / 2.0)
}

/// Milliseconds between two GPU clock reads, or `None` when the clock gave no
/// usable interval (a very short command can report equal or reversed times).
pub(super) fn gpu_interval_ms(start: f64, end: f64) -> Option<f64> {
    let ms = (end - start) * 1000.0;
    (ms.is_finite() && ms > 0.0).then_some(ms)
}

impl MetalDevice {
    pub(super) fn timed_command(
        &mut self,
        buffers: &[BufferRef],
        pipeline: Option<&PipelineRef>,
        blocks: u32,
        lanes: u32,
        bytes: usize,
    ) -> Result<f64, Error> {
        self.timed_command_checked(buffers, pipeline, blocks, lanes, bytes)?
            .ok_or_else(|| Error::Device("Metal GPU clock returned no interval".into()))
    }

    /// `Err` only when the command did not reach `Completed`; a completed
    /// command whose GPU clock gave no interval is `Ok(None)`.
    pub(super) fn timed_command_checked(
        &mut self,
        buffers: &[BufferRef],
        pipeline: Option<&PipelineRef>,
        blocks: u32,
        lanes: u32,
        bytes: usize,
    ) -> Result<Option<f64>, Error> {
        self.admit()?;
        let command = command_buffer(&self.queue)?;
        if let Some(pipeline) = pipeline {
            let refs = buffers.iter().map(|buffer| &**buffer).collect::<Vec<_>>();
            self.encode(&command, pipeline, &refs, blocks, lanes)?;
        } else {
            let encoder = command
                .blitCommandEncoder()
                .ok_or_else(|| Error::Unavailable("Metal blit encoder unavailable".into()))?;
            // SAFETY: the benchmark passes two retained buffers each covering
            // `bytes`, with zero offset, and waits for a terminal status.
            unsafe {
                encoder.copyFromBuffer_sourceOffset_toBuffer_destinationOffset_size(
                    &buffers[0],
                    0,
                    &buffers[1],
                    0,
                    bytes,
                );
            }
            encoder.endEncoding();
        }
        let mut work = MetalWork::prepared(
            command.clone(),
            buffers.to_vec(),
            pipeline.cloned().into_iter().collect(),
        );
        work.submit()?;
        let work = self.await_terminal(work, Instant::now())?;
        if work.state != WorkState::Completed {
            return Err(Error::Device(format!(
                "Metal benchmark command failed: {:?}",
                command.error()
            )));
        }
        Ok(gpu_interval_ms(
            command.GPUStartTime(),
            command.GPUEndTime(),
        ))
    }

    /// One 64 Mi generated chain on resident shared buffers, compared with an
    /// equally resident device-to-device copy (10 warmups, 30 timed samples).
    pub fn benchmark_fused_chain(&mut self, plan: &Verified) -> Result<BenchResult, Error> {
        use invar_device::DagOp;
        let ir = plan.ir();
        let dag = ir.dag.as_ref().ok_or(Error::Limit)?;
        let shape = dag.nodes.iter().map(|node| node.op).collect::<Vec<_>>();
        if self.is_poisoned()
            || plan.capabilities() != self.caps
            || ir.length != 64 * 1024 * 1024
            || dag.inputs.len() != 1
            || dag.outputs.len() != 3
            || dag.stage_outputs != [1, 2, 3]
            || shape
                != [
                    DagOp::Input(0),
                    DagOp::WrappingAdd {
                        input: 0,
                        constant: 3,
                    },
                    DagOp::WrappingMul {
                        input: 1,
                        constant: 7,
                    },
                    DagOp::WrappingAdd {
                        input: 2,
                        constant: 11,
                    },
                ]
        {
            return Err(Error::Limit);
        }
        let bytes = ir.length as usize * 4;
        self.check_submission(
            bytes.max(4),
            bytes
                .checked_mul(5)
                .and_then(|n| n.checked_add(4))
                .ok_or(Error::ResourceLimit("Metal benchmark byte count overflow"))?,
        )?;
        let source =
            emit_msl(plan).map_err(|e| Error::Toolchain(format!("MSL emission: {e:?}")))?;
        let (pipeline, _) = self.compile(&source, "invar_map_v1")?;
        let input = self.buffer(bytes)?;
        let outputs = (0..3)
            .map(|_| self.buffer(bytes))
            .collect::<Result<Vec<_>, _>>()?;
        let copy = self.buffer(bytes)?;
        let n = self.buffer(4)?;
        // SAFETY: retained shared allocations cover the full typed ranges;
        // commands have not been submitted, and every value is initialized.
        unsafe {
            ptr::write(n.contents().as_ptr().cast::<u32>(), ir.length);
            let values = std::slice::from_raw_parts_mut(
                input.contents().as_ptr().cast::<u32>(),
                ir.length as usize,
            );
            values.fill(1);
        }
        let mut kernel_buffers = Vec::with_capacity(5);
        kernel_buffers.push(input.clone());
        kernel_buffers.extend(outputs.iter().cloned());
        kernel_buffers.push(n);
        let copy_buffers = [input.clone(), copy.clone()];
        self.timed_command(
            &kernel_buffers,
            Some(&pipeline),
            ir.geometry.blocks,
            ir.geometry.lanes,
            bytes,
        )?;
        // SAFETY: the preceding timed command reached Completed; retained
        // shared outputs each contain all 64 Mi initialized u32 values.
        let outputs_correct = unsafe {
            outputs
                .iter()
                .zip([4u32, 28, 39])
                .all(|(buffer, expected)| {
                    std::slice::from_raw_parts(
                        buffer.contents().as_ptr().cast::<u32>(),
                        ir.length as usize,
                    )
                    .iter()
                    .all(|value| *value == expected)
                })
        };
        if !outputs_correct {
            return Err(Error::Device(
                "resident map output differs from reference".into(),
            ));
        }
        for _ in 0..10 {
            self.timed_command(&copy_buffers, None, 0, 0, bytes)?;
            self.timed_command(
                &kernel_buffers,
                Some(&pipeline),
                ir.geometry.blocks,
                ir.geometry.lanes,
                bytes,
            )?;
        }
        let mut copy_samples = Vec::with_capacity(30);
        let mut kernel_samples = Vec::with_capacity(30);
        for index in 0..30 {
            if index % 2 == 0 {
                copy_samples.push(self.timed_command(&copy_buffers, None, 0, 0, bytes)?);
                kernel_samples.push(self.timed_command(
                    &kernel_buffers,
                    Some(&pipeline),
                    ir.geometry.blocks,
                    ir.geometry.lanes,
                    bytes,
                )?);
            } else {
                kernel_samples.push(self.timed_command(
                    &kernel_buffers,
                    Some(&pipeline),
                    ir.geometry.blocks,
                    ir.geometry.lanes,
                    bytes,
                )?);
                copy_samples.push(self.timed_command(&copy_buffers, None, 0, 0, bytes)?);
            }
        }
        let copy_median_ms = median(copy_samples)?;
        let generated_median_ms = median(kernel_samples)?;
        let copy_gbps = 2.0 * bytes as f64 / copy_median_ms / 1_000_000.0;
        // One input read and three stage outputs written; no other traffic.
        let required_bytes = 4 * bytes;
        let generated_gbps = required_bytes as f64 / generated_median_ms / 1_000_000.0;
        Ok(BenchResult {
            copy_median_ms,
            generated_median_ms,
            copy_gbps,
            generated_gbps,
            bandwidth_fraction: generated_gbps / copy_gbps,
            source_hash: source_hash(&source),
            length: ir.length as usize,
            warmups: 10,
            samples: 30,
            outputs_correct,
            required_bytes,
            logical_bytes: required_bytes,
        })
    }

    pub(super) fn timed_collective(
        &mut self,
        ir: &invar_device::Ir,
        pipelines: &[PipelineRef],
        buffers: &CollectiveBuffers,
        scan: bool,
    ) -> Result<f64, Error> {
        self.admit()?;
        let command = command_buffer(&self.queue)?;
        self.encode_collective(&command, ir, pipelines, buffers, scan)?;
        let buffers = buffers.all();
        let mut work = MetalWork::prepared(command.clone(), buffers, pipelines.to_vec());
        work.submit()?;
        let work = self.await_terminal(work, Instant::now())?;
        if work.state != WorkState::Completed {
            return Err(Error::Device(format!(
                "Metal benchmark collective failed: {:?}",
                command.error()
            )));
        }
        let ms = (command.GPUEndTime() - command.GPUStartTime()) * 1000.0;
        if !ms.is_finite() || ms <= 0.0 {
            return Err(Error::Device("Metal GPU clock returned no interval".into()));
        }
        Ok(ms)
    }

    /// Sum reduction or inclusive affine scan on 64 Mi resident elements.
    /// Every output word is checked before the timed sample loop.
    pub fn benchmark_collective(
        &mut self,
        plan: &Verified,
        seed_value: u32,
    ) -> Result<BenchResult, Error> {
        use invar_device::{Coefficient, ReduceLaw};
        let ir = plan.ir();
        let collective = ir.collective.as_ref().ok_or(Error::Limit)?;
        let (scan, input_count) = match collective.kind {
            CollectiveKind::Reduction {
                law: ReduceLaw::WrappingSum,
                ..
            } if collective.inputs.len() == 1 => (false, 1),
            CollectiveKind::AffineScan {
                a: Coefficient::Input(0),
                b: Coefficient::Input(1),
                prefix: PrefixMode::Inclusive,
                ..
            } if collective.inputs.len() == 2 => (true, 2),
            _ => return Err(Error::Limit),
        };
        if self.is_poisoned() || plan.capabilities() != self.caps || ir.length != 64 * 1024 * 1024 {
            return Err(Error::Limit);
        }
        let bytes = ir.length as usize * 4;
        let total = collective_bytes(ir, input_count, scan)
            .and_then(|n| n.checked_add(bytes))
            .ok_or(Error::Limit)?;
        self.check_submission(bytes.max(4), total)?;
        let source =
            emit_msl(plan).map_err(|e| Error::Toolchain(format!("MSL emission: {e:?}")))?;
        let compiled = ["invar_block_v1", "invar_propagate_v1", "invar_writeback_v1"]
            .map(|name| self.compile(&source, name))
            .into_iter()
            .collect::<Result<Vec<_>, _>>()?;
        let pipelines = compiled
            .into_iter()
            .map(|(pipeline, _)| pipeline)
            .collect::<Vec<_>>();
        let buffers = self.collective_buffers(ir, input_count, scan, seed_value)?;
        let copy = self.buffer(bytes)?;
        for buffer in &buffers.inputs {
            // SAFETY: each retained shared input holds `ir.length` u32 slots and
            // no command using it has been submitted.
            unsafe {
                std::slice::from_raw_parts_mut(
                    buffer.contents().as_ptr().cast::<u32>(),
                    ir.length as usize,
                )
                .fill(1);
            }
        }
        self.timed_collective(ir, &pipelines, &buffers, scan)?;
        // SAFETY: the command reached Completed, and output has exactly the
        // collective's verified element count in retained shared storage.
        let values = unsafe {
            std::slice::from_raw_parts(
                buffers.output.contents().as_ptr().cast::<u32>(),
                collective.output.elements as usize,
            )
        };
        let outputs_correct = if scan {
            values[..ir.length as usize]
                .iter()
                .enumerate()
                .all(|(i, value)| *value == seed_value.wrapping_add(i as u32).wrapping_add(1))
                && values[ir.length as usize] == seed_value.wrapping_add(ir.length)
        } else {
            values == [seed_value.wrapping_add(ir.length)]
        };
        if !outputs_correct {
            return Err(Error::Device(
                "resident collective output differs from reference".into(),
            ));
        }
        let copy_buffers = [buffers.inputs[0].clone(), copy];
        for _ in 0..10 {
            self.timed_command(&copy_buffers, None, 0, 0, bytes)?;
            self.timed_collective(ir, &pipelines, &buffers, scan)?;
        }
        let mut copy_samples = Vec::with_capacity(30);
        let mut kernel_samples = Vec::with_capacity(30);
        for index in 0..30 {
            if index % 2 == 0 {
                copy_samples.push(self.timed_command(&copy_buffers, None, 0, 0, bytes)?);
                kernel_samples.push(self.timed_collective(ir, &pipelines, &buffers, scan)?);
            } else {
                kernel_samples.push(self.timed_collective(ir, &pipelines, &buffers, scan)?);
                copy_samples.push(self.timed_command(&copy_buffers, None, 0, 0, bytes)?);
            }
        }
        let copy_median_ms = median(copy_samples)?;
        let generated_median_ms = median(kernel_samples)?;
        let copy_gbps = 2.0 * bytes as f64 / copy_median_ms / 1_000_000.0;
        // Gate on required bytes: every input read once and every output word
        // (including the terminal state) written once. Re-reads, summaries and
        // carries are implementation traffic, reported but never credited.
        let output_bytes = collective.output.elements as usize * 4;
        let required_bytes = bytes * input_count + output_bytes;
        let summary_bytes = ir.geometry.blocks as usize * if scan { 8 } else { 4 };
        let logical_bytes = if scan {
            // block reads inputs and writes summaries; propagate reads them and
            // writes prefixes; writeback rereads inputs and prefixes, writes output.
            2 * bytes * input_count + output_bytes + 4 * summary_bytes
        } else {
            bytes + 2 * summary_bytes + 8
        };
        let generated_gbps = required_bytes as f64 / generated_median_ms / 1_000_000.0;
        Ok(BenchResult {
            copy_median_ms,
            generated_median_ms,
            copy_gbps,
            generated_gbps,
            bandwidth_fraction: generated_gbps / copy_gbps,
            source_hash: source_hash(&source),
            length: ir.length as usize,
            warmups: 10,
            samples: 30,
            outputs_correct,
            required_bytes,
            logical_bytes,
        })
    }

    /// Host wall time of one minimal compute dispatch: encode, submit and
    /// observe completion of a one-thread kernel that writes one word.
    pub fn measure_dispatch(&mut self) -> Result<f64, Error> {
        self.admit()?;
        let (pipeline, _) = self.compile(invar_device::PROBE_MSL, invar_device::PROBE_FUNCTION)?;
        let flag = self.buffer(4)?;
        // SAFETY: the retained shared buffer holds one u32 and no command
        // using it has been submitted yet.
        unsafe { ptr::write(flag.contents().as_ptr().cast::<u32>(), 0) };
        let begun = Instant::now();
        let command = command_buffer(&self.queue)?;
        self.encode(&command, &pipeline, &[&flag], 1, 1)?;
        let mut work = MetalWork::prepared(command.clone(), vec![flag.clone()], vec![pipeline]);
        work.submit()?;
        let work = self.await_terminal(work, begun)?;
        let ms = begun.elapsed().as_secs_f64() * 1000.0;
        // SAFETY: the command is terminal and the retained buffer holds one u32.
        let written = unsafe { ptr::read(flag.contents().as_ptr().cast::<u32>()) };
        if work.state != WorkState::Completed || written != 1 {
            return Err(Error::Device("Metal calibration dispatch failed".into()));
        }
        Ok(ms)
    }

    /// One measured host-to-host shared-memory copy through the Metal queue.
    pub fn measure_roundtrip(&mut self, bytes: usize) -> Result<f64, Error> {
        self.admit()?;
        if bytes == 0 {
            return Err(Error::Limit);
        }
        self.check_submission(
            bytes,
            bytes
                .checked_mul(2)
                .ok_or(Error::ResourceLimit("Metal roundtrip byte count overflow"))?,
        )?;
        let input = self.buffer(bytes)?;
        let output = self.buffer(bytes)?;
        let mut host = Vec::new();
        host.try_reserve_exact(bytes)
            .map_err(|_| Error::ResourceLimit("Metal host staging allocation failed"))?;
        host.resize(bytes, 0x5a);
        let begun = Instant::now();
        // SAFETY: retained shared input covers exactly `bytes`, and the GPU
        // command has not been submitted. Hardware calibration checks all bytes.
        unsafe {
            ptr::copy_nonoverlapping(host.as_ptr(), input.contents().as_ptr().cast(), bytes);
        }
        let command = command_buffer(&self.queue)?;
        let encoder = command
            .blitCommandEncoder()
            .ok_or_else(|| Error::Unavailable("Metal blit encoder unavailable".into()))?;
        // SAFETY: both retained buffers are `bytes` long, offsets are zero,
        // and the boundary retains them until completion is confirmed.
        unsafe {
            encoder.copyFromBuffer_sourceOffset_toBuffer_destinationOffset_size(
                &input, 0, &output, 0, bytes,
            );
        }
        encoder.endEncoding();
        let mut work = MetalWork::prepared(
            command.clone(),
            vec![input.clone(), output.clone()],
            Vec::new(),
        );
        work.submit()?;
        let work = self.await_terminal(work, begun)?;
        if work.state != WorkState::Completed {
            return Err(Error::Device(format!(
                "Metal blit failed: {:?}",
                command.error()
            )));
        }
        let ms = begun.elapsed().as_secs_f64() * 1000.0;
        // SAFETY: completed shared output remains retained and covers `bytes`.
        let copied =
            unsafe { std::slice::from_raw_parts(output.contents().as_ptr().cast::<u8>(), bytes) };
        if copied != host {
            return Err(Error::Device("Metal calibration copy mismatch".into()));
        }
        Ok(ms)
    }
}

#[cfg(test)]
mod tests {
    use super::gpu_interval_ms;

    #[test]
    fn a_gpu_clock_without_an_interval_is_none_not_an_error() {
        assert_eq!(gpu_interval_ms(1.0, 1.0), None);
        assert_eq!(gpu_interval_ms(2.0, 1.0), None);
        assert_eq!(gpu_interval_ms(f64::NAN, 1.0), None);
        assert_eq!(gpu_interval_ms(0.0, f64::INFINITY), None);
        assert!((gpu_interval_ms(1.0, 1.002).unwrap() - 2.0).abs() < 1e-9);
    }
}
