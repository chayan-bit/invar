//! Generated-track samples for the M5-01 harness: verified map and
//! collective plans timed on resident buffers (10 warmups, 30 samples) plus
//! independent host round trips. Every output word is checked against the
//! reference before any sample is taken.
//!
//! Ported from the reviewed benchmark branch (`c19c5264`, "sample verified
//! u32 generated collectives"), which `src/bin/invar-bench.rs` already calls;
//! the only change is that a chunked scan reports its real dispatch count.

use super::*;
use run::collective_bytes;

/// Raw generated-kernel samples from a verified ordinary-source plan.
#[derive(Debug)]
pub struct GeneratedBenchResult {
    pub source_hash: u64,
    pub compile_ms: f64,
    pub device_resident_ms: Vec<f64>,
    pub host_to_host_ms: Vec<f64>,
    pub transfer_bytes: usize,
    pub workspace_bytes: usize,
    pub launches: usize,
}

impl MetalDevice {
    /// Time one checked map plan with retained resident buffers. The ordinary
    /// runner supplies independent host round trips with the same inputs.
    pub fn benchmark_generated_map(
        &mut self,
        plan: &Verified,
        inputs: &[&[u32]],
        expected: &[Vec<u32>],
    ) -> Result<GeneratedBenchResult, Error> {
        let ir = plan.ir();
        let input_count = ir.dag.as_ref().map_or(1, |dag| dag.inputs.len());
        let output_count = ir.dag.as_ref().map_or(1, |dag| dag.outputs.len());
        let bytes = (ir.length as usize).checked_mul(4).ok_or(Error::Limit)?;
        let transfer_bytes = bytes
            .checked_mul(input_count + output_count)
            .ok_or(Error::Limit)?;
        if ir.length == 0
            || ir.collective.is_some()
            || plan.capabilities() != self.caps
            || inputs.len() != input_count
            || expected.len() != output_count
            || inputs
                .iter()
                .any(|values| values.len() != ir.length as usize)
            || expected
                .iter()
                .any(|values| values.len() != ir.length as usize)
        {
            return Err(Error::Limit);
        }
        self.admit()?;
        if transfer_bytes as u64 > invar_device::MAX_DAG_TRANSFER_BYTES {
            return Err(Error::ResourceLimit("map exceeds 8 GiB transfer budget"));
        }
        self.check_submission(
            bytes,
            transfer_bytes
                .checked_add(4)
                .ok_or(Error::ResourceLimit("map byte count overflow"))?,
        )?;
        let source =
            emit_msl(plan).map_err(|e| Error::Toolchain(format!("MSL emission: {e:?}")))?;
        let started = Instant::now();
        let (pipeline, _) = self.compile(&source, "invar_map_v1")?;
        let compile_ms = started.elapsed().as_secs_f64() * 1000.0;
        let input_buffers = (0..input_count)
            .map(|_| self.buffer(bytes))
            .collect::<Result<Vec<_>, _>>()?;
        let output_buffers = (0..output_count)
            .map(|_| self.buffer(bytes))
            .collect::<Result<Vec<_>, _>>()?;
        let n = self.buffer(4)?;
        // SAFETY: all shared allocations cover the exact initialized input
        // ranges and no command has been submitted yet.
        unsafe {
            ptr::write(n.contents().as_ptr().cast::<u32>(), ir.length);
            for (values, buffer) in inputs.iter().zip(&input_buffers) {
                ptr::copy_nonoverlapping(
                    values.as_ptr(),
                    buffer.contents().as_ptr().cast::<u32>(),
                    values.len(),
                );
            }
        }
        let buffers = input_buffers
            .iter()
            .chain(&output_buffers)
            .cloned()
            .chain(std::iter::once(n))
            .collect::<Vec<_>>();
        self.timed_command(
            &buffers,
            Some(&pipeline),
            ir.geometry.blocks,
            ir.geometry.lanes,
            bytes,
        )?;
        // SAFETY: timed_command observed Completed; each output allocation
        // holds the verified number of u32 elements.
        let correct = unsafe {
            output_buffers
                .iter()
                .zip(expected)
                .all(|(buffer, reference)| {
                    std::slice::from_raw_parts(
                        buffer.contents().as_ptr().cast::<u32>(),
                        ir.length as usize,
                    ) == reference
                })
        };
        if !correct {
            return Err(Error::Device(
                "generated map output differs from reference".into(),
            ));
        }
        for _ in 0..10 {
            self.timed_command(
                &buffers,
                Some(&pipeline),
                ir.geometry.blocks,
                ir.geometry.lanes,
                bytes,
            )?;
        }
        let mut device_resident_ms = Vec::with_capacity(30);
        for _ in 0..30 {
            device_resident_ms.push(self.timed_command(
                &buffers,
                Some(&pipeline),
                ir.geometry.blocks,
                ir.geometry.lanes,
                bytes,
            )?);
        }
        for _ in 0..10 {
            let run = self.run_fused_map(plan, inputs)?;
            if run.outputs != expected {
                return Err(Error::Device("host round trip map mismatch".into()));
            }
        }
        let mut host_to_host_ms = Vec::with_capacity(30);
        for _ in 0..30 {
            let run = self.run_fused_map(plan, inputs)?;
            if run.outputs != expected {
                return Err(Error::Device("host round trip map mismatch".into()));
            }
            host_to_host_ms.push(run.host_to_host_ms);
        }
        Ok(GeneratedBenchResult {
            source_hash: source_hash(&source),
            compile_ms,
            device_resident_ms,
            host_to_host_ms,
            transfer_bytes,
            workspace_bytes: 0,
            launches: 1,
        })
    }

    /// Time the three-pass reduction or inclusive affine scan on resident data.
    pub fn benchmark_generated_collective(
        &mut self,
        plan: &Verified,
        inputs: &[&[u32]],
        seed: u32,
        expected: &[u32],
    ) -> Result<GeneratedBenchResult, Error> {
        use invar_device::{Coefficient, ReduceLaw};
        let ir = plan.ir();
        let collective = ir.collective.as_ref().ok_or(Error::Limit)?;
        let scan = match collective.kind {
            CollectiveKind::Reduction {
                law: ReduceLaw::WrappingSum,
                ..
            } if inputs.len() == 1 => false,
            CollectiveKind::AffineScan {
                a: Coefficient::Input(0),
                b: Coefficient::Input(1),
                prefix: PrefixMode::Inclusive,
                ..
            } if inputs.len() == 2 => true,
            _ => return Err(Error::Limit),
        };
        let bytes = (ir.length as usize).checked_mul(4).ok_or(Error::Limit)?;
        let transfer_bytes = bytes
            .checked_mul(inputs.len())
            .and_then(|n| n.checked_add(expected.len().checked_mul(4)?))
            .ok_or(Error::Limit)?;
        let workspace_bytes = collective_bytes(ir, inputs.len(), scan)
            .and_then(|n| n.checked_sub(transfer_bytes))
            .ok_or(Error::Limit)?;
        if ir.length == 0
            || self.is_poisoned()
            || plan.capabilities() != self.caps
            || inputs.len() != collective.inputs.len()
            || inputs
                .iter()
                .any(|values| values.len() != ir.length as usize)
            || expected.len() != collective.output.elements as usize
        {
            return Err(Error::Limit);
        }
        self.admit()?;
        let source =
            emit_msl(plan).map_err(|e| Error::Toolchain(format!("MSL emission: {e:?}")))?;
        let started = Instant::now();
        let pipelines = ["invar_block_v1", "invar_propagate_v1", "invar_writeback_v1"]
            .map(|name| self.compile(&source, name))
            .into_iter()
            .map(|result| result.map(|(pipeline, _)| pipeline))
            .collect::<Result<Vec<_>, _>>()?;
        let compile_ms = started.elapsed().as_secs_f64() * 1000.0;
        let buffers = self.collective_buffers(ir, inputs.len(), scan, seed)?;
        // SAFETY: retained shared inputs cover these exact ranges and no
        // command has been submitted with them yet.
        unsafe {
            for (values, buffer) in inputs.iter().zip(&buffers.inputs) {
                ptr::copy_nonoverlapping(
                    values.as_ptr(),
                    buffer.contents().as_ptr().cast::<u32>(),
                    values.len(),
                );
            }
        }
        self.timed_collective(ir, &pipelines, &buffers, scan)?;
        // SAFETY: timed_collective observed Completed and the output buffer
        // covers the verified collective output element count.
        let actual = unsafe {
            std::slice::from_raw_parts(
                buffers.output.contents().as_ptr().cast::<u32>(),
                expected.len(),
            )
        };
        if actual != expected {
            return Err(Error::Device(
                "generated collective output differs from reference".into(),
            ));
        }
        for _ in 0..10 {
            self.timed_collective(ir, &pipelines, &buffers, scan)?;
        }
        let mut device_resident_ms = Vec::with_capacity(30);
        for _ in 0..30 {
            device_resident_ms.push(self.timed_collective(ir, &pipelines, &buffers, scan)?);
        }
        for _ in 0..10 {
            let run = self.run_collective(plan, inputs, seed)?;
            if run.prefixes != expected[..expected.len() - 1]
                || run.final_state != *expected.last().ok_or(Error::Limit)?
            {
                return Err(Error::Device("host round trip collective mismatch".into()));
            }
        }
        let mut host_to_host_ms = Vec::with_capacity(30);
        for _ in 0..30 {
            let run = self.run_collective(plan, inputs, seed)?;
            if run.prefixes != expected[..expected.len() - 1]
                || run.final_state != *expected.last().ok_or(Error::Limit)?
            {
                return Err(Error::Device("host round trip collective mismatch".into()));
            }
            host_to_host_ms.push(run.host_to_host_ms);
        }
        Ok(GeneratedBenchResult {
            source_hash: source_hash(&source),
            compile_ms,
            device_resident_ms,
            host_to_host_ms,
            transfer_bytes,
            workspace_bytes,
            // A scan encodes block, propagate and writeback once per chunk.
            launches: if scan {
                3 * run::scan_chunks(ir.geometry.blocks)
            } else {
                3
            },
        })
    }
}
