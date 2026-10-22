//! M4-07 bounded summary scans in one command buffer: a status clear and one
//! single-pass look-back scan for prefixes, or reduce and final composition
//! for the final state. Inputs are validated before compilation; output is
//! read only after the command buffer reaches `Completed`.

use super::*;
use invar_device::summary as ir;

#[derive(Debug)]
pub struct SummaryResult {
    /// Inclusive prefixes (`steps * width` words) or the final state (`width`).
    pub output: Vec<u32>,
    pub source_hash: u64,
    pub transfer_bytes: u64,
    pub blocks: u32,
    /// Predecessor tiles this run reduced itself instead of reading their
    /// published summary (look-back fallbacks).
    pub fallbacks: u32,
    pub compile_ms: f64,
    pub kernel_ms: f64,
    pub host_to_host_ms: f64,
}

/// Resident comparison of one summary scan with plain copies of its
/// coefficient buffer on the same GPU (10 warmups, 30 interleaved samples).
#[derive(Debug)]
pub struct SummaryBench {
    /// Output of the first command on the resident buffers.
    pub output: Vec<u32>,
    pub summary_median_ms: f64,
    /// `invar_copy_v1`: a `uint4` device-to-device copy kernel.
    pub copy_kernel_median_ms: f64,
    /// Blit-encoder copy of the same bytes, the M3 copy reference.
    pub copy_blit_median_ms: f64,
    pub copy_bytes: u64,
    pub required_bytes: u64,
    /// Read plus write bytes per second of the faster copy.
    pub copy_gbps: f64,
    pub summary_gbps: f64,
    pub bandwidth_fraction: f64,
    pub warmups: usize,
    pub samples: usize,
    /// Look-back fallbacks of the first command.
    pub fallbacks: u32,
}

const WARMUPS: usize = 10;
const SAMPLES: usize = 30;
const COPY: &str = r#"// invar-copy-v1
#include <metal_stdlib>
using namespace metal;
kernel void invar_copy_v1(device const uint4* in [[buffer(0)]], device uint4* out [[buffer(1)]],
    device const uint* n [[buffer(2)]], uint i [[thread_position_in_grid]]) {
  if (i < n[0]) out[i] = in[i];
}
"#;

/// Retained buffers and pipelines of one staged scan.
struct Staged {
    pipelines: [PipelineRef; 2],
    coefficients: BufferRef,
    seed: BufferRef,
    /// Look-back slots (prefixes) or tile aggregates (final state).
    status: BufferRef,
    output: BufferRef,
    counts: BufferRef,
    blocks: u32,
    output_words: usize,
    prefix: bool,
}

impl Staged {
    fn buffers(&self) -> Vec<BufferRef> {
        [
            &self.coefficients,
            &self.seed,
            &self.status,
            &self.output,
            &self.counts,
        ]
        .map(Clone::clone)
        .to_vec()
    }

    fn status_words(blocks: u32, prefix: bool) -> usize {
        if prefix {
            blocks as usize * ir::METAL_SLOT_WORDS as usize + 2
        } else {
            blocks as usize * 16
        }
    }

    fn fallbacks(&self) -> u32 {
        if !self.prefix {
            return 0;
        }
        // SAFETY: callers read only after a Completed command; the status
        // buffer holds `status_words` u32s and the counter is the last one.
        unsafe {
            ptr::read(
                self.status
                    .contents()
                    .as_ptr()
                    .cast::<u32>()
                    .add(Self::status_words(self.blocks, true) - 1),
            )
        }
    }

    fn read_output(&self) -> Vec<u32> {
        let mut values = vec![0u32; self.output_words];
        // SAFETY: callers read only after a Completed command; the retained
        // output buffer covers every copied u32.
        unsafe {
            ptr::copy_nonoverlapping(
                self.output.contents().as_ptr().cast(),
                values.as_mut_ptr(),
                self.output_words,
            );
        }
        values
    }
}

fn median(mut values: Vec<f64>) -> Result<f64, Error> {
    if values.len() != SAMPLES || values.iter().any(|v| !v.is_finite() || *v <= 0.0) {
        return Err(Error::Device("invalid Metal GPU timing".into()));
    }
    values.sort_by(f64::total_cmp);
    Ok((values[SAMPLES / 2 - 1] + values[SAMPLES / 2]) / 2.0)
}

impl MetalDevice {
    fn stage_summary(
        &mut self,
        plan: &ir::Verified,
        coefficients: &[u32],
        seed: &[u32],
        look_back: ir::LookBack,
    ) -> Result<(Staged, u64, f64), Error> {
        let source = ir::emit_msl_with(plan, look_back)
            .map_err(|e| Error::Toolchain(format!("MSL emission: {e:?}")))?;
        let hash = source_hash(&source);
        let started = Instant::now();
        let prefix = plan.plan().observation == ir::Observation::InclusivePrefixes;
        let names = if prefix {
            ["invar_summary_clear_v3", "invar_summary_scan_v3"]
        } else {
            ["invar_summary_reduce_v1", "invar_summary_final_v3"]
        };
        let pipelines = [
            self.compile(&source, names[0])?.0,
            self.compile(&source, names[1])?.0,
        ];
        // The block scan forms SIMD groups of 32 from consecutive threads.
        if pipelines.iter().any(|p| p.threadExecutionWidth() != 32) {
            return Err(Error::Toolchain(
                "Metal summary scan requires 32-wide SIMD groups".into(),
            ));
        }
        let compile_ms = started.elapsed().as_secs_f64() * 1000.0;
        let blocks = plan.blocks();
        let staged = Staged {
            pipelines,
            coefficients: self.buffer(coefficients.len() * 4)?,
            seed: self.buffer(seed.len() * 4)?,
            status: self.buffer(Staged::status_words(blocks, prefix) * 4)?,
            output: self.buffer(plan.output_words() * 4)?,
            counts: self.buffer(8)?,
            blocks,
            output_words: plan.output_words(),
            prefix,
        };
        // SAFETY: each retained shared buffer was just allocated with room for
        // exactly these words, and no command references it yet.
        unsafe {
            ptr::copy_nonoverlapping(
                coefficients.as_ptr(),
                staged.coefficients.contents().as_ptr().cast(),
                coefficients.len(),
            );
            ptr::copy_nonoverlapping(
                seed.as_ptr(),
                staged.seed.contents().as_ptr().cast(),
                seed.len(),
            );
            ptr::copy_nonoverlapping(
                [plan.plan().steps, blocks].as_ptr(),
                staged.counts.contents().as_ptr().cast(),
                2,
            );
        }
        Ok((staged, hash, compile_ms))
    }

    /// Encode the verified schedule; buffer slots follow the emitter.
    fn encode_summary(
        &self,
        command: &ProtocolObject<dyn MTLCommandBuffer>,
        s: &Staged,
    ) -> Result<(), Error> {
        let [first, second] = &s.pipelines;
        if !s.prefix {
            self.encode(
                command,
                first,
                &[&s.coefficients, &s.status, &s.counts],
                s.blocks,
                ir::LANES,
            )?;
            return self.encode(
                command,
                second,
                &[&s.status, &s.seed, &s.output, &s.counts],
                1,
                ir::LANES,
            );
        }
        let words =
            u32::try_from(Staged::status_words(s.blocks, true)).map_err(|_| Error::Limit)?;
        self.encode(
            command,
            first,
            &[&s.status, &s.counts],
            words.div_ceil(ir::LANES),
            ir::LANES,
        )?;
        self.encode(
            command,
            second,
            &[&s.coefficients, &s.seed, &s.output, &s.status, &s.counts],
            s.blocks,
            ir::LANES,
        )
    }

    /// Submit one command, wait for `Completed`, and return its GPU interval.
    fn run_command(
        &mut self,
        buffers: Vec<BufferRef>,
        pipelines: Vec<PipelineRef>,
        encode: impl FnOnce(&Self, &ProtocolObject<dyn MTLCommandBuffer>) -> Result<(), Error>,
    ) -> Result<f64, Error> {
        let command = self
            .queue
            .commandBuffer()
            .ok_or_else(|| Error::Unavailable("Metal command buffer unavailable".into()))?;
        encode(self, &command)?;
        let mut work = MetalWork::prepared(command.clone(), buffers, pipelines);
        work.submit()?;
        let work = self.await_terminal(work, Instant::now())?;
        if work.state != WorkState::Completed {
            return Err(Error::Device(format!(
                "Metal summary command failed: {:?}",
                command.error()
            )));
        }
        Ok((command.GPUEndTime() - command.GPUStartTime()) * 1000.0)
    }

    /// Run a verified summary scan. `coefficients` and `seed` use the source
    /// layout: step-major table entries or row-major d x d matrices.
    pub fn run_summary(
        &mut self,
        plan: &ir::Verified,
        coefficients: &[u32],
        seed: &[u32],
    ) -> Result<SummaryResult, Error> {
        self.run_summary_with(plan, coefficients, seed, ir::LookBack::DEFAULT)
    }

    /// `run_summary` with an explicit look-back schedule; results must not
    /// depend on it, which the adversarial tests check.
    pub fn run_summary_with(
        &mut self,
        plan: &ir::Verified,
        coefficients: &[u32],
        seed: &[u32],
        look_back: ir::LookBack,
    ) -> Result<SummaryResult, Error> {
        self.admit()?;
        if plan.capabilities() != self.caps {
            return Err(Error::Limit);
        }
        ir::check_inputs(plan, coefficients, seed).map_err(|e| Error::Device(e.to_string()))?;
        if plan.plan().steps == 0 {
            let source =
                ir::emit_msl(plan).map_err(|e| Error::Toolchain(format!("MSL emission: {e:?}")))?;
            let output = match plan.plan().observation {
                ir::Observation::FinalState => seed.to_vec(),
                ir::Observation::InclusivePrefixes => Vec::new(),
            };
            return Ok(SummaryResult {
                output,
                source_hash: source_hash(&source),
                transfer_bytes: 0,
                blocks: plan.blocks(),
                fallbacks: 0,
                compile_ms: 0.0,
                kernel_ms: 0.0,
                host_to_host_ms: 0.0,
            });
        }
        let (staged, hash, compile_ms) = self.stage_summary(plan, coefficients, seed, look_back)?;
        let host_started = Instant::now();
        let kernel_ms = self.run_command(
            staged.buffers(),
            staged.pipelines.to_vec(),
            |device, command| device.encode_summary(command, &staged),
        )?;
        Ok(SummaryResult {
            output: staged.read_output(),
            source_hash: hash,
            transfer_bytes: plan.transfer_bytes(),
            blocks: staged.blocks,
            fallbacks: staged.fallbacks(),
            compile_ms,
            kernel_ms,
            host_to_host_ms: host_started.elapsed().as_secs_f64() * 1000.0,
        })
    }

    /// Time the resident scan against copies of its coefficient buffer. The
    /// caller compares `output` with its oracle; timing is never a gate here.
    pub fn benchmark_summary(
        &mut self,
        plan: &ir::Verified,
        coefficients: &[u32],
        seed: &[u32],
    ) -> Result<SummaryBench, Error> {
        self.admit()?;
        if plan.capabilities() != self.caps
            || plan.plan().steps == 0
            || !coefficients.len().is_multiple_of(4)
        {
            return Err(Error::Limit);
        }
        ir::check_inputs(plan, coefficients, seed).map_err(|e| Error::Device(e.to_string()))?;
        let copy = self.compile(COPY, "invar_copy_v1")?.0;
        let (staged, _, _) = self.stage_summary(plan, coefficients, seed, ir::LookBack::DEFAULT)?;
        let bytes = coefficients.len() * 4;
        let quads = u32::try_from(coefficients.len() / 4).map_err(|_| Error::Limit)?;
        let destination = self.buffer(bytes)?;
        let quad_count = self.buffer(4)?;
        // SAFETY: freshly allocated 4-byte shared buffer, not yet referenced.
        unsafe { ptr::write(quad_count.contents().as_ptr().cast::<u32>(), quads) };
        let copy_buffers = vec![
            staged.coefficients.clone(),
            destination.clone(),
            quad_count.clone(),
        ];
        let timed = |device: &mut Self, kind: usize| -> Result<f64, Error> {
            match kind {
                0 => device.run_command(
                    staged.buffers(),
                    staged.pipelines.to_vec(),
                    |device, command| device.encode_summary(command, &staged),
                ),
                1 => device.run_command(
                    copy_buffers.clone(),
                    vec![copy.clone()],
                    |device, command| {
                        device.encode(
                            command,
                            &copy,
                            &[&staged.coefficients, &destination, &quad_count],
                            quads.div_ceil(256),
                            256,
                        )
                    },
                ),
                _ => device.run_command(copy_buffers.clone(), Vec::new(), |_, command| {
                    let encoder = command.blitCommandEncoder().ok_or_else(|| {
                        Error::Unavailable("Metal blit encoder unavailable".into())
                    })?;
                    // SAFETY: both retained buffers cover `bytes` at offset
                    // zero and the command is awaited to a terminal state.
                    unsafe {
                        encoder.copyFromBuffer_sourceOffset_toBuffer_destinationOffset_size(
                            &staged.coefficients,
                            0,
                            &destination,
                            0,
                            bytes,
                        );
                    }
                    encoder.endEncoding();
                    Ok(())
                }),
            }
        };
        timed(self, 0)?;
        let output = staged.read_output();
        let fallbacks = staged.fallbacks();
        for _ in 0..WARMUPS {
            for kind in 0..3 {
                timed(self, kind)?;
            }
        }
        let mut samples = [Vec::new(), Vec::new(), Vec::new()];
        for index in 0..SAMPLES {
            let order = if index % 2 == 0 { [0, 1, 2] } else { [2, 1, 0] };
            for kind in order {
                samples[kind].push(timed(self, kind)?);
            }
        }
        let [summary, kernel, blit] = samples;
        let (summary, kernel, blit) = (median(summary)?, median(kernel)?, median(blit)?);
        let copy_gbps = 2.0 * bytes as f64 / kernel.min(blit) / 1e6;
        let summary_gbps = plan.transfer_bytes() as f64 / summary / 1e6;
        Ok(SummaryBench {
            output,
            summary_median_ms: summary,
            copy_kernel_median_ms: kernel,
            copy_blit_median_ms: blit,
            copy_bytes: bytes as u64,
            required_bytes: plan.transfer_bytes(),
            copy_gbps,
            summary_gbps,
            bandwidth_fraction: summary_gbps / copy_gbps,
            warmups: WARMUPS,
            samples: SAMPLES,
            fallbacks,
        })
    }
}
