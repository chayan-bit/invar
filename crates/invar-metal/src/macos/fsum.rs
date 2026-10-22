//! v0.2 deterministic float reductions on Metal. The kernels return per-group
//! partials (tree nodes or exact digit rows); the host finishes them with the
//! same shared code as the CPU engines, so only the partials cross the bus.

use super::*;
use invar_device::fsum::{self as fs, Launch, Op};

/// Partials of one completed submission.
pub struct FsumResult {
    pub launch: Launch,
    /// Tree: one node word per group, at level `launch.node_level()`.
    pub nodes: Vec<u32>,
    /// Exact: `groups * ROW` words (digit sums, then the flag word).
    pub rows: Vec<i64>,
    pub source_hash: u64,
    pub compile_ms: f64,
    pub kernel_ms: f64,
    pub transfer_bytes: usize,
}

/// Resident timings of one benchmark size, in milliseconds of GPU time.
#[derive(Debug)]
pub struct FsumBench {
    pub elements: usize,
    pub median_ms: f64,
    pub baseline_median_ms: f64,
}

/// Handwritten unordered Metal sum: grid-stride float4 loads, `simd_sum`, one
/// partial per group. It is the speed reference, not a deterministic result.
const BASELINE: &str = r#"#include <metal_stdlib>
using namespace metal;
kernel void baseline_sum(device const float4* a [[buffer(0)]],
                         device const float4* unused [[buffer(1)]],
                         device float* out [[buffer(2)]],
                         constant uint* p [[buffer(3)]],
                         device uint* guard [[buffer(4)]],
                         uint tid [[thread_position_in_threadgroup]],
                         uint group [[threadgroup_position_in_grid]],
                         uint gid [[thread_position_in_grid]],
                         uint total [[threads_per_grid]],
                         uint lane [[thread_index_in_simdgroup]],
                         uint simd [[simdgroup_index_in_threadgroup]]) {
    float acc = 0.0f;
    for (uint i = gid; i < p[0] / 4u; i += total) {
        float4 v = a[i];
        acc += (v.x + v.y) + (v.z + v.w);
    }
    acc = simd_sum(acc);
    threadgroup float partial[8];
    if (lane == 0u) { partial[simd] = acc; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0u) {
        float s = 0.0f;
        for (uint k = 0u; k < 8u; ++k) { s += partial[k]; }
        out[group] = s;
    }
}
"#;

fn check_operands(op: Op, a: &[u32], b: Option<&[u32]>) -> Result<(), Error> {
    if op.is_dot() != b.is_some() || b.is_some_and(|b| b.len() != a.len()) {
        return Err(Error::Limit);
    }
    Ok(())
}

impl MetalDevice {
    /// The pipeline for `op` and the milliseconds spent creating it (0 on a
    /// cache hit). The library is compiled ahead unless the device is a
    /// developer runtime-source device.
    fn fsum_pipeline(&mut self, op: Op) -> Result<(PipelineRef, f64), Error> {
        let started = Instant::now();
        let (pipeline, hit) = self.compile(&fs::msl(), op.kernel())?;
        if pipeline.maxTotalThreadsPerThreadgroup() < fs::LANES as usize {
            return Err(Error::ResourceLimit(
                "fsum geometry exceeds Metal pipeline threadgroup limit",
            ));
        }
        let ms = if hit {
            0.0
        } else {
            started.elapsed().as_secs_f64() * 1000.0
        };
        Ok((pipeline, ms))
    }

    fn upload_words<T: Copy>(&self, values: &[T]) -> Result<BufferRef, Error> {
        let buffer = self.buffer(std::mem::size_of_val(values).max(4))?;
        // SAFETY: the new shared buffer holds at least `size_of_val(values)`
        // bytes, is disjoint from the source slice, and no command uses it yet.
        unsafe {
            ptr::copy_nonoverlapping(
                values.as_ptr(),
                buffer.contents().as_ptr().cast(),
                values.len(),
            );
        }
        Ok(buffer)
    }

    /// Upload, run one launch, and return its partials. `b` is the dot operand.
    pub fn run_fsum(
        &mut self,
        op: Op,
        a: &[u32],
        b: Option<&[u32]>,
        leaf_shift: u32,
    ) -> Result<FsumResult, Error> {
        self.admit()?;
        check_operands(op, a, b)?;
        let launch = Launch::new(op, a.len(), leaf_shift).map_err(|_| Error::Limit)?;
        let output_bytes = launch.output_words() * if op.is_exact() { 8 } else { 4 };
        let input_bytes = a.len() * 4;
        let transfer_bytes = input_bytes * (1 + usize::from(b.is_some())) + output_bytes + 16;
        crate::check_aggregate_limit(transfer_bytes)?;
        self.check_submission(input_bytes.max(output_bytes), transfer_bytes)?;
        let (pipeline, compile_ms) = self.fsum_pipeline(op)?;
        let first = self.upload_words(a)?;
        let second = match b {
            Some(b) => self.upload_words(b)?,
            None => first.clone(),
        };
        let output = self.buffer(output_bytes)?;
        let params = self.upload_words(&launch.params)?;
        // Metal zero-fills new buffers, so the guard starts clear.
        let guard = self.buffer(4)?;
        let buffers = vec![first, second, output, params, guard];
        let ms = self.timed_command(&buffers, Some(&pipeline), launch.groups, fs::LANES, 0)?;
        // SAFETY: Completed is terminal; slot 4 is a retained one-word buffer.
        let fired = unsafe { ptr::read(buffers[4].contents().as_ptr().cast::<u32>()) };
        if fired != 0 {
            return Err(Error::SubnormalFallback);
        }
        let mut result = FsumResult {
            launch,
            nodes: Vec::new(),
            rows: Vec::new(),
            source_hash: source_hash(&fs::msl()),
            compile_ms,
            kernel_ms: ms,
            transfer_bytes,
        };
        // SAFETY: the command reached Completed (terminal), and the retained
        // output buffer holds exactly `output_words` words of the read type.
        unsafe {
            let words = buffers[2].contents().as_ptr();
            if op.is_exact() {
                result.rows = read_words::<i64>(words, launch.output_words());
            } else {
                result.nodes = read_words::<u32>(words, launch.output_words());
            }
        }
        Ok(result)
    }

    /// Resident medians of the fsum kernel and the handwritten baseline sum on
    /// the same `n` words (10 warmups, 30 timed samples each).
    pub fn benchmark_fsum(
        &mut self,
        op: Op,
        a: &[u32],
        leaf_shift: u32,
    ) -> Result<FsumBench, Error> {
        let launch = Launch::new(op, a.len(), leaf_shift).map_err(|_| Error::Limit)?;
        let (pipeline, _) = self.fsum_pipeline(op)?;
        let (baseline, _) = self.compile(BASELINE, "baseline_sum")?;
        let input = self.upload_words(a)?;
        let output = self.buffer(launch.output_words().max(4096) * 8)?;
        let params = self.upload_words(&launch.params)?;
        let guard = self.buffer(4)?;
        let buffers = vec![input.clone(), input, output, params, guard];
        let baseline_groups = (a.len() as u32 / 4).div_ceil(fs::LANES * 16).clamp(1, 4096);
        let runs = [(&pipeline, launch.groups), (&baseline, baseline_groups)];
        // Interleave the two kernels so clock and thermal drift hit both.
        let mut times = [Vec::new(), Vec::new()];
        for sample in 0..40 {
            for (index, (pipeline, groups)) in runs.iter().enumerate() {
                let ms = self.timed_command(&buffers, Some(pipeline), *groups, fs::LANES, 0)?;
                if sample >= 10 {
                    times[index].push(ms);
                }
            }
        }
        let [fsum_times, baseline_times] = times;
        let median_ms = bench::median(fsum_times)?;
        let baseline_median_ms = bench::median(baseline_times)?;
        Ok(FsumBench {
            elements: a.len(),
            median_ms,
            baseline_median_ms,
        })
    }
}

/// Copy `count` words of `T` out of a completed shared buffer.
///
/// # Safety
/// `source` must point at `count` initialized `T` words that no command writes.
unsafe fn read_words<T: Copy + Default>(source: *mut std::ffi::c_void, count: usize) -> Vec<T> {
    let mut values = vec![T::default(); count];
    // SAFETY: the caller guarantees `count` readable words; `values` is a
    // fresh disjoint allocation of the same length.
    unsafe { ptr::copy_nonoverlapping(source.cast::<T>(), values.as_mut_ptr(), count) };
    values
}
