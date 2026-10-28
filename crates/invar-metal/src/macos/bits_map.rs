//! Elementwise maps over raw 64-bit encodings, used by the invar-math-cr-v1
//! Metal differential and benchmarks. The caller supplies MSL whose kernel has
//! the fixed ABI `(device const ulong *x [[buffer(0)]], device const ulong *y
//! [[buffer(1)]], device ulong *out [[buffer(2)]], constant uint &n
//! [[buffer(3)]], uint i [[thread_position_in_grid]])`; this module gets its
//! pipeline through the device's library cache (a compiled-ahead library, or
//! source compiled with the pinned options on a device opened with
//! `open_runtime_source`), runs it on fresh shared buffers and returns the
//! output only after a Completed command (unknown completion quarantines).

use super::*;

const LANES: u32 = 32;

/// One compiled map kernel.
pub struct BitsMap {
    pipeline: PipelineRef,
    pub compile_ms: f64,
}

/// Output encodings and GPU time of one map dispatch.
pub struct BitsMapRun {
    pub output: Vec<u64>,
    pub kernel_ms: f64,
}

impl MetalDevice {
    /// The pipeline for `kernel` of `source`.
    pub fn load_bits_map(&mut self, source: &str, kernel: &str) -> Result<BitsMap, Error> {
        let started = Instant::now();
        let (pipeline, _) = self.compile(source, kernel)?;
        Ok(BitsMap {
            pipeline,
            compile_ms: started.elapsed().as_secs_f64() * 1000.0,
        })
    }

    /// Run `map` over `x` and `y` (equal, nonzero lengths below 2^32) and wait
    /// for completion. Each dispatch must finish within the device deadline,
    /// so callers chunk large inputs.
    pub fn run_bits_map(
        &mut self,
        map: &BitsMap,
        x: &[u64],
        y: &[u64],
    ) -> Result<BitsMapRun, Error> {
        self.reap();
        if self.is_poisoned() {
            return Err(Error::Device("Metal device has unconfirmed work".into()));
        }
        let n = u32::try_from(x.len()).map_err(|_| Error::Limit)?;
        if x.len() != y.len() || n == 0 {
            return Err(Error::Limit);
        }
        let bytes = x.len().checked_mul(8).ok_or(Error::Limit)?;
        self.check_submission(bytes, bytes.checked_mul(3).ok_or(Error::Limit)?)?;
        let buffers = [
            self.buffer(bytes)?,
            self.buffer(bytes)?,
            self.buffer(bytes)?,
            self.buffer(4)?,
        ];
        // SAFETY: each fresh shared buffer holds at least the copied element
        // count, is disjoint from the host slices, and no command uses it yet.
        unsafe {
            ptr::copy_nonoverlapping(x.as_ptr(), buffers[0].contents().as_ptr().cast(), x.len());
            ptr::copy_nonoverlapping(y.as_ptr(), buffers[1].contents().as_ptr().cast(), y.len());
            ptr::write_bytes(buffers[2].contents().as_ptr().cast::<u64>(), 0, x.len());
            ptr::write(buffers[3].contents().as_ptr().cast::<u32>(), n);
        }
        let command = command_buffer(&self.queue)?;
        let refs = buffers.each_ref().map(|b| &**b);
        self.encode(&command, &map.pipeline, &refs, n.div_ceil(LANES), LANES)?;
        let mut work = MetalWork::prepared(
            command.clone(),
            buffers.to_vec(),
            vec![map.pipeline.clone()],
        );
        work.submit()?;
        let work = self.await_terminal(work, Instant::now())?;
        if work.state != WorkState::Completed {
            return Err(Error::Device(format!(
                "Metal command failed: {:?}",
                command.error()
            )));
        }
        let kernel_ms = (command.GPUEndTime() - command.GPUStartTime()) * 1000.0;
        let mut output = vec![0u64; x.len()];
        // SAFETY: Completed is terminal; the retained output buffer holds
        // `x.len()` u64 slots and the destination is disjoint.
        unsafe {
            ptr::copy_nonoverlapping(
                buffers[2].contents().as_ptr().cast(),
                output.as_mut_ptr(),
                x.len(),
            )
        };
        Ok(BitsMapRun { output, kernel_ms })
    }
}
