//! P4a path-sum launcher: ordered stages of one source on one stream, with
//! `(args by value, part, out, stops)` parameters and 256-thread groups.
//! Compiler-generated and handwritten (P4a profile) kernels share it, so both sides
//! of a gate comparison pay the same launch, synchronization and copy.
use super::*;

pub const PATHSUM_GROUP: u32 = 256;

/// `stops` words for a launcher of `groups` groups: one per group, since
/// kernels may write `stops[blockIdx.x]`, and never fewer than one group's
/// reduce width.
fn stop_words(groups: usize) -> usize {
    groups.max(PATHSUM_GROUP as usize)
}

/// The by-value `Args` block; each kernel reads its own declared prefix.
#[derive(Clone, Copy, Debug, Default)]
#[repr(C)]
struct ArgBlock([u32; 16]);

// SAFETY: a plain `repr(C)` array of u32 with no padding or pointers.
unsafe impl cudarc::driver::DeviceRepr for ArgBlock {}

pub struct PathSum {
    stream: Arc<CudaStream>,
    functions: Vec<CudaFunction>,
    args: ArgBlock,
    part: CudaSlice<u32>,
    out: CudaSlice<u32>,
    stops: CudaSlice<u32>,
    start: CudaEvent,
    end: CudaEvent,
    groups: usize,
}

impl CudaDevice {
    /// Compile `kernels` of `source` for at most `groups` groups per stage,
    /// each publishing one `element_bytes` partial; `out` holds one element.
    pub fn pathsum(
        &mut self,
        source: &str,
        kernels: &[&str],
        groups: usize,
        element_bytes: usize,
    ) -> Result<PathSum, Error> {
        if self.is_poisoned()
            || !element_bytes.is_multiple_of(4)
            || element_bytes == 0
            || groups == 0
        {
            return Err(Error::Limit);
        }
        let bytes = groups.checked_mul(element_bytes).ok_or(Error::Limit)?;
        validate_transfer_len(bytes).map_err(|_| Error::Limit)?;
        let words = bytes / 4;
        let image = compile_gnet_cubin(source, self.caps)?.image;
        let module = self
            .context
            .load_module(Ptx::from_binary(image))
            .map_err(driver_error)?;
        let functions = kernels
            .iter()
            .map(|name| module.load_function(name).map_err(driver_error))
            .collect::<Result<Vec<_>, _>>()?;
        let event = || {
            self.context
                .new_event(Some(CUevent_flags::CU_EVENT_DEFAULT))
                .map_err(driver_error)
        };
        Ok(PathSum {
            stream: self.stream.clone(),
            functions,
            args: ArgBlock::default(),
            part: self
                .stream
                .alloc_zeros::<u32>(words)
                .map_err(driver_error)?,
            out: self
                .stream
                .alloc_zeros::<u32>(element_bytes / 4)
                .map_err(driver_error)?,
            stops: self
                .stream
                .alloc_zeros::<u32>(stop_words(groups))
                .map_err(driver_error)?,
            start: event()?,
            end: event()?,
            groups,
        })
    }
}

impl PathSum {
    /// Groups whose `stops[blockIdx.x]` word the launcher's buffer covers.
    pub fn stop_capacity(&self) -> usize {
        self.stops.len()
    }

    pub fn write_args(&mut self, args: &[u8]) -> Result<(), Error> {
        if args.len() > 64 || !args.len().is_multiple_of(4) {
            return Err(Error::Limit);
        }
        self.args = ArgBlock::default();
        for (slot, chunk) in self.args.0.iter_mut().zip(args.chunks_exact(4)) {
            *slot = u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
        }
        Ok(())
    }

    /// Launch `(kernel index, groups)` stages in order, wait, and return the
    /// event-timed device interval in milliseconds.
    pub fn dispatch(&mut self, stages: &[(usize, u32)]) -> Result<f64, Error> {
        // Refuse a bad stage before the first launch, so a refusal never leaves
        // earlier kernels running against buffers the caller may then reuse.
        if stages.iter().any(|&(kernel, groups)| {
            kernel >= self.functions.len()
                || groups == 0
                || groups as usize > self.groups
                || groups as usize > self.stops.len()
        }) {
            return Err(Error::Limit);
        }
        self.start.record(&self.stream).map_err(driver_error)?;
        for &(kernel, groups) in stages {
            let mut launch = self.stream.launch_builder(&self.functions[kernel]);
            launch
                .arg(&self.args)
                .arg(&mut self.part)
                .arg(&mut self.out)
                .arg(&mut self.stops);
            // SAFETY: every path-sum source declares `(Args, Part*, Part*,
            // uint*)` with Args a prefix of the 64-byte block; partial and stop
            // writes stay below `groups` elements, which both allocations cover, and
            // the stream is synchronized before any buffer is reused or freed,
            // including when a later launch fails.
            let launched = unsafe {
                launch.launch(LaunchConfig {
                    grid_dim: (groups, 1, 1),
                    block_dim: (PATHSUM_GROUP, 1, 1),
                    shared_mem_bytes: 0,
                })
            };
            if let Err(error) = launched {
                let _ = self.stream.synchronize();
                return Err(driver_error(error));
            }
        }
        if let Err(error) = self
            .end
            .record(&self.stream)
            .and_then(|()| self.end.synchronize())
        {
            let _ = self.stream.synchronize();
            return Err(driver_error(error));
        }
        let ms = f64::from(self.start.elapsed_ms(&self.end).map_err(driver_error)?);
        if !ms.is_finite() || ms < 0.0 {
            return Err(Error::Driver("CUDA event returned no interval".into()));
        }
        Ok(ms)
    }

    /// The first `words` u32 words of the output element.
    pub fn read_out(&mut self, words: usize) -> Result<Vec<u32>, Error> {
        if words > self.out.len() {
            return Err(Error::Limit);
        }
        let mut host = vec![0u32; self.out.len()];
        self.stream
            .memcpy_dtoh(&self.out, &mut host)
            .map_err(driver_error)?;
        self.stream.synchronize().map_err(driver_error)?;
        host.truncate(words);
        Ok(host)
    }
}
