//! P4b task-region runner (ABI `gnet-task-v1`, cascade schedule): one
//! persistent kernel over four device buffers, one launch per call. Lanes
//! claim arena nodes from a device-wide ready list and resume ready joins in
//! place; the grid is clamped to the blocks the device keeps resident, so
//! every launched lane runs at once. Source admission and the observation
//! decode live in root (`invar::general_net::task`).
use super::*;

/// Launch shape of an emitted cascade region.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CascadeShape {
    /// 32-bit words of the params, state, node and list buffers.
    pub words: [usize; 4],
    pub threads: u32,
    /// Requested blocks (clamped to the resident capacity).
    pub blocks: u32,
}

/// The state buffer after the root finished, and timing.
#[derive(Clone, Debug)]
pub struct TaskResult {
    pub state: Vec<u32>,
    pub submissions: u32,
    /// Event-timed device span of the launch, seconds.
    pub gpu_seconds: f64,
}

pub struct CascadeWork {
    kernel: CudaFunction,
    buffers: Vec<CudaSlice<u32>>,
    stream: Arc<CudaStream>,
    start: CudaEvent,
    end: CudaEvent,
    threads: u32,
    blocks: u32,
    words: [usize; 4],
    /// Sequence of the last call (tags the nodes it publishes).
    seq: u32,
}

const STATE_WORDS: usize = 32;
/// Call sequences stay below this so `seq << 3` keeps every bit.
const SEQ_LIMIT: u32 = 1 << 29;

impl CudaDevice {
    pub fn prepare_cascade(
        &mut self,
        source: &str,
        kernel: &str,
        shape: CascadeShape,
        compiler: &Path,
    ) -> Result<CascadeWork, Error> {
        self.admit()?;
        let total = shape
            .words
            .iter()
            .try_fold(0usize, |a, w| a.checked_add(w.checked_mul(4)?))
            .ok_or(Error::Limit)?;
        if shape.words.contains(&0)
            || shape.words[1] < STATE_WORDS
            || shape.threads == 0
            || shape.blocks == 0
            || total > invar_device::MAX_GNET_STORAGE_BYTES
        {
            return Err(Error::Limit);
        }
        let cubin = compile_gnet_cubin_bounded(source, compiler, self.caps)?;
        let module = self
            .context
            .load_module(Ptx::from_binary(cubin.image))
            .map_err(driver_error)?;
        let kernel = module.load_function(kernel).map_err(driver_error)?;
        if kernel.max_threads_per_block().map_err(driver_error)? < shape.threads as i32 {
            return Err(Error::Limit);
        }
        let per_sm = kernel
            .occupancy_max_active_blocks_per_multiprocessor(shape.threads, 0, None)
            .map_err(driver_error)?;
        let sms = self
            .context
            .attribute(driver_sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)
            .map_err(driver_error)?;
        let resident = per_sm.saturating_mul(u32::try_from(sms).map_err(|_| Error::Limit)?);
        if resident == 0 {
            return Err(Error::Limit);
        }
        let buffers = shape
            .words
            .iter()
            .map(|w| self.stream.alloc_zeros::<u32>(*w).map_err(driver_error))
            .collect::<Result<Vec<_>, _>>()?;
        let timed = || {
            self.context
                .new_event(Some(CUevent_flags::CU_EVENT_DEFAULT))
                .map_err(driver_error)
        };
        Ok(CascadeWork {
            kernel,
            buffers,
            stream: self.stream.clone(),
            start: timed()?,
            end: timed()?,
            threads: shape.threads,
            blocks: shape.blocks.min(resident),
            words: shape.words,
            seq: 0,
        })
    }
}

impl CascadeWork {
    /// Blocks of the launch after the residency clamp.
    pub fn blocks(&self) -> u32 {
        self.blocks
    }

    /// Run one call with entry arguments `params` to completion. A stalled
    /// cascade (state word 1 = 2, a defect) is an error, never a result.
    pub fn call(&mut self, params: &[u32]) -> Result<TaskResult, Error> {
        if params.len() > self.words[0] {
            return Err(Error::Limit);
        }
        let stream = self.stream.clone();
        self.seq = self.seq % (SEQ_LIMIT - 1) + 1;
        if self.seq == 1 {
            // A wrapped sequence could meet an old tag: clear the nodes.
            stream
                .memset_zeros(&mut self.buffers[2])
                .map_err(driver_error)?;
        }
        let mut state = vec![0u32; STATE_WORDS];
        state[25] = self.seq;
        stream
            .memcpy_htod(params, &mut self.buffers[0].slice_mut(0..params.len()))
            .map_err(driver_error)?;
        stream
            .memcpy_htod(&state[..], &mut self.buffers[1].slice_mut(0..STATE_WORDS))
            .map_err(driver_error)?;
        self.start.record(&stream).map_err(driver_error)?;
        let mut launch = stream.launch_builder(&self.kernel);
        for buffer in &mut self.buffers {
            launch.arg(buffer);
        }
        // SAFETY: the task ABI declares exactly these four u32 buffers; every
        // capacity was checked at allocation and the work retains them and
        // the module until the stream is synchronized below.
        unsafe {
            launch.launch(LaunchConfig {
                grid_dim: (self.blocks, 1, 1),
                block_dim: (self.threads, 1, 1),
                shared_mem_bytes: 0,
            })
        }
        .map_err(driver_error)?;
        self.end.record(&stream).map_err(driver_error)?;
        stream
            .memcpy_dtoh(&self.buffers[1].slice(0..STATE_WORDS), &mut state[..])
            .map_err(driver_error)?;
        stream.synchronize().map_err(driver_error)?;
        let gpu_seconds = f64::from(self.start.elapsed_ms(&self.end).map_err(driver_error)?) / 1e3;
        if state[1] != 1 {
            return Err(Error::Driver(format!(
                "task cascade stalled (state word 1 = {})",
                state[1]
            )));
        }
        Ok(TaskResult {
            state,
            submissions: 1,
            gpu_seconds,
        })
    }
}

/// Handwritten two-kernel baseline (bench only): a grid kernel then a
/// single-block top kernel over five buffers.
pub struct BaselineWork {
    grid: CudaFunction,
    top: CudaFunction,
    buffers: Vec<CudaSlice<u32>>,
    stream: Arc<CudaStream>,
    start: CudaEvent,
    end: CudaEvent,
}

impl CudaDevice {
    pub fn prepare_baseline(
        &mut self,
        source: &str,
        grid: &str,
        top: &str,
        words: [usize; 5],
        compiler: &Path,
    ) -> Result<BaselineWork, Error> {
        self.admit()?;
        if words.contains(&0) || words[0] < 4 || words[4] < 2 {
            return Err(Error::Limit);
        }
        let cubin = compile_gnet_cubin_bounded(source, compiler, self.caps)?;
        let module = self
            .context
            .load_module(Ptx::from_binary(cubin.image))
            .map_err(driver_error)?;
        let grid = module.load_function(grid).map_err(driver_error)?;
        let top = module.load_function(top).map_err(driver_error)?;
        let buffers = words
            .iter()
            .map(|w| self.stream.alloc_zeros::<u32>(*w).map_err(driver_error))
            .collect::<Result<Vec<_>, _>>()?;
        let timed = || {
            self.context
                .new_event(Some(CUevent_flags::CU_EVENT_DEFAULT))
                .map_err(driver_error)
        };
        Ok(BaselineWork {
            grid,
            top,
            buffers,
            stream: self.stream.clone(),
            start: timed()?,
            end: timed()?,
        })
    }
}

impl BaselineWork {
    /// Run with `params = [a, b, depth, groups]`; return the two output words
    /// (value, trap) and the event-timed device span in seconds.
    pub fn call(&mut self, params: [i32; 4]) -> Result<([i32; 2], f64), Error> {
        let groups = u32::try_from(params[3]).map_err(|_| Error::Limit)?;
        if groups == 0 || groups > 256 {
            return Err(Error::Limit);
        }
        let stream = self.stream.clone();
        let words = params.map(|p| p as u32);
        stream
            .memcpy_htod(&words[..], &mut self.buffers[0].slice_mut(0..4))
            .map_err(driver_error)?;
        self.start.record(&stream).map_err(driver_error)?;
        for (function, blocks) in [(&self.grid, groups), (&self.top, 1)] {
            let mut launch = stream.launch_builder(function);
            for buffer in &mut self.buffers {
                launch.arg(buffer);
            }
            // SAFETY: the baseline source declares exactly these five u32
            // buffers; capacities were checked at allocation and the work
            // retains them until the stream is synchronized below.
            unsafe {
                launch.launch(LaunchConfig {
                    grid_dim: (blocks, 1, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                })
            }
            .map_err(driver_error)?;
        }
        self.end.record(&stream).map_err(driver_error)?;
        let mut out = [0u32; 2];
        stream
            .memcpy_dtoh(&self.buffers[4].slice(0..2), &mut out[..])
            .map_err(driver_error)?;
        stream.synchronize().map_err(driver_error)?;
        let seconds = f64::from(self.start.elapsed_ms(&self.end).map_err(driver_error)?) / 1e3;
        Ok((out.map(|w| w as i32), seconds))
    }
}
