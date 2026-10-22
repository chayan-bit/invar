//! M4-07 bounded summary scans: one single-pass look-back launch, one owned
//! completion fence. Inputs are staged privately and validated before
//! compilation; no output is read before the fence confirms the device finished.

use super::*;
use invar_device::summary as ir;

#[derive(Debug)]
pub struct SummaryResult {
    /// Inclusive prefixes (`steps * width` words) or the final state (`width`).
    pub output: Vec<u32>,
    pub source_hash: u64,
    pub cubin_bytes: usize,
    pub transfer_bytes: u64,
    pub blocks: u32,
    pub compile_ms: f64,
    /// Workspace zeroing and the scan kernel, between stream events after H2D
    /// and before D2H.
    pub kernel_ms: f32,
    pub host_to_host_ms: f64,
    pub nvrtc_log: String,
}

/// Device-resident comparison of one summary scan with plain copies of its
/// coefficient buffer on the same GPU (10 warmups, 30 interleaved samples).
#[derive(Debug)]
pub struct SummaryBench {
    /// Output of the first launch on the resident buffers, read after a fence.
    pub output: Vec<u32>,
    pub summary: BenchMetric,
    /// `invar_copy_v1`: a grid-stride `uint4` device-to-device copy kernel.
    pub copy_kernel: BenchMetric,
    /// `cuMemcpyDtoD` of the same bytes, the M3 copy reference.
    pub copy_driver: BenchMetric,
    pub copy_bytes: u64,
    pub required_bytes: u64,
    /// Read plus write bytes per second of the faster copy.
    pub copy_gbps: f64,
    /// Required bytes (coefficients and seed read once, output written once).
    pub summary_gbps: f64,
    pub bandwidth_fraction: f64,
    pub warmups: usize,
    pub samples: usize,
}

const WARMUPS: usize = 10;
const SAMPLES: usize = 30;
const COPY: &str = r#"// invar-copy-v1
extern "C" __global__ void invar_copy_v1(const uint4* in, uint4* out, unsigned long long n) {
  for (unsigned long long i = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x; i < n;
       i += (unsigned long long)gridDim.x * blockDim.x) out[i] = in[i];
}
"#;

/// The benchmark copy kernel and its destination, owned like the scan.
struct CopyKernel {
    _module: Arc<cudarc::driver::CudaModule>,
    function: CudaFunction,
    destination: CudaSlice<u32>,
}

/// Every allocation, module and event the submitted stream may still touch.
struct SummaryWork {
    _module: Arc<cudarc::driver::CudaModule>,
    scan: CudaFunction,
    coefficients: CudaSlice<u32>,
    seed: CudaSlice<u32>,
    output: CudaSlice<u32>,
    status: CudaSlice<u32>,
    ticket: CudaSlice<u32>,
    output_host: PinnedHostSlice<u32>,
    copy: Option<CopyKernel>,
    kernels_start: CudaEvent,
    kernels_end: CudaEvent,
    completed: CudaEvent,
    steps: u32,
    blocks: u32,
}

impl SummaryWork {
    /// Enqueue the verified schedule on `stream`; arguments follow the emitter.
    fn launch(&mut self, stream: &Arc<CudaStream>) -> Result<(), Error> {
        stream
            .memset_zeros(&mut self.status)
            .map_err(driver_error)?;
        stream
            .memset_zeros(&mut self.ticket)
            .map_err(driver_error)?;
        let mut scan = stream.launch_builder(&self.scan);
        scan.arg(&self.coefficients)
            .arg(&self.seed)
            .arg(&mut self.output)
            .arg(&mut self.status)
            .arg(&mut self.ticket)
            .arg(&self.steps);
        // SAFETY: the verified emitter fixes (coefficients, seed, output,
        // status, ticket, n). Tickets 0..blocks index `blocks` zeroed 24-word
        // status slots; every output write is below n * width (inclusive) or
        // width (final) words; coefficients hold n steps.
        unsafe {
            scan.launch(LaunchConfig {
                grid_dim: (self.blocks, 1, 1),
                block_dim: (ir::LANES, 1, 1),
                shared_mem_bytes: 0,
            })
        }
        .map(|_| ())
        .map_err(driver_error)
    }

    fn launch_copy(&mut self, stream: &Arc<CudaStream>) -> Result<(), Error> {
        let kernel = self.copy.as_mut().ok_or(Error::Limit)?;
        let quads = (self.coefficients.len() / 4) as u64;
        let mut copy = stream.launch_builder(&kernel.function);
        copy.arg(&self.coefficients)
            .arg(&mut kernel.destination)
            .arg(&quads);
        // SAFETY: both allocations hold `4 * quads` words (checked by the
        // caller), CUDA allocations are 256-byte aligned, and the grid-stride
        // loop bounds every index below `quads`.
        unsafe {
            copy.launch(LaunchConfig {
                grid_dim: (quads.div_ceil(256).min(1 << 20) as u32, 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            })
        }
        .map(|_| ())
        .map_err(driver_error)
    }

    fn timed(&mut self, stream: &Arc<CudaStream>, kind: usize) -> Result<f32, Error> {
        self.kernels_start.record(stream).map_err(driver_error)?;
        match kind {
            0 => self.launch(stream)?,
            1 => self.launch_copy(stream)?,
            _ => {
                let kernel = self.copy.as_mut().ok_or(Error::Limit)?;
                stream
                    .memcpy_dtod(&self.coefficients, &mut kernel.destination)
                    .map_err(driver_error)?
            }
        }
        self.kernels_end.record(stream).map_err(driver_error)?;
        self.kernels_end.synchronize().map_err(driver_error)?;
        self.kernels_start
            .elapsed_ms(&self.kernels_end)
            .map_err(driver_error)
    }
}

struct Prepared {
    work: Box<SummaryWork>,
    hash: u64,
    cubin_bytes: usize,
    compile_ms: f64,
    log: String,
}

impl CudaDevice {
    /// Compile the verified schedule and stage validated inputs on the stream.
    fn prepare_summary(
        &mut self,
        plan: &ir::Verified,
        coefficients: &[u32],
        seed: &[u32],
        compiler: &Path,
    ) -> Result<Prepared, Error> {
        let source = ir::emit_cuda(plan).map_err(|e| Error::Compile(format!("{e:?}")))?;
        let hash = source_hash(&source);
        let started = Instant::now();
        let compiled = compile_cubin_bounded(&source, compiler, self.caps)?;
        let compile_ms = started.elapsed().as_secs_f64() * 1000.0;
        let cubin_bytes = compiled.image.len();
        let module = self
            .context
            .load_module(Ptx::from_binary(compiled.image))
            .map_err(driver_error)?;
        let scan = module
            .load_function("invar_summary_scan_v1")
            .map_err(driver_error)?;
        let blocks = plan.blocks();
        let status_words = blocks as usize * ir::STATUS_WORDS as usize;
        // SAFETY: each allocation is written before any read on this stream:
        // H2D fills coefficients and seed, every launch zeroes status and
        // ticket first, and the scan writes every output word.
        let (coefficients_dev, seed_dev, output, status, ticket) = unsafe {
            (
                self.stream.alloc::<u32>(coefficients.len()),
                self.stream.alloc::<u32>(seed.len()),
                self.stream.alloc::<u32>(plan.output_words()),
                self.stream.alloc::<u32>(status_words),
                self.stream.alloc::<u32>(1),
            )
        };
        // SAFETY: zero-filled immediately below before any slice API or D2H.
        let mut output_host = unsafe { self.context.alloc_pinned::<u32>(plan.output_words()) }
            .map_err(driver_error)?;
        let ptr = output_host.as_mut_ptr().map_err(driver_error)?;
        // SAFETY: ptr owns exactly output_words writable u32 slots.
        unsafe { std::ptr::write_bytes(ptr, 0, plan.output_words()) };
        let event = || {
            self.context
                .new_event(Some(CUevent_flags::CU_EVENT_DEFAULT))
                .map_err(driver_error)
        };
        let mut work = Box::new(SummaryWork {
            _module: module,
            scan,
            coefficients: coefficients_dev.map_err(driver_error)?,
            seed: seed_dev.map_err(driver_error)?,
            output: output.map_err(driver_error)?,
            status: status.map_err(driver_error)?,
            ticket: ticket.map_err(driver_error)?,
            output_host,
            copy: None,
            kernels_start: event()?,
            kernels_end: event()?,
            completed: event()?,
            steps: plan.plan().steps,
            blocks,
        });
        let staged = (|| -> Result<(), Error> {
            self.stream
                .memcpy_htod(coefficients, &mut work.coefficients)
                .map_err(driver_error)?;
            self.stream
                .memcpy_htod(seed, &mut work.seed)
                .map_err(driver_error)
        })();
        if let Err(error) = staged {
            self.hold(work);
            return Err(error);
        }
        Ok(Prepared {
            work,
            hash,
            cubin_bytes,
            compile_ms,
            log: compiled.log,
        })
    }

    /// Wait for `work.completed` within the kernel deadline; on timeout the
    /// work is held, never freed.
    fn finish_summary(&mut self, work: Box<SummaryWork>) -> Result<Box<SummaryWork>, Error> {
        let waited = Instant::now();
        while !work.completed.is_complete() {
            if waited.elapsed() >= KERNEL_DEADLINE {
                self.hold(work);
                return Err(Error::Timeout);
            }
            std::thread::sleep(Duration::from_millis(1));
        }
        self.context.check_err().map_err(driver_error)?;
        Ok(work)
    }

    /// Run a verified summary scan. `coefficients` and `seed` use the source
    /// layout: step-major table entries or row-major d x d matrices.
    pub fn run_summary(
        &mut self,
        plan: &ir::Verified,
        coefficients: &[u32],
        seed: &[u32],
        compiler: &Path,
    ) -> Result<SummaryResult, Error> {
        self.admit()?;
        if plan.capabilities() != self.caps {
            return Err(Error::Limit);
        }
        ir::check_inputs(plan, coefficients, seed).map_err(|e| Error::Driver(e.to_string()))?;
        if plan.plan().steps == 0 {
            let source = ir::emit_cuda(plan).map_err(|e| Error::Compile(format!("{e:?}")))?;
            let output = match plan.plan().observation {
                ir::Observation::FinalState => seed.to_vec(),
                ir::Observation::InclusivePrefixes => Vec::new(),
            };
            return Ok(SummaryResult {
                output,
                source_hash: source_hash(&source),
                cubin_bytes: 0,
                transfer_bytes: 0,
                blocks: plan.blocks(),
                compile_ms: 0.0,
                kernel_ms: 0.0,
                host_to_host_ms: 0.0,
                nvrtc_log: String::new(),
            });
        }
        let host_started = Instant::now();
        let Prepared {
            mut work,
            hash,
            cubin_bytes,
            compile_ms,
            log,
        } = self.prepare_summary(plan, coefficients, seed, compiler)?;
        let submitted = (|| -> Result<(), Error> {
            let w = &mut *work;
            w.kernels_start.record(&self.stream).map_err(driver_error)?;
            w.launch(&self.stream)?;
            w.kernels_end.record(&self.stream).map_err(driver_error)?;
            self.stream
                .memcpy_dtoh(&w.output, &mut w.output_host)
                .map_err(driver_error)?;
            w.completed.record(&self.stream).map_err(driver_error)
        })();
        if let Err(error) = submitted {
            self.hold(work);
            return Err(error);
        }
        let work = self.finish_summary(work)?;
        let kernel_ms = work
            .kernels_start
            .elapsed_ms(&work.kernels_end)
            .map_err(driver_error)?;
        let output = work.output_host.as_slice().map_err(driver_error)?.to_vec();
        Ok(SummaryResult {
            output,
            source_hash: hash,
            cubin_bytes,
            transfer_bytes: plan.transfer_bytes(),
            blocks: plan.blocks(),
            compile_ms,
            kernel_ms,
            host_to_host_ms: host_started.elapsed().as_secs_f64() * 1000.0 - compile_ms,
            nvrtc_log: log,
        })
    }

    /// Time the resident scan against copies of its coefficient buffer. The
    /// caller compares `output` with its oracle; timing is never a gate here.
    pub fn benchmark_summary(
        &mut self,
        plan: &ir::Verified,
        coefficients: &[u32],
        seed: &[u32],
        compiler: &Path,
    ) -> Result<SummaryBench, Error> {
        self.admit()?;
        if plan.capabilities() != self.caps
            || plan.plan().steps == 0
            || !coefficients.len().is_multiple_of(4)
        {
            return Err(Error::Limit);
        }
        ir::check_inputs(plan, coefficients, seed).map_err(|e| Error::Driver(e.to_string()))?;
        let copy_image = compile_cubin_bounded(COPY, compiler, self.caps)?.image;
        let copy_module = self
            .context
            .load_module(Ptx::from_binary(copy_image))
            .map_err(driver_error)?;
        let copy_function = copy_module
            .load_function("invar_copy_v1")
            .map_err(driver_error)?;
        // SAFETY: the copy destination is only ever written by the timed
        // copies and never read.
        let destination =
            unsafe { self.stream.alloc::<u32>(coefficients.len()) }.map_err(driver_error)?;
        let Prepared { mut work, .. } = self.prepare_summary(plan, coefficients, seed, compiler)?;
        work.copy = Some(CopyKernel {
            _module: copy_module,
            function: copy_function,
            destination,
        });
        let stream = self.stream.clone();
        let measured = (|| -> Result<[Vec<f32>; 3], Error> {
            work.launch(&stream)?;
            stream
                .memcpy_dtoh(&work.output, &mut work.output_host)
                .map_err(driver_error)?;
            stream.synchronize().map_err(driver_error)?;
            for _ in 0..WARMUPS {
                for kind in 0..3 {
                    work.timed(&stream, kind)?;
                }
            }
            let mut samples = [Vec::new(), Vec::new(), Vec::new()];
            for index in 0..SAMPLES {
                let order = if index % 2 == 0 { [0, 1, 2] } else { [2, 1, 0] };
                for kind in order {
                    samples[kind].push(work.timed(&stream, kind)?);
                }
            }
            Ok(samples)
        })();
        let [summary, copy_kernel, copy_driver] = match measured {
            Ok(samples) => samples,
            Err(error) => {
                self.hold(work);
                return Err(error);
            }
        };
        let output = work.output_host.as_slice().map_err(driver_error)?.to_vec();
        let (summary, copy_kernel, copy_driver) = (
            bench::metric(summary)?,
            bench::metric(copy_kernel)?,
            bench::metric(copy_driver)?,
        );
        let copy_bytes = coefficients.len() as u64 * 4;
        let fastest = copy_kernel.median_ms.min(copy_driver.median_ms);
        let copy_gbps = 2.0 * copy_bytes as f64 / f64::from(fastest) / 1e6;
        let summary_gbps = plan.transfer_bytes() as f64 / f64::from(summary.median_ms) / 1e6;
        Ok(SummaryBench {
            output,
            summary,
            copy_kernel,
            copy_driver,
            copy_bytes,
            required_bytes: plan.transfer_bytes(),
            copy_gbps,
            summary_gbps,
            bandwidth_fraction: summary_gbps / copy_gbps,
            warmups: WARMUPS,
            samples: SAMPLES,
        })
    }
}
