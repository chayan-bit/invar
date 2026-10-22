//! v0.2 deterministic float reductions on the owned stream. The kernels
//! return per-block partials (tree nodes or exact digit rows); the host
//! finishes them with the shared code. NVRTC keeps IEEE subnormals (no
//! `--ftz`) and the kernels use `__fadd_rn`/`__fmul_rn`, which never contract.
//! Work that misses its deadline or fails to enqueue is held, never freed.

use super::*;
use invar_device::fsum::{self as fs, Launch, Op};

#[derive(Debug)]
pub struct FsumResult {
    pub launch: Launch,
    /// Tree: one node word per block, at level `launch.node_level()`.
    pub nodes: Vec<u32>,
    /// Exact: `blocks * ROW` words (digit sums, then the flag word).
    pub rows: Vec<i64>,
    pub source_hash: u64,
    pub compile_ms: f64,
    pub kernel_ms: f32,
    pub transfer_bytes: usize,
}

/// Resident GPU medians of one size (10 warmups, 30 timed samples).
#[derive(Debug)]
pub struct FsumBench {
    pub elements: usize,
    pub median_ms: f32,
}

/// Everything a launch touches; dropped only after its event completes.
struct FsumWork {
    a: CudaSlice<u32>,
    b: Option<CudaSlice<u32>>,
    nodes: CudaSlice<u32>,
    rows: CudaSlice<i64>,
    _functions: Vec<CudaFunction>,
    start: CudaEvent,
    end: CudaEvent,
}

fn check_operands(op: Op, a: &[u32], b: Option<&[u32]>) -> Result<(), Error> {
    if op.is_dot() != b.is_some() || b.is_some_and(|b| b.len() != a.len()) {
        return Err(Error::Limit);
    }
    Ok(())
}

impl CudaDevice {
    /// Compile both kernels once per opened device.
    fn fsum_functions(&mut self, compiler: &Path) -> Result<(Vec<CudaFunction>, f64), Error> {
        if let Some(functions) = &self.fsum {
            return Ok((functions.clone(), 0.0));
        }
        let started = Instant::now();
        let compiled = compile_cubin_bounded(&fs::cuda(), compiler, self.caps)?;
        let module = self
            .context
            .load_module(Ptx::from_binary(compiled.image))
            .map_err(driver_error)?;
        let functions = [Op::SumTree, Op::SumExact]
            .iter()
            .map(|op| module.load_function(op.kernel()).map_err(driver_error))
            .collect::<Result<Vec<_>, _>>()?;
        self.fsum = Some(functions.clone());
        Ok((functions, started.elapsed().as_secs_f64() * 1000.0))
    }

    fn launch_fsum(
        &self,
        work: &mut FsumWork,
        function: &CudaFunction,
        launch: &Launch,
    ) -> Result<(), Error> {
        let [n, dot, shift, digits] = launch.params;
        let mut builder = self.stream.launch_builder(function);
        builder.arg(&work.a).arg(work.b.as_ref().unwrap_or(&work.a));
        if launch.op.is_exact() {
            builder.arg(&mut work.rows).arg(&n).arg(&dot).arg(&digits);
        } else {
            builder.arg(&mut work.nodes).arg(&n).arg(&dot).arg(&shift);
        }
        // SAFETY: the fixed kernels take (a, b, out, n, dot, shift-or-digits)
        // in this order; `Launch` sized `out` for every block and bounds every
        // index by `n`, which equals both operand lengths. FsumWork retains all
        // arguments and the module until its completion event.
        unsafe {
            builder.launch(LaunchConfig {
                grid_dim: (launch.groups, 1, 1),
                block_dim: (fs::LANES, 1, 1),
                shared_mem_bytes: 0,
            })
        }
        .map_err(driver_error)?;
        Ok(())
    }

    fn fsum_work(&self, launch: &Launch, a: &[u32], b: Option<&[u32]>) -> Result<FsumWork, Error> {
        let event = || {
            self.context
                .new_event(Some(CUevent_flags::CU_EVENT_DEFAULT))
                .map_err(driver_error)
        };
        let words = launch.output_words();
        Ok(FsumWork {
            a: self.stream.clone_htod(a).map_err(allocation_error)?,
            b: b.map(|b| self.stream.clone_htod(b))
                .transpose()
                .map_err(allocation_error)?,
            nodes: self
                .stream
                .alloc_zeros::<u32>(words.max(1))
                .map_err(allocation_error)?,
            rows: self
                .stream
                .alloc_zeros::<i64>(words.max(1))
                .map_err(allocation_error)?,
            _functions: Vec::new(),
            start: event()?,
            end: event()?,
        })
    }

    /// Upload, run one launch, and return its partials. `b` is the dot operand.
    pub fn run_fsum(
        &mut self,
        op: Op,
        a: &[u32],
        b: Option<&[u32]>,
        leaf_shift: u32,
        compiler: &Path,
    ) -> Result<FsumResult, Error> {
        self.admit()?;
        check_operands(op, a, b)?;
        let launch = Launch::new(op, a.len(), leaf_shift).map_err(|_| Error::Limit)?;
        let transfer_bytes =
            a.len() * 4 * (1 + usize::from(b.is_some())) + launch.output_words() * 12;
        crate::validate_transfer_budget(transfer_bytes, transfer_bytes)?;
        self.check_submission(transfer_bytes)?;
        let (functions, compile_ms) = self.fsum_functions(compiler)?;
        let function = functions[usize::from(op.is_exact())].clone();
        let mut work = Box::new(self.fsum_work(&launch, a, b)?);
        work._functions = functions;
        let submitted = (|| -> Result<(), Error> {
            work.start.record(&self.stream).map_err(driver_error)?;
            self.launch_fsum(&mut work, &function, &launch)?;
            work.end.record(&self.stream).map_err(driver_error)
        })();
        if let Err(error) = submitted {
            self.hold(work);
            return Err(error);
        }
        let waited = Instant::now();
        while !work.end.is_complete() {
            if waited.elapsed() >= KERNEL_DEADLINE {
                self.hold(work);
                return Err(Error::Timeout);
            }
            std::thread::sleep(Duration::from_micros(200));
        }
        let kernel_ms = work.start.elapsed_ms(&work.end).map_err(driver_error)?;
        let words = launch.output_words();
        let (mut nodes, mut rows) = (Vec::new(), Vec::new());
        if op.is_exact() {
            rows = self.stream.clone_dtoh(&work.rows).map_err(driver_error)?;
            rows.truncate(words);
        } else {
            nodes = self.stream.clone_dtoh(&work.nodes).map_err(driver_error)?;
            nodes.truncate(words);
        }
        Ok(FsumResult {
            launch,
            nodes,
            rows,
            source_hash: source_hash(&fs::cuda()),
            compile_ms,
            kernel_ms,
            transfer_bytes,
        })
    }

    /// Resident kernel median on `a` (sum ops), timed with stream events.
    pub fn benchmark_fsum(
        &mut self,
        op: Op,
        a: &[u32],
        leaf_shift: u32,
        compiler: &Path,
    ) -> Result<FsumBench, Error> {
        self.admit()?;
        let launch = Launch::new(op, a.len(), leaf_shift).map_err(|_| Error::Limit)?;
        let (functions, _) = self.fsum_functions(compiler)?;
        let function = functions[usize::from(op.is_exact())].clone();
        let mut work = self.fsum_work(&launch, a, None)?;
        let mut times = Vec::with_capacity(30);
        for sample in 0..40 {
            work.start.record(&self.stream).map_err(driver_error)?;
            self.launch_fsum(&mut work, &function, &launch)?;
            work.end.record(&self.stream).map_err(driver_error)?;
            work.end.synchronize().map_err(driver_error)?;
            if sample >= 10 {
                times.push(work.start.elapsed_ms(&work.end).map_err(driver_error)?);
            }
        }
        times.sort_by(f32::total_cmp);
        Ok(FsumBench {
            elements: a.len(),
            median_ms: (times[14] + times[15]) / 2.0,
        })
    }
}
