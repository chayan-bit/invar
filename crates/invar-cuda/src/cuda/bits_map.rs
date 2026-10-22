//! Elementwise maps over raw 64-bit encodings, used by the invar-math-cr-v1
//! CUDA differential and benchmarks. The caller supplies a translation unit
//! whose kernel has the fixed ABI `(const u64 *x, const u64 *y, u64 *out,
//! u64 n)` and the NVRTC options that make it exact; this module only
//! compiles, loads, launches and times it. Every call synchronizes before it
//! returns, so no device work outlives the borrowed buffers.

use super::*;

/// Compiled programs here hold up to forty CORE-MATH functions.
const MAX_MAP_CUBIN_BYTES: usize = 64 * 1024 * 1024;
const MAX_MAP_LOG_BYTES: usize = 64 * 1024;
const BLOCK: u32 = 256;

/// One loaded map kernel.
pub struct BitsMap {
    _module: Arc<cudarc::driver::CudaModule>,
    function: CudaFunction,
    pub cubin_bytes: usize,
    pub compile_ms: f64,
    pub log: String,
}

/// Output encodings and the device time of the kernel alone.
pub struct BitsMapRun {
    pub output: Vec<u64>,
    pub kernel_ms: f32,
}

fn full_log(program: &NvrtcProgram) -> String {
    let mut bytes = 0usize;
    // SAFETY: `program` owns a live NVRTC handle; `bytes` is a writable local.
    if unsafe { nvrtc_sys::nvrtcGetProgramLogSize(program.0, &mut bytes) }
        .result()
        .is_err()
        || bytes == 0
    {
        return String::new();
    }
    let mut log = vec![0u8; bytes.min(MAX_MAP_LOG_BYTES)];
    if bytes > log.len() {
        return format!("NVRTC log of {bytes} bytes exceeds the cap");
    }
    // SAFETY: `log` owns exactly the reported size while the program is live.
    if unsafe { nvrtc_sys::nvrtcGetProgramLog(program.0, log.as_mut_ptr().cast()) }
        .result()
        .is_err()
    {
        return String::new();
    }
    log.truncate(log.iter().position(|&c| c == 0).unwrap_or(log.len()));
    String::from_utf8_lossy(&log).into_owned()
}

/// Compile `source` for the device's own architecture under `options`.
pub fn compile_with_options(
    source: &str,
    options: &[&str],
    caps: Capabilities,
) -> Result<CompiledCubin, Error> {
    let Capabilities::Cuda { sm, .. } = caps else {
        return Err(Error::Unavailable);
    };
    // SAFETY: optional-library presence probe precedes every NVRTC symbol access.
    if !unsafe { nvrtc_sys::is_culib_present() } {
        return Err(Error::Unavailable);
    }
    let text = CString::new(source).map_err(|_| Error::Limit)?;
    let program = NvrtcProgram(
        nvrtc_result::create_program(&text, Some(c"invar_bits_map.cu"))
            .map_err(|e| Error::Compile(e.to_string()))?,
    );
    let architecture = format!("--gpu-architecture=sm_{sm}");
    let mut all: Vec<&str> = options.to_vec();
    all.push(&architecture);
    // SAFETY: program owns the live handle and source; options contain no NULs.
    let compiled = unsafe { nvrtc_result::compile_program(program.0, &all) };
    let log = full_log(&program);
    if let Err(error) = compiled {
        return Err(Error::Compile(format!("{error}: {log}")));
    }
    let mut bytes = 0usize;
    // SAFETY: live successfully compiled program; valid writable local size.
    unsafe { nvrtc_sys::nvrtcGetCUBINSize(program.0, &mut bytes) }
        .result()
        .map_err(|e| Error::Compile(e.to_string()))?;
    if bytes == 0 || bytes > MAX_MAP_CUBIN_BYTES {
        return Err(Error::Limit);
    }
    let mut image = vec![0u8; bytes];
    // SAFETY: image owns exactly the size returned by the live NVRTC program.
    unsafe { nvrtc_sys::nvrtcGetCUBIN(program.0, image.as_mut_ptr().cast()) }
        .result()
        .map_err(|e| Error::Compile(e.to_string()))?;
    Ok(CompiledCubin { image, log })
}

impl CudaDevice {
    /// Compile and load `kernel` from `source`.
    pub fn load_bits_map(
        &mut self,
        source: &str,
        options: &[&str],
        kernel: &str,
    ) -> Result<BitsMap, Error> {
        self.admit()?;
        let started = Instant::now();
        let compiled = compile_with_options(source, options, self.caps)?;
        let compile_ms = started.elapsed().as_secs_f64() * 1000.0;
        let cubin_bytes = compiled.image.len();
        let module = self
            .context
            .load_module(Ptx::from_binary(compiled.image))
            .map_err(driver_error)?;
        let function = module.load_function(kernel).map_err(driver_error)?;
        Ok(BitsMap {
            _module: module,
            function,
            cubin_bytes,
            compile_ms,
            log: compiled.log,
        })
    }

    /// Run `map` over `x` and `y` (equal lengths) and wait for completion.
    pub fn run_bits_map(
        &mut self,
        map: &BitsMap,
        x: &[u64],
        y: &[u64],
    ) -> Result<BitsMapRun, Error> {
        self.admit()?;
        if x.len() != y.len() || x.is_empty() {
            return Err(Error::Limit);
        }
        let n = x.len() as u64;
        let bytes = x.len().checked_mul(24).ok_or(Error::Limit)?;
        if bytes > self.context.mem_get_info().map_err(driver_error)?.0 {
            return Err(Error::ResourceLimit(
                "bits map buffers exceed free CUDA memory",
            ));
        }
        let x_dev = self.stream.clone_htod(x).map_err(allocation_error)?;
        let y_dev = self.stream.clone_htod(y).map_err(allocation_error)?;
        let mut out = self
            .stream
            .alloc_zeros::<u64>(x.len())
            .map_err(allocation_error)?;
        let flags = Some(CUevent_flags::CU_EVENT_DEFAULT);
        let start = self.context.new_event(flags).map_err(driver_error)?;
        let end = self.context.new_event(flags).map_err(driver_error)?;
        let blocks = u32::try_from(n.div_ceil(u64::from(BLOCK))).map_err(|_| Error::Limit)?;
        start.record(&self.stream).map_err(driver_error)?;
        let mut launch = self.stream.launch_builder(&map.function);
        launch.arg(&x_dev).arg(&y_dev).arg(&mut out).arg(&n);
        // SAFETY: the kernel ABI is (const u64*, const u64*, u64*, u64) and
        // reads or writes index i only when i < n; all three allocations hold
        // n initialized elements and stay alive until the synchronize below.
        unsafe {
            launch.launch(LaunchConfig {
                grid_dim: (blocks, 1, 1),
                block_dim: (BLOCK, 1, 1),
                shared_mem_bytes: 0,
            })
        }
        .map_err(driver_error)?;
        end.record(&self.stream).map_err(driver_error)?;
        let output = self.stream.clone_dtoh(&out).map_err(driver_error)?;
        self.stream.synchronize().map_err(driver_error)?;
        let kernel_ms = start.elapsed_ms(&end).map_err(driver_error)?;
        Ok(BitsMapRun { output, kernel_ms })
    }
}
