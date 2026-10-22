//! Bit transpose with fresh private storage and an owning completion fence.
//! Arbitrary u32/binary32 bits are preserved. Root owns source admission and
//! logical publication; no output is exposed before actual device completion.

use super::*;
use cudarc::driver::{CudaModule, DevicePtr};
use invar_device::transpose as ir;

#[derive(Debug)]
pub struct TransposeResult {
    /// Fresh physical output, including zeroed padding; logical NaN/inf bits are legal.
    pub output: Vec<u32>,
    pub source_hash: u64,
    pub cubin_bytes: usize,
    pub transfer_bytes: usize,
    pub compile_ms: f64,
    pub nvrtc_log: String,
}

struct Shape {
    lengths: [usize; 2],
    logical: usize,
    offset: usize,
    transfer_bytes: usize,
}

fn plan_options(source: &str, caps: Capabilities) -> Result<Vec<String>, Error> {
    if source.len() > invar_device::MAX_SOURCE_BYTES || source.contains('\0') {
        return Err(Error::Limit);
    }
    let options = ir::target_options(caps).map_err(|_| Error::Unavailable)?;
    if !matches!(caps, Capabilities::Cuda { .. }) {
        return Err(Error::Unavailable);
    }
    if !matches!(
        source.lines().next(),
        Some(
            "// invar-transpose-bits-v1 version=1 layout-version=1 element-bytes=4"
                | "// invar-stencil-u32-v1 version=1 numeric=wrapping-u32-v1"
        )
    ) || !source
        .lines()
        .nth(1)
        .is_some_and(|line| line.ends_with(&format!("caps={caps:?}")))
        || !source
            .lines()
            .any(|line| line == format!("// required-options: {}", options.join(" ")))
    {
        return Err(Error::Compile(
            "transpose source/capability/options identity mismatch".into(),
        ));
    }
    Ok(options)
}

fn shape(plan: &ir::Verified, caps: Capabilities, input_len: usize) -> Result<Shape, Error> {
    if plan.capabilities() != caps {
        return Err(Error::Limit);
    }
    let input = &plan.ir().input;
    let output = &plan.ir().output;
    let lengths = [
        usize::try_from(input.elements).map_err(|_| Error::Limit)?,
        usize::try_from(output.elements).map_err(|_| Error::Limit)?,
    ];
    if input_len != lengths[0] {
        return Err(Error::Limit);
    }
    let [rows, cols] = plan.ir().dimensions;
    let logical = usize::try_from(u64::from(rows) * u64::from(cols)).map_err(|_| Error::Limit)?;
    let offset = usize::try_from(output.offset).map_err(|_| Error::Limit)?;
    if offset
        .checked_add(logical)
        .is_none_or(|end| end > lengths[1])
    {
        return Err(Error::Limit);
    }
    let transfer_bytes = lengths[1]
        .checked_mul(2)
        .and_then(|n| n.checked_add(lengths[0]))
        .and_then(|n| n.checked_mul(4))
        .ok_or(Error::Limit)?;
    // Two private host/device copies plus the returned output allocation.
    let owned_bytes = lengths[0]
        .checked_add(lengths[1])
        .and_then(|n| n.checked_mul(2))
        .and_then(|n| n.checked_add(lengths[1]))
        .and_then(|n| n.checked_mul(4))
        .ok_or(Error::Limit)?;
    crate::validate_transfer_budget(transfer_bytes, owned_bytes)?;
    if owned_bytes as u64 > ir::MAX_OWNED_BYTES {
        return Err(Error::ResourceLimit("M4-02 owned storage exceeds 5 GiB"));
    }
    Ok(Shape {
        lengths,
        logical,
        offset,
        transfer_bytes,
    })
}

fn zeros(len: usize) -> Result<Vec<u32>, Error> {
    let mut words = Vec::new();
    words
        .try_reserve_exact(len)
        .map_err(|_| Error::ResourceLimit("CUDA transpose output allocation failed"))?;
    words.resize(len, 0);
    Ok(words)
}

fn validate_output(shape: &Shape, values: &[u32]) -> Result<(), Error> {
    if values.len() != shape.lengths[1] {
        return Err(Error::Driver("transpose output length refusal".into()));
    }
    let end = shape
        .offset
        .checked_add(shape.logical)
        .ok_or(Error::Limit)?;
    if values
        .get(..shape.offset)
        .ok_or(Error::Limit)?
        .iter()
        .chain(values.get(end..).ok_or(Error::Limit)?.iter())
        .any(|&word| word != 0)
    {
        return Err(Error::Driver("transpose output padding modified".into()));
    }
    Ok(())
}

/// Word-copy helper entry. Call only in the bounded compiler child. The actual
/// NVRTC version must exactly match caps. CUBIN requires sm_N instead of the
/// plan's virtual compute_N target; word-copy options remain ordered. The legacy
/// integer helper
/// omits modify-stack-limit=false, so it cannot compile this sealed plan.
pub fn compile_transpose_cubin(source: &str, caps: Capabilities) -> Result<CompiledCubin, Error> {
    let mut options = plan_options(source, caps)?;
    // SAFETY: optional-library presence probe precedes every NVRTC symbol access.
    if !unsafe { nvrtc_sys::is_culib_present() } {
        return Err(Error::Unavailable);
    }
    let (mut major, mut minor) = (0, 0);
    // SAFETY: NVRTC writes initialized local i32s; its library is present.
    unsafe { nvrtc_sys::nvrtcVersion(&mut major, &mut minor) }
        .result()
        .map_err(|e| Error::Compile(e.to_string()))?;
    let Capabilities::Cuda {
        major: wanted_major,
        minor: wanted_minor,
        sm,
    } = caps
    else {
        return Err(Error::Unavailable);
    };
    if (major, minor) != (wanted_major as i32, wanted_minor as i32) {
        return Err(Error::Compile(format!(
            "NVRTC {major}.{minor}; expected {wanted_major}.{wanted_minor}"
        )));
    }
    options[1] = format!("--gpu-architecture=sm_{sm}");
    let text = CString::new(source).map_err(|_| Error::Limit)?;
    let program = NvrtcProgram(
        nvrtc_result::create_program(&text, Some(c"invar_transpose_bits_v1.cu"))
            .map_err(|e| Error::Compile(e.to_string()))?,
    );
    let options: Vec<&str> = options.iter().map(String::as_str).collect();
    // SAFETY: program owns the live handle and source; options contain no NULs.
    let compiled = unsafe { nvrtc_result::compile_program(program.0, &options) };
    let log = program_log(&program)?;
    if let Err(error) = compiled {
        return Err(Error::Compile(format!("{error}: {log}")));
    }
    let mut bytes = 0usize;
    // SAFETY: live successfully compiled program; valid writable local size.
    unsafe { nvrtc_sys::nvrtcGetCUBINSize(program.0, &mut bytes) }
        .result()
        .map_err(|e| Error::Compile(e.to_string()))?;
    if bytes == 0 || bytes > MAX_CUBIN_BYTES {
        return Err(Error::Limit);
    }
    let mut image = Vec::new();
    image
        .try_reserve_exact(bytes)
        .map_err(|_| Error::OutOfMemory)?;
    image.resize(bytes, 0);
    // SAFETY: image owns exactly the size returned by the live NVRTC program.
    unsafe { nvrtc_sys::nvrtcGetCUBIN(program.0, image.as_mut_ptr().cast()) }
        .result()
        .map_err(|e| Error::Compile(e.to_string()))?;
    Ok(CompiledCubin { image, log })
}

fn program_log(program: &NvrtcProgram) -> Result<String, Error> {
    let mut bytes = 0usize;
    // SAFETY: program retains its live NVRTC handle; bytes is a writable local.
    unsafe { nvrtc_sys::nvrtcGetProgramLogSize(program.0, &mut bytes) }
        .result()
        .map_err(|e| Error::Compile(e.to_string()))?;
    if bytes == 0 {
        return Ok(String::new());
    }
    if bytes > 64 * 1024 {
        return Err(Error::Limit);
    }
    let mut log = Vec::new();
    log.try_reserve_exact(bytes)
        .map_err(|_| Error::OutOfMemory)?;
    log.resize(bytes, 0u8);
    // SAFETY: log owns the complete reported size; no truncated foreign buffer.
    unsafe { nvrtc_sys::nvrtcGetProgramLog(program.0, log.as_mut_ptr().cast()) }
        .result()
        .map_err(|e| Error::Compile(e.to_string()))?;
    log.truncate(
        log.iter()
            .position(|&c| c == 0)
            .unwrap_or(log.len())
            .min(MAX_ERROR_BYTES),
    );
    Ok(String::from_utf8_lossy(&log).into_owned())
}

pub(super) fn compile_bounded(
    source: &str,
    compiler: &Path,
    caps: Capabilities,
    timeout: Duration,
) -> Result<CompiledCubin, Error> {
    plan_options(source, caps)?;
    let Capabilities::Cuda { major, minor, sm } = caps else {
        return Err(Error::Unavailable);
    };
    let mut command = Command::new(compiler);
    command.args([
        major.to_string(),
        minor.to_string(),
        sm.to_string(),
        "transpose-bits-v1".into(),
    ]);
    super::compiler_child::compile(&mut command, source, MAX_CUBIN_BYTES, timeout, false)
}

/// One completion algorithm used by real CUDA work and deterministic fault
/// tests. The guard quarantines on failure, cancellation or unwinding before
/// confirmed quiescence. No host read is permitted by this path before then.
pub(super) trait PendingWork {
    fn submit(&mut self) -> Result<(), Error>;
    fn complete(&self) -> Result<bool, Error>;
    fn read(&self) -> Result<Vec<u32>, Error>;
}
struct Pending<W, H: FnOnce(Box<W>)> {
    work: Option<Box<W>>,
    hold: Option<H>,
    quiescent: bool,
}
impl<W, H: FnOnce(Box<W>)> Drop for Pending<W, H> {
    fn drop(&mut self) {
        if !self.quiescent
            && let (Some(work), Some(hold)) = (self.work.take(), self.hold.take())
        {
            hold(work);
        }
    }
}
pub(super) fn finish<W: PendingWork, H: FnOnce(Box<W>)>(
    work: Box<W>,
    hold: H,
    timeout: Duration,
    mut stop: impl FnMut() -> Option<Error>,
) -> Result<Vec<u32>, Error> {
    let mut pending = Pending {
        work: Some(work),
        hold: Some(hold),
        quiescent: false,
    };
    pending
        .work
        .as_deref_mut()
        .ok_or(Error::Poisoned)?
        .submit()?;
    let started = Instant::now();
    loop {
        if let Some(error) = stop() {
            return Err(error);
        }
        if started.elapsed() >= timeout {
            return Err(Error::Timeout);
        }
        if pending.work.as_deref().ok_or(Error::Poisoned)?.complete()? {
            pending.quiescent = true;
            return pending.work.as_deref().ok_or(Error::Poisoned)?.read();
        }
        std::thread::sleep(Duration::from_millis(1));
    }
}

struct TransposeWork {
    _context: Arc<CudaContext>,
    _module: Arc<CudaModule>,
    stream: Arc<CudaStream>,
    function: CudaFunction,
    input_host: PinnedHostSlice<u32>,
    input_dev: CudaSlice<u32>,
    output_host: PinnedHostSlice<u32>,
    output_dev: CudaSlice<u32>,
    completed: CudaEvent,
    blocks: u32,
}
impl PendingWork for TransposeWork {
    fn submit(&mut self) -> Result<(), Error> {
        self._context.check_err().map_err(driver_error)?;
        self.stream
            .memcpy_htod(&self.input_host, &mut self.input_dev)
            .map_err(driver_error)?;
        // The fresh zeroed output upload initializes padding and every allocation
        // byte before D2H can observe it, including excess masked output slots.
        self.stream
            .memcpy_htod(&self.output_host, &mut self.output_dev)
            .map_err(driver_error)?;
        let mut launch = self.stream.launch_builder(&self.function);
        launch.arg(&self.input_dev).arg(&mut self.output_dev);
        // SAFETY: sealed source fixes two unsigned-int pointers with checked
        // full capacities, offsets/strides and actual base alignment. Fresh
        // input/output allocations are disjoint and fully initialized. A fixed
        // 16x16x1 block and 1x1x1 grid satisfy the emitter's geometry; its two
        // unconditional tile barriers protect the padded static shared tile.
        // Every owner remains retained until a terminal same-stream fence.
        unsafe {
            launch.launch(LaunchConfig {
                grid_dim: (self.blocks, 1, 1),
                block_dim: (16, 16, 1),
                shared_mem_bytes: 0,
            })
        }
        .map_err(driver_error)?;
        self.stream
            .memcpy_dtoh(&self.output_dev, &mut self.output_host)
            .map_err(driver_error)?;
        self.completed.record(&self.stream).map_err(driver_error)
    }
    fn complete(&self) -> Result<bool, Error> {
        self._context.bind_to_thread().map_err(driver_error)?;
        self._context.check_err().map_err(driver_error)?;
        // SAFETY: the retained event was recorded after D2H on this context's
        // retained stream; query only inspects completion of its live handle.
        match unsafe { driver_sys::cuEventQuery(self.completed.cu_event()) }.result() {
            Ok(()) => Ok(true),
            Err(DriverError(CUresult::CUDA_ERROR_NOT_READY)) => Ok(false),
            Err(error) => Err(driver_error(error)),
        }
    }
    fn read(&self) -> Result<Vec<u32>, Error> {
        let values = self.output_host.as_slice().map_err(driver_error)?;
        let mut result = Vec::new();
        result
            .try_reserve_exact(values.len())
            .map_err(|_| Error::ResourceLimit("CUDA transpose output allocation failed"))?;
        result.extend_from_slice(values);
        Ok(result)
    }
}

fn validate_address(ptr: u64, alignment: u64) -> Result<(), Error> {
    if ptr == 0 || !alignment.is_power_of_two() || alignment < 4 || !ptr.is_multiple_of(alignment) {
        return Err(Error::Limit);
    }
    Ok(())
}

pub(super) fn pinned(
    context: &Arc<CudaContext>,
    values: Option<&[u32]>,
    len: usize,
) -> Result<PinnedHostSlice<u32>, Error> {
    // SAFETY: nonzero bounded pinned capacity is immediately zero-initialized
    // before any slice/transfer API; optional exact-length source is disjoint.
    let mut host = unsafe { context.alloc_pinned::<u32>(len) }.map_err(allocation_error)?;
    let ptr = host.as_mut_ptr().map_err(driver_error)?;
    // SAFETY: ptr owns len writable u32 slots of this fresh allocation.
    unsafe { std::ptr::write_bytes(ptr, 0, len) };
    if let Some(values) = values {
        if values.len() != len {
            return Err(Error::Limit);
        }
        // SAFETY: source length is checked against the disjoint initialized
        // pinned capacity and remains live through this synchronous host copy.
        unsafe { std::ptr::copy_nonoverlapping(values.as_ptr(), ptr, values.len()) };
    }
    Ok(host)
}

fn admit_before_storage<T>(
    admission: impl FnOnce() -> Result<(), Error>,
    allocation: impl FnOnce() -> Result<T, Error>,
) -> Result<T, Error> {
    admission()?;
    allocation()
}

impl CudaDevice {
    /// Run a verified bit transpose. Exact physical input capacity is
    /// required. Output is fresh physical storage; callers publish only the
    /// verified logical region after source/fuel/key/generation checks.
    /// Empty output domains enqueue nothing and read no input words.
    pub fn run_transpose(
        &mut self,
        plan: &ir::Verified,
        input: &[u32],
        compiler: &Path,
    ) -> Result<TransposeResult, Error> {
        self.run_transpose_with_fault(plan, input, compiler, super::StencilFault::None)
    }
    /// Same completed word path with the shared grid submission fault points.
    pub fn run_transpose_with_fault(
        &mut self,
        plan: &ir::Verified,
        input: &[u32],
        compiler: &Path,
        fault: super::StencilFault,
    ) -> Result<TransposeResult, Error> {
        self.admit()?;
        match fault {
            super::StencilFault::OutOfMemory => return Err(Error::OutOfMemory),
            super::StencilFault::LaunchFailure => {
                return Err(Error::Driver("injected pre-submission rejection".into()));
            }
            _ => {}
        }
        let shape = shape(plan, self.caps, input.len())?;
        let device_bytes = shape.lengths[0]
            .checked_add(shape.lengths[1])
            .and_then(|n| n.checked_mul(4))
            .ok_or(Error::Limit)?;
        if device_bytes > self.context.mem_get_info().map_err(driver_error)?.0 {
            return Err(Error::ResourceLimit(
                "M4-02 buffers exceed free CUDA device memory",
            ));
        }
        let source = ir::emit_cuda(plan).map_err(|e| Error::Compile(format!("{e:?}")))?;
        if plan_options(&source, self.caps)? != plan.nvrtc_options() {
            return Err(Error::Limit);
        }
        let hash = source_hash(&source);
        if shape.logical == 0 {
            return Ok(TransposeResult {
                output: zeros(shape.lengths[1])?,
                source_hash: hash,
                cubin_bytes: 0,
                transfer_bytes: 0,
                compile_ms: 0.0,
                nvrtc_log: String::new(),
            });
        }
        // Synchronously snapshot all raw bits into privately owned pinned
        // storage. No finite-word filter: transpose performs no arithmetic.
        let input_host = admit_before_storage(
            || self.check_submission(shape.transfer_bytes),
            || pinned(&self.context, Some(input), shape.lengths[0]),
        )?;
        let output_host = pinned(&self.context, None, shape.lengths[1])?;
        let started = Instant::now();
        let compiled = compile_bounded(&source, compiler, self.caps, COMPILE_DEADLINE)?;
        let compile_ms = started.elapsed().as_secs_f64() * 1000.0;
        let cubin_bytes = compiled.image.len();
        let module = self
            .context
            .load_module(Ptx::from_binary(compiled.image))
            .map_err(driver_error)?;
        let function = module
            .load_function(ir::KERNEL_NAME)
            .map_err(driver_error)?;
        if !(0..=4096).contains(&function.local_size_bytes().map_err(driver_error)?) {
            return Err(Error::Limit);
        }
        // SAFETY: submit uploads initialized owned staging before any kernel
        // read; the owning work retains both allocations through completion.
        let input_dev =
            unsafe { self.stream.alloc::<u32>(shape.lengths[0]) }.map_err(allocation_error)?;
        // SAFETY: the fresh zeroed output upload initializes the full capacity.
        let output_dev =
            unsafe { self.stream.alloc::<u32>(shape.lengths[1]) }.map_err(allocation_error)?;
        for (buffer, metadata) in [&input_dev, &output_dev]
            .into_iter()
            .zip([&plan.ir().input, &plan.ir().output])
        {
            let (ptr, _lease) = buffer.device_ptr(&self.stream);
            validate_address(ptr, metadata.alignment)?;
        }
        let completed = self.context.new_event(None).map_err(driver_error)?;
        let work = Box::new(TransposeWork {
            _context: self.context.clone(),
            _module: module,
            stream: self.stream.clone(),
            function,
            input_host,
            input_dev,
            output_host,
            output_dev,
            completed,
            blocks: (u64::from(plan.ir().dimensions[0]).div_ceil(16)
                * u64::from(plan.ir().dimensions[1]).div_ceil(16))
            .clamp(1, 65535) as u32,
        });
        let output = finish(
            work,
            |work| self.hold(work),
            KERNEL_DEADLINE,
            || (fault == super::StencilFault::PartialCompletion).then_some(Error::Timeout),
        )?;
        validate_output(&shape, &output)?;
        Ok(TransposeResult {
            output,
            source_hash: hash,
            cubin_bytes,
            transfer_bytes: shape.transfer_bytes,
            compile_ms,
            nvrtc_log: compiled.log,
        })
    }
}

#[cfg(test)]
#[path = "transpose_tests.rs"]
mod tests;
