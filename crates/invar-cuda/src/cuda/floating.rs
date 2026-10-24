//! Strict contraction execution with fresh private storage and an owning fence.
//! The root owns source-loop admission and publication; this module returns only
//! completed finite physical C storage. It never invokes the integer cache.

use super::transpose::{PendingWork, admit_before_storage, finish};
use super::*;
use cudarc::driver::{CudaModule, DevicePtr};
use invar_device::{f32_probe, floating as ir, tc};

#[derive(Debug)]
pub struct FloatingResult {
    /// Fresh physical C allocation, including zeroed prefix/tail padding.
    pub output: Vec<u32>,
    pub source_hash: u64,
    pub cubin_bytes: usize,
    pub transfer_bytes: usize,
    pub compile_ms: f64,
    pub nvrtc_log: String,
}

struct Shape {
    lengths: [usize; 3],
    logical: usize,
    offset: usize,
    transfer_bytes: usize,
}

fn finite(word: u32) -> bool {
    word & 0x7f800000 != 0x7f800000
}

fn validate_staging(plan: &ir::Verified, inputs: [&[u32]; 2]) -> Result<(), Error> {
    if plan.ir().inputs[0].id == plan.ir().inputs[1].id && inputs[0] != inputs[1] {
        return Err(Error::StaleBorrow);
    }
    if inputs.into_iter().flatten().any(|&word| !finite(word)) {
        return Err(Error::Driver("floating nonfinite input refusal".into()));
    }
    Ok(())
}

/// The numeric policy a contraction source header declares: strict, or the
/// explicit `tc-f16-f32-v1` contract. Both compile under the same options.
fn source_policy(source: &str) -> Option<&'static str> {
    let first = source.lines().next()?;
    [ir::NUMERIC_POLICY, ir::TC_POLICY]
        .into_iter()
        .find(|policy| {
            first
                == format!(
                    "// invar-f32-contract-v1 numeric={policy} numeric-version=1 layout-version=1"
                )
        })
}

fn strict_options(source: &str, caps: Capabilities) -> Result<Vec<String>, Error> {
    if source.len() > invar_device::MAX_SOURCE_BYTES || source.contains('\0') {
        return Err(Error::Limit);
    }
    let options = ir::cuda_strict_options(caps).ok_or(Error::Unavailable)?;
    if source_policy(source).is_none()
        || !source
            .lines()
            .nth(1)
            .is_some_and(|line| line.ends_with(&format!("caps={caps:?}")))
        || !source
            .lines()
            .any(|line| line == format!("// required-options: {}", options.join(" ")))
    {
        return Err(Error::Compile(
            "floating source/capability/options identity mismatch".into(),
        ));
    }
    Ok(options)
}

fn shape(plan: &ir::Verified, caps: Capabilities, lengths: [usize; 2]) -> Result<Shape, Error> {
    if plan.capabilities() != caps {
        return Err(Error::Limit);
    }
    let mut physical = [0; 3];
    for (i, buffer) in plan
        .ir()
        .inputs
        .iter()
        .chain(std::iter::once(&plan.ir().output))
        .enumerate()
    {
        physical[i] = usize::try_from(buffer.elements).map_err(|_| Error::Limit)?;
    }
    if lengths != [physical[0], physical[1]] {
        return Err(Error::Limit);
    }
    let [rows, _, cols] = plan.ir().dimensions;
    let logical = usize::try_from(u64::from(rows) * u64::from(cols)).map_err(|_| Error::Limit)?;
    let offset = usize::try_from(plan.ir().output.offset).map_err(|_| Error::Limit)?;
    if offset
        .checked_add(logical)
        .is_none_or(|end| end > physical[2])
    {
        return Err(Error::Limit);
    }
    // Empty input allocations use one initialized dummy word; K=0 kernels
    // never load either dummy. Bound host+device storage and result reservation.
    let inputs = physical[0]
        .max(1)
        .checked_add(physical[1].max(1))
        .ok_or(Error::Limit)?;
    let transfer_bytes = inputs
        .checked_add(physical[2].checked_mul(2).ok_or(Error::Limit)?)
        .and_then(|n| n.checked_mul(4))
        .ok_or(Error::Limit)?;
    let owned_bytes = inputs
        .checked_add(physical[2])
        .and_then(|n| n.checked_mul(2))
        .and_then(|n| n.checked_add(physical[2]))
        .and_then(|n| n.checked_mul(4))
        .ok_or(Error::Limit)?;
    crate::validate_transfer_budget(transfer_bytes, owned_bytes)?;
    Ok(Shape {
        lengths: physical,
        logical,
        offset,
        transfer_bytes,
    })
}

fn zeros(len: usize) -> Result<Vec<u32>, Error> {
    let mut words = Vec::new();
    words
        .try_reserve_exact(len)
        .map_err(|_| Error::ResourceLimit("CUDA floating output allocation failed"))?;
    words.resize(len, 0);
    Ok(words)
}

fn validate_output(shape: &Shape, values: &[u32]) -> Result<(), Error> {
    if values.len() != shape.lengths[2] || values.iter().any(|&word| !finite(word)) {
        return Err(Error::Driver(
            "floating output length/nonfinite refusal".into(),
        ));
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
        return Err(Error::Driver("floating output padding modified".into()));
    }
    Ok(())
}

/// The D1 probe source must be exactly the shared generated probe for `caps`.
fn probe_options(source: &str, caps: Capabilities) -> Result<Vec<String>, Error> {
    if source.len() > invar_device::MAX_SOURCE_BYTES || source.contains('\0') {
        return Err(Error::Limit);
    }
    if Some(source) != invar_device::f32_probe::emit_cuda(caps).as_deref() {
        return Err(Error::Compile(
            "floating probe source identity mismatch".into(),
        ));
    }
    ir::cuda_strict_options(caps).ok_or(Error::Unavailable)
}

/// Strict helper entry. Call only in the bounded compiler child. The actual
/// NVRTC version must exactly match caps. CUBIN requires sm_N instead of the
/// plan's virtual compute_N target; numeric options are unchanged and ordered.
pub fn compile_floating_cubin(source: &str, caps: Capabilities) -> Result<CompiledCubin, Error> {
    let options = strict_options(source, caps)?;
    if source_policy(source) != Some(ir::NUMERIC_POLICY) {
        return Err(Error::Compile(
            "floating helper mode/policy mismatch".into(),
        ));
    }
    compile_strict(source, caps, options, c"invar_f32_contract_v1.cu")
}

/// `tc-f16-f32-v1` helper entry: the same pinned options and NVRTC identity,
/// for a source whose header declares the explicit Tensor Core contract.
pub fn compile_tc_cubin(source: &str, caps: Capabilities) -> Result<CompiledCubin, Error> {
    let options = strict_options(source, caps)?;
    if source_policy(source) != Some(ir::TC_POLICY) {
        return Err(Error::Compile("tc helper mode/policy mismatch".into()));
    }
    compile_strict(source, caps, options, c"invar_tc_f16_f32_v1.cu")
}

/// Probe helper entry, compiled under exactly the strict contraction options.
pub fn compile_floating_probe_cubin(
    source: &str,
    caps: Capabilities,
) -> Result<CompiledCubin, Error> {
    let options = probe_options(source, caps)?;
    compile_strict(source, caps, options, c"invar_f32_probe_v1.cu")
}

/// Compile `source` under pinned `options` once NVRTC matches `caps` exactly;
/// shared by the floating, probe, tensor-core and transpose helpers.
pub(super) fn compile_strict(
    source: &str,
    caps: Capabilities,
    mut options: Vec<String>,
    file: &std::ffi::CStr,
) -> Result<CompiledCubin, Error> {
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
        nvrtc_result::create_program(&text, Some(file))
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

/// Helper-child protocol mode for the D1 probe kernel.
pub const PROBE_MODE: &str = "strict-f32-probe-v1";

fn compile_bounded(
    source: &str,
    compiler: &Path,
    caps: Capabilities,
    timeout: Duration,
) -> Result<CompiledCubin, Error> {
    strict_options(source, caps)?;
    let mode = source_policy(source).ok_or(Error::Limit)?;
    compile_child(source, compiler, caps, timeout, mode)
}

fn compile_probe_bounded(
    source: &str,
    compiler: &Path,
    caps: Capabilities,
    timeout: Duration,
) -> Result<CompiledCubin, Error> {
    probe_options(source, caps)?;
    compile_child(source, compiler, caps, timeout, PROBE_MODE)
}

fn compile_child(
    source: &str,
    compiler: &Path,
    caps: Capabilities,
    timeout: Duration,
    mode: &str,
) -> Result<CompiledCubin, Error> {
    let Capabilities::Cuda { major, minor, sm } = caps else {
        return Err(Error::Unavailable);
    };
    let mut command = Command::new(compiler);
    command.args([
        major.to_string(),
        minor.to_string(),
        sm.to_string(),
        mode.into(),
    ]);
    super::compiler_child::compile(&mut command, source, MAX_CUBIN_BYTES, timeout, false)
}

struct FloatingWork {
    _context: Arc<CudaContext>,
    _module: Arc<CudaModule>,
    stream: Arc<CudaStream>,
    function: CudaFunction,
    input_host: [PinnedHostSlice<u32>; 2],
    input_dev: [CudaSlice<u32>; 2],
    output_host: PinnedHostSlice<u32>,
    output_dev: CudaSlice<u32>,
    completed: CudaEvent,
    /// (groups, threads): the emitted kernel's verified one-dimensional launch.
    launch: (u32, u32),
}
impl PendingWork for FloatingWork {
    fn submit(&mut self) -> Result<(), Error> {
        self._context.check_err().map_err(driver_error)?;
        for (host, dev) in self.input_host.iter().zip(&mut self.input_dev) {
            self.stream.memcpy_htod(host, dev).map_err(driver_error)?;
        }
        // The fresh zeroed C upload initializes padding and every allocation
        // byte before D2H can observe it, including excess masked output slots.
        self.stream
            .memcpy_htod(&self.output_host, &mut self.output_dev)
            .map_err(driver_error)?;
        let mut launch = self.stream.launch_builder(&self.function);
        launch
            .arg(&self.input_dev[0])
            .arg(&self.input_dev[1])
            .arg(&mut self.output_dev);
        // SAFETY: verified source fixes three A/B/C 32-bit pointer arguments;
        // owned words carry checked f32/integer storage or probe packets.
        // Capacities include the f32-ops private status suffix when present.
        // Inputs and fresh output have separate aligned allocations, and
        // verification bounds every emitted load/store and rounding site.
        // One-dimensional geometry from the emitter that produced this
        // kernel (direct kernels are grid-strided; tiled kernels trap on any
        // other block size), static shared memory only; all owners retained.
        unsafe {
            launch.launch(LaunchConfig {
                grid_dim: (self.launch.0, 1, 1),
                block_dim: (self.launch.1, 1, 1),
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
            .map_err(|_| Error::ResourceLimit("CUDA floating output allocation failed"))?;
        result.extend_from_slice(values);
        Ok(result)
    }
}

fn pinned(
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
        if values.len() > len {
            return Err(Error::Limit);
        }
        // SAFETY: source length is checked against the disjoint initialized
        // pinned capacity and remains live through this synchronous host copy.
        unsafe { std::ptr::copy_nonoverlapping(values.as_ptr(), ptr, values.len()) };
    }
    Ok(host)
}

impl CudaDevice {
    /// Execute a verified rank-1 f32 program with privately staged packed words.
    /// The appended status word is checked only after the owning fence completes.
    pub fn run_f32_ops(
        &mut self,
        plan: &invar_device::v2::Verified,
        packed: &[u32],
        compiler: &Path,
    ) -> Result<Vec<u32>, Error> {
        self.admit()?;
        let p = invar_device::f32_ops::program(plan).map_err(|_| Error::Limit)?;
        if plan.capabilities() != self.caps || packed.len() != p.packed_len() {
            return Err(Error::Limit);
        }
        let report = self.strict_f32_probe(compiler)?;
        if !report.admits_strict() || !invar_device::f32_ops::admits_trig(p, report) {
            return Err(Error::Unavailable);
        }
        if p.output_len() == 0 {
            return Ok(Vec::new());
        }
        let source =
            invar_device::f32_ops::emit(plan).map_err(|e| Error::Compile(format!("{e:?}")))?;
        let n = p.output_len().checked_add(1).ok_or(Error::Limit)?;
        let input = [
            pinned(&self.context, Some(packed), packed.len().max(1))?,
            pinned(&self.context, None, 1)?,
        ];
        let mut initial = vec![0; n];
        initial[n - 1] = ir::guard::UNWRITTEN;
        let output = pinned(&self.context, Some(&initial), n)?;
        let compiled = compile_bounded(&source, compiler, self.caps, COMPILE_DEADLINE)?;
        let mut words = self.submit_owned(
            compiled.image,
            invar_device::f32_ops::KERNEL_NAME,
            input,
            output,
            [packed.len(), 1, n],
            [4; 3],
            (1, 32),
        )?;
        let status = words.pop().ok_or(Error::Limit)?;
        if ir::guard::decode_bits(status, invar_device::f32_ops::CUDA_STATUS)
            .map_err(Error::InvalidGuard)?
            != 0
        {
            return Err(Error::IntegerOverflow);
        }
        Ok(words)
    }
    /// Run a verified strict contraction. Exact physical input capacities are
    /// required. Output is fresh physical C storage; callers publish only the
    /// verified logical region after source/fuel/key/generation checks.
    /// Empty output domains enqueue nothing and read neither input allocation.
    pub fn run_contraction(
        &mut self,
        plan: &ir::Verified,
        inputs: [&[u32]; 2],
        compiler: &Path,
    ) -> Result<FloatingResult, Error> {
        let source = ir::emit_cuda(plan).map_err(|e| Error::Compile(format!("{e:?}")))?;
        self.run_emitted(plan, inputs, compiler, &source, (1, 32))
    }

    /// The M4-03 tiled schedule of the same verified plan, with the same
    /// admission, D1 probe, staging, completion and output checks.
    pub fn run_tiled_contraction(
        &mut self,
        plan: &ir::Verified,
        inputs: [&[u32]; 2],
        compiler: &Path,
    ) -> Result<FloatingResult, Error> {
        if plan.is_tc() {
            return self.run_tc_contraction(plan, inputs, compiler);
        }
        let source =
            invar_device::gemm::emit_cuda(plan).map_err(|e| Error::Compile(format!("{e:?}")))?;
        let geometry = invar_device::gemm::geometry(plan);
        self.run_emitted(
            plan,
            inputs,
            compiler,
            &source,
            (geometry.groups, geometry.threads),
        )
    }

    /// `tc-f16-f32-v1`: inputs are rounded to binary16 here (an element past
    /// the binary16 range or rounding to a binary16 subnormal is refused
    /// before any compile or submission), then
    /// the `mma.sync` schedule runs with the same admission and checks.
    fn run_tc_contraction(
        &mut self,
        plan: &ir::Verified,
        inputs: [&[u32]; 2],
        compiler: &Path,
    ) -> Result<FloatingResult, Error> {
        shape(plan, self.caps, [inputs[0].len(), inputs[1].len()])?;
        let source = tc::emit_cuda(plan).map_err(|e| Error::Compile(format!("{e:?}")))?;
        let refused = |r: tc::InputRange| Error::Driver(r.to_string());
        let a = tc::round_inputs(0, inputs[0]).map_err(refused)?;
        let b = tc::round_inputs(1, inputs[1]).map_err(refused)?;
        let geometry = tc::geometry(plan);
        self.run_emitted(
            plan,
            [&a, &b],
            compiler,
            &source,
            (geometry.groups, geometry.threads),
        )
    }

    fn run_emitted(
        &mut self,
        plan: &ir::Verified,
        inputs: [&[u32]; 2],
        compiler: &Path,
        source: &str,
        launch: (u32, u32),
    ) -> Result<FloatingResult, Error> {
        self.admit()?;
        let shape = shape(plan, self.caps, [inputs[0].len(), inputs[1].len()])?;
        let device_bytes = shape
            .lengths
            .iter()
            .try_fold(0usize, |sum, &n| {
                sum.checked_add(n.max(1)).ok_or(Error::Limit)
            })?
            .checked_mul(4)
            .ok_or(Error::Limit)?;
        let (free, _) = self.context.mem_get_info().map_err(driver_error)?;
        if device_bytes > free {
            return Err(Error::MatrixResourceLimit {
                resource: "CUDA free device memory",
                requested: device_bytes,
                limit: free,
            });
        }
        if strict_options(source, self.caps)? != plan.nvrtc_options()
            || source_policy(source) != Some(plan.ir().numeric_policy.as_str())
        {
            return Err(Error::Limit);
        }
        // D1: a device whose probe is not exact never runs a strict region.
        if !self.strict_f32_probe(compiler)?.admits_strict() {
            return Err(Error::Unavailable);
        }
        let hash = source_hash(source);
        if shape.logical == 0 {
            return Ok(FloatingResult {
                output: zeros(shape.lengths[2])?,
                source_hash: hash,
                cubin_bytes: 0,
                transfer_bytes: 0,
                compile_ms: 0.0,
                nvrtc_log: String::new(),
            });
        }
        // Stage privately and reject nonfinite input before any compiler child
        // or asynchronous CUDA submission. Caller pointers never reach a kernel.
        let input_host = admit_before_storage(
            || self.check_submission(shape.transfer_bytes),
            || {
                Ok([
                    pinned(&self.context, Some(inputs[0]), shape.lengths[0].max(1))?,
                    pinned(&self.context, Some(inputs[1]), shape.lengths[1].max(1))?,
                ])
            },
        )?;
        validate_staging(
            plan,
            [
                input_host[0].as_slice().map_err(driver_error)?,
                input_host[1].as_slice().map_err(driver_error)?,
            ],
        )?;
        let output_host = pinned(&self.context, None, shape.lengths[2])?;
        let started = Instant::now();
        let compiled = compile_bounded(source, compiler, self.caps, COMPILE_DEADLINE)?;
        let compile_ms = started.elapsed().as_secs_f64() * 1000.0;
        let cubin_bytes = compiled.image.len();
        let alignments = [
            plan.ir().inputs[0].alignment,
            plan.ir().inputs[1].alignment,
            plan.ir().output.alignment,
        ];
        let output = self.submit_owned(
            compiled.image,
            ir::KERNEL_NAME,
            input_host,
            output_host,
            shape.lengths,
            alignments,
            launch,
        )?;
        validate_output(&shape, &output)?;
        Ok(FloatingResult {
            output,
            source_hash: hash,
            cubin_bytes,
            transfer_bytes: shape.transfer_bytes,
            compile_ms,
            nvrtc_log: compiled.log,
        })
    }

    /// D1 per-process preflight, cached for this device: the shared probe
    /// table compiled under exactly the strict options and submitted through
    /// the same owned, fenced path as contractions. Strict work is refused
    /// unless the report admits it.
    pub fn strict_f32_probe(&mut self, compiler: &Path) -> Result<&f32_probe::Report, Error> {
        if self.strict_probe.is_none() {
            self.admit()?;
            let source = f32_probe::emit_cuda(self.caps).ok_or(Error::Unavailable)?;
            let compiled = compile_probe_bounded(&source, compiler, self.caps, COMPILE_DEADLINE)?;
            let words = f32_probe::words();
            // Two words per case: result and guard flag.
            let cases = f32_probe::CASES.len() * 2;
            // Both words of each case start unwritten.
            let sentinel = f32_probe::unwritten();
            let input_host = [
                pinned(&self.context, Some(&words), words.len())?,
                pinned(&self.context, None, 1)?,
            ];
            let output_host = pinned(&self.context, Some(&sentinel), cases)?;
            let observed = self.submit_owned(
                compiled.image,
                f32_probe::KERNEL_NAME,
                input_host,
                output_host,
                [words.len(), 1, cases],
                [4; 3],
                (1, 32),
            )?;
            let report = f32_probe::Report::evaluate(&observed).map_err(Error::InvalidProbe)?;
            self.strict_probe = Some(report);
        }
        self.strict_probe.as_ref().ok_or(Error::Poisoned)
    }

    /// Load one strict CUBIN and run it on fresh device copies of owned staging
    /// with the emitter's (groups, threads) launch. Every buffer is retained
    /// until the fence confirms quiescence; returns the completed output words.
    #[allow(clippy::too_many_arguments)] // Each owned input is moved into the fenced work.
    fn submit_owned(
        &mut self,
        image: Vec<u8>,
        kernel: &str,
        input_host: [PinnedHostSlice<u32>; 2],
        output_host: PinnedHostSlice<u32>,
        lengths: [usize; 3],
        alignments: [u64; 3],
        launch: (u32, u32),
    ) -> Result<Vec<u32>, Error> {
        let module = self
            .context
            .load_module(Ptx::from_binary(image))
            .map_err(driver_error)?;
        let function = module.load_function(kernel).map_err(driver_error)?;
        if !(0..=4096).contains(&function.local_size_bytes().map_err(driver_error)?) {
            return Err(Error::Limit);
        }
        // SAFETY: no allocation is read before the submit method uploads fully
        // initialized owned staging; FloatingWork retains all buffers to fence.
        let input_dev = [
            unsafe { self.stream.alloc::<u32>(lengths[0].max(1)) }.map_err(allocation_error)?,
            unsafe { self.stream.alloc::<u32>(lengths[1].max(1)) }.map_err(allocation_error)?,
        ];
        // SAFETY: the private initialized C upload writes all physical words.
        let output_dev =
            unsafe { self.stream.alloc::<u32>(lengths[2]) }.map_err(allocation_error)?;
        for (buffer, alignment) in input_dev
            .iter()
            .chain(std::iter::once(&output_dev))
            .zip(alignments)
        {
            let (ptr, _lease) = buffer.device_ptr(&self.stream);
            if ptr == 0 || ptr % alignment != 0 {
                return Err(Error::Limit);
            }
        }
        let completed = self.context.new_event(None).map_err(driver_error)?;
        let work = Box::new(FloatingWork {
            _context: self.context.clone(),
            _module: module,
            stream: self.stream.clone(),
            function,
            input_host,
            input_dev,
            output_host,
            output_dev,
            completed,
            launch,
        });
        finish(work, |work| self.hold(work), KERNEL_DEADLINE, || None)
    }
}

#[cfg(test)]
#[path = "floating_tests.rs"]
mod tests;
