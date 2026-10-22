//! The one CUDA Driver boundary. The device and NVRTC select the CUDA target.

use std::{
    collections::VecDeque,
    ffi::CString,
    path::Path,
    process::Command,
    sync::Arc,
    time::{Duration, Instant},
};

use cudarc::driver::{
    CudaContext, CudaEvent, CudaFunction, CudaSlice, CudaStream, LaunchConfig, PinnedHostSlice,
    PushKernelArg,
};
use cudarc::driver::{
    result::DriverError,
    sys::{self as driver_sys, CUevent_flags, CUresult},
};
use cudarc::nvrtc::{Ptx, result as nvrtc_result, sys as nvrtc_sys};
use invar_device::{Capabilities, Ir, Verified, emit_cuda, source_hash};

use crate::{
    Device, Error, Quarantine, SubmitError, validate_device_capacity, validate_transfer_len,
};

mod bench;
mod bits_map;
mod collective;
mod collective_bench;
mod compiler_child;
mod floating;
mod fsum;
mod fused;
mod resident;
pub use resident::ResidentMap;
pub mod gnet;
mod indexed;
mod law_collective;
mod pathsum;
mod segmented;
mod sorting;
mod spatial;
mod stencil_u32;
mod summary;
pub mod task;
mod transpose;
pub use bench::{BenchMetric, BenchResult};
pub use bits_map::{BitsMap, BitsMapRun, compile_with_options};
pub use collective::{CollectiveFault, CollectiveResult};
pub use collective_bench::CollectiveBenchResult;
pub use floating::{
    FloatingResult, PROBE_MODE, compile_floating_cubin, compile_floating_probe_cubin,
    compile_tc_cubin,
};
pub use fsum::{FsumBench, FsumResult};
pub use fused::{FusedFault, FusedResult};
pub use pathsum::{PATHSUM_GROUP, PathSum};
pub use segmented::SegmentedResult;
pub use sorting::SortResult;
pub use spatial::SpatialResult;
pub use stencil_u32::{StencilFault, StencilResult};
pub use summary::{SummaryBench, SummaryResult};
pub use transpose::{TransposeResult, compile_transpose_cubin};

pub const MAX_ERROR_BYTES: usize = 512;

const MAX_CUBIN_BYTES: usize = 1024 * 1024;
/// Program-specific general-net modules carry every generated continuation,
/// child projection and discovery routine; NVRTC 12.9 emits up to ~8.5 MiB
/// for the GNET-03 adversarial corpus (3.8 MiB for the largest runnable
/// fixture). Still a hard cap on compiler output and module load.
pub const MAX_GNET_CUBIN_BYTES: usize = 16 * 1024 * 1024;
const KERNEL_DEADLINE: Duration = Duration::from_secs(10);
const COMPILE_DEADLINE: Duration = Duration::from_secs(20);
/// Uncached NVRTC 12.9 (sm_120) wall time grows with the number of generated
/// continuations and discovery sites: measured 13-25 s for ordinary GNET-03
/// fixtures and 127 s for the widest adversarial program (174 states, ten
/// discovered call sites). Still a hard bound on the compiler subprocess.
const GNET_COMPILE_DEADLINE: Duration = Duration::from_secs(300);
const MAX_MODULES: usize = 8;
const MAX_QUARANTINE: usize = 8;

struct NvrtcProgram(nvrtc_sys::nvrtcProgram);

impl Drop for NvrtcProgram {
    fn drop(&mut self) {
        // SAFETY: this handle is created exactly once by NVRTC below, remains
        // live throughout compilation, and is destroyed exactly once here.
        let _ = unsafe { nvrtc_result::destroy_program(self.0) };
    }
}

pub struct CompiledCubin {
    pub image: Vec<u8>,
    pub log: String,
}

/// The numeric options `compile_gnet_cubin` adds after the architecture and
/// language standard. Compiled-ahead images record them in their header.
pub const GNET_STRICT_NVRTC_OPTIONS: [&str; 4] = [
    "--fmad=false",
    "--ftz=false",
    "--prec-div=true",
    "--prec-sqrt=true",
];

pub fn compile_cubin(source: &str, caps: Capabilities) -> Result<CompiledCubin, Error> {
    compile_cubin_with_cap(
        source,
        caps,
        invar_device::MAX_SOURCE_BYTES,
        MAX_CUBIN_BYTES,
        false,
    )
}
pub fn compile_gnet_cubin(source: &str, caps: Capabilities) -> Result<CompiledCubin, Error> {
    compile_cubin_with_cap(
        source,
        caps,
        invar_device::MAX_GNET_SOURCE_BYTES,
        MAX_GNET_CUBIN_BYTES,
        true,
    )
}
fn compile_cubin_with_cap(
    source: &str,
    caps: Capabilities,
    source_cap: usize,
    cubin_cap: usize,
    strict: bool,
) -> Result<CompiledCubin, Error> {
    if source.len() > source_cap || source.contains('\0') {
        return Err(Error::Limit);
    }
    let mut major = 0;
    let mut minor = 0;
    // SAFETY: both pointers address initialized i32 locals; NVRTC writes only
    // their values. The version check rejects every unpinned compiler.
    unsafe { nvrtc_sys::nvrtcVersion(&mut major, &mut minor) }
        .result()
        .map_err(|e| Error::Compile(e.to_string()))?;
    let Capabilities::Cuda {
        major: expected_major,
        minor: expected_minor,
        sm,
    } = caps
    else {
        return Err(Error::Unavailable);
    };
    if (major, minor) != (expected_major as i32, expected_minor as i32) {
        return Err(Error::Compile(format!(
            "NVRTC {major}.{minor}; expected {expected_major}.{expected_minor}"
        )));
    }
    let text = CString::new(source).map_err(|_| Error::Limit)?;
    let program = NvrtcProgram(
        nvrtc_result::create_program(&text, Some(c"invar_map_v1.cu"))
            .map_err(|e| Error::Compile(e.to_string()))?,
    );
    let architecture = format!("--gpu-architecture=sm_{sm}");
    let mut options = vec![architecture.as_str(), "--std=c++17"];
    if strict {
        options.extend(GNET_STRICT_NVRTC_OPTIONS);
    }
    if let Err(error) =
        // SAFETY: `program` owns a live NVRTC handle, and these options contain
        // no interior NUL. NvrtcProgram keeps the source alive until drop.
        unsafe { nvrtc_result::compile_program(program.0, &options) }
    {
        // SAFETY: compilation failed but the NVRTC program remains live.
        let log = unsafe { nvrtc_result::get_program_log(program.0) }
            .map(|chars| {
                chars
                    .into_iter()
                    .take(MAX_ERROR_BYTES)
                    .map(nvrtc_log_byte)
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();
        return Err(Error::Compile(format!(
            "{error}: {}",
            String::from_utf8_lossy(&log)
        )));
    }
    let mut bytes = 0usize;
    // SAFETY: NVRTC writes one usize to the valid local; compilation succeeded.
    unsafe { nvrtc_sys::nvrtcGetCUBINSize(program.0, &mut bytes) }
        .result()
        .map_err(|e| Error::Compile(e.to_string()))?;
    if bytes == 0 || bytes > cubin_cap {
        return Err(Error::Limit);
    }
    let mut image = vec![0u8; bytes];
    // SAFETY: NVRTC reports `bytes` as the required CUBIN size, and `image`
    // owns exactly that many initialized writable bytes while program is live.
    unsafe { nvrtc_sys::nvrtcGetCUBIN(program.0, image.as_mut_ptr().cast()) }
        .result()
        .map_err(|e| Error::Compile(e.to_string()))?;
    // SAFETY: successful compilation leaves `program` live for log retrieval.
    let raw_log = unsafe { nvrtc_result::get_program_log(program.0) }
        .map_err(|e| Error::Compile(e.to_string()))?;
    let log_bytes = raw_log
        .into_iter()
        .take(MAX_ERROR_BYTES)
        .map(nvrtc_log_byte)
        .take_while(|c| *c != 0)
        .collect::<Vec<_>>();
    Ok(CompiledCubin {
        image,
        log: String::from_utf8_lossy(&log_bytes).into_owned(),
    })
}

/// Reinterpret one NVRTC log `c_char` as its byte. `c_char` is `i8` on some
/// targets (x86_64 Linux) and `u8` on others (aarch64 Linux); this keeps the
/// exact bit pattern on both without a target-specific cast.
fn nvrtc_log_byte(c: std::ffi::c_char) -> u8 {
    u8::from_ne_bytes(c.to_ne_bytes())
}

/// Keep NVRTC's synchronous compiler behind a killable, capped child process.
pub fn compile_cubin_bounded(
    source: &str,
    compiler: &Path,
    caps: Capabilities,
) -> Result<CompiledCubin, Error> {
    compile_cubin_bounded_with_cap(source, compiler, caps, false)
}
pub fn compile_gnet_cubin_bounded(
    source: &str,
    compiler: &Path,
    caps: Capabilities,
) -> Result<CompiledCubin, Error> {
    compile_cubin_bounded_with_cap(source, compiler, caps, true)
}
fn compile_cubin_bounded_with_cap(
    source: &str,
    compiler: &Path,
    caps: Capabilities,
    gnet: bool,
) -> Result<CompiledCubin, Error> {
    let (source_cap, cubin_cap, deadline) = if gnet {
        (
            invar_device::MAX_GNET_SOURCE_BYTES,
            MAX_GNET_CUBIN_BYTES,
            GNET_COMPILE_DEADLINE,
        )
    } else {
        (
            invar_device::MAX_SOURCE_BYTES,
            MAX_CUBIN_BYTES,
            COMPILE_DEADLINE,
        )
    };
    if source.len() > source_cap || source.contains('\0') {
        return Err(Error::Limit);
    }
    let Capabilities::Cuda { major, minor, sm } = caps else {
        return Err(Error::Unavailable);
    };
    let mut command = Command::new(compiler);
    command.args([major.to_string(), minor.to_string(), sm.to_string()]);
    if gnet {
        command.arg("--gnet-v5");
    }
    compiler_child::compile(&mut command, source, cubin_cap, deadline, gnet)
}

/// The loadable NVRTC's version. Build-time only: this loads NVRTC.
pub fn nvrtc_version() -> Result<(u32, u32), Error> {
    // SAFETY: the probe only tries to load the compiler library; an absent
    // library fails here before cudarc's lazy loader can panic.
    if !unsafe { nvrtc_sys::is_culib_present() } {
        return Err(Error::Unavailable);
    }
    let mut major = 0;
    let mut minor = 0;
    // SAFETY: NVRTC writes two initialized i32 locals; its library is present.
    unsafe { nvrtc_sys::nvrtcVersion(&mut major, &mut minor) }
        .result()
        .map_err(|error| Error::Compile(error.to_string()))?;
    Ok((
        u32::try_from(major).map_err(|_| Error::Unavailable)?,
        u32::try_from(minor).map_err(|_| Error::Unavailable)?,
    ))
}

fn driver_error(error: DriverError) -> Error {
    match error.0 {
        CUresult::CUDA_ERROR_OUT_OF_MEMORY => Error::OutOfMemory,
        CUresult::CUDA_ERROR_INVALID_HANDLE => Error::InvalidHandle,
        _ => Error::Driver(error.to_string().chars().take(MAX_ERROR_BYTES).collect()),
    }
}

fn allocation_error(error: DriverError) -> Error {
    match driver_error(error) {
        Error::OutOfMemory => Error::ResourceLimit("CUDA host/device allocation failed"),
        other => other,
    }
}

pub struct CudaDevice {
    context: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    name: String,
    caps: Capabilities,
    driver: u32,
    quarantine: Quarantine<Held>,
    modules: VecDeque<ModuleEntry>,
    /// D1 strict binary32 probe verdict, run at most once per opened device.
    strict_probe: Option<invar_device::f32_probe::Report>,
    /// v0.2 fsum tree and exact functions, compiled once per opened device.
    fsum: Option<Vec<CudaFunction>>,
}

/// Work whose completion was not confirmed, owned until `fence` completes.
/// The fence is recorded on the same stream after the work, so its completion
/// implies the work's. A missing fence (record failed) is never reaped.
struct Held {
    _work: Box<dyn std::any::Any>,
    fence: Option<CudaEvent>,
}

struct ModuleEntry {
    ir: Ir,
    caps: Capabilities,
    source: String,
    names: Vec<String>,
    functions: Vec<CudaFunction>,
    cubin_bytes: usize,
}

struct LoadedFunctions {
    functions: Vec<CudaFunction>,
    cubin_bytes: usize,
    compile_ms: f64,
    nvrtc_log: String,
    cache_hit: bool,
}

pub enum CudaWork {
    Empty,
    Active {
        input: PinnedHostSlice<u8>,
        output: PinnedHostSlice<u8>,
        buffer: CudaSlice<u8>,
        event: CudaEvent,
        stream: Arc<CudaStream>,
        recorded: bool,
    },
}

struct MapWork {
    input_host: PinnedHostSlice<u32>,
    output_host: PinnedHostSlice<u32>,
    input_dev: CudaSlice<u32>,
    output_dev: CudaSlice<u32>,
    function: CudaFunction,
    h2d_start: CudaEvent,
    h2d_end: CudaEvent,
    kernel_end: CudaEvent,
    event: CudaEvent,
    stream: Arc<CudaStream>,
    length: u32,
}

#[derive(Debug)]
pub struct MapResult {
    pub values: Vec<u32>,
    pub source_hash: u64,
    pub cubin_bytes: usize,
    pub transfer_bytes: usize,
    pub blocks: u32,
    pub lanes: u32,
    pub compile_ms: f64,
    pub h2d_ms: f32,
    pub kernel_ms: f32,
    pub d2h_ms: f32,
    pub nvrtc_log: String,
    pub cache_hit: bool,
}

impl CudaDevice {
    /// Open device zero with its actual CUDA architecture and NVRTC version.
    pub fn open() -> Result<Self, Error> {
        let (major, minor) = nvrtc_version()?;
        let mut device = Self::open_driver()?;
        if major != 12 || minor < 4 {
            return Err(Error::Unavailable);
        }
        device.caps = Capabilities::Cuda {
            major,
            minor,
            sm: device.sm(),
        };
        Ok(device)
    }

    /// Open device zero through the Driver API alone, for compiled-ahead
    /// images. Never probes or loads NVRTC. Its capabilities carry compiler
    /// version 0.0, which no pinned NVRTC matches, so every source-compiling
    /// route on this device refuses instead of compiling.
    pub fn open_driver() -> Result<Self, Error> {
        // SAFETY: the probe only tries to load the driver library; an absent
        // library fails here before cudarc's lazy loader can panic.
        if !unsafe { driver_sys::is_culib_present() } {
            return Err(Error::Unavailable);
        }
        let context = CudaContext::new(0).map_err(driver_error)?;
        let (compute_major, compute_minor) = context.compute_capability().map_err(driver_error)?;
        let sm = compute_major
            .checked_mul(10)
            .and_then(|value| value.checked_add(compute_minor))
            .and_then(|value| u32::try_from(value).ok())
            .ok_or(Error::Unavailable)?;
        if sm < 50 {
            return Err(Error::Unavailable);
        }
        let mut driver = 0;
        // SAFETY: the driver library is present and initialized by the
        // context above; it writes one initialized i32 local.
        unsafe { driver_sys::cuDriverGetVersion(&mut driver) }
            .result()
            .map_err(driver_error)?;
        let driver = u32::try_from(driver).map_err(|_| Error::Unavailable)?;
        let name = context.name().map_err(driver_error)?;
        let stream = context.new_stream().map_err(driver_error)?;
        Ok(Self {
            context,
            stream,
            name,
            caps: Capabilities::Cuda {
                major: 0,
                minor: 0,
                sm,
            },
            driver,
            quarantine: Quarantine::new(MAX_QUARANTINE),
            modules: VecDeque::new(),
            strict_probe: None,
            fsum: None,
        })
    }

    /// The device's compute capability as `major * 10 + minor` (e.g. 120).
    pub fn sm(&self) -> u32 {
        match self.caps {
            Capabilities::Cuda { sm, .. } => sm,
            _ => 0,
        }
    }

    /// `cuDriverGetVersion`, e.g. 13000 for a CUDA 13.0 driver.
    pub fn driver_version(&self) -> u32 {
        self.driver
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn capabilities(&self) -> Capabilities {
        self.caps
    }

    /// Whether dispatch is refused: unreaped work remains, or quarantine
    /// overflow leaked work whose completion can never be confirmed.
    pub fn is_poisoned(&self) -> bool {
        self.quarantine.blocks_dispatch()
    }

    pub fn quarantine_len(&self) -> usize {
        self.quarantine.len()
    }

    /// Release quarantined work whose stream fence has completed.
    pub fn reap(&mut self) {
        self.quarantine
            .reap(|held| held.fence.as_ref().is_some_and(CudaEvent::is_complete));
    }

    /// Keep `work` and everything it owns alive until a later stream fence
    /// confirms quiescence. Never frees, publishes or replays it.
    fn hold<T: 'static>(&mut self, work: Box<T>) {
        let fence = self
            .context
            .new_event(None)
            .ok()
            .filter(|event| event.record(&self.stream).is_ok());
        self.quarantine.push(Held { _work: work, fence });
    }

    /// Reap confirmed work, then refuse dispatch while any remains.
    fn admit(&mut self) -> Result<(), Error> {
        self.reap();
        if self.is_poisoned() {
            return Err(Error::Poisoned);
        }
        Ok(())
    }

    /// Cache only functions compiled from this process's freshly verified IR.
    /// Each function owns its loaded module through an Arc; submitted work keeps
    /// its own clone alive even when an idle cache entry is evicted.
    fn load_functions(
        &mut self,
        plan: &Verified,
        source: &str,
        names: &[&str],
        compiler: &Path,
    ) -> Result<LoadedFunctions, Error> {
        if plan.capabilities() != self.caps || names.is_empty() {
            return Err(Error::Limit);
        }
        if let Some(index) = self.modules.iter().position(|entry| {
            entry.ir == *plan.ir()
                && entry.caps == self.caps
                && entry.source == source
                && entry
                    .names
                    .iter()
                    .map(String::as_str)
                    .eq(names.iter().copied())
        }) {
            let entry = self.modules.remove(index).ok_or(Error::InvalidHandle)?;
            let loaded = LoadedFunctions {
                functions: entry.functions.clone(),
                cubin_bytes: entry.cubin_bytes,
                compile_ms: 0.0,
                nvrtc_log: String::new(),
                cache_hit: true,
            };
            self.modules.push_back(entry);
            return Ok(loaded);
        }
        let started = Instant::now();
        let compiled = compile_cubin_bounded(source, compiler, self.caps)?;
        let compile_ms = started.elapsed().as_secs_f64() * 1000.0;
        let cubin_bytes = compiled.image.len();
        let module = self
            .context
            .load_module(Ptx::from_binary(compiled.image))
            .map_err(driver_error)?;
        let functions = names
            .iter()
            .map(|name| module.load_function(name).map_err(driver_error))
            .collect::<Result<Vec<_>, _>>()?;
        self.modules.push_back(ModuleEntry {
            ir: plan.ir().clone(),
            caps: self.caps,
            source: source.to_owned(),
            names: names.iter().map(|name| (*name).to_owned()).collect(),
            functions: functions.clone(),
            cubin_bytes,
        });
        if self.modules.len() > MAX_MODULES {
            self.modules.pop_front();
        }
        Ok(LoadedFunctions {
            functions,
            cubin_bytes,
            compile_ms,
            nvrtc_log: compiled.log,
            cache_hit: false,
        })
    }

    /// Advisory available memory. Submission rechecks it before allocation.
    pub fn available_capacity(&self) -> Result<usize, Error> {
        self.context
            .mem_get_info()
            .map(|(free, _)| free)
            .map_err(|_| Error::ResourceLimit("CUDA free-memory query failed"))
    }

    fn check_submission(&self, bytes: usize) -> Result<(), Error> {
        let (free, _) = self
            .context
            .mem_get_info()
            .map_err(|_| Error::ResourceLimit("CUDA free-memory query failed"))?;
        validate_device_capacity(bytes, free)
    }

    /// Compile one verified map with pinned NVRTC and submit it on this owned stream.
    pub fn run_map(
        &mut self,
        plan: &Verified,
        input: &[u32],
        compiler: &Path,
    ) -> Result<MapResult, Error> {
        self.run_map_with_fault(plan, input, compiler, FusedFault::None)
    }

    #[doc(hidden)]
    pub fn run_map_with_fault(
        &mut self,
        plan: &Verified,
        input: &[u32],
        compiler: &Path,
        fault: FusedFault,
    ) -> Result<MapResult, Error> {
        self.admit()?;
        if plan.ir().dag.is_some()
            || plan.capabilities() != self.caps
            || input.len() != plan.ir().length as usize
        {
            return Err(Error::Limit);
        }
        match fault {
            FusedFault::FalseProfitability => return Err(Error::Unprofitable),
            FusedFault::TransferOutOfMemory => return Err(Error::OutOfMemory),
            FusedFault::LaunchFailure => return Err(Error::InvalidHandle),
            _ => {}
        }
        validate_transfer_len(
            input
                .len()
                .checked_mul(8)
                .ok_or(Error::ResourceLimit("CUDA map byte count overflow"))?,
        )?;
        let source = emit_cuda(plan).map_err(|e| Error::Compile(format!("{e:?}")))?;
        let hash = source_hash(&source);
        if input.is_empty() {
            return Ok(MapResult {
                values: Vec::new(),
                source_hash: hash,
                cubin_bytes: 0,
                transfer_bytes: 0,
                blocks: plan.ir().geometry.blocks,
                lanes: plan.ir().geometry.lanes,
                compile_ms: 0.0,
                h2d_ms: 0.0,
                kernel_ms: 0.0,
                d2h_ms: 0.0,
                nvrtc_log: String::new(),
                cache_hit: false,
            });
        }
        self.check_submission(input.len() * 8)?;
        let mut loaded = self.load_functions(plan, &source, &["invar_map_v1"], compiler)?;
        let function = loaded.functions.remove(0);
        // SAFETY: the pinned input is fully written by the copy below before
        // any H2D transfer can read it.
        let mut input_host =
            unsafe { self.context.alloc_pinned::<u32>(input.len()) }.map_err(allocation_error)?;
        // SAFETY: the pinned output is zero-filled below before the D2H API
        // borrows it, and the completion event gates every host read.
        let mut output_host =
            unsafe { self.context.alloc_pinned::<u32>(input.len()) }.map_err(allocation_error)?;
        let src_ptr = input_host.as_mut_ptr().map_err(driver_error)?;
        let dst_ptr = output_host.as_mut_ptr().map_err(driver_error)?;
        // SAFETY: both pinned allocations have `input.len()` writable u32 slots;
        // the source slice is disjoint and remains live for this synchronous copy.
        unsafe { std::ptr::copy_nonoverlapping(input.as_ptr(), src_ptr, input.len()) };
        // SAFETY: the D2H transfer's mutable-slice API requires initialized
        // host storage, and all `input.len()` u32 slots belong to output_host.
        unsafe { std::ptr::write_bytes(dst_ptr, 0, input.len()) };
        // SAFETY: all device slots are written before any kernel or host read;
        // the boxed work retains both allocations until completion is proven.
        let input_dev =
            unsafe { self.stream.alloc::<u32>(input.len()) }.map_err(allocation_error)?;
        // SAFETY: the kernel writes every masked output slot before the D2H
        // copy on the same stream reads it; MapWork retains the allocation.
        let output_dev =
            unsafe { self.stream.alloc::<u32>(input.len()) }.map_err(allocation_error)?;
        // cudarc defaults to CU_EVENT_DISABLE_TIMING; elapsed_ms requires timed events.
        let h2d_start = self
            .context
            .new_event(Some(CUevent_flags::CU_EVENT_DEFAULT))
            .map_err(driver_error)?;
        let h2d_end = self
            .context
            .new_event(Some(CUevent_flags::CU_EVENT_DEFAULT))
            .map_err(driver_error)?;
        let kernel_end = self
            .context
            .new_event(Some(CUevent_flags::CU_EVENT_DEFAULT))
            .map_err(driver_error)?;
        let event = self
            .context
            .new_event(Some(CUevent_flags::CU_EVENT_DEFAULT))
            .map_err(driver_error)?;
        let mut work = Box::new(MapWork {
            input_host,
            output_host,
            input_dev,
            output_dev,
            function,
            h2d_start,
            h2d_end,
            kernel_end,
            event,
            stream: self.stream.clone(),
            length: plan.ir().length,
        });
        let submitted = (|| -> Result<(), Error> {
            work.h2d_start.record(&work.stream).map_err(driver_error)?;
            work.stream
                .memcpy_htod(&work.input_host, &mut work.input_dev)
                .map_err(driver_error)?;
            work.h2d_end.record(&work.stream).map_err(driver_error)?;
            let geometry = plan.ir().geometry;
            let mut launch = work.stream.launch_builder(&work.function);
            launch
                .arg(&work.input_dev)
                .arg(&mut work.output_dev)
                .arg(&work.length);
            // SAFETY: `Verified` proves exact contiguous u32 buffers, disjoint
            // input/output, in-bounds masked indexing, fixed ABI argument order,
            // and legal geometry. MapWork retains all arguments and module.
            unsafe {
                launch.launch(LaunchConfig {
                    grid_dim: (geometry.blocks, 1, 1),
                    block_dim: (geometry.lanes, 1, 1),
                    shared_mem_bytes: 0,
                })
            }
            .map_err(driver_error)?;
            work.kernel_end.record(&work.stream).map_err(driver_error)?;
            if fault == FusedFault::PartialCompletion {
                return Err(Error::Timeout);
            }
            work.stream
                .memcpy_dtoh(&work.output_dev, &mut work.output_host)
                .map_err(driver_error)?;
            work.event.record(&work.stream).map_err(driver_error)?;
            Ok(())
        })();
        if let Err(error) = submitted {
            // Any failed enqueue may already be in flight. No destructor may
            // free pinned/device storage or unload the module before quiescence.
            self.hold(work);
            return Err(error);
        }
        let started = Instant::now();
        while !work.event.is_complete() {
            if started.elapsed() >= KERNEL_DEADLINE {
                self.hold(work);
                return Err(Error::Timeout);
            }
            std::thread::sleep(Duration::from_millis(1));
        }
        let h2d_ms = work
            .h2d_start
            .elapsed_ms(&work.h2d_end)
            .map_err(driver_error)?;
        let kernel_ms = work
            .h2d_end
            .elapsed_ms(&work.kernel_end)
            .map_err(driver_error)?;
        let d2h_ms = work
            .kernel_end
            .elapsed_ms(&work.event)
            .map_err(driver_error)?;
        let slice = work.output_host.as_slice().map_err(driver_error)?;
        let mut values = Vec::new();
        values
            .try_reserve_exact(slice.len())
            .map_err(|_| Error::ResourceLimit("host output allocation failed"))?;
        values.extend_from_slice(slice);
        Ok(MapResult {
            values,
            source_hash: hash,
            cubin_bytes: loaded.cubin_bytes,
            transfer_bytes: input.len() * 2 * std::mem::size_of::<u32>(),
            blocks: plan.ir().geometry.blocks,
            lanes: plan.ir().geometry.lanes,
            compile_ms: loaded.compile_ms,
            h2d_ms,
            kernel_ms,
            d2h_ms,
            nvrtc_log: loaded.nvrtc_log,
            cache_hit: loaded.cache_hit,
        })
    }
}

impl Device for CudaDevice {
    type Work = CudaWork;

    fn prepare(&mut self, bytes: &[u8]) -> Result<Self::Work, Error> {
        crate::validate_roundtrip_len(bytes.len())?;
        self.check_submission(bytes.len())?;
        if bytes.is_empty() {
            return Ok(CudaWork::Empty);
        }
        // SAFETY: the caller's slice is copied into this separately owned
        // pinned storage before any transfer is enqueued; the bounded nonzero
        // length is checked by Runtime.
        let mut input =
            unsafe { self.context.alloc_pinned::<u8>(bytes.len()) }.map_err(allocation_error)?;
        // SAFETY: zero-filled below before the D2H mutable-slice API sees it;
        // CudaWork retains it until the recorded event completes.
        let mut output =
            unsafe { self.context.alloc_pinned::<u8>(bytes.len()) }.map_err(allocation_error)?;
        let destination = input.as_mut_ptr().map_err(driver_error)?;
        let output_ptr = output.as_mut_ptr().map_err(driver_error)?;
        // SAFETY: `destination` points to `bytes.len()` writable pinned u8
        // slots owned by `input`; source has the same length and is disjoint.
        unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), destination, bytes.len()) };
        // SAFETY: `output_ptr` owns `bytes.len()` writable u8 slots. cudarc's
        // subsequent mutable-slice transfer requires initialized storage.
        unsafe { std::ptr::write_bytes(output_ptr, 0, bytes.len()) };
        // SAFETY: The allocation is initialized by the H2D copy before any
        // D2H read on this one stream, and its lifetime is held in CudaWork.
        let buffer = unsafe { self.stream.alloc::<u8>(bytes.len()) }.map_err(allocation_error)?;
        let event = self.context.new_event(None).map_err(driver_error)?;
        Ok(CudaWork::Active {
            input,
            output,
            buffer,
            event,
            stream: self.stream.clone(),
            recorded: false,
        })
    }

    fn submit(&mut self, work: &mut Self::Work) -> Result<(), SubmitError> {
        let CudaWork::Active {
            input,
            output,
            buffer,
            event,
            stream,
            recorded,
        } = work
        else {
            return Ok(());
        };
        stream
            .memcpy_htod(input, buffer)
            .map_err(|error| SubmitError::InFlightUnknown(driver_error(error)))?;
        stream
            .memcpy_dtoh(buffer, output)
            .map_err(|error| SubmitError::InFlightUnknown(driver_error(error)))?;
        event
            .record(stream)
            .map_err(|error| SubmitError::InFlightUnknown(driver_error(error)))?;
        *recorded = true;
        Ok(())
    }

    fn complete(&mut self, work: &mut Self::Work) -> Result<bool, Error> {
        match work {
            CudaWork::Empty => Ok(true),
            CudaWork::Active {
                event, recorded, ..
            } => Ok(*recorded && event.is_complete()),
        }
    }

    fn read(&mut self, work: &mut Self::Work) -> Result<Vec<u8>, Error> {
        match work {
            CudaWork::Empty => Ok(Vec::new()),
            CudaWork::Active { output, .. } => {
                let bytes = output.as_slice().map_err(driver_error)?;
                let mut result = Vec::new();
                result
                    .try_reserve_exact(bytes.len())
                    .map_err(|_| Error::ResourceLimit("CUDA readback allocation failed"))?;
                result.extend_from_slice(bytes);
                Ok(result)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::nvrtc_log_byte;

    #[test]
    fn compiler_log_reader_drains_past_the_retained_overflow_sentinel() {
        use super::{MAX_ERROR_BYTES, compiler_child::drain_capped};
        use std::io::Cursor;
        for length in [0, MAX_ERROR_BYTES, MAX_ERROR_BYTES + 1, 1024 * 1024] {
            let mut source = Cursor::new(vec![b'w'; length]);
            let log = drain_capped(&mut source, MAX_ERROR_BYTES + 1).unwrap();
            assert_eq!(log, vec![b'w'; length.min(MAX_ERROR_BYTES + 1)]);
            assert_eq!(source.position(), length as u64);
        }
    }

    #[cfg(unix)]
    #[test]
    fn compiler_log_reader_keeps_pipe_open_until_child_completion() {
        use super::{MAX_ERROR_BYTES, compiler_child::drain_capped};
        use std::process::{Command, Stdio};
        let mut child = Command::new("sh")
            .args(["-c", "head -c 1048576 /dev/zero >&2 || exit 90"])
            .stderr(Stdio::piped())
            .spawn()
            .unwrap();
        let log = drain_capped(child.stderr.take().unwrap(), MAX_ERROR_BYTES + 1).unwrap();
        assert_eq!(log.len(), MAX_ERROR_BYTES + 1);
        assert!(
            child.wait().unwrap().success(),
            "reader closed the pipe early"
        );
    }

    /// A helper that leaves a descendant on its pipes, or never exits, must
    /// neither hang the bounded compile nor leak the descendant.
    #[cfg(unix)]
    #[test]
    fn bounded_compile_lingering_or_hung_helper_ends_typed_within_the_deadline() {
        use super::{
            COMPILE_DEADLINE, Capabilities, Error, compile_cubin_bounded,
            compiler_child::tests::{descendant, fixture, lingers},
        };
        use std::time::{Duration, Instant};
        const CAPS: Capabilities = Capabilities::Cuda {
            major: 12,
            minor: 4,
            sm: 86,
        };
        const SOURCE: &str = "extern \"C\" __global__ void k() {}";
        let begun = Instant::now();
        let compiled = compile_cubin_bounded(SOURCE, &fixture("lingering.sh"), CAPS).unwrap();
        assert!(
            begun.elapsed() < Duration::from_secs(5),
            "{:?}",
            begun.elapsed()
        );
        assert_eq!(compiled.image, b"mock-cubin");
        assert!(!lingers(descendant(&compiled.log)), "{}", compiled.log);
        let begun = Instant::now();
        let result = compile_cubin_bounded(SOURCE, &fixture("lingering-log.sh"), CAPS);
        assert!(
            matches!(result, Err(Error::Limit)),
            "{:?}",
            result.as_ref().err()
        );
        assert!(
            begun.elapsed() < Duration::from_secs(5),
            "{:?}",
            begun.elapsed()
        );
        let begun = Instant::now();
        let result = compile_cubin_bounded(SOURCE, &fixture("never-exits.sh"), CAPS);
        let took = begun.elapsed();
        assert!(
            matches!(result, Err(Error::Timeout)),
            "{:?}",
            result.as_ref().err()
        );
        assert!(
            took >= COMPILE_DEADLINE && took < COMPILE_DEADLINE + Duration::from_secs(3),
            "{took:?}"
        );
    }

    #[test]
    fn nvrtc_log_bytes_preserve_bits_for_signed_and_unsigned_c_char() {
        // Includes high-bit bytes, which are negative where c_char is signed.
        for byte in 0..=u8::MAX {
            let c = std::ffi::c_char::from_ne_bytes([byte]);
            assert_eq!(nvrtc_log_byte(c), byte);
        }
    }
}
