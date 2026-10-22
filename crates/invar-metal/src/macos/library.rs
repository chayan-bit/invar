//! The one place compute pipelines come from. A production device holds only
//! compiled-ahead libraries; a developer device may also compile MSL source.
//! Every route (map, DAG, collectives, general-net, calibration) goes through
//! one bounded per-device cache keyed by the SHA-256 of the exact MSL and the
//! function name, so two different kernels never share an entry.
use std::{
    ffi::c_void,
    ptr::NonNull,
    sync::{
        Arc, Condvar, Mutex,
        atomic::{AtomicU64, Ordering},
    },
};

use block2::RcBlock;
use dispatch2::{DispatchData, DispatchRetained};
use objc2_foundation::NSError;
use objc2_metal::{
    MTLComputePipelineDescriptor, MTLComputePipelineReflection, MTLFunction, MTLPipelineOption,
};
use sha2::{Digest, Sha256};

use super::*;
use crate::{LibraryError, SOURCE_TRIPWIRE};

pub(super) use invar_device::hex;

/// Compiled-ahead libraries one device may hold.
pub const MAX_LIBRARIES: usize = 64;
/// Source-compiled libraries a developer device keeps (one per MSL source).
const MAX_SOURCE_LIBRARIES: usize = 8;
/// Pipeline states kept per device; the least recently used goes first.
pub const MAX_PIPELINES: usize = 16;
/// Default wall bound on one uncached pipeline creation. Creation is
/// asynchronous, so a kernel that would block the compiler for hours (#45:
/// 300-600 KB general-net kernels past a 9,000 s alarm) is refused instead.
/// `INVAR_METAL_PIPELINE_DEADLINE_SECS` overrides it per process.
pub const PIPELINE_DEADLINE: Duration = Duration::from_secs(900);
/// The typed refusal of a pipeline that missed its deadline. It is returned
/// before any buffer is bound or command encoded.
pub const PIPELINE_DEADLINE_REASON: &str = "metal_pipeline_deadline";

type LibraryRef = Retained<ProtocolObject<dyn MTLLibrary>>;
type Key = ([u8; 32], String);
struct PipelineEntry {
    key: Key,
    pipeline: PipelineRef,
    archive_key: [u8; 32],
    archive_bytes: usize,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CacheReport {
    /// Pipeline states currently cached.
    pub pipelines: usize,
    /// Pipeline states created since the device opened.
    pub pipeline_creations: u64,
    /// Calls to `newLibraryWithSource` since the device opened.
    pub source_compiles: u64,
    /// Compiled-ahead libraries loaded.
    pub libraries: usize,
    pub archive: super::archive::Report,
}

#[derive(Clone, Debug)]
pub struct PipelineMeasurement {
    pub total_ms: f64,
    pub library_ms: f64,
    /// Includes archive admission and publication when enabled.
    pub pipeline_ms: f64,
    /// Only the Metal pipeline creation call, excluding cache file IO.
    pub pipeline_creation_ms: f64,
    pub max_threads: usize,
    pub execution_width: usize,
    pub static_threadgroup_bytes: usize,
    pub archive: super::archive::Report,
}

#[derive(Default)]
pub(super) struct Libraries {
    runtime_source: bool,
    prebuilt: Vec<([u8; 32], LibraryRef, Option<[u8; 32]>)>,
    compiled: VecDeque<([u8; 32], LibraryRef)>,
    pipelines: VecDeque<PipelineEntry>,
    pipeline_creations: u64,
    source_compiles: u64,
    archives: Option<super::archive::Cache>,
    library_ms: f64,
    pipeline_ms: f64,
    pub(super) bound: Bound,
}

impl Libraries {
    pub(super) fn new(runtime_source: bool) -> Self {
        let deadline = std::env::var("INVAR_METAL_PIPELINE_DEADLINE_SECS")
            .ok()
            .and_then(|s| s.parse().ok())
            .map_or(PIPELINE_DEADLINE, Duration::from_secs);
        Self {
            runtime_source,
            bound: Bound {
                deadline,
                ..Bound::default()
            },
            archives: std::env::var_os("INVAR_METAL_ARCHIVE_DIR")
                .map(std::path::PathBuf::from)
                .or_else(|| {
                    std::env::var_os("HOME").map(|home| {
                        std::path::PathBuf::from(home).join(".cache/invar/metal-archives")
                    })
                })
                .map(super::archive::Cache::new),
            ..Self::default()
        }
    }
}

fn msl_sha256(source: &str) -> [u8; 32] {
    Sha256::digest(source.as_bytes()).into()
}

impl MetalDevice {
    /// A developer device that may compile MSL source at run time. Debugging
    /// only: production opens with `open` and loads compiled-ahead libraries.
    pub fn open_runtime_source() -> Result<Self, Error> {
        let mut device = Self::open()?;
        device.libraries = Libraries::new(true);
        Ok(device)
    }

    /// Bound every later uncached pipeline creation on this device.
    pub fn set_pipeline_deadline(&mut self, deadline: Duration) {
        self.libraries.bound.deadline = deadline;
    }

    /// Pipeline creations that completed after their deadline. Each was
    /// dropped by its completion handler: never cached, bound or dispatched.
    pub fn late_pipeline_completions(&self) -> u64 {
        self.libraries.bound.late.load(Ordering::Acquire)
    }

    pub fn cache_report(&self) -> CacheReport {
        CacheReport {
            pipelines: self.libraries.pipelines.len(),
            pipeline_creations: self.libraries.pipeline_creations,
            source_compiles: self.libraries.source_compiles,
            libraries: self.libraries.prebuilt.len(),
            archive: self
                .libraries
                .archives
                .as_ref()
                .map(|cache| cache.report)
                .unwrap_or_default(),
        }
    }

    /// Load one compiled-ahead library from validated bytes. Its functions
    /// must be exactly `functions`; the bytes are copied, never re-read.
    pub fn load_library(
        &mut self,
        msl_sha256: [u8; 32],
        metallib: &[u8],
        functions: &[&str],
    ) -> Result<(), Error> {
        self.load_library_identified(msl_sha256, metallib, functions, None)
    }

    /// Load AOT bytes with the validated compiler/options/program identity from
    /// the container header. It participates in every persistent archive key.
    pub fn load_library_with_identity(
        &mut self,
        msl_sha256: [u8; 32],
        metallib: &[u8],
        functions: &[&str],
        identity: &str,
    ) -> Result<(), Error> {
        if identity.is_empty() || identity.len() > 64 * 1024 {
            return Err(Error::Limit);
        }
        let mut hash = Sha256::new();
        hash.update((identity.len() as u64).to_le_bytes());
        hash.update(identity.as_bytes());
        hash.update(Sha256::digest(metallib));
        self.load_library_identified(
            msl_sha256,
            metallib,
            functions,
            Some(hash.finalize().into()),
        )
    }

    fn load_library_identified(
        &mut self,
        msl_sha256: [u8; 32],
        metallib: &[u8],
        functions: &[&str],
        compiler: Option<[u8; 32]>,
    ) -> Result<(), Error> {
        if self
            .libraries
            .prebuilt
            .iter()
            .any(|(d, _, _)| *d == msl_sha256)
        {
            return Err(Error::Library(LibraryError::Duplicate));
        }
        if self.libraries.prebuilt.len() == MAX_LIBRARIES || metallib.is_empty() {
            return Err(Error::Limit);
        }
        let data = dispatch_data(metallib)?;
        let library = self
            .device
            .newLibraryWithData_error(&data)
            .map_err(|e| Error::Library(LibraryError::Rejected(e.to_string())))?;
        let mut found = library
            .functionNames()
            .iter()
            .map(|name| name.to_string())
            .collect::<Vec<_>>();
        let mut wanted = functions.iter().map(|f| f.to_string()).collect::<Vec<_>>();
        found.sort();
        wanted.sort();
        if found != wanted {
            return Err(Error::Library(LibraryError::EntryPoint(format!(
                "library exports {found:?}, expected {wanted:?}"
            ))));
        }
        self.libraries
            .prebuilt
            .push((msl_sha256, library, compiler));
        // A developer may register AOT after compiling this source. Its
        // compiler/library identity must replace the old pipeline identity.
        self.libraries
            .pipelines
            .retain(|entry| entry.key.0 != msl_sha256);
        Ok(())
    }

    /// The pipeline for function `name` of the kernel whose MSL is `source`,
    /// and whether it came from the cache.
    pub(super) fn compile(
        &mut self,
        source: &str,
        name: &str,
    ) -> Result<(PipelineRef, bool), Error> {
        let key = (msl_sha256(source), name.to_owned());
        let cache = &mut self.libraries.pipelines;
        if let Some(at) = cache.iter().position(|entry| entry.key == key) {
            let entry = cache.remove(at).expect("position is in bounds");
            let pipeline = entry.pipeline.clone();
            let archive_key = entry.archive_key;
            let archive_bytes = entry.archive_bytes;
            cache.push_back(entry);
            if let Some(cache) = &mut self.libraries.archives {
                cache.report.last = super::archive::State::MemoryHit;
                cache.report.key = archive_key;
                cache.report.bytes = archive_bytes;
                cache.report.write_refusal = None;
                cache.report.pipeline_creation_ns = 0;
            }
            return Ok((pipeline, true));
        }
        let begun = Instant::now();
        let (library, compiler) = self.library(source, &key.0)?;
        self.libraries.library_ms = begun.elapsed().as_secs_f64() * 1000.0;
        let begun = Instant::now();
        let function = library
            .newFunctionWithName(&NSString::from_str(name))
            .ok_or_else(|| Error::Library(LibraryError::EntryPoint(name.into())))?;
        let pipeline =
            if let (Some(cache), Some(compiler)) = (&mut self.libraries.archives, compiler) {
                let bound = &self.libraries.bound;
                cache.pipeline(&self.device, &function, &key.0, &compiler, name, bound)?
            } else {
                if let Some(cache) = &mut self.libraries.archives {
                    cache.report.last = super::archive::State::Disabled;
                    cache.report.key = [0; 32];
                    cache.report.bytes = 0;
                    cache.report.write_refusal = None;
                }
                let begun = Instant::now();
                let pipeline = self.pipeline(&function)?;
                if let Some(cache) = &mut self.libraries.archives {
                    cache.report.pipeline_creation_ns =
                        u64::try_from(begun.elapsed().as_nanos()).unwrap_or(u64::MAX);
                }
                pipeline
            };
        self.libraries.pipeline_ms = begun.elapsed().as_secs_f64() * 1000.0;
        self.libraries.pipeline_creations += 1;
        let archive = self.cache_report().archive;
        let cache = &mut self.libraries.pipelines;
        if cache.len() == MAX_PIPELINES {
            cache.pop_front();
        }
        cache.push_back(PipelineEntry {
            key,
            pipeline: pipeline.clone(),
            archive_key: archive.key,
            archive_bytes: archive.bytes,
        });
        Ok((pipeline, false))
    }

    /// Compile only, without allocating buffers or dispatching a kernel. The
    /// serialized archive byte count is a portable on-disk pipeline-size proxy.
    pub fn measure_pipeline(
        &mut self,
        source: &str,
        name: &str,
    ) -> Result<PipelineMeasurement, Error> {
        if source.len() > invar_device::MAX_GNET_SOURCE_BYTES
            || name.is_empty()
            || name.len() > 128
            || !name.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'_')
        {
            return Err(Error::Limit);
        }
        let begun = Instant::now();
        let (pipeline, memory_hit) = self.compile(source, name)?;
        Ok(PipelineMeasurement {
            total_ms: begun.elapsed().as_secs_f64() * 1000.0,
            library_ms: if memory_hit {
                0.0
            } else {
                self.libraries.library_ms
            },
            pipeline_ms: if memory_hit {
                0.0
            } else {
                self.libraries.pipeline_ms
            },
            max_threads: pipeline.maxTotalThreadsPerThreadgroup(),
            execution_width: pipeline.threadExecutionWidth(),
            static_threadgroup_bytes: pipeline.staticThreadgroupMemoryLength(),
            pipeline_creation_ms: self.cache_report().archive.pipeline_creation_ns as f64
                / 1_000_000.0,
            archive: self.cache_report().archive,
        })
    }

    fn pipeline(&self, function: &ProtocolObject<dyn MTLFunction>) -> Result<PipelineRef, Error> {
        self.libraries.bound.function(&self.device, function)
    }

    /// The library built from exactly this MSL: compiled-ahead first; source
    /// compilation only on a developer device, once per source.
    fn library(
        &mut self,
        source: &str,
        digest: &[u8; 32],
    ) -> Result<(LibraryRef, Option<[u8; 32]>), Error> {
        let libraries = &mut self.libraries;
        if let Some((_, library, compiler)) =
            libraries.prebuilt.iter().find(|(d, _, _)| d == digest)
        {
            return Ok((library.clone(), *compiler));
        }
        if !libraries.runtime_source {
            return Err(Error::Library(LibraryError::Missing {
                msl_sha256: hex(digest),
            }));
        }
        let runtime_compiler = Some(
            Sha256::digest(format!(
                "system-Metal-source:{}",
                objc2_foundation::NSProcessInfo::processInfo().operatingSystemVersionString()
            ))
            .into(),
        );
        if let Some((_, library)) = libraries.compiled.iter().find(|(d, _)| d == digest) {
            return Ok((library.clone(), runtime_compiler));
        }
        let library = self.compile_library_source(source)?;
        let compiled = &mut self.libraries.compiled;
        if compiled.len() == MAX_SOURCE_LIBRARIES {
            compiled.pop_front();
        }
        compiled.push_back((*digest, library.clone()));
        Ok((library, runtime_compiler))
    }

    /// The only `newLibraryWithSource` call: pinned MSL 3.1, fast math off.
    fn compile_library_source(&mut self, source: &str) -> Result<LibraryRef, Error> {
        if std::env::var_os(SOURCE_TRIPWIRE).is_some() {
            eprintln!("{SOURCE_TRIPWIRE}: runtime MSL source compilation was reached");
            std::process::abort();
        }
        self.libraries.source_compiles += 1;
        let options = MTLCompileOptions::new();
        options.setLanguageVersion(MTLLanguageVersion::Version3_1);
        #[allow(deprecated)]
        options.setFastMathEnabled(false);
        self.device
            .newLibraryWithSource_options_error(&NSString::from_str(source), Some(&options))
            .map_err(|e| Error::Toolchain(format!("Metal source compilation failed: {e}")))
    }
}

/// The deadline and late-completion count shared by one device's
/// asynchronous pipeline creations.
pub(super) struct Bound {
    deadline: Duration,
    late: Arc<AtomicU64>,
}

impl Default for Bound {
    fn default() -> Self {
        Self {
            deadline: PIPELINE_DEADLINE,
            late: Arc::default(),
        }
    }
}

type Completion = Result<PipelineRef, String>;

enum State {
    Waiting,
    Done(Completion),
    Abandoned,
}

/// One creation's rendezvous between its waiter and its completion handler.
/// The handler owns an `Arc`, so it stays valid however late Metal calls it.
struct Slot {
    state: Mutex<State>,
    ready: Condvar,
    late: Arc<AtomicU64>,
    /// The absolute deadline, fixed before the creation is issued; `None`
    /// only when it lies beyond what `Instant` can represent.
    until: Option<Instant>,
}

impl Slot {
    /// Runs on a Metal queue. A completion for an abandoned creation is
    /// counted and dropped here, so it is never published.
    fn deliver(
        &self,
        pipeline: *mut ProtocolObject<dyn MTLComputePipelineState>,
        error: *mut NSError,
    ) {
        // SAFETY: Metal passes each argument at +0 or null, valid for the
        // duration of the handler call; retaining takes our own reference.
        let result = match unsafe { Retained::retain(pipeline) } {
            Some(pipeline) => Ok(pipeline),
            None => Err(unsafe { Retained::retain(error) }
                .map_or_else(|| "no pipeline and no error".into(), |e| e.to_string())),
        };
        let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        match *state {
            State::Waiting => {
                *state = State::Done(result);
                self.ready.notify_all();
            }
            State::Abandoned => {
                self.late.fetch_add(1, Ordering::AcqRel);
            }
            State::Done(_) => {}
        }
    }

    /// A completion is accepted only if the waiter observes it, under the
    /// lock, before the absolute deadline. One that won the lock race after
    /// the deadline expired is counted late and dropped like any other.
    fn wait(&self) -> Result<PipelineRef, Error> {
        let remaining = self.until.map_or(Duration::MAX, |until| {
            until.saturating_duration_since(Instant::now())
        });
        let state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        let (mut state, _) = self
            .ready
            .wait_timeout_while(state, remaining, |s| matches!(s, State::Waiting))
            .unwrap_or_else(|e| e.into_inner());
        let expired = self.until.is_some_and(|until| Instant::now() >= until);
        match std::mem::replace(&mut *state, State::Abandoned) {
            State::Done(_) if expired => {
                self.late.fetch_add(1, Ordering::AcqRel);
                Err(Error::ResourceLimit(PIPELINE_DEADLINE_REASON))
            }
            State::Done(Ok(pipeline)) => Ok(pipeline),
            State::Done(Err(e)) => Err(Error::Toolchain(format!(
                "Metal pipeline creation failed: {e}"
            ))),
            State::Waiting | State::Abandoned => {
                Err(Error::ResourceLimit(PIPELINE_DEADLINE_REASON))
            }
        }
    }
}

impl Bound {
    fn slot(&self) -> Arc<Slot> {
        Arc::new(Slot {
            state: Mutex::new(State::Waiting),
            ready: Condvar::new(),
            late: self.late.clone(),
            until: Instant::now().checked_add(self.deadline),
        })
    }

    /// `newComputePipelineStateWithFunction:completionHandler:` under the deadline.
    fn function(
        &self,
        device: &ProtocolObject<dyn MTLDevice>,
        function: &ProtocolObject<dyn MTLFunction>,
    ) -> Result<PipelineRef, Error> {
        let slot = self.slot();
        let handler = RcBlock::new({
            let slot = slot.clone();
            move |pipeline: *mut ProtocolObject<dyn MTLComputePipelineState>,
                  error: *mut NSError| slot.deliver(pipeline, error)
        });
        // SAFETY: `handler` is a live block of the declared signature; Metal
        // copies it before returning and calls the copy exactly once.
        unsafe {
            device.newComputePipelineStateWithFunction_completionHandler(
                function,
                RcBlock::as_ptr(&handler),
            )
        };
        slot.wait()
    }

    /// `newComputePipelineStateWithDescriptor:options:completionHandler:`
    /// under the deadline; the reflection object is not requested.
    pub(super) fn descriptor(
        &self,
        device: &ProtocolObject<dyn MTLDevice>,
        descriptor: &MTLComputePipelineDescriptor,
        options: MTLPipelineOption,
    ) -> Result<PipelineRef, Error> {
        let slot = self.slot();
        let handler = RcBlock::new({
            let slot = slot.clone();
            move |pipeline: *mut ProtocolObject<dyn MTLComputePipelineState>,
                  _: *mut MTLComputePipelineReflection,
                  error: *mut NSError| slot.deliver(pipeline, error)
        });
        // SAFETY: as in `function`.
        unsafe {
            device.newComputePipelineStateWithDescriptor_options_completionHandler(
                descriptor,
                options,
                RcBlock::as_ptr(&handler),
            )
        };
        slot.wait()
    }
}

/// A dispatch data object holding a copy of `bytes`.
fn dispatch_data(bytes: &[u8]) -> Result<DispatchRetained<DispatchData>, Error> {
    unsafe extern "C" {
        // libdispatch, part of libSystem. A null destructor is
        // DISPATCH_DATA_DESTRUCTOR_DEFAULT: the buffer is copied immediately.
        fn dispatch_data_create(
            buffer: *const c_void,
            size: usize,
            queue: *const c_void,
            destructor: *const c_void,
        ) -> *mut DispatchData;
    }
    // SAFETY: `bytes` is valid for `bytes.len()` reads for this call, and the
    // default destructor copies them before returning, so no borrow escapes.
    let raw = unsafe {
        dispatch_data_create(
            bytes.as_ptr().cast(),
            bytes.len(),
            std::ptr::null(),
            std::ptr::null(),
        )
    };
    let raw = NonNull::new(raw).ok_or(Error::OutOfMemory)?;
    // SAFETY: dispatch_data_create returns a +1 reference we now own.
    Ok(unsafe { DispatchRetained::from_raw(raw) })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A kernel no shader or archive cache has seen: its constant is new to
    /// this call, so the asynchronous creation cannot complete synchronously.
    fn fresh_kernel() -> String {
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .subsec_nanos()
            ^ std::process::id();
        format!(
            "#include <metal_stdlib>\nusing namespace metal;\n\
             kernel void deadline_probe(device uint *out [[buffer(0)]], uint i [[thread_position_in_grid]]) \
             {{ out[i] = i * {nonce}u + {nonce}u; }}\n"
        )
    }

    fn wait_for_late(device: &MetalDevice, count: u64) {
        let begun = Instant::now();
        while device.late_pipeline_completions() < count {
            assert!(
                begun.elapsed() < Duration::from_secs(120),
                "no late completion"
            );
            std::thread::sleep(Duration::from_millis(10));
        }
        assert_eq!(device.late_pipeline_completions(), count);
    }

    /// Both creation routes: the plain function route, and the archive
    /// rebuild route (a fresh private archive directory always misses).
    fn missed_deadline_then_recovery(archive: Option<std::path::PathBuf>) {
        let mut device = MetalDevice::open_runtime_source().expect("Metal device");
        device.libraries.archives = archive.map(super::archive::Cache::new);
        let source = fresh_kernel();
        device.set_pipeline_deadline(Duration::ZERO);
        let refused = device.compile(&source, "deadline_probe");
        assert!(
            matches!(refused, Err(Error::ResourceLimit(PIPELINE_DEADLINE_REASON))),
            "{:?}",
            refused.map(|_| ())
        );
        assert_eq!(device.cache_report().pipelines, 0);
        assert_eq!(device.cache_report().pipeline_creations, 0);
        // The late pipeline reaches only its handler, which drops it.
        wait_for_late(&device, 1);
        assert_eq!(device.cache_report().pipelines, 0);
        device.set_pipeline_deadline(PIPELINE_DEADLINE);
        let (_, hit) = device.compile(&source, "deadline_probe").unwrap();
        assert!(!hit);
        assert_eq!(device.cache_report().pipelines, 1);
        assert_eq!(device.late_pipeline_completions(), 1);
    }

    #[test]
    fn completion_that_wins_the_lock_after_the_deadline_is_refused_and_counted_late() {
        // No device: a null pipeline and error deliver a failed creation.
        let none = || (std::ptr::null_mut(), std::ptr::null_mut());
        let expired = Bound {
            deadline: Duration::ZERO,
            late: Arc::default(),
        };
        let slot = expired.slot();
        let (pipeline, error) = none();
        slot.deliver(pipeline, error);
        assert!(matches!(
            slot.wait(),
            Err(Error::ResourceLimit(PIPELINE_DEADLINE_REASON))
        ));
        assert_eq!(expired.late.load(Ordering::Acquire), 1);
        // The same delivery inside the deadline is published as it came.
        let open = Bound::default();
        let slot = open.slot();
        let (pipeline, error) = none();
        slot.deliver(pipeline, error);
        assert!(matches!(slot.wait(), Err(Error::Toolchain(_))));
        assert_eq!(open.late.load(Ordering::Acquire), 0);
    }

    #[test]
    fn missed_pipeline_deadline_is_a_typed_refusal_and_the_late_pipeline_is_dropped() {
        missed_deadline_then_recovery(None);
    }

    #[test]
    fn missed_archive_rebuild_deadline_is_a_typed_refusal_and_the_late_pipeline_is_dropped() {
        let root = std::env::temp_dir().join(format!(
            "invar-metal-deadline-{}-{}",
            std::process::id(),
            fresh_kernel().len()
        ));
        missed_deadline_then_recovery(Some(root.clone()));
        let _ = std::fs::remove_dir_all(root);
    }
}
