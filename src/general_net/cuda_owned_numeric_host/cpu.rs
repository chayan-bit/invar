//! ABI12 stage 2c S2: CPU group model as a SUBPROCESS backend. The unchanged
//! emitted ABI12 CUDA text, the stage-1 shim (with thread-local indices for
//! real threads) and a fixed driver are compiled into an executable with a
//! fixed argv; each epoch is one bounded child process. No library is loaded
//! into this process and there is no `unsafe` here. This is CPU modelling,
//! never device execution or device qualification.
//!
//! Child protocol: the scratch directory (0700, unique) holds `epoch.in`;
//! the child (argv = [exe], cwd = scratch, cleared environment) applies
//! `RLIMIT_AS`, runs every lane once with 32 worker threads of fixed stack
//! size and a dispatcher synchronized by real `std::barrier`s, and writes
//! `epoch.out` (renamed from a temporary file). A clean exit 0 is Complete;
//! a reaped non-zero exit or a reaped kill is FailedQuiescent with no
//! readback; only a child that cannot be confirmed reaped is Unknown.
use super::{
    Backend, BackendKind, Buffers, Error, ReceiptSeal, Session, Slots, State, root, sealed,
};
use crate::general_net::compiled::{self, FenceObservation, Submission};
use crate::general_net::provenance::owned_numeric_execution as receipt;
use toolchain::{
    COMPILE_ADDRESS_SPACE_BYTES, COMPILE_DISK_BYTES, COMPILE_FILE_BYTES, COMPILE_MAX_PROCESSES,
    Guard, Program,
};

mod toolchain;
use sha2::{Digest, Sha256};
use std::{
    cell::RefCell,
    fs,
    io::{Read, Write},
    os::unix::{fs::DirBuilderExt, process::CommandExt},
    path::{Path, PathBuf},
    process::{Child, Command, Stdio},
    sync::atomic::{AtomicU64, Ordering},
    time::{Duration, Instant},
};

/// Fixed product compiler flags (part of the model identity). Fully static,
/// so the executable's bytes bind its runtime code, and no linker plugin (F1).
pub(crate) const FLAGS_V2: &[&str] = &[
    "-std=c++20",
    "-O2",
    "-pthread",
    "-fno-strict-aliasing",
    "-fno-fast-math",
    "-ffp-contract=off",
    "-fvisibility=hidden",
    "-static",
    "-fno-use-linker-plugin",
];
/// Test-only sanitizer profiles, recorded separately (their flags are part of
/// the model identity, so a sanitizer build never shares an identity with
/// the product profile).
#[cfg(all(test, target_os = "linux"))]
pub(crate) const UBSAN_FLAGS_V1: &[&str] = &[
    "-std=c++20",
    "-O2",
    "-pthread",
    "-fno-strict-aliasing",
    "-fno-fast-math",
    "-ffp-contract=off",
    "-fvisibility=hidden",
    "-fsanitize=undefined",
    "-fno-sanitize-recover=all",
];
#[cfg(all(test, target_os = "linux"))]
pub(crate) const ASAN_FLAGS_V1: &[&str] = &[
    "-std=c++20",
    "-O2",
    "-pthread",
    "-fno-strict-aliasing",
    "-fno-fast-math",
    "-ffp-contract=off",
    "-fvisibility=hidden",
    "-fsanitize=address",
];
/// Worker thread stack, set explicitly with `pthread_attr_setstacksize`.
pub(crate) const STACK_BYTES: usize = 1024 * 1024;
/// Address space for code, loader, libc/libstdc++ and one malloc arena,
/// beyond the buffers and the 33 thread stacks.
pub(crate) const CHILD_OVERHEAD_BYTES: usize = 128 * 1024 * 1024;
/// Parent-side read/write chunk.
const IO_CHUNK: usize = 64 * 1024;
const HEADER_WORDS: usize = 9;
const MAGIC: u64 = 0x3231_454e_4756_4e49; // ASCII INVGNE12, little-endian
const COMPILE_DEADLINE: Duration = Duration::from_secs(120);
const PROBE_DEADLINE: Duration = Duration::from_secs(30);
pub(crate) const MAX_EPOCH_DEADLINE: Duration = Duration::from_secs(120);
/// Parent-side memory beyond the child cap (F4): the epoch digest list
/// (at most 1024 x 32 B) and a bound on the model record: 6 toolchain
/// programs plus the executable's runtime list, each with at most 64
/// libraries and paths of at most 4096 B (hard caps in `toolchain`).
pub(crate) const EPOCH_DIGEST_BYTES: usize = 1024 * 32;
pub(crate) const RECORD_BYTES: usize = 4 * 1024 * 1024;
/// Steady-state per-session disk cap (F4/R2): the post-compile scratch check
/// plus three device-state files (`epoch.in`, `epoch.tmp`/`epoch.out`,
/// `epoch.stderr`), each hard-capped by the child's `RLIMIT_FSIZE`. Not a
/// whole-session peak bound: compile-stage peak disk is bounded only by the
/// hard per-file cap times the files alive together (modelled).
pub(crate) const SESSION_DISK_BYTES: u64 = 512 * 1024 * 1024;
const KERNEL: &str = "invar_gnet_owned_numeric_resume_v1";

/// The stage-1 shim, with thread-local indices so every worker thread owns
/// its `threadIdx`/`blockIdx` like a CUDA thread.
pub(crate) const SHIM_V1: &str = r#"#define __device__
#define __global__
struct Dim { unsigned x = 1, y = 1, z = 1; };
Dim gridDim, blockDim{32, 1, 1};
thread_local Dim threadIdx{0, 0, 0}, blockIdx{0, 0, 0};
"#;

/// Fixed driver template; `@...@` fields are filled with numeric constants.
pub(crate) const DRIVER_V2: &str = r#"
#include <atomic>
#include <barrier>
#include <cstdint>
#include <cstdio>
#include <vector>
#include <pthread.h>
#include <sys/resource.h>
static const unsigned long long MAGIC = @MAGIC@ULL;
static const unsigned long long WORDS[7] = {@WORDS@};
static const unsigned LANES = @LANES@, BLOCKS = LANES / 32;
static const unsigned long long AS_CAP = @AS_CAP@ULL;
static const unsigned long long FSIZE = @FSIZE@ULL;
static const size_t STACK = @STACK@;
static std::vector<unsigned long long> buf[7];
static std::barrier<>* start_gate;
static std::barrier<>* finish_gate;
static std::atomic<unsigned> current{0};
static std::atomic<bool> stop{false};
static void* worker(void* arg) {
    threadIdx.x = (unsigned)(uintptr_t)arg;
    for (;;) {
        start_gate->arrive_and_wait();
        if (stop.load(std::memory_order_acquire)) return nullptr;
        blockIdx.x = current.load(std::memory_order_acquire);
        dev::@KERNEL@((const dev::gnet_w*)buf[0].data(), (dev::gnet_w*)buf[1].data(),
            (dev::gnet_u*)buf[2].data(), (const dev::gnet_u*)buf[3].data(),
            (dev::gnet_w*)buf[4].data(), (dev::gnet_u*)buf[5].data(),
            (dev::GnetTask*)buf[6].data());
        finish_gate->arrive_and_wait();
    }
}
int main(int argc, char**) {
    if (argc != 1) return 64;
    struct rlimit cap = {AS_CAP, AS_CAP};
    if (setrlimit(RLIMIT_AS, &cap) != 0) return 65;
    struct rlimit fsize = {FSIZE, FSIZE};
    if (setrlimit(RLIMIT_FSIZE, &fsize) != 0) return 65;
    FILE* in = std::fopen("epoch.in", "rb");
    if (!in) return 66;
    unsigned long long head[9];
    if (std::fread(head, 8, 9, in) != 9 || head[0] != MAGIC) return 67;
    for (int i = 0; i < 7; ++i) if (head[2 + i] != WORDS[i]) return 67;
    for (int i = 0; i < 7; ++i) {
        buf[i].resize(WORDS[i]);
        if (std::fread(buf[i].data(), 8, WORDS[i], in) != WORDS[i]) return 68;
    }
    if (std::fgetc(in) != EOF) return 68;
    std::fclose(in);
    gridDim.x = BLOCKS;
    std::barrier<> start(33), finish(33);
    start_gate = &start;
    finish_gate = &finish;
    pthread_attr_t attr;
    if (pthread_attr_init(&attr) != 0 || pthread_attr_setstacksize(&attr, STACK) != 0) return 69;
    pthread_t threads[32];
    for (unsigned t = 0; t < 32; ++t)
        if (pthread_create(&threads[t], &attr, worker, (void*)(uintptr_t)t) != 0) return 69;
    for (unsigned b = 0; b < BLOCKS; ++b) {
        current.store(b, std::memory_order_release);
        start.arrive_and_wait();
        finish.arrive_and_wait();
    }
    stop.store(true, std::memory_order_release);
    start.arrive_and_wait();
    for (unsigned t = 0; t < 32; ++t) pthread_join(threads[t], nullptr);
    head[1] += 1;
    FILE* out = std::fopen("epoch.tmp", "wb");
    if (!out) return 70;
    if (std::fwrite(head, 8, 9, out) != 9) return 70;
    for (int i = 0; i < 7; ++i)
        if (std::fwrite(buf[i].data(), 8, WORDS[i], out) != WORDS[i]) return 70;
    if (std::fclose(out) != 0 || std::rename("epoch.tmp", "epoch.out") != 0) return 70;
    return 0;
}
"#;

fn sha256(bytes: &[u8]) -> [u8; 32] {
    Sha256::digest(bytes).into()
}

/// Bytes of host memory the model may use for `device_bytes`: the child's
/// address-space cap (buffers, 33 stacks, fixed overhead) plus parent I/O.
pub(crate) fn child_cap(device_bytes: usize) -> Option<usize> {
    device_bytes
        .checked_add(33usize.checked_mul(STACK_BYTES)?)?
        .checked_add(CHILD_OVERHEAD_BYTES)
}

/// Typed model identity (D6). Every digest is of bytes this backend wrote,
/// ran or produced.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct ModelRecord {
    pub compiler_path: PathBuf,
    pub compiler_sha256: [u8; 32],
    pub compiler_version_sha256: [u8; 32],
    pub host_triple: String,
    pub flags: &'static [&'static str],
    pub shim_sha256: [u8; 32],
    pub driver_sha256: [u8; 32],
    pub model_source_sha256: [u8; 32],
    pub executable_sha256: [u8; 32],
    pub kernel_sha256: [u8; 32],
    pub stack_bytes: usize,
    pub child_address_space_bytes: usize,
    pub lanes: usize,
    pub words: [usize; 7],
    /// Shell, driver, cc1plus, collect2, as and ld, each with the shared
    /// libraries its interpreter maps (F1).
    pub toolchain: Vec<Program>,
    /// True when the executable has no ELF interpreter (product profile).
    pub runtime_static: bool,
    /// The executable's shared libraries (empty when static).
    pub runtime_libraries: Vec<(PathBuf, [u8; 32])>,
    /// Compile-stage bounds by class (F2/R1/R2).
    pub compile_stage: receipt::CompileStage,
    /// Steady-state session disk after compiling (not a peak bound).
    pub steady_state_disk_bytes: u64,
    /// Every recorded toolchain file was re-hashed after the compile (R3).
    pub toolchain_reverified: bool,
}

/// Unique 0700 scratch directory, removed when the work is dropped.
struct Scratch(PathBuf);
impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}
fn scratch() -> Result<Scratch, String> {
    static NEXT: AtomicU64 = AtomicU64::new(1);
    for _ in 0..128 {
        let nonce = NEXT.fetch_add(1, Ordering::Relaxed);
        let path =
            std::env::temp_dir().join(format!("invar-gnet12-model-{}-{nonce}", std::process::id()));
        match fs::DirBuilder::new().mode(0o700).create(&path) {
            Ok(()) => return Ok(Scratch(path)),
            Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(e) => return Err(e.to_string()),
        }
    }
    Err("scratch directory collisions".into())
}

/// The executable and its device state files for one session.
pub(crate) struct Work {
    dir: Scratch,
    exe: PathBuf,
    record: ModelRecord,
    device_bytes: usize,
    /// Completed epochs; a completed `epoch.out` must carry this number.
    completed: u64,
    child: Option<(Child, Instant)>,
    deadline: Duration,
    /// sha256 of the initial device state file.
    initial_sha256: [u8; 32],
    /// sha256 of every completed epoch's output file, in order (capacity
    /// reserved for the fence budget; never grows past it).
    epochs: RefCell<Vec<[u8; 32]>>,
}
impl Work {
    pub(crate) fn record(&self) -> &ModelRecord {
        &self.record
    }
}

/// Subprocess CPU group model of the emitted ABI12 CUDA text.
pub(crate) struct CpuGroupModel {
    compiler: PathBuf,
    deadline: Duration,
    poisoned: bool,
    /// Test-only: run this program instead of the model executable, to
    /// exercise timeout, failure and kill handling.
    #[cfg(test)]
    pub(super) substitute: Option<(PathBuf, Vec<String>)>,
    /// Test-only: report the child as unconfirmable.
    #[cfg(test)]
    pub(super) unconfirmable: bool,
    /// Test-only: (flags, require a static executable) instead of the
    /// product profile (`FLAGS_V2`, static required).
    #[cfg(test)]
    pub(super) profile: Option<(&'static [&'static str], bool)>,
    /// Test-only: a smaller compile-stage address-space cap.
    #[cfg(test)]
    pub(super) compile_address_space: Option<u64>,
    /// Test-only: a smaller session disk cap.
    #[cfg(test)]
    pub(super) disk_cap: Option<u64>,
    /// Test-only: the compile `PATH` instead of this process's.
    #[cfg(test)]
    pub(super) path_env: Option<std::ffi::OsString>,
    /// Test-only: run after the compile, before the toolchain is re-verified.
    #[cfg(test)]
    pub(super) after_compile: Option<Box<dyn Fn()>>,
}
impl CpuGroupModel {
    pub(crate) fn new(compiler: &Path, deadline: Duration) -> Result<Self, String> {
        if deadline.is_zero() || deadline > MAX_EPOCH_DEADLINE {
            return Err("epoch deadline".into());
        }
        Ok(Self {
            compiler: compiler.to_path_buf(),
            deadline,
            poisoned: false,
            #[cfg(test)]
            substitute: None,
            #[cfg(test)]
            unconfirmable: false,
            #[cfg(test)]
            profile: None,
            #[cfg(test)]
            compile_address_space: None,
            #[cfg(test)]
            disk_cap: None,
            #[cfg(test)]
            path_env: None,
            #[cfg(test)]
            after_compile: None,
        })
    }
    pub(crate) fn poisoned(&self) -> bool {
        self.poisoned
    }

    /// Write the initial device state and return the sha256 of its bytes.
    fn write_image(path: &Path, image: &Buffers, words: [usize; 7]) -> Result<[u8; 32], String> {
        let mut file = std::io::BufWriter::with_capacity(
            IO_CHUNK,
            fs::File::create(path).map_err(|e| e.to_string())?,
        );
        let mut head = [0u64; HEADER_WORDS];
        head[0] = MAGIC;
        for (slot, w) in head[2..].iter_mut().zip(words) {
            *slot = w as u64;
        }
        let mut hash = Sha256::new();
        let mut put = |w: u64| {
            hash.update(w.to_le_bytes());
            file.write_all(&w.to_le_bytes()).map_err(|e| e.to_string())
        };
        for w in head {
            put(w)?;
        }
        for w in &image.input {
            put(*w as u64)?;
        }
        for w in &image.output {
            put(*w as u64)?;
        }
        for w in &image.trace {
            put(*w)?;
        }
        for w in &image.config {
            put(*w)?;
        }
        for w in &image.heap {
            put(*w as u64)?;
        }
        for w in &image.cleanup {
            put(*w)?;
        }
        hash.update(&image.tasks);
        file.write_all(&image.tasks).map_err(|e| e.to_string())?;
        file.flush().map_err(|e| e.to_string())?;
        Ok(hash.finalize().into())
    }
}

impl sealed::Sealed for CpuGroupModel {}
impl Backend for CpuGroupModel {
    type Work = Work;
    const KIND: BackendKind = BackendKind::CpuGroupModel;
    fn host_bytes(device_bytes: usize) -> Option<usize> {
        child_cap(device_bytes)?
            .checked_add(2 * IO_CHUNK)?
            .checked_add(EPOCH_DIGEST_BYTES)?
            .checked_add(RECORD_BYTES)
    }
    fn prepare(&mut self, artifact: &compiled::Artifact, image: &Buffers) -> Result<Work, String> {
        if self.poisoned {
            return Err("poisoned CPU group model".into());
        }
        if artifact.abi_version() != 12 || artifact.target() != compiled::Target::Cuda {
            return Err("CPU group model requires an ABI12 CUDA artifact".into());
        }
        let text = artifact.source();
        if text.contains("#include") || text.matches("__global__").count() != 1 {
            return Err("ABI12 text profile".into());
        }
        // F5: lanes and every buffer length from the plan facts.
        let f = artifact
            .owned_numeric_cuda_facts()
            .map_err(|e| format!("{e:?}"))?;
        let red = f.red_zone_words;
        let layout = || "workset layout".to_owned();
        let lanes = image
            .output
            .len()
            .checked_sub(red)
            .filter(|n| n % f.output_words == 0)
            .map(|n| n / f.output_words)
            .filter(|l| *l > 0 && l % f.lane_block == 0 && *l <= f.max_roots)
            .ok_or_else(layout)?;
        let expect = |per_lane: usize| per_lane.checked_mul(lanes).and_then(|n| n.checked_add(red));
        let trace = image.trace.len().checked_sub(red).ok_or_else(layout)?;
        if Some(image.input.len()) != expect(f.input_words)
            || trace % (lanes * f.event_words) != 0
            || image.config.len() != f.config_words + red
            || Some(image.heap.len()) != expect(f.heap_words)
            || Some(image.cleanup.len()) != expect(f.cleanup_words)
            || Some(image.tasks.len()) != expect(f.task_words).and_then(|w| w.checked_mul(8))
        {
            return Err(layout());
        }
        let words = [
            image.input.len(),
            image.output.len(),
            image.trace.len(),
            image.config.len(),
            image.heap.len(),
            image.cleanup.len(),
            image.tasks.len() / 8,
        ];
        let device_bytes = words
            .iter()
            .try_fold(0usize, |n, w| n.checked_add(w.checked_mul(8)?))
            .ok_or_else(layout)?;
        let cap = child_cap(device_bytes).ok_or("child cap")?;
        let state_bytes = (device_bytes + HEADER_WORDS * 8) as u64;
        // F4: per-session disk, refused before any compiler call.
        let steady_state_disk_bytes = state_bytes
            .checked_mul(3)
            .and_then(|n| n.checked_add(COMPILE_DISK_BYTES))
            .ok_or("disk bound")?;
        #[cfg(test)]
        let disk_cap = self.disk_cap.unwrap_or(SESSION_DISK_BYTES);
        #[cfg(not(test))]
        let disk_cap = SESSION_DISK_BYTES;
        if steady_state_disk_bytes > disk_cap {
            return Err("session disk bound".into());
        }
        #[cfg(test)]
        let (flags, require_static) = self.profile.unwrap_or((FLAGS_V2, true));
        #[cfg(not(test))]
        let (flags, require_static): (&'static [&'static str], bool) = (FLAGS_V2, true);
        #[cfg(test)]
        let address_space = self
            .compile_address_space
            .unwrap_or(COMPILE_ADDRESS_SPACE_BYTES);
        #[cfg(not(test))]
        let address_space = COMPILE_ADDRESS_SPACE_BYTES;
        let guard = Guard {
            address_space_bytes: address_space,
            file_bytes: COMPILE_FILE_BYTES,
            deadline: PROBE_DEADLINE,
        };
        #[cfg(test)]
        let path_env = self
            .path_env
            .clone()
            .unwrap_or_else(|| std::env::var_os("PATH").unwrap_or_default());
        #[cfg(not(test))]
        let path_env = std::env::var_os("PATH").unwrap_or_default();
        let dir = scratch()?;
        let compiler = toolchain::resolve(&self.compiler, &path_env)?;
        let compiler_sha256 = toolchain::file_sha256(&compiler, toolchain::MAX_PROGRAM_BYTES)?;
        let (ok, version_out, version_err) = toolchain::guarded(
            &compiler,
            &["--version".as_ref()],
            &[],
            &dir.0,
            "compiler-version",
            guard,
        )?;
        let (ok2, triple, _) = toolchain::guarded(
            &compiler,
            &["-dumpmachine".as_ref()],
            &[],
            &dir.0,
            "compiler-dumpmachine",
            guard,
        )?;
        if !ok || !ok2 {
            return Err("compiler probe failed".into());
        }
        let host_triple = String::from_utf8_lossy(&triple).trim().to_owned();
        if host_triple.is_empty() || host_triple.len() > 128 {
            return Err("compiler host triple".into());
        }
        let mut version = version_out;
        version.extend_from_slice(&version_err);
        let programs = toolchain::toolchain(&compiler, &path_env, &dir.0, guard)?;
        let words_text = words.map(|w| w.to_string()).join(",");
        let driver = DRIVER_V2
            .replace("@MAGIC@", &MAGIC.to_string())
            .replace("@WORDS@", &words_text)
            .replace("@LANES@", &lanes.to_string())
            .replace("@AS_CAP@", &cap.to_string())
            .replace("@FSIZE@", &state_bytes.to_string())
            .replace("@STACK@", &STACK_BYTES.to_string())
            .replace("@KERNEL@", KERNEL);
        let source = format!("{SHIM_V1}namespace dev {{\n{text}\n}}\n{driver}");
        fs::write(dir.0.join("model.cpp"), &source).map_err(|e| e.to_string())?;
        let exe = dir.0.join("model");
        let mut args: Vec<&std::ffi::OsStr> = flags.iter().map(|f| f.as_ref()).collect();
        args.extend(["model.cpp", "-o", "model"].map(std::ffi::OsStr::new));
        let (built, _, err) = toolchain::guarded(
            &compiler,
            &args,
            &[("PATH", &path_env), ("TMPDIR", dir.0.as_os_str())],
            &dir.0,
            "compile",
            Guard {
                deadline: COMPILE_DEADLINE,
                ..guard
            },
        )?;
        if !built {
            return Err(format!(
                "model compile failed: {}",
                String::from_utf8_lossy(&err)
            ));
        }
        #[cfg(test)]
        if let Some(hook) = &self.after_compile {
            hook();
        }
        // R3: every recorded toolchain file must be unchanged after the
        // compile (not atomic pinning; see the receipt's hash window).
        toolchain::verify(&programs)?;
        // F1: the product executable must be static; otherwise its runtime
        // libraries are listed and hashed.
        let runtime_static = toolchain::elf_interp(&exe)?.is_none();
        if require_static && !runtime_static {
            return Err("model executable is dynamically linked".into());
        }
        let runtime_libraries = toolchain::libraries(&exe, &dir.0, "libraries-model", guard)?;
        // F2/F4: the compile stage's scratch footprint.
        if toolchain::dir_bytes(&dir.0)? > COMPILE_DISK_BYTES {
            return Err("compile stage disk".into());
        }
        let executable_sha256 = toolchain::file_sha256(&exe, 64 * 1024 * 1024)?;
        let initial_sha256 = Self::write_image(&dir.0.join("epoch.in"), image, words)?;
        let mut epochs = Vec::new();
        epochs
            .try_reserve_exact(1024)
            .map_err(|_| "epoch digest reservation")?;
        let record = ModelRecord {
            compiler_path: compiler,
            compiler_sha256,
            compiler_version_sha256: sha256(&version),
            host_triple,
            flags,
            shim_sha256: sha256(SHIM_V1.as_bytes()),
            driver_sha256: sha256(DRIVER_V2.as_bytes()),
            model_source_sha256: sha256(source.as_bytes()),
            executable_sha256,
            kernel_sha256: sha256(text.as_bytes()),
            stack_bytes: STACK_BYTES,
            child_address_space_bytes: cap,
            lanes,
            words,
            toolchain: programs,
            runtime_static,
            runtime_libraries,
            compile_stage: receipt::CompileStage {
                hard_address_space_bytes: address_space,
                hard_file_bytes: COMPILE_FILE_BYTES,
                hard_probe_deadline_ms: PROBE_DEADLINE.as_millis() as u64,
                hard_compile_deadline_ms: COMPILE_DEADLINE.as_millis() as u64,
                modelled_max_processes: COMPILE_MAX_PROCESSES,
                post_compile_scratch_bytes: COMPILE_DISK_BYTES,
            },
            steady_state_disk_bytes,
            toolchain_reverified: true,
        };
        Ok(Work {
            dir,
            exe,
            record,
            device_bytes,
            completed: 0,
            child: None,
            deadline: self.deadline,
            initial_sha256,
            epochs: RefCell::new(epochs),
        })
    }

    fn submit(&mut self, work: &mut Work) -> Submission {
        if self.poisoned || work.child.is_some() {
            return Submission::NotSubmitted;
        }
        // The previous completed (and validated) epoch's output is the device
        // state now.
        let out = work.dir.0.join("epoch.out");
        if work.completed > 0 && fs::rename(&out, work.dir.0.join("epoch.in")).is_err() {
            return Submission::NotSubmitted;
        }
        // One truncated stderr file per session (bounded by RLIMIT_FSIZE).
        let stderr = match fs::File::create(work.dir.0.join("epoch.stderr")) {
            Ok(f) => f,
            Err(_) => return Submission::NotSubmitted,
        };
        #[cfg(test)]
        let mut command = match &self.substitute {
            Some((program, args)) => {
                let mut c = Command::new(program);
                c.args(args);
                c
            }
            None => Command::new(&work.exe),
        };
        #[cfg(not(test))]
        let mut command = Command::new(&work.exe);
        match command
            .current_dir(&work.dir.0)
            .env_clear()
            .env("MALLOC_ARENA_MAX", "1")
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(stderr)
            .process_group(0)
            .spawn()
        {
            Ok(child) => {
                work.child = Some((child, Instant::now()));
                Submission::Submitted
            }
            Err(_) => Submission::NotSubmitted,
        }
    }

    fn poll(&mut self, work: &mut Work) -> FenceObservation {
        let Some((child, started)) = work.child.as_mut() else {
            return FenceObservation::Unknown;
        };
        #[cfg(test)]
        if self.unconfirmable {
            return FenceObservation::Unknown;
        }
        match child.try_wait() {
            Ok(Some(status)) => {
                work.child = None;
                if status.success() {
                    work.completed += 1;
                    FenceObservation::Complete
                } else {
                    FenceObservation::FailedQuiescent
                }
            }
            Ok(None) if started.elapsed() >= work.deadline => {
                invar_foreign::numeric::terminate_compiler_group(child.id());
                let _ = child.kill();
                match child.wait() {
                    Ok(_) => {
                        work.child = None;
                        FenceObservation::FailedQuiescent
                    }
                    Err(_) => FenceObservation::Unknown,
                }
            }
            Ok(None) => FenceObservation::Pending,
            Err(_) => FenceObservation::Unknown,
        }
    }

    fn read_into(&self, work: &Work, slots: Slots<'_>) -> Result<(), String> {
        let path = work.dir.0.join("epoch.out");
        let meta = fs::symlink_metadata(&path).map_err(|e| e.to_string())?;
        if !meta.is_file() || meta.len() != (HEADER_WORDS * 8 + work.device_bytes) as u64 {
            return Err("epoch output length".into());
        }
        let mut file = fs::File::open(&path).map_err(|e| e.to_string())?;
        let mut hash = Sha256::new();
        let mut word = [0u8; 8];
        let mut next = |file: &mut fs::File| -> Result<u64, String> {
            file.read_exact(&mut word).map_err(|e| e.to_string())?;
            hash.update(word);
            Ok(u64::from_le_bytes(word))
        };
        let mut head = [0u64; HEADER_WORDS];
        for h in &mut head {
            *h = next(&mut file)?;
        }
        if head[0] != MAGIC
            || head[1] != work.completed
            || head[2..]
                .iter()
                .zip(work.record.words)
                .any(|(h, w)| *h != w as u64)
        {
            return Err("epoch output header".into());
        }
        let mut chunk = [0u8; IO_CHUNK];
        let mut fill = |file: &mut fs::File,
                        dst: &mut dyn FnMut(usize, u64),
                        n: usize|
         -> Result<(), String> {
            let mut at = 0;
            while at < n {
                let take = (n - at).min(IO_CHUNK / 8);
                file.read_exact(&mut chunk[..take * 8])
                    .map_err(|e| e.to_string())?;
                hash.update(&chunk[..take * 8]);
                for (i, w) in chunk[..take * 8].chunks_exact(8).enumerate() {
                    dst(at + i, u64::from_le_bytes(w.try_into().expect("8 bytes")));
                }
                at += take;
            }
            Ok(())
        };
        let Slots {
            input,
            output,
            trace,
            config,
            heap,
            cleanup,
            tasks,
        } = slots;
        let n = input.len();
        fill(&mut file, &mut |i, w| input[i] = w as i64, n)?;
        let n = output.len();
        fill(&mut file, &mut |i, w| output[i] = w as i64, n)?;
        let n = trace.len();
        fill(&mut file, &mut |i, w| trace[i] = w, n)?;
        let n = config.len();
        fill(&mut file, &mut |i, w| config[i] = w, n)?;
        let n = heap.len();
        fill(&mut file, &mut |i, w| heap[i] = w as i64, n)?;
        let n = cleanup.len();
        fill(&mut file, &mut |i, w| cleanup[i] = w, n)?;
        file.read_exact(tasks).map_err(|e| e.to_string())?;
        hash.update(&*tasks);
        let mut epochs = work.epochs.borrow_mut();
        if epochs.len() == epochs.capacity() {
            return Err("epoch digest capacity".into());
        }
        epochs.push(hash.finalize().into());
        Ok(())
    }

    fn poison(&mut self) {
        self.poisoned = true;
    }
}

fn le_digest<T: Copy>(words: &[T], to: impl Fn(T) -> u64) -> [u8; 32] {
    let mut hash = Sha256::new();
    for w in words {
        hash.update(to(*w).to_le_bytes());
    }
    hash.finalize().into()
}

/// Only a published CPU-group-model session can obtain an execution receipt;
/// the evidence backend is this concrete sealed type, never a label.
impl Session<'_, CpuGroupModel> {
    pub(crate) fn execution_receipt(&self) -> Result<receipt::ExecutionReceipt, Error> {
        if self.state != State::Published {
            return Err(Error::State(self.state));
        }
        let fence = self.fence.as_ref().ok_or(Error::State(self.state))?;
        let (record, initial_sha256, epochs) = fence
            .observe(|w: &Work| {
                (
                    w.record.clone(),
                    w.initial_sha256,
                    w.epochs.borrow().clone(),
                )
            })
            .map_err(|_| Error::State(self.state))?;
        let terminals: Vec<(u64, [u8; 32])> = (0..self.config.roots)
            .map(|r| {
                let output = root(&self.previous, self.lane, r).output;
                (output[0] as u64, le_digest(output, |w| w as u64))
            })
            .collect();
        let c = self.config;
        receipt::cpu_group_model(
            ReceiptSeal(()),
            receipt::Run {
                compile: self.plan.artifact(),
                plan_nonce: self.plan.nonce(),
                config: [
                    c.roots as u64,
                    c.fuel,
                    c.transitions,
                    c.trace_events as u64,
                    c.quantum,
                    c.max_submissions,
                ],
                input_sha256: le_digest(&self.previous.input, |w| w as u64),
                initial_sha256,
                epochs: &epochs,
                terminals: &terminals,
                work_reserved: self.ledger.reserved(),
                work_used: self.ledger.used(),
                host_bytes: self.host_bytes,
                model: receipt::ModelIdentity {
                    compiler_path: &record.compiler_path,
                    compiler_sha256: record.compiler_sha256,
                    compiler_version_sha256: record.compiler_version_sha256,
                    host_triple: &record.host_triple,
                    flags: record.flags,
                    shim_sha256: record.shim_sha256,
                    driver_sha256: record.driver_sha256,
                    model_source_sha256: record.model_source_sha256,
                    executable_sha256: record.executable_sha256,
                    kernel_sha256: record.kernel_sha256,
                    stack_bytes: record.stack_bytes,
                    child_address_space_bytes: record.child_address_space_bytes,
                    lanes: record.lanes,
                    words: record.words,
                    toolchain: record
                        .toolchain
                        .iter()
                        .map(|p| receipt::ProgramIdentity {
                            role: p.role,
                            path: &p.path,
                            sha256: p.sha256,
                            libraries: &p.libraries,
                        })
                        .collect(),
                    runtime_static: record.runtime_static,
                    runtime_libraries: &record.runtime_libraries,
                    compile_stage: record.compile_stage,
                    steady_state_disk_bytes: record.steady_state_disk_bytes,
                    steady_state_disk_cap: SESSION_DISK_BYTES,
                    toolchain_reverified: record.toolchain_reverified,
                },
            },
        )
        .map_err(Error::Provenance)
    }
}

// The model is GNU/Linux-only by design: a fully static ELF driver
// (`FLAGS_V2`), its `PT_INTERP` and loader listing, and the GNU driver's
// collect2/as/ld identity. Elsewhere the model's `prepare` fails closed, so
// these tests do not apply there.
#[cfg(all(test, target_os = "linux"))]
#[path = "../../../tests/support/cuda_owned_numeric_host_cpu_model.rs"]
mod tests;
