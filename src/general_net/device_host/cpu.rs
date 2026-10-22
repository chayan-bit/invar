//! Explicit CPU execution model of the emitted device body, never a GPU claim.
//! A persistent confined compiler library owns twelve real allocations and a
//! 32-thread barrier group across every source-derived continuation epoch.
use super::*;
use invar_foreign::device::Driver;
use std::{
    cell::RefCell,
    fs,
    path::{Path, PathBuf},
    process::Command,
    sync::atomic::{AtomicU64, Ordering},
};
mod source;
/// Exact refusal of an ABI12 (`cuda-owned-numeric-continuation-v1`) artifact.
pub const UNSUPPORTED_ABI12: &str = "CPU device model: unsupported ABI 12 artifact";
pub struct CpuModel {
    poisoned: bool,
    compiler: PathBuf,
}
impl Default for CpuModel {
    fn default() -> Self {
        Self {
            poisoned: false,
            compiler: std::env::var_os("CXX")
                .map(PathBuf::from)
                .unwrap_or_else(|| "c++".into()),
        }
    }
}
impl CpuModel {
    pub fn with_compiler(path: &Path) -> Self {
        Self {
            poisoned: false,
            compiler: path.into(),
        }
    }
}
/// CPU seconds for each compiler process. Over the 538 distinct runnable
/// corpus models (-O1, UBSan, g++ 14, aarch64 staging) compile cost is 5.7
/// CPU-s at the median, 15.4 at p90 and 23.5 at most.
const COMPILE_CPU_SECONDS: u32 = 60;
/// Wall backstop, the GNET NVRTC deadline. On a loaded runner models of at
/// most 20 CPU-s took over 60 s of wall time, so wall time measures load.
const COMPILE_WALL_SECONDS: &str = "300";
/// A compiled model library: compiler, emitted-source digest and bytes.
type Compiled = (PathBuf, [u8; 32], Vec<u8>);
thread_local! {
    /// The library this thread compiled last, by compiler and emitted source.
    /// Replays of one program (fuel cuts, adversarial runs) reuse its bytes
    /// instead of recompiling; every `Work` still loads its own file, so the
    /// library globals (group barrier, threadgroup statics) are never shared.
    static LAST: RefCell<Option<Compiled>> = const { RefCell::new(None) };
}
impl CpuModel {
    /// Compile `code` into `library` within the deadline and return its bytes.
    /// A failed compile keeps `directory` as evidence.
    fn compile(
        &self,
        directory: &mut Directory,
        code: &str,
        library: &Path,
    ) -> Result<Vec<u8>, String> {
        let source = directory.0.join("model.cpp");
        fs::write(&source, code).map_err(|e| e.to_string())?;
        // The CPU limit bounds the compiler's own work whatever the host load;
        // the wall deadline only catches a compiler that stalls without it.
        let output = Command::new("timeout")
            .args(["--kill-after=2", COMPILE_WALL_SECONDS, "sh", "-c"])
            .arg(format!(
                "ulimit -t {COMPILE_CPU_SECONDS} && exec \"$0\" \"$@\""
            ))
            .arg(&self.compiler)
            .args([
                "-std=c++20",
                "-O1",
                "-pthread",
                "-fno-fast-math",
                "-ffp-contract=off",
                "-fPIC",
                "-shared",
                "-fvisibility=hidden",
                "-fsanitize=undefined",
                "-fno-sanitize-recover=all",
            ])
            .arg(&source)
            .arg("-o")
            .arg(library)
            .output()
            .map_err(|e| e.to_string())?;
        if !output.status.success() {
            directory.1 = false;
            let _ = fs::write(directory.0.join("compiler.stdout"), &output.stdout);
            let _ = fs::write(directory.0.join("compiler.stderr"), &output.stderr);
            return Err(format!(
                "CPU device model compile {:?}; evidence {}: {}",
                output.status.code(),
                directory.0.display(),
                String::from_utf8_lossy(&output.stderr)
            ));
        }
        fs::read(library).map_err(|e| e.to_string())
    }
}
struct Directory(PathBuf, bool);
impl Drop for Directory {
    fn drop(&mut self) {
        if self.1 {
            let _ = fs::remove_dir_all(&self.0);
        }
    }
}
pub struct Work {
    driver: Driver,
    ranges: [BufferRange; 12],
    _directory: Directory,
}
impl Backend for CpuModel {
    type Work = Work;
    fn label() -> provenance::FenceBackend {
        provenance::FenceBackend::CpuGroupModel
    }
    fn poisoned(&self) -> bool {
        self.poisoned
    }
    fn poison(&mut self) {
        self.poisoned = true
    }
    fn prepare(
        &mut self,
        artifact: &compiled::Artifact,
        bytes: [usize; 12],
        input: &[u64],
        config: [u64; 6],
    ) -> Result<Work, String> {
        if self.poisoned {
            return Err("poisoned CPU device model".into());
        }
        // ABI12 is a compile receipt only: refused before any directory,
        // compiler or driver call. Other artifacts keep their prior refusals.
        if artifact.abi_version() == 12 {
            return Err(UNSUPPORTED_ABI12.into());
        }
        static NEXT: AtomicU64 = AtomicU64::new(1);
        let nonce = NEXT
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |n| n.checked_add(1))
            .map_err(|_| "model generation exhausted")?;
        let mut directory = Directory(
            std::env::temp_dir().join(format!("invar-device-model-{}-{nonce}", std::process::id())),
            true,
        );
        fs::create_dir(&directory.0).map_err(|e| e.to_string())?;
        let digest: [u8; 32] = Sha256::digest(artifact.source().as_bytes()).into();
        let symbol = format!(
            "invar_{}_gnet_driver_v1",
            digest
                .iter()
                .map(|b| format!("{b:02x}"))
                .collect::<String>()
        );
        let code = source::emit(artifact, &symbol)?;
        let library = directory
            .0
            .join(format!("model.{}", std::env::consts::DLL_EXTENSION));
        let digest: [u8; 32] = Sha256::digest(code.as_bytes()).into();
        let cached = LAST.with_borrow(|last| {
            last.as_ref()
                .filter(|(compiler, d, _)| *compiler == self.compiler && *d == digest)
                .map(|(_, _, image)| image.clone())
        });
        let image = match cached {
            Some(image) => {
                fs::write(&library, &image).map_err(|e| e.to_string())?;
                image
            }
            None => {
                let image = self.compile(&mut directory, &code, &library)?;
                LAST.set(Some((self.compiler.clone(), digest, image.clone())));
                image
            }
        };
        let hash: [u8; 32] = Sha256::digest(&image).into();
        let driver = Driver::load(&library, hash, &symbol, bytes, input, config)
            .map_err(|e| format!("model prepare:{e:?}"))?;
        let ids = driver.addresses();
        let ranges = std::array::from_fn(|i| BufferRange {
            device: u64::from(std::process::id()),
            allocation: ids[i],
            offset: 0,
            bytes: bytes[i],
        });
        Ok(Work {
            driver,
            ranges,
            _directory: directory,
        })
    }
    fn ranges(work: &Work) -> [BufferRange; 12] {
        work.ranges
    }
    fn submit(work: &mut Work) -> Submission {
        match work.driver.submit() {
            Ok(()) => Submission::Submitted,
            Err(invar_foreign::device::Error::Busy | invar_foreign::device::Error::Protocol) => {
                Submission::NotSubmitted
            }
            Err(_) => Submission::Unknown,
        }
    }
    fn fence(work: &mut Work) -> FenceObservation {
        match work.driver.fence() {
            Ok(true) => FenceObservation::Complete,
            Ok(false) => FenceObservation::Pending,
            Err(_) => FenceObservation::Unknown,
        }
    }
    fn read(work: &Work) -> Result<[Vec<u64>; 12], String> {
        work.driver.read().map_err(|e| format!("model read:{e:?}"))
    }
}
