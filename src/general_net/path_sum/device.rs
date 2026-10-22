//! Launchers for one lowering: the CPU model (always), Metal and CUDA.
//!
//! The CPU model compiles the same source as C++17 with `PROOF_CHECK`, which
//! turns every narrow operator into a checked one that aborts on a result
//! outside its type: an unsound licence fails loudly instead of agreeing by
//! accident. It is an execution model of the kernel, never a GPU claim.

use std::{
    io::{BufRead, BufReader, Write},
    path::PathBuf,
    process::{Child, ChildStdin, ChildStdout, Command, Stdio},
    sync::atomic::{AtomicU64, Ordering},
};

use super::{
    GROUP, Lowering, MAX_LANES,
    host::{Launch, Launcher, Shape},
};

/// Kernel entry points in every lowering's source, in launcher order.
pub const KERNELS: [&str; 4] = [
    "pathsum_root",
    "pathsum_fused",
    "pathsum_map",
    "pathsum_reduce",
];

pub fn kernel_index(name: &str) -> usize {
    KERNELS
        .iter()
        .position(|k| *k == name)
        .unwrap_or(usize::MAX)
}

/// Group partials a launch may publish.
pub fn max_groups() -> usize {
    MAX_LANES.div_ceil(GROUP) as usize
}

pub struct CpuModel {
    child: Child,
    input: ChildStdin,
    output: BufReader<ChildStdout>,
    directory: PathBuf,
    params: usize,
}

impl CpuModel {
    /// Compile `lowering`'s source with `$CXX` (default `c++`).
    pub fn compile(lowering: &Lowering) -> Result<Self, String> {
        static NEXT: AtomicU64 = AtomicU64::new(1);
        let nonce = NEXT.fetch_add(1, Ordering::Relaxed);
        let directory =
            std::env::temp_dir().join(format!("invar-pathsum-{}-{nonce}", std::process::id()));
        std::fs::create_dir_all(&directory).map_err(|e| e.to_string())?;
        let source = directory.join("model.cpp");
        let binary = directory.join("model");
        std::fs::write(&source, lowering.source()).map_err(|e| e.to_string())?;
        let compiler = std::env::var_os("CXX").unwrap_or_else(|| "c++".into());
        let output = Command::new(compiler)
            .args([
                "-std=c++17",
                "-O1",
                "-DPROOF_CHECK",
                "-fsanitize=undefined",
                "-fno-sanitize-recover=all",
                "-w",
            ])
            .arg(&source)
            .arg("-o")
            .arg(&binary)
            .output()
            .map_err(|e| e.to_string())?;
        if !output.status.success() {
            return Err(format!(
                "CPU model compile failed: {}",
                String::from_utf8_lossy(&output.stderr)
            ));
        }
        let mut child = Command::new(&binary)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|e| e.to_string())?;
        let input = child.stdin.take().ok_or("stdin")?;
        let output = BufReader::new(child.stdout.take().ok_or("stdout")?);
        Ok(Self {
            child,
            input,
            output,
            directory,
            params: lowering.param_count().max(1),
        })
    }
}

impl Drop for CpuModel {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
        let _ = std::fs::remove_dir_all(&self.directory);
    }
}

impl Launcher for CpuModel {
    fn launch(&mut self, lowering: &Lowering, launch: &Launch) -> Result<Vec<u32>, String> {
        let shape = u32::from(launch.shape != Shape::Root);
        let root: Vec<String> = launch.args[..self.params]
            .iter()
            .map(|w| (*w as i32).to_string())
            .collect();
        writeln!(self.input, "{} {shape} {}", launch.lanes, root.join(" "))
            .and_then(|()| self.input.flush())
            .map_err(|e| format!("CPU model input: {e}"))?;
        let mut line = String::new();
        let read = self
            .output
            .read_line(&mut line)
            .map_err(|e| format!("CPU model output: {e}"))?;
        if read == 0 {
            let mut stderr = String::new();
            if let Some(mut err) = self.child.stderr.take() {
                let _ = std::io::Read::read_to_string(&mut err, &mut stderr);
            }
            return Err(format!(
                "CPU model aborted (narrow proof check or sanitizer): {stderr}"
            ));
        }
        let words = line
            .split_whitespace()
            .map(|w| w.parse::<u32>().map_err(|e| e.to_string()))
            .collect::<Result<Vec<_>, _>>()?;
        if words.len() != lowering.out_words() {
            return Err(format!("CPU model wrote {} words", words.len()));
        }
        Ok(words)
    }
}

pub fn stages(launch: &Launch) -> Vec<(usize, u32)> {
    launch
        .stages()
        .into_iter()
        .map(|(name, groups)| (kernel_index(name), groups))
        .collect()
}

#[cfg(all(feature = "metal", target_os = "macos"))]
pub struct Metal {
    kernels: invar_metal::PathSum,
    /// GPU interval of the last launch.
    pub gpu_ms: f64,
}

#[cfg(all(feature = "metal", target_os = "macos"))]
impl Metal {
    pub fn new(device: &mut invar_metal::MetalDevice, lowering: &Lowering) -> Result<Self, String> {
        let kernels = device
            .pathsum(
                lowering.source(),
                &KERNELS,
                max_groups(),
                lowering.out_words() * 4,
            )
            .map_err(|e| format!("Metal path-sum: {e:?}"))?;
        Ok(Self {
            kernels,
            gpu_ms: 0.0,
        })
    }
}

#[cfg(all(feature = "metal", target_os = "macos"))]
impl Launcher for Metal {
    fn launch(&mut self, lowering: &Lowering, launch: &Launch) -> Result<Vec<u32>, String> {
        let stages: Vec<(usize, usize)> = stages(launch)
            .into_iter()
            .map(|(k, g)| (k, g as usize))
            .collect();
        self.kernels
            .write_args(&launch.arg_bytes())
            .map_err(|e| format!("{e:?}"))?;
        self.gpu_ms = self
            .kernels
            .dispatch(&stages)
            .map_err(|e| format!("{e:?}"))?;
        self.kernels
            .read_out(lowering.out_words())
            .map_err(|e| format!("{e:?}"))
    }
}

#[cfg(feature = "cuda")]
pub struct Cuda {
    kernels: invar_cuda::cuda::PathSum,
    /// Device interval of the last launch.
    pub gpu_ms: f64,
}

#[cfg(feature = "cuda")]
impl Cuda {
    pub fn new(
        device: &mut invar_cuda::cuda::CudaDevice,
        lowering: &Lowering,
    ) -> Result<Self, String> {
        let kernels = device
            .pathsum(
                lowering.source(),
                &KERNELS,
                max_groups(),
                lowering.out_words() * 4,
            )
            .map_err(|e| format!("CUDA path-sum: {e:?}"))?;
        Ok(Self {
            kernels,
            gpu_ms: 0.0,
        })
    }
}

#[cfg(feature = "cuda")]
impl Launcher for Cuda {
    fn launch(&mut self, lowering: &Lowering, launch: &Launch) -> Result<Vec<u32>, String> {
        self.kernels
            .write_args(&launch.arg_bytes())
            .map_err(|e| format!("{e:?}"))?;
        self.gpu_ms = self
            .kernels
            .dispatch(&stages(launch))
            .map_err(|e| format!("{e:?}"))?;
        self.kernels
            .read_out(lowering.out_words())
            .map_err(|e| format!("{e:?}"))
    }
}
