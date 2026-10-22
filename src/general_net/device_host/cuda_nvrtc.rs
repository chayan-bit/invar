//! Developer and test CUDA adapter: compiles the generated device source with
//! the bounded NVRTC helper at run time, like the Metal runtime-source route.
//! Production runs use compiled-ahead images through `cuda::Cuda`, which never
//! loads a compiler. ABI13 strict-f32 entries run only here for now, because
//! their D1 numeric probe is still compiled at run time.
use super::*;
use std::{
    collections::HashMap,
    path::{Path, PathBuf},
    sync::Arc,
};

/// Opt-in for forced CUDA without `--artifact` (debugging and hardware tests).
pub const RUNTIME_NVRTC_ENV: &str = "INVAR_CUDA_RUNTIME_NVRTC";

/// Why one ahead-of-time compile was refused. The budget is the bounded
/// NVRTC child's own: a wall deadline and a cubin size cap per compile.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CompileRefusal {
    Deadline,
    Limit,
    Compile,
    Unavailable,
    Other,
}
impl CompileRefusal {
    pub fn class(self) -> &'static str {
        match self {
            Self::Deadline => "compile_deadline",
            Self::Limit => "compile_limit",
            Self::Compile => "compile_error",
            Self::Unavailable => "compiler_unavailable",
            Self::Other => "compile_other",
        }
    }
}
impl From<invar_cuda::Error> for CompileRefusal {
    fn from(error: invar_cuda::Error) -> Self {
        match error {
            invar_cuda::Error::Timeout => Self::Deadline,
            invar_cuda::Error::Limit => Self::Limit,
            invar_cuda::Error::Compile(_) => Self::Compile,
            invar_cuda::Error::Unavailable => Self::Unavailable,
            _ => Self::Other,
        }
    }
}

/// Compile one general-net device source ahead of time in its own bounded
/// `invar-nvrtc` child (callers may run several at once). Nothing is loaded.
pub fn compile_ahead(
    compiler: &Path,
    caps: invar_device::Capabilities,
    source: &str,
) -> Result<Vec<u8>, CompileRefusal> {
    invar_cuda::cuda::compile_gnet_cubin_bounded(source, compiler, caps)
        .map(|cubin| cubin.image)
        .map_err(CompileRefusal::from)
}

/// Cubins compiled ahead for one device, keyed by the SHA-256 of the exact
/// device source they were compiled from.
pub struct Cubins {
    caps: invar_device::Capabilities,
    images: HashMap<[u8; 32], Vec<u8>>,
}
impl Cubins {
    pub fn new(caps: invar_device::Capabilities) -> Self {
        Self {
            caps,
            images: HashMap::new(),
        }
    }
    pub fn insert(&mut self, source: &str, image: Vec<u8>) {
        self.images.insert(Sha256::digest(source).into(), image);
    }
    fn get(&self, source: &str) -> Option<&[u8]> {
        let key: [u8; 32] = Sha256::digest(source).into();
        self.images.get(&key).map(Vec::as_slice)
    }
}

pub struct CudaNvrtc {
    device: invar_cuda::cuda::CudaDevice,
    compiler: PathBuf,
    cubins: Option<Arc<Cubins>>,
    poisoned: bool,
    numeric: Option<serde_json::Value>,
}
impl CudaNvrtc {
    pub fn open(compiler: &Path) -> Result<Self, Error> {
        Ok(Self {
            device: invar_cuda::cuda::CudaDevice::open()
                .map_err(|e| Error::Backend(format!("CUDA open:{e:?}")))?,
            compiler: compiler.into(),
            cubins: None,
            poisoned: false,
            numeric: None,
        })
    }
    pub fn capabilities(&self) -> invar_device::Capabilities {
        self.device.capabilities()
    }
    /// Load cubins compiled ahead for this device's capabilities instead of
    /// compiling the same source again; other sources still compile here.
    pub fn with_cubins(mut self, cubins: Arc<Cubins>) -> Result<Self, Error> {
        if cubins.caps != self.capabilities() {
            return Err(Error::Backend("cubins compiled for another device".into()));
        }
        self.cubins = Some(cubins);
        Ok(self)
    }
}
impl Backend for CudaNvrtc {
    type Work = invar_cuda::cuda::gnet::Work;
    fn label() -> provenance::FenceBackend {
        provenance::FenceBackend::CudaDriver
    }
    fn poisoned(&self) -> bool {
        self.poisoned || self.device.is_poisoned()
    }
    fn poison(&mut self) {
        self.poisoned = true
    }
    fn numeric_receipt(&self) -> Option<&serde_json::Value> {
        self.numeric.as_ref()
    }
    fn prepare(
        &mut self,
        artifact: &compiled::Artifact,
        bytes: [usize; 12],
        input: &[u64],
        config: [u64; 6],
    ) -> Result<Self::Work, String> {
        if artifact.target() != Target::Cuda || !matches!(artifact.abi_version(), 5 | 13) {
            return Err("CUDA device artifact required".into());
        }
        if artifact.abi_version() == 13 {
            let report = self
                .device
                .strict_f32_probe(&self.compiler)
                .map_err(|e| format!("D1 numeric preflight:{e:?}"))?;
            self.numeric = Some(super::numeric_probe_receipt(report));
        }
        let ahead = self.cubins.as_ref().and_then(|c| c.get(artifact.source()));
        let compiled;
        let image = match ahead {
            Some(image) => image,
            None => {
                compiled = invar_cuda::cuda::compile_gnet_cubin_bounded(
                    artifact.source(),
                    &self.compiler,
                    self.device.capabilities(),
                )
                .map_err(|e| format!("CUDA prepare:{e:?}"))?;
                &compiled.image
            }
        };
        let kernel = self
            .device
            .load_gnet_image(image)
            .map_err(|e| format!("CUDA prepare:{e:?}"))?;
        self.device
            .prepare_gnet(&kernel, bytes, input, config)
            .map_err(|e| format!("CUDA prepare:{e:?}"))
    }
    fn ranges(work: &Self::Work) -> [BufferRange; 12] {
        <cuda::Cuda as Backend>::ranges(work)
    }
    fn submit(work: &mut Self::Work) -> Submission {
        <cuda::Cuda as Backend>::submit(work)
    }
    fn fence(work: &mut Self::Work) -> FenceObservation {
        <cuda::Cuda as Backend>::fence(work)
    }
    fn read(work: &Self::Work) -> Result<[Vec<u64>; 12], String> {
        <cuda::Cuda as Backend>::read(work)
    }
}

impl super::host_frame::RegionBackend for CudaNvrtc {
    fn execute(
        &mut self,
        program: &compiled::body_map::Compiled,
        image: &super::host_frame::Image,
        groups: usize,
    ) -> Result<super::host_frame::Image, Error> {
        if self.poisoned() || program.lanes.target() != Target::Cuda {
            return Err(Error::Poisoned);
        }
        let report = self
            .device
            .strict_f32_probe(&self.compiler)
            .map_err(|e| Error::Backend(format!("D1:{e:?}")))?;
        self.numeric = Some(super::numeric_probe_receipt(report));
        self.device
            .run_gnet_region(program.device_source(), image, groups, &self.compiler)
            .map_err(|e| {
                self.poisoned = true;
                Error::Backend(format!("P2 CUDA:{e:?}"))
            })
    }
}
