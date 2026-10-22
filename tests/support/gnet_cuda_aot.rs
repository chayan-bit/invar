//! Test harness build step for the compiled-ahead CUDA route. The production
//! `Cuda` backend only runs loaded images; this wrapper plays `invar build`
//! for each plan it is asked to admit (NVRTC in the bounded helper
//! subprocess), loads the container through the same untrusted-bytes path,
//! and then delegates every device operation unchanged. Images are ABI5
//! only, so ABI13 strict-f32 plans run through the runtime-NVRTC adapter,
//! which compiles their D1 probe; `nvrtc_runs` counts them.
#![allow(dead_code)]

use invar::general_net::{
    compiled::{self, BufferRange, FenceObservation, Submission},
    device_host::{Backend, Plan, cuda::Cuda, cuda_aot, cuda_nvrtc::CudaNvrtc},
    provenance,
};

pub struct Aot {
    pub cuda: Cuda,
    pub builds: usize,
    nvrtc: Option<CudaNvrtc>,
    use_nvrtc: bool,
    pub nvrtc_runs: usize,
}

pub fn compiler() -> &'static std::path::Path {
    std::path::Path::new(env!("CARGO_BIN_EXE_invar-nvrtc"))
}

pub fn open() -> Aot {
    Aot {
        cuda: Cuda::open().expect("CUDA device"),
        builds: 0,
        nvrtc: None,
        use_nvrtc: false,
        nvrtc_runs: 0,
    }
}

impl Backend for Aot {
    type Work = <Cuda as Backend>::Work;
    fn label() -> provenance::FenceBackend {
        Cuda::label()
    }
    fn poisoned(&self) -> bool {
        self.cuda.poisoned() || self.nvrtc.as_ref().is_some_and(|n| n.poisoned())
    }
    fn poison(&mut self) {
        self.cuda.poison();
        if let Some(n) = self.nvrtc.as_mut() {
            n.poison();
        }
    }
    fn numeric_receipt(&self) -> Option<&serde_json::Value> {
        self.nvrtc
            .as_ref()
            .filter(|_| self.use_nvrtc)
            .and_then(|n| n.numeric_receipt())
    }
    fn admit(&mut self, plan: &Plan) -> Result<(), String> {
        self.use_nvrtc = plan.artifact().abi_version() == 13;
        if self.use_nvrtc {
            if self.nvrtc.is_none() {
                self.nvrtc = Some(CudaNvrtc::open(compiler()).map_err(|e| format!("{e:?}"))?);
            }
            self.nvrtc_runs += 1;
            return Ok(());
        }
        if self.cuda.admit(plan).is_err() {
            let image = cuda_aot::build(plan, compiler(), self.cuda.sm())?;
            self.cuda
                .load(plan, &image)
                .map_err(|e| format!("load:{e:?}"))?;
            self.builds += 1;
        }
        self.cuda.admit(plan)
    }
    fn prepare(
        &mut self,
        artifact: &compiled::Artifact,
        bytes: [usize; 12],
        input: &[u64],
        config: [u64; 6],
    ) -> Result<Self::Work, String> {
        match self.nvrtc.as_mut().filter(|_| self.use_nvrtc) {
            Some(nvrtc) => nvrtc.prepare(artifact, bytes, input, config),
            None => self.cuda.prepare(artifact, bytes, input, config),
        }
    }
    fn ranges(work: &Self::Work) -> [BufferRange; 12] {
        Cuda::ranges(work)
    }
    fn submit(work: &mut Self::Work) -> Submission {
        Cuda::submit(work)
    }
    fn fence(work: &mut Self::Work) -> FenceObservation {
        Cuda::fence(work)
    }
    fn read(work: &Self::Work) -> Result<[Vec<u64>; 12], String> {
        Cuda::read(work)
    }
}
