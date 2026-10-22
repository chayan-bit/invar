//! Actual CUDA adapter. It runs compiled-ahead images only: the device opens
//! through the Driver API, an image is validated and matched to its program
//! before `cuModuleLoadData`, and no compiler is loaded or spawned. CPU
//! qualification never invokes it and cannot substitute for its receipt.
use super::{cuda_aot, *};
use std::collections::VecDeque;

/// Loaded images kept per device; the oldest is unloaded past this bound
/// once no submitted work holds it.
const MAX_IMAGES: usize = 8;

struct Loaded {
    header: cuda_aot::Header,
    kernel: invar_cuda::cuda::gnet::Kernel,
    receipt: serde_json::Value,
}

pub struct Cuda {
    device: invar_cuda::cuda::CudaDevice,
    images: VecDeque<Loaded>,
    current: Option<usize>,
    poisoned: bool,
}
impl Cuda {
    pub fn open() -> Result<Self, Error> {
        Ok(Self {
            device: invar_cuda::cuda::CudaDevice::open_driver()
                .map_err(|e| Error::Backend(format!("CUDA open:{e:?}")))?,
            images: VecDeque::new(),
            current: None,
            poisoned: false,
        })
    }
    pub fn sm(&self) -> u32 {
        self.device.sm()
    }
    pub fn driver_version(&self) -> u32 {
        self.device.driver_version()
    }
    pub fn name(&self) -> &str {
        self.device.name()
    }
    /// Validate a container from untrusted bytes, match it to `plan` and this
    /// device, and load its CUBIN. Every refusal is typed; none recompiles.
    pub fn load(&mut self, plan: &Plan, bytes: &[u8]) -> Result<serde_json::Value, Error> {
        let image = cuda_aot::decode(bytes).map_err(Error::Image)?;
        image.header().verify(plan).map_err(Error::Image)?;
        image
            .header()
            .admit_device(self.sm(), self.driver_version())
            .map_err(Error::Image)?;
        let kernel = self
            .device
            .load_gnet_image(image.cubin())
            .map_err(|e| Error::Image(cuda_aot::Error::Rejected(format!("{e:?}"))))?;
        let receipt = image.receipt();
        self.images.push_back(Loaded {
            header: image.header().clone(),
            kernel,
            receipt: receipt.clone(),
        });
        if self.images.len() > MAX_IMAGES {
            self.images.pop_front();
        }
        self.current = None;
        Ok(receipt)
    }
    /// The load receipt of the image matching `plan`, if one is loaded.
    pub fn receipt(&self, plan: &Plan) -> Option<&serde_json::Value> {
        self.find(plan).map(|i| &self.images[i].receipt)
    }
    fn find(&self, plan: &Plan) -> Option<usize> {
        self.images
            .iter()
            .rposition(|loaded| loaded.header.verify(plan).is_ok())
    }
}
impl Backend for Cuda {
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
    fn admit(&mut self, plan: &Plan) -> Result<(), String> {
        self.current = self.find(plan);
        self.current.map(|_| ()).ok_or_else(|| {
            "no compiled-ahead CUDA image is loaded for this program; build one with `invar build --target cuda`".into()
        })
    }
    fn prepare(
        &mut self,
        artifact: &compiled::Artifact,
        bytes: [usize; 12],
        input: &[u64],
        config: [u64; 6],
    ) -> Result<Self::Work, String> {
        if artifact.target() != Target::Cuda || artifact.abi_version() != 5 {
            return Err("CUDA device artifact required".into());
        }
        let loaded = self
            .current
            .and_then(|i| self.images.get(i))
            .filter(|loaded| {
                loaded.header.device_source_sha256
                    == <[u8; 32]>::from(Sha256::digest(artifact.source()))
            })
            .ok_or("CUDA prepare: the admitted image does not match this artifact")?;
        self.device
            .prepare_gnet(&loaded.kernel, bytes, input, config)
            .map_err(|e| format!("CUDA prepare:{e:?}"))
    }
    fn ranges(work: &Self::Work) -> [BufferRange; 12] {
        let ids = work.addresses();
        let bytes = work.bytes();
        std::array::from_fn(|i| BufferRange {
            device: work.context_identity(),
            allocation: ids[i],
            offset: 0,
            bytes: bytes[i],
        })
    }
    fn submit(work: &mut Self::Work) -> Submission {
        match work.submit() {
            Ok(()) => Submission::Submitted,
            Err(_) => Submission::Unknown,
        }
    }
    fn fence(work: &mut Self::Work) -> FenceObservation {
        if work.complete() {
            FenceObservation::Complete
        } else {
            FenceObservation::Pending
        }
    }
    fn read(work: &Self::Work) -> Result<[Vec<u64>; 12], String> {
        work.read().map_err(|e| format!("CUDA read:{e:?}"))
    }
}
