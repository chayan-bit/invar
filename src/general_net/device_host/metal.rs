//! Actual Metal command adapter. It runs compiled-ahead libraries: a library
//! is validated and matched to its program before `newLibraryWithData`, and
//! the production device has no source compiler to fall back to. The CPU
//! group model never opens this device.
use super::*;
use crate::metal_aot::{self, Route};
pub struct Metal {
    device: invar_metal::MetalDevice,
    poisoned: bool,
    numeric: Option<serde_json::Value>,
}
impl Metal {
    pub fn open() -> Result<Self, Error> {
        Self::with(invar_metal::MetalDevice::open())
    }
    /// Developer debugging only: compile the generated MSL at run time.
    pub fn open_runtime_source() -> Result<Self, Error> {
        Self::with(invar_metal::MetalDevice::open_runtime_source())
    }
    fn with(device: Result<invar_metal::MetalDevice, invar_metal::Error>) -> Result<Self, Error> {
        Ok(Self {
            device: device.map_err(|e| Error::Backend(format!("Metal open:{e:?}")))?,
            poisoned: false,
            numeric: None,
        })
    }
    /// Match a validated container to `plan` (same entry, ordinary source,
    /// generated MSL, entry points and D1 probe) and load its libraries. Every
    /// refusal is typed; none compiles source instead.
    pub fn load(
        &mut self,
        plan: &Plan,
        image: &metal_aot::Image,
    ) -> Result<serde_json::Value, metal_aot::Error> {
        let header = image.header();
        header.verify(Route::Gnet, plan.entry(), plan.source_digest())?;
        header.verify_gnet(plan.artifact())?;
        image.install(&mut self.device)?;
        Ok(image.receipt())
    }
    pub fn cache_report(&self) -> invar_metal::CacheReport {
        self.device.cache_report()
    }
}
impl Backend for Metal {
    type Work = invar_metal::gnet::Work;
    fn label() -> provenance::FenceBackend {
        provenance::FenceBackend::MetalCommand
    }
    fn poisoned(&self) -> bool {
        self.poisoned || self.device.is_poisoned()
    }
    fn poison(&mut self) {
        self.poisoned = true;
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
        if artifact.target() != Target::Metal || !matches!(artifact.abi_version(), 5 | 13) {
            return Err("Metal device artifact required".into());
        }
        if artifact.abi_version() == 13 {
            let report = self
                .device
                .strict_f32_probe()
                .map_err(|e| format!("D1 numeric preflight:{e:?}"))?;
            self.numeric = Some(super::numeric_probe_receipt(report));
        }
        self.device
            .prepare_gnet(artifact.source(), bytes, input, config)
            .map_err(|e| format!("Metal prepare:{e:?}"))
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
        // This boundary returns errors only before commit; a committed command
        // always remains retained and is observed through its actual status.
        match work.submit() {
            Ok(()) => Submission::Submitted,
            Err(_) => Submission::NotSubmitted,
        }
    }
    fn fence(work: &mut Self::Work) -> FenceObservation {
        match work.completion() {
            invar_metal::gnet::Completion::Pending => FenceObservation::Pending,
            invar_metal::gnet::Completion::Complete => FenceObservation::Complete,
            invar_metal::gnet::Completion::FailedQuiescent => FenceObservation::FailedQuiescent,
        }
    }
    fn read(work: &Self::Work) -> Result<[Vec<u64>; 12], String> {
        work.read().map_err(|e| format!("Metal read:{e:?}"))
    }
}

impl super::host_frame::RegionBackend for Metal {
    fn execute(
        &mut self,
        program: &compiled::body_map::Compiled,
        image: &super::host_frame::Image,
        groups: usize,
    ) -> Result<super::host_frame::Image, Error> {
        if self.poisoned() {
            return Err(Error::Poisoned);
        }
        if program.target() != Target::Metal {
            return Err(Error::Input);
        }
        let report = self
            .device
            .strict_f32_probe()
            .map_err(|e| Error::Backend(format!("D1:{e:?}")))?;
        self.numeric = Some(super::numeric_probe_receipt(report));
        self.device
            .run_gnet_region(program.device_source(), image, groups)
            .map_err(|e| {
                self.poisoned = true;
                Error::Backend(format!("P2 Metal:{e:?}"))
            })
    }
}
