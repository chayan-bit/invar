//! Source-bound packets reaching the owning strict Metal contraction runtime.
//! The same reverified plan as CUDA, emitted as MSL with an underflow guard.
//! The D1 probe runs once at open; a device it admits neither exactly nor under
//! the guard refuses every packet before submission. A dispatch whose guard
//! fires is discarded and reported as `SubnormalFallback`, so the source loop
//! runs on the CPU; nothing unproven is ever published.
use super::packet::{logical_output, packet_inputs, packet_shape, response_packet};
use super::*;
use invar_cuda::{Error as DeviceError, SubmitError};
use invar_device::f32_probe::Report;
use sha2::{Digest, Sha256};

pub struct Metal {
    source: Vec<u8>,
    entry: String,
    device: invar_metal::MetalDevice,
    probe: Report,
}
impl Metal {
    /// Explicit GPU open plus the per-process D1 probe. Pure packet logic
    /// never calls this. Production passes the compiled-ahead container built
    /// by `invar build ... --family tensor`; it must be the build of exactly
    /// `source` and `entry`, and a kernel missing from it is refused, never
    /// compiled. `Kernels::RuntimeSource` is the developer path.
    pub fn open(
        source: &[u8],
        entry: &str,
        kernels: crate::schedule::metal::Kernels<'_>,
    ) -> Result<Self, invar_metal::Error> {
        if source.len() > 1024 * 1024 {
            return Err(invar_metal::Error::Limit);
        }
        let checked = crate::front::parse(source)
            .and_then(|program| crate::check::check(program, false))
            .map_err(|e| invar_metal::Error::Device(format!("source: {e:?}")))?;
        crate::tensor::Contraction::from_checked(&checked, entry)
            .map_err(|e| invar_metal::Error::Device(format!("source contract: {e:?}")))?;
        let mut device = open_device(source, entry, kernels)?;
        let probe = device.strict_f32_probe()?.clone();
        Ok(Self {
            source: source.to_vec(),
            entry: entry.into(),
            device,
            probe,
        })
    }
    pub fn capabilities(&self) -> Capabilities {
        self.device.capabilities()
    }
    /// The D1 verdict that governs every strict packet on this device.
    pub fn probe(&self) -> &Report {
        &self.probe
    }
}
fn open_device(
    source: &[u8],
    entry: &str,
    kernels: crate::schedule::metal::Kernels<'_>,
) -> Result<invar_metal::MetalDevice, invar_metal::Error> {
    use crate::schedule::metal::Kernels;
    use invar_metal::{Error, LibraryError};
    match kernels {
        Kernels::Prebuilt(image) => {
            image
                .header()
                .verify(
                    crate::metal_aot::Route::Tensor,
                    entry,
                    Sha256::digest(source).into(),
                )
                .map_err(|e| Error::Library(LibraryError::Rejected(format!("{e:?}"))))?;
            let mut device = invar_metal::MetalDevice::open()?;
            image
                .install(&mut device)
                .map_err(|e| Error::Library(LibraryError::Rejected(format!("{e:?}"))))?;
            Ok(device)
        }
        Kernels::RuntimeSource => invar_metal::MetalDevice::open_runtime_source(),
        Kernels::Missing => Err(Error::Unavailable(
            "Metal runs compiled-ahead libraries: build them with `invar build PATH --target metal --entry ENTRY --family tensor --fuel FUEL --output FILE` and pass `--artifact FILE`".into(),
        )),
    }
}

/// Every Metal kernel a `source-device PATH tensor ENTRY metal FUEL` run on a
/// device with `caps` can dispatch: the D1 probe, then each contraction the
/// source reaches within `fuel`. Invar source has no external input, so one
/// exact CPU run discovers them; a run with less fuel reaches a subset.
pub fn kernels(
    source: &[u8],
    entry: &str,
    fuel: u64,
    caps: Capabilities,
) -> Result<Vec<(String, Vec<String>)>, String> {
    struct Recorder<'a> {
        source: &'a [u8],
        entry: &'a str,
        caps: Capabilities,
        msl: &'a mut Vec<String>,
        failed: &'a mut Option<String>,
    }
    impl Device for Recorder<'_> {
        type Work = <CpuDevice as Device>::Work;
        fn prepare(&mut self, bytes: &[u8]) -> Result<Self::Work, DeviceError> {
            let emitted = prepare_packet(self.source, self.entry, self.caps, bytes)
                .map_err(|e| format!("{e:?}"))
                .and_then(|work| {
                    invar_device::floating::emit_msl(work.plan.plan()).map_err(|e| format!("{e:?}"))
                });
            match emitted {
                Ok(msl) if !self.msl.contains(&msl) => self.msl.push(msl),
                Ok(_) => {}
                Err(error) => *self.failed = Some(error),
            }
            CpuDevice.prepare(bytes)
        }
        fn submit(&mut self, work: &mut Self::Work) -> Result<(), SubmitError> {
            CpuDevice.submit(work)
        }
        fn complete(&mut self, work: &mut Self::Work) -> Result<bool, DeviceError> {
            CpuDevice.complete(work)
        }
        fn read(&mut self, work: &mut Self::Work) -> Result<Vec<u8>, DeviceError> {
            CpuDevice.read(work)
        }
    }
    let (mut msl, mut failed) = (Vec::new(), None);
    let recorder = Recorder {
        source,
        entry,
        caps,
        msl: &mut msl,
        failed: &mut failed,
    };
    run_source(source, fuel, &mut Runtime::new(recorder, 2), caps)
        .map_err(|e| format!("host:{e:?}"))?;
    if let Some(error) = failed {
        return Err(format!("contraction emission:{error}"));
    }
    let mut kernels = vec![(
        invar_device::f32_probe::emit_msl(),
        vec![invar_device::f32_probe::KERNEL_NAME.to_owned()],
    )];
    for msl in msl {
        kernels.push((msl, vec![invar_device::floating::KERNEL_NAME.to_owned()]));
    }
    Ok(kernels)
}

pub struct Work {
    plan: crate::tensor::Emission,
    packet: Vec<u8>,
    inputs: [Vec<u32>; 2],
    response: Option<Vec<u8>>,
}
fn prepare_packet(
    source: &[u8],
    entry: &str,
    caps: Capabilities,
    bytes: &[u8],
) -> Result<Work, DeviceError> {
    let (dimensions, [a, b, c]) = packet_shape(bytes)?;
    let plan = crate::tensor::emit_msl(source, entry, dimensions, [a as u64, b as u64], caps)
        .map_err(|_| DeviceError::InvalidHandle)?;
    if plan.key().to_hex().as_bytes() != &bytes[40..104]
        || plan.plan().ir().output.elements != c as u64
    {
        return Err(DeviceError::StaleBorrow);
    }
    let inputs = packet_inputs(bytes, [a, b])?;
    let mut packet = Vec::new();
    packet
        .try_reserve_exact(bytes.len())
        .map_err(|_| DeviceError::OutOfMemory)?;
    packet.extend_from_slice(bytes);
    Ok(Work {
        plan,
        packet,
        inputs,
        response: None,
    })
}
impl Device for Metal {
    type Work = Work;
    fn prepare(&mut self, bytes: &[u8]) -> Result<Work, DeviceError> {
        if !(self.probe.admits_strict()
            || self
                .probe
                .admits_strict_with_fallback(invar_device::f32_probe::CONTRACTION_OPS))
        {
            return Err(DeviceError::Unavailable);
        }
        self.device.reap();
        if self.device.is_poisoned() {
            return Err(DeviceError::Poisoned);
        }
        prepare_packet(&self.source, &self.entry, self.capabilities(), bytes)
    }
    fn submit(&mut self, work: &mut Work) -> Result<(), SubmitError> {
        if work.response.is_some() {
            return Err(SubmitError::FailedQuiescent(DeviceError::InvalidHandle));
        }
        let result = self
            .device
            .run_contraction(work.plan.plan(), [&work.inputs[0], &work.inputs[1]]);
        match result {
            Ok(result) => {
                let words = logical_output(work.plan.plan(), &result.output)
                    .map_err(SubmitError::FailedQuiescent)?;
                work.response = Some(
                    response_packet(&work.packet, &words).map_err(SubmitError::FailedQuiescent)?,
                );
                Ok(())
            }
            // Missed deadlines leave the command quarantined in the device.
            Err(invar_metal::Error::Pending) => {
                Err(SubmitError::InFlightUnknown(DeviceError::Timeout))
            }
            Err(invar_metal::Error::Unavailable(_)) => {
                Err(SubmitError::FailedQuiescent(DeviceError::Unavailable))
            }
            Err(invar_metal::Error::SubnormalFallback) => {
                Err(SubmitError::FailedQuiescent(DeviceError::SubnormalFallback))
            }
            Err(invar_metal::Error::InvalidGuard(error)) => Err(SubmitError::FailedQuiescent(
                DeviceError::InvalidGuard(error),
            )),
            Err(invar_metal::Error::InvalidProbe(error)) => Err(SubmitError::FailedQuiescent(
                DeviceError::InvalidProbe(error),
            )),
            Err(error) => Err(SubmitError::FailedQuiescent(DeviceError::Driver(format!(
                "{error:?}"
            )))),
        }
    }
    fn complete(&mut self, work: &mut Work) -> Result<bool, DeviceError> {
        if work.response.is_some() {
            return Ok(true);
        }
        self.device.reap();
        Ok(!self.device.is_poisoned())
    }
    fn read(&mut self, work: &mut Work) -> Result<Vec<u8>, DeviceError> {
        work.response.take().ok_or(DeviceError::InvalidHandle)
    }
}
