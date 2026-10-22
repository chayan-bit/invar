//! Source-bound packets reaching the owning strict CUDA contraction runtime.
//! No original source array pointer is passed through this adapter.
use super::packet::{logical_output, packet_inputs, packet_shape, response_packet};
use super::*;
use invar_cuda::{Error as DeviceError, SubmitError};
use std::path::{Path, PathBuf};

pub struct Cuda {
    source: Vec<u8>,
    entry: String,
    compiler: PathBuf,
    device: invar_cuda::cuda::CudaDevice,
}
impl Cuda {
    /// Explicit GPU open. Qualification of pure packet logic never calls this.
    pub fn open(source: &[u8], entry: &str, compiler: &Path) -> Result<Self, DeviceError> {
        if source.len() > 1024 * 1024 {
            return Err(DeviceError::Limit);
        }
        let mut owned_source = Vec::new();
        owned_source
            .try_reserve_exact(source.len())
            .map_err(|_| DeviceError::OutOfMemory)?;
        owned_source.extend_from_slice(source);
        let checked = crate::front::parse(source)
            .and_then(|program| crate::check::check(program, false))
            .map_err(|_| DeviceError::InvalidHandle)?;
        crate::tensor::Contraction::from_checked(&checked, entry)
            .map_err(|_| DeviceError::InvalidHandle)?;
        Ok(Self {
            source: owned_source,
            entry: entry.into(),
            compiler: compiler.into(),
            device: invar_cuda::cuda::CudaDevice::open()?,
        })
    }
    pub fn capabilities(&self) -> Capabilities {
        self.device.capabilities()
    }
    pub fn device_name(&self) -> &str {
        self.device.name()
    }
    /// The per-process D1 verdict for this device, run on first use.
    pub fn probe(&mut self) -> Result<&invar_device::f32_probe::Report, DeviceError> {
        self.device.strict_f32_probe(&self.compiler)
    }
}
pub struct Work {
    plan: crate::tensor::Emission,
    packet: Vec<u8>,
    inputs: [Vec<u32>; 2],
    response: Option<Vec<u8>>,
    terminal: bool,
}
fn prepare_packet(
    source: &[u8],
    entry: &str,
    caps: Capabilities,
    bytes: &[u8],
) -> Result<Work, DeviceError> {
    let (dimensions, [a, b, c]) = packet_shape(bytes)?;
    let plan = crate::tensor::emit_cuda(source, entry, dimensions, [a as u64, b as u64], caps)
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
        terminal: false,
    })
}
impl Device for Cuda {
    type Work = Work;
    fn prepare(&mut self, bytes: &[u8]) -> Result<Work, DeviceError> {
        self.device.reap();
        if self.device.is_poisoned() {
            return Err(DeviceError::Poisoned);
        }
        // D1: refuse before submission on a device the probe does not admit.
        if !self.probe()?.admits_strict() {
            return Err(DeviceError::Unavailable);
        }
        prepare_packet(&self.source, &self.entry, self.capabilities(), bytes)
    }
    fn submit(&mut self, work: &mut Work) -> Result<(), SubmitError> {
        if work.terminal || work.response.is_some() {
            return Err(SubmitError::FailedQuiescent(DeviceError::InvalidHandle));
        }
        let result = self.device.run_contraction(
            work.plan.plan(),
            [&work.inputs[0], &work.inputs[1]],
            &self.compiler,
        );
        match result {
            Ok(result) => {
                work.terminal = true;
                let words = logical_output(work.plan.plan(), &result.output)
                    .map_err(SubmitError::FailedQuiescent)?;
                work.response = Some(
                    response_packet(&work.packet, &words).map_err(SubmitError::FailedQuiescent)?,
                );
                Ok(())
            }
            Err(error) => {
                // Error names cannot establish quiescence. Reap actual fences.
                self.device.reap();
                if self.device.is_poisoned() {
                    Err(SubmitError::InFlightUnknown(error))
                } else {
                    work.terminal = true;
                    Err(SubmitError::FailedQuiescent(error))
                }
            }
        }
    }
    fn complete(&mut self, work: &mut Work) -> Result<bool, DeviceError> {
        if work.terminal {
            return Ok(true);
        }
        self.device.reap();
        Ok(!self.device.is_poisoned())
    }
    fn read(&mut self, work: &mut Work) -> Result<Vec<u8>, DeviceError> {
        if !work.terminal {
            return Err(DeviceError::Poisoned);
        }
        work.response.take().ok_or(DeviceError::InvalidHandle)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    const SOURCE:&[u8]=b"fn mat(a:read Array<f32>,b:read Array<f32>,rows:u32,inner:u32,cols:u32)->Array<f32>{var c=filled_f32(rows*cols,0.0);for i in 0u32..rows{for j in 0u32..cols{var sum:f32=0.0;for k in 0u32..inner{sum=sum+a[i*inner+k]*b[k*cols+j];}c[i*cols+j]=sum;}}return c;}";
    #[test]
    fn source_packet_reverification_binds_source_key_dimensions_generation_and_input_bytes() {
        let caps = Capabilities::cuda_12_9_sm89();
        let plan = crate::tensor::emit_cuda(SOURCE, "mat", [2, 2, 2], [4, 4], caps).unwrap();
        let a = [F32Bits::POS_ZERO; 4];
        let b = [F32Bits::POS_ZERO; 4];
        let packet = super::super::request(&plan, [&a, &b]).unwrap();
        let work = prepare_packet(SOURCE, "mat", caps, &packet).unwrap();
        assert_eq!(work.inputs, [vec![0; 4], vec![0; 4]]);
        assert!(work.response.is_none());
        assert!(!work.terminal);
        for at in [4, 16, 32, 40, 104, 112, 144] {
            let mut changed = packet.clone();
            changed[at] ^= 1;
            assert!(
                prepare_packet(SOURCE, "mat", caps, &changed).is_err(),
                "mutation at{at}"
            );
        }
        // A distinct nonzero generation, independent of the process counter.
        let generation = u64::from_le_bytes(packet[104..112].try_into().unwrap());
        let mut changed = packet.clone();
        changed[104..112].copy_from_slice(&generation.checked_add(1).unwrap_or(1).to_le_bytes());
        assert!(prepare_packet(SOURCE, "mat", caps, &changed).is_err());
        let changed = String::from_utf8(SOURCE.to_vec())
            .unwrap()
            .replace("sum+a", "sum-a");
        assert!(prepare_packet(changed.as_bytes(), "mat", caps, &packet).is_err());
        assert!(
            prepare_packet(
                SOURCE,
                "mat",
                Capabilities::Cuda {
                    major: 12,
                    minor: 4,
                    sm: 86
                },
                &packet
            )
            .is_err()
        );
    }
    #[test]
    fn physical_output_publication_selects_only_verified_logical_region_and_empty_domain() {
        let caps = Capabilities::cuda_12_9_sm89();
        let emitted = crate::tensor::emit_cuda(SOURCE, "mat", [2, 2, 2], [4, 4], caps).unwrap();
        let mut ir = emitted.plan().ir().clone();
        ir.output.offset = 3;
        ir.output.elements = 11;
        let origin = invar_device::Provenance {
            source_identity: ir.source_identity,
            graph_revision: ir.graph_revision,
        };
        let plan = invar_device::floating::verify(ir, caps, origin).unwrap();
        let physical = [0, 0, 0, 1, 2, 3, 4, 0, 0, 0, 0];
        assert_eq!(logical_output(&plan, &physical).unwrap(), vec![1, 2, 3, 4]);
        assert!(logical_output(&plan, &physical[..10]).is_err());
        let empty = crate::tensor::emit_cuda(SOURCE, "mat", [0, 2, 3], [0, 6], caps).unwrap();
        assert!(logical_output(empty.plan(), &[]).unwrap().is_empty());
    }
}
