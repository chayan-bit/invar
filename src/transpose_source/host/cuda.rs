//! Source-reverified private transpose packets and owning CUDA publication.
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
    /// Explicit GPU open; the pure packet tests never call this method.
    pub fn open(source: &[u8], entry: &str, compiler: &Path) -> Result<Self, DeviceError> {
        if source.len() > 1024 * 1024 {
            return Err(DeviceError::Limit);
        }
        let checked = crate::front::parse(source)
            .and_then(|p| crate::check::check(p, false))
            .map_err(|_| DeviceError::InvalidHandle)?;
        Contract::from_checked(&checked, entry).map_err(|_| DeviceError::InvalidHandle)?;
        let mut owned = Vec::new();
        owned
            .try_reserve_exact(source.len())
            .map_err(|_| DeviceError::OutOfMemory)?;
        owned.extend_from_slice(source);
        Ok(Self {
            source: owned,
            entry: entry.into(),
            compiler: compiler.into(),
            device: invar_cuda::cuda::CudaDevice::open()?,
        })
    }
    pub fn capabilities(&self) -> Capabilities {
        self.device.capabilities()
    }
}
pub struct Work {
    plan: super::super::Emission,
    packet: Vec<u8>,
    input: Vec<u32>,
    response: Option<Vec<u8>>,
    terminal: bool,
}
fn prepare_packet(
    source: &[u8],
    entry: &str,
    caps: Capabilities,
    bytes: &[u8],
) -> Result<Work, DeviceError> {
    if bytes.get(..4) != Some(b"itr1")
        || bytes.len() < HEADER
        || bytes.len() > invar_cuda::MAX_TRANSFER_BYTES
    {
        return Err(DeviceError::InvalidHandle);
    }
    let u32_at = |i| u32::from_le_bytes(bytes[i..i + 4].try_into().expect("bounded header"));
    let u64_at = |i| u64::from_le_bytes(bytes[i..i + 8].try_into().expect("bounded header"));
    let dimensions = [u32_at(4), u32_at(8)];
    let input_len = usize::try_from(u64_at(16)).map_err(|_| DeviceError::Limit)?;
    let output_len = usize::try_from(u64_at(24)).map_err(|_| DeviceError::Limit)?;
    if input_len > invar_device::transpose::MAX_LOGICAL_ELEMENTS as usize
        || output_len > invar_device::transpose::MAX_LOGICAL_ELEMENTS as usize
        || u64_at(32) != 0
        || u64_at(104) == 0
        || HEADER.checked_add(input_len.checked_mul(4).ok_or(DeviceError::Limit)?)
            != Some(bytes.len())
        || Sha256::digest(&bytes[HEADER..]).as_slice() != &bytes[112..144]
    {
        return Err(DeviceError::InvalidHandle);
    }
    let checked = crate::front::parse(source)
        .and_then(|p| crate::check::check(p, false))
        .map_err(|_| DeviceError::InvalidHandle)?;
    let contract =
        Contract::from_checked(&checked, entry).map_err(|_| DeviceError::InvalidHandle)?;
    let plan = contract
        .emit_cuda(dimensions, input_len as u64, caps)
        .map_err(|_| DeviceError::InvalidHandle)?;
    let mut key = Vec::with_capacity(64);
    for byte in artifact_key(source, &contract, &plan) {
        key.extend_from_slice(format!("{byte:02x}").as_bytes());
    }
    if u32_at(12) != u32::from(contract.element() == Type::F32)
        || plan.plan().ir().output.elements != output_len as u64
        || key != bytes[40..104]
    {
        return Err(DeviceError::StaleBorrow);
    }
    let mut input = Vec::new();
    input
        .try_reserve_exact(input_len)
        .map_err(|_| DeviceError::OutOfMemory)?;
    for word in bytes[HEADER..].chunks_exact(4) {
        input.push(u32::from_le_bytes(word.try_into().expect("four bytes")));
    }
    let mut packet = Vec::new();
    packet
        .try_reserve_exact(bytes.len())
        .map_err(|_| DeviceError::OutOfMemory)?;
    packet.extend_from_slice(bytes);
    Ok(Work {
        plan,
        packet,
        input,
        response: None,
        terminal: false,
    })
}
fn logical_output(
    plan: &invar_device::transpose::Verified,
    physical: &[u32],
) -> Result<Vec<u32>, DeviceError> {
    let out = &plan.ir().output;
    if physical.len() as u64 != out.elements {
        return Err(DeviceError::InvalidHandle);
    }
    let [rows, cols] = plan.ir().dimensions;
    let count = (rows as usize)
        .checked_mul(cols as usize)
        .ok_or(DeviceError::Limit)?;
    let mut words = Vec::new();
    words
        .try_reserve_exact(count)
        .map_err(|_| DeviceError::OutOfMemory)?;
    // Source fresh output is flattened by its transposed [cols,rows] shape.
    for col in 0..cols {
        for row in 0..rows {
            let index = out
                .offset
                .checked_add(
                    u64::from(col)
                        .checked_mul(out.strides[0])
                        .ok_or(DeviceError::Limit)?,
                )
                .and_then(|n| n.checked_add(u64::from(row).checked_mul(out.strides[1])?))
                .and_then(|n| usize::try_from(n).ok())
                .ok_or(DeviceError::Limit)?;
            words.push(*physical.get(index).ok_or(DeviceError::InvalidHandle)?);
        }
    }
    Ok(words)
}
impl Device for Cuda {
    type Work = Work;
    fn prepare(&mut self, bytes: &[u8]) -> Result<Work, DeviceError> {
        self.device.reap();
        if self.device.is_poisoned() {
            return Err(DeviceError::Poisoned);
        }
        prepare_packet(&self.source, &self.entry, self.capabilities(), bytes)
    }
    fn submit(&mut self, work: &mut Work) -> Result<(), SubmitError> {
        if work.terminal || work.response.is_some() {
            return Err(SubmitError::FailedQuiescent(DeviceError::InvalidHandle));
        }
        match self
            .device
            .run_transpose(work.plan.plan(), &work.input, &self.compiler)
        {
            Ok(result) => {
                work.terminal = true;
                let words = logical_output(work.plan.plan(), &result.output)
                    .map_err(SubmitError::FailedQuiescent)?;
                let mut response = Vec::new();
                response
                    .try_reserve_exact(HEADER + words.len() * 4)
                    .map_err(|_| SubmitError::FailedQuiescent(DeviceError::OutOfMemory))?;
                response.extend_from_slice(&work.packet[..HEADER]);
                for word in words {
                    response.extend_from_slice(&word.to_le_bytes());
                }
                work.response = Some(response);
                Ok(())
            }
            Err(error) => {
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
    const SOURCE:&[u8]=b"fn transpose(a:read Array<u32>,rows:u32,cols:u32)->Array<u32>{var b=filled_u32(rows*cols,0u32);for r in 0u32..rows{for c in 0u32..cols{b[c*rows+r]=a[r*cols+c];}}return b;}";
    fn contract(source: &[u8]) -> Contract {
        let checked = crate::front::parse(source)
            .and_then(|p| crate::check::check(p, false))
            .unwrap();
        Contract::from_checked(&checked, "transpose").unwrap()
    }
    #[test]
    fn packets_bind_exact_source_caps_type_dimensions_generation_and_whole_input() {
        let caps = Capabilities::cuda_12_9_sm89();
        let c = contract(SOURCE);
        let plan = c.emit_cuda([2, 3], 8, caps).unwrap();
        let input = [3, 7, 11, 13, 17, u32::MAX, 123, 456];
        let bytes = request(SOURCE, &c, &plan, &input).unwrap();
        let work = prepare_packet(SOURCE, "transpose", caps, &bytes).unwrap();
        assert_eq!(work.input, input);
        assert!(work.response.is_none());
        for at in [4, 8, 12, 16, 24, 32, 40, 112, 144, 175] {
            let mut changed = bytes.clone();
            changed[at] ^= 1;
            assert!(
                prepare_packet(SOURCE, "transpose", caps, &changed).is_err(),
                "at {at}"
            );
        }
        let mut zero_generation = bytes.clone();
        zero_generation[104..112].fill(0);
        assert!(prepare_packet(SOURCE, "transpose", caps, &zero_generation).is_err());
        let changed = [SOURCE, b" "].concat();
        assert!(prepare_packet(&changed, "transpose", caps, &bytes).is_err());
        assert!(
            prepare_packet(
                SOURCE,
                "transpose",
                Capabilities::Cuda {
                    major: 12,
                    minor: 4,
                    sm: 86
                },
                &bytes
            )
            .is_err()
        );
        let floating = String::from_utf8(SOURCE.to_vec())
            .unwrap()
            .replace("u32>", "f32>")
            .replace("filled_u32", "filled_f32")
            .replace("0u32);", "0.0);");
        let c = contract(floating.as_bytes());
        let plan = c.emit_cuda([2, 3], 8, caps).unwrap();
        let input = [
            0, 0x80000000, 0x7f800000, 0x7fc00001, 0xff800000, 0x3f800000, 123, 456,
        ];
        let bytes = request(floating.as_bytes(), &c, &plan, &input).unwrap();
        assert_eq!(
            prepare_packet(floating.as_bytes(), "transpose", caps, &bytes)
                .unwrap()
                .input,
            input
        );
    }
    #[test]
    fn logical_publication_selects_transposed_offsets_strides_and_empty_domains() {
        let caps = Capabilities::cuda_12_9_sm89();
        let c = contract(SOURCE);
        let emitted = c.emit_cuda([2, 3], 6, caps).unwrap();
        let mut ir = emitted.plan().ir().clone();
        ir.output.offset = 2;
        ir.output.elements = 10;
        let origin = invar_device::Provenance {
            source_identity: ir.source_identity,
            graph_revision: ir.graph_revision,
        };
        let plan = invar_device::transpose::verify(ir, caps, origin).unwrap();
        let words = [99, 99, 3, 13, 7, 17, 11, u32::MAX, 99, 99];
        assert_eq!(
            logical_output(&plan, &words).unwrap(),
            vec![3, 13, 7, 17, 11, u32::MAX]
        );
        assert!(logical_output(&plan, &words[..9]).is_err());
        let mut unsupported = plan.ir().clone();
        unsupported.output.strides = [3, 1];
        assert!(invar_device::transpose::verify(unsupported, caps, origin).is_err());
        let empty = c.emit_cuda([0, 3], 0, caps).unwrap();
        assert!(logical_output(empty.plan(), &[]).unwrap().is_empty());
    }
}
