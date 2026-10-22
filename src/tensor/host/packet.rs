//! Device-adapter halves of the private `itf1` request protocol shared by the
//! CPU, CUDA and Metal tensor adapters: header/digest validation, input
//! decoding, logical publication and the response packet.
use super::HEADER;
use invar_cuda::Error;
use sha2::{Digest, Sha256};

/// Offset of the 32-byte digest field; every header byte before it is bound.
const DIGEST: usize = 112;

/// SHA-256 over the header before the digest field (magic, dimensions,
/// lengths, artifact key and generation) followed by the payload words.
/// `bytes` must hold at least a full header.
pub(crate) fn packet_digest(bytes: &[u8]) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(&bytes[..DIGEST]);
    hasher.update(&bytes[HEADER..]);
    hasher.finalize().into()
}

/// Validate the header, lengths and header-plus-payload digest of an
/// untrusted packet. Returns the dimensions and the A/B/C lengths in elements.
pub(crate) fn packet_shape(bytes: &[u8]) -> Result<([u32; 3], [usize; 3]), Error> {
    if bytes.get(..4) != Some(b"itf1")
        || bytes.len() < HEADER
        || bytes.len() > invar_cuda::MAX_TRANSFER_BYTES
    {
        return Err(Error::InvalidHandle);
    }
    let u32_at = |i| u32::from_le_bytes(bytes[i..i + 4].try_into().expect("bounded header"));
    let u64_at = |i| u64::from_le_bytes(bytes[i..i + 8].try_into().expect("bounded header"));
    let dimensions = [u32_at(4), u32_at(8), u32_at(12)];
    let lengths = [u64_at(16), u64_at(24), u64_at(32)].map(|n| usize::try_from(n).ok());
    let [Some(a), Some(b), Some(c)] = lengths else {
        return Err(Error::Limit);
    };
    if [a, b, c].iter().any(|n| *n > crate::eval::MAX_ARRAY_LEN)
        || HEADER.checked_add(
            a.checked_add(b)
                .and_then(|n| n.checked_mul(4))
                .ok_or(Error::Limit)?,
        ) != Some(bytes.len())
        || u64_at(104) == 0
        || packet_digest(bytes) != bytes[DIGEST..HEADER]
    {
        return Err(Error::InvalidHandle);
    }
    Ok((dimensions, [a, b, c]))
}

/// Privately copy the A and B words that follow a validated header.
#[cfg(any(test, feature = "cuda", all(feature = "metal", target_os = "macos")))]
pub(crate) fn packet_inputs(bytes: &[u8], lengths: [usize; 2]) -> Result<[Vec<u32>; 2], Error> {
    let mut inputs: [Vec<u32>; 2] = std::array::from_fn(|_| Vec::new());
    let mut at = HEADER;
    for (out, n) in inputs.iter_mut().zip(lengths) {
        let end = n
            .checked_mul(4)
            .and_then(|bytes| at.checked_add(bytes))
            .ok_or(Error::Limit)?;
        let words = bytes.get(at..end).ok_or(Error::InvalidHandle)?;
        out.try_reserve_exact(n).map_err(|_| Error::OutOfMemory)?;
        for word in words.chunks_exact(4) {
            out.push(u32::from_le_bytes(word.try_into().expect("four bytes")));
        }
        at = end;
    }
    Ok(inputs)
}

/// Copy only verified logical C words. Physical fresh padding is never source output.
#[cfg(any(test, feature = "cuda", all(feature = "metal", target_os = "macos")))]
pub(crate) fn logical_output(
    plan: &invar_device::floating::Verified,
    physical: &[u32],
) -> Result<Vec<u32>, Error> {
    let out = &plan.ir().output;
    if physical.len() as u64 != out.elements {
        return Err(Error::InvalidHandle);
    }
    let [rows, _, cols] = plan.ir().dimensions;
    let count = (rows as usize)
        .checked_mul(cols as usize)
        .ok_or(Error::Limit)?;
    let mut words = Vec::new();
    words
        .try_reserve_exact(count)
        .map_err(|_| Error::OutOfMemory)?;
    for row in 0..rows {
        for col in 0..cols {
            let index = out
                .offset
                .checked_add(
                    u64::from(row)
                        .checked_mul(out.strides[0])
                        .ok_or(Error::Limit)?,
                )
                .and_then(|n| n.checked_add(u64::from(col).checked_mul(out.strides[1])?))
                .and_then(|n| usize::try_from(n).ok())
                .ok_or(Error::Limit)?;
            words.push(*physical.get(index).ok_or(Error::InvalidHandle)?);
        }
    }
    Ok(words)
}

/// The request header followed by the logical output words.
#[cfg(any(test, feature = "cuda", all(feature = "metal", target_os = "macos")))]
pub(crate) fn response_packet(packet: &[u8], words: &[u32]) -> Result<Vec<u8>, Error> {
    let header = packet.get(..HEADER).ok_or(Error::InvalidHandle)?;
    let mut response = Vec::new();
    response
        .try_reserve_exact(HEADER + words.len() * 4)
        .map_err(|_| Error::OutOfMemory)?;
    response.extend_from_slice(header);
    for word in words {
        response.extend_from_slice(&word.to_le_bytes());
    }
    Ok(response)
}

#[cfg(test)]
mod tests {
    use super::super::{CpuDevice, request, response};
    use super::*;
    use crate::numeric_contract::F32Bits;
    use invar_cuda::Device;
    const SOURCE:&[u8]=b"fn mat(a:read Array<f32>,b:read Array<f32>,rows:u32,inner:u32,cols:u32)->Array<f32>{var c=filled_f32(rows*cols,0.0);for i in 0u32..rows{for j in 0u32..cols{var sum:f32=0.0;for k in 0u32..inner{sum=sum+a[i*inner+k]*b[k*cols+j];}c[i*cols+j]=sum;}}return c;}";

    /// A fresh 2x2x2 request of ones; each call takes a new generation.
    fn ones() -> Vec<u8> {
        let caps = invar_device::Capabilities::cuda_12_9_sm89();
        let plan = crate::tensor::emit(SOURCE, "mat", [2, 2, 2], [4, 4], caps).unwrap();
        let one = [F32Bits::from_bits(1.0f32.to_bits()); 4];
        request(&plan, [&one, &one]).unwrap()
    }
    fn generation(bytes: &[u8]) -> u64 {
        u64::from_le_bytes(bytes[104..112].try_into().unwrap())
    }
    /// A generation distinct from `bytes`' own and nonzero, whatever the
    /// process-wide counter has reached.
    fn other_generation(bytes: &[u8]) -> u64 {
        generation(bytes).checked_add(1).unwrap_or(1)
    }

    #[test]
    fn digest_binds_every_header_field_and_a_distinct_nonzero_generation() {
        let packet = ones();
        assert!(packet_shape(&packet).is_ok());
        assert!(CpuDevice.prepare(&packet).is_ok());
        // Dimensions, A/B/C lengths, key, generation, digest and payload.
        for at in [4, 8, 12, 16, 24, 32, 40, 103, 104, 111, 112, 143, 144, 159] {
            let mut changed = packet.clone();
            changed[at] ^= 1;
            assert!(packet_shape(&changed).is_err(), "mutation at {at}");
            assert!(CpuDevice.prepare(&changed).is_err(), "CPU mutation at {at}");
        }
        let mut changed = packet.clone();
        changed[104..112].copy_from_slice(&other_generation(&packet).to_le_bytes());
        assert_ne!(generation(&changed), 0);
        assert!(packet_shape(&changed).is_err());
        assert!(CpuDevice.prepare(&changed).is_err());
        // Resealed, the same bytes validate again: the digest is the binding.
        let digest = packet_digest(&changed);
        changed[DIGEST..HEADER].copy_from_slice(&digest);
        assert!(packet_shape(&changed).is_ok());
    }

    #[test]
    fn host_refuses_a_response_with_a_mismatched_generation() {
        let request = ones();
        let mut cpu = CpuDevice;
        let mut work = cpu.prepare(&request).unwrap();
        cpu.submit(&mut work).unwrap();
        let reply = cpu.read(&mut work).unwrap();
        let values = response(&reply, &request, 4).unwrap();
        let bits: Vec<u32> = values.iter().map(|v| v.to_bits()).collect();
        assert_eq!(bits, vec![2.0f32.to_bits(); 4]);
        let mut stale = reply.clone();
        stale[104..112].copy_from_slice(&other_generation(&request).to_le_bytes());
        assert!(response(&stale, &request, 4).is_none());
        // A reply to an earlier request with the same inputs is not accepted
        // for a later one.
        let later = ones();
        assert_ne!(generation(&later), generation(&request));
        assert!(response(&reply, &later, 4).is_none());
    }
}
