//! Stable four-pass byte radix core, also embedded in standalone native hosts.
//! No interpreter, driver, unsafe code, or third-party runtime is needed.
use std::ops::Range;

pub const MAX_ELEMENTS: usize = 1 << 28;
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum KeyKind {
    U32,
    I32,
    F32,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Error {
    LengthMismatch,
    Offset { index: usize },
    Limit,
    Allocation,
    Workers,
    WorkerFailure(&'static str),
    Internal,
}

pub fn reserve<T>(len: usize) -> Result<Vec<T>, Error> {
    let mut out = Vec::new();
    out.try_reserve_exact(len).map_err(|_| Error::Allocation)?;
    Ok(out)
}
pub fn canonical(bits: u32, kind: KeyKind) -> u32 {
    if kind == KeyKind::F32 && bits & 0x7f800000 == 0x7f800000 && bits & 0x007fffff != 0 {
        0x7fc00000
    } else {
        bits
    }
}
fn radix_key(bits: u32, kind: KeyKind) -> u32 {
    let bits = canonical(bits, kind);
    match kind {
        KeyKind::U32 => bits,
        KeyKind::I32 => bits ^ 0x80000000,
        KeyKind::F32 if bits & 0x80000000 != 0 => !bits,
        KeyKind::F32 => bits ^ 0x80000000,
    }
}
pub fn ranges(len: usize, offsets: Option<&[u32]>) -> Result<Vec<Range<usize>>, Error> {
    if len > MAX_ELEMENTS || offsets.is_some_and(|v| v.len() > MAX_ELEMENTS) {
        return Err(Error::Limit);
    }
    let mut out = reserve(offsets.map_or(1, |v| v.len().saturating_sub(1)))?;
    if let Some(offsets) = offsets {
        if offsets.first() != Some(&0) {
            return Err(Error::Offset { index: 0 });
        }
        let mut previous = 0;
        for (index, &offset) in offsets.iter().enumerate() {
            if offset < previous || u64::from(offset) > len as u64 {
                return Err(Error::Offset { index });
            }
            previous = offset;
        }
        if offsets.last().copied().map(u64::from) != Some(len as u64) {
            return Err(Error::Offset {
                index: offsets.len() - 1,
            });
        }
        out.extend(offsets.windows(2).map(|v| v[0] as usize..v[1] as usize));
    } else {
        out.push(0..len);
    }
    Ok(out)
}
/// One stable byte pass. Input and output have equal, independently owned lengths.
pub fn byte_pass(
    input: &[[u32; 2]],
    output: &mut [[u32; 2]],
    shift: u32,
) -> Result<[usize; 256], Error> {
    if input.len() != output.len() || ![0, 8, 16, 24].contains(&shift) {
        return Err(Error::Internal);
    }
    let mut counts = [0usize; 256];
    for pair in input {
        counts[((pair[0] >> shift) & 255) as usize] += 1;
    }
    let mut next = [0usize; 256];
    let mut sum = 0;
    for (index, &count) in counts.iter().enumerate() {
        next[index] = sum;
        sum += count;
    }
    for &pair in input {
        let digit = ((pair[0] >> shift) & 255) as usize;
        *output.get_mut(next[digit]).ok_or(Error::Internal)? = pair;
        next[digit] += 1;
    }
    Ok(counts)
}
pub fn serial_pairs(pairs: &mut [[u32; 2]]) -> Result<(), Error> {
    let mut scratch = reserve(pairs.len())?;
    scratch.resize(pairs.len(), [0; 2]);
    for shift in [0, 8, 16, 24] {
        byte_pass(pairs, &mut scratch, shift)?;
        pairs.copy_from_slice(&scratch);
    }
    Ok(())
}
pub fn order_with(
    keys: &[u32],
    kind: KeyKind,
    offsets: Option<&[u32]>,
    sorter: impl Fn(&mut [[u32; 2]]) -> Result<(), Error>,
) -> Result<Vec<u32>, Error> {
    let ranges = ranges(keys.len(), offsets)?;
    let mut pairs = reserve(keys.len())?;
    pairs.extend(
        keys.iter()
            .enumerate()
            .map(|(i, &key)| [radix_key(key, kind), i as u32]),
    );
    for range in ranges {
        sorter(&mut pairs[range])?;
    }
    let mut order = reserve(keys.len())?;
    order.extend(pairs.iter().map(|v| v[1]));
    Ok(order)
}
pub fn sort_with(
    keys: &mut [u32],
    payload: Option<&mut [u32]>,
    kind: KeyKind,
    offsets: Option<&[u32]>,
    sorter: impl Fn(&mut [[u32; 2]]) -> Result<(), Error>,
) -> Result<(), Error> {
    if payload.as_ref().is_some_and(|v| v.len() != keys.len()) {
        return Err(Error::LengthMismatch);
    }
    let ranges = ranges(keys.len(), offsets)?;
    let order = order_with(keys, kind, offsets, sorter)?;
    let mut sorted = reserve(keys.len())?;
    for &i in &order {
        sorted.push(*keys.get(i as usize).ok_or(Error::Internal)?);
    }
    for range in ranges {
        for key in &mut sorted[range] {
            *key = canonical(*key, kind);
        }
    }
    let sorted_payload = payload
        .as_ref()
        .map(|input| {
            let mut output = reserve(input.len())?;
            for &i in &order {
                output.push(*input.get(i as usize).ok_or(Error::Internal)?);
            }
            Ok::<_, Error>(output)
        })
        .transpose()?;
    keys.copy_from_slice(&sorted);
    if let (Some(payload), Some(output)) = (payload, sorted_payload) {
        payload.copy_from_slice(&output);
    }
    Ok(())
}
