//! Affine index expansion for the device plan: bounded by the device element limit and reserved
//! fallibly, so a large admitted domain is refused as a typed error instead of aborting.
use super::Error;
use invar_device::indexed as device;

pub(super) fn expand(lo: u32, hi: u32, a: u32, b: u32) -> Result<Vec<u32>, Error> {
    let width = hi.saturating_sub(lo);
    let mut indices = Vec::new();
    if width > device::MAX_ELEMENTS || indices.try_reserve_exact(width as usize).is_err() {
        return Err(Error::Device(device::Error::Limit));
    }
    for i in lo..hi {
        indices.push(
            a.checked_mul(i)
                .and_then(|v| v.checked_add(b))
                .ok_or(Error::Binding)?,
        );
    }
    Ok(indices)
}
