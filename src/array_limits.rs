//! Language allocation limits, separate from per-plan budgets and live-device
//! admission. The live ceiling matches the #123 2^28-element device policy.
pub const MAX_ARRAY_LEN: usize = 1 << 24;
pub const MAX_LIVE_ELEMENTS: usize = 1 << 28;
/// #123's aggregate storage ceiling. Backends must also check live capacity.
pub const MAX_DEVICE_BYTES: usize = invar_device::MAX_GNET_STORAGE_BYTES;
