//! Target-independent owned-array storage ABI. Handles are allocation IDs,
//! never raw addresses; compaction updates descriptors and preserves all loans.
use super::Error;
use crate::{front::Type, mir};

pub const OUTPUT_WORDS: usize = 10;
pub const CLEANUP_WORDS: usize = 4;
pub const DESCRIPTOR_WORDS: usize = 4;
pub const KERNEL_NAME: &str = "invar_gnet_owned_v2";

#[derive(Clone, Copy, Debug)]
pub struct OwnedLimits {
    pub elements: usize,
    pub allocations: usize,
    pub cleanup_events: usize,
}
impl Default for OwnedLimits {
    fn default() -> Self {
        Self {
            elements: 65_536,
            allocations: 1024,
            cleanup_events: 10_000,
        }
    }
}

#[derive(Clone, Debug)]
pub struct OwnedLayout {
    elements: usize,
    allocations: usize,
    cleanup_events: usize,
    heap_words: usize,
}
impl OwnedLayout {
    pub(super) fn new(limits: OwnedLimits) -> Result<Self, Error> {
        if limits.elements > crate::array_limits::MAX_LIVE_ELEMENTS
            || limits.allocations == 0
            || limits.allocations > 10_000
            || limits.cleanup_events > 100_000
        {
            return Err(Error::Limit);
        }
        let heap_words = limits
            .allocations
            .checked_mul(DESCRIPTOR_WORDS)
            .and_then(|n| n.checked_add(limits.elements))
            .ok_or(Error::Limit)?;
        Ok(Self {
            elements: limits.elements,
            allocations: limits.allocations,
            cleanup_events: limits.cleanup_events,
            heap_words,
        })
    }
    pub fn elements(&self) -> usize {
        self.elements
    }
    pub fn allocations(&self) -> usize {
        self.allocations
    }
    pub fn cleanup_events(&self) -> usize {
        self.cleanup_events
    }
    pub fn heap_words(&self) -> usize {
        self.heap_words
    }
    pub fn cleanup_words(&self) -> usize {
        self.cleanup_events * CLEANUP_WORDS
    }
    pub fn storage_bytes(&self) -> usize {
        (self.heap_words + self.cleanup_words()) * 8
    }
    pub fn output_words(&self) -> usize {
        OUTPUT_WORDS
    }
    pub fn kernel_name(&self) -> &'static str {
        KERNEL_NAME
    }
}

pub(super) fn array(ty: &Type) -> bool {
    matches!(ty, Type::ArrayI32 | Type::ArrayU32)
}
pub(super) fn operation(op: &mir::Operation) -> bool {
    matches!(
        op,
        mir::Operation::Move(_)
            | mir::Operation::ArrayRef { .. }
            | mir::Operation::Array(_)
            | mir::Operation::Index { .. }
            | mir::Operation::Length(_)
            | mir::Operation::StoreIndex { .. }
    )
}
