//! Checked, bounded logical-to-byte maps for the M4 layout contract.
//!
//! This module validates metadata only. It grants neither ownership nor a
//! pointer-lifetime guarantee and does not establish write disjointness.

use std::ops::Range;

pub const VERSION: u32 = 1;

pub const MAX_RANK: usize = 4;
pub const MAX_EXTENT: u64 = (1_u64 << 31) - 1;
pub const MAX_BUFFER_BYTES: u64 = 1_u64 << 31;

/// Frozen device element widths, without numeric-policy implications.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LayoutElement {
    U32,
    F32,
    F16,
}

impl LayoutElement {
    pub const fn bytes(self) -> u64 {
        match self {
            Self::U32 | Self::F32 => 4,
            Self::F16 => 2,
        }
    }
}

/// Metadata validation failures; these are not source-language trap codes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LayoutError {
    RankLimit,
    RankMismatch,
    ExtentLimit,
    BufferLimit,
    ArithmeticOverflow,
    InvalidAlignment,
    Misaligned,
    PhysicalBounds,
    IndexBounds,
    AccessWidth,
}

/// Untrusted descriptor. Strides and offset are measured in elements.
pub struct LayoutSpec<'a> {
    pub element: LayoutElement,
    pub shape: &'a [u64],
    pub strides: &'a [u64],
    pub offset: u64,
    pub allocation_bytes: u64,
    pub base_alignment: u64,
}

/// A validated map. Fields are private so validation cannot be bypassed.
#[derive(Debug, PartialEq, Eq)]
pub struct CheckedLayout {
    element: LayoutElement,
    rank: usize,
    shape: [u64; MAX_RANK],
    strides: [u64; MAX_RANK],
    origin_bytes: u64,
    allocation_bytes: u64,
    logical_elements: u64,
}

impl LayoutSpec<'_> {
    /// Validate in O(rank), using fixed storage and no pointer operations.
    ///
    /// An empty domain permits a one-past origin but never an element access.
    /// Nonnegative strides make the last corner the maximum physical address.
    pub fn verify(&self) -> Result<CheckedLayout, LayoutError> {
        let rank = self.shape.len();
        if rank > MAX_RANK {
            return Err(LayoutError::RankLimit);
        }
        if self.strides.len() != rank {
            return Err(LayoutError::RankMismatch);
        }
        if self.shape.iter().any(|&extent| extent > MAX_EXTENT) {
            return Err(LayoutError::ExtentLimit);
        }
        if self.allocation_bytes > MAX_BUFFER_BYTES {
            return Err(LayoutError::BufferLimit);
        }
        if !self.base_alignment.is_power_of_two() {
            return Err(LayoutError::InvalidAlignment);
        }
        let width = self.element.bytes();
        if self.base_alignment < width {
            return Err(LayoutError::Misaligned);
        }
        let origin_bytes = self
            .offset
            .checked_mul(width)
            .ok_or(LayoutError::ArithmeticOverflow)?;
        if origin_bytes > self.allocation_bytes {
            return Err(LayoutError::PhysicalBounds);
        }
        let empty = self.shape.contains(&0);
        let mut logical_elements = if empty { 0 } else { 1_u64 };
        let mut last_element = self.offset;
        if !empty {
            for (&extent, &stride) in self.shape.iter().zip(self.strides) {
                logical_elements = logical_elements
                    .checked_mul(extent)
                    .ok_or(LayoutError::ArithmeticOverflow)?;
                let delta = (extent - 1)
                    .checked_mul(stride)
                    .ok_or(LayoutError::ArithmeticOverflow)?;
                last_element = last_element
                    .checked_add(delta)
                    .ok_or(LayoutError::ArithmeticOverflow)?;
            }
            let logical_bytes = logical_elements
                .checked_mul(width)
                .ok_or(LayoutError::ArithmeticOverflow)?;
            if logical_bytes > MAX_BUFFER_BYTES {
                return Err(LayoutError::BufferLimit);
            }
            let end = last_element
                .checked_add(1)
                .and_then(|value| value.checked_mul(width))
                .ok_or(LayoutError::ArithmeticOverflow)?;
            if end > self.allocation_bytes {
                return Err(LayoutError::PhysicalBounds);
            }
        }
        let mut shape = [0; MAX_RANK];
        let mut strides = [0; MAX_RANK];
        // rank was bounded above; slices have matching validated lengths.
        shape[..rank].copy_from_slice(self.shape);
        strides[..rank].copy_from_slice(self.strides);
        Ok(CheckedLayout {
            element: self.element,
            rank,
            shape,
            strides,
            origin_bytes,
            allocation_bytes: self.allocation_bytes,
            logical_elements,
        })
    }
}

impl CheckedLayout {
    pub fn logical_elements(&self) -> u64 {
        self.logical_elements
    }

    pub fn shape(&self) -> &[u64] {
        &self.shape[..self.rank]
    }

    /// Return exactly one logical element's half-open byte range in O(rank).
    ///
    /// A backend asking for a widened load is rejected even if adjacent bytes
    /// happen to be allocated. Vector access requires a separate future proof.
    pub fn access(&self, indices: &[u64], width: u64) -> Result<Range<u64>, LayoutError> {
        if width != self.element.bytes() {
            return Err(LayoutError::AccessWidth);
        }
        if indices.len() != self.rank {
            return Err(LayoutError::RankMismatch);
        }
        if self.logical_elements == 0 {
            return Err(LayoutError::IndexBounds);
        }
        let mut start = self.origin_bytes;
        for ((&index, &extent), &stride) in indices
            .iter()
            .zip(self.shape())
            .zip(&self.strides[..self.rank])
        {
            if index >= extent {
                return Err(LayoutError::IndexBounds);
            }
            let delta = index
                .checked_mul(stride)
                .and_then(|value| value.checked_mul(width))
                .ok_or(LayoutError::ArithmeticOverflow)?;
            start = start
                .checked_add(delta)
                .ok_or(LayoutError::ArithmeticOverflow)?;
        }
        let end = start
            .checked_add(width)
            .ok_or(LayoutError::ArithmeticOverflow)?;
        if end > self.allocation_bytes {
            return Err(LayoutError::PhysicalBounds);
        }
        Ok(start..end)
    }
}
