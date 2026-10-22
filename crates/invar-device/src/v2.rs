//! Canonical physical metadata for legacy u32 and strict f32 device plans.
//! Verification proves byte maps, not ownership of a live pointer or a fence.
use crate::{Capabilities, Provenance, Span};

pub const VERSION: u32 = 2;
pub const LAYOUT_VERSION: u32 = 1;
pub const MAX_RANK: usize = 4;
pub const MAX_BYTES: u64 = 1 << 31;
pub const MAX_EXTENT: u64 = (1 << 31) - 1;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ElemType {
    U32,
    F32,
    F16,
}
impl ElemType {
    pub const fn bytes(self) -> u64 {
        match self {
            Self::U32 | Self::F32 => 4,
            Self::F16 => 2,
        }
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NumericKind {
    WrappingU32,
    StrictF32,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NumericPolicy {
    pub version: u32,
    pub kind: NumericKind,
}
impl NumericPolicy {
    pub const STRICT_F32: Self = Self {
        version: 1,
        kind: NumericKind::StrictF32,
    };
    pub const WRAPPING_U32: Self = Self {
        version: 1,
        kind: NumericKind::WrappingU32,
    };
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Layout {
    Strided {
        shape: Vec<u64>,
        strides_in_elements: Vec<u64>,
        offset: u64,
    },
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Buffer {
    pub id: u32,
    pub element: ElemType,
    pub allocation_bytes: u64,
    pub alignment: u64,
    pub writable: bool,
    pub layout: Layout,
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Ir {
    pub version: u32,
    pub layout_version: u32,
    pub numeric: NumericPolicy,
    pub span: Span,
    pub origin: Provenance,
    pub buffers: Vec<Buffer>,
    pub kernel: Kernel,
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Kernel {
    LegacyU32(Box<crate::Ir>),
    Contraction { dimensions: [u32; 3] },
    F32(crate::f32_ops::Program),
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Error {
    Version(u32),
    LayoutVersion(u32),
    Numeric,
    Source,
    Graph,
    Rank,
    Extent,
    Bounds,
    Overflow,
    Alignment,
    AccessWidth,
    Alias,
    Capability,
}
#[derive(Clone, Debug)]
pub struct Verified {
    ir: Ir,
    caps: Capabilities,
}
impl Verified {
    /// Complete admission/cache identity under the pinned Rust 1.95.0 Debug
    /// encoding; source provenance stays here rather than in generated code.
    pub fn identity(&self) -> String {
        format!("invar-device-ir-v2:{:?}:{:?}", self.caps, self.ir)
    }
    pub fn ir(&self) -> &Ir {
        &self.ir
    }
    pub fn capabilities(&self) -> Capabilities {
        self.caps
    }
    pub fn access(
        &self,
        slot: usize,
        index: &[u64],
        width: u64,
    ) -> Result<std::ops::Range<u64>, Error> {
        let b = self.ir.buffers.get(slot).ok_or(Error::Bounds)?;
        if width != b.element.bytes() {
            return Err(Error::AccessWidth);
        }
        let Layout::Strided {
            shape,
            strides_in_elements,
            offset,
        } = &b.layout;
        if index.len() != shape.len() {
            return Err(Error::Rank);
        }
        let mut position = *offset;
        for ((i, n), s) in index.iter().zip(shape).zip(strides_in_elements) {
            if i >= n {
                return Err(Error::Bounds);
            }
            position = position
                .checked_add(i.checked_mul(*s).ok_or(Error::Overflow)?)
                .ok_or(Error::Overflow)?;
        }
        let start = position.checked_mul(width).ok_or(Error::Overflow)?;
        let end = start.checked_add(width).ok_or(Error::Overflow)?;
        if end > b.allocation_bytes {
            return Err(Error::Bounds);
        }
        Ok(start..end)
    }
}

pub fn verify(ir: Ir, origin: Provenance, caps: Capabilities) -> Result<Verified, Error> {
    if ir.version != VERSION {
        return Err(Error::Version(ir.version));
    }
    if ir.layout_version != LAYOUT_VERSION {
        return Err(Error::LayoutVersion(ir.layout_version));
    }
    match &ir.kernel {
        Kernel::LegacyU32(old) => {
            if old.version != crate::IR_VERSION {
                return Err(Error::Version(old.version));
            }
            crate::verify((**old).clone(), origin, caps).map_err(|_| Error::Numeric)?;
            if legacy(old) != ir {
                return Err(Error::Graph);
            }
        }
        Kernel::Contraction {
            dimensions: [r, k, c],
        } => {
            if ir.numeric != NumericPolicy::STRICT_F32 || ir.buffers.len() != 3 {
                return Err(Error::Numeric);
            }
            for (slot, (b, shape)) in ir
                .buffers
                .iter()
                .zip([[*r, *k], [*k, *c], [*r, *c]])
                .enumerate()
            {
                let Layout::Strided { shape: actual, .. } = &b.layout;
                if b.element != ElemType::F32
                    || b.writable != (slot == 2)
                    || actual != &shape.map(u64::from)
                {
                    return Err(Error::Bounds);
                }
            }
            if u64::from(*r)
                .checked_mul(u64::from(*k))
                .and_then(|n| n.checked_mul(u64::from(*c)))
                .is_none_or(|n| n > crate::floating::MAX_MULTIPLY_ADDS)
            {
                return Err(Error::Bounds);
            }
        }
        Kernel::F32(p) => crate::f32_ops::validate(&ir, p)?,
    }
    verify_metadata(ir, origin, caps)
}

pub(crate) fn verify_metadata(
    ir: Ir,
    origin: Provenance,
    caps: Capabilities,
) -> Result<Verified, Error> {
    if ir.version != VERSION {
        return Err(Error::Version(ir.version));
    }
    if ir.layout_version != LAYOUT_VERSION {
        return Err(Error::LayoutVersion(ir.layout_version));
    }
    if ir.numeric.version != 1 {
        return Err(Error::Numeric);
    }
    if ir.span.start >= ir.span.end || ir.origin.source_identity == 0 {
        return Err(Error::Source);
    }
    if ir.origin != origin {
        return Err(Error::Graph);
    }
    match caps {
        Capabilities::Cuda {
            major: 12,
            minor,
            sm,
        } if minor >= 4 && sm >= 50 => {}
        Capabilities::Metal {
            apple_family,
            msl_major: 3,
            msl_minor: 1,
        } if apple_family >= 7 => {}
        _ => return Err(Error::Capability),
    }
    if ir.buffers.is_empty()
        || ir.buffers.len()
            > if matches!(caps, Capabilities::Metal { .. }) {
                31
            } else {
                32
            }
    {
        return Err(Error::Bounds);
    }
    for (slot, b) in ir.buffers.iter().enumerate() {
        let Layout::Strided {
            shape,
            strides_in_elements: strides,
            offset,
        } = &b.layout;
        let width = b.element.bytes();
        if shape.len() > MAX_RANK || shape.len() != strides.len() {
            return Err(Error::Rank);
        }
        if shape.iter().any(|n| *n > MAX_EXTENT) {
            return Err(Error::Extent);
        }
        if !b.alignment.is_power_of_two() || b.alignment < width {
            return Err(Error::Alignment);
        }
        if b.allocation_bytes > MAX_BYTES {
            return Err(Error::Bounds);
        }
        if offset.checked_mul(width).ok_or(Error::Overflow)? > b.allocation_bytes {
            return Err(Error::Bounds);
        }
        if !shape.contains(&0) {
            let mut elements = 1u64;
            let mut last = *offset;
            for (n, s) in shape.iter().zip(strides) {
                elements = elements.checked_mul(*n).ok_or(Error::Overflow)?;
                last = last
                    .checked_add((n - 1).checked_mul(*s).ok_or(Error::Overflow)?)
                    .ok_or(Error::Overflow)?;
            }
            if elements.checked_mul(width).ok_or(Error::Overflow)? > MAX_BYTES
                || last
                    .checked_add(1)
                    .and_then(|n| n.checked_mul(width))
                    .ok_or(Error::Overflow)?
                    > b.allocation_bytes
            {
                return Err(Error::Bounds);
            }
            // A sorted-stride separation proof admits row-major, permutations,
            // and padded writes; overlapping writes have no implicit law.
            if b.writable {
                let mut axes = shape
                    .iter()
                    .zip(strides)
                    .filter(|(n, _)| **n > 1)
                    .collect::<Vec<_>>();
                axes.sort_by_key(|(_, s)| **s);
                let mut covered = 1u64;
                for (n, s) in axes {
                    if *s < covered {
                        return Err(Error::Alias);
                    }
                    covered = covered
                        .checked_add((n - 1).checked_mul(*s).ok_or(Error::Overflow)?)
                        .ok_or(Error::Overflow)?;
                }
            }
        }
        for prior in &ir.buffers[..slot] {
            if prior.id == b.id
                && (prior.writable
                    || b.writable
                    || prior.allocation_bytes != b.allocation_bytes
                    || prior.alignment != b.alignment
                    || prior.element != b.element)
            {
                return Err(Error::Alias);
            }
        }
    }
    Ok(Verified { ir, caps })
}

/// Normalize the readable v1 format without changing its frozen u32 semantics.
pub(crate) fn legacy(ir: &crate::Ir) -> Ir {
    let buffers = if let Some(dag) = &ir.dag {
        dag.inputs
            .iter()
            .map(|b| (b, false))
            .chain(dag.outputs.iter().map(|b| (b, true)))
            .collect::<Vec<_>>()
    } else if let Some(c) = &ir.collective {
        c.inputs
            .iter()
            .map(|b| (b, false))
            .chain(std::iter::once((&ir.output, true)))
            .collect()
    } else {
        vec![(&ir.input, false), (&ir.output, true)]
    };
    Ir {
        version: VERSION,
        layout_version: LAYOUT_VERSION,
        numeric: NumericPolicy::WRAPPING_U32,
        span: ir.span,
        origin: Provenance {
            source_identity: ir.source_identity,
            graph_revision: ir.graph_revision,
        },
        kernel: Kernel::LegacyU32(Box::new(ir.clone())),
        buffers: buffers
            .into_iter()
            .map(|(b, writable)| Buffer {
                id: b.id,
                element: ElemType::U32,
                allocation_bytes: u64::from(b.elements) * 4,
                alignment: 4,
                writable,
                layout: Layout::Strided {
                    shape: vec![u64::from(b.elements)],
                    strides_in_elements: vec![1],
                    offset: 0,
                },
            })
            .collect(),
    }
}

pub(crate) fn contraction(ir: &crate::floating::Ir) -> Ir {
    Ir {
        version: VERSION,
        layout_version: ir.layout_version,
        numeric: NumericPolicy {
            version: ir.numeric_version,
            kind: NumericKind::StrictF32,
        },
        span: ir.span,
        origin: Provenance {
            source_identity: ir.source_identity,
            graph_revision: ir.graph_revision,
        },
        kernel: Kernel::Contraction {
            dimensions: ir.dimensions,
        },
        buffers: ir
            .inputs
            .iter()
            .map(|b| (b, false))
            .chain(std::iter::once((&ir.output, true)))
            .map(|(b, writable)| Buffer {
                id: b.id,
                element: ElemType::F32,
                allocation_bytes: b.elements * 4,
                alignment: b.alignment,
                writable,
                layout: Layout::Strided {
                    shape: b.shape.map(u64::from).to_vec(),
                    strides_in_elements: b.strides.to_vec(),
                    offset: b.offset,
                },
            })
            .collect(),
    }
}
