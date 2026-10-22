//! Safe metadata policy, shared with the root issue test without a Python link dependency.

pub const MAX_ELEMENTS: usize = 1_000_000;
pub const MAX_BYTES: usize = 1_048_576;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Dtype {
    U8,
    I32,
    U32,
    F32,
    F64,
    /// NumPy `bool_`: one byte per element, each exactly 0 or 1 (HA-1.4).
    Bool,
}

impl Dtype {
    pub fn parse(name: &str) -> Option<Self> {
        match name {
            "uint8" => Some(Self::U8),
            "int32" => Some(Self::I32),
            "uint32" => Some(Self::U32),
            "float32" => Some(Self::F32),
            "float64" => Some(Self::F64),
            "bool" => Some(Self::Bool),
            _ => None,
        }
    }

    pub fn size(self) -> usize {
        match self {
            Self::U8 | Self::Bool => 1,
            Self::I32 | Self::U32 | Self::F32 => 4,
            Self::F64 => 8,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Metadata {
    pub dtype: Dtype,
    pub rank: usize,
    pub length: usize,
    pub contiguous: bool,
    pub aligned: bool,
    pub writable: bool,
    pub owns_data: bool,
    pub address: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Failure {
    Dtype,
    Rank,
    Shape,
    Contiguity,
    Alignment,
    Writability,
    Ownership,
    Limit,
    Alias,
    /// A `bool_` byte other than 0 or 1: not a valid `Bool`.
    BoolValue,
}

impl Failure {
    pub fn code(self) -> &'static str {
        match self {
            Self::Dtype => "PY_ARRAY_DTYPE",
            Self::Rank => "PY_ARRAY_RANK",
            Self::Shape => "PY_ARRAY_SHAPE",
            Self::Contiguity => "PY_ARRAY_CONTIGUITY",
            Self::Alignment => "PY_ARRAY_ALIGNMENT",
            Self::Writability => "PY_ARRAY_WRITABILITY",
            Self::Ownership => "PY_ARRAY_OWNERSHIP",
            Self::Limit => "PY_BUFFER_LIMIT",
            Self::Alias => "PY_BUFFER_ALIAS",
            Self::BoolValue => "PY_ARRAY_BOOL_VALUE",
        }
    }
}

/// Check metadata before creating any element reference. Only NumPy-owned
/// allocations are admitted; arbitrary externally backed arrays cannot be pinned here.
pub fn validate(
    metadata: Metadata,
    expected: Dtype,
    length: Option<usize>,
    output: bool,
) -> Result<(), Failure> {
    if metadata.dtype != expected {
        return Err(Failure::Dtype);
    }
    if metadata.rank != 1 {
        return Err(Failure::Rank);
    }
    if length.is_some_and(|length| length != metadata.length) {
        return Err(Failure::Shape);
    }
    if !metadata.contiguous {
        return Err(Failure::Contiguity);
    }
    if !metadata.aligned || !metadata.address.is_multiple_of(expected.size()) {
        return Err(Failure::Alignment);
    }
    if output && !metadata.writable {
        return Err(Failure::Writability);
    }
    let maximum = if expected == Dtype::U8 {
        MAX_BYTES
    } else {
        MAX_ELEMENTS
    };
    if metadata.length > maximum
        || metadata
            .length
            .checked_mul(expected.size())
            .and_then(|bytes| metadata.address.checked_add(bytes))
            .is_none()
    {
        return Err(Failure::Limit);
    }
    if !metadata.owns_data {
        return Err(Failure::Ownership);
    }
    Ok(())
}

/// Half-open extents: empty buffers never alias. Read/read overlap is permitted.
pub fn overlaps(left: Metadata, right: Metadata) -> Result<bool, Failure> {
    let end = |value: Metadata| {
        value
            .length
            .checked_mul(value.dtype.size())
            .and_then(|bytes| value.address.checked_add(bytes))
            .ok_or(Failure::Limit)
    };
    Ok(left.length != 0
        && right.length != 0
        && left.address < end(right)?
        && right.address < end(left)?)
}

/// Whether every byte of a `bool_` buffer is 0 or 1. Run this before any `&[bool]` is formed over the memory:
/// another byte value is not a valid Rust `bool`, and NumPy lets a `uint8` view write one.
pub fn bool_bytes_valid(bytes: &[u8]) -> bool {
    bytes.iter().all(|byte| *byte <= 1)
}
