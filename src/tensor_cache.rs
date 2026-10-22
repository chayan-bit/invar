//! Validated content keys for untrusted tensor cache hints.
//!
//! A key binds supplied source/proof/compiler descriptors and emitted bytes.
//! It does not authenticate a proof, check source recognition or authorize code
//! execution, aliasing, option legality, allocation ownership or a device fence.
//! Consumers must reverify current source/MIR/device admission before a hint hit.
#![forbid(unsafe_code)]

use crate::{
    layout_contract::{self, LayoutElement, LayoutError, LayoutSpec},
    numeric_contract::NumericPolicy,
};
use sha2::{Digest, Sha256};

const FORMAT_VERSION: u32 = 2;
const MAX_SOURCE_BYTES: usize = 1024 * 1024;
const MAX_EMITTED_BYTES: usize = 1024 * 1024;
const MAX_TEXT_BYTES: usize = 4096;
const MAX_OPTIONS: usize = 64;

/// Borrowed descriptors supplied by the checked source/MIR/device boundary.
/// Layout offsets/strides are in elements; allocations are measured in bytes.
pub struct KeyInput<'a> {
    pub source: &'a [u8],
    pub entry: &'a str,
    pub source_identity: u64,
    pub graph_revision: u64,
    pub function_contract: [u8; 32],
    pub numeric_policy: NumericPolicy,
    pub layout_version: u32,
    pub dimensions: [u32; 3],
    pub layouts: [LayoutSpec<'a>; 3],
    pub compiler_build: &'a str,
    pub target: &'a str,
    pub compiler_options: &'a [&'a str],
    pub emitted_source: &'a [u8],
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Error {
    InputLimit,
    UnsupportedPolicy,
    LayoutVersion(u32),
    Element { buffer: usize },
    Rank { buffer: usize },
    Shape { buffer: usize },
    Layout { buffer: usize, error: LayoutError },
    Mismatch,
}

/// Opaque canonical key. No imported hash can manufacture verified admission.
#[derive(Debug, PartialEq, Eq)]
pub struct CacheKey([u8; 32]);

fn validate(input: &KeyInput<'_>) -> Result<(), Error> {
    if input.source.len() > MAX_SOURCE_BYTES
        || input.emitted_source.len() > MAX_EMITTED_BYTES
        || input.entry.is_empty()
        || input.entry.len() > MAX_TEXT_BYTES
        || input.compiler_build.is_empty()
        || input.compiler_build.len() > MAX_TEXT_BYTES
        || input.target.is_empty()
        || input.target.len() > MAX_TEXT_BYTES
        || input.compiler_options.len() > MAX_OPTIONS
        || input
            .compiler_options
            .iter()
            .any(|s| s.len() > MAX_TEXT_BYTES)
    {
        return Err(Error::InputLimit);
    }
    if ![NumericPolicy::STRICT_F32, NumericPolicy::TC_F16_F32].contains(&input.numeric_policy) {
        return Err(Error::UnsupportedPolicy);
    }
    if input.layout_version != layout_contract::VERSION {
        return Err(Error::LayoutVersion(input.layout_version));
    }
    let [rows, inner, cols] = input.dimensions.map(u64::from);
    let shapes = [[rows, inner], [inner, cols], [rows, cols]];
    for (buffer, (layout, expected)) in input.layouts.iter().zip(shapes).enumerate() {
        if layout.element != LayoutElement::F32 {
            return Err(Error::Element { buffer });
        }
        if layout.shape.len() != 2 || layout.strides.len() != 2 {
            return Err(Error::Rank { buffer });
        }
        if layout.shape != expected {
            return Err(Error::Shape { buffer });
        }
        layout
            .verify()
            .map_err(|error| Error::Layout { buffer, error })?;
    }
    Ok(())
}

/// Length framing prevents concatenation collisions without delimiters/escaping.
/// All integers in key format2 are unsigned fixed-width little-endian values.
fn bytes(hash: &mut Sha256, value: &[u8]) {
    hash.update((value.len() as u64).to_le_bytes());
    hash.update(value);
}

impl CacheKey {
    /// Validate all descriptors, then hash canonical format2 in field order.
    /// Work is O(source + emitted bytes + options); layout checks are O(rank).
    /// This performs no allocation proportional to dimensions and no file IO.
    pub fn new(input: &KeyInput<'_>) -> Result<Self, Error> {
        validate(input)?;
        let mut hash = Sha256::new();
        bytes(&mut hash, b"invar.tensor-cache");
        hash.update(FORMAT_VERSION.to_le_bytes());
        hash.update(invar_device::v2::VERSION.to_le_bytes());
        bytes(&mut hash, input.source);
        bytes(&mut hash, input.entry.as_bytes());
        hash.update(input.source_identity.to_le_bytes());
        hash.update(input.graph_revision.to_le_bytes());
        hash.update(input.function_contract);
        hash.update(input.numeric_policy.version().to_le_bytes());
        bytes(&mut hash, input.numeric_policy.kind().as_bytes());
        bytes(&mut hash, input.numeric_policy.id().as_bytes());
        hash.update(input.layout_version.to_le_bytes());
        for dim in input.dimensions {
            hash.update(dim.to_le_bytes());
        }
        for layout in &input.layouts {
            // Validated F32 tag and width remain explicit versioned fields.
            hash.update(2_u32.to_le_bytes());
            hash.update(layout.element.bytes().to_le_bytes());
            hash.update(2_u32.to_le_bytes());
            for dim in layout.shape {
                hash.update(dim.to_le_bytes());
            }
            for stride in layout.strides {
                hash.update(stride.to_le_bytes());
            }
            hash.update(layout.offset.to_le_bytes());
            hash.update(layout.allocation_bytes.to_le_bytes());
            hash.update(layout.base_alignment.to_le_bytes());
        }
        bytes(&mut hash, input.compiler_build.as_bytes());
        bytes(&mut hash, input.target.as_bytes());
        hash.update((input.compiler_options.len() as u64).to_le_bytes());
        for option in input.compiler_options {
            bytes(&mut hash, option.as_bytes());
        }
        bytes(&mut hash, input.emitted_source);
        Ok(Self(hash.finalize().into()))
    }

    pub fn to_hex(&self) -> String {
        self.0.iter().map(|byte| format!("{byte:02x}")).collect()
    }

    /// Revalidate current inputs before comparing the key. An invalid descriptor
    /// remains an error even when a stored hint purports to match its identity.
    pub fn verify(&self, input: &KeyInput<'_>) -> Result<(), Error> {
        if *self == Self::new(input)? {
            Ok(())
        } else {
            Err(Error::Mismatch)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn issue_m4_01_tensor_cache_corrupt_stored_digest_is_rejected() {
        let shape = [0, 0];
        let strides = [0, 1];
        let input = KeyInput {
            source: b"source",
            entry: "entry",
            source_identity: 0,
            graph_revision: 0,
            function_contract: [0; 32],
            numeric_policy: NumericPolicy::STRICT_F32,
            layout_version: 1,
            dimensions: [0, 0, 0],
            layouts: std::array::from_fn(|_| LayoutSpec {
                element: LayoutElement::F32,
                shape: &shape,
                strides: &strides,
                offset: 0,
                allocation_bytes: 0,
                base_alignment: 4,
            }),
            compiler_build: "build",
            target: "cpu",
            compiler_options: &[],
            emitted_source: b"emission",
        };
        let mut key = CacheKey::new(&input).unwrap();
        key.0[17] ^= 1;
        assert_eq!(key.verify(&input), Err(Error::Mismatch));
    }
}
