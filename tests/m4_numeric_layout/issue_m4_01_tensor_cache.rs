#![forbid(unsafe_code)]
use invar::{
    layout_contract::{LayoutElement, LayoutError, LayoutSpec},
    numeric_contract::{NumericPolicy, PolicyError},
    tensor_cache::{CacheKey, Error, KeyInput},
};

const OPTIONS: &[&str] = &["--std=c++17", "--fmad=false", "--ftz=false"];

/// Descriptor fixtures only. Main separately owns actual source/MIR recognition
/// and binding a checked function contract to real emitted device bytes.
#[derive(Clone)]
struct Fixture {
    source: Vec<u8>,
    entry: String,
    source_identity: u64,
    revision: u64,
    contract: [u8; 32],
    policy: NumericPolicy,
    version: u32,
    dimensions: [u32; 3],
    shapes: [[u64; 2]; 3],
    strides: [[u64; 2]; 3],
    offsets: [u64; 3],
    allocations: [u64; 3],
    alignments: [u64; 3],
    elements: [LayoutElement; 3],
    build: String,
    target: String,
    emitted: Vec<u8>,
}
impl Fixture {
    fn new() -> Self {
        Self {
            source: b"checked source fixture".to_vec(),
            entry: "matmul".to_owned(),
            source_identity: 77,
            revision: 3,
            contract: [0x5a; 32],
            policy: NumericPolicy::STRICT_F32,
            version: 1,
            dimensions: [2, 3, 5],
            shapes: [[2, 3], [3, 5], [2, 5]],
            strides: [[3, 1], [5, 1], [5, 1]],
            offsets: [0; 3],
            allocations: [128; 3],
            alignments: [16; 3],
            elements: [LayoutElement::F32; 3],
            build: "rustc1.95+nvrtc12.4".to_owned(),
            target: "cuda:compute_86".to_owned(),
            emitted: b"program-specific emitted fixture".to_vec(),
        }
    }
    fn input<'a>(&'a self, options: &'a [&'a str]) -> KeyInput<'a> {
        KeyInput {
            source: &self.source,
            entry: &self.entry,
            source_identity: self.source_identity,
            graph_revision: self.revision,
            function_contract: self.contract,
            numeric_policy: self.policy,
            layout_version: self.version,
            dimensions: self.dimensions,
            layouts: std::array::from_fn(|i| LayoutSpec {
                element: self.elements[i],
                shape: &self.shapes[i],
                strides: &self.strides[i],
                offset: self.offsets[i],
                allocation_bytes: self.allocations[i],
                base_alignment: self.alignments[i],
            }),
            compiler_build: &self.build,
            target: &self.target,
            compiler_options: options,
            emitted_source: &self.emitted,
        }
    }
    fn key(&self) -> CacheKey {
        CacheKey::new(&self.input(OPTIONS)).unwrap()
    }
    fn dimensions(&mut self, d: [u32; 3]) {
        self.dimensions = d;
        let [r, k, c] = d.map(u64::from);
        self.shapes = [[r, k], [k, c], [r, c]];
        self.strides = [[k, 1], [c, 1], [c, 1]];
    }
}

#[test]
fn issue_m4_01_tensor_cache_key_is_independent_of_storage() {
    let fixture = Fixture::new();
    let key = fixture.key();
    let independent = fixture.clone();
    assert_eq!(key, independent.key());
    key.verify(&independent.input(OPTIONS)).unwrap();
    assert_eq!(key.to_hex().len(), 64);
    println!("tensor_cache_key={}", key.to_hex());
}

#[test]
fn issue_m4_01_tensor_cache_every_source_proof_and_artifact_field_invalidates() {
    let fixture = Fixture::new();
    let key = fixture.key();
    for field in 0..8 {
        let mut changed = fixture.clone();
        match field {
            0 => changed.source.push(b' '),
            1 => changed.entry.push('2'),
            2 => changed.source_identity += 1,
            3 => changed.revision += 1,
            4 => changed.contract[11] ^= 1,
            5 => changed.build.push_str(".changed"),
            6 => changed.target = "cuda:compute_89".to_owned(),
            7 => changed.emitted.push(b'\n'),
            _ => unreachable!(),
        }
        assert_ne!(key, changed.key(), "field {field}");
        assert_eq!(key.verify(&changed.input(OPTIONS)), Err(Error::Mismatch));
    }
    for axis in 0..3 {
        let mut changed = fixture.clone();
        let mut d = changed.dimensions;
        d[axis] += 1;
        changed.dimensions(d);
        assert_ne!(key, changed.key());
        assert_eq!(key.verify(&changed.input(OPTIONS)), Err(Error::Mismatch));
    }
}

#[test]
fn issue_m4_01_tensor_cache_each_buffer_layout_field_invalidates() {
    let fixture = Fixture::new();
    let key = fixture.key();
    for buffer in 0..3 {
        for field in 0..5 {
            let mut changed = fixture.clone();
            match field {
                0 => changed.strides[buffer][0] += 1,
                1 => changed.strides[buffer][1] += 1,
                2 => changed.offsets[buffer] += 1,
                3 => changed.allocations[buffer] += 4,
                4 => changed.alignments[buffer] *= 2,
                _ => unreachable!(),
            }
            assert_ne!(key, changed.key(), "buffer {buffer}, field {field}");
            assert_eq!(key.verify(&changed.input(OPTIONS)), Err(Error::Mismatch));
        }
    }
}

#[test]
fn issue_m4_01_tensor_cache_lengths_option_order_and_bytes_are_unambiguous() {
    let mut left = Fixture::new();
    left.source = b"ab".to_vec();
    left.entry = "c".to_owned();
    let mut right = left.clone();
    right.source = b"a".to_vec();
    right.entry = "bc".to_owned();
    assert_ne!(left.key(), right.key());
    let fixture = Fixture::new();
    for (a, b) in [
        (&["ab", "c"][..], &["a", "bc"][..]),
        (&["a", "b"][..], &["b", "a"][..]),
        (&[][..], &[""][..]),
        (&["x\0y"][..], &["x", "y"][..]),
        (&["a"][..], &["a", "a"][..]),
    ] {
        let key = CacheKey::new(&fixture.input(a)).unwrap();
        assert_eq!(key.verify(&fixture.input(b)), Err(Error::Mismatch));
    }
    let mut changed = fixture.clone();
    changed.source.push(0);
    assert_ne!(fixture.key(), changed.key());
    // Different compiler options must miss; option legality is still the actual
    // compiler/admission owner's job, never granted by a cache key.
    assert_eq!(
        fixture.key().verify(&fixture.input(&["--use_fast_math"])),
        Err(Error::Mismatch)
    );
}

#[test]
fn issue_m4_01_tensor_cache_rejects_versions_types_rank_and_shape() {
    let fixture = Fixture::new();
    let key = fixture.key();
    for version in [0, 2, u32::MAX] {
        let mut changed = fixture.clone();
        changed.version = version;
        assert_eq!(
            key.verify(&changed.input(OPTIONS)),
            Err(Error::LayoutVersion(version))
        );
    }
    // NumericPolicy is opaque. Invalid policy bytes fail at its public
    // constructor before KeyInput can exist; version 2 (docs/numeric.md) uses
    // its own kind spellings, so a v1 kind under it is refused.
    assert_eq!(
        NumericPolicy::from_parts(3, "strict-f32"),
        Err(PolicyError::UnsupportedVersion(3))
    );
    assert_eq!(
        NumericPolicy::from_parts(2, "strict-f32"),
        Err(PolicyError::UnsupportedKind)
    );
    assert_eq!(
        NumericPolicy::from_parts(1, "relaxed-f32"),
        Err(PolicyError::UnsupportedKind)
    );
    for buffer in 0..3 {
        for element in [LayoutElement::U32, LayoutElement::F16] {
            let mut changed = fixture.clone();
            changed.elements[buffer] = element;
            assert_eq!(
                key.verify(&changed.input(OPTIONS)),
                Err(Error::Element { buffer })
            );
        }
        let mut changed = fixture.clone();
        changed.shapes[buffer][1] += 1;
        assert_eq!(
            key.verify(&changed.input(OPTIONS)),
            Err(Error::Shape { buffer })
        );
        let short = [1];
        let mut input = fixture.input(OPTIONS);
        input.layouts[buffer].shape = &short;
        assert_eq!(key.verify(&input), Err(Error::Rank { buffer }));
        let mut input = fixture.input(OPTIONS);
        input.layouts[buffer].strides = &short;
        assert_eq!(key.verify(&input), Err(Error::Rank { buffer }));
        let rank3 = [1, 2, 3];
        let mut input = fixture.input(OPTIONS);
        input.layouts[buffer].shape = &rank3;
        assert_eq!(key.verify(&input), Err(Error::Rank { buffer }));
    }
}

#[test]
fn issue_m4_01_tensor_cache_revalidates_bounds_alignment_and_overflow() {
    let fixture = Fixture::new();
    let key = fixture.key();
    for buffer in 0..3 {
        for (field, value, error) in [
            (0, 0, LayoutError::InvalidAlignment),
            (0, 3, LayoutError::InvalidAlignment),
            (0, 2, LayoutError::Misaligned),
            (1, 4, LayoutError::PhysicalBounds),
            (1, (1u64 << 31) + 1, LayoutError::BufferLimit),
            (2, u64::MAX, LayoutError::ArithmeticOverflow),
            (3, u64::MAX, LayoutError::ArithmeticOverflow),
        ] {
            let mut changed = fixture.clone();
            match field {
                0 => changed.alignments[buffer] = value,
                1 => changed.allocations[buffer] = value,
                2 => changed.offsets[buffer] = value,
                3 => changed.strides[buffer][0] = value,
                _ => unreachable!(),
            }
            assert_eq!(
                key.verify(&changed.input(OPTIONS)),
                Err(Error::Layout { buffer, error })
            );
        }
    }
    let mut changed = fixture.clone();
    changed.dimensions([1, u32::MAX, 0]);
    assert_eq!(
        key.verify(&changed.input(OPTIONS)),
        Err(Error::Layout {
            buffer: 0,
            error: LayoutError::ExtentLimit
        })
    );
}

#[test]
fn issue_m4_01_tensor_cache_zero_axes_one_past_and_bounded_strided_maps() {
    for dims in [[0, 3, 5], [2, 0, 5], [2, 3, 0], [0, 0, 0]] {
        let mut fixture = Fixture::new();
        fixture.dimensions(dims);
        for i in 0..3 {
            if fixture.shapes[i].contains(&0) {
                fixture.allocations[i] = 4;
                fixture.offsets[i] = 1;
                fixture.strides[i] = [u64::MAX, u64::MAX];
            }
        }
        let key = fixture.key();
        key.verify(&fixture.input(OPTIONS)).unwrap();
        let mut changed = fixture.clone();
        changed.offsets[0] += 1;
        assert!(key.verify(&changed.input(OPTIONS)).is_err());
    }
    let mut fixture = Fixture::new();
    fixture.strides[0] = [1, 2];
    fixture.strides[1] = [6, 1];
    fixture.offsets = [1, 2, 3];
    let key = fixture.key();
    key.verify(&fixture.input(OPTIONS)).unwrap();
    let checked = fixture.input(OPTIONS).layouts[0].verify().unwrap();
    assert_eq!(checked.access(&[1, 2], 4).unwrap(), 24..28);
    assert_eq!(checked.access(&[1, 2], 8), Err(LayoutError::AccessWidth));
}

#[test]
fn issue_m4_01_tensor_cache_small_shape_grid_has_exact_maps_and_stable_keys() {
    for rows in 0..=4 {
        for inner in 0..=4 {
            for cols in 0..=4 {
                let mut fixture = Fixture::new();
                fixture.dimensions([rows, inner, cols]);
                let key = fixture.key();
                key.verify(&fixture.input(OPTIONS)).unwrap();
                for layout in fixture.input(OPTIONS).layouts {
                    let checked = layout.verify().unwrap();
                    let [r, c] = [layout.shape[0], layout.shape[1]];
                    assert_eq!(checked.logical_elements(), r * c);
                    if r > 0 && c > 0 {
                        assert_eq!(
                            checked.access(&[r - 1, c - 1], 4).unwrap(),
                            (r * c - 1) * 4..r * c * 4
                        );
                    } else {
                        assert_eq!(checked.access(&[0, 0], 4), Err(LayoutError::IndexBounds));
                    }
                }
            }
        }
    }
}

#[test]
fn issue_m4_01_tensor_cache_bounded_byte_and_option_inputs() {
    let fixture = Fixture::new();
    let key = fixture.key();
    for field in 0..5 {
        let mut changed = fixture.clone();
        match field {
            0 => changed.source = vec![0; 1024 * 1024 + 1],
            1 => changed.emitted = vec![0; 1024 * 1024 + 1],
            2 => changed.entry = "e".repeat(4097),
            3 => changed.build = "b".repeat(4097),
            4 => changed.target = "t".repeat(4097),
            _ => unreachable!(),
        }
        assert_eq!(key.verify(&changed.input(OPTIONS)), Err(Error::InputLimit));
    }
    assert_eq!(
        key.verify(&fixture.input(&["x"; 65])),
        Err(Error::InputLimit)
    );
    let huge = "o".repeat(4097);
    assert_eq!(key.verify(&fixture.input(&[&huge])), Err(Error::InputLimit));
    for field in 0..3 {
        let mut changed = fixture.clone();
        match field {
            0 => changed.entry.clear(),
            1 => changed.build.clear(),
            2 => changed.target.clear(),
            _ => unreachable!(),
        }
        assert_eq!(key.verify(&changed.input(OPTIONS)), Err(Error::InputLimit));
    }
}
