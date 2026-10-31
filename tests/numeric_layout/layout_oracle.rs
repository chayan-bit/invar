//! Issue #1 (M4-01).
use invar::layout_contract::{
    LayoutElement as E, LayoutError as X, LayoutSpec, MAX_BUFFER_BYTES, MAX_EXTENT,
};

fn spec<'a>(shape: &'a [u64], strides: &'a [u64], bytes: u64) -> LayoutSpec<'a> {
    LayoutSpec {
        element: E::U32,
        shape,
        strides,
        offset: 0,
        allocation_bytes: bytes,
        base_alignment: 4,
    }
}

#[test]
fn layout_oracle_contiguous_and_transposed() {
    let row = spec(&[2, 3], &[3, 1], 24).verify().expect("row-major");
    let transpose = spec(&[3, 2], &[1, 3], 24).verify().expect("transpose");
    assert_eq!(row.logical_elements(), 6);
    for i in 0..2 {
        for j in 0..3 {
            let byte = 12 * i + 4 * j;
            assert_eq!(row.access(&[i, j], 4), Ok(byte..byte + 4));
            assert_eq!(transpose.access(&[j, i], 4), Ok(byte..byte + 4));
        }
    }
}

#[test]
fn layout_oracle_offset_and_holes() {
    let mut input = spec(&[2, 2], &[5, 2], 40);
    input.offset = 2;
    let map = input.verify().expect("strided");
    assert_eq!(map.access(&[0, 0], 4), Ok(8..12));
    assert_eq!(map.access(&[0, 1], 4), Ok(16..20));
    assert_eq!(map.access(&[1, 0], 4), Ok(28..32));
    assert_eq!(map.access(&[1, 1], 4), Ok(36..40));
}

#[test]
fn layout_oracle_empty_domain_ignores_unreachable_stride_products() {
    let shape = [MAX_EXTENT, MAX_EXTENT, MAX_EXTENT, 0];
    let strides = [u64::MAX; 4];
    let empty = spec(&shape, &strides, 0).verify().expect("empty domain");
    assert_eq!(empty.logical_elements(), 0);
    assert_eq!(empty.access(&[0, 0, 0, 0], 4), Err(X::IndexBounds));
}

#[test]
fn layout_oracle_empty_origin_bounds() {
    let mut input = spec(&[0], &[1], 8);
    input.offset = 2;
    let empty = input.verify().expect("one past origin");
    assert_eq!(empty.access(&[0], 4), Err(X::IndexBounds));
    input.offset = 3;
    assert_eq!(input.verify(), Err(X::PhysicalBounds));
}

#[test]
fn layout_oracle_rank_zero_scalar() {
    let scalar = spec(&[], &[], 4).verify().expect("scalar");
    assert_eq!(scalar.logical_elements(), 1);
    assert_eq!(scalar.access(&[], 4), Ok(0..4));
    assert_eq!(spec(&[], &[], 0).verify(), Err(X::PhysicalBounds));
}

#[test]
fn layout_oracle_frozen_element_widths() {
    for (element, width) in [(E::U32, 4), (E::F32, 4), (E::F16, 2)] {
        let mut input = spec(&[2], &[1], 2 * width);
        input.element = element;
        input.base_alignment = width;
        let map = input.verify().expect("element width");
        assert_eq!(map.access(&[1], width), Ok(width..width * 2));
        assert_eq!(map.access(&[1], width * 2), Err(X::AccessWidth));
        assert_eq!(map.access(&[1], 0), Err(X::AccessWidth));
    }
}

#[test]
fn layout_oracle_descriptor_rejections() {
    assert_eq!(spec(&[1; 5], &[1; 5], 4).verify(), Err(X::RankLimit));
    assert_eq!(spec(&[1, 1], &[1], 4).verify(), Err(X::RankMismatch));
    assert_eq!(
        spec(&[MAX_EXTENT + 1], &[0], 4).verify(),
        Err(X::ExtentLimit)
    );
    assert_eq!(
        spec(&[1], &[1], MAX_BUFFER_BYTES + 1).verify(),
        Err(X::BufferLimit)
    );
    assert_eq!(spec(&[2], &[1], 7).verify(), Err(X::PhysicalBounds));
}

#[test]
fn layout_oracle_logical_limit_even_with_broadcast() {
    assert_eq!(spec(&[MAX_EXTENT], &[0], 4).verify(), Err(X::BufferLimit));
    let exact = spec(&[MAX_BUFFER_BYTES / 4], &[1], MAX_BUFFER_BYTES)
        .verify()
        .expect("exact limit");
    assert_eq!(
        exact.access(&[MAX_BUFFER_BYTES / 4 - 1], 4),
        Ok(MAX_BUFFER_BYTES - 4..MAX_BUFFER_BYTES)
    );
}

#[test]
fn layout_oracle_alignment_metadata() {
    let mut input = spec(&[1], &[1], 4);
    for alignment in [0, 3, 6] {
        input.base_alignment = alignment;
        assert_eq!(input.verify(), Err(X::InvalidAlignment));
    }
    for alignment in [1, 2] {
        input.base_alignment = alignment;
        assert_eq!(input.verify(), Err(X::Misaligned));
    }
    input.base_alignment = 64;
    assert!(input.verify().is_ok());
}

#[test]
fn layout_oracle_checked_arithmetic() {
    let mut input = spec(&[0], &[1], 0);
    input.offset = u64::MAX;
    assert_eq!(input.verify(), Err(X::ArithmeticOverflow));
    assert_eq!(
        spec(&[3], &[u64::MAX], 4).verify(),
        Err(X::ArithmeticOverflow)
    );
    assert_eq!(
        spec(&[2, 2], &[u64::MAX, 1], 4).verify(),
        Err(X::ArithmeticOverflow)
    );
    assert_eq!(
        spec(&[MAX_EXTENT; 4], &[0; 4], 4).verify(),
        Err(X::ArithmeticOverflow)
    );
    assert_eq!(
        spec(&[2], &[u64::MAX], 4).verify(),
        Err(X::ArithmeticOverflow)
    );
    assert_eq!(
        spec(&[2], &[u64::MAX / 4], 4).verify(),
        Err(X::ArithmeticOverflow)
    );
    assert_eq!(
        spec(&[MAX_EXTENT, MAX_EXTENT, 4], &[0; 3], 4).verify(),
        Err(X::ArithmeticOverflow)
    );
}

#[test]
fn layout_oracle_rank_four_singletons() {
    let map = spec(&[1; 4], &[u64::MAX; 4], 4)
        .verify()
        .expect("rank four singleton");
    assert_eq!(map.logical_elements(), 1);
    assert_eq!(map.access(&[0; 4], 4), Ok(0..4));
}

#[test]
fn layout_oracle_access_rejections() {
    let map = spec(&[2], &[1], 16)
        .verify()
        .expect("extra allocation padding");
    assert_eq!(map.access(&[2], 4), Err(X::IndexBounds));
    assert_eq!(map.access(&[u64::MAX], 4), Err(X::IndexBounds));
    assert_eq!(map.access(&[], 4), Err(X::RankMismatch));
    assert_eq!(map.access(&[0, 0], 4), Err(X::RankMismatch));
    assert_eq!(map.access(&[0], 8), Err(X::AccessWidth));
}

#[test]
fn layout_oracle_overlapping_read_maps_do_not_imply_unique_access() {
    let map = spec(&[2, 3], &[0, 1], 12).verify().expect("read broadcast");
    assert_eq!(map.access(&[0, 2], 4), Ok(8..12));
    assert_eq!(map.access(&[1, 2], 4), Ok(8..12));
}

#[test]
fn layout_oracle_exhaustive_small_maps_against_enumeration() {
    for rows in 0..4_u64 {
        for columns in 0..4_u64 {
            for row_stride in 0..5_u64 {
                for column_stride in 0..5_u64 {
                    let shape = [rows, columns];
                    let strides = [row_stride, column_stride];
                    let mut offsets = Vec::new();
                    for i in 0..rows {
                        for j in 0..columns {
                            offsets.push(4 * (1 + i * row_stride + j * column_stride));
                        }
                    }
                    let needed = offsets.iter().map(|v| v + 4).max().unwrap_or(4);
                    let mut input = spec(&shape, &strides, needed);
                    input.offset = 1;
                    let map = input.verify().expect("enumerated domain fits");
                    assert_eq!(map.logical_elements(), rows * columns);
                    let mut observed = Vec::new();
                    for i in 0..rows {
                        for j in 0..columns {
                            let access = map.access(&[i, j], 4).expect("valid index");
                            assert_eq!(access.end - access.start, 4);
                            observed.push(access.start);
                        }
                    }
                    assert_eq!(observed, offsets);
                    if !offsets.is_empty() {
                        input.allocation_bytes -= 1;
                        assert_eq!(input.verify(), Err(X::PhysicalBounds));
                    }
                }
            }
        }
    }
}
