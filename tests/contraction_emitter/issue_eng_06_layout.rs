//! ENG-06 (#61) part E: `Layout` gains a strided view with rank, extents,
//! strides and element type (the ENG-03 storage types), and `Geometry` gains a
//! third dimension. Every current family still requires `ContiguousU32` and
//! one plane, so a strided buffer or a second plane is refused until a family
//! admits it, and the admitted map's kernel is unchanged.
#![forbid(unsafe_code)]

use invar::{admission, check::check, device_ir, front::parse, mir, plan::Execution};
use invar_device::{
    Capabilities, ElementType, Error, Geometry, Ir, Layout, MAX_RANK, Provenance, Strided,
};

const SOURCE: &str = "fn main() -> i32 { let xs: Array<u32> = [1u32, 2u32, 3u32]; var ys: Array<u32> = filled_u32(len(xs), 0u32); for i in 0u32 .. len(xs) { ys[i] = wrapping_add(xs[i], 7u32); } return 0; }";

fn admitted() -> Ir {
    let checked = check(parse(SOURCE.as_bytes()).unwrap(), true).unwrap();
    let program = mir::lower(&checked).unwrap();
    let report = admission::analyze(SOURCE.as_bytes(), &program).unwrap();
    let region = &report.regions()[0];
    assert!(matches!(region.execution(), Execution::Map(_)));
    device_ir::from_region(region, 3).unwrap()
}

fn verify(ir: Ir) -> Result<invar_device::Verified, Error> {
    let origin = Provenance {
        source_identity: ir.source_identity,
        graph_revision: ir.graph_revision,
    };
    invar_device::verify(ir, origin, Capabilities::cuda_12_9_sm89())
}

#[test]
fn issue_eng_06_layout_element_types_are_sized_and_storage_types_marked() {
    let all = [
        (ElementType::U32, 4, false),
        (ElementType::I32, 4, false),
        (ElementType::F32, 4, false),
        (ElementType::F64, 8, false),
        (ElementType::F16, 2, true),
        (ElementType::Bf16, 2, true),
        (ElementType::I8, 1, true),
        (ElementType::U8, 1, true),
        (ElementType::Bool, 1, false),
    ];
    for (element, bytes, storage) in all {
        assert_eq!(element.bytes(), bytes, "{element:?}");
        assert_eq!(element.storage_only(), storage, "{element:?}");
    }
    assert_eq!(all.len(), 9);
}

#[test]
fn issue_eng_06_layout_strided_views_verify_rank_and_bounds() {
    let dense = Strided::row_major(ElementType::F32, &[2, 3, 4]).unwrap();
    assert_eq!(dense.rank, 3);
    assert_eq!(dense.extents, [2, 3, 4, 0]);
    assert_eq!(dense.strides, [12, 4, 1, 0]);
    assert_eq!(dense.span(), Some(24));
    assert_eq!(dense.verify(24), Ok(()));
    assert_eq!(dense.verify(23), Err(Error::Bounds));
    // A transposed view of a 3x5 matrix spans the same 15 elements.
    let transposed = Strided {
        element: ElementType::F16,
        rank: 2,
        extents: [5, 3, 0, 0],
        strides: [1, 5, 0, 0],
    };
    assert_eq!(transposed.span(), Some(15));
    assert_eq!(transposed.verify(15), Ok(()));
    // A padded row (ld 8 for 6 columns) over 4 rows: 3 * 8 + 6 elements.
    let padded = Strided {
        element: ElementType::Bf16,
        rank: 2,
        extents: [4, 6, 0, 0],
        strides: [8, 1, 0, 0],
    };
    assert_eq!(padded.span(), Some(30));
    // An empty extent addresses nothing.
    let empty = Strided::row_major(ElementType::U8, &[3, 0]).unwrap();
    assert_eq!(empty.span(), Some(0));
    assert_eq!(empty.verify(0), Ok(()));
    // Rank 0, rank above MAX_RANK and stray data in unused dimensions.
    assert!(Strided::row_major(ElementType::U32, &[]).is_none());
    assert!(Strided::row_major(ElementType::U32, &[1; MAX_RANK + 1]).is_none());
    let mut stray = dense;
    stray.extents[3] = 1;
    assert_eq!(stray.verify(u64::MAX), Err(Error::Layout));
    let mut stray = dense;
    stray.strides[3] = 1;
    assert_eq!(stray.verify(u64::MAX), Err(Error::Layout));
    let mut rank = dense;
    rank.rank = 0;
    assert_eq!(rank.verify(u64::MAX), Err(Error::Layout));
    rank.rank = MAX_RANK as u8 + 1;
    assert_eq!(rank.verify(u64::MAX), Err(Error::Layout));
    // Overflowing spans are refused, not wrapped.
    let huge = Strided {
        element: ElementType::F64,
        rank: 2,
        extents: [u64::MAX, 2, 0, 0],
        strides: [u64::MAX, 1, 0, 0],
    };
    assert_eq!(huge.span(), None);
    assert_eq!(huge.verify(u64::MAX), Err(Error::Bounds));
}

#[test]
fn issue_eng_06_layout_current_families_refuse_strided_buffers_and_planes() {
    let ir = admitted();
    assert_eq!(ir.geometry.planes, 1);
    assert_eq!(
        ir.geometry,
        Geometry::new(ir.geometry.blocks, ir.geometry.lanes)
    );
    let kernel = invar_device::emit_cuda(&verify(ir.clone()).unwrap()).unwrap();
    // The admitted map's kernel is unchanged (issue_m3_02's pinned hash).
    assert_eq!(invar_device::source_hash(&kernel), 0xdbbc_b16d_ac2c_5e72);
    let strided = Layout::Strided(Strided::row_major(ElementType::U32, &[3]).unwrap());
    let mut input = ir.clone();
    input.input.layout = strided;
    assert_eq!(verify(input).err(), Some(Error::Layout));
    let mut output = ir.clone();
    output.output.layout = strided;
    assert_eq!(verify(output).err(), Some(Error::Layout));
    for planes in [0, 2] {
        let mut ir = ir.clone();
        ir.geometry.planes = planes;
        assert_eq!(verify(ir).err(), Some(Error::Geometry), "{planes}");
    }
}
