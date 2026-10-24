#![allow(dead_code)]
use invar::{admission, check::check, device_ir, front::parse, mir, plan::Execution};
use invar_device::{
    AddressSpace, Barrier, Error as VerifyError, Geometry, Ir, LaneMask, Layout, Numeric, Op,
    Provenance, Transfer,
};

pub const SOURCE: &str = "fn main() -> i32 { let xs: Array<u32> = [1u32, 2u32, 3u32]; var ys: Array<u32> = filled_u32(len(xs), 0u32); for i in 0u32 .. len(xs) { ys[i] = wrapping_add(xs[i], 7u32); } return 0; }";

pub fn observed_source() -> String {
    SOURCE.replace("return 0;", "if ys[0u32] != 8u32 { return 1; } if ys[1u32] != 9u32 { return 1; } if ys[2u32] != 10u32 { return 1; } return 0;")
}

pub fn admitted() -> Ir {
    let checked = check(parse(SOURCE.as_bytes()).unwrap(), true).unwrap();
    let program = mir::lower(&checked).unwrap();
    let report = admission::analyze(SOURCE.as_bytes(), &program).unwrap();
    let region = &report.regions()[0];
    assert!(matches!(region.execution(), Execution::Map(_)));
    device_ir::from_region(region, 3).unwrap()
}

pub fn provenance(ir: &Ir) -> Provenance {
    Provenance {
        source_identity: ir.source_identity,
        graph_revision: ir.graph_revision,
    }
}

pub type Mutation = (&'static str, VerifyError, Box<dyn Fn(&mut Ir)>);

pub fn malformed_cases() -> Vec<Mutation> {
    vec![
        (
            "overread",
            VerifyError::Bounds,
            Box::new(|ir| ir.input.elements = 2),
        ),
        (
            "alias",
            VerifyError::Alias,
            Box::new(|ir| ir.output.id = ir.input.id),
        ),
        (
            "stale version",
            VerifyError::Version,
            Box::new(|ir| ir.version = 0),
        ),
        (
            "stale graph",
            VerifyError::Graph,
            Box::new(|ir| ir.graph_revision += 1),
        ),
        (
            "geometry",
            VerifyError::Geometry,
            Box::new(|ir| {
                ir.geometry = Geometry {
                    blocks: 0,
                    lanes: 256,
                }
            }),
        ),
        (
            "arithmetic",
            VerifyError::Operations,
            Box::new(|ir| ir.ops = vec![Op::CheckedAdd(1)]),
        ),
        (
            "missing barrier",
            VerifyError::Barrier,
            Box::new(|ir| {
                ir.requires_barrier = true;
                ir.barrier = Barrier::None;
            }),
        ),
        (
            "nonuniform barrier",
            VerifyError::Barrier,
            Box::new(|ir| {
                ir.requires_barrier = true;
                ir.barrier = Barrier::Conditional;
            }),
        ),
        (
            "lane mask",
            VerifyError::Mask,
            Box::new(|ir| ir.lane_mask = LaneMask::All),
        ),
        (
            "completion",
            VerifyError::Completion,
            Box::new(|ir| ir.completion = false),
        ),
        (
            "layout",
            VerifyError::Layout,
            Box::new(|ir| ir.input.layout = Layout::Unknown),
        ),
        (
            "address",
            VerifyError::Layout,
            Box::new(|ir| ir.input.address = AddressSpace::Host),
        ),
        (
            "transfer",
            VerifyError::Transfer,
            Box::new(|ir| ir.transfer = Transfer::Unknown),
        ),
        (
            "numeric",
            VerifyError::Numeric,
            Box::new(|ir| ir.numeric = Numeric::CheckedU32),
        ),
    ]
}
