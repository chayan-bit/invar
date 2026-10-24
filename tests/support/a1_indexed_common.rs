#![allow(dead_code)]
use invar::{check, front, mir};

pub const FUEL: u64 = 1_000_000;

pub const CASES: [&[u8]; 6] = [
    include_bytes!("../fixtures/a1-indexed/gather.inv"),
    include_bytes!("../fixtures/a1-indexed/scatter.inv"),
    include_bytes!("../fixtures/a1-indexed/histogram.inv"),
    include_bytes!("../fixtures/a1-indexed/affine.inv"),
    include_bytes!("../../bench/general-net-v01/current/frontier_graph_a.inv"),
    include_bytes!("../../bench/general-net-v01/current/frontier_graph_b.inv"),
];

pub fn lower(s: &[u8]) -> mir::Program {
    mir::lower(&check::check(front::parse(s).unwrap(), true).unwrap()).unwrap()
}

pub fn equal(a: &invar::eval::Observation, b: &invar::eval::Observation) {
    assert_eq!(a.result, b.result);
    assert_eq!(a.steps_used, b.steps_used);
    assert_eq!(a.cleanup_trace, b.cleanup_trace);
}

pub fn static_refusals() -> Vec<String> {
    let affine = std::str::from_utf8(CASES[3]).unwrap();
    vec![
        affine.replace("for i in 0u32..", "for i in 1u32.."),
        affine.replace("2u32*i", "0u32*i"),
        affine.replace("2u32*i", "4294967295u32*i"),
        affine.replace("filled_u32(5u32", "filled_u32(4u32"),
        affine.replace("filled_u32(5u32", "filled_u32(4097u32"),
        std::str::from_utf8(CASES[2])
            .unwrap()
            .replace("wrapping_add(out[ix[i]],a[i])", "out[ix[i]] + a[i]"),
    ]
}
