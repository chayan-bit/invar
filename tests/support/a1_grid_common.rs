#![allow(dead_code)]
use invar::{check, eval, front, mir};

#[path = "a1_grid_source.rs"]
pub mod fixtures;

pub const FUEL: u64 = 10_000_000;

pub fn program(source: &str) -> mir::Program {
    mir::lower(&check::check(front::parse(source.as_bytes()).unwrap(), true).unwrap()).unwrap()
}

pub fn parity(actual: &eval::Observation, serial: &eval::Observation) {
    assert_eq!(actual.result, serial.result);
    assert_eq!(actual.steps_used, serial.steps_used);
    assert_eq!(actual.cleanup_trace, serial.cleanup_trace);
}

pub fn sources() -> Vec<String> {
    let mut sources = vec![
        fixtures::transpose(false, 17, 31, false, true),
        fixtures::transpose(true, 17, 31, false, true),
    ];
    for clamp in [false, true] {
        for steps in [1, 2, 3] {
            sources.push(fixtures::stencil(clamp, false, 3, 5, steps));
        }
    }
    sources
}
