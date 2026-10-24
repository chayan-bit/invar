#![allow(dead_code)]
use invar::{check, front, mir};

pub const SCALAR: &str = include_str!("../a1_tensor/scalar.inv");

pub const DIRECT: &str = include_str!("../a1_tensor/direct.inv");

pub const TC: &str = include_str!("../a1_tensor/tc.inv");

pub fn program(source: &str) -> mir::Program {
    mir::lower(&check::check(front::parse(source.as_bytes()).unwrap(), true).unwrap()).unwrap()
}
