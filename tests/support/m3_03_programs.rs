#![allow(dead_code)]
use invar::{check::check, front::parse, mir};

pub const CHAIN: &str = "fn main() -> i32 { \
    let xs: Array<u32> = [1u32,2u32,3u32]; \
    var a: Array<u32> = filled_u32(len(xs),0u32); \
    for i in 0u32 .. len(xs) { a[i] = wrapping_add(xs[i],3u32); } \
    let first: Array<u32> = move a; \
    var b: Array<u32> = filled_u32(len(first),0u32); \
    for j in 0u32 .. len(first) { b[j] = wrapping_mul(first[j],7u32); } \
    let second: Array<u32> = move b; \
    var c: Array<u32> = filled_u32(len(second),0u32); \
    for k in 0u32 .. len(second) { c[k] = wrapping_add(second[k],11u32); } \
    if c[0u32] != 39u32 { return 1; } \
    if c[1u32] != 46u32 { return 1; } \
    if c[2u32] != 53u32 { return 1; } return 0; }";

pub const ZIP_DAG: &str = "fn main() -> i32 { \
    let xs: Array<u32> = [1u32,2u32,3u32]; \
    let ys: Array<u32> = [4u32,5u32,6u32]; \
    let zs: Array<u32> = [7u32,8u32,9u32]; \
    var a: Array<u32> = filled_u32(len(xs),0u32); \
    for i in 0u32 .. len(xs) { a[i] = wrapping_add(xs[i],ys[i]); } \
    let first: Array<u32> = move a; \
    var b: Array<u32> = filled_u32(len(first),0u32); \
    for j in 0u32 .. len(first) { b[j] = wrapping_mul(first[j],zs[j]); } \
    if b[0u32] != 35u32 { return 1; } \
    if b[1u32] != 56u32 { return 1; } \
    if b[2u32] != 81u32 { return 1; } return 0; }";

pub const LARGE_CHAIN: &str = "fn main() -> i32 { \
    let xs: Array<u32> = filled_u32(67108864u32,1u32); \
    var a: Array<u32> = filled_u32(len(xs),0u32); \
    for i in 0u32 .. len(xs) { a[i] = wrapping_add(xs[i],3u32); } \
    let first: Array<u32> = move a; \
    var b: Array<u32> = filled_u32(len(first),0u32); \
    for j in 0u32 .. len(first) { b[j] = wrapping_mul(first[j],7u32); } \
    let second: Array<u32> = move b; \
    var c: Array<u32> = filled_u32(len(second),0u32); \
    for k in 0u32 .. len(second) { c[k] = wrapping_add(second[k],11u32); } return 0; }";

pub fn checked_program(source: &str) -> (invar::check::CheckedProgram, mir::Program) {
    let checked = check(parse(source.as_bytes()).unwrap(), true).unwrap();
    let program = mir::lower(&checked).unwrap();
    (checked, program)
}
