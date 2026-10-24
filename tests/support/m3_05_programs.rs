#![allow(dead_code)]
pub const PAIR: &str = "fn main() -> i32 { let data: Array<u32> = [1u32,2u32,3u32]; \
    var first: Array<u32> = filled_u32(len(data),0u32); \
    for i in 0u32 .. len(data) { first[i] = wrapping_add(data[i],3u32); } \
    let stable: Array<u32> = move first; \
    var second: Array<u32> = filled_u32(len(stable),0u32); \
    for j in 0u32 .. len(stable) { second[j] = wrapping_mul(stable[j],7u32); } \
    if second[2u32] == 42u32 { return 0; } return 1; }";

pub const REVERSED_ZIP: &str = "fn main() -> i32 { let xs: Array<u32> = [1u32,2u32,3u32]; \
    let ys: Array<u32> = [2u32,3u32,5u32]; \
    var first: Array<u32> = filled_u32(len(xs),0u32); \
    for i in 0u32 .. len(xs) { first[i] = wrapping_add(xs[i],3u32); } \
    let mid: Array<u32> = move first; \
    var second: Array<u32> = filled_u32(len(mid),0u32); \
    for j in 0u32 .. len(mid) { second[j] = wrapping_mul(ys[j],mid[j]); } \
    if second[0u32] != 8u32 { return 1; } \
    if second[2u32] != 30u32 { return 1; } return 0; }";
