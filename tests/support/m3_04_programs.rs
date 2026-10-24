#![allow(dead_code)]
pub const SUM: &str = "fn main() -> i32 { let xs: Array<u32> = [1u32,2u32,3u32]; var acc: u32 = 7u32; for i in 0u32 .. len(xs) { acc = wrapping_add(acc,xs[i]); } if acc == 13u32 { return 0; } return 1; }";

pub const PRODUCT: &str = "fn main() -> i32 { let xs: Array<u32> = [2u32,3u32,5u32]; var acc: u32 = 3u32; for i in 0u32 .. len(xs) { acc = wrapping_mul(acc,xs[i]); } if acc == 90u32 { return 0; } return 1; }";

pub const INCLUSIVE: &str = "fn main() -> i32 { let aa: Array<u32> = [2u32,3u32,5u32]; let bb: Array<u32> = [7u32,11u32,13u32]; var state: u32 = 17u32; var out: Array<u32> = filled_u32(len(aa),0u32); for i in 0u32 .. len(aa) { state = wrapping_add(wrapping_mul(aa[i],state),bb[i]); out[i] = state; } if out[2u32] == 683u32 { return 0; } return 1; }";

pub const EXCLUSIVE: &str = "fn main() -> i32 { let aa: Array<u32> = [2u32,3u32,5u32]; let bb: Array<u32> = [7u32,11u32,13u32]; var state: u32 = 17u32; var out: Array<u32> = filled_u32(len(aa),0u32); for i in 0u32 .. len(aa) { out[i] = state; state = wrapping_add(wrapping_mul(aa[i],state),bb[i]); } if out[0u32] == 17u32 { return 0; } return 1; }";

pub const FINAL: &str = "fn main() -> i32 { let aa: Array<u32> = [2u32,3u32,5u32]; let bb: Array<u32> = [7u32,11u32,13u32]; var state: u32 = 17u32; for i in 0u32 .. len(aa) { state = wrapping_add(wrapping_mul(aa[i],state),bb[i]); } if state == 683u32 { return 0; } return 1; }";
