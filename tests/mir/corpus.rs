//! Programs shared by the MIR differential suites: the reference evaluator and the MIR evaluator
//! (and, in `issue_m1_02`, the native backend) must agree on each one. Fixed corpus, no random seed.

/// `(source, fuel)`.
pub const CORPUS: &[(&str, u64)] = &[
    ("fn main() -> i32 { return 42; }", 10_000),
    ("fn main() -> i32 { return 42; }", 1),
    (
        "fn main() -> i32 { if true { return 1; } else { return 1 / 0; } }",
        10_000,
    ),
    (
        "fn main() -> i32 { if false and (1 / 0 == 0) { return 1; } else { return 2; } }",
        10_000,
    ),
    (
        "fn main() -> i32 { if true or (1 / 0 == 0) { return 3; } else { return 4; } }",
        10_000,
    ),
    (
        "fn main() -> i32 { var x: i32 = 0; while x < 2 { x = x + 1; } return x; }",
        10_000,
    ),
    (
        "fn main() -> i32 { var x: u32 = 0u32; for i in 0u32 .. 2u32 { x = x + i; } return 1; }",
        10_000,
    ),
    (
        "fn main() -> i32 { for i in 0u32 .. 2u32 { let a: Array<i32> = [1]; let b: Array<i32> = move a; } return 0; }",
        10_000,
    ),
    (
        "fn main() -> i32 { let xs: Array<i32> = []; if len(xs) == 0u32 { return 1; } else { return 0; } }",
        10_000,
    ),
    (
        "fn main() -> i32 { let xs: Array<i32> = filled_i32(3u32, 7); return xs[2u32]; }",
        10_000,
    ),
    (
        "fn f(n: i32) -> i32 { if n == 0 { return 1; } else { return n * f(n - 1); } } fn main() -> i32 { return f(5); }",
        10_000,
    ),
    (
        "fn main() -> i32 { return -2147483648i32 / -1i32; }",
        10_000,
    ),
    (
        "fn main() -> i32 { return -2147483648i32 % -1i32; }",
        10_000,
    ),
    (
        "fn main() -> i32 { let xs: Array<i32> = [1]; return xs[2u32]; }",
        10_000,
    ),
    (
        "fn main() -> i32 { var xs: Array<i32> = [1]; xs[2u32] = 1 / 0; return 0; }",
        10_000,
    ),
    (
        "fn main() -> i32 { var xs: Array<i32> = [1]; xs[2u32] = 7; return 0; }",
        10_000,
    ),
    (
        "fn main() -> i32 { for i in 4294967295u32 .. 4294967295u32 {} return 8; }",
        10_000,
    ),
    (
        "fn make() -> Array<i32> { return [9]; } fn main() -> i32 { let xs: Array<i32> = make(); return xs[0u32]; }",
        10_000,
    ),
    ("fn main() -> i32 { while true {} return 0; }", 8),
    (include_str!("../../examples/sum.inv"), 10_000),
    (include_str!("../../examples/ownership.inv"), 10_000),
    (
        "fn read_one(a: read Array<i32>) -> i32 { return a[0u32]; } fn bump(a: mut Array<i32>) -> Unit { a[0u32] = a[0u32] + read_one(a); return; } fn main() -> i32 { var xs: Array<i32> = [2]; bump(xs); return xs[0u32]; }",
        10_000,
    ),
];
