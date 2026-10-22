//! ENG-10 (#56): held-out domain benchmark suites (ML, games, robotics).
//!
//! Every program is ordinary Invar source: a leaf kernel written as loops over
//! flat arrays, plus a generated `main` that fills its inputs from a seeded
//! recurrence and returns 0 only when every output equals the Rust oracle
//! bit for bit. The oracles in `oracle.rs` are plain Rust, never Invar.
//!
//! The split unit is the kernel: all instances (shape and seed variants) of
//! one kernel share a split, so a tuning path never sees the loop structure of
//! a held-out kernel. The split is a seeded 25% of the kernels per suite.
//! Nothing here freezes a held-out set; the freeze is gated by #52.
#![allow(dead_code)] // shared by the bench binary and its tests; each uses part
use super::hash::sha256;

pub mod cli;
pub mod games;
pub mod keying;
pub mod manifest;
pub mod ml;
pub mod oracle;
pub mod report;
pub mod robotics;

pub const SUITES: [&str; 3] = ["ml", "games", "robotics"];
pub const SEED: u64 = 0x646f_6d61_696e_7331;

/// One seeded input array: `len` elements from [`oracle::fill`] with `seed`.
pub struct Input {
    pub len: u32,
    pub seed: u32,
}

/// The values a program must produce.
pub enum Wanted {
    F32(Vec<f32>),
    U32(Vec<u32>),
}

/// One shape and seed variant of a kernel: the inputs, the call expression
/// (over `in0`, `in1`, ...) and the oracle's answer.
pub struct Case {
    pub shape: Vec<(&'static str, u32)>,
    pub inputs: Vec<Input>,
    pub call: String,
    pub wanted: Wanted,
}

pub struct Program {
    pub suite: &'static str,
    pub kernel: &'static str,
    pub id: String,
    pub shape: Vec<(&'static str, u32)>,
    /// `(len, seed)` of each generated input array.
    pub inputs: Vec<(u32, u32)>,
    pub source: String,
}

/// A kernel's loops and the shape and seed variants that exercise them.
pub struct Kernel {
    pub name: &'static str,
    pub leaf: &'static str,
    pub cases: Vec<Case>,
}

const HELPERS: &str = "fn fill(n: u32, seed: u32) -> Array<f32> {
    var out: Array<f32> = filled_f32(n, 0.0);
    var s: u32 = seed;
    for i in 0u32..n {
        s = wrapping_add(wrapping_mul(s, 1664525u32), 1013904223u32);
        out[i] = (f32_from_u32((s / 65536u32) % 2001u32) - 1000.0) * 0.001;
    }
    return move out;
}
";

const COMPARE_F32: &str = "fn mismatches(got: read Array<f32>, wanted: read Array<f32>) -> u32 {
    if len(got) != len(wanted) { return 1u32 + len(wanted); }
    var bad: u32 = 0u32;
    for q in 0u32..len(wanted) {
        if got[q] != wanted[q] { bad = bad + 1u32; }
        if wanted[q] == 0.0 {
            if 1.0 / got[q] != 1.0 / wanted[q] { bad = bad + 1u32; }
        }
    }
    return bad;
}
";

const COMPARE_U32: &str = "fn mismatches(got: read Array<u32>, wanted: read Array<u32>) -> u32 {
    if len(got) != len(wanted) { return 1u32 + len(wanted); }
    var bad: u32 = 0u32;
    for q in 0u32..len(wanted) {
        if got[q] != wanted[q] { bad = bad + 1u32; }
    }
    return bad;
}
";

/// Shortest decimal that parses back to exactly `x` (Invar decimal literals
/// are exact and carry no exponent; Rust's `Display` never prints one).
pub fn f32_literal(x: f32) -> String {
    assert!(x.is_finite(), "oracle produced a non-finite value");
    let text = format!("{x}");
    if text.contains('.') {
        text
    } else {
        format!("{text}.0")
    }
}

fn array_literal<T>(xs: &[T], one: impl Fn(&T) -> String) -> String {
    format!("[{}]", xs.iter().map(one).collect::<Vec<_>>().join(", "))
}

fn assemble(kernel: &Kernel, case: &Case) -> String {
    let mut main = String::from("fn main() -> i32 {\n");
    for (k, input) in case.inputs.iter().enumerate() {
        main.push_str(&format!(
            "    let in{k}: Array<f32> = fill({}u32, {}u32);\n",
            input.len, input.seed
        ));
    }
    let (ty, wanted, compare) = match &case.wanted {
        Wanted::F32(xs) => ("f32", array_literal(xs, |x| f32_literal(*x)), COMPARE_F32),
        Wanted::U32(xs) => ("u32", array_literal(xs, |x| format!("{x}u32")), COMPARE_U32),
    };
    main.push_str(&format!(
        "    let got: Array<{ty}> = {};\n    let wanted: Array<{ty}> = {wanted};\n    if mismatches(got, wanted) != 0u32 {{ return 1; }}\n    return 0;\n}}\n",
        case.call
    ));
    format!("{HELPERS}{compare}{}\n{main}", kernel.leaf.trim_end())
}

pub fn dims(shape: &[(&'static str, u32)]) -> String {
    shape
        .iter()
        .map(|(name, value)| format!("{name}{value}"))
        .collect::<Vec<_>>()
        .join("_")
}

pub fn kernels(suite: &str) -> Result<Vec<Kernel>, String> {
    match suite {
        "ml" => Ok(ml::kernels()),
        "games" => Ok(games::kernels()),
        "robotics" => Ok(robotics::kernels()),
        other => Err(format!("unknown suite {other}")),
    }
}

/// Every program of `suite`, in kernel then case order.
pub fn programs(suite: &'static str) -> Result<Vec<Program>, String> {
    let mut out = Vec::new();
    for kernel in kernels(suite)? {
        for (ordinal, case) in kernel.cases.iter().enumerate() {
            out.push(Program {
                suite,
                kernel: kernel.name,
                id: format!("{suite}-{}-{}-c{ordinal}", kernel.name, dims(&case.shape)),
                shape: case.shape.clone(),
                inputs: case.inputs.iter().map(|i| (i.len, i.seed)).collect(),
                source: assemble(&kernel, case),
            });
        }
    }
    if out.is_empty() {
        return Err(format!("suite {suite} has no programs"));
    }
    Ok(out)
}

fn xorshift(state: &mut u64) -> u64 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    *state
}

/// The held-out kernels of `suite`: a seeded 25% (at least one) of the sorted
/// kernel names, chosen by a shuffle that does not look at any result.
pub fn held_out_kernels(suite: &str, names: &[&'static str]) -> Vec<&'static str> {
    let mut sorted = names.to_vec();
    sorted.sort_unstable();
    sorted.dedup();
    let mut state = SEED ^ u64::from_str_radix(&sha256(suite.as_bytes())[..16], 16).unwrap_or(1);
    for i in (1..sorted.len()).rev() {
        sorted.swap(i, xorshift(&mut state) as usize % (i + 1));
    }
    let count = sorted.len().div_ceil(4).max(1);
    let mut held = sorted[..count.min(sorted.len())].to_vec();
    held.sort_unstable();
    held
}
