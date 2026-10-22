#![allow(dead_code)]
use invar::{admission, check, front, mir, plan::PlanKind};

pub const PAIR: &str = "fn pair(aa: read Array<u32>, bb: read Array<u32>, n: u32, seed_a: u32, seed_b: u32) -> Array<u32> { var sa = seed_a; var sb = seed_b; for i in 0u32..n { sa = wrapping_mul(aa[i],sa); sb = wrapping_add(wrapping_mul(aa[i],sb),bb[i]); } return [sa,sb]; }";

pub fn compile(source: &str) -> (invar::check::CheckedProgram, mir::Program) {
    let checked = check::check(front::parse(source.as_bytes()).unwrap(), true).unwrap();
    let program = mir::lower(&checked).unwrap();
    mir::verify(&program).unwrap();
    (checked, program)
}

pub fn inputs(n: usize) -> (Vec<u32>, Vec<u32>) {
    let mut seed = 0x5a17_2026_0910_0001u64;
    let mut next = || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        seed as u32
    };
    (
        (0..n).map(|_| next() | 1).collect(),
        (0..n).map(|_| next()).collect(),
    )
}

// Independently written arithmetic modulo 2^32, with no production law call.
pub fn oracle(a: &[u32], b: &[u32], seed: [u32; 2]) -> [u32; 2] {
    a.iter().zip(b).fold(seed, |[x, y], (&a, &b)| {
        [
            (u64::from(a) * u64::from(x)) as u32,
            (u64::from(a) * u64::from(y) + u64::from(b)) as u32,
        ]
    })
}

pub fn source(a: &[u32], b: &[u32], seed: [u32; 2]) -> String {
    let values = |xs: &[u32]| {
        xs.iter()
            .map(|v| format!("{v}u32"))
            .collect::<Vec<_>>()
            .join(",")
    };
    let expected = oracle(a, b, seed);
    format!(
        "{PAIR} fn main() -> i32 {{ let aa: Array<u32> = [{}]; let bb: Array<u32> = [{}]; let answer = pair(aa,bb,{}u32,{}u32,{}u32); if answer[0u32] == {}u32 and answer[1u32] == {}u32 {{ return 0; }} return 1; }}",
        values(a),
        values(b),
        a.len(),
        seed[0],
        seed[1],
        expected[0],
        expected[1]
    )
}

pub fn region(source: &str, program: &mir::Program) -> invar::admission::Report {
    let report = admission::analyze(source.as_bytes(), program).unwrap();
    assert!(
        report
            .regions()
            .iter()
            .any(|r| r.plan_kind() == PlanKind::Reduction)
    );
    report
}
