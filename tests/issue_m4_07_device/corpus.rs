//! Shared M4-07 device corpus. Every case is an ordinary checked source scan,
//! prepared through the real source contract; the device plan comes from that
//! preparation. `independent` is a separate sequential reference.
#![allow(dead_code)]

use invar::{
    bounded_summary::Budget,
    check, front,
    summary_source::{Contract, Family, Prepared},
};

pub const BUDGET: Budget = Budget {
    max_steps: 1 << 20,
    max_bytes: 1 << 33,
    max_composition_ops: 1 << 40,
};

/// Largest source array the summary contract admits (1,000,000 words).
pub const MAX_SOURCE_WORDS: u32 = 1_000_000;

pub struct Case {
    pub name: String,
    pub family: Family,
    pub prefix: bool,
    pub steps: u32,
    pub coefficients: Vec<u32>,
    pub seed: Vec<u32>,
}

pub fn step_words(family: Family) -> u32 {
    match family {
        Family::Table(k) => u32::from(k),
        Family::Matrix(d) => u32::from(d) * u32::from(d),
    }
}

/// The recognized ordinary-source loop for `family`, final-state or prefixes.
pub fn source(family: Family, prefix: bool) -> String {
    match family {
        Family::Table(k) => format!(
            "fn run(coeff: read Array<u32>, seed: u32, count: u32) -> {} {{ var state = seed; {} for i in 0u32..count {{ state = coeff[i * {k}u32 + state]; {} }} return {}; }}",
            if prefix { "Array<u32>" } else { "u32" },
            if prefix {
                "var out = filled_u32(count,0u32);"
            } else {
                ""
            },
            if prefix { "out[i] = state;" } else { "" },
            if prefix { "out" } else { "state" }
        ),
        Family::Matrix(d) => {
            let d = usize::from(d);
            let mut s = "fn run(coeff: read Array<u32>, seed: read Array<u32>, count: u32) -> Array<u32> { ".to_owned();
            for row in 0..d {
                s += &format!("var x{row} = seed[{row}u32]; ");
            }
            if prefix {
                s += &format!("var out = filled_u32(count * {d}u32,0u32); ");
            }
            s += "for i in 0u32..count { ";
            for row in 0..d {
                let mut dot = "0u32".to_owned();
                for col in 0..d {
                    dot = format!(
                        "wrapping_add({dot},wrapping_mul(coeff[i * {}u32 + {}u32],x{col}))",
                        d * d,
                        row * d + col
                    );
                }
                s += &format!("let y{row} = {dot}; ");
            }
            for row in 0..d {
                s += &format!("x{row} = y{row}; ");
            }
            if prefix {
                for row in 0..d {
                    s += &format!("out[i * {d}u32 + {row}u32] = x{row}; ");
                }
                s += "} return out; }";
            } else {
                let names = (0..d).map(|j| format!("x{j}")).collect::<Vec<_>>();
                s += &format!("}} return [{}]; }}", names.join(","));
            }
            s
        }
    }
}

pub fn contract(family: Family, prefix: bool) -> Contract {
    let checked = check::check(
        front::parse(source(family, prefix).as_bytes()).unwrap(),
        false,
    )
    .expect("corpus source checks");
    Contract::from_checked(&checked, "run").expect("corpus source is a bounded scan")
}

pub fn prepare(contract: &Contract, case: &Case) -> Prepared {
    contract
        .prepare(&case.coefficients, &case.seed, case.steps, BUDGET)
        .unwrap_or_else(|e| panic!("{}: {e:?}", case.name))
}

/// Sequential source-order reference in device output layout.
pub fn independent(family: Family, prefix: bool, coefficients: &[u32], seed: &[u32]) -> Vec<u32> {
    let mut state = seed.to_vec();
    let mut all = Vec::new();
    for step in coefficients.chunks_exact(step_words(family) as usize) {
        state = match family {
            Family::Table(_) => vec![step[state[0] as usize]],
            Family::Matrix(d) => {
                let d = usize::from(d);
                (0..d)
                    .map(|row| {
                        let sum = (0..d).fold(0u128, |sum, col| {
                            sum + u128::from(step[row * d + col]) * u128::from(state[col])
                        });
                        (sum % (1u128 << 32)) as u32
                    })
                    .collect()
            }
        };
        if prefix {
            all.extend_from_slice(&state);
        }
    }
    if prefix { all } else { state }
}

struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> u32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 32) as u32
    }
}

/// Noncommuting data: random table entries, or full-width matrix words with
/// every third word u32::MAX so products wrap.
pub fn random(family: Family, steps: u32, salt: u64) -> (Vec<u32>, Vec<u32>) {
    let mut lcg = Lcg(salt ^ 0x9e37_79b9_7f4a_7c15);
    let words = steps * step_words(family);
    match family {
        Family::Table(k) => (
            (0..words).map(|_| lcg.next() % u32::from(k)).collect(),
            vec![u32::from(k) - 1],
        ),
        Family::Matrix(d) => (
            (0..words)
                .map(|i| if i % 3 == 0 { u32::MAX } else { lcg.next() })
                .collect(),
            [u32::MAX, 0x8000_0000, 3, 11][..usize::from(d)].to_vec(),
        ),
    }
}

pub const FAMILIES: [Family; 8] = [
    Family::Table(1),
    Family::Table(2),
    Family::Table(3),
    Family::Table(16),
    Family::Matrix(1),
    Family::Matrix(2),
    Family::Matrix(3),
    Family::Matrix(4),
];

/// Zero, singleton, odd and ragged lengths, plus the largest source size.
pub const LENGTHS: [u32; 17] = [
    0, 1, 2, 3, 5, 7, 15, 16, 17, 255, 256, 257, 4095, 4096, 4097, 8193, 12289,
];

/// Bijective tables and odd-determinant matrices. Long random runs collapse to
/// a constant map or the zero matrix within a tile, which then absorbs every
/// earlier tile; these never collapse, so a skipped, repeated or reordered tile
/// in a look-back changes every later output.
pub fn invertible(family: Family, steps: u32, salt: u64) -> (Vec<u32>, Vec<u32>) {
    let mut lcg = Lcg(salt ^ 0x5851_f42d_4c95_7f2d);
    let mut words = Vec::new();
    for _ in 0..steps {
        match family {
            Family::Table(k) => {
                let mut p: Vec<u32> = (0..u32::from(k)).collect();
                for i in (1..p.len()).rev() {
                    p.swap(i, lcg.next() as usize % (i + 1));
                }
                words.extend(p);
            }
            Family::Matrix(d) => {
                // L (odd diagonal, random below) times U (unit diagonal,
                // random above): the determinant is odd, so it is a unit.
                let d = usize::from(d);
                let (mut l, mut u) = (vec![0u32; d * d], vec![0u32; d * d]);
                for r in 0..d {
                    for c in 0..d {
                        match r.cmp(&c) {
                            std::cmp::Ordering::Equal => {
                                (l[r * d + c], u[r * d + c]) = (lcg.next() | 1, 1)
                            }
                            std::cmp::Ordering::Greater => l[r * d + c] = lcg.next(),
                            std::cmp::Ordering::Less => u[r * d + c] = lcg.next(),
                        }
                    }
                }
                for r in 0..d {
                    for c in 0..d {
                        words.push((0..d).fold(0u32, |a, k| {
                            a.wrapping_add(l[r * d + k].wrapping_mul(u[k * d + c]))
                        }));
                    }
                }
            }
        }
    }
    let seed = match family {
        Family::Table(k) => vec![u32::from(k) - 1],
        Family::Matrix(d) => [u32::MAX, 0x8000_0000, 3, 11][..usize::from(d)].to_vec(),
    };
    (words, seed)
}

/// Every SIMD-group, round and tile boundary of `family` on both targets, each
/// at -1, 0, +1 and twice plus one.
pub fn boundaries(family: Family) -> Vec<u32> {
    use invar_device::{Capabilities, summary};
    let device = match family {
        Family::Table(k) => summary::Family::Table(k),
        Family::Matrix(d) => summary::Family::Matrix(d),
    };
    let metal = Capabilities::Metal {
        apple_family: 9,
        msl_major: 3,
        msl_minor: 1,
    };
    let mut lengths = Vec::new();
    for (caps, observation, rounds) in [
        (
            Capabilities::cuda_12_9_sm89(),
            summary::Observation::InclusivePrefixes,
            1,
        ),
        (
            Capabilities::cuda_12_9_sm89(),
            summary::Observation::FinalState,
            1,
        ),
        (metal, summary::Observation::InclusivePrefixes, 1),
        (
            metal,
            summary::Observation::FinalState,
            summary::METAL_ROUNDS,
        ),
    ] {
        let items = summary::items(device, observation, caps);
        for unit in [32 * items, 256 * items, 256 * items * rounds] {
            lengths.extend([unit - 1, unit, unit + 1, 2 * unit + 1]);
        }
    }
    lengths.sort_unstable();
    lengths.dedup();
    lengths
}

/// Lengths of 9 and 33 tiles plus a ragged tail on each target, so look-backs
/// cross several windows.
pub fn multi_tile(family: Family) -> Vec<u32> {
    use invar_device::{Capabilities, summary};
    let device = match family {
        Family::Table(k) => summary::Family::Table(k),
        Family::Matrix(d) => summary::Family::Matrix(d),
    };
    let metal = Capabilities::Metal {
        apple_family: 9,
        msl_major: 3,
        msl_minor: 1,
    };
    let largest = MAX_SOURCE_WORDS / step_words(family);
    [Capabilities::cuda_12_9_sm89(), metal]
        .into_iter()
        .flat_map(|caps| {
            let tile = 256 * summary::items(device, summary::Observation::InclusivePrefixes, caps);
            [9 * tile + 5, 33 * tile + 7]
        })
        .filter(|n| *n <= largest)
        .collect()
}

fn case(
    name: String,
    family: Family,
    prefix: bool,
    coefficients: Vec<u32>,
    seed: Vec<u32>,
) -> Case {
    Case {
        steps: coefficients.len() as u32 / step_words(family),
        name,
        family,
        prefix,
        coefficients,
        seed,
    }
}

pub fn cases() -> Vec<Case> {
    let mut all = Vec::new();
    for (f, family) in FAMILIES.into_iter().enumerate() {
        let largest = MAX_SOURCE_WORDS / step_words(family);
        let mut lengths = LENGTHS.to_vec();
        lengths.extend(boundaries(family).into_iter().filter(|n| *n < largest));
        lengths.push(largest);
        lengths.sort_unstable();
        lengths.dedup();
        for (l, steps) in lengths.into_iter().enumerate() {
            for prefix in [false, true] {
                let (coefficients, seed) = random(family, steps, (f * 64 + l) as u64);
                all.push(case(
                    format!("random {family:?} n={steps} prefix={prefix}"),
                    family,
                    prefix,
                    coefficients,
                    seed,
                ));
            }
        }
    }
    for (f, family) in FAMILIES.into_iter().enumerate() {
        for steps in multi_tile(family) {
            for prefix in [false, true] {
                let (coefficients, seed) = invertible(family, steps, f as u64);
                all.push(case(
                    format!("invertible {family:?} n={steps} prefix={prefix}"),
                    family,
                    prefix,
                    coefficients,
                    seed,
                ));
            }
        }
    }
    // Every ordered triple of the four 2-state tables, and every ordered pair
    // of the 27 3-state tables, observed at every prefix.
    let tables = |k: u32| -> Vec<Vec<u32>> {
        (0..k.pow(k))
            .map(|n| (0..k).map(|s| n / k.pow(s) % k).collect())
            .collect()
    };
    let two = tables(2);
    let mut triples = Vec::new();
    for a in &two {
        for b in &two {
            for c in &two {
                triples.extend([a, b, c].into_iter().flatten());
            }
        }
    }
    all.push(case(
        "exhaustive K=2 triples".into(),
        Family::Table(2),
        true,
        triples,
        vec![1],
    ));
    let three = tables(3);
    let mut pairs = Vec::new();
    for a in &three {
        for b in &three {
            pairs.extend([a, b].into_iter().flatten());
        }
    }
    for seed in 0..3 {
        all.push(case(
            format!("exhaustive K=3 pairs seed={seed}"),
            Family::Table(3),
            true,
            pairs.clone(),
            vec![seed],
        ));
    }
    // Identity, zero, all-ones-wrapping and nilpotent matrix fixtures.
    let identity4 = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];
    let nilpotent4 = [0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0];
    let seed4 = vec![u32::MAX, 0x8000_0000, 3, 11];
    for (name, word) in [
        ("identity", identity4.to_vec()),
        ("zero", vec![0; 16]),
        ("all-max", vec![u32::MAX; 16]),
        ("nilpotent", nilpotent4.to_vec()),
    ] {
        all.push(case(
            format!("{name} d=4 x5000"),
            Family::Matrix(4),
            true,
            word.repeat(5000),
            seed4.clone(),
        ));
    }
    let identity16 = (0..16).collect::<Vec<u32>>();
    all.push(case(
        "identity K=16 x5000".into(),
        Family::Table(16),
        true,
        identity16.repeat(5000),
        vec![9],
    ));
    all
}
