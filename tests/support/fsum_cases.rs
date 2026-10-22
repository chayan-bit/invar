//! Shared adversarial inputs for the v0.2 deterministic reduction tests.
#![allow(dead_code)]

pub const CANONICAL_NAN: u32 = 0x7fc0_0000;
pub const MAX: u32 = 0x7f7f_ffff;
pub const INF: u32 = 0x7f80_0000;
pub const NEG: u32 = 0x8000_0000;

pub fn bits(value: f32) -> u32 {
    value.to_bits()
}

/// SplitMix64: deterministic, dependency-free test randomness.
pub struct Rng(pub u64);

impl Rng {
    pub fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    /// A binary32 word biased towards hard cases: subnormals, zeros, extreme
    /// exponents, near-cancelling magnitudes and rare infinities and NaNs.
    pub fn word(&mut self) -> u32 {
        let raw = self.next();
        let sign = (raw as u32) & NEG;
        let fraction = (raw >> 8) as u32 & 0x007f_ffff;
        match raw >> 58 {
            0 => sign | INF,
            1 => 0x7f80_0000 | fraction | 1,
            2..=5 => sign | fraction,
            6 | 7 => sign,
            8..=11 => sign | (((raw >> 40) as u32 % 6 + 248) << 23) | fraction,
            12..=19 => sign | ((127 + (raw >> 40) as u32 % 3) << 23) | fraction,
            _ => sign | (((raw >> 40) as u32 % 254 + 1) << 23) | fraction,
        }
    }

    /// A finite, mostly normal word with a narrow exponent spread.
    pub fn finite(&mut self) -> u32 {
        let raw = self.next();
        ((raw as u32) & NEG)
            | ((120 + (raw >> 40) as u32 % 16) << 23)
            | ((raw >> 8) as u32 & 0x7f_ffff)
    }

    pub fn shuffle<T>(&mut self, items: &mut [T]) {
        for index in (1..items.len()).rev() {
            let other = (self.next() % (index as u64 + 1)) as usize;
            items.swap(index, other);
        }
    }
}

fn ones(n: usize) -> Vec<u32> {
    vec![bits(1.0); n]
}

/// (label, a, b) with `a.len() == b.len()`.
pub fn adversarial_vectors() -> Vec<(String, Vec<u32>, Vec<u32>)> {
    let mut cases: Vec<(String, Vec<u32>)> = vec![
        ("empty".into(), vec![]),
        ("single negative zero".into(), vec![NEG]),
        ("single subnormal".into(), vec![1]),
        ("single NaN payload".into(), vec![0xffc0_1234]),
        ("single max".into(), vec![MAX]),
        (
            "cancellation".into(),
            vec![bits(1e30), bits(1.0), bits(-1e30)],
        ),
        (
            "cancellation to tiny".into(),
            vec![bits(3.0), 0x3380_0000, bits(-3.0)],
        ),
        ("dynamic range".into(), vec![MAX, 1, NEG | MAX, 1, 1]),
        (
            "all subnormal".into(),
            (1..=40).map(|i| i * 0x0001_0101).collect(),
        ),
        (
            "subnormals cross normal".into(),
            vec![0x007f_ffff, 1, 0x007f_ffff, NEG | 2],
        ),
        ("half-ulp tie to even".into(), vec![bits(1.0), 0x3380_0000]),
        ("half-ulp tie odd".into(), vec![0x3f80_0001, 0x3380_0000]),
        (
            "tie broken by sticky".into(),
            vec![bits(1.0), 0x3380_0000, 1],
        ),
        ("overflow".into(), vec![MAX, MAX]),
        (
            "overflow cancels".into(),
            vec![MAX, MAX, NEG | MAX, NEG | MAX],
        ),
        ("inf and nan".into(), vec![INF, bits(2.0), 0x7fc0_0001]),
        ("opposite inf".into(), vec![INF, NEG | INF]),
        ("mixed zeros".into(), vec![NEG, 0, NEG]),
        ("negative zeros".into(), vec![NEG; 17]),
        ("negative zeros n=129".into(), vec![NEG; 129]),
    ];
    for n in [7, 8, 9, 15, 16, 17, 31, 63, 64, 65, 129] {
        let mut values = ones(n);
        values[n / 2] = bits(16_777_216.0);
        cases.push((format!("ulp-sensitive ones n={n}"), values));
    }
    let mut rng = Rng(0xf5);
    for n in [1000usize, 4096, 10_007] {
        cases.push((
            format!("random n={n}"),
            (0..n).map(|_| rng.word()).collect(),
        ));
        let half = (0..n / 2).map(|_| rng.finite()).collect::<Vec<_>>();
        let mut paired = half.iter().flat_map(|&x| [x, x ^ NEG]).collect::<Vec<_>>();
        paired.push(1);
        rng.shuffle(&mut paired);
        cases.push((format!("exact cancellation n={}", paired.len()), paired));
    }
    cases
        .into_iter()
        .map(|(label, a)| {
            let b = a
                .iter()
                .enumerate()
                .map(|(i, &x)| match i % 5 {
                    0 => bits(1.0),
                    1 => x ^ NEG,
                    2 => 0x3f80_0001,
                    3 => rng.word(),
                    _ => 0,
                })
                .collect::<Vec<_>>();
            (label, a, b)
        })
        .collect()
}
