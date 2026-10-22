//! Shared M4-03 tiled-contraction corpus: the direct-cell source family,
//! adversarial shapes, binary32 word patterns and two independent oracles.
//! Used by the no-device CPU suite and both hardware suites.
#![allow(dead_code)]

use invar::numeric_contract::F32Bits;

pub const DIRECT: &str = "fn mat(a: read Array<f32>, b: read Array<f32>, rows: u32, inner: u32, cols: u32) -> Array<f32> { var c = filled_f32(rows * cols,0.0); for i in 0u32..rows { for j in 0u32..cols { for k in 0u32..inner { c[i * cols + j] = c[i * cols + j] + a[i * inner + k] * b[k * cols + j]; } } } return move c; }";

/// One, 1xK by Kx1, skinny, K = 0, exact tiles, and every non-multiple of the
/// 64x64x16 tile (edges in rows, columns and k at once).
pub const SHAPES: [[u32; 3]; 14] = [
    [1, 1, 1],
    [1, 37, 1],
    [1, 300, 77],
    [77, 300, 1],
    [70, 1, 70],
    [5, 0, 7],
    [64, 16, 64],
    [65, 17, 63],
    [63, 33, 65],
    [17, 13, 33],
    [128, 64, 128],
    [129, 31, 130],
    [2, 3, 130],
    [130, 3, 2],
];

/// Historical large regression shapes; the 2^28 boundary has separate tests.
pub const LARGEST: [[u32; 3]; 2] = [[256, 1024, 256], [1000, 67, 1000]];

/// Normal-range words whose products and partial sums stay normal or exactly
/// zero (signed zeros, cancellation, rounding at 1 + 2^-23).
pub const NORMAL: [u32; 8] = [
    0x3f80_0000, // 1
    0xbfc0_0000, // -1.5
    0x3fc0_0000, // 1.5
    0x3f00_0000, // 0.5
    0x8000_0000, // -0
    0x4010_0000, // 2.25
    0x3f80_0001, // 1 + 2^-23
    0xc040_0000, // -3
];

/// Gradual underflow: subnormal inputs and subnormal results.
pub const SUBNORMAL: [u32; 8] = [
    0x0000_0001, // 2^-149
    0x3f80_0000, // 1
    0x0080_0000, // 2^-126
    0x3f00_0000, // 0.5
    0x8000_0003, // -3 * 2^-149
    0x3fc0_0000, // 1.5
    0x007f_ffff, // largest subnormal
    0x8000_0000, // -0
];

/// Infinities and a NaN: IEEE propagation inside the kernel; device runtimes
/// refuse them before submission and the source loop keeps the CPU result.
pub const NONFINITE: [u32; 6] = [
    0x7f80_0000,
    0x3f80_0000,
    0xff80_0000,
    0x0000_0000,
    0x7fc1_2345,
    0x7f7f_ffff,
];

pub fn words(pattern: &[u32], len: usize, stride: usize) -> Vec<u32> {
    (0..len)
        .map(|i| pattern[(i * stride + 1) % pattern.len()])
        .collect()
}

/// Pseudo-random normal words in [-2, 2) with full 23-bit mantissas, from a
/// fixed xorshift seed, so large shapes exercise rounding in every cell.
pub fn random_words(len: usize, seed: u64) -> Vec<u32> {
    let mut state = seed | 1;
    (0..len)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let mantissa = (state >> 20) as u32 & 0x007f_ffff;
            let sign = ((state >> 50) as u32 & 1) << 31;
            sign | 0x3f80_0000 | mantissa
        })
        .collect()
}

/// Software binary32 oracle: exact integer/rational F32Bits operations in
/// increasing k with a separately rounded multiply then add, from +0.
pub fn oracle([rows, inner, cols]: [u32; 3], a: &[u32], b: &[u32]) -> Vec<u32> {
    let [rows, inner, cols] = [rows, inner, cols].map(|n| n as usize);
    let mut out = Vec::with_capacity(rows * cols);
    for i in 0..rows {
        for j in 0..cols {
            let mut sum = F32Bits::POS_ZERO;
            for k in 0..inner {
                let product =
                    F32Bits::from_bits(a[i * inner + k]).mul(F32Bits::from_bits(b[k * cols + j]));
                sum = sum.add(product);
            }
            out.push(sum.to_bits());
        }
    }
    out
}

/// Host IEEE oracle for the large shapes: Rust never contracts `a * b + c`
/// into an FMA and keeps gradual underflow, so this is the same contract
/// computed by the host FPU, independent of every generated kernel.
pub fn host_oracle([rows, inner, cols]: [u32; 3], a: &[u32], b: &[u32]) -> Vec<u32> {
    let [rows, inner, cols] = [rows, inner, cols].map(|n| n as usize);
    let mut out = Vec::with_capacity(rows * cols);
    for i in 0..rows {
        for j in 0..cols {
            let mut sum = 0.0f32;
            for k in 0..inner {
                let product = f32::from_bits(a[i * inner + k]) * f32::from_bits(b[k * cols + j]);
                sum += product;
            }
            out.push(sum.to_bits());
        }
    }
    out
}

/// Bitwise equality except that every NaN equals every NaN (D1/D2).
pub fn strict_equal(actual: &[u32], expected: &[u32]) -> bool {
    actual.len() == expected.len()
        && actual
            .iter()
            .zip(expected)
            .all(|(&x, &y)| F32Bits::from_bits(x).strict_eq(F32Bits::from_bits(y)))
}

/// FMA discriminator: every cell is 1 * -1 + (1 + 2^-23)(1 - 2^-23). Separate
/// roundings give exactly +0; a fused second step keeps -2^-46.
pub fn fma_discriminator(rows: u32, cols: u32) -> ([u32; 3], Vec<u32>, Vec<u32>) {
    let a = (0..rows).flat_map(|_| [0x3f80_0000, 0x3f80_0001]).collect();
    let b = [0xbf80_0000u32, 0x3f7f_fffe]
        .iter()
        .flat_map(|&w| std::iter::repeat_n(w, cols as usize))
        .collect();
    ([rows, 2, cols], a, b)
}

pub fn is_subnormal(word: u32) -> bool {
    word & 0x7f80_0000 == 0 && word & 0x007f_ffff != 0
}

// --- tc-f16-f32-v1 (D5 as amended 2026-10-14) ---------------------------

/// The same direct-cell family with the explicit Tensor Core step.
pub const TC_DIRECT: &str = "fn mat(a: read Array<f32>, b: read Array<f32>, rows: u32, inner: u32, cols: u32) -> Array<f32> { var c = filled_f32(rows * cols,0.0); for i in 0u32..rows { for j in 0u32..cols { for k in 0u32..inner { c[i * cols + j] = tc_mul_add(a[i * inner + k], b[k * cols + j], c[i * cols + j]); } } } return move c; }";

/// Pseudo-random words spread over the binary16 range: magnitudes from
/// 2^-26 (rounds to 0 or the smallest subnormal) through binary16 subnormals
/// and normals below 2^15, with full binary32 mantissas so every element
/// rounds, and both signs.
pub fn tc_words(len: usize, seed: u64) -> Vec<u32> {
    words_in(len, seed, 101, 41)
}

/// Like `tc_words`, with normal binary16 magnitudes only (2^-14 .. 2^14).
pub fn tc_normal_words(len: usize, seed: u64) -> Vec<u32> {
    words_in(len, seed, 113, 29)
}

/// Binary16 subnormals `n x 2^-24`, `n` in 1..=1023, both signs.
pub fn tc_subnormal_words(len: usize, seed: u64) -> Vec<u32> {
    (0..len as u64)
        .map(|i| {
            let n = (i * 7919 + seed * 104_729) % 1023 + 1;
            let v = n as f32 * 2f32.powi(-24);
            (if (i + seed).is_multiple_of(3) { -v } else { v }).to_bits()
        })
        .collect()
}

fn words_in(len: usize, seed: u64, low: u32, span: u64) -> Vec<u32> {
    let mut state = seed | 1;
    (0..len)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let mantissa = (state >> 20) as u32 & 0x007f_ffff;
            let sign = ((state >> 50) as u32 & 1) << 31;
            // Biased exponents low..low + span: 2^-26 .. 2^14 for `tc_words`.
            let exponent = low + ((state >> 44) % span) as u32;
            sign | exponent << 23 | mantissa
        })
        .collect()
}

/// The D5 counterexample, all binary16 values: `a = b = [1, 2^-12 x (K - 1)]`,
/// so the products are 1 and K - 1 copies of 2^-24. Source order gives 1
/// (every `1 + 2^-24` ties to even); the exact sum is `1 + (K - 1) 2^-24`.
pub fn tc_counterexample(k: u32) -> ([u32; 3], Vec<u32>, Vec<u32>) {
    let mut a = vec![0x3980_0000; k as usize]; // 2^-12
    a[0] = 0x3f80_0000;
    (([1, k, 1]), a.clone(), a)
}

/// Host f32 summation of binary16-rounded products in a chosen order, as a
/// device might accumulate: `tile` products are summed first, then added to
/// the running accumulator in source order (tile 1 = plain source order);
/// `reverse` sums each tile from its last product. `truncate` rounds every
/// addition toward zero instead of to nearest.
pub fn tc_ordered(
    [rows, inner, cols]: [u32; 3],
    a: &[u32],
    b: &[u32],
    tile: usize,
    reverse: bool,
    truncate: bool,
) -> Vec<u32> {
    let [rows, inner, cols] = [rows, inner, cols].map(|n| n as usize);
    let round = |w: u32| f32::from_bits(invar_device::tc::f16_round(w).expect("binary16 range"));
    let add = |x: f32, y: f32| {
        let s = x + y;
        if !truncate || s == 0.0 {
            return s;
        }
        // The exact sum's error term; step toward zero when s rounded away.
        let (big, small) = if x.abs() >= y.abs() { (x, y) } else { (y, x) };
        let error = small - (s - big);
        if error != 0.0 && (error > 0.0) != (s > 0.0) {
            f32::from_bits(s.to_bits() - 1)
        } else {
            s
        }
    };
    let mut out = Vec::with_capacity(rows * cols);
    for i in 0..rows {
        for j in 0..cols {
            let mut acc = 0.0f32;
            for k0 in (0..inner).step_by(tile.max(1)) {
                let mut part = 0.0f32;
                let span = k0..(k0 + tile.max(1)).min(inner);
                let ks: Vec<usize> = if reverse {
                    span.rev().collect()
                } else {
                    span.collect()
                };
                for k in ks {
                    part = add(part, round(a[i * inner + k]) * round(b[k * cols + j]));
                }
                acc = add(acc, part);
            }
            out.push(acc.to_bits());
        }
    }
    out
}

/// Hardware cases for `tc-f16-f32-v1`: the counterexample at K = 16, 32,
/// 4096 and the family's max K, every adversarial shape with binary16-range
/// random words, the binary16-subnormal cases of the second D5 amendment
/// (the sm_120 pair, a leading-zero sweep, a subnormal at every k position,
/// all-subnormal and mixed tiles, subnormals at K = 16, 32, 4096 and max),
/// near-total cancellation, and the largest admitted shapes.
/// A labelled `tc-f16-f32-v1` case: dimensions, A words, B words.
pub type TcCase = (String, [u32; 3], Vec<u32>, Vec<u32>);

const F16_65504: u32 = 0x477f_e000;
/// 1.9990234375 x 2^-12, the largest binary16 below 2^-11.
const F16_SMALL: u32 = 0x39ff_e000;

fn pow2(e: i32) -> u32 {
    2f32.powi(e).to_bits()
}

pub fn tc_cases() -> Vec<TcCase> {
    let mut cases: Vec<TcCase> = Vec::new();
    for k in [16, 32, 4096, 1_000_000] {
        let (dims, a, b) = tc_counterexample(k);
        cases.push(("counterexample".into(), dims, a, b));
    }
    for (n, dims) in SHAPES.iter().enumerate() {
        let [r, k, c] = dims.map(|v| v as usize);
        let a = tc_words(r * k, 31 + n as u64);
        let b = tc_words(k * c, 131 + n as u64);
        cases.push(("random".into(), *dims, a, b));
    }
    // The sm_120 pair: 2^-22 x -20048 and 2^-23 x -0.2342529296875, whose
    // second product the Tensor Core alone drops (ratio 1.55 in a full cell).
    cases.push((
        "subnormal-sm120-pair".into(),
        [1, 2, 1],
        vec![pow2(-22), pow2(-23)],
        vec![0xc69c_a000, 0xbe6f_e000],
    ));
    // Leading-zero sweep: row i leads with 2^(i - 24) x 65504 (10 - i leading
    // zeros in the subnormal operand, row 10 normal) over fifteen products of
    // 1.999 x 2^-12 squared just below the Tensor Core's window.
    let mut a = vec![F16_SMALL; 11 * 16];
    for i in 0..11 {
        a[i * 16] = pow2(i as i32 - 24);
    }
    let mut b = vec![F16_SMALL; 16];
    b[0] = F16_65504;
    cases.push(("subnormal-leading-zero-sweep".into(), [11, 16, 1], a, b));
    // The worst lead (2^-24 x 65504) at every k position of a k-tile.
    for p in 0..16 {
        let (mut a, mut b) = (vec![F16_SMALL; 16], vec![F16_SMALL; 16]);
        a[p] = pow2(-24);
        b[p] = F16_65504;
        cases.push((format!("subnormal-at-k{p}"), [1, 16, 1], a, b));
    }
    // All-subnormal tiles: A subnormal over random B, and both subnormal.
    let dims = [16, 16, 8];
    let a = tc_subnormal_words(256, 1);
    cases.push(("subnormal-tile-a".into(), dims, a.clone(), tc_words(128, 3)));
    cases.push((
        "subnormal-tile-ab".into(),
        dims,
        a,
        tc_subnormal_words(128, 2),
    ));
    // Mixed tiles: A is subnormal on a checkerboard of 16x16 tiles and
    // normal elsewhere, B normal, so some k-tiles need the correction.
    let dims = [48, 64, 24];
    let normal = tc_normal_words(48 * 64, 41);
    let sub = tc_subnormal_words(48 * 64, 43);
    let a = (0..48 * 64usize)
        .map(|e| {
            let (i, k) = (e / 64, e % 64);
            if (i / 16 + k / 16).is_multiple_of(2) {
                sub[e]
            } else {
                normal[e]
            }
        })
        .collect();
    cases.push((
        "subnormal-mixed-tiles".into(),
        dims,
        a,
        tc_normal_words(64 * 24, 47),
    ));
    // Subnormals in every k-tile at K = 16, 32, 4096 and the family's max.
    for (dims, seed) in [
        ([2, 16, 3], 51),
        ([2, 32, 3], 53),
        ([2, 4096, 3], 55),
        ([1, 1_000_000, 1], 57),
    ] {
        let [r, k, c] = dims.map(|v| v as usize);
        let normal = tc_normal_words(r * k, seed);
        let sub = tc_subnormal_words(r * k, seed + 1);
        let a = (0..r * k)
            .map(|e| {
                if e.is_multiple_of(3) {
                    sub[e]
                } else {
                    normal[e]
                }
            })
            .collect();
        cases.push((
            "subnormal-k".into(),
            dims,
            a,
            tc_normal_words(k * c, seed + 2),
        ));
    }
    // Binary16 subnormal factors (2^-24 .. 2^-15) times normals.
    let dims = [33, 70, 17];
    let a = (0..33 * 70)
        .map(|i| 0x3380_0000 + ((i as u32 * 7919) % 512) * 0x0001_0000)
        .collect();
    let b = tc_words(70 * 17, 5);
    cases.push(("f16-subnormal".into(), dims, a, b));
    // Cancellation: +-2^14 pairs around small residues, sum |p| far above |sum|.
    let dims = [16, 64, 8];
    let a = (0..16 * 64)
        .map(|i| match i % 4 {
            0 => 0x4680_0000, // 16384
            1 => 0xc680_0000, // -16384
            _ => 0x3a00_0000 + (i as u32 % 97) * 0x0000_2000,
        })
        .collect();
    let b = (0..64 * 8u32)
        .map(|i| {
            if i.is_multiple_of(3) {
                0x3f80_0000
            } else {
                0x3f80_2000
            }
        })
        .collect();
    cases.push(("cancellation".into(), dims, a, b));
    for dims in LARGEST {
        let [r, k, c] = dims.map(|v| v as usize);
        cases.push((
            "largest".into(),
            dims,
            tc_words(r * k, 77),
            tc_words(k * c, 79),
        ));
    }
    cases
}
