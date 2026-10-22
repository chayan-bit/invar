//! Pure integer D5 oracle, also embedded in standalone native hosts.
/// Largest admitted inner extent: the bound's derivation needs
/// `(9/8 * K + 2) * 2^-23` small enough (see the amendment); the i128
/// accumulator alone would be exact up to 2^46 terms.
pub const MAX_INNER: u32 = 1 << 20;
/// Round finite binary32 bits to the nearest binary16 value (ties to even),
/// returned as the binary32 bits of that value. `None` for an infinity, a NaN
/// or a value that rounds past 65504 (binary16 overflow).
pub fn f16_round(word: u32) -> Option<u32> {
    let sign = word & 0x8000_0000;
    let magnitude = word & 0x7fff_ffff;
    if magnitude >= 0x7f80_0000 {
        return None;
    }
    if magnitude == 0 {
        return Some(word);
    }
    let biased = (magnitude >> 23) as i32;
    let (significand, exponent) = match biased {
        0 => (magnitude, -149),
        _ => ((magnitude & 0x007f_ffff) | 0x0080_0000, biased - 150),
    };
    let top = 31 - significand.leading_zeros() as i32 + exponent;
    // Binary16 spacing at this magnitude: 2^(max(top, -14) - 10).
    let quantum = top.max(-14) - 10;
    let shift = quantum - exponent;
    let q = if shift <= 0 {
        u64::from(significand) << -shift
    } else if shift > 25 {
        0
    } else {
        let kept = u64::from(significand >> shift);
        let rest = significand & ((1u32 << shift) - 1);
        let half = 1u32 << (shift - 1);
        kept + u64::from(rest > half || (rest == half && kept & 1 == 1))
    };
    if q == 0 {
        return Some(sign);
    }
    // q * 2^quantum, at most 2048 * 2^5 = 65536 before the overflow check.
    let msb = 63 - q.leading_zeros() as i32;
    let value_exponent = msb + quantum;
    // q < 2^11 except a rounding carry, so exponent 15 tops out at 65504.
    if value_exponent > 15 {
        return None;
    }
    let mantissa = ((q << (23 - msb)) as u32) & 0x007f_ffff;
    Some(sign | (((value_exponent + 127) as u32) << 23) | mantissa)
}

/// A refused input element (an infinity, a NaN or a value that rounds past
/// 65504): slot 0 = A, 1 = B.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct InputRange {
    pub input: usize,
    pub index: usize,
}

impl std::fmt::Display for InputRange {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "tc-f16-f32-v1 input {} element {} outside the binary16 range",
            self.input, self.index
        )
    }
}

/// Round every element to binary16 (as binary32 words), refusing the first
/// element outside the binary16 finite range. Binary16 subnormals are
/// admitted (the CUDA schedule adds their products on CUDA cores).
pub fn round_inputs(input: usize, words: &[u32]) -> Result<Vec<u32>, InputRange> {
    words
        .iter()
        .enumerate()
        .map(|(index, &w)| f16_round(w).ok_or(InputRange { input, index }))
        .collect()
}

/// A binary16-valued binary32 word as an exact integer multiple of 2^-24.
fn units(word: u32) -> i64 {
    let magnitude = word & 0x7fff_ffff;
    if magnitude == 0 {
        return 0;
    }
    let significand = i64::from((magnitude & 0x007f_ffff) | 0x0080_0000);
    // Binary16 values are normal binary32 multiples of 2^-24, so this is exact.
    let shift = (magnitude >> 23) as i32 - 150 + 24;
    let value = if shift >= 0 {
        significand << shift
    } else {
        significand >> -shift
    };
    if word >> 31 == 1 { -value } else { value }
}

/// Round an exact multiple of 2^-48 to the nearest binary32 (ties to even).
/// Admitted sums are below 2^52 in magnitude and nonzero sums are at least
/// 2^-48, so the result is always a normal number or +0.
pub(super) fn round_units48(value: i128) -> u32 {
    if value == 0 {
        return 0;
    }
    let sign = if value < 0 { 0x8000_0000 } else { 0 };
    let m = value.unsigned_abs();
    let length = 128 - m.leading_zeros() as i32;
    let mut exponent = length - 1 - 48;
    let mut significand = if length <= 24 {
        m << (24 - length)
    } else {
        let shift = (length - 24) as u32;
        let kept = m >> shift;
        let rest = m & ((1u128 << shift) - 1);
        let half = 1u128 << (shift - 1);
        kept + u128::from(rest > half || (rest == half && kept & 1 == 1))
    };
    if significand == 1 << 24 {
        significand >>= 1;
        exponent += 1;
    }
    sign | (((exponent + 127) as u32) << 23) | ((significand as u32) & 0x007f_ffff)
}

/// Why a result set fails the amended D5 check.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Failure {
    Shape,
    Input(InputRange),
    /// `result` is outside the bound around `exact` (the rounded exact sum).
    Cell {
        cell: usize,
        result: u32,
        exact: u32,
        ratio: f64,
    },
}

/// Passing evidence: the largest observed `|r - exact| / bound`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Report {
    pub cells: usize,
    pub max_ratio: f64,
}

struct Rounded {
    a: Vec<i64>,
    b: Vec<i64>,
}

fn rounded([rows, inner, cols]: [u32; 3], a: &[u32], b: &[u32]) -> Result<Rounded, Failure> {
    let [rows, inner, cols] = [rows, inner, cols].map(|n| n as usize);
    if inner > MAX_INNER as usize
        || a.len() != rows.checked_mul(inner).ok_or(Failure::Shape)?
        || b.len() != inner.checked_mul(cols).ok_or(Failure::Shape)?
    {
        return Err(Failure::Shape);
    }
    let a = round_inputs(0, a).map_err(Failure::Input)?;
    let b = round_inputs(1, b).map_err(Failure::Input)?;
    Ok(Rounded {
        a: a.into_iter().map(units).collect(),
        b: b.into_iter().map(units).collect(),
    })
}

/// Exact sum and absolute sum of one cell's products, in units of 2^-48.
fn cell(r: &Rounded, [_, inner, cols]: [usize; 3], i: usize, j: usize) -> (i128, i128) {
    (0..inner).fold((0, 0), |(sum, magnitude), k| {
        let p = i128::from(r.a[i * inner + k]) * i128::from(r.b[k * cols + j]);
        (sum + p, magnitude + p.abs())
    })
}

/// The reference result: each cell's exact sum of binary16-rounded products,
/// rounded once to nearest even (`+0` for an exact zero). Dense row-major
/// `[rows, inner] x [inner, cols]` inputs.
pub fn reference(dims: [u32; 3], a: &[u32], b: &[u32]) -> Result<Vec<u32>, Failure> {
    let r = rounded(dims, a, b)?;
    let d = dims.map(|n| n as usize);
    Ok((0..d[0] * d[2])
        .map(|c| round_units48(cell(&r, d, c / d[2], c % d[2]).0))
        .collect())
}

/// Floor and ceiling of a finite binary32 value times 2^71, or `None` when its
/// magnitude is at least 2^126 (beyond every admissible result).
fn scaled71(word: u32) -> Option<(i128, i128)> {
    let magnitude = word & 0x7fff_ffff;
    let biased = (magnitude >> 23) as i32;
    let (m, q) = match biased {
        0 => (i128::from(magnitude), -149),
        _ => (
            i128::from((magnitude & 0x007f_ffff) | 0x0080_0000),
            biased - 150,
        ),
    };
    let shift = q + 71;
    let (floor, ceil) = if shift >= 0 {
        if shift > 102 {
            return None;
        }
        (m << shift, m << shift)
    } else if -shift > 24 {
        (0, i128::from(m != 0))
    } else {
        let kept = m >> -shift;
        (kept, kept + i128::from(m & ((1 << -shift) - 1) != 0))
    };
    Some(if word >> 31 == 1 {
        (-ceil, -floor)
    } else {
        (floor, ceil)
    })
}

/// Check `out` (dense row-major `[rows, cols]` binary32 words) against the
/// amended D5 bound `|r - exact| <= 2 * K * 2^-23 * sum |p|`, exactly: both
/// sides are integers at scale 2^-71 except `r`, handled by floor/ceiling.
pub fn verify(dims: [u32; 3], a: &[u32], b: &[u32], out: &[u32]) -> Result<Report, Failure> {
    let r = rounded(dims, a, b)?;
    let d = dims.map(|n| n as usize);
    if out.len() != d[0] * d[2] {
        return Err(Failure::Shape);
    }
    let k = i128::from(dims[1]);
    let mut max_ratio = 0.0f64;
    for (c, &word) in out.iter().enumerate() {
        let (sum, magnitude) = cell(&r, d, c / d[2], c % d[2]);
        let (centre, slack) = (sum << 23, 2 * k * magnitude);
        let exact = round_units48(sum);
        let error = (f32::from_bits(word) as f64 - sum as f64 * 2f64.powi(-48)).abs();
        let bound = slack as f64 * 2f64.powi(-71);
        let ratio = if bound > 0.0 {
            error / bound
        } else if error == 0.0 {
            0.0
        } else {
            f64::INFINITY
        };
        let within = word & 0x7f80_0000 != 0x7f80_0000
            && scaled71(word)
                .is_some_and(|(floor, ceil)| floor >= centre - slack && ceil <= centre + slack);
        if !within {
            return Err(Failure::Cell {
                cell: c,
                result: word,
                exact,
                ratio,
            });
        }
        max_ratio = max_ratio.max(ratio);
    }
    Ok(Report {
        cells: out.len(),
        max_ratio,
    })
}
