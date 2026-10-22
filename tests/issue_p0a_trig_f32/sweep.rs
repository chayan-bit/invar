//! Shared exhaustive-sweep helpers for `invar-trig-f32-v1`: a stable digest
//! (FNV-1a 64 over the little-endian result words of each 2^24-input block,
//! then over the 256 block digests in order), the libm-based correctly rounded
//! reference and the subnormal audit backend.
#![allow(dead_code)]
use invar_device::trig_f32::{self, Arithmetic, Host};

pub const BLOCK_BITS: u32 = 24;
pub const BLOCKS: u32 = 1 << (32 - BLOCK_BITS);
pub const FUNCTIONS: [&str; 3] = ["sin", "cos", "sqrt"];

pub fn fnv(hash: u64, word: u64, bytes: usize) -> u64 {
    (0..bytes).fold(hash, |h, i| {
        (h ^ ((word >> (8 * i)) & 0xff)).wrapping_mul(0x0000_0100_0000_01b3)
    })
}
pub const FNV_SEED: u64 = 0xcbf2_9ce4_8422_2325;

/// Fold per-block digests (in block order) into the sweep digest.
pub fn combine(blocks: &[u64]) -> u64 {
    blocks.iter().fold(FNV_SEED, |h, &b| fnv(h, b, 8))
}

pub fn host(function: usize, x: u32) -> u32 {
    match function {
        0 => trig_f32::host_sin(x),
        1 => trig_f32::host_cos(x),
        _ => trig_f32::host_sqrt(x),
    }
}

pub fn threads() -> usize {
    std::env::var("INVAR_TRIG_THREADS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or_else(|| std::thread::available_parallelism().map_or(1, |n| n.get()))
        .max(1)
}

/// Monotone integer key: adjacent finite binary32 values differ by one.
pub fn key(bits: u32) -> i64 {
    if bits & 0x8000_0000 != 0 {
        -i64::from(bits & 0x7fff_ffff)
    } else {
        i64::from(bits)
    }
}

/// The correctly rounded binary32 result from an f64 libm value, or `None`
/// when the f64 value lies within two f64 ulps of a binary32 rounding
/// boundary (libm's < 1 ulp error could then flip the rounding).
pub fn correctly_rounded(reference: f64) -> Option<u32> {
    let rounded = reference as f32;
    let bits = rounded.to_bits();
    let ulp64 = f64::from_bits(reference.abs().to_bits() + 1) - reference.abs();
    for neighbour in [bits.wrapping_sub(1), bits.wrapping_add(1)] {
        let other = f32::from_bits(neighbour);
        if !other.is_finite() || other.signum() != rounded.signum() && rounded != 0.0 {
            continue;
        }
        let midpoint = (f64::from(rounded) + f64::from(other)) / 2.0;
        if (reference - midpoint).abs() <= 2.0 * ulp64 {
            return None;
        }
    }
    Some(bits)
}

/// Spacing of binary32 values at |v| (2^-149 below the normal range).
pub fn ulp32(v: f64) -> f64 {
    let a = v.abs();
    if a < f64::from(f32::MIN_POSITIVE) {
        return 2f64.powi(-149);
    }
    2f64.powi(a.log2().floor() as i32 - 23)
}

/// The host backend that also records the smallest nonzero magnitude any
/// floating operand, result or exact fma product reaches, and every
/// operation that rounds a nonzero exact value to zero.
#[derive(Clone, Copy, Debug)]
pub struct Audit {
    pub smallest: f64,
    pub smallest_input: u32,
    pub underflows: u64,
    pub input: u32,
}

impl Default for Audit {
    fn default() -> Self {
        Self {
            smallest: f64::INFINITY,
            smallest_input: 0,
            underflows: 0,
            input: 0,
        }
    }
}

impl Audit {
    fn note(&mut self, values: &[f64], result: u32, exact_nonzero: bool) {
        for &v in values {
            let a = v.abs();
            if a != 0.0 && a.is_finite() && a < self.smallest {
                self.smallest = a;
                self.smallest_input = self.input;
            }
        }
        if exact_nonzero && f32::from_bits(result) == 0.0 {
            self.underflows += 1;
        }
    }
    pub fn merge(&mut self, other: &Self) {
        if other.smallest < self.smallest {
            self.smallest = other.smallest;
            self.smallest_input = other.smallest_input;
        }
        self.underflows += other.underflows;
    }
}

fn v(w: u32) -> f64 {
    f64::from(f32::from_bits(w))
}

impl Arithmetic for Audit {
    fn add(&mut self, a: u32, b: u32) -> u32 {
        let r = Host.add(a, b);
        self.note(&[v(a), v(b), v(r)], r, v(a) + v(b) != 0.0);
        r
    }
    fn sub(&mut self, a: u32, b: u32) -> u32 {
        let r = Host.sub(a, b);
        self.note(&[v(a), v(b), v(r)], r, v(a) - v(b) != 0.0);
        r
    }
    fn mul(&mut self, a: u32, b: u32) -> u32 {
        let r = Host.mul(a, b);
        // The f64 product of two binary32 values is exact.
        self.note(&[v(a), v(b), v(a) * v(b), v(r)], r, v(a) * v(b) != 0.0);
        r
    }
    fn fused(&mut self, a: u32, b: u32, c: u32) -> u32 {
        let r = Host.fused(a, b, c);
        let product = v(a) * v(b);
        let exact_nonzero = product != -v(c);
        self.note(&[v(a), v(b), v(c), product, v(r)], r, exact_nonzero);
        r
    }
    fn root(&mut self, a: u32) -> u32 {
        let r = Host.root(a);
        self.note(&[v(a), v(r)], r, false);
        r
    }
    fn convert(&mut self, a: u32) -> u32 {
        let r = Host.convert(a);
        self.note(&[v(r)], r, false);
        r
    }
}
