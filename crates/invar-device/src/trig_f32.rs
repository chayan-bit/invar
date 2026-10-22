//! `invar-trig-f32-v1`: the frozen strict binary32 `sin`, `cos` and `sqrt`.
//!
//! This module is the single source of truth for the operation sequence.
//! [`sincos`] and [`sqrt`] are written once against [`Ops`]; every engine
//! replays them through a backend: the root's exact `F32Bits` oracle, the
//! [`Host`] evaluator, LLVM text for native CPU code, CUDA C and MSL. No
//! emitter spells the algorithm itself, so no emitter can drift.
//!
//! Source (attributed, see `docs/trig-f32-v1.md`): Arm optimized-routines,
//! AdvSIMD single-precision `sincosf`, commit
//! `97362b2eab91d35d30c4ac2c8c79dcd4dae4331b`,
//! `math/aarch64/advsimd/v_sincosf_common.h` (fast path, |x| <= 2^20) and
//! `math/aarch64/advsimd/v_trigf_fallback.h` (large-argument reduction),
//! Copyright (c) 2023-2026 Arm Limited, MIT OR Apache-2.0 WITH LLVM-exception.
//! Constants and the order of every rounding are kept; the deviations are an
//! explicit tiny-argument path, canonical NaN for NaN and infinite inputs, and
//! integer spellings of exact conversions (each listed where it occurs).
//!
//! Every floating operation is one IEEE 754 binary32 basic operation (`+`,
//! `-`, `*`, `sqrt`, exact `u32` conversion) or an explicit fused `fma`, each
//! rounded to nearest even. Everything else is exact 32-bit integer work on
//! encodings. The exhaustive audit in `docs/trig-f32-v1.md` shows that no
//! floating operation of this sequence reads or produces a subnormal for any
//! input, so flush-to-zero devices compute the same bits.

pub const CONTRACT: &str = "invar-trig-f32-v1";
/// Canonical quiet NaN returned for NaN inputs, infinite trig inputs and
/// negative nonzero `sqrt` inputs. NaN payloads are not propagated.
pub const CANONICAL_NAN: u32 = 0x7fc0_0000;
pub const SIN_HELPER: &str = "invar_trig_f32_v1";
pub const SQRT_HELPER: &str = "invar_sqrt_f32_v1";

const SIGN: u32 = 0x8000_0000;
const ABS: u32 = 0x7fff_ffff;
const INFINITY: u32 = 0x7f80_0000;
const MIN_NORMAL: u32 = 0x0080_0000;
const ONE: u32 = 0x3f80_0000;
const MINUS_HALF: u32 = 0xbf00_0000;
/// 2^-12: below it the correctly rounded sin(x) is x and cos(x) is 1.
const TINY: u32 = 0x3980_0000;
/// 2^20, the upper bound (inclusive) of the Arm fast path.
const RANGE: u32 = 0x4980_0000;
/// 0x1.8p23: adding it rounds |v| < 2^22 to an integer held in the low bits.
const SHIFT: u32 = 0x4b40_0000;
/// 2^-75, the exact scale of the subnormal `sqrt` path.
const TWO_POW_MINUS_75: u32 = 0x1a00_0000;

// v_sincosf_common.h, `v_sincosf_data` (hex-float spelling in comments).
const INV_PIO2: u32 = 0x3f22_f983; // 0x1.45f306p-1
const PIO2_1: u32 = 0x3fc9_0fdb; // 0x1.921fb6p+0
const PIO2_2: u32 = 0xb33b_bd2e; // -0x1.777a5cp-25
const PIO2_3: u32 = 0xa6f7_2ced; // -0x1.ee59dap-50
const S0: u32 = 0xbe2a_aaa3; // -0x1.555546p-3
const S1: u32 = 0x3c08_83b0; // 0x1.11076p-7
const S2: u32 = 0xb94c_a75a; // -0x1.994eb4p-13
const C0: u32 = 0x3d2a_aaa5; // 0x1.55554ap-5
const C1: u32 = 0xbab6_060d; // -0x1.6c0c1ap-10
const C2: u32 = 0x37cc_f077; // 0x1.99e0eep-16
// v_trigf_fallback.h, `eval_data` (s0..s2 equal the fast path's).
const PIO4_HI: u32 = 0x3f49_0fdb; // 0x1.921fb6p-1
const PIO4_LO: u32 = 0xb2bb_bd2e; // -0x1.777a5cp-26
const E_C1: u32 = 0x3d2a_aaa5; // 0x1.55554ap-5
const E_C2: u32 = 0xbab6_060d; // -0x1.6c0c1ap-10
const E_C3: u32 = 0x37cc_f077; // 0x1.99e0eep-16

/// `FOUR_OVER_PI[16][4]` of v_trigf_fallback.h, row-major: row i holds a
/// 4-term binary32 expansion of frac((4/pi) * 2^(8 * (i - 3))) in [0.5, 1.5).
pub const FOUR_OVER_PI: [u32; 64] = [
    0x3f80_0001,
    0xb33a_0cf9,
    0xa60d_8ddf,
    0x99ad_603e, //
    0x3f80_00a3,
    0xb14f_9236,
    0xa45d_f56b,
    0x94f6_2a0c, //
    0x3f80_a2fa,
    0xb379_2363,
    0xa6ef_ab58,
    0x98f6_2a0c, //
    0x3fa2_f983,
    0x335c_9c88,
    0x2629_4fe1,
    0x196a_fa3f, //
    0x3f73_06dd,
    0xb2c6_efab,
    0xa630_1ec5,
    0x9982_e0ad, //
    0x3f83_6e4e,
    0x3308_2a54,
    0xa4f6_2a0c,
    0x187a_9a6f, //
    0x3f5c_9c88,
    0x3229_4fe1,
    0x256a_fa3f,
    0x98b2_c890, //
    0x3f1c_882a,
    0x32a7_f09d,
    0x263e_8faa,
    0x99b2_23f2, //
    0x3f08_2a54,
    0xb0f6_2a0c,
    0x247a_9a6f,
    0x96fc_9276, //
    0x3f95_29fc,
    0x329d_5f48,
    0xa5ac_b224,
    0x17db_6296, //
    0x3fa9_fc27,
    0x332f_a3ea,
    0x26d3_7703,
    0x1a5b_14ad, //
    0x3f78_4eb0,
    0xb2b8_2b2d,
    0x266e_06db,
    0x18a5_664f, //
    0x3fa7_57d2,
    0xb1ac_b224,
    0x23db_6296,
    0x974d_8779, //
    0x3f2f_a3ea,
    0x32d3_7703,
    0x265b_14ad,
    0x9958_778e, //
    0x3f23_ea6a,
    0xb288_fc92,
    0xa66b_5336,
    0x98ef_1bf0, //
    0x3f6a_69bc,
    0xb2fc_9276,
    0x262c_c9e2,
    0x18e4_1080, //
];

/// `SINCOS_K_PI_OVER_4[8][2]`: { sin(k pi/4), cos(k pi/4) } for k < 8.
pub const SINCOS_K_PI_OVER_4: [u32; 16] = [
    0,
    ONE,
    0x3f35_04f3,
    0x3f35_04f3,
    ONE,
    0,
    0x3f35_04f3,
    0xbf35_04f3, //
    0,
    0xbf80_0000,
    0xbf35_04f3,
    0xbf35_04f3,
    0xbf80_0000,
    0,
    0xbf35_04f3,
    0x3f35_04f3, //
];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Table {
    FourOverPi,
    SinCosKPiOver4,
}

impl Table {
    pub const fn words(self) -> &'static [u32] {
        match self {
            Self::FourOverPi => &FOUR_OVER_PI,
            Self::SinCosKPiOver4 => &SINCOS_K_PI_OVER_4,
        }
    }
    const fn name(self) -> &'static str {
        match self {
            Self::FourOverPi => "invar_trig_f32_v1_four_over_pi",
            Self::SinCosKPiOver4 => "invar_trig_f32_v1_sincos_k",
        }
    }
    /// Index mask; table lengths are powers of two, so every access is in bounds.
    const fn mask(self) -> u32 {
        self.words().len() as u32 - 1
    }
}

/// The primitive operations of the contract. `W` is a 32-bit word holding
/// either binary32 bits or an unsigned integer. Floating methods round once to
/// nearest even; integer methods are exact and wrap modulo 2^32; shift amounts
/// are taken modulo 32 so every backend has the same total semantics.
pub trait Ops {
    type W: Copy;
    fn word(&mut self, value: u32) -> Self::W;
    fn fadd(&mut self, a: Self::W, b: Self::W) -> Self::W;
    fn fsub(&mut self, a: Self::W, b: Self::W) -> Self::W;
    fn fmul(&mut self, a: Self::W, b: Self::W) -> Self::W;
    /// `a * b + c` with a single rounding.
    fn fma(&mut self, a: Self::W, b: Self::W, c: Self::W) -> Self::W;
    fn fsqrt(&mut self, a: Self::W) -> Self::W;
    /// Unsigned integer to binary32; the contract only converts values below 2^24.
    fn u32_to_f32(&mut self, a: Self::W) -> Self::W;
    fn and(&mut self, a: Self::W, b: Self::W) -> Self::W;
    fn or(&mut self, a: Self::W, b: Self::W) -> Self::W;
    fn xor(&mut self, a: Self::W, b: Self::W) -> Self::W;
    fn iadd(&mut self, a: Self::W, b: Self::W) -> Self::W;
    fn isub(&mut self, a: Self::W, b: Self::W) -> Self::W;
    fn shl(&mut self, a: Self::W, amount: Self::W) -> Self::W;
    /// Logical right shift.
    fn shr(&mut self, a: Self::W, amount: Self::W) -> Self::W;
    /// Arithmetic right shift of the two's-complement word.
    fn sar(&mut self, a: Self::W, amount: Self::W) -> Self::W;
    /// 1 when `a < b` as unsigned integers, else 0.
    fn ult(&mut self, a: Self::W, b: Self::W) -> Self::W;
    /// `then` when `condition != 0`, else `otherwise`.
    fn select(&mut self, condition: Self::W, then: Self::W, otherwise: Self::W) -> Self::W;
    /// `table[index & (len - 1)]`.
    fn table(&mut self, table: Table, index: Self::W) -> Self::W;
    /// Structured control flow: only the taken arm's operations execute.
    fn branch<const N: usize>(
        &mut self,
        condition: Self::W,
        then: impl FnOnce(&mut Self) -> [Self::W; N],
        otherwise: impl FnOnce(&mut Self) -> [Self::W; N],
    ) -> [Self::W; N];
}

/// sin(x) when `cosine == 0`, cos(x) when `cosine == 1`.
pub fn sincos<O: Ops>(o: &mut O, x: O::W, cosine: O::W) -> O::W {
    let abs = o.word(ABS);
    let a = o.and(x, abs);
    let tiny = o.word(TINY);
    let is_tiny = o.ult(a, tiny);
    let [y] = o.branch(
        is_tiny,
        // Deviation: |x| < 2^-12 returns the correctly rounded value without
        // arithmetic. This keeps sin(-0) = -0 (the Arm fast path yields +0)
        // and keeps subnormal inputs away from flushing devices.
        |o| {
            let one = o.word(ONE);
            [o.select(cosine, one, x)]
        },
        |o| {
            let infinity = o.word(INFINITY);
            let finite = o.ult(a, infinity);
            o.branch(
                finite,
                |o| {
                    let limit = o.word(RANGE + 1);
                    let fast = o.ult(a, limit);
                    o.branch(
                        fast,
                        |o| [fast_path(o, x, cosine)],
                        |o| [large_path(o, x, cosine)],
                    )
                },
                // Deviation: NaN and infinite inputs return the canonical NaN.
                |o| [o.word(CANONICAL_NAN)],
            )
        },
    );
    y
}

/// `v_sincosf_inline` for 2^-12 <= |x| <= 2^20.
fn fast_path<O: Ops>(o: &mut O, x: O::W, cosine: O::W) -> O::W {
    let shift = o.word(SHIFT);
    let inv_pio2 = o.word(INV_PIO2);
    // n = rint(x / (pi/2)): q0 = shift + x * inv_pio2 is an integer in
    // [2^23, 2^24), so q0 - shift is exact.
    let q0 = o.fma(x, inv_pio2, shift);
    let q = o.fsub(q0, shift);
    // Arm converts q with vcvtq_s32_f32; |n| < 2^20 makes bits(q0) - bits(shift)
    // the same two's-complement integer.
    let n = o.isub(q0, shift);
    let sign = o.word(SIGN);
    let minus_q = o.xor(q, sign);
    // r = x - q * pi/2 in three fused steps (Arm vfmsq: r - q * c).
    let p1 = o.word(PIO2_1);
    let p2 = o.word(PIO2_2);
    let p3 = o.word(PIO2_3);
    let r = o.fma(minus_q, p1, x);
    let r = o.fma(minus_q, p2, r);
    let r = o.fma(minus_q, p3, r);
    // sin(r) ~= r + r^3 * (s0 + r^2 * (s1 + r^2 * s2)).
    let r2 = o.fmul(r, r);
    let r3 = o.fmul(r, r2);
    let s0 = o.word(S0);
    let s1 = o.word(S1);
    let s2 = o.word(S2);
    let s = o.fma(r2, s2, s1);
    let s = o.fma(r2, s, s0);
    let s = o.fma(r3, s, r);
    // cos(r) ~= 1 + r^2 * (-0.5 + r^2 * c0 + r^4 * (c1 + r^2 * c2)).
    let r4 = o.fmul(r2, r2);
    let c0 = o.word(C0);
    let c1 = o.word(C1);
    let c2 = o.word(C2);
    let minus_half = o.word(MINUS_HALF);
    let one = o.word(ONE);
    let p = o.fma(r2, c2, c1);
    let c = o.fma(r2, c0, minus_half);
    let c = o.fma(r4, p, c);
    let c = o.fma(c, r2, one);
    // cos(x) is sin(x) one quadrant later; the swap and sign are Arm's
    // `n & 1` select and `(n & 2) << 30` (resp. `(n + 1) & 2`) sign flips.
    let m = o.iadd(n, cosine);
    let bit0 = o.word(1);
    let bit1 = o.word(2);
    let thirty = o.word(30);
    let odd = o.and(m, bit0);
    let value = o.select(odd, c, s);
    let quadrant = o.and(m, bit1);
    let flip = o.shl(quadrant, thirty);
    o.xor(value, flip)
}

/// `vrndaq_f32` (round to integral, ties away from zero) on encodings.
fn round_away<O: Ops>(o: &mut O, v: O::W) -> O::W {
    let abs = o.word(ABS);
    let sign_mask = o.word(SIGN);
    let a = o.and(v, abs);
    let sign = o.and(v, sign_mask);
    let twenty_three = o.word(23);
    let exponent = o.shr(a, twenty_three);
    // For exponents 127..=149 the fraction occupies the low 150 - e bits.
    let base = o.word(150);
    let bits = o.isub(base, exponent);
    let one = o.word(1);
    let below = o.isub(bits, one);
    let half = o.shl(one, below);
    let unit = o.shl(one, bits);
    let zero = o.word(0);
    let mask = o.isub(zero, unit);
    let raised = o.iadd(a, half);
    let rounded = o.and(raised, mask);
    let general = o.or(sign, rounded);
    let one_bits = o.word(ONE);
    let signed_one = o.or(sign, one_bits);
    let e126 = o.word(126);
    let e127 = o.word(127);
    let e150 = o.word(150);
    let lt126 = o.ult(exponent, e126);
    let lt127 = o.ult(exponent, e127);
    let lt150 = o.ult(exponent, e150);
    let integral = o.select(lt150, general, v);
    let at_least_half = o.select(lt127, signed_one, integral);
    o.select(lt126, sign, at_least_half)
}

/// Fast two-sum (requires |a| >= |b|): (hi, lo) with hi + lo = a + b exactly.
fn fast_two_sum<O: Ops>(o: &mut O, a: O::W, b: O::W) -> (O::W, O::W) {
    let hi = o.fadd(a, b);
    let t = o.fsub(hi, a);
    let lo = o.fsub(b, t);
    (hi, lo)
}

/// TwoProd: hi = a * b rounded, lo = a * b - hi exactly (Arm `vfmaq(-hi, a, b)`).
fn two_prod<O: Ops>(o: &mut O, a: O::W, b: O::W) -> (O::W, O::W) {
    let hi = o.fmul(a, b);
    let sign = o.word(SIGN);
    let minus_hi = o.xor(hi, sign);
    let lo = o.fma(a, b, minus_hi);
    (hi, lo)
}

/// `sincos_fallback` for finite |x| > 2^20.
fn large_path<O: Ops>(o: &mut O, x: O::W, cosine: O::W) -> O::W {
    // large_range_reduction: move x into [2^32, 2^40) keeping the mantissa and
    // the exponent modulo 8, and pick the 4/pi row from the exponent's high
    // bits. Arm's unsigned shift also carries the sign bit, which adds 256 and
    // changes neither the row (modulo 16) nor the offset (modulo 8).
    let twenty_three = o.word(23);
    let byte = o.word(0xff);
    let shifted = o.shr(x, twenty_three);
    let exponent = o.and(shifted, byte);
    let bias = o.word(127 + 32);
    let x_e_m32 = o.isub(exponent, bias);
    let three = o.word(3);
    let seven = o.word(7);
    let fifteen = o.word(15);
    let row_offset = o.sar(x_e_m32, three);
    let row = o.iadd(row_offset, three);
    let row = o.and(row, fifteen);
    let masked = o.and(x_e_m32, seven);
    let biased = o.iadd(masked, bias);
    let new_exponent = o.shl(biased, twenty_three);
    let sign_mask = o.word(SIGN);
    let mantissa_mask = o.word(0x007f_ffff);
    let sign = o.and(x, sign_mask);
    let mantissa = o.and(x, mantissa_mask);
    let magnitude = o.or(new_exponent, mantissa);
    let x_reduced = o.or(sign, magnitude);
    let two = o.word(2);
    let first = o.shl(row, two);
    let d0 = o.table(Table::FourOverPi, first);
    let one = o.word(1);
    let second = o.iadd(first, one);
    let d1 = o.table(Table::FourOverPi, second);
    let third = o.iadd(first, two);
    let d2 = o.table(Table::FourOverPi, third);
    let fourth = o.iadd(first, three);
    let d3 = o.table(Table::FourOverPi, fourth);
    let (_, ph_lo) = two_prod(o, x_reduced, d0);
    let (pm_hi, pm_lo) = two_prod(o, x_reduced, d1);
    let (pl_hi, pl_lo) = two_prod(o, x_reduced, d2);
    // ph.hi is a multiple of 2^8 and cannot change k mod 8.
    let sum_hi = o.fadd(ph_lo, pm_hi);
    let kd = round_away(o, sum_hi);
    let lowered = o.fsub(ph_lo, kd);
    let y_hi = o.fadd(lowered, pm_hi);
    let (mid_hi, mid_lo) = fast_two_sum(o, pm_lo, pl_hi);
    let y_l = o.fma(x_reduced, d3, pl_lo);
    let (y_hi, y_lo) = fast_two_sum(o, y_hi, mid_hi);
    let tail = o.fadd(mid_lo, y_l);
    let y_lo = o.fadd(y_lo, tail);
    // Arm converts kd with vcvtnq_s32_f32; kd is integral with |kd| < 2^18,
    // so bits(kd + 0x1.8p23) - bits(0x1.8p23) is the same integer.
    let shift = o.word(SHIFT);
    let kd_shifted = o.fadd(kd, shift);
    let k = o.isub(kd_shifted, shift);
    // sincos_eval: r = y * pi/4, sin(r) and cos(r) - 1.
    let pio4_hi = o.word(PIO4_HI);
    let pio4_lo = o.word(PIO4_LO);
    let low = o.fmul(y_lo, pio4_hi);
    let corr = o.fma(y_hi, pio4_lo, low);
    let r = o.fma(y_hi, pio4_hi, corr);
    let r2 = o.fmul(r, r);
    let r3 = o.fmul(r2, r);
    let s0 = o.word(S0);
    let s1 = o.word(S1);
    let s2 = o.word(S2);
    let sin_r = o.fma(r2, s2, s1);
    let sin_r = o.fma(r2, sin_r, s0);
    let sin_r = o.fma(r3, sin_r, r);
    let c1 = o.word(E_C1);
    let c2 = o.word(E_C2);
    let c3 = o.word(E_C3);
    let minus_half = o.word(MINUS_HALF);
    let cosm1 = o.fma(r2, c3, c2);
    let cosm1 = o.fma(r2, cosm1, c1);
    let cosm1 = o.fma(r2, cosm1, minus_half);
    let cosm1 = o.fmul(cosm1, r2);
    // Angle addition with sin(k pi/4) and cos(k pi/4).
    let octant = o.and(k, seven);
    let pair = o.shl(octant, one);
    let sin_k = o.table(Table::SinCosKPiOver4, pair);
    let cos_index = o.iadd(pair, one);
    let cos_k = o.table(Table::SinCosKPiOver4, cos_index);
    let sin_k_cosm1 = o.fmul(sin_k, cosm1);
    let sine = o.fma(cos_k, sin_r, sin_k_cosm1);
    let sine = o.fadd(sine, sin_k);
    let cos_k_cosm1 = o.fmul(cos_k, cosm1);
    let minus_sin_k = o.xor(sin_k, sign_mask);
    let cosine_value = o.fma(minus_sin_k, sin_r, cos_k_cosm1);
    let cosine_value = o.fadd(cosine_value, cos_k);
    o.select(cosine, cosine_value, sine)
}

/// Correctly rounded square root. Subnormal inputs are scaled exactly, so the
/// device `sqrt` only sees normal operands: x = m * 2^-149 = (2m) * 2^-150 and
/// sqrt(x) = sqrt(2m) * 2^-75, where 2m < 2^24 converts exactly and the final
/// power-of-two product is exact and normal.
pub fn sqrt<O: Ops>(o: &mut O, x: O::W) -> O::W {
    let abs = o.word(ABS);
    let a = o.and(x, abs);
    let infinity = o.word(INFINITY);
    let nan = o.ult(infinity, a);
    let thirty_one = o.word(31);
    let negative = o.shr(x, thirty_one);
    let zero = o.word(0);
    let nonzero = o.ult(zero, a);
    let negative_nonzero = o.and(negative, nonzero);
    let invalid = o.or(nan, negative_nonzero);
    let [y] = o.branch(
        invalid,
        |o| [o.word(CANONICAL_NAN)],
        |o| {
            let min_normal = o.word(MIN_NORMAL);
            let small = o.ult(a, min_normal);
            o.branch(
                small,
                |o| {
                    let one = o.word(1);
                    let twice = o.shl(a, one);
                    let t = o.u32_to_f32(twice);
                    let root = o.fsqrt(t);
                    let scale = o.word(TWO_POW_MINUS_75);
                    let scaled = o.fmul(root, scale);
                    // Signed zeros are returned unchanged.
                    [o.select(nonzero, scaled, x)]
                },
                |o| [o.fsqrt(x)],
            )
        },
    );
    y
}

/// Word evaluation under one binary32 arithmetic. The integer, table and
/// control operations of [`Ops`] are exact and shared by every implementer,
/// so an evaluator only states how it rounds the six floating operations.
pub trait Arithmetic {
    fn add(&mut self, a: u32, b: u32) -> u32;
    fn sub(&mut self, a: u32, b: u32) -> u32;
    fn mul(&mut self, a: u32, b: u32) -> u32;
    /// `a * b + c` with one rounding.
    fn fused(&mut self, a: u32, b: u32, c: u32) -> u32;
    fn root(&mut self, a: u32) -> u32;
    fn convert(&mut self, a: u32) -> u32;
}

impl<A: Arithmetic> Ops for A {
    type W = u32;
    fn word(&mut self, value: u32) -> u32 {
        value
    }
    fn fadd(&mut self, a: u32, b: u32) -> u32 {
        self.add(a, b)
    }
    fn fsub(&mut self, a: u32, b: u32) -> u32 {
        self.sub(a, b)
    }
    fn fmul(&mut self, a: u32, b: u32) -> u32 {
        self.mul(a, b)
    }
    fn fma(&mut self, a: u32, b: u32, c: u32) -> u32 {
        self.fused(a, b, c)
    }
    fn fsqrt(&mut self, a: u32) -> u32 {
        self.root(a)
    }
    fn u32_to_f32(&mut self, a: u32) -> u32 {
        self.convert(a)
    }
    fn and(&mut self, a: u32, b: u32) -> u32 {
        a & b
    }
    fn or(&mut self, a: u32, b: u32) -> u32 {
        a | b
    }
    fn xor(&mut self, a: u32, b: u32) -> u32 {
        a ^ b
    }
    fn iadd(&mut self, a: u32, b: u32) -> u32 {
        a.wrapping_add(b)
    }
    fn isub(&mut self, a: u32, b: u32) -> u32 {
        a.wrapping_sub(b)
    }
    fn shl(&mut self, a: u32, amount: u32) -> u32 {
        a << (amount & 31)
    }
    fn shr(&mut self, a: u32, amount: u32) -> u32 {
        a >> (amount & 31)
    }
    fn sar(&mut self, a: u32, amount: u32) -> u32 {
        ((a as i32) >> (amount & 31)) as u32
    }
    fn ult(&mut self, a: u32, b: u32) -> u32 {
        u32::from(a < b)
    }
    fn select(&mut self, condition: u32, then: u32, otherwise: u32) -> u32 {
        if condition != 0 { then } else { otherwise }
    }
    fn table(&mut self, table: Table, index: u32) -> u32 {
        table.words()[(index & table.mask()) as usize]
    }
    fn branch<const N: usize>(
        &mut self,
        condition: u32,
        then: impl FnOnce(&mut Self) -> [u32; N],
        otherwise: impl FnOnce(&mut Self) -> [u32; N],
    ) -> [u32; N] {
        if condition != 0 {
            then(self)
        } else {
            otherwise(self)
        }
    }
}

/// The host's IEEE binary32 arithmetic. Rust never enables flush-to-zero,
/// and `mul_add` is the correctly rounded fused operation on every supported
/// host (a hardware instruction or a correctly rounded `fmaf`).
#[derive(Clone, Copy, Debug, Default)]
pub struct Host;

fn f(w: u32) -> f32 {
    f32::from_bits(w)
}

impl Arithmetic for Host {
    fn add(&mut self, a: u32, b: u32) -> u32 {
        (f(a) + f(b)).to_bits()
    }
    fn sub(&mut self, a: u32, b: u32) -> u32 {
        (f(a) - f(b)).to_bits()
    }
    fn mul(&mut self, a: u32, b: u32) -> u32 {
        (f(a) * f(b)).to_bits()
    }
    fn fused(&mut self, a: u32, b: u32, c: u32) -> u32 {
        f(a).mul_add(f(b), f(c)).to_bits()
    }
    fn root(&mut self, a: u32) -> u32 {
        f(a).sqrt().to_bits()
    }
    fn convert(&mut self, a: u32) -> u32 {
        (a as f32).to_bits()
    }
}

pub fn host_sin(x: u32) -> u32 {
    sincos(&mut Host, x, 0)
}
pub fn host_cos(x: u32) -> u32 {
    sincos(&mut Host, x, 1)
}
pub fn host_sqrt(x: u32) -> u32 {
    sqrt(&mut Host, x)
}

/// Text targets that replay the sequence.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Lang {
    Cuda,
    Msl,
    Llvm,
    /// GNET C++/CUDA/MSL: exact integer software binary32 primitives.
    Gnet,
}

/// Source emitter. Every operation becomes exactly one target operation with
/// the same operands in the same order; values are 32-bit words.
struct Text {
    lang: Lang,
    out: String,
    names: Vec<String>,
    next: usize,
    labels: usize,
    block: String,
}

impl Text {
    fn new(lang: Lang, params: &[&str]) -> Self {
        let names = params
            .iter()
            .map(|p| match lang {
                Lang::Llvm => format!("%{p}"),
                _ => (*p).to_owned(),
            })
            .collect();
        Self {
            lang,
            out: String::new(),
            names,
            next: 0,
            labels: 0,
            block: "entry".to_owned(),
        }
    }
    fn n(&self, w: usize) -> &str {
        &self.names[w]
    }
    fn fresh(&mut self) -> String {
        let id = self.next;
        self.next += 1;
        match self.lang {
            Lang::Llvm => format!("%t{id}"),
            _ => format!("t{id}"),
        }
    }
    fn word_type(&self) -> &'static str {
        match self.lang {
            Lang::Cuda | Lang::Gnet => "unsigned",
            Lang::Msl => "uint",
            Lang::Llvm => "i32",
        }
    }
    /// Bind a C expression to a fresh constant word.
    fn c(&mut self, expr: String) -> usize {
        let name = self.fresh();
        let ty = self.word_type();
        self.out
            .push_str(&format!("  const {ty} {name} = {expr};\n"));
        self.names.push(name);
        self.names.len() - 1
    }
    fn float(&self, w: usize) -> String {
        match self.lang {
            Lang::Cuda => format!("__uint_as_float({})", self.n(w)),
            _ => format!("as_type<float>({})", self.n(w)),
        }
    }
    fn bits(&self, expr: String) -> String {
        match self.lang {
            Lang::Cuda => format!("__float_as_uint({expr})"),
            _ => format!("as_type<uint>({expr})"),
        }
    }
    fn line(&mut self, line: String) {
        self.out.push_str("  ");
        self.out.push_str(&line);
        self.out.push('\n');
    }
    /// LLVM: decode a word to float.
    fn ll_float(&mut self, w: usize) -> String {
        let name = self.fresh();
        let line = format!("{name} = bitcast i32 {} to float", self.n(w));
        self.line(line);
        name
    }
    /// LLVM: encode a float result as a word.
    fn ll_word(&mut self, value: String) -> usize {
        let name = self.fresh();
        self.line(format!("{name} = bitcast float {value} to i32"));
        self.names.push(name);
        self.names.len() - 1
    }
    fn ll_float_op(&mut self, op: &str, args: &[usize]) -> usize {
        let args = args.iter().map(|&w| self.ll_float(w)).collect::<Vec<_>>();
        let result = self.fresh();
        let line = match (op, args.as_slice()) {
            ("fma", [a, b, c]) => {
                format!("{result} = call float @llvm.fma.f32(float {a}, float {b}, float {c})")
            }
            ("sqrt", [a]) => format!("{result} = call float @llvm.sqrt.f32(float {a})"),
            (op, [a, b]) => format!("{result} = {op} float {a}, {b}"),
            _ => unreachable!("floating operation arity is fixed by Ops"),
        };
        self.line(line);
        self.ll_word(result)
    }
    fn ll_int(&mut self, op: &str, a: usize, b: usize) -> usize {
        let name = self.fresh();
        let line = format!("{name} = {op} i32 {}, {}", self.n(a), self.n(b));
        self.line(line);
        self.names.push(name);
        self.names.len() - 1
    }
    fn ll_shift(&mut self, op: &str, a: usize, amount: usize) -> usize {
        let masked = self.fresh();
        let line = format!("{masked} = and i32 {}, 31", self.n(amount));
        self.line(line);
        let name = self.fresh();
        let line = format!("{name} = {op} i32 {}, {masked}", self.n(a));
        self.line(line);
        self.names.push(name);
        self.names.len() - 1
    }
    fn ll_condition(&mut self, w: usize) -> String {
        let name = self.fresh();
        let line = format!("{name} = icmp ne i32 {}, 0", self.n(w));
        self.line(line);
        name
    }
    fn c_float_op(&mut self, cuda: &str, msl: &str, args: &[usize]) -> usize {
        if self.lang == Lang::Gnet {
            let helper = match cuda {
                "__fadd_rn" => "gnet_f32_add",
                "__fsub_rn" => "gnet_f32_sub",
                "__fmul_rn" => "gnet_f32_mul",
                "__fmaf_rn" => "gnet_f32_fma",
                "__fsqrt_rn" => "gnet_f32_sqrt",
                _ => unreachable!("fixed floating primitive"),
            };
            return self.c(format!(
                "{helper}({})",
                args.iter()
                    .map(|&w| self.n(w))
                    .collect::<Vec<_>>()
                    .join(", ")
            ));
        }
        let args = args.iter().map(|&w| self.float(w)).collect::<Vec<_>>();
        let expr = match (self.lang, args.as_slice()) {
            (Lang::Cuda, args) => format!("{cuda}({})", args.join(", ")),
            (_, [a, b]) => format!("({a} {msl} {b})"),
            (_, args) => format!("{msl}({})", args.join(", ")),
        };
        let bits = self.bits(expr);
        self.c(bits)
    }
    fn c_int(&mut self, op: &str, a: usize, b: usize) -> usize {
        let expr = format!("({} {op} {})", self.n(a), self.n(b));
        self.c(expr)
    }
}

impl Ops for Text {
    type W = usize;
    fn word(&mut self, value: u32) -> usize {
        self.names.push(match self.lang {
            Lang::Llvm => (value as i32).to_string(),
            _ => format!("{value}u"),
        });
        self.names.len() - 1
    }
    fn fadd(&mut self, a: usize, b: usize) -> usize {
        match self.lang {
            Lang::Llvm => self.ll_float_op("fadd", &[a, b]),
            _ => self.c_float_op("__fadd_rn", "+", &[a, b]),
        }
    }
    fn fsub(&mut self, a: usize, b: usize) -> usize {
        match self.lang {
            Lang::Llvm => self.ll_float_op("fsub", &[a, b]),
            _ => self.c_float_op("__fsub_rn", "-", &[a, b]),
        }
    }
    fn fmul(&mut self, a: usize, b: usize) -> usize {
        match self.lang {
            Lang::Llvm => self.ll_float_op("fmul", &[a, b]),
            _ => self.c_float_op("__fmul_rn", "*", &[a, b]),
        }
    }
    fn fma(&mut self, a: usize, b: usize, c: usize) -> usize {
        match self.lang {
            Lang::Llvm => self.ll_float_op("fma", &[a, b, c]),
            _ => self.c_float_op("__fmaf_rn", "fma", &[a, b, c]),
        }
    }
    fn fsqrt(&mut self, a: usize) -> usize {
        match self.lang {
            Lang::Llvm => self.ll_float_op("sqrt", &[a]),
            _ => self.c_float_op("__fsqrt_rn", "sqrt", &[a]),
        }
    }
    fn u32_to_f32(&mut self, a: usize) -> usize {
        match self.lang {
            Lang::Llvm => {
                let converted = self.fresh();
                let line = format!("{converted} = uitofp i32 {} to float", self.n(a));
                self.line(line);
                self.ll_word(converted)
            }
            Lang::Cuda => {
                let expr = format!("__float_as_uint(__uint2float_rn({}))", self.n(a));
                self.c(expr)
            }
            Lang::Msl => {
                let expr = format!("as_type<uint>(float({}))", self.n(a));
                self.c(expr)
            }
            Lang::Gnet => self.c(format!("gnet_f32_from_u32({})", self.n(a))),
        }
    }
    fn and(&mut self, a: usize, b: usize) -> usize {
        match self.lang {
            Lang::Llvm => self.ll_int("and", a, b),
            _ => self.c_int("&", a, b),
        }
    }
    fn or(&mut self, a: usize, b: usize) -> usize {
        match self.lang {
            Lang::Llvm => self.ll_int("or", a, b),
            _ => self.c_int("|", a, b),
        }
    }
    fn xor(&mut self, a: usize, b: usize) -> usize {
        match self.lang {
            Lang::Llvm => self.ll_int("xor", a, b),
            _ => self.c_int("^", a, b),
        }
    }
    fn iadd(&mut self, a: usize, b: usize) -> usize {
        match self.lang {
            Lang::Llvm => self.ll_int("add", a, b),
            _ => self.c_int("+", a, b),
        }
    }
    fn isub(&mut self, a: usize, b: usize) -> usize {
        match self.lang {
            Lang::Llvm => self.ll_int("sub", a, b),
            _ => self.c_int("-", a, b),
        }
    }
    fn shl(&mut self, a: usize, amount: usize) -> usize {
        match self.lang {
            Lang::Llvm => self.ll_shift("shl", a, amount),
            _ => {
                let expr = format!("({} << ({} & 31u))", self.n(a), self.n(amount));
                self.c(expr)
            }
        }
    }
    fn shr(&mut self, a: usize, amount: usize) -> usize {
        match self.lang {
            Lang::Llvm => self.ll_shift("lshr", a, amount),
            _ => {
                let expr = format!("({} >> ({} & 31u))", self.n(a), self.n(amount));
                self.c(expr)
            }
        }
    }
    fn sar(&mut self, a: usize, amount: usize) -> usize {
        match self.lang {
            Lang::Llvm => self.ll_shift("ashr", a, amount),
            Lang::Cuda => {
                let expr = format!(
                    "(unsigned)((int){} >> ({} & 31u))",
                    self.n(a),
                    self.n(amount)
                );
                self.c(expr)
            }
            Lang::Msl => {
                let expr = format!(
                    "as_type<uint>(as_type<int>({}) >> ({} & 31u))",
                    self.n(a),
                    self.n(amount)
                );
                self.c(expr)
            }
            Lang::Gnet => {
                // Spell sign extension in unsigned arithmetic, independent of
                // implementation-defined signed conversions and right shifts.
                let a = self.n(a);
                let b = self.n(amount);
                self.c(format!(
                    "(({a} >> ({b} & 31u)) | ((0u - ({a} >> 31u)) & ~(0xffffffffu >> ({b} & 31u))))"
                ))
            }
        }
    }
    fn ult(&mut self, a: usize, b: usize) -> usize {
        match self.lang {
            Lang::Llvm => {
                let flag = self.fresh();
                let line = format!("{flag} = icmp ult i32 {}, {}", self.n(a), self.n(b));
                self.line(line);
                let name = self.fresh();
                self.line(format!("{name} = zext i1 {flag} to i32"));
                self.names.push(name);
                self.names.len() - 1
            }
            _ => {
                let expr = format!("({} < {} ? 1u : 0u)", self.n(a), self.n(b));
                self.c(expr)
            }
        }
    }
    fn select(&mut self, condition: usize, then: usize, otherwise: usize) -> usize {
        match self.lang {
            Lang::Llvm => {
                let flag = self.ll_condition(condition);
                let name = self.fresh();
                let line = format!(
                    "{name} = select i1 {flag}, i32 {}, i32 {}",
                    self.n(then),
                    self.n(otherwise)
                );
                self.line(line);
                self.names.push(name);
                self.names.len() - 1
            }
            _ => {
                let expr = format!(
                    "({} != 0u ? {} : {})",
                    self.n(condition),
                    self.n(then),
                    self.n(otherwise)
                );
                self.c(expr)
            }
        }
    }
    fn table(&mut self, table: Table, index: usize) -> usize {
        let len = table.words().len();
        let mask = table.mask();
        match self.lang {
            Lang::Llvm => {
                let masked = self.fresh();
                let line = format!("{masked} = and i32 {}, {mask}", self.n(index));
                self.line(line);
                let wide = self.fresh();
                self.line(format!("{wide} = zext i32 {masked} to i64"));
                let pointer = self.fresh();
                self.line(format!(
                    "{pointer} = getelementptr [{len} x i32], ptr @{}, i64 0, i64 {wide}",
                    table.name()
                ));
                let name = self.fresh();
                self.line(format!("{name} = load i32, ptr {pointer}"));
                self.names.push(name);
                self.names.len() - 1
            }
            _ => {
                let expr = format!("{}[{} & {mask}u]", table.name(), self.n(index));
                self.c(expr)
            }
        }
    }
    fn branch<const N: usize>(
        &mut self,
        condition: usize,
        then: impl FnOnce(&mut Self) -> [usize; N],
        otherwise: impl FnOnce(&mut Self) -> [usize; N],
    ) -> [usize; N] {
        if self.lang == Lang::Llvm {
            let flag = self.ll_condition(condition);
            let id = self.labels;
            self.labels += 1;
            let (then_label, else_label, merge) = (
                format!("then{id}"),
                format!("else{id}"),
                format!("merge{id}"),
            );
            self.line(format!(
                "br i1 {flag}, label %{then_label}, label %{else_label}"
            ));
            self.out.push_str(&format!("{then_label}:\n"));
            self.block = then_label;
            let taken = then(self);
            let taken_end = self.block.clone();
            self.line(format!("br label %{merge}"));
            self.out.push_str(&format!("{else_label}:\n"));
            self.block = else_label;
            let other = otherwise(self);
            let other_end = self.block.clone();
            self.line(format!("br label %{merge}"));
            self.out.push_str(&format!("{merge}:\n"));
            self.block = merge;
            return std::array::from_fn(|i| {
                let name = self.fresh();
                let line = format!(
                    "{name} = phi i32 [ {}, %{taken_end} ], [ {}, %{other_end} ]",
                    self.n(taken[i]),
                    self.n(other[i])
                );
                self.line(line);
                self.names.push(name);
                self.names.len() - 1
            });
        }
        let ty = self.word_type();
        let outputs: [String; N] = std::array::from_fn(|_| self.fresh());
        self.line(format!("{ty} {};", outputs.join(", ")));
        let condition = self.n(condition).to_owned();
        self.line(format!("if ({condition} != 0u) {{"));
        let taken = then(self);
        for (output, &value) in outputs.iter().zip(&taken) {
            let line = format!("{output} = {};", self.n(value));
            self.line(line);
        }
        self.line("} else {".to_owned());
        let other = otherwise(self);
        for (output, &value) in outputs.iter().zip(&other) {
            let line = format!("{output} = {};", self.n(value));
            self.line(line);
        }
        self.line("}".to_owned());
        outputs.map(|output| {
            self.names.push(output);
            self.names.len() - 1
        })
    }
}

fn table_text(lang: Lang, table: Table) -> String {
    let words = table.words();
    match lang {
        Lang::Llvm => format!(
            "@{} = private unnamed_addr constant [{} x i32] [{}]\n",
            table.name(),
            words.len(),
            words
                .iter()
                .map(|w| format!("i32 {}", *w as i32))
                .collect::<Vec<_>>()
                .join(", ")
        ),
        Lang::Cuda | Lang::Msl | Lang::Gnet => format!(
            "{} {}[{}] = {{{}}};\n",
            if lang == Lang::Gnet {
                "GNET_TRIG_CONSTANT unsigned"
            } else if lang == Lang::Cuda {
                "__device__ static const unsigned"
            } else {
                "constant uint"
            },
            table.name(),
            words.len(),
            words
                .iter()
                .map(|w| format!("{w:#x}u"))
                .collect::<Vec<_>>()
                .join(",")
        ),
    }
}

fn function(
    lang: Lang,
    name: &str,
    params: &[&str],
    body: impl FnOnce(&mut Text) -> usize,
) -> String {
    let mut text = Text::new(lang, params);
    let result = body(&mut text);
    let result = text.n(result).to_owned();
    match lang {
        Lang::Llvm => format!(
            "define private i32 @{name}({}) \"denormal-fp-math\"=\"ieee,ieee\" \"denormal-fp-math-f32\"=\"ieee,ieee\" {{\nentry:\n{}  ret i32 {result}\n}}\n",
            params
                .iter()
                .map(|p| format!("i32 %{p}"))
                .collect::<Vec<_>>()
                .join(", "),
            text.out
        ),
        Lang::Cuda | Lang::Msl | Lang::Gnet => {
            let (qualifier, ty) = if lang == Lang::Gnet {
                ("GNET_FN unsigned", "unsigned")
            } else if lang == Lang::Cuda {
                ("__device__ static inline unsigned", "unsigned")
            } else {
                ("static inline uint", "uint")
            };
            format!(
                "{qualifier} {name}({}) {{\n{}  return {result};\n}}\n",
                params
                    .iter()
                    .map(|p| format!("{ty} {p}"))
                    .collect::<Vec<_>>()
                    .join(", "),
                text.out
            )
        }
    }
}

/// Helper definitions for a target: `invar_trig_f32_v1(x, cosine)` when
/// `trig`, and `invar_sqrt_f32_v1(x)` when `root`. Words in, word out. LLVM
/// text also needs `declare float @llvm.fma.f32(float, float, float)` and,
/// with `root`, `declare float @llvm.sqrt.f32(float)` from the caller.
pub fn helpers(lang: Lang, trig: bool, root: bool) -> String {
    let mut out = format!("// {CONTRACT}: generated from invar_device::trig_f32\n");
    if lang == Lang::Llvm {
        out = format!("; {CONTRACT}: generated from invar_device::trig_f32\n");
    }
    if lang == Lang::Gnet {
        out.push_str("static_assert(sizeof(unsigned)==4,\"GNET trig words require32 bits\");\n");
    }
    if lang == Lang::Gnet && root {
        // sqrt() supplies positive normal operands, +0 or +infinity. For a 24-bit
        // significand s and even unbiased exponent e, sqrt(s << 23) has a
        // 24-bit integer floor q. Round up iff remainder > q: the midpoint
        // square is q*q + q + 1/4, so an integer radicand can never tie.
        // All squares fit in 48 bits. No host/device floating arithmetic.
        out.push_str(
            r#"
GNET_FN unsigned gnet_f32_sqrt(unsigned a) {
  if (a == 0u || a == 2139095040u) return a;
  int e = (int)(a >> 23) - 127;
  gnet_u s = (a & 8388607u) | 8388608u;
  if (e % 2 != 0) { s <<= 1; --e; }
  const gnet_u n = s << 23;
  gnet_u q = 0;
  for (int bit = 23; bit >= 0; --bit) {
    const gnet_u t = q | (1ULL << bit);
    if (t*t <= n) q = t;
  }
  if (n-q*q > q) ++q;
  return (unsigned)gnet_f32_pack(q, e/2, 0);
}
"#,
        );
    }
    if trig {
        out.push_str(&table_text(lang, Table::FourOverPi));
        out.push_str(&table_text(lang, Table::SinCosKPiOver4));
        out.push_str(&function(lang, SIN_HELPER, &["x", "cosine"], |t| {
            sincos(t, 0, 1)
        }));
    }
    if root {
        out.push_str(&function(lang, SQRT_HELPER, &["x"], |t| sqrt(t, 0)));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hex(v: f64) -> u32 {
        (v as f32).to_bits()
    }

    #[test]
    fn constants_match_their_hex_float_spellings() {
        // Rust has no hex float literals; these decimal values are exact.
        assert_eq!(
            f32::from_bits(INV_PIO2) as f64,
            0.636_619_746_685_028_076_171_875
        );
        assert_eq!(hex(1.5 * 8_388_608.0), SHIFT);
        assert_eq!(hex(1_048_576.0), RANGE);
        assert_eq!(f32::from_bits(TINY), 2f32.powi(-12));
        assert_eq!(f32::from_bits(TWO_POW_MINUS_75), 2f32.powi(-75));
        assert_eq!(f32::from_bits(PIO2_1), std::f32::consts::FRAC_PI_2);
        assert_eq!(f32::from_bits(PIO4_HI), std::f32::consts::FRAC_PI_4);
        // pio2_2 and pio4_lo are the same tail at different scales.
        assert_eq!(f32::from_bits(PIO2_2) / 2.0, f32::from_bits(PIO4_LO));
        // pio2_1 + pio2_2 + pio2_3 approximates pi/2 to about 2^-72.
        let sum = f32::from_bits(PIO2_1) as f64 + f32::from_bits(PIO2_2) as f64;
        assert!((sum - std::f64::consts::FRAC_PI_2).abs() < 2f64.powi(-47));
        assert_eq!(f32::from_bits(0x3f35_04f3), std::f32::consts::FRAC_1_SQRT_2);
    }

    #[test]
    fn round_away_matches_the_host_for_every_exponent_and_tie() {
        let mut cases = vec![0, SIGN, 0x3f00_0000, 0xbf00_0000, 0x3fc0_0000, 0x4020_0000];
        for e in 100..=160u32 {
            for m in [0, 1, 0x40_0000, 0x40_0001, 0x3f_ffff, 0x7f_ffff] {
                cases.push((e << 23) | m);
                cases.push(SIGN | (e << 23) | m);
            }
        }
        for v in cases {
            assert_eq!(
                round_away(&mut Host, v),
                f(v).round().to_bits(),
                "{v:#010x}"
            );
        }
    }

    #[test]
    fn special_values_follow_the_contract() {
        for x in [
            0x7f80_0000,
            0xff80_0000,
            0x7fc0_0000,
            0xffc0_1234,
            0x7f80_0001,
        ] {
            assert_eq!(host_sin(x), CANONICAL_NAN);
            assert_eq!(host_cos(x), CANONICAL_NAN);
        }
        for x in [
            0,
            SIGN,
            1,
            0x8000_0001,
            0x007f_ffff,
            0x3980_0000 - 1,
            0xb980_0000 - 1,
        ] {
            assert_eq!(host_sin(x), x);
            assert_eq!(host_cos(x), ONE);
        }
        assert_eq!(host_sqrt(SIGN), SIGN);
        assert_eq!(host_sqrt(0), 0);
        assert_eq!(host_sqrt(INFINITY), INFINITY);
        for x in [
            0xbf80_0000,
            0xff80_0000,
            0x8000_0001,
            0x7fc0_0001,
            0xffff_ffff,
        ] {
            assert_eq!(host_sqrt(x), CANONICAL_NAN);
        }
    }

    #[test]
    fn sqrt_is_correctly_rounded_on_subnormals_and_boundaries() {
        // The host sqrt is IEEE; the scaled path must agree with it bit for bit.
        for x in (1..MIN_NORMAL)
            .step_by(4099)
            .chain([1, 2, 3, MIN_NORMAL - 1, MIN_NORMAL])
        {
            assert_eq!(host_sqrt(x), f(x).sqrt().to_bits(), "{x:#010x}");
        }
        for x in (MIN_NORMAL..INFINITY).step_by(65_537) {
            assert_eq!(host_sqrt(x), f(x).sqrt().to_bits(), "{x:#010x}");
        }
    }

    #[test]
    fn host_is_close_to_libm_on_a_sample() {
        // A sanity bound only; the contract's exhaustive measurement is in
        // docs/trig-f32-v1.md.
        for x in (0..0x7f80_0000u32).step_by(1_000_003) {
            for (value, reference) in [
                (host_sin(x), (f(x) as f64).sin()),
                (host_cos(x), (f(x) as f64).cos()),
            ] {
                let ulp = 2f64.powi((reference.abs().log2().floor() as i32).max(-126) - 23);
                assert!(
                    ((f(value) as f64) - reference).abs() <= 2.0 * ulp,
                    "{x:#010x}"
                );
            }
        }
    }

    #[test]
    fn emitters_spell_every_floating_operation_once() {
        // Count the primitive floating operations the sequence performs.
        #[derive(Default)]
        struct Count(usize, usize, usize);
        let count = |lang: Lang, text: &str| -> Count {
            match lang {
                Lang::Cuda => Count(
                    text.matches("__fmaf_rn(").count(),
                    text.matches("__fmul_rn(").count()
                        + text.matches("__fadd_rn(").count()
                        + text.matches("__fsub_rn(").count(),
                    text.matches("__fsqrt_rn(").count(),
                ),
                Lang::Msl => Count(
                    text.matches(" fma(").count() + text.matches("(fma(").count(),
                    text.matches(") * as_type").count()
                        + text.matches(") + as_type").count()
                        + text.matches(") - as_type").count(),
                    text.matches("(sqrt(").count(),
                ),
                Lang::Llvm => Count(
                    text.matches("@llvm.fma.f32(float %").count(),
                    text.matches(" = fmul float").count()
                        + text.matches(" = fadd float").count()
                        + text.matches(" = fsub float").count(),
                    text.matches("@llvm.sqrt.f32(float %").count(),
                ),
                Lang::Gnet => Count(
                    text.matches("= gnet_f32_fma(").count(),
                    text.matches("= gnet_f32_mul(").count()
                        + text.matches("= gnet_f32_add(").count()
                        + text.matches("= gnet_f32_sub(").count(),
                    text.matches("= gnet_f32_sqrt(").count(),
                ),
            }
        };
        let mut reference = Count::default();
        struct Tally<'a>(&'a mut Count);
        impl Ops for Tally<'_> {
            type W = ();
            fn word(&mut self, _: u32) {}
            fn fadd(&mut self, _: (), _: ()) {
                self.0.1 += 1;
            }
            fn fsub(&mut self, _: (), _: ()) {
                self.0.1 += 1;
            }
            fn fmul(&mut self, _: (), _: ()) {
                self.0.1 += 1;
            }
            fn fma(&mut self, _: (), _: (), _: ()) {
                self.0.0 += 1;
            }
            fn fsqrt(&mut self, _: ()) {
                self.0.2 += 1;
            }
            fn u32_to_f32(&mut self, _: ()) {}
            fn and(&mut self, _: (), _: ()) {}
            fn or(&mut self, _: (), _: ()) {}
            fn xor(&mut self, _: (), _: ()) {}
            fn iadd(&mut self, _: (), _: ()) {}
            fn isub(&mut self, _: (), _: ()) {}
            fn shl(&mut self, _: (), _: ()) {}
            fn shr(&mut self, _: (), _: ()) {}
            fn sar(&mut self, _: (), _: ()) {}
            fn ult(&mut self, _: (), _: ()) {}
            fn select(&mut self, _: (), _: (), _: ()) {}
            fn table(&mut self, _: Table, _: ()) {}
            fn branch<const N: usize>(
                &mut self,
                _: (),
                then: impl FnOnce(&mut Self) -> [(); N],
                otherwise: impl FnOnce(&mut Self) -> [(); N],
            ) -> [(); N] {
                then(self);
                otherwise(self)
            }
        }
        sincos(&mut Tally(&mut reference), (), ());
        sqrt(&mut Tally(&mut reference), ());
        assert_eq!((reference.0, reference.1, reference.2), (25, 28, 2));
        for lang in [Lang::Cuda, Lang::Msl, Lang::Llvm, Lang::Gnet] {
            let text = helpers(lang, true, true);
            let seen = count(lang, &text);
            assert_eq!(
                (seen.0, seen.1, seen.2),
                (reference.0, reference.1, reference.2),
                "{lang:?}\n{text}"
            );
        }
    }
}
