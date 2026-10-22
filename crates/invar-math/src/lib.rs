//! The reviewed FFI boundary to the vendored CORE-MATH library.
//!
//! CORE-MATH (MIT, https://gitlab.inria.fr/core-math/core-math, commit
//! `dae5dd8c0d5bd2691b45de28c9edbe12f067c3d3`) provides correctly rounded
//! binary32 and binary64 functions. `build.rs` compiles the unmodified
//! sources with `-ffp-contract=off` into one static archive. The safe entry
//! points below take and return raw encodings so callers never depend on host
//! floating-point state. The same C text is the source of the CUDA and MSL
//! emitters ([`source`]), so every engine runs one algorithm.
#![deny(unsafe_op_in_unsafe_fn)]

/// The correctly rounded functions CORE-MATH supplies to `invar-math-cr-v1`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Function {
    Cbrt,
    Hypot,
    Sin,
    Cos,
    Tan,
    Asin,
    Acos,
    Atan,
    Atan2,
    Sinh,
    Cosh,
    Tanh,
    Exp,
    Exp2,
    Expm1,
    Log,
    Log2,
    Log10,
    Log1p,
    Pow,
}

impl Function {
    pub const ALL: [Self; 20] = [
        Self::Cbrt,
        Self::Hypot,
        Self::Sin,
        Self::Cos,
        Self::Tan,
        Self::Asin,
        Self::Acos,
        Self::Atan,
        Self::Atan2,
        Self::Sinh,
        Self::Cosh,
        Self::Tanh,
        Self::Exp,
        Self::Exp2,
        Self::Expm1,
        Self::Log,
        Self::Log2,
        Self::Log10,
        Self::Log1p,
        Self::Pow,
    ];

    /// The CORE-MATH directory and binary64 base name, e.g. `sin`.
    pub const fn name(self) -> &'static str {
        match self {
            Self::Cbrt => "cbrt",
            Self::Hypot => "hypot",
            Self::Sin => "sin",
            Self::Cos => "cos",
            Self::Tan => "tan",
            Self::Asin => "asin",
            Self::Acos => "acos",
            Self::Atan => "atan",
            Self::Atan2 => "atan2",
            Self::Sinh => "sinh",
            Self::Cosh => "cosh",
            Self::Tanh => "tanh",
            Self::Exp => "exp",
            Self::Exp2 => "exp2",
            Self::Expm1 => "expm1",
            Self::Log => "log",
            Self::Log2 => "log2",
            Self::Log10 => "log10",
            Self::Log1p => "log1p",
            Self::Pow => "pow",
        }
    }

    pub const fn binary(self) -> bool {
        matches!(self, Self::Hypot | Self::Atan2 | Self::Pow)
    }
}

mod ffi {
    unsafe extern "C" {
        pub fn cr_cbrtf(x: f32) -> f32;
        pub fn cr_hypotf(x: f32, y: f32) -> f32;
        pub fn cr_sinf(x: f32) -> f32;
        pub fn cr_cosf(x: f32) -> f32;
        pub fn cr_tanf(x: f32) -> f32;
        pub fn cr_asinf(x: f32) -> f32;
        pub fn cr_acosf(x: f32) -> f32;
        pub fn cr_atanf(x: f32) -> f32;
        pub fn cr_atan2f(y: f32, x: f32) -> f32;
        pub fn cr_sinhf(x: f32) -> f32;
        pub fn cr_coshf(x: f32) -> f32;
        pub fn cr_tanhf(x: f32) -> f32;
        pub fn cr_expf(x: f32) -> f32;
        pub fn cr_exp2f(x: f32) -> f32;
        pub fn cr_expm1f(x: f32) -> f32;
        pub fn cr_logf(x: f32) -> f32;
        pub fn cr_log2f(x: f32) -> f32;
        pub fn cr_log10f(x: f32) -> f32;
        pub fn cr_log1pf(x: f32) -> f32;
        pub fn cr_powf(x: f32, y: f32) -> f32;
        pub fn cr_cbrt(x: f64) -> f64;
        pub fn cr_hypot(x: f64, y: f64) -> f64;
        pub fn cr_sin(x: f64) -> f64;
        pub fn cr_cos(x: f64) -> f64;
        pub fn cr_tan(x: f64) -> f64;
        pub fn cr_asin(x: f64) -> f64;
        pub fn cr_acos(x: f64) -> f64;
        pub fn cr_atan(x: f64) -> f64;
        pub fn cr_atan2(y: f64, x: f64) -> f64;
        pub fn cr_sinh(x: f64) -> f64;
        pub fn cr_cosh(x: f64) -> f64;
        pub fn cr_tanh(x: f64) -> f64;
        pub fn cr_exp(x: f64) -> f64;
        pub fn cr_exp2(x: f64) -> f64;
        pub fn cr_expm1(x: f64) -> f64;
        pub fn cr_log(x: f64) -> f64;
        pub fn cr_log2(x: f64) -> f64;
        pub fn cr_log10(x: f64) -> f64;
        pub fn cr_log1p(x: f64) -> f64;
        pub fn cr_pow(x: f64, y: f64) -> f64;
        pub fn invar_math_hold_nearest() -> core::ffi::c_int;
        pub fn invar_math_restore_rounding(mode: core::ffi::c_int);
        pub fn invar_math_fe_upward() -> core::ffi::c_int;
        pub fn invar_math_rounding() -> core::ffi::c_int;
    }
}

/// Test support: switch the calling thread to the upward rounding mode (or back
/// to round-to-nearest) and read the current mode.
#[doc(hidden)]
pub fn set_upward_rounding(upward: bool) {
    // SAFETY: fenv calls on the calling thread's own floating-point state.
    unsafe {
        if upward {
            ffi::invar_math_restore_rounding(ffi::invar_math_fe_upward());
        } else {
            ffi::invar_math_hold_nearest();
        }
    }
}

#[doc(hidden)]
pub fn upward_rounding() -> bool {
    // SAFETY: as above.
    unsafe { ffi::invar_math_rounding() == ffi::invar_math_fe_upward() }
}

/// Evaluate `f` in binary32 on encodings. The second operand is ignored by
/// unary functions; `atan2` takes `(y, x)` in that order like C.
pub fn f32_bits(f: Function, x: u32, y: u32) -> u32 {
    let (x, y) = (f32::from_bits(x), f32::from_bits(y));
    // SAFETY: every CORE-MATH entry point is a pure function of its scalar
    // arguments with no pointer parameters or preconditions; the only global
    // state it reads, the rounding mode, is held at round-to-nearest here.
    let result = unsafe {
        let mode = ffi::invar_math_hold_nearest();
        let result = match f {
            Function::Cbrt => ffi::cr_cbrtf(x),
            Function::Hypot => ffi::cr_hypotf(x, y),
            Function::Sin => ffi::cr_sinf(x),
            Function::Cos => ffi::cr_cosf(x),
            Function::Tan => ffi::cr_tanf(x),
            Function::Asin => ffi::cr_asinf(x),
            Function::Acos => ffi::cr_acosf(x),
            Function::Atan => ffi::cr_atanf(x),
            Function::Atan2 => ffi::cr_atan2f(x, y),
            Function::Sinh => ffi::cr_sinhf(x),
            Function::Cosh => ffi::cr_coshf(x),
            Function::Tanh => ffi::cr_tanhf(x),
            Function::Exp => ffi::cr_expf(x),
            Function::Exp2 => ffi::cr_exp2f(x),
            Function::Expm1 => ffi::cr_expm1f(x),
            Function::Log => ffi::cr_logf(x),
            Function::Log2 => ffi::cr_log2f(x),
            Function::Log10 => ffi::cr_log10f(x),
            Function::Log1p => ffi::cr_log1pf(x),
            Function::Pow => ffi::cr_powf(x, y),
        };
        ffi::invar_math_restore_rounding(mode);
        result
    };
    result.to_bits()
}

/// Evaluate `f` in binary64 on encodings; see [`f32_bits`].
pub fn f64_bits(f: Function, x: u64, y: u64) -> u64 {
    let (x, y) = (f64::from_bits(x), f64::from_bits(y));
    // SAFETY: as in `f32_bits`, pure scalar functions without preconditions.
    let result = unsafe {
        let mode = ffi::invar_math_hold_nearest();
        let result = match f {
            Function::Cbrt => ffi::cr_cbrt(x),
            Function::Hypot => ffi::cr_hypot(x, y),
            Function::Sin => ffi::cr_sin(x),
            Function::Cos => ffi::cr_cos(x),
            Function::Tan => ffi::cr_tan(x),
            Function::Asin => ffi::cr_asin(x),
            Function::Acos => ffi::cr_acos(x),
            Function::Atan => ffi::cr_atan(x),
            Function::Atan2 => ffi::cr_atan2(x, y),
            Function::Sinh => ffi::cr_sinh(x),
            Function::Cosh => ffi::cr_cosh(x),
            Function::Tanh => ffi::cr_tanh(x),
            Function::Exp => ffi::cr_exp(x),
            Function::Exp2 => ffi::cr_exp2(x),
            Function::Expm1 => ffi::cr_expm1(x),
            Function::Log => ffi::cr_log(x),
            Function::Log2 => ffi::cr_log2(x),
            Function::Log10 => ffi::cr_log10(x),
            Function::Log1p => ffi::cr_log1p(x),
            Function::Pow => ffi::cr_pow(x, y),
        };
        ffi::invar_math_restore_rounding(mode);
        result
    };
    result.to_bits()
}

/// The static archive `build.rs` compiled, for linking native executables.
pub const ARCHIVE: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/libinvar_coremath.a"));

/// The vendored C translation unit for `f` in binary32 (`binary64` false) or
/// binary64, and the local headers it includes as `(name, text)`.
pub fn source(
    f: Function,
    binary64: bool,
) -> (&'static str, &'static [(&'static str, &'static str)]) {
    sources::get(f, binary64)
}

mod cxx;
mod device;
mod msl;
mod sources;
pub use device::{
    CUDA_ANONYMOUS_STRUCT_NVRTC, CUDA_MIN_NVRTC, CUDA_OPTIONS, cuda_min_nvrtc, cuda_source,
    cuda_symbol,
};
pub use msl::{MSL_PRELUDE, msl_source, msl_symbol};

#[cfg(test)]
mod tests;
