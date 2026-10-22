//! invar-math-cr-v1 on CUDA: one elementwise kernel per library function and
//! type, over raw encodings. CORE-MATH functions compile from the vendored C
//! (`invar_math::cuda_source`); the exact functions use CUDA operations that
//! are IEEE exact under `invar_math::CUDA_OPTIONS`. Results that are NaN leave
//! as the canonical quiet NaN, as on the CPU engines.

use super::{MathFn, core_math};

/// The kernel every generated map exports.
pub const CUDA_KERNEL: &str = "invar_math_map";

/// The C expression for an exact function over operands `a` and `b`.
fn exact(f: MathFn, binary64: bool) -> String {
    let s = if binary64 { "" } else { "f" };
    let one = if binary64 { "1.0" } else { "1.0f" };
    match f {
        MathFn::Sqrt => format!("sqrt{s}(a)"),
        MathFn::Floor => format!("floor{s}(a)"),
        MathFn::Ceil => format!("ceil{s}(a)"),
        MathFn::Trunc => format!("trunc{s}(a)"),
        MathFn::Round => format!("round{s}(a)"),
        MathFn::Abs => format!("fabs{s}(a)"),
        MathFn::Copysign => format!("copysign{s}(a, b != b ? {one} : b)"),
        MathFn::Min => {
            "a != a ? b : b != b ? a : a == b ? (signbit(a) ? a : b) : a < b ? a : b".into()
        }
        MathFn::Max => {
            "a != a ? b : b != b ? a : a == b ? (signbit(a) ? b : a) : a > b ? a : b".into()
        }
        _ => unreachable!("{} is a CORE-MATH function", f.name()),
    }
}

/// A complete NVRTC translation unit computing `f` elementwise; compile it
/// with `invar_math::CUDA_OPTIONS`.
pub fn cuda_map_source(f: MathFn, binary64: bool) -> String {
    let (mut source, call) = match core_math(f) {
        Some(c) => {
            let args = if f.arity() == 2 { "a, b" } else { "a" };
            (
                invar_math::cuda_source(&[(c, binary64)]),
                format!("{}({args})", invar_math::cuda_symbol(c, binary64)),
            )
        }
        None => (
            String::from("// invar-math-cr-v1 exact\n"),
            exact(f, binary64),
        ),
    };
    let body = if binary64 {
        format!(
            "double a = __longlong_as_double((long long)x[i]), b = __longlong_as_double((long long)y[i]);\n  \
             double r = {call};\n  out[i] = r != r ? 0x7ff8000000000000ull : (unsigned long long)__double_as_longlong(r);"
        )
    } else {
        format!(
            "float a = __int_as_float((int)(unsigned)x[i]), b = __int_as_float((int)(unsigned)y[i]);\n  \
             float r = {call};\n  out[i] = r != r ? 0x7fc00000ull : (unsigned long long)(unsigned)__float_as_int(r);"
        )
    };
    source.push_str(&format!(
        "extern \"C\" __global__ void {CUDA_KERNEL}(const unsigned long long *x, const unsigned long long *y, \
         unsigned long long *out, unsigned long long n) {{\n  \
         unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;\n  \
         if (i >= n) return;\n  {body}\n}}\n"
    ));
    source
}

/// The oldest NVRTC that compiles [`cuda_map_source`] of `f` (#225).
pub fn cuda_map_min_nvrtc(f: MathFn, binary64: bool) -> (u32, u32) {
    core_math(f).map_or(invar_math::CUDA_MIN_NVRTC, |c| {
        invar_math::cuda_min_nvrtc(&[(c, binary64)])
    })
}

/// The refusal for a kernel that needs NVRTC `required` when `loaded` is older.
#[cfg(any(feature = "cuda", test))]
fn nvrtc_refusal(required: (u32, u32), loaded: (u32, u32)) -> Option<String> {
    (loaded < required).then(|| {
        format!(
            "{} CUDA kernels with anonymous-struct unions need NVRTC {}.{} or newer; \
             the loaded NVRTC is {}.{} (#225)",
            super::CONTRACT,
            required.0,
            required.1,
            loaded.0,
            loaded.1
        )
    })
}

/// Refuse before NVRTC runs when the loaded NVRTC is older than `required`,
/// so the caller sees the requirement instead of a compiler log.
#[cfg(feature = "cuda")]
pub fn nvrtc_admits(required: (u32, u32)) -> Result<(), invar_cuda::Error> {
    let loaded = invar_cuda::cuda::nvrtc_version()?;
    nvrtc_refusal(required, loaded).map_or(Ok(()), |m| Err(invar_cuda::Error::Compile(m)))
}

/// Compile and load [`cuda_map_source`] of `f` after [`nvrtc_admits`].
#[cfg(feature = "cuda")]
pub fn load_cuda_map(
    device: &mut invar_cuda::cuda::CudaDevice,
    f: MathFn,
    binary64: bool,
) -> Result<invar_cuda::cuda::BitsMap, invar_cuda::Error> {
    nvrtc_admits(cuda_map_min_nvrtc(f, binary64))?;
    device.load_bits_map(
        &cuda_map_source(f, binary64),
        &invar_math::CUDA_OPTIONS,
        CUDA_KERNEL,
    )
}

/// The binary32 expression for an exact function on Metal, over the
/// prelude's binary32 encodings `F` (exact through `D` or on the encoding).
fn msl_exact(f: MathFn) -> &'static str {
    match f {
        MathFn::Sqrt => "invar_f_sqrt(a)",
        MathFn::Floor => "invar_f_floor(a)",
        MathFn::Ceil => "invar_f_ceil(a)",
        MathFn::Trunc => "invar_f_trunc(a)",
        MathFn::Round => "invar_f_round(a)",
        MathFn::Abs => "invar_f_fabs(a)",
        MathFn::Copysign => "invar_f_copysign(a, invar_f_isnan(b.u) ? F::raw(0x3f800000u) : b)",
        MathFn::Min => "invar_f_minimum_number(a, b)",
        MathFn::Max => "invar_f_maximum_number(a, b)",
        _ => unreachable!("{} is a CORE-MATH function", f.name()),
    }
}

/// A complete MSL unit computing binary32 `f` elementwise with the kernel
/// [`CUDA_KERNEL`] (`x`, `y`, `out` as `ulong` buffers 0-2, `n` as `uint` in
/// buffer 3). Compiled as host C++ it instead defines
/// `invar_math_map_host(x, y, out, n)`, so the same text is checked on a CPU.
pub fn msl_map_source(f: MathFn) -> String {
    let (mut source, call) = match core_math(f) {
        Some(c) => {
            let args = if f.arity() == 2 { "a, b" } else { "a" };
            (
                invar_math::msl_source(&[c]),
                format!("{}({args})", invar_math::msl_symbol(c)),
            )
        }
        None => (
            format!("{}// invar-math-cr-v1 exact\n", invar_math::MSL_PRELUDE),
            msl_exact(f).to_owned(),
        ),
    };
    source.push_str(&format!(
        "static inline ulong invar_map_one(ulong xi, ulong yi) {{\n  \
         F a = F::raw((uint)xi), b = F::raw((uint)yi);\n  \
         F r = {call};\n  \
         return invar_f_isnan(r.u) ? 0x7fc00000ul : (ulong)r.u;\n}}\n\
         #ifdef __METAL_VERSION__\n\
         kernel void {CUDA_KERNEL}(device const ulong *x [[buffer(0)]], device const ulong *y [[buffer(1)]], \
         device ulong *out [[buffer(2)]], constant uint &n [[buffer(3)]], uint i [[thread_position_in_grid]]) {{\n  \
         if (i < n) out[i] = invar_map_one(x[i], y[i]);\n}}\n\
         #else\n\
         extern \"C\" void invar_math_map_host(const ulong *x, const ulong *y, ulong *out, ulong n) {{\n  \
         for (ulong i = 0; i < n; i++) out[i] = invar_map_one(x[i], y[i]);\n}}\n\
         #endif\n"
    ));
    source
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nvrtc_floor_names_both_versions_and_follows_the_source() {
        let message = nvrtc_refusal((12, 9), (12, 4)).unwrap();
        assert!(message.contains("need NVRTC 12.9 or newer"), "{message}");
        assert!(message.contains("loaded NVRTC is 12.4"), "{message}");
        assert!(message.starts_with("invar-math-cr-v1 "), "{message}");
        assert_eq!(nvrtc_refusal((12, 9), (12, 9)), None);
        assert_eq!(nvrtc_refusal((12, 9), (13, 0)), None);
        assert_eq!(nvrtc_refusal((12, 4), (12, 4)), None);
        assert_eq!(cuda_map_min_nvrtc(MathFn::Cos, true), (12, 9));
        assert_eq!(cuda_map_min_nvrtc(MathFn::Cos, false), (12, 4));
        // Exact functions emit no CORE-MATH text.
        assert_eq!(cuda_map_min_nvrtc(MathFn::Floor, true), (12, 4));
    }
}
