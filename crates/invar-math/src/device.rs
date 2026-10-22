//! Device translation units built from the vendored CORE-MATH C text.
//!
//! The CUDA emitter compiles the unmodified upstream files as device C++ under
//! NVRTC. Each file is spliced with its local headers, wrapped in its own
//! namespace and followed by `#undef` of every macro it defined, so several
//! functions share one program without collisions. The prelude replaces the
//! host headers: fixed-width integers, a round-to-nearest `fegetround`, inert
//! exception flags, and the host architecture macros removed so every target
//! takes upstream's portable path. `crate::cxx` turns the C99 designated
//! initializers C++ rejects into positional ones. Only operations exact on
//! the device are used: IEEE `+ - * /`, `sqrt`, `fma`, rounding to integers,
//! `fabs`, `copysign`, `fmin` and `fmax`, with contraction off.

use super::{Function, source};

/// NVRTC options that keep every operation the upstream C performs exact.
pub const CUDA_OPTIONS: [&str; 8] = [
    "--fmad=false",
    "--ftz=false",
    "--prec-div=true",
    "--prec-sqrt=true",
    "-default-device",
    "--device-int128",
    "--std=c++20",
    "-w",
];

const CUDA_PRELUDE: &str = r#"// invar-math-cr-v1 CUDA prelude
#undef __x86_64__
#undef __i386__
#undef __aarch64__
#undef __arm64__
#undef __AVX__
#undef __SSE4_1__
#undef __ARM_ARCH
#undef __FMA__
typedef signed char int8_t;
typedef unsigned char uint8_t;
typedef short int16_t;
typedef unsigned short uint16_t;
typedef int int32_t;
typedef unsigned int uint32_t;
typedef long long int64_t;
typedef unsigned long long uint64_t;
typedef unsigned long long uintptr_t;
#define INT64_C(c) c##LL
#define UINT64_C(c) c##ULL
#define NAN __builtin_bit_cast(float, 0x7fc00000u)
#define INFINITY __builtin_bit_cast(float, 0x7f800000u)
#define FE_INVALID 1
#define FE_DIVBYZERO 4
#define FE_OVERFLOW 8
#define FE_UNDERFLOW 16
#define FE_INEXACT 32
#define FE_ALL_EXCEPT 61
#define FE_TONEAREST 0
#define FE_DOWNWARD 0x400
#define FE_UPWARD 0x800
#define FE_TOWARDZERO 0xc00
typedef unsigned int fexcept_t;
typedef unsigned int fenv_t;
static inline int fegetround(void) { return FE_TONEAREST; }
static inline int feraiseexcept(int) { return 0; }
static inline int feclearexcept(int) { return 0; }
static inline int fetestexcept(int) { return 0; }
static inline int fegetexceptflag(fexcept_t *f, int) { *f = 0; return 0; }
static inline int fesetexceptflag(const fexcept_t *, int) { return 0; }
static inline int feholdexcept(fenv_t *e) { *e = 0; return 0; }
static inline int feupdateenv(const fenv_t *) { return 0; }
#define printf(...) 0
#define exit(code) __trap()
#define __builtin_clz(x) __clz((int)(x))
#define __builtin_clzll(x) __clzll((long long)(x))
#define __builtin_ctz(x) (__ffs((int)(x)) - 1)
#define __builtin_ctzll(x) (__ffsll((long long)(x)) - 1)
// C's builtins take and return exactly double or float; fixed-type wrappers
// keep C's argument conversions where CUDA would pick an overload.
#define INVAR_D(name, cuda) static inline double invar_##name(double a) { return cuda(a); }
#define INVAR_F(name, cuda) static inline float invar_##name(float a) { return cuda(a); }
INVAR_D(fabs, fabs) INVAR_D(floor, floor) INVAR_D(sqrt, sqrt) INVAR_D(round, round) INVAR_D(rint, rint)
INVAR_F(fabsf, fabsf) INVAR_F(sqrtf, sqrtf) INVAR_F(roundf, roundf)
static inline double invar_fma(double a, double b, double c) { return fma(a, b, c); }
static inline float invar_fmaf(float a, float b, float c) { return fmaf(a, b, c); }
static inline double invar_fmax(double a, double b) { return fmax(a, b); }
static inline float invar_fmaxf(float a, float b) { return fmaxf(a, b); }
static inline double invar_fmin(double a, double b) { return fmin(a, b); }
static inline float invar_fminf(float a, float b) { return fminf(a, b); }
static inline double invar_copysign(double a, double b) { return copysign(a, b); }
static inline float invar_copysignf(float a, float b) { return copysignf(a, b); }
#define __builtin_fabs invar_fabs
#define __builtin_fabsf invar_fabsf
#define __builtin_floor invar_floor
#define __builtin_fma invar_fma
#define __builtin_fmaf invar_fmaf
#define __builtin_fmax invar_fmax
#define __builtin_fmaxf invar_fmaxf
#define __builtin_fmin invar_fmin
#define __builtin_fminf invar_fminf
#define __builtin_sqrt invar_sqrt
#define __builtin_sqrtf invar_sqrtf
#define __builtin_round invar_round
#define __builtin_roundf invar_roundf
#define __builtin_roundeven invar_rint
#define __builtin_copysign invar_copysign
#define __builtin_copysignf invar_copysignf
#define __builtin_isnan isnan
#define __builtin_inff() INFINITY
#define __builtin_nanf(tag) NAN
// GCC's __builtin_mul_overflow: the exact product of any two 64-bit integers,
// stored modulo 2^64 into *r, reporting whether it fits in *r's type.
template <class A, class B, class R>
static inline bool invar_mul_overflow(A a, B b, R *r) {
  static_assert(sizeof(A) == 8 && sizeof(B) == 8 && sizeof(R) == 8, "64-bit operands only");
  bool na = (A)-1 < 0 && (long long)a < 0, nb = (B)-1 < 0 && (long long)b < 0;
  unsigned __int128 ua = na ? (unsigned __int128)(0ull - (unsigned long long)a) : (unsigned long long)a;
  unsigned __int128 ub = nb ? (unsigned __int128)(0ull - (unsigned long long)b) : (unsigned long long)b;
  unsigned __int128 p = ua * ub;
  bool negative = na != nb && p != 0;
  *r = (R)(unsigned long long)(negative ? (unsigned __int128)0 - p : p);
  if ((R)-1 < 0) return negative ? p > ((unsigned __int128)1 << 63) : p >= ((unsigned __int128)1 << 63);
  return negative || p >> 64 != 0;
}
#define __builtin_mul_overflow(a, b, r) invar_mul_overflow((a), (b), (r))
"#;

/// The namespace that holds `f`'s translation unit.
pub(crate) fn namespace(f: Function, binary64: bool) -> String {
    format!("invar_cm_{}_{}", f.name(), if binary64 { 64 } else { 32 })
}

/// The qualified device entry point, e.g. `invar_cm_sin_64::cr_sin`.
pub fn cuda_symbol(f: Function, binary64: bool) -> String {
    let suffix = if binary64 { "" } else { "f" };
    format!("{}::cr_{}{suffix}", namespace(f, binary64), f.name())
}

/// Splice local headers, drop host includes and FENV pragmas, and record every
/// macro name the text defines.
pub(crate) fn splice(
    text: &str,
    headers: &[(&str, &str)],
    out: &mut String,
    defined: &mut Vec<String>,
) {
    for line in text.lines() {
        let trimmed = line.trim_start();
        if let Some(rest) = trimmed.strip_prefix("#include") {
            let rest = rest.trim();
            if let Some(name) = rest.strip_prefix('"').and_then(|r| r.split('"').next()) {
                let header = headers.iter().find(|(n, _)| *n == name);
                let (_, body) =
                    header.unwrap_or_else(|| panic!("unlisted CORE-MATH header {name}"));
                splice(body, headers, out, defined);
            }
            out.push('\n');
            continue;
        }
        if trimmed.starts_with("#pragma") {
            out.push('\n');
            continue;
        }
        if let Some(rest) = trimmed.strip_prefix("#define") {
            let name: String = rest
                .trim_start()
                .chars()
                .take_while(|c| c.is_alphanumeric() || *c == '_')
                .collect();
            if !defined.contains(&name) {
                defined.push(name);
            }
        }
        out.push_str(line);
        out.push('\n');
    }
}

/// The oldest NVRTC every kernel of this library compiles under.
pub const CUDA_MIN_NVRTC: (u32, u32) = (12, 4);
/// The oldest NVRTC that accepts an anonymous struct inside a union, which
/// several CORE-MATH sources declare; NVRTC 12.4 rejects it.
pub const CUDA_ANONYMOUS_STRUCT_NVRTC: (u32, u32) = (12, 9);

/// The oldest NVRTC that compiles [`cuda_source`] of `functions`.
pub fn cuda_min_nvrtc(functions: &[(Function, bool)]) -> (u32, u32) {
    let anonymous = functions.iter().any(|&(f, binary64)| {
        let (text, headers) = source(f, binary64);
        let mut unit = String::new();
        splice(text, headers, &mut unit, &mut Vec::new());
        crate::cxx::has_anonymous_struct_union(&unit)
    });
    if anonymous {
        CUDA_ANONYMOUS_STRUCT_NVRTC
    } else {
        CUDA_MIN_NVRTC
    }
}

/// One NVRTC translation unit holding the requested functions, deduplicated.
/// Callers append their kernels and call [`cuda_symbol`].
pub fn cuda_source(functions: &[(Function, bool)]) -> String {
    let mut out = String::from(CUDA_PRELUDE);
    let mut seen = Vec::new();
    for &(f, binary64) in functions {
        if seen.contains(&(f, binary64)) {
            continue;
        }
        seen.push((f, binary64));
        let (text, headers) = source(f, binary64);
        let mut defined = Vec::new();
        out.push_str(&format!("namespace {} {{\n", namespace(f, binary64)));
        let mut unit = String::new();
        splice(text, headers, &mut unit, &mut defined);
        out.push_str(&crate::cxx::rewrite(&unit));
        out.push_str("\n}\n");
        for name in defined {
            out.push_str(&format!("#undef {name}\n"));
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn anonymous_struct_unions_raise_the_nvrtc_floor() {
        // The binary64 units NVRTC 12.4 rejected on an H100 test host.
        for f in [
            Function::Cos,
            Function::Tan,
            Function::Atan2,
            Function::Log,
            Function::Log10,
            Function::Pow,
        ] {
            assert_eq!(
                cuda_min_nvrtc(&[(f, true)]),
                CUDA_ANONYMOUS_STRUCT_NVRTC,
                "{f:?}"
            );
        }
        // Every binary32 unit compiled under NVRTC 12.4 there.
        for f in Function::ALL {
            assert_eq!(cuda_min_nvrtc(&[(f, false)]), CUDA_MIN_NVRTC, "{f:?}");
        }
        let all: Vec<_> = Function::ALL.map(|f| (f, true)).to_vec();
        assert_eq!(cuda_min_nvrtc(&all), CUDA_ANONYMOUS_STRUCT_NVRTC);
        assert_eq!(cuda_min_nvrtc(&[]), CUDA_MIN_NVRTC);
        assert!(CUDA_MIN_NVRTC < CUDA_ANONYMOUS_STRUCT_NVRTC);
    }

    #[test]
    fn cuda_source_is_self_contained_and_namespaced() {
        let all: Vec<_> = [false, true]
            .into_iter()
            .flat_map(|b| Function::ALL.map(|f| (f, b)))
            .collect();
        let text = cuda_source(&all);
        assert!(!text.lines().any(|l| l.trim_start().starts_with("#include")));
        assert!(!text.contains("STDC FENV_ACCESS"));
        for &(f, b) in &all {
            let symbol = cuda_symbol(f, b);
            let (space, name) = symbol.split_once("::").unwrap();
            assert_eq!(
                text.matches(&format!("namespace {space} {{")).count(),
                1,
                "{symbol}"
            );
            assert!(
                text.contains(&format!("{name} (")) || text.contains(&format!("{name}(")),
                "{symbol}"
            );
        }
        // Every macro a unit defines is undefined before the next unit.
        assert!(text.contains("#undef DINT_H") && text.contains("#undef U128"));
        assert_eq!(
            cuda_source(&[(Function::Sin, true), (Function::Sin, true)]),
            cuda_source(&[(Function::Sin, true)])
        );
    }
}
