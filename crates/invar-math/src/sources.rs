//! The vendored CORE-MATH C text, embedded for the CUDA and MSL emitters.
//! Generated from the file list in `build.rs`; `tests.rs` pins the bytes.

use super::Function;

type Headers = &'static [(&'static str, &'static str)];

macro_rules! file {
    ($path:literal) => {
        include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/core-math/", $path))
    };
}

pub(super) fn get(f: Function, binary64: bool) -> (&'static str, Headers) {
    match (f, binary64) {
        (Function::Cbrt, false) => (file!("binary32/cbrt/cbrtf.c"), &[]),
        (Function::Cbrt, true) => (file!("binary64/cbrt/cbrt.c"), &[]),
        (Function::Hypot, false) => (file!("binary32/hypot/hypotf.c"), &[]),
        (Function::Hypot, true) => (file!("binary64/hypot/hypot.c"), &[]),
        (Function::Sin, false) => (file!("binary32/sin/sinf.c"), &[]),
        (Function::Sin, true) => (file!("binary64/sin/sin.c"), &[]),
        (Function::Cos, false) => (file!("binary32/cos/cosf.c"), &[]),
        (Function::Cos, true) => (file!("binary64/cos/cos.c"), &[]),
        (Function::Tan, false) => (file!("binary32/tan/tanf.c"), &[]),
        (Function::Tan, true) => (file!("binary64/tan/tan.c"), &[]),
        (Function::Asin, false) => (file!("binary32/asin/asinf.c"), &[]),
        (Function::Asin, true) => (file!("binary64/asin/asin.c"), &[]),
        (Function::Acos, false) => (file!("binary32/acos/acosf.c"), &[]),
        (Function::Acos, true) => (file!("binary64/acos/acos.c"), &[]),
        (Function::Atan, false) => (file!("binary32/atan/atanf.c"), &[]),
        (Function::Atan, true) => (file!("binary64/atan/atan.c"), &[]),
        (Function::Atan2, false) => (file!("binary32/atan2/atan2f.c"), &[]),
        (Function::Atan2, true) => (
            file!("binary64/atan2/atan2.c"),
            &[("tint.h", file!("binary64/atan2/tint.h"))],
        ),
        (Function::Sinh, false) => (file!("binary32/sinh/sinhf.c"), &[]),
        (Function::Sinh, true) => (file!("binary64/sinh/sinh.c"), &[]),
        (Function::Cosh, false) => (file!("binary32/cosh/coshf.c"), &[]),
        (Function::Cosh, true) => (file!("binary64/cosh/cosh.c"), &[]),
        (Function::Tanh, false) => (file!("binary32/tanh/tanhf.c"), &[]),
        (Function::Tanh, true) => (file!("binary64/tanh/tanh.c"), &[]),
        (Function::Exp, false) => (file!("binary32/exp/expf.c"), &[]),
        (Function::Exp, true) => (file!("binary64/exp/exp.c"), &[]),
        (Function::Exp2, false) => (file!("binary32/exp2/exp2f.c"), &[]),
        (Function::Exp2, true) => (file!("binary64/exp2/exp2.c"), &[]),
        (Function::Expm1, false) => (file!("binary32/expm1/expm1f.c"), &[]),
        (Function::Expm1, true) => (file!("binary64/expm1/expm1.c"), &[]),
        (Function::Log, false) => (file!("binary32/log/logf.c"), &[]),
        (Function::Log, true) => (
            file!("binary64/log/log.c"),
            &[("dint.h", file!("binary64/log/dint.h"))],
        ),
        (Function::Log2, false) => (file!("binary32/log2/log2f.c"), &[]),
        (Function::Log2, true) => (file!("binary64/log2/log2.c"), &[]),
        (Function::Log10, false) => (file!("binary32/log10/log10f.c"), &[]),
        (Function::Log10, true) => (
            file!("binary64/log10/log10.c"),
            &[("dint.h", file!("binary64/log10/dint.h"))],
        ),
        (Function::Log1p, false) => (file!("binary32/log1p/log1pf.c"), &[]),
        (Function::Log1p, true) => (
            file!("binary64/log1p/log1p.c"),
            &[("dint.h", file!("binary64/log1p/dint.h"))],
        ),
        (Function::Pow, false) => (file!("binary32/pow/powf.c"), &[]),
        (Function::Pow, true) => (
            file!("binary64/pow/pow.c"),
            &[
                ("pow.h", file!("binary64/pow/pow.h")),
                ("dint.h", file!("binary64/pow/dint.h")),
                ("qint.h", file!("binary64/pow/qint.h")),
            ],
        ),
    }
}
