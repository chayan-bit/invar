//! Compile the vendored CORE-MATH sources once, with strict IEEE flags, into
//! `libinvar_coremath.a`. The same archive is linked into this crate and
//! embedded for native executables (see `ARCHIVE`).
use std::path::Path;

const FUNCTIONS: [&str; 20] = [
    "cbrt", "hypot", "sin", "cos", "tan", "asin", "acos", "atan", "atan2", "sinh", "cosh", "tanh",
    "exp", "exp2", "expm1", "log", "log2", "log10", "log1p", "pow",
];

fn main() {
    let root = Path::new("core-math");
    println!("cargo:rerun-if-changed=core-math");
    let mut build = cc::Build::new();
    for name in FUNCTIONS {
        build.file(root.join(format!("binary32/{name}/{name}f.c")));
        build.file(root.join(format!("binary64/{name}/{name}.c")));
    }
    // Round-to-nearest bracket for every call (see src/nearest.c).
    println!("cargo:rerun-if-changed=src/nearest.c");
    build.file("src/nearest.c");
    // CORE-MATH is GNU C (__int128, __builtin_*, <x86intrin.h>), which cl.exe
    // cannot compile; MSVC targets build it with clang-cl, whose Clang driver
    // flags must be wrapped in /clang: or they are ignored with only a warning.
    let msvc = std::env::var("CARGO_CFG_TARGET_ENV").is_ok_and(|env| env == "msvc");
    let clang = |flag: &str| {
        if msvc {
            format!("/clang:{flag}")
        } else {
            flag.to_owned()
        }
    };
    build.prefer_clang_cl_over_msvc(msvc);
    // Correct rounding is proved for IEEE binary64 arithmetic with only the
    // explicit __builtin_fma fused: no contraction, no fast-math, no errno.
    build
        .opt_level(2)
        .flag(clang("-ffp-contract=off"))
        .flag_if_supported(clang("-fno-fast-math"))
        .flag_if_supported(clang("-fno-math-errno"))
        .flag_if_supported(clang("-fexcess-precision=standard"))
        .define("_GNU_SOURCE", None)
        .warnings(false)
        .cargo_warnings(false);
    let compiler = build.get_compiler();
    assert!(
        !msvc || compiler.is_like_clang_cl(),
        "CORE-MATH needs clang-cl on MSVC targets, found {}: install the Visual Studio \
         component \"C++ Clang Compiler for Windows\" or set CC to clang-cl",
        compiler.path().display()
    );
    build.compile("invar_coremath");
    let archive = Path::new(&std::env::var("OUT_DIR").expect("cargo sets OUT_DIR"))
        .join("libinvar_coremath.a");
    println!("cargo:archive={}", archive.display());
}
