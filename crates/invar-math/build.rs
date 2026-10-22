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
    // Correct rounding is proved for IEEE binary64 arithmetic with only the
    // explicit __builtin_fma fused: no contraction, no fast-math, no errno.
    build
        .opt_level(2)
        .flag("-ffp-contract=off")
        .flag_if_supported("-fno-fast-math")
        .flag_if_supported("-fno-math-errno")
        .flag_if_supported("-fexcess-precision=standard")
        .define("_GNU_SOURCE", None)
        .warnings(false)
        .cargo_warnings(false);
    build.compile("invar_coremath");
    let archive = Path::new(&std::env::var("OUT_DIR").expect("cargo sets OUT_DIR"))
        .join("libinvar_coremath.a");
    println!("cargo:archive={}", archive.display());
}
