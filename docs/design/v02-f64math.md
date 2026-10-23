# v0.2 f64math: executable f64 and the invar-math-cr-v1 library

Status: in progress on `feat/v02-f64math`.
Objective: kinematics, estimation and control can be written in Invar with identical bits on every engine that supports the type.

## 1. Reuse survey (written before implementation)

The workstream needs correctly rounded binary32 and binary64 elementary functions in three places: the CPU engines, a CUDA emitter and an MSL emitter.

| Candidate | License | What it is | Verdict |
| --- | --- | --- | --- |
| CORE-MATH (gitlab.inria.fr/core-math/core-math) | MIT | Correctly rounded binary32 and binary64 C, one file per function, with published worst-case files | **Picked** as the single source of truth |
| `core-math` crate 1.4.0 (jdh8) | MIT | Rust bindings that build CORE-MATH with `cc` and `-march=native` | Rejected: `-march=native` makes the archive host-specific, it pins an older upstream (85430b3e) and it does not expose the C text the device emitters need |
| `pxfm` 0.1.30 | BSD-3 / Apache-2.0 | Pure Rust port of many CORE-MATH routines | Rejected: a second implementation to keep bit-equal with upstream, partial coverage of the list, and nothing reusable for CUDA or MSL |
| `libm` 0.2.16 | MIT / Apache-2.0 | musl port, faithfully rounded | Rejected: not correctly rounded, so engines would not be bit-identical |
| CRlibm | LGPL | Older correctly rounded binary64 library | Rejected: license, unmaintained, binary64 only |
| P0a and its compiled GNET device path (`invar-trig-f32-v1`) | in repo | Strict f32 sin, cos and sqrt plus the device plumbing | Superseded, see section 6 |

Decision: vendor the unmodified CORE-MATH C sources at upstream commit `dae5dd8c0d5bd2691b45de28c9edbe12f067c3d3` (2026-10-15) into `crates/invar-math/core-math/`, with `LICENSE`, `UPSTREAM` and `SHA256SUMS` recorded and checked by a unit test.
`crates/invar-math/build.rs` compiles the 40 translation units (20 functions in two formats) with `-O2 -ffp-contract=off -fno-fast-math -fno-math-errno -fexcess-precision=standard` into one static archive.
The same C text is exposed through `invar_math::source` for the device emitters, so every engine runs one algorithm.
The independent oracle for differential tests is CORE-MATH's own upstream build, compiled separately from a fresh clone, not this archive.

## 2. Type rules (strict-f64-v1)

`f64` is a primitive beside `f32`, with `Array<f64>`, record fields, variant and `Result` payloads, generic instantiation and `extern c fn` value arguments.
Arithmetic is IEEE 754 binary64 with round-to-nearest-even, gradual underflow and no contraction; only an explicit `fma` fuses.
The exact oracle is `src/numeric_contract/ieee.rs`, a rational single-rounding model shared by binary32 and binary64, versioned as `strict-f64-v1` in `NumericPolicy`.
NaNs form one class: comparisons follow IEEE, and values that leave a library function or a float-to-float conversion are the canonical quiet NaN (`0x7fc00000`, `0x7ff8000000000000`).
There is no float remainder, no float view and no implicit conversion between any two numeric types.

Literals: a decimal literal may carry an `f32` or `f64` suffix.
An unsuffixed literal takes its float type from context (an annotation, a parameter, the other operand or a return type) and defaults to `f32`, which keeps every existing program unchanged.
Any other identifier glued to a decimal (`1.0f16`, `1.0x`, `1.0f`) is `LEX_INVALID_LITERAL`.
The 256 significant digit cap of strict-f32-v1 is shared, so magnitudes below about 1e-254 are computed, not written.

## 3. Conversions (ruling C3, C15)

Constructors and conversions are type-suffixed: `f32_from_i32`, `f32_from_u32`, `f64_from_i32`, `f64_from_u32`, `f64_from_f32`, `f32_from_f64`, `i32_from_f32`, `u32_from_f32`, `i32_from_f64`, `u32_from_f64`, `filled_f64`.
Integer to float and `f64_from_f32` are exact; `f32_from_f64` rounds to nearest even.
Float to integer truncates toward zero and traps `INTEGER_OVERFLOW` at the call span when the input is NaN or the truncated value is out of range, for both f32 and f64 (C3).
Main had no float-to-integer conversion before, so this is a spec addition rather than a change to f32.

## 4. The invar-math-cr-v1 library

Functions are overloaded by argument type (C15) and every result is the correctly rounded value of the exact mathematical function.

| Group | Functions | Implementation |
| --- | --- | --- |
| Roots | `sqrt`, `cbrt`, `hypot` | `sqrt` is IEEE exact (oracle, LLVM `llvm.sqrt`), the others are CORE-MATH |
| Trigonometric | `sin`, `cos`, `tan`, `asin`, `acos`, `atan`, `atan2(y, x)` | CORE-MATH |
| Hyperbolic | `sinh`, `cosh`, `tanh` | CORE-MATH |
| Exponential | `exp`, `exp2`, `expm1`, `log`, `log2`, `log10`, `log1p`, `pow` | CORE-MATH |
| Exact | `floor`, `ceil`, `trunc`, `round`, `abs`, `copysign`, `min`, `max` | Oracle; natively LLVM intrinsics, with `min` and `max` as compare-and-select IR |

Documented choices:
- `round` rounds half away from zero (C `round`, `llvm.round`).
- `copysign(x, y)` treats a NaN sign source as positive, so the result never depends on a NaN's sign bit.
- `min` and `max` are IEEE 754-2019 minimumNumber and maximumNumber: a NaN operand is ignored, both NaN gives NaN, and `-0 < +0`.
- Every library result that is NaN is the canonical quiet NaN.

The names are reserved intrinsics, so a function, parameter or local named `sin` or `min` is `NAME_DUPLICATE`.
One existing test helper named `round` was renamed.

## 5. Engines

| Engine | f32 | f64 | Mechanism |
| --- | --- | --- | --- |
| reference evaluator | yes | yes | `src/eval/floating.rs` over the oracle and `invar_math` |
| MIR evaluator | yes | yes | same module, so both evaluators share one implementation |
| native-serial (LLVM) | yes | yes | `src/native/floating.rs`: bitcast i64 carriers, `fadd`/`fmul`/`fdiv` without fast-math flags, `llvm.{sqrt,floor,ceil,trunc,round,fabs,copysign}`, min and max as explicit compare-and-select IR (x86 `llvm.minimumnum.f64` does not order signed zeros), and calls to `cr_*` linked from the embedded CORE-MATH archive plus `-lm` for its fenv calls |
| cpu-parallel | yes | yes | runs the MIR evaluator semantics; f64 regions fall back to serial with the existing reason until the float map family lands |
| CUDA | yes | yes | `invar::math::cuda_map_source`: the vendored C compiled by NVRTC, see below |
| Metal | yes | refused | `invar::math::msl_map_source` for f32, see below; auto placement records `metal-no-f64` on `mir-serial` for every function that computes f64, which runs on the CPU; a forced `--engine metal` run with no f32 or integer region refuses with `METAL_UNAVAILABLE` |
| GNET | f32 only | refused | the general net has no binary64 encoding; `Rule::F64` is unsupported and `accepts_type` rejects f64 |

The CUDA and Metal rows are emitters verified bit for bit on the GPU through elementwise map kernels.
No device placement family lowers a library call yet, so programs that call these functions place on the CPU engines today; the emitters are what a family will call when it does.

Native executables carry f64 arrays as host array kind 4 (`Vec<i64>` of encodings).
`extern c fn` accepts `f64` value arguments as `INV_TAG_F64_V1` (5), whose `scalar` holds the binary64 encoding.

### CUDA emitter

`invar_math::cuda_source` builds one NVRTC translation unit from the unmodified vendored C.
Each function's file is spliced with its local headers, wrapped in its own namespace and followed by an `#undef` of every macro it defined, so any set of functions shares one program.
A prelude replaces the host headers: fixed-width integer typedefs, `fegetround` returning round-to-nearest, inert exception flags, and fixed-type wrappers for the GCC builtins upstream calls.
The host architecture macros are undefined, so every target takes upstream's portable code path.
`crates/invar-math/src/cxx.rs` makes the three C constructs C++ rejects acceptable without changing their meaning: designated initializers of anonymous structs inside unions become positional (the struct moves to the front of its union), the byte-order conditional resolves to its little-endian branch, and the statements a `goto` skips move into a block the jump passes whole.
Each rewrite refuses shapes it does not recognize, and a variable still used after a label fails to compile rather than changing meaning.
`invar_math::CUDA_OPTIONS` (`--fmad=false --ftz=false --prec-div=true --prec-sqrt=true -default-device --device-int128 --std=c++20`) keeps every operation upstream performs exact: IEEE `+ - * /`, `sqrt` and `fma`, rounding to integers, `fabs`, `copysign`, `fmin` and `fmax`.
The exact library functions use CUDA operations that are exact under the same options.
`invar_cuda::cuda::BitsMap` compiles, loads and runs one elementwise kernel over raw encodings; NaN results leave as the canonical quiet NaN.
The anonymous structs inside unions need NVRTC 12.9 or newer: NVRTC 12.4 rejects them, and every other unit compiles from 12.4.
`invar_math::cuda_min_nvrtc` derives the floor from the emitted units, and `invar::math::load_cuda_map` checks the loaded NVRTC against it before compiling, so an older NVRTC is a compile refusal that names both versions instead of a compiler log.

### Metal emitter

Metal has no binary64, but CORE-MATH's binary32 functions keep their intermediates in double, and their error analysis assumes IEEE binary64 operations.
`crates/invar-math/src/msl_prelude.h` therefore defines `double` as `D`, a struct holding a binary64 encoding whose `+ - * /`, `fma`, `sqrt`, conversions, comparisons and rounding to integers are computed with 64-bit integer arithmetic and round to nearest even exactly once, with gradual underflow.
Packing and rounding follow Berkeley SoftFloat 3e's `roundPackToF64` and `roundPackToF32`; the operations reduce to one 128-bit exact sum or product followed by that rounding, a 63-step restoring division and a 55-step digit-by-digit square root.
The 128-bit integers of the Payne-Hanek reductions in `sinf`, `cosf` and `tanf` are a two-word struct with C's modulo semantics.
This is not f64 emulation for Invar programs: `D` exists only inside the f32 library, and Invar's own f64 still refuses Metal with `metal-no-f64`.
`crates/invar-math/src/msl.rs` makes the C text valid MSL without changing its meaning.
It removes comments and applies the C++ rewrites above, turns every unsuffixed floating literal into `D::raw(<its exact binary64 encoding>)` (decimal literals through Rust's correctly rounded parser, hexadecimal ones exactly or refused), moves `static const` tables into the `constant` address space (function-local tables are hoisted with the macros their function defines and renamed, and tables whose initializers compute stay function-local), marks table views `constant` and result pointers `thread`, and maps the GCC builtins to fixed-type functions.
Metal flushes binary32 subnormal operands and results even with fast math disabled (the first Metal run returned 0 for `sinf(0x1p-149)` and missed one subnormal `exp2f` result), so `float` is defined as `F`, a struct holding a binary32 encoding.
Its `+ - * /` and `sqrt` compute in `D` and round once more to binary32, which is still correctly rounded because 53 >= 2 * 24 + 2; its `fma` rounds to odd in binary64 first and then to nearest even in binary32, which is correct because 53 >= 24 + 2; its comparisons, `fabs`, `copysign`, `fmin`, `fmax`, minimumNumber and maximumNumber work on the encodings.
No binary32 arithmetic reaches the GPU's float unit, so the result does not depend on its subnormal mode.
The conversions `D` and `F` need at compile time (table initializers in the `constant` address space) are `constexpr`; they count leading zeros with a loop only during constant evaluation and with `metal::clz` on the device.
The binary64-to-binary32 rounding is kept out of line because the macOS 27 AGX backend compiler crashes (`XPC_ERROR_CONNECTION_INTERRUPTED`) when it is inlined after `cbrtf`'s final mask of the low word.
Without `__METAL_VERSION__` the prelude maps the address spaces away and the same unit compiles as host C++, so CI checks the transform and the binary64 unit without a GPU.
`invar_metal::MetalDevice::load_bits_map` takes the pipeline through the device library cache (the differential and benchmarks open the device with `open_runtime_source`, since these kernels are verification harnesses and not compiled ahead), and `run_bits_map` runs the map kernel in chunks under the per-command deadline.

## 6. Numeric contract versions (ruling C4)

`invar-math-cr-v1` is the v0.2 default math contract.
It supersedes `invar-trig-f32-v1` from P0a and its compiled GNET device path, whose strict but not correctly rounded f32 `sin` and `cos` can differ in the last bit; `sqrt` is identical in both.
`invar explain --format json` and the `invar run --format json` receipt record `numeric_contracts` (`binary64: strict-f64-v1`, `math: invar-math-cr-v1`) whenever a program uses f64 or a library function; programs that use neither keep their previous documents byte for byte.
The MIR program's `numeric_policy` stays `strict-f32-v1` for binary32 and its identity hash is unchanged.
The v0.2 release notes must state the sin/cos bit change against trig-f32-v1.
There is no `--numeric-contract` flag: trig-f32-v1 stays the f32 default until #48 makes `invar-math-cr-v1` the default.

## 7. Acceptance evidence

| Check | Evidence |
| --- | --- |
| strict-f64-v1 oracle | `tests/issue_v02_f64_oracle.rs` (policy, literals, signed zero, NaN, fma, sqrt, underflow, conversions, exact functions, 600-round host differential) |
| CORE-MATH vendoring | `crates/invar-math/src/tests.rs` (checksum pin, spot values) |
| Source semantics on reference, MIR at four fuels and native | `tests/issue_v02_f64_source.rs` |
| f64 foreign argument | `tests/issue_l1_04_bytes_strings_cabi.rs::v02_f64_foreign_scalar_argument_matches_its_oracle` |
| Panda FK vs Pinocchio within 1e-12 on reference, cpu-parallel and native through the CLI | `tests/issue_v02_fk_panda.rs`, `examples/fk_panda.inv` |
| Every function and type lowered identically on reference, MIR and native | `tests/issue_v02_math_engines.rs` |
| CPU library vs an independent upstream CORE-MATH build: worst cases, specials, 10^6 random inputs per function and type, every 256th binary32 encoding of each univariate function, and a mutated coefficient | `tests/issue_v02_math_differential.rs` (ignored, heavy) |
| CUDA vs the same upstream build and the strict oracles, the same inputs, a mutated coefficient in the device source, and all forty functions in one program | `tests/issue_v02_math_cuda.rs` (ignored, GPU) |
| C-to-C++ rewrites and device source shape | `crates/invar-math/src/cxx.rs`, `crates/invar-math/src/device.rs` unit tests |
| Metal binary64 and binary32 units vs the host FPU (10^6 edge-biased cases per operation, subnormals and fma cancellation included), the prelude warning-free under `-Werror`, and every f32 function's MSL compiled as host C++ vs the CPU library | `tests/issue_v02_math_msl.rs` (runs in CI), `crates/invar-math/oracle/softdouble_check.cpp` |
| Metal vs the upstream build on the same inputs as CUDA (f32), the exact functions vs the strict oracles, and a mutated coefficient | `tests/issue_v02_math_msl.rs::metal` (ignored, Metal GPU) |
| MSL transform rules (literals, hoisting, shadowed tables, address spaces) | `crates/invar-math/src/msl.rs` unit tests |
| Metal refuses f64 with `metal-no-f64` | `tests/issue_v02_fk_panda.rs::v02_f64_on_metal_runs_on_the_cpu_with_metal_no_f64` |
| Performance next to libm, CUDA `__nv_*` and `metal::precise` | `bench/v02_math_bench.rs` (ignored), numbers in section 8 |

The Panda example reproduces the four Pinocchio 4.1.0 golden poses of `examples/cells/reach_place/components/pinocchio_fk/golden/panda_fk.json` with every rotation and translation entry within 1e-12; a host double model of the same chain differs from the goldens by at most 4.2e-15.

## 8. Performance (ruling C13)

Correct rounding costs speed; there is no gate in this workstream, and the numbers are reported as measured.
Each figure is the median of five timed repeats after one warm-up, from `bench/v02_math_bench.rs` (ignored), with the host's one-minute load average beside it.

### Metal, binary32, against `metal::precise`

Apple M4, macOS 27, 2^22 inputs in [0.01, 0.99] per function (`y` the same for `atan2` and `pow`), run exclusively, load1 4.58 and 4.60 for two consecutive runs.
Kernel time in milliseconds for the whole input set; the range covers the two runs, which differ by up to 4x on some functions because the GPU clock is not pinned.
"Not CR" counts the inputs where `metal::precise` differs from the correctly rounded result, and "max ulp" is its largest distance.

| Function | invar ms | precise ms | not CR (of 4194304) | max ulp |
| --- | --- | --- | --- | --- |
| sin | 65.9-66.0 | 1.07 | 1839172 | 3 |
| cos | 65.5-65.6 | 1.06 | 1839582 | 2 |
| tan | 18.4-30.0 | 0.68-0.85 | 2288501 | 4 |
| asin | 50.1-73.8 | 0.73-0.96 | 1585523 | 3 |
| acos | 43.6-85.2 | 0.69-0.75 | 1852408 | 3 |
| atan | 23.1-49.9 | 0.94-2.11 | 1274633 | 3 |
| atan2 | 48.8-112.2 | 1.25-1.47 | 1473871 | 3 |
| sinh | 18.2-72.1 | 0.67-1.15 | 1440702 | 2 |
| cosh | 17.6-73.2 | 0.68-0.75 | 602657 | 1 |
| tanh | 23.6-73.3 | 0.68-0.95 | 1161863 | 2 |
| exp | 9.5 | 0.66-0.67 | 1127342 | 1 |
| exp2 | 16.3-17.5 | 0.90-1.51 | 1048145 | 1 |
| log | 26.7-49.3 | 0.68-1.00 | 1911260 | 2 |
| log2 | 9.3-9.4 | 0.67-0.68 | 2020389 | 3 |
| log10 | 11.5-11.6 | 0.66-0.72 | 1282636 | 3 |
| pow | 28.0-110.0 | 1.68-1.86 | 1379244 | 2 |
| cbrt | 41.5-70.2 | n/a | | |
| hypot | 7.7-32.9 | n/a | | |
| expm1 | 21.9-73.6 | n/a | | |
| log1p | 11.0-11.1 | n/a | | |

Metal has no `precise` cbrt, hypot, expm1 or log1p.
The library is 12x to 114x slower than `metal::precise` on Metal because every binary32 and binary64 operation is emulated on integers (`D` and `F` in section 5); `metal::precise` misses correct rounding on 14% to 55% of these inputs by up to 4 ulp.
