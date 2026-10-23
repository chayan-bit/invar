# Trig contract v1 (`invar-trig-f32-v1`)

This document is normative for the source intrinsics `sin`, `cos` and `sqrt` and freezes decision D6.
It extends [`strict-f32-v1`](numeric-v1.md): every rule there still holds, and this contract only adds three intrinsics.

`sin(x: f32) -> f32`, `cos(x: f32) -> f32` and `sqrt(x: f32) -> f32` take and return IEEE 754 binary32 values without implicit conversion.
`sin` and `cos` are a frozen software sequence of IEEE basic operations and explicit fused multiply-adds.
They are not correctly rounded and they are never a vendor or libm `sin`/`cos`.
`sqrt` is correctly rounded on every input.
Every engine (exact CPU oracle, reference evaluator, MIR evaluator, native LLVM, CUDA and Metal) must return the same 32-bit word for every input, including the NaN encoding.

## Selecting this contract (`--numeric-contract trig-f32-v1`)

Ruling C4 (#48) makes `invar-math-cr-v1` the v0.2 default for f32 `sin` and `cos` and keeps this contract for one release behind an option.
- `invar run`, `invar explain` and `invar build` take `--numeric-contract trig-f32-v1`. It applies to every program the process checks.
- After typing, every f32 `sin` and `cos` call then takes the name of its helper (`sin@trig-f32-v1`, `cos@trig-f32-v1`; not identifiers, so source cannot write them). MIR carries the choice, and the `numeric_contracts.trig` record of explain and run receipts names this contract.
- `sqrt` is correctly rounded in both contracts and keeps its name. f64 calls and every other library function stay `invar-math-cr-v1`.
- Until the default switch lands, the default already runs this contract, so the option changes no bits. The reference evaluator, MIR, cpu-parallel and native-serial run the named helpers. The device paths (GNET compiled kernels, CUDA and Metal task regions) do not recognize the named helpers yet; that is #48 part B.
- Code: `src/cli.rs`, `math::select_trig_f32_v1`, `src/check/numeric/trig.rs`. Tests: `tests/issue_c4_trig_option.rs`.

## Single source of truth

The sequence is written once, in `crates/invar-device/src/trig_f32.rs`, as two generic functions `sincos` and `sqrt` over the `Ops` trait.
No engine spells the algorithm itself; each one replays those functions through a backend:

| Engine | Backend | Floating primitives |
| --- | --- | --- |
| Exact oracle (`F32Bits::sin`, `cos`, `sqrt_sequence` in `src/numeric_contract.rs`) | `Exact`: the BigRational strict-f32-v1 operations | `F32Bits::{add, sub, mul, fma, sqrt, from_u32}` |
| Reference and MIR evaluators, CPU source regions, GNET oracle | `Host` (`host_sin`, `host_cos`, `host_sqrt`) | Rust `f32` `+ - *`, `mul_add`, `sqrt`, `as f32` |
| Native CPU (`src/native/emit.rs`) | LLVM text from `helpers(Lang::Llvm, ..)` | `fadd`, `fsub`, `fmul` without fast-math flags, `@llvm.fma.f32`, `@llvm.sqrt.f32`, `uitofp` |
| CUDA (`f32_ops::emit`) | CUDA C from `helpers(Lang::Cuda, ..)` | `__fadd_rn`, `__fsub_rn`, `__fmul_rn`, `__fmaf_rn`, `__fsqrt_rn`, `__uint2float_rn` |
| Metal (`f32_ops::emit`) | MSL from `helpers(Lang::Msl, ..)` under `#pragma METAL fp contract(off)` and disabled fast math | `+ - *`, `fma`, `sqrt`, `float(uint)` |
| Compiled GNET C++, CUDA and MSL | `helpers(Lang::Gnet, ..)` over the existing bounded integer software helpers | `gnet_f32_{add,sub,mul,fma,from_u32}`, integer correctly rounded `gnet_f32_sqrt` |

Every other step is exact 32-bit integer work on encodings (masks, shifts, integer add and subtract, unsigned compare, select and table loads).
The CUDA spellings are the explicit round-to-nearest intrinsics, so `-fmad`, `-ftz` and `-prec-sqrt` cannot change them.
The compiled GNET path performs no floating arithmetic. Its square-root primitive
is generated from this same source file and preserves zero and positive infinity.
For a positive normal operand, normalize the unbiased exponent to an even `e`
and form the integer radicand `n = significand << 23` (at most 48 bits).
Compute its integer square root `q` by 24 bounded trial bits, then increment
exactly when `n - q*q > q`. The midpoint square is `q*q + q + 1/4`, so an
integer radicand cannot tie. Packing `q` at exponent `e/2` gives the correctly
rounded binary32 result. The shared `sqrt` sequence above this primitive handles
NaNs, negative inputs and the exact subnormal scale. This route therefore needs
no hardware sqrt accuracy or denormal guarantee.
Native LLVM functions carry `"denormal-fp-math"="ieee,ieee"` and no fast-math flags, and `@llvm.fma.f32` is the only fused operation.
The unit test `emitters_spell_every_floating_operation_once` checks that each emitted text has exactly the sequence's 25 fused, 28 other and 2 square-root operations.

## Source algorithm and attribution

The `sin` and `cos` sequence is Arm optimized-routines, AdvSIMD single-precision `sincosf`:

- repository `https://github.com/ARM-software/optimized-routines`, commit `97362b2eab91d35d30c4ac2c8c79dcd4dae4331b`;
- `math/aarch64/advsimd/v_sincosf_common.h` (`v_sincosf_inline`, the fast path for |x| <= 2^20);
- `math/aarch64/advsimd/v_trigf_fallback.h` (`large_range_reduction`, `sincos_eval`, the 4/pi and sin/cos(k pi/4) tables);
- Copyright (c) 2023-2026 Arm Limited, `SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception`, which is compatible with this repository.

It was chosen over the other candidates surveyed:

| Candidate | Why not |
| --- | --- |
| CORE-MATH `sinf`/`cosf` (correctly rounded) | the reduction and evaluation use binary64, which Metal does not have |
| musl/FreeBSD `__sindf`/`__rem_pio2f` | binary64 kernels and an integer Payne-Hanek reduction with 64-bit products |
| glibc `s_sinf.c` (Szabolcs Nagy) | binary64 polynomial and reduction |
| SLEEF `sinf_u1` | binary32-only too, but evaluated in double-float (float-float) arithmetic throughout, a longer sequence for the same 1-ULP class |
| Arm `v_sincosf` | binary32-only throughout, fma-based, a float-only large reduction with 4-term expansions of 4/pi, branch-free inside each path, published bound 1.17 + 0.5 ULP (sin) and 1.31 + 0.5 ULP (cos) below 2^20 and about 1.43 + 0.5 ULP above |

Constants, table entries and the order of every rounding are kept from Arm.
The deviations, each marked where it occurs in `trig_f32.rs`:

1. **Tiny path.** For |x| < 2^-12 the result is `x` for `sin` and `1.0` for `cos`, computed without arithmetic.
   Both are correctly rounded there, it keeps `sin(-0) = -0` (Arm's fast path returns `+0`), and it keeps subnormal arguments away from flushing devices.
2. **Special values.** NaN and infinite arguments return the canonical NaN `0x7fc00000` (Arm propagates the input NaN through arithmetic).
3. **Exact conversions as integer work.** Arm's `vcvtq_s32_f32(q)` becomes `bits(q0) - bits(0x1.8p23)`, exact because |n| < 2^20.
   Arm's `vcvtnq_s32_f32(kd)` becomes `bits(kd + 0x1.8p23) - bits(0x1.8p23)`, exact because kd is integral with |kd| < 2^18.
   Arm's `vrndaq_f32` (round to integral, ties away) is spelled with integer operations on the encoding (`round_away`), checked against the host for every exponent and tie.
4. **Exponent indexing.** The large path takes the 8-bit exponent field where Arm shifts the word including the sign bit; the extra 256 changes neither the table row (modulo 16) nor the exponent offset (modulo 8).
5. **One function for both results.** The fast path selects with `m = n + cosine`, which is Arm's `sincos` selection with the quadrant shifted by one; the large path computes Arm's sine and cosine angle-addition formulas and selects one.
   The selected value is the same word Arm's `sincosf` returns for that output.

## Exact operation sequence

`a = bits(x) & 0x7fffffff`.
Branches are structured: only the taken arm executes.

1. If `a < 0x39800000` (2^-12): return `cosine ? 0x3f800000 : x`.
2. Else if `a >= 0x7f800000`: return `0x7fc00000`.
3. Else if `a <= 0x49800000` (2^20), the **fast path**:
   1. `q0 = fma(x, 0x1.45f306p-1, 0x1.8p23)`, `q = q0 - 0x1.8p23`, `n = bits(q0) - bits(0x1.8p23)`.
   2. `r = fma(-q, 0x1.921fb6p+0, x)`, then `r = fma(-q, -0x1.777a5cp-25, r)`, then `r = fma(-q, -0x1.ee59dap-50, r)`.
   3. `r2 = r * r`, `r3 = r * r2`.
   4. `s = fma(r2, -0x1.994eb4p-13, 0x1.11076p-7)`, `s = fma(r2, s, -0x1.555546p-3)`, `s = fma(r3, s, r)`.
   5. `r4 = r2 * r2`, `p = fma(r2, 0x1.99e0eep-16, -0x1.6c0c1ap-10)`, `c = fma(r2, 0x1.55554ap-5, -0.5)`, `c = fma(r4, p, c)`, `c = fma(c, r2, 1.0)`.
   6. `m = n + cosine`; the result is `(m & 1 ? c : s)` with its sign bit flipped by `(m & 2) << 30`.
4. Else the **large path** (finite |x| > 2^20):
   1. `e = (bits(x) >> 23) & 0xff`, `t = e - 159`, `row = ((t >>arith 3) + 3) & 15`, and `xr` is `x` with its exponent field replaced by `(t & 7) + 159`, so `xr` lies in [2^32, 2^40) with the sign of `x`.
   2. `d0..d3 = FOUR_OVER_PI[4 row .. 4 row + 3]`.
      With `two_prod(a, b) = (h = a * b, fma(a, b, -h))`: `(_, ph_lo) = two_prod(xr, d0)`, `(pm_hi, pm_lo) = two_prod(xr, d1)`, `(pl_hi, pl_lo) = two_prod(xr, d2)`.
   3. `kd = round_away(ph_lo + pm_hi)`, `y_hi = (ph_lo - kd) + pm_hi`.
      With `fast_two_sum(a, b) = (h = a + b, b - (h - a))`: `(mid_hi, mid_lo) = fast_two_sum(pm_lo, pl_hi)`, `y_l = fma(xr, d3, pl_lo)`, `(y_hi, y_lo) = fast_two_sum(y_hi, mid_hi)`, `y_lo = y_lo + (mid_lo + y_l)`, `k = bits(kd + 0x1.8p23) - bits(0x1.8p23)`.
   4. `r = fma(y_hi, 0x1.921fb6p-1, fma(y_hi, -0x1.777a5cp-26, y_lo * 0x1.921fb6p-1))`, `r2 = r * r`, `r3 = r2 * r`.
   5. `sr = fma(r3, fma(r2, fma(r2, -0x1.994eb4p-13, 0x1.11076p-7), -0x1.555546p-3), r)`.
      `cm = fma(r2, fma(r2, fma(r2, 0x1.99e0eep-16, -0x1.6c0c1ap-10), 0x1.55554ap-5), -0.5) * r2`.
   6. `sk, ck = SINCOS_K_PI_OVER_4[2 (k & 7)], [2 (k & 7) + 1]`.
      `sin = fma(ck, sr, sk * cm) + sk`, `cos = fma(-sk, sr, ck * cm) + ck`, and the result is the one selected by `cosine`.

Inner calls evaluate innermost first; every line is one rounding, in the order written.
The table words are listed in `trig_f32.rs` as binary32 encodings converted from Arm's hex-float literals and checked against their spellings by a unit test.

`sqrt(x)`:

1. NaN, or a negative nonzero `x` (including `-inf` and negative subnormals): return `0x7fc00000`.
2. `+0` and `-0` are returned unchanged, and `+inf` returns `+inf`.
3. A positive subnormal `x = m * 2^-149` returns `sqrt(float(2 m)) * 2^-75`: `2 m < 2^24` converts exactly, the square root of a normal operand is correctly rounded, and the power-of-two product is exact and normal.
4. Otherwise return `sqrt(x)` correctly rounded.

## Special values, signed zeros and NaN payloads

| Input | `sin` | `cos` | `sqrt` |
| --- | --- | --- | --- |
| `+0` / `-0` | `+0` / `-0` | `1.0` | `+0` / `-0` |
| subnormal `x` | `x` | `1.0` | positive: correctly rounded; negative: `0x7fc00000` |
| `+inf` / `-inf` | `0x7fc00000` | `0x7fc00000` | `+inf` / `0x7fc00000` |
| any NaN (quiet or signaling, any sign or payload) | `0x7fc00000` | `0x7fc00000` | `0x7fc00000` |

The result never depends on an input NaN's payload or sign.
Source programs still observe NaNs only as a class, as strict-f32-v1 requires; the bit-level tests pin the canonical word.

## Subnormals and flushing devices

The exhaustive CPU audit (`p0a_trig_exhaustive_cpu`) records, for every input, the smallest nonzero magnitude reached by any floating operand, result or exact fma product of the sequence, and every operation that rounds a nonzero exact value to zero.
Measured over all 2^32 inputs: the smallest such magnitude is exactly 2^-126, the smallest normal (the `sqrt` operand for input `0x00800000`), and no operation underflows to zero.
So no floating operation of the sequence reads or produces a subnormal, and a device that flushes subnormals (FTZ or DAZ) returns the same words.
Subnormal arguments reach `sin` and `cos` only through the integer tiny path and reach `sqrt` only through the exact integer-to-float scaling.
This is why the CUDA and Metal emitters need no dispatch guard for these nodes, unlike strict-f32-v1 arithmetic on a flushing Metal device (see `numeric-v1.md`, device admission).

Device admission (`f32_ops::admits_trig`) still runs the live strict probe: every probe case that mismatches on the device for an operation the helpers use (`+`, `-`, `*`, `fma` for trig; `*`, `uint` conversion and `sqrt` for `sqrt`) must be a subnormal-class case, or the plan is refused before submission.
CUDA additionally requires `admits_strict()` as for every strict plan.

## Accuracy

Measured on the CPU over all 2^32 inputs against the correctly rounded result (binary64 libm rounded once to binary32; inputs whose binary64 value lies within two binary64 ULPs of a binary32 rounding boundary are reported as ambiguous and excluded from the integer count):

| Function | Max distance from the correctly rounded result | Max error (ULP, real-valued) | Max error, magnitude up to 2^20 | Max error, magnitude above 2^20 | Results 0 / 1 / 2 ULP from correctly rounded |
| --- | --- | --- | --- | --- | --- |
| `sin` | 2 ULP (first at `0x442b2762`) | 1.9243 at `0x51edf31e` | 1.6689 | 1.9243 | 3,880,147,584 / 397,698,034 / 344,428 |
| `cos` | 2 ULP (first at `0x44166f5f`) | 1.9345 at `0x75d13c2d` | 1.8067 | 1.9345 | 3,876,779,250 / 401,065,718 / 345,088 |

The bound of the contract is therefore **2 ULP** from the correctly rounded result for every finite input, and the measured real-valued errors match Arm's published bounds (1.17 + 0.5 and 1.31 + 0.5 below 2^20, 1.42 + 0.5 and 1.43 + 0.5 above).
58 of the 2 x 4,278,190,080 finite evaluations had an ambiguous binary64 reference and are left out of the distance count only; their real-valued error is included.
The inputs nearest to a multiple of pi are `0x6ff9be45` for `sin` (|sin x| = 3.2e-9) and `0x6f79be45` for `cos` (|cos x| = 1.6e-9); both are golden rows.

`sqrt` equals the correctly rounded square root on every input (checked against `f64::sqrt` rounded to binary32, which is exact for square roots because 53 >= 2 * 24 + 2).

## Verification

| Check | Where | Result |
| --- | --- | --- |
| Golden hard cases on oracle and host, special values, sampled oracle stride | `tests/issue_p0a_trig_f32.rs` (default) | in CI |
| Source intrinsics on eval, MIR eval and native, CPU source regions, device-source embedding, admission | `tests/issue_p0a_trig_f32.rs` (default, `native` feature for native) | in CI |
| CPU exhaustive: oracle stride, ULP bound, sqrt rounding, NaN policy, subnormal audit, full digest | `p0a_trig_exhaustive_cpu` | 2^32 inputs, 779 s on 8 threads: digests equal `FULL_DIGEST`, 16,711,936 inputs (every 257th) replayed on the exact oracle with 0 mismatches, `sqrt` correctly rounded everywhere, NaN policy holds, audit as above |
| Native LLVM exhaustive vs host | `p0a_trig_exhaustive_native` | 2^32 inputs x 3 functions, 0 mismatches, 28 s, digests equal `FULL_DIGEST` |
| Metal exhaustive vs host | `p0a_trig_exhaustive_metal` | 2^32 inputs x 3 functions on an Apple M4 through the production MSL emitter, 0 mismatches, about 220 s in four slices |
| CUDA exhaustive vs host | `p0a_trig_exhaustive_cuda` | pending a CUDA host |

The exact oracle costs about 50 to 100 microseconds per evaluation (BigRational arithmetic), so replaying all 2^32 x 3 evaluations on it would take months of CPU time.
The host backend replays the same generic sequence with IEEE binary32 operations, which round exactly as the oracle's operations do, and the stride check confirms that on 16.7 million inputs spread over every exponent.

The full-sweep digest (`FULL_DIGEST` in the test) is FNV-1a 64 over the little-endian result words of each 2^24-input block, folded over the 256 block digests in order, per function.
Pending CUDA commands, on a host with a CUDA 12.4-12.9 driver and NVRTC:

```sh
cargo test --locked --features cuda --test issue_p0a_trig_f32 -- --ignored p0a_trig_exhaustive_cuda --nocapture
INVAR_TRIG_FIRST_CHUNK=0 INVAR_TRIG_CHUNKS=64 cargo test --locked --features cuda --test issue_p0a_trig_f32 -- --ignored p0a_trig_exhaustive_cuda --nocapture
```

The second form bounds one run to 64 of the 4295 one-million-input chunks.
