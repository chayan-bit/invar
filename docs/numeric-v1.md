# Numeric contract v1

This document is normative for M4-01 and freezes decisions D1 and D2.
The default policy is `{ version: 1, kind: "strict-f32" }`, named `strict-f32-v1`.
An unknown policy version or kind is refused with a typed diagnostic rather than approximated.
Version 2 ([numeric-v2.md](numeric-v2.md)) adds declared fold tiers and leaves every rule here unchanged.

## Source grammar

`f32` is an IEEE 754 binary32 scalar and `Array<f32>` is an affine owned array.
A decimal literal has grammar `digit+ "." digit+` and is rounded once from its exact decimal value to binary32 using round to nearest, ties to even.
Decimal literals contain at most 256 digits and unary negation is separate, including for negative zero.
`filled_f32(length: u32, value: f32) -> Array<f32>` constructs an array under the existing allocation limit.
`+`, `-`, `*`, `/`, unary `-`, and the six comparison operators accept matching f32 operands without implicit integer conversion.
`fma(a: f32, b: f32, c: f32) -> f32` computes the exact product plus addend with one rounding.
`f32_from_u32` and `f32_from_i32` round the exact integer to nearest even.
`sin`, `cos` and `sqrt` (each `f32 -> f32`) follow the frozen [`invar-trig-f32-v1`](trig-f32-v1.md) contract: `sqrt` is correctly rounded, and `sin` and `cos` are a fixed sequence of the operations above that is bit-identical on every engine but not correctly rounded.
`i32_from_f32(value: f32) -> i32` first requires a finite value in the closed interval `[i32::MIN, i32::MAX]`, then truncates toward zero.
An invalid `i32_from_f32` traps `INTEGER_OVERFLOW` at the source call, before publication of any subsequent result.
`Quantity<f32, Unit>` checks units before erasing the unit tag to f32 and must generate identical code to the equivalent bare f32 program.

## Strict semantics

Every arithmetic operation rounds independently to nearest even.
Subnormal inputs and results are preserved without flush to zero or denormals as zero.
Signed zeros are preserved according to IEEE arithmetic, and exact cancellation under nearest even produces positive zero except addition of two negative zeros.
Overflow produces the signed infinity and invalid floating arithmetic produces NaN rather than an integer arithmetic trap.
NaN payloads and NaN signs are unobservable in M4 and every NaN result compares as the same NaN class for engine agreement.
Source comparisons are IEEE unordered for NaNs, so every ordered comparison and equality is false and inequality is true.
Positive and negative zero compare equal.
Strict engine agreement requires identical non-NaN words and NaN-class equality.
Implicit FMA, contraction, reassociation, and silent fast math are forbidden.
A strict reduction is a source-order left fold from its specified initial value with one rounding per addition.
A strict inclusive scan emits each prefix of that same left fold, including the initial value in the first addition.
An empty reduction returns its initial value and an empty scan has no output elements.
A strict rank-2 contraction initializes each cell to positive zero and visits the inner axis in increasing source order with separate multiply and add roundings.
Parallelism may distribute independent cells or map elements but must preserve the ordered operations within each cell or prefix.

## Explicit relaxed contracts

Explicit `fma` changes only that operation's rounding and does not relax adjacent operations.
`reassoc_add(acc, x)` explicitly permits reassociation of its reduction, with error bounded by `(n - 1) * 2^-24 * sum(abs(x_i))` against the source-order reference for the finite domain covered by that contract.
`tc_mul_add` selects the separate `tc-f16-f32-v1` contract and does not change ordinary f32 operations.
Ordinary checking admits it only as the direct-cell step of a recognized fresh-zero matrix contraction.
Reference, serial MIR, the CPU net oracle and CPU/auto fallback round inputs to binary16, retain source control flow/fuel/bounds/cleanup, and return the once-rounded exact sum of products.
Nonfinite or overflowing reached inputs report `TC_INPUT_RANGE`; an unreached contraction does not inspect its inputs.
The CPU net oracle executes its own activations, fuel, traps and cleanup; only binary16 rounding and exact D5 arithmetic are shared. Native executables use the same pure integer D5 oracle at successful function completion; native library export remains explicitly unsupported for TC.
Device completion may return any result within the amended D5 bound. A quiescent device refusal uses the exact serial path; unknown completion still forbids replay or publication.
Unsupported relaxed syntax or policies must be refused and cannot silently become strict or fast arithmetic.
M4-01 does not implement a Tensor Core schedule or claim performance acceptance.

## Device admission and fallback

Every process probes the live device under the pinned strict compiler options before admitting arithmetic, comparisons, or conversions.
CUDA requires disabled FMA contraction and FTZ and explicit precise division and square-root options.
Metal requires disabled fast math and `#pragma METAL fp contract(off)`.
Division remains CPU-only unless device admission establishes its required rounding and subnormal behavior.
Source `sin`, `cos` and `sqrt` run on a device only through the `invar-trig-f32-v1` helpers ([`trig-f32-v1.md`](trig-f32-v1.md)), which never read or produce a subnormal; they are admitted when every probe mismatch of the operations they use is a subnormal-class case.
A flushing Metal device may execute an operation only behind a sound dispatch guard covering every possible divergence under the documented device model.
A flagged dispatch is discarded after confirmed completion and rerun on the exact CPU path with an explicit fallback reason.
Unknown completion quarantines work and forbids publication and CPU replay.
An operation whose guard cannot be justified is refused before submission.
Policy, IR version, layout version, target capability, compiler options, and generated source all participate in artifact and cache identity.

## Semantic fixtures

The committed `issue_m4_01_numeric_oracle`, `issue_m4_01_decimal_literals`, and `issue_m4_01_source_f32` fixtures establish literal rounding, signed zeros, infinities, NaNs, cancellation, conversion traps, and independent exact CPU outcomes.
The committed `issue_m4_01_device/corpus.rs` establishes zero, one, odd, rectangular, and subnormal contraction cases.
Device or mock-only outcomes do not establish native, CPU-parallel, or hardware acceptance for a missing engine.
