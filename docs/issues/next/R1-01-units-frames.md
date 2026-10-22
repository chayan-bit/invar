# R1-01 - Units of measure and coordinate-frame types

Release: v0.1 (milestone R1); moved from v0.2 on 2026-09-26 because the system integration track (S1) needs it; expanded from a planning contract into a full contract.

Dependencies: L1-03.

## Goal

Make physical units and coordinate frames part of the type system at zero runtime cost.

## Motivation

Invar's robotics direction is an OEM supplier model: each new manufacturer's hardware should be cheap to support, because robot-specific facts become typed compiler inputs instead of per-customer hand-written code.

## Scope

- Dimension-checked unit types (length, angle, time, mass, force, torque and derived units) with explicit scale units (for example mm and m, rad and deg).
- Frame-tagged vectors, points, rotations and transforms; arithmetic across frames requires an explicit transform value whose source and target frames match.
- Explicit, checked conversions between scales; no implicit conversion.
- Units and frames erase completely before MIR codegen, so generated CPU, CUDA and Metal code is identical to the unit-free program.

## System integration impact (from S1)

Added 2026-09-26 by the integration plan (`docs/system-integration-v1.md` sections 2 and 6); the scope above is unchanged.

- Representation: quantities are `Quantity<T, Unit>` over the L1-03 generics, with the SI base dimensions (length, mass, time, current, temperature, amount, luminous intensity) plus angle as a tracked dimension, and dimension algebra (product, quotient, integer power) at type level.
- Frames are nominal types declared with `frame Name` or `frame Name : Parent`; transforms are `Transform<From, To>`; composition `Transform<A, B> * Transform<B, C>` yields `Transform<A, C>` and any other composition is rejected.
- Rates and durations are ordinary unit types: `Hz` (dimension time^-1) and `s`, `ms`, `us`, `ns`; `1 / Quantity<f64, s>` has dimension `Hz`.
- Component port types (S1-01) reuse exactly these types; there is no separate port type system and no translation layer.
- The declared frame tree has static closure checks: every frame used is reachable from exactly one root, the parent relation is acyclic, and a frame declared twice is rejected; the S1-03 check that every used edge has a transform source builds on this.
- Unit and frame diagnostics are registered in the AI-01 registry in the range the registry assigns to the core checker; suggested fixes cover the unambiguous scale conversions (for example insert `to_m(x)` for a millimetre value).

## Semantics

A quantity's runtime representation is its numeric type `T`; the unit and frame are erased before MIR, so MIR, native, CUDA and Metal outputs are unchanged relative to the unit-free program.
Scale conversions are explicit functions with exact rational factors applied under the program's numeric policy; no conversion is implicit.
Addition and subtraction require identical unit and frame; multiplication and division combine dimensions; comparison requires identical unit.
Angles in `rad` and `deg` are distinct scales of the angle dimension.
R1-01 lands before M4-01 in the v0.1 order, so `T` ranges over the numeric types implemented at that point (the integer types); the kinematics corpus uses fixed-point scales (for example `Quantity<i32, um>`, `Quantity<i32, urad>`), and `f32` quantities follow automatically once M4-01 lands because units erase before MIR.
Compile-time constants in system declarations (S1-01) are exact decimal literals with a unit, evaluated as exact rationals and serialized as canonical decimal strings in `invar-system-v1` (design note section 1, "Numbers before M4-01"); R1-01 provides their unit typing.

## Acceptance tests

Add the integration test binary `issue_r1_01_units_frames` (`tests/issue_r1_01_units_frames.rs`).
Run `cargo test --locked issue_r1_01_units_frames` in Debug and Release and require a nonzero test count.
Positive cases:
- `mm + m`, `rad + deg`, and adding a vector in frame A to a vector in frame B without a transform are rejected with AI-01 diagnostics that carry a suggested fix where one is unambiguous, and the AI-01 fix-application test proves the fixed program checks;
- generated code for a unit-typed program is byte-identical to the unit-free equivalent on every engine that emits text (MIR dump and LLVM IR; CUDA and MSL text where the program is device-eligible);
- reference and native engines agree on a corpus of at least 20 kinematics-style programs (planar two-link forward kinematics, transform chains, velocity from displacement and duration);
- dimension algebra fixtures: force times length is torque-dimensioned; length divided by duration is velocity; `1 / s` equals `Hz`;
- frame-tree fixtures: a three-level tree with composed transforms checks.

## Negative cases

Each is rejected with a registered code: mismatched units in `+`, `-` or comparison; composing `Transform<A, B>` with `Transform<C, D>` where `B != C`; a frame whose parent is undeclared; a parent cycle; two roots; a frame declared twice; an implicit scale conversion; a unit used as a value.

## Done criteria

Grammar, spec fixtures, reference, checker, MIR erasure and native agreement are covered; the test binary passes in Debug and Release with a nonzero count; every new diagnostic is registered under AI-01.

## Out of scope

No symbolic algebra, no runtime unit tags, no automatic frame-graph search, no non-SI unit systems beyond explicit scale units, no affine temperature units.
