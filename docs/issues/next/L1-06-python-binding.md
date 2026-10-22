# L1-06 - Minimal Python binding over the C ABI

Problem/result: let existing Python and NumPy software call compiled Invar kernels directly, so that Invar can be embedded where numeric work already lives, without hand-rolled FFI and without copying arrays.

Dependencies: L1-04.

Release: v0.1 (milestone L1).

## Scope

Add a separate crate `crates/invar-python` built on PyO3 and the `numpy` crate (reuse maintained crates; no hand-rolled FFI), packaged with maturin inside a Nix dev shell.
The module compiles Invar source, lists exported functions with their signatures, and calls exported functions through the L1-04 C ABI.
NumPy arrays pass zero-copy through the buffer protocol; dtype, shape, contiguity, alignment, writability and ownership are validated at the boundary before any call.
Compile, check and trap errors map to Python exception classes carrying the AI-01 diagnostic JSON (`invar-diagnostics-v1`) as a structured attribute.
The engine is selectable (`reference`, `native-serial`, `cpu-parallel`, `auto`, and `cuda` or `metal` when built with that feature), and explain output is available as the AI-01 explain document.

## Semantics

A call either returns results or raises; it never returns partial output.
An input that fails validation raises before any Invar code runs.
Output arrays are either caller-provided (validated like inputs) or allocated by the binding and owned by Python; ownership transfer follows the L1-04 rules exactly.
The Global Interpreter Lock is released only around calls whose inputs are validated and pinned for the call's duration.
`unsafe` code appears only inside `crates/invar-python`, each block with a SAFETY comment and each invariant with a test; the core crate keeps `#![forbid(unsafe_code)]`.

## Acceptance tests

Add the integration test binary `issue_l1_06_python_binding` and a pytest suite run from the Nix dev shell (for example `nix develop .#python --command pytest crates/invar-python/tests`).
Run `cargo test --locked issue_l1_06_python_binding` in Debug and Release and require a nonzero test count; the pytest run must also report a nonzero test count.
Positive cases:
- results equal the reference evaluator on the full corpus subset expressible through the ABI (every program whose signature maps to ABI scalars, arrays and byte strings), with the subset count reported;
- zero-copy is proven by a test that observes the same buffer address on both sides;
- `u32`, `i32` and `f32` arrays, empty arrays, and large bounded arrays round-trip;
- every exception class carries schema-valid AI-01 JSON with the expected reason code.

## Negative cases

Wrong dtype, wrong rank or shape, non-contiguous or misaligned arrays, read-only output buffers, aliasing inputs and outputs where the signature forbids it, arrays freed or resized during a call, oversized inputs, unknown function names, and wrong argument counts each raise a typed exception and never crash the interpreter.

## Done criteria

The crate builds reproducibly in the Nix dev shell on macOS (aarch64) and Linux (x86_64); the Rust and pytest suites pass with nonzero counts; the reference-equivalence subset count is recorded; the survey of PyO3, `numpy`, maturin and alternatives is reported with license, maintenance and unsafe surface.

## Out of scope

No PyPI publishing, no wheels for other platforms, no async API, no Python callbacks into Invar, no pandas or other framework integration, and no C ABI changes beyond what L1-04 defines.
