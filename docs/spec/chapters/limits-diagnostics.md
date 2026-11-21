## 12. Limits

[N12.1.1] The reference limits are call depth 64, array length 2^24 elements, 2^28 live elements and default fuel 1,000,000, and they are tool limits and not source semantics. `src/array_limits.rs::MAX_LIVE_ELEMENTS`
[N12.1.2] The checker limits are 32 modules, 256 imports, 128 records of 1 to 16 fields, 128 variants of 1 to 16 arms, 128 frames and 128 lowered functions. `src/check/modules.rs::MAX_RECORDS`
[N12.1.3] The MIR limits are 128 functions, 4,096 blocks and 8,192 locals per function, and 100,000 instructions. `src/mir.rs::MAX_INSTRUCTIONS`
[N12.1.4] Limit exhaustion is a typed tool or resource outcome and never a source trap or a proof of divergence. `docs/native-toolchain-v1.md`
[N12.1.5] The implementation uses checked or explicit wrapping arithmetic and typed outcomes, so host panics, overflow profiles and Rust drop order never become Invar semantics. `docs/rust-engineering.md`

## 13. Diagnostics and explain schemas

### 13.1 Diagnostics

[N13.1.1] Diagnostics use the schema `invar-diagnostics-v3` with `schema`, `schema_version`, `status` and `diagnostics`, and a document carries at most 64 diagnostics. `schema/invar-diagnostics-v3.schema.json`
[N13.1.2] A diagnostic has a stable `code`, a `severity`, a message, a primary span with byte offsets and one-based line and column, secondary spans, notes and suggested fixes. `tests/diagnostics/machine_diagnostics/contracts.rs::machine_diagnostics_positions_and_fix`
[N13.1.3] Codes, span semantics and fix edits are the contract and messages may change wording. `schema/diagnostic-codes-v1.json`
[N13.1.4] The status is one of `ok`, `source_error`, `tool_error`, `trap` and `evaluation_limit`, with exit codes 0, 1, 2, 3 and 4. `schema/invar-diagnostics-v3.schema.json`

### 13.2 Registry

[N13.2.1] Every emitted code is in a committed registry, `schema/diagnostic-codes-v1.json` plus the extension files, and a test scans the source in both directions. `tests/diagnostics/machine_diagnostics/registry.rs::machine_diagnostics_registry_covers_emitted_literals`
[N13.2.2] Each code declares `reachability` of `source`, `invariant` or `external`, and checked source never emits an `invariant` code. `tests/diagnostics/machine_diagnostics/documents.rs::machine_diagnostics_corpus_has_no_invariant_codes`
[N13.2.3] Only `source` codes carry fixes. `tests/diagnostics/machine_diagnostics/registry.rs::machine_diagnostics_registry_covers_emitted_literals`
[N13.2.4] The `S1xxx` ranges are 1000 system dialect, 1100 to 1399 graph checks, 1400 foreign manifests, 1500 ROS 2, 1600 robot and cell, 1700 vocabulary, 1800 acceptance and 1900 reserved. `schema/diagnostic-codes-v1.json`

### 13.3 Explain

[N13.3.1] `invar explain PATH --format json` emits `invar-explain-v6` with `source_identity` and `regions`, and success is exit 0 even when every region is rejected. `docs/explain.md`
[N13.3.2] Each region has `function`, `span`, `status`, `first_rejected_obligation`, `rule_revision`, `numeric_policy`, `chosen_plan`, `fallback_plan`, `runtime_guard`, `facts`, `execution` and ordered `obligations`. `schema/explain-v6.schema.json`
[N13.3.3] Obligation evidence is `checked_invariant`, `trusted_rule`, `runtime_guard` or `unknown`, and unknown evidence rejects admission. `docs/explain.md`
[N13.3.4] `explain --engine cuda|metal|auto` adds `device_execution` described by `invar-explain-device-v5`. `schema/invar-explain-device-v5.schema.json`
[N13.3.5] Persisted admission documents are descriptive, and validation re-analyses current MIR and source bytes. `docs/explain.md`
[N13.3.6] `source_identity` is `fnv1a64:` over the source bytes and establishes identity and not authenticity. `docs/explain.md`

```invar outcome=value:0
fn main() -> i32 { return 0; }
```

## 14. Python binding surface

[N14.1.1] The extension module `invar_python` exposes `compile_source`, `validate_array`, `validate_buffers`, `capabilities`, the class `CheckedModule` and exceptions rooted at `InvarError`. `crates/invar-python/src/lib.rs::invar_python`
[N14.1.2] `compile_source(source, path, engine, exports)` checks a complete source, and `native-serial` needs explicit exports and a native-enabled core build. `crates/invar-python/src/lib.rs::compile_source`
[N14.1.3] Selected exports have at most eight scalar parameters of `i32` or `u32`, and calls validate names, types, arity and fuel before any source work. `docs/scalar-exports-v1.md`
[N14.1.4] `capabilities()` reports what this build supports, and it is the authority for the binding surface. `crates/invar-python/src/lib.rs::capabilities`
[N14.1.5] The reference array entry copies, the GIL is not released, and external array owners, scheduled host arrays and `take` buffers are unavailable. `crates/invar-python/src/lib.rs::external_array_owners`
[N14.1.6] Exports accept `read` and `mut` `Array<i32>`, `Array<u32>`, `Array<f32>`, `Array<f64>` and `Array<Bool>` parameters, which `native-serial` passes zero-copy at the NumPy address with rollback on failure and the reference path copies. `crates/invar-python/src/lib.rs::array_export_paths`
[N14.1.8] `CheckedModule.run_main` runs `main` on any engine of `reference`, `native-serial`, `cpu-parallel`, `auto`, `cuda` and `metal`, and `receipt()` returns the receipt of the last successful `run_main`. `crates/invar-python/src/module.rs::run_main`
[N14.1.7] The private wheel qualification is Linux x86_64 with glibc 2.41 and CPython 3.12 only. `crates/invar-python/README.md`

## 15. Example library and coverage

[N15.1.1] Every Invar source file under `examples/` is listed exactly once in `examples/index.json` with its path, intent, family, tags, spec section and expected result. `tests/spec_examples.rs::validate_index`
[N15.1.2] Each family is either covered by at least one example or listed under `pending_families` with a reason. `tests/spec_examples.rs::validate_index`
[N15.1.3] Each runnable example runs on the reference evaluator and on every engine it declares, check and system examples run their declared check commands, and `native-serial` runs only in a build with the `native` feature. `tests/spec_examples.rs::run_example`
[N15.1.4] System examples carry their expected diagnostics and their IR or generated-file hashes, checked twice for byte identity. `tests/spec_examples.rs::tree_hash`

