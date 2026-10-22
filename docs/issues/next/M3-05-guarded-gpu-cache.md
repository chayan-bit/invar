# M3-05 — Guarded GPU dispatch, cache, and failure integration

Problem/result: integrate guarded GPU dispatch with admission records, CPU fallback, bounded caching, and explicit failure semantics.

Dependencies: M3-04.

Contract and implementation: define a versioned cache key from source identity, typed contract, rule/toolchain/device/layout policy, generated artifact hash, and relevant inputs. Validate cache entries before use; discard stale/corrupt entries. Guards are total and nonobservable, and false/unknown selects CPU. Use the exact Prepared/Submitted/Completed/FailedQuiescent/InFlightUnknown state machine from M3-01. Device completion and borrowed storage remain live until acknowledged; partial visible writes cannot be replayed without an explicit contract.

Positive cases: valid cache hit/miss, guard true/false, device unavailable, compilation failure, runtime launch failure, cancellation, and CPU fallback preserve required observations. Negative cases: forged/stale key, changed source/rule/toolchain, corrupted artifact, unsafe cache metadata, partial mutation, and unknown admission fail closed without bypassing verification.

Acceptance interface and evidence: add `issue_m3_05_guarded_gpu_integration`; run `cargo test --locked issue_m3_05_guarded_gpu_integration` with deterministic mock-device tests and the recorded hardware matrix when available. Record cache key fields, guard decisions, artifact validation, failure prefixes, completion ownership, and CPU fallback outputs.

Exclusions/stop: no public plugin marketplace, multi-device policy, arbitrary kernel admission, or production-readiness claim. Stop when guarded dispatch is auditable, bounded, and always retains a legal CPU path.

Bound cache size, artifact validation time, guard cost, in-flight completions, and retained device memory. Cache corruption is recoverable by discard and rebuild, while repeated validation failure is reported. Never let a cache hit bypass current admission, verifier, or completion checks.

Cache evidence must include both cold and warm paths, with the same source and policy checks visible in each path.

## Concrete integration requirements

Implement cuda/auto/explain interfaces from docs/implementation-plan.md. Forced cuda reports an ineligible/unavailable error rather than silently choosing CPU. Auto may select CPU before launch; retry after device failure requires fresh unpublished outputs and confirmed quiescence/retained resources. A hung device never licenses freeing or reusing in-flight buffers. Demonstrate real cache cold/warm paths and injected errors alongside actual device execution; hardware acceptance cannot be replaced by mocks.

Follow the shared acceptance interface in `docs/implementation-plan.md`: list and verify a nonzero test selection, run the issue suite in Debug and Release, and retain full milestone Cargo gates. Commands are required deliverables, not existing functionality in the planning checkout.

## Cache trust contract

Treat persistent cache bytes as untrusted. Version 1 persists only bounded schedule-choice hints and descriptive source/manifest data. Reconstruct the current region from trusted source, rerun admission/physical verification, and regenerate CUDA from the current verified device IR. Recompile with pinned NVRTC on each cold process; reuse compiled modules only in memory within the current process, bound to live context and validated current-plan identity. Never load PTX/cubin or accept executable CUDA text from disk/imported cache, even when a self-stored hash matches. A hash is identity, not authentication or semantic verification. Test replaced artifact plus replaced digest, valid-looking forged metadata and an injected CUDA body: none may reach Driver module load. Persistent binary caching is deferred until a separately trusted authentication or regeneration mechanism is specified. InFlightUnknown retains/quarantines resources and prohibits replay until quiescence is confirmed.
