# M3-01 — Rust CUDA driver runtime ownership and completion

Problem/result: introduce one pinned CUDA driver boundary that owns device allocations, streams, events, and completion lifetimes safely.

Dependencies: M2-05.

Contract and implementation: choose and record one supported CUDA driver/toolkit/target matrix from backend research. Wrap foreign handles behind safe Rust types and explicit typed errors; document thread affinity, synchronization, allocation, cancellation, and completion ownership. A borrow remains live until device completion, not merely submission. Preserve CPU fallback and source-ordered failure prefix; no unsafe code leaks into Invar core.

Positive cases: bounded allocation/copy/launch/completion and cleanup paths release resources exactly once; a real host/device byte-copy round trip agrees with the original bytes. Exercise launch outcomes through the mock boundary here; M3-02 owns the first generated-kernel launch so this runtime issue does not depend on its successor. Negative cases: unavailable device, invalid handle, launch failure, timeout, cancellation before completion, stale borrow, OOM, and driver error return typed outcomes; fallback is allowed only before submission or after confirmed quiescence.

Acceptance interface and evidence: add `issue_m3_01_cuda_runtime`; run `cargo test --locked issue_m3_01_cuda_runtime` with a fake/mock boundary for default CI and the pinned hardware matrix when available. Record driver/toolkit versions, device, ownership traces, unavailable hardware, and no-GPU local results.

Exclusions/stop: no kernel generation, device IR, multi-device scheduling, or public deployment. Stop when the narrow runtime contract is reviewable and CPU fallback works without a CUDA host.

Bound allocation size, in-flight operations, polling duration, and diagnostic payloads; bounded polling does not imply bounded physical device completion. Keep mock tests deterministic and make hardware tests additive. A driver API's unsafe surface stays inside the reviewed boundary; safe Rust ownership does not by itself prove device completion.

Record whether each check was observed locally, on pinned hardware, or unavailable; do not merge those evidence classes.

## Concrete integration requirements

Introduce opt-in feature cuda and explicit GPU qualification commands from docs/implementation-plan.md. Start with one context/stream, flat fixed-width ABI and fresh outputs. Isolate unsafe FFI in a reviewed boundary with context-bound resources; event completion, not launch success, ends the loan. Add a real allocate/copy/complete/free smoke test. Mock tests are necessary but do not complete this issue without the required authorized device run.

Follow the shared acceptance interface in `docs/implementation-plan.md`: list and verify a nonzero test selection, run the issue suite in Debug and Release, and retain full milestone Cargo gates. Commands are required deliverables, not existing functionality in the planning checkout.

## Required incomplete-completion state

Implement `Prepared -> Submitted -> Completed | FailedQuiescent | InFlightUnknown`. Only Completed may publish. Prepared or FailedQuiescent with fresh unpublished outputs may select fallback; both quiescent terminal states may release resources. A timeout/cancel request with no confirmed quiescence yields an owning pending token and transitions to InFlightUnknown. Its drop transfers resources to a bounded runtime quarantine, never frees/reuses them or retries computation. Retain context, stream, event, device allocations and owned pinned staging; initial transfers must not depend on a caller buffer whose Rust lifetime can end. Poison further submissions on that context until completion or a documented driver teardown guarantees quiescence. Quarantine reaching its budget disables GPU dispatch. The synchronous CLI reports an environment/pending failure rather than pretending cancellation completed. Process termination is an environment boundary, not proof of in-process safe release. Inject timeout then drop the token: assert no free, publication or CPU replay occurs; later confirmed completion releases exactly once.
