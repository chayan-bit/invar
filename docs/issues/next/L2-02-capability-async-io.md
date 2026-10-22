# L2-02 — Capability-based async IO and hosted acceptance

Problem/result: add a narrow hosted asynchronous IO profile with explicit capabilities, cancellation, and external-effect acceptance.

Dependencies: L2-01.

Contract and implementation: define capability values, permitted filesystem/network-like operations, effect declarations, buffering, deadlines, cancellation acknowledgment, close/drain order, and environment responses. Update grammar/spec, checker, evaluator/oracle, MIR/native serial semantics, and host adapters. Capabilities constrain Invar access but are not an OS sandbox. Hosted tests use deterministic local fixtures or approved services; no implicit network is required for ordinary checks.

Positive cases: bounded fixture reads/writes, capability denial, EOF, partial IO, cancellation after acknowledgment, timeout reporting, and task-scope cleanup agree with the declared trace contract. Negative cases: missing capability, path/length overflow, use after close, cancellation before safe release, unbounded buffering, external failure, unauthorized host access, and nondeterministic ordering reject or return typed environmental error.

Acceptance interface and evidence: add `issue_l2_02_capability_async_io`; run `cargo test --locked issue_l2_02_capability_async_io` and require a nonzero test count (zero-test success fails) for offline fixtures, then the separately documented hosted acceptance command only when its environment is available. Record capability declarations, fixture/service versions, effect traces, cleanup/lifetime evidence, and unavailable-hosted checks.

Exclusions/stop: no general web framework, unconstrained sockets, ambient filesystem access, distributed runtime, or claim of sandbox security. Stop when the bounded capability profile is reproducible offline and hosted gaps remain explicit.

Bound capabilities, bytes, concurrent operations, deadlines, buffers, diagnostics, and hosted resource cost. A denied or unavailable operation must never silently become a successful empty result.

Hosted evidence is additive to deterministic offline fixtures and cannot replace them.

## System integration impact (from S1)

Added 2026-09-26 by the integration plan (`docs/system-integration-v1.md`); existing content above is unchanged except where this section says so.

- v0.1 component execution does not depend on this issue: generated targets run on ROS 2 and ros2_control (S1-04), so L2 concurrency and IO are not prerequisites of the integration track.
