# L2-01 — Structured tasks, channels, and cancellation semantics

Problem/result: add structured concurrency with explicit task lifetime, bounded channels, and cancellation acknowledgment while preserving source observations and ownership.

Dependencies: L1-05, M2-05.

Contract and implementation: specify task scopes, spawn/join order, channel capacity and close behavior, send/receive failure, cancellation states, draining, blocking, suspension, and every exit edge. Update grammar/spec, checker, evaluator/oracle, typed MIR, native serial semantics, and diagnostics before any runtime implementation. A task cannot retain a borrow past scope/cancellation acknowledgment; visible writes before failure remain ordered. Unknown progress is not success.

Positive cases: bounded child completion, channel send/receive/close, cancellation acknowledgment, nested scopes, normal return, and task-failure/cancellation and normal-scope cleanup under the new L2 trace contract agree across reference and native serial paths. Negative cases: leaked task, unacknowledged cancellation, closed/full channel misuse, deadlock-prone unbounded wait, borrowed storage release, task escape, and unsupported scheduling reject or return typed outcome.

Acceptance interface and evidence: add `issue_l2_01_structured_tasks`; run `cargo test --locked issue_l2_01_structured_tasks` and require a nonzero test count (zero-test success fails). Record trace semantics, scheduler bounds, ownership/lifetime events, oracle/native outcomes, and deterministic failure-prefix fixtures.

Exclusions/stop: no async IO, work-stealing runtime, unbounded channels, implicit cancellation, or source concurrency model inferred from Rust implementation details. Stop when one bounded task/channel subset is fully specified and tested.

Bound task count, channel capacity, message size, wait time, and trace output. A timeout is an incomplete/tool outcome, not proof that a source program diverges.

The checker must expose the ownership reason for each rejected task escape or cancellation path.

Compile-time source errors never run cleanup; bootstrap terminal traps retain their no-language-unwinding meaning unless a separately explicit semantic revision is made. Reference traces are the acceptance oracle for ordering and cleanup; scheduler timing is not.

## System integration impact (from S1)

Added 2026-09-26 by the integration plan (`docs/system-integration-v1.md`); existing content above is unchanged except where this section says so.

- v0.1 component execution does not depend on this issue: generated targets run on ROS 2 and ros2_control (S1-04), so L2 concurrency and IO are not prerequisites of the integration track.
