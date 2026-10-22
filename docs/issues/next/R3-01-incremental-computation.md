# R3-01 - Incremental computation

Release: v0.4 (milestone R3); planning contract, to be expanded into a full contract before implementation.

Dependencies: R1-05.

## Goal

Update results incrementally when inputs change a little, with proof that the update equals recomputation.

## Motivation

Invar's robotics direction is an OEM supplier model: each new manufacturer's hardware should be cheap to support, because robot-specific facts become typed compiler inputs instead of per-customer hand-written code.

## Scope

- `update(previous, retained, changes) == recompute(updated)` proven per supported operation.
- A cost model that chooses repair versus recompute.
- Motivating workloads: occupancy grids and SLAM map updates.

## Acceptance sketch

- Per-operation equivalence proofs or exhaustive bounded checks, plus randomized tests against recomputation.
- Repair beats recompute on small change sets and the cost model picks correctly on the benchmark corpus.
- Integration test binary `issue_r3_01_incremental`.

## Out of scope

No general self-adjusting computation over arbitrary programs.
