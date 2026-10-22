# R2-01 - Record and replay with source-level explanations

Release: v0.3 (milestone R2); planning contract, to be expanded into a full contract before implementation.

Dependencies: R1-06.

## Goal

Record optimized executions and replay them with source-level explanations.

## Motivation

Invar's robotics direction is an OEM supplier model: each new manufacturer's hardware should be cheap to support, because robot-specific facts become typed compiler inputs instead of per-customer hand-written code.

## Scope

- Bounded recording of inputs, plan decisions and nondeterministic sources for a run.
- Deterministic replay on the reference evaluator and on the recorded engine.
- Source-level explanations that map optimized execution (fused regions, schedules, device choices) back to source spans.

## Acceptance sketch

- Replay reproduces recorded outputs bit for bit under the declared numeric policy.
- Explanations reference valid spans and AI-01 codes for every optimized region.
- Integration test binary `issue_r2_01_record_replay`.

## Out of scope

No full-system recording, no time-travel debugging UI.

## System integration impact (from S1)

Added 2026-09-26 by the integration plan (`docs/system-integration-v1.md`); existing content above is unchanged except where this section says so.

- Record and replay cover the whole component graph in MCAP, deterministic under the declared connection policies, not only single kernels.
- MCAP schemas are generated from the `invar-system-v1` IR, so recorded topics and port types cannot drift.
