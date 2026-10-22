# R1-06 - Real-time execution profile

Release: v0.2 (milestone R1); planning contract, to be expanded into a full contract before implementation.

Dependencies: R1-05.

## Goal

Provide a real-time profile for control-loop regions, separated from best-effort code.

## Motivation

Invar's robotics direction is an OEM supplier model: each new manufacturer's hardware should be cheap to support, because robot-specific facts become typed compiler inputs instead of per-customer hand-written code.

## Scope

- A profile that forbids allocation and unbounded loops and bounds memory in marked control-loop regions.
- A static worst-case execution-time bound for those regions on simple CPUs (in-order cores with documented timing), using or porting a proven WCET analysis approach selected by a survey.
- A clear boundary between real-time regions and best-effort code, checked by the compiler.

## Acceptance sketch

- Profile violations are rejected with AI-01 diagnostics.
- Measured execution times on the chosen reference CPU never exceed the static bound over a bounded stress run.
- Best-effort code calling into real-time regions is allowed; the reverse is rejected.
- Integration test binary `issue_r1_06_realtime_profile`.

## Out of scope

No WCET for GPUs or out-of-order desktop CPUs, no RTOS implementation.

## System integration impact (from S1)

Added 2026-09-26 by the integration plan (`docs/system-integration-v1.md`); existing content above is unchanged except where this section says so.

- R1-06 supplies measured WCET per component and kernel; S1-03 reports then replace declared budgets with measured values where R1-06 evidence exists and state which values remain declared.
- Until R1-06 evidence exists, every generated report says deadlines are declared, not measured (design note section 6).
