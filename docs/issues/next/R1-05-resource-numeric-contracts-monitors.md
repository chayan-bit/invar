# R1-05 - Resource and numeric contracts plus generated runtime safety monitors

Release: v0.2 (milestone R1); planning contract, to be expanded into a full contract before implementation.

Dependencies: R1-04.

## Goal

Let a program declare resource, numeric and safety envelopes, and prove or monitor them.

## Motivation

Invar's robotics direction is an OEM supplier model: each new manufacturer's hardware should be cheap to support, because robot-specific facts become typed compiler inputs instead of per-customer hand-written code.

## Scope

- Contracts for no allocation, bounded temporary memory, exact versus bounded-error arithmetic, joint, velocity and torque envelopes, and bounds on learned-policy outputs.
- Static proof where the compiler can establish the contract; otherwise a generated runtime monitor with a declared fallback controller.
- Latency remains a preference, never a guarantee, in this issue.

## Acceptance sketch

- Each contract kind has positive fixtures proven statically and fixtures that need a generated monitor.
- Injected violations trigger the monitor and switch to the declared fallback controller within one control period, with a typed event.
- Monitors add bounded, reported overhead.
- Integration test binary `issue_r1_05_contracts_monitors`.

## Out of scope

No WCET guarantees (R1-06), no certification claims (R2-02).

## System integration impact (from S1)

Added 2026-09-26 by the integration plan (`docs/system-integration-v1.md`); existing content above is unchanged except where this section says so.

- Runtime monitors are generated from component contracts in the system graph (port ranges, rates, latency budgets, frame-validity and resource limits declared in S1-01 and checked by S1-03) as well as from kernel contracts.
- Monitor violations are reported through `invar-diagnostics-v1` codes and recorded in MCAP (R2-01).
