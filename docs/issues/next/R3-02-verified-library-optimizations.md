# R3-02 - Library-provided verified optimizations

Release: v0.4 (milestone R3); planning contract, to be expanded into a full contract before implementation.

Dependencies: R3-01.

## Goal

Let libraries ship optimizations the compiler can trust without trusting the library author.

## Motivation

Invar's robotics direction is an OEM supplier model: each new manufacturer's hardware should be cheap to support, because robot-specific facts become typed compiler inputs instead of per-customer hand-written code.

## Scope

- An optimization package: operation spec, implementation, preconditions, machine-checkable equivalence evidence and separate performance evidence.
- A small trusted checker that validates the equivalence evidence before the compiler uses the optimization.

## Acceptance sketch

- Valid packages are applied and match the reference evaluator.
- Packages with broken evidence, unmet preconditions or missing performance evidence are rejected.
- Integration test binary `issue_r3_02_verified_optimizations`.

## Out of scope

No unverified rewrite plugins, no general theorem prover.
