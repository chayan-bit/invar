# R1-04 - Automatic differentiation (forward and reverse mode)

Release: v0.2 (milestone R1); planning contract, to be expanded into a full contract before implementation.

Dependencies: R1-03.

## Goal

Differentiate Invar programs in forward and reverse mode on every engine.

## Motivation

Invar's robotics direction is an OEM supplier model: each new manufacturer's hardware should be cheap to support, because robot-specific facts become typed compiler inputs instead of per-customer hand-written code.

## Scope

- Forward mode (dual numbers) and reverse mode (tape-free where the structure allows) over the numeric subset, including the R1-03 batched linear algebra.
- Derivatives of kinematics and dynamics from R1-02 models.
- Explicit handling of non-differentiable points with typed diagnostics.

## Acceptance sketch

- Derivatives match finite differences under a declared tolerance and match the reference evaluator's derivative on CPU, CUDA and Metal.
- Forward and reverse mode agree with each other on every corpus program.
- Integration test binary `issue_r1_04_autodiff`.

## Out of scope

No higher-order derivatives beyond Hessian-vector products, no differentiable simulation.
