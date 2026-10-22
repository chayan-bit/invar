# R1-03 - Batched small fixed-size linear algebra

Release: v0.2 (milestone R1); planning contract, to be expanded into a full contract before implementation.

Dependencies: R1-01.

## Goal

Run thousands of small fixed-size linear algebra problems at library speed on CPU, CUDA and Metal.

## Motivation

Invar's robotics direction is an OEM supplier model: each new manufacturer's hardware should be cheap to support, because robot-specific facts become typed compiler inputs instead of per-customer hand-written code.

## Scope

- Fixed-size 3x3, 4x4 and 6x6 matrices, quaternions and small dense solves (LU, Cholesky, QR) as first-class values.
- Batched execution across thousands of instances with generated schedules on CPU, CUDA and Metal.
- Numeric policy follows the v0.1 strict and relaxed contracts.

## Acceptance sketch

- Gated through the M5-01 harness against cuBLAS batched and MAGMA batched on CUDA, Eigen fixed-size on CPU, and a reviewed pinned Metal kernel or MPS equivalent.
- Results agree with the reference evaluator under the declared numeric policy on every engine.
- Integration test binary `issue_r1_03_batched_linalg`.

## Out of scope

No large dense or sparse linear algebra, no eigen-decompositions beyond small symmetric cases.
