# R2-02 - Certification evidence generation

Release: v0.3 (milestone R2); planning contract, to be expanded into a full contract before implementation.

Dependencies: R2-01.

## Goal

Generate evidence packs that support functional-safety certification of Invar-built software.

## Motivation

Invar's robotics direction is an OEM supplier model: each new manufacturer's hardware should be cheap to support, because robot-specific facts become typed compiler inputs instead of per-customer hand-written code.

## Scope

- Requirements-to-code-to-test traceability from annotated source and tests.
- Reference-equivalence results presented as translation-validation evidence.
- Evidence packs targeting IEC 61508, ISO 13849 and ISO 10218, stating what is and is not evidenced.

## Acceptance sketch

- A sample project produces a complete, reproducible evidence pack with traceability matrices and equivalence records.
- Missing traces or failed equivalence make the pack generation fail.
- Integration test binary `issue_r2_02_certification_evidence`.

## Out of scope

No claim of certification, no tool qualification of the Rust toolchain.

## System integration impact (from S1)

Added 2026-09-26 by the integration plan (`docs/system-integration-v1.md`); existing content above is unchanged except where this section says so.

- Evidence packs include S1-03 graph-check results, S1-10 timing-analysis results, S1-02 contract-test records and S1-08 license results, each with its proven-versus-measured statement.
