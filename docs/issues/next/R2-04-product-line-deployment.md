# R2-04 - Product-line configuration and protected deployment

Release: v0.3 (milestone R2); planning contract, to be expanded into a full contract before implementation.

Dependencies: R2-03, S1-01.

## Goal

Ship per-customer variants from one source safely.

## Motivation

Invar's robotics direction is an OEM supplier model: each new manufacturer's hardware should be cheap to support, because robot-specific facts become typed compiler inputs instead of per-customer hand-written code.

## Scope

- Compile-time checked per-customer variants (robot model, I/O, contracts) from one source tree.
- Specialized-binary-only shipping, so customers receive compiled variants rather than source.
- Over-the-air updates that prove compatibility (ABI, contracts, robot model) before install.

## Acceptance sketch

- Every declared variant builds and passes its contracts; an invalid variant is rejected at compile time.
- An incompatible update is refused before install with a typed reason.
- Integration test binary `issue_r2_04_product_line`.

## Out of scope

No update server, no licensing or DRM system.

## System integration impact (from S1)

Added 2026-09-26 by the integration plan (`docs/system-integration-v1.md`); existing content above is unchanged except where this section says so.

- Product-line configuration adds variants over the S1-01 `product` declaration and does not redefine it (design note section 7).
