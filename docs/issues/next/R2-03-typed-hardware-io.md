# R2-03 - Typed hardware I/O specifications

Release: v0.3 (milestone R2); planning contract, to be expanded into a full contract before implementation.

Dependencies: R1-06, S1-02.

## Goal

Describe hardware I/O in typed specifications and generate parsers and timing-checked I/O.

## Motivation

Invar's robotics direction is an OEM supplier model: each new manufacturer's hardware should be cheap to support, because robot-specific facts become typed compiler inputs instead of per-customer hand-written code.

## Scope

- Typed specifications for EtherCAT, CAN and sensor packet formats.
- Generated parsers and serializers with bounds and validity checks.
- Capability-typed I/O so that only code holding a capability can touch a device, with timing checks against the R1-06 profile.

## Acceptance sketch

- Generated parsers round-trip every valid packet and reject malformed packets with typed errors, fuzzed with a maintained fuzzer.
- I/O without a capability or outside its timing budget is rejected at compile time.
- Integration test binary `issue_r2_03_hardware_io`.

## Out of scope

No fieldbus master implementation, no kernel drivers.

## System integration impact (from S1)

Added 2026-09-26 by the integration plan (`docs/system-integration-v1.md`); existing content above is unchanged except where this section says so.

- Hardware adapters are components with S1-02 foreign manifests (for example IgH EtherCAT, open62541, libmodbus), so their licenses, isolation and contracts go through S1-03 and S1-02 like any other component.
