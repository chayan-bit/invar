# Documentation index

This page says what each document in `docs/` is for and how much to trust it.
There are three kinds: current specifications (checked against the code), design notes (the reasoning behind a subsystem), and process documents (how work is validated).
There is no archive directory: documents that were superseded are deleted, and their history is in git.

## Start here

| Document | Read it to |
| --- | --- |
| [README](../README.md) | See what Invar is and build it |
| [Glossary](glossary.md) | Look up a project word: admission, oracle, family, receipt, cr-v1 and more |
| [Architecture](architecture.md) | Follow one program through the code, then read the design direction and trust boundaries |
| [Examples](../examples/README.md) | Run verified programs |

## Current specifications

These are normative.
Where one disagrees with the code, one of the two is wrong and the disagreement is a bug.

| Document | Specifies |
| --- | --- |
| [v0.1 language specification](spec/invar-v0.1.md) | The language and the tool surface; every code block runs and every normative statement cites a file, both checked by `tests/issue_ai_02_spec_examples.rs` |
| [Grammar](../spec/grammar.ebnf) | The complete syntax |
| [Placement profile v1](../spec/placement-v1.md) | `--engine auto` placement receipts and calibration profiles |
| [Explain JSON](explain.md) | The `invar-explain-v5` admission receipt and device execution reports |
| [Numeric contracts](numeric.md) | `strict-f32-v1`, fold tiers and storage types |
| [Shape and layout contract](layout-v1.md) | Layout contract version 1 for the M4 families |
| [Typed MIR v1](mir-v1.md) | The mid-level IR |
| [Typed interaction graph v1](typed-interaction-v1.md) | The typed net used for checked rewrites and extraction |
| [GNET core contract](../src/general_net/CONTRACT.md) | The general-net boundary, its frozen interfaces and its compiled ABIs |

## Subsystem contracts

Each of these documents one implemented subsystem in enough detail to change it safely.
They describe the code on `main`, and their status lines say when a contract is frozen.

| Document | Subsystem |
| --- | --- |
| [Native toolchain and trap ABI](native-toolchain-v1.md) | LLVM CPU backend |
| [CUDA target contract](cuda-target-v1.md) | NVRTC, compute capability and limits |
| [Bounded schedule selection](bounded-schedule-dispatch.md) | Device schedule choice and residency |
| [Indexed device paths](indexed-device-v1.md) | Gather, scatter, histogram and frontier on devices |
| [Host array arguments (HA-1)](host-array-arguments-ha1.md) | Passing arrays from Rust and Python hosts (frozen) |
| [Scalar exports](scalar-exports-v1.md) | `invar::exports` |
| [Ordered affine-pair reduction](pair-reduction-v1.md) | A noncommutative lawful reduction |
| [Net normal form](net-normal-form-v01.md) | Regions, laws and certificates for general nets |
| [PNF family admission](pnf-family-admission-v01.md) | Family admission from the normal form |
| [Path-sum lowering](path-sum-v01.md) | Recursion lowered to one kernel |
| [Task region](gnet-task-v01.md) | Residual fork-join lowering |
| [Law collective](law-collective-v01.md) | Certificate-bound ordered collectives |
| [Loaded numeric owned host](numeric-owned-host-v1.md) | The ABI11 CPU numeric host |
| [Planner wire](planner-wire-v1.md) | The robot planner transport |
| [Debugging](debugging.md) | Source-level debugging of native programs |

## Design notes

These explain why the code is shaped as it is.
They are direction, not promises, and they say so in their status lines.

| Document | Covers |
| --- | --- |
| [Architecture](architecture.md) | A code tour, then: one meaning with many admitted implementations; evidence and trust |
| [Parallelism foundations](parallelism-foundation.md) | Why interaction nets, and what they must and need not do |
| [System integration](system-integration-v1.md) | Invar as the robot system description (decided and implemented) |

## Process

| Document | Covers |
| --- | --- |
| [Validation](validation.md) | Required commands, adversarial cases, oracles, CI and remote CUDA runs |
| [Rust engineering](rust-engineering.md) | Rules for Rust code, APIs, errors, safety and tests |
| [CUDA-class acceptance](performance-acceptance.md) | The protocol for performance claims |

## Reference

| Document | Covers |
| --- | --- |
| [Python binding survey](python-binding-survey.md) | The dependency survey behind the Python package |

## Where else documentation lives

- Many directories under `src/` have a README with their entry points.
- Module headers (`//!` comments at the top of each `src/*.rs` file) say what the file is and where to start reading.
- [tools/README.md](../tools/README.md), [test-support/README.md](../test-support/README.md) and [examples/README.md](../examples/README.md) cover the benchmark tooling, shared test helpers and the example library.
- Planned work and open gaps are tracked in [GitHub issues](https://github.com/chayan-bit/invar/issues), not in documents.
