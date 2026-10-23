<div align="center">

# Invar

**A Rust-implemented language for safe native code and automatic parallelism grounded in typed interaction semantics.**

[![CI](https://github.com/chayan-bit/invar/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/chayan-bit/invar/actions/workflows/ci.yml)
[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Rust 1.95](https://img.shields.io/badge/rust-1.95-orange.svg)](rust-toolchain.toml)
[![Status: pre-release](https://img.shields.io/badge/status-pre--release-yellow.svg)](https://github.com/chayan-bit/invar/issues)

</div>

Write ordinary sequential code.
Make ownership and observable effects understandable.
Let the compiler choose an equivalent implementation when it can establish legality and a useful cost advantage.

```text
fn sum(xs: read Array<i32>) -> i32 {
    var result: i32 = 0;
    for i in 0 .. len(xs) {
        result = result + xs[i];
    }
    return result;
}
```

This sum checks overflow in source order.
It cannot simply become a tree reduction: changing the grouping can change which addition traps.
Explicit modular reductions provide a lawful parallel alternative.

## Why Invar

- **Defined behavior in every build mode.** Evaluation order, arithmetic, traps and cleanup do not change between debug and release.
- **Ownership without a garbage collector.** Moves and borrows are checked; there are no hidden deep copies and no mandatory tracing GC.
- **Parallelism only when it is legal.** The compiler admits a parallel plan only when it can establish legality and a cost advantage; unknown legality means the sequential fallback.
- **Ordinary native execution.** The CPU path needs no graph runtime and no task pool.
- **Tooling for people and tools.** Diagnostics are structured, and a local interface is available to humans and external coding tools.
- **An honest trust boundary.** Compiler trust, bounded testing and actual proof are reported separately.

## Status

Invar is a pre-release development snapshot, not a complete v0.1 release.
Contributions are welcome.

Implemented on `main`: source checking, reference and MIR evaluators, typed interaction nets, LLVM native CPU execution, bounded modules/records/variants/`Result`, borrowed views, generics and closures, bytes and UTF-8 strings, an explicit C ABI, and integer units and nominal frames.
Guarded CUDA and Metal code generation exists for admitted program families, including the v0.2 explicit stable sorting and spatial grid queries (Appendix B of the [specification](docs/spec/invar-v0.1.md)).
Integration, full automatic parallelism and GPU performance acceptance remain in progress; open work is tracked in [GitHub issues](https://github.com/chayan-bit/invar/issues).
No safety theorem, native performance result, parallel speedup, or AI-authoring advantage is claimed.

## Quick start

Install [rustup](https://rustup.rs/); the pinned toolchain (Rust 1.95.0, see [rust-toolchain.toml](rust-toolchain.toml)) is installed on first use.

```sh
git clone https://github.com/chayan-bit/invar.git
cd invar
cargo build --locked
cargo run --locked --bin invar -- run examples/sum.inv --engine reference
```

[examples/sum.inv](examples/sum.inv) sums `[1, 2, 3, 4]`, so the program returns 10.
The [examples README](examples/README.md) describes the rest of the verified example library.

## Usage

The command name is `invar` and source files use `.inv`.

| Command | What it does |
| --- | --- |
| `invar check PATH [--format json]` | Checks syntax, names, types, ownership, call loans and lexical view lifetimes |
| `invar run PATH --engine reference [--fuel N] [--format json]` | Executes a program on the reference evaluator |
| `invar system gen ros2 PATH --out DIR` | Emits a deterministic ROS 2 workspace from a checked system description |

Foreign programs require an explicit absolute `--c-library PATH`; the [versioned C header](include/invar_foreign_v1.h) defines their narrow scalar/buffer ABI.
The optional `native` feature adds [LLVM CPU build and run](docs/native-toolchain-v1.md).
The [bootstrap specification](spec/bootstrap.md) records the supported subsets and limits.

The parser accepts at most 1 MiB of UTF-8 source and nesting depth 32.
The reference evaluator limits call depth to 64 and fuel to the positive decimal CLI value (default 1,000,000 steps).
These are tool limits, not source-language values.

## Development

The [justfile](justfile) collects the common commands:

```sh
just check       # format, lint, full Debug and Release suites, doctests
just test-quick  # fast inner loop
```

The full suite is large; `cargo nextest run --locked --profile ci` runs it in parallel with the timeouts in [nextest.toml](.config/nextest.toml).
CI runs CPU-only checks on GitHub-hosted runners.
CUDA and Metal suites are ignored by default and are not run in CI; see [CONTRIBUTING.md](CONTRIBUTING.md) for how to run them on matching hardware.

## Documentation

| Document | Purpose |
| --- | --- |
| [v0.1 language specification](docs/spec/invar-v0.1.md) | Compact specification whose every code block and cited statement is checked by the test suite |
| [Bootstrap specification](spec/bootstrap.md) | Detailed normative text for the implemented language |
| [Grammar](spec/grammar.ebnf) | Complete syntax for that scope |
| [Placement](spec/placement-v1.md) | Placement profiles and receipts for `--engine auto` |
| [Examples](examples/README.md) | Verified example library with an index of expected results |
| [Architecture](docs/architecture.md) | Long-term design and trusted boundaries |
| [Parallelism foundations](docs/parallelism-foundation.md) | Interaction calculus, structured operations, and automatic GPU generation |
| [Numeric contracts](docs/numeric.md) | f32 semantics, fold tiers and storage types |
| [Explain JSON](docs/explain.md) | `invar explain` documents and device execution reports |
| [System integration](docs/system-integration-v1.md) | Invar as the robot system description |
| [CUDA-class acceptance](docs/performance-acceptance.md) | Frozen workload families, fair baselines and hardware performance gates |
| [Rust engineering](docs/rust-engineering.md) | Project-wide Rust code, API, error, safety, and test rules |
| [Validation](docs/validation.md) | Conformance cases, CI and evidence gates |
| [Debugging](docs/debugging.md) | Source-level debugging of native programs |

Other files in `docs/` specify individual contracts (native toolchain, CUDA target, layouts, normal form and families).
Planned work and open gaps are tracked in [GitHub issues](https://github.com/chayan-bit/invar/issues).

## Contributing

Bug reports, documentation fixes, tests and implementation work are all welcome.
Read [CONTRIBUTING.md](CONTRIBUTING.md) first, and follow the [Code of Conduct](CODE_OF_CONDUCT.md).
For a substantial change, open an issue first so the design can be agreed before you invest in it.

## Security

Please report vulnerabilities privately through GitHub; see [SECURITY.md](SECURITY.md).

## License

Invar is licensed under the [Apache License, Version 2.0](LICENSE); see also [NOTICE](NOTICE).
The vendored CORE-MATH sources in `crates/invar-math/core-math/` keep their upstream MIT license.
Unless you explicitly state otherwise, any contribution you intentionally submit for inclusion in this project is licensed under Apache-2.0, without any additional terms or conditions.
