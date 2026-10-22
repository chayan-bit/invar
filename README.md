# Invar

**A Rust-implemented language for safe native code and automatic parallelism grounded in typed interaction semantics.**

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

## Status

Invar is pre-release and open source; contributions are welcome.
It is a development snapshot, not a complete v0.1 release.

Implemented on `main`: source checking, reference and MIR evaluators, typed interaction nets, LLVM native CPU execution, bounded modules/records/variants/`Result`, borrowed views, generics and closures, bytes and UTF-8 strings, an explicit C ABI, and integer units and nominal frames.
Guarded CUDA and Metal code generation exists for admitted program families, including the v0.2 [explicit stable sorting](docs/design/v02-sort.md) and [spatial grid queries](docs/design/v02-spatial.md).
Integration, full automatic parallelism and GPU performance acceptance remain in progress; see [remaining work](docs/remaining-work.md).
No safety theorem, native performance result, parallel speedup, or AI-authoring advantage is claimed.

The command name is `invar` and source files use `.inv`.
`invar check PATH [--format json]` checks syntax, names, types, ownership, call loans and lexical view lifetimes.
`invar run PATH --engine reference [--fuel N] [--format json]` executes a program on the reference evaluator.
Foreign programs require an explicit absolute `--c-library PATH`; the [versioned C header](include/invar_foreign_v1.h) defines their narrow scalar/buffer ABI.
`invar system gen ros2 PATH --out DIR` emits a deterministic ROS 2 workspace from a checked system description.
The optional `native` feature adds [LLVM CPU build and run](docs/native-toolchain-v1.md).
The [bootstrap specification](spec/bootstrap.md) records the supported subsets and limits.

## Build and test

The toolchain is pinned in `rust-toolchain.toml` (Rust 1.95.0); `rustup` installs it on first use.
From the repository root:

```sh
cargo build --locked
cargo run --locked --bin invar -- run examples/sum.inv --engine reference
cargo fmt --all -- --check
cargo clippy --locked --all-targets -- -D warnings
cargo test --locked
```

The full suite is large; `cargo nextest run --locked --profile ci` runs it in parallel with the timeouts in `.config/nextest.toml`.
The `justfile` collects the same commands (`just check`, `just test-quick`).
GPU tests are ignored by default and need the `cuda` or `metal` feature plus matching hardware (`just test-gpu-cuda`, `just test-metal`).

The parser accepts at most 1 MiB of UTF-8 source and nesting depth 32.
The reference evaluator limits call depth to 64 and fuel to the positive decimal CLI value (default 1,000,000 steps).
These are tool limits, not source-language values.

## Commitments

- Defined evaluation, arithmetic, trap, and cleanup behavior in every build mode.
- Moves and borrows; no mandatory tracing GC, hidden deep copies, or model service.
- Ordinary native execution without a mandatory graph runtime or task pool.
- Automatic parallelism only for admitted transformations. Unknown legality means fallback.
- Structured diagnostics and local interfaces for humans and external coding tools.
- Honest separation of compiler trust, bounded testing, and actual proof.

## Documentation

| Document | Purpose |
| --- | --- |
| [v0.1 language specification](docs/spec/invar-v0.1.md) | Compact specification whose every code block and cited statement is checked by the test suite |
| [Bootstrap specification](spec/bootstrap.md) | Detailed normative text for the implemented language |
| [Grammar](spec/grammar.ebnf) | Complete syntax for that scope |
| [Examples](examples/README.md) | Verified example library with an index of expected results |
| [Architecture](docs/architecture.md) | Long-term design and trusted boundaries |
| [Implementation plan](docs/implementation-plan.md) | Compiler pipeline, mathematical laws, contracts and delivery sequence |
| [Parallelism foundations](docs/parallelism-foundation.md) | Interaction calculus, structured operations, and automatic GPU generation |
| [CUDA-class acceptance](docs/performance-acceptance.md) | Frozen workload families, fair baselines and hardware performance gates |
| [Rust engineering](docs/rust-engineering.md) | Project-wide Rust code, API, error, safety, and test rules |
| [Validation](docs/validation.md) | Conformance cases and evidence gates |
| [Roadmap](docs/roadmap.md) | Reference core, native and interaction core, CPU/GPU parallelism |
| [Implementation issues](docs/issues/README.md) | Ordered, bounded work items |
| [Contributing](CONTRIBUTING.md) | Development and review workflow |

## License

Invar is licensed under the [Apache License, Version 2.0](LICENSE); see also [NOTICE](NOTICE).
The vendored CORE-MATH sources in `crates/invar-math/core-math/` keep their upstream MIT license.
Unless you explicitly state otherwise, any contribution you intentionally submit for inclusion in this project is licensed under Apache-2.0, without any additional terms or conditions.
