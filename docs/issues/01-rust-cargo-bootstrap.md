# M0-01 — Rust Cargo bootstrap

Objective: establish a reproducible Rust 2024 package with a library, binary, locked dependencies, and real Cargo tests.

Prerequisites: none. Own `Cargo.toml`, `Cargo.lock`, `rust-toolchain.toml`, `src/`, `tests/`, `.github/workflows/`, and build documentation; do not prescaffold later compiler subsystems.

Requirements: use Rust 2024, choose and record the exact tested stable toolchain and `rust-version` without inventing a version, pin that toolchain in `rust-toolchain.toml`, and commit `Cargo.lock`. Follow [Rust engineering](../rust-engineering.md). Start with one package exposing both library and binary targets. Add `#![forbid(unsafe_code)]` to both targets, warnings-as-errors through Clippy, and a real smoke test exercising `target/debug/invar --help` (with the matching release binary in release tests). CI must run the locked Debug and Release gates on the recorded toolchain. Provision dependencies before offline checks, then verify locked offline operation. Use standard library facilities first; scoped reviewed dependencies such as `serde_json` are allowed where they remove JSON correctness risk.

Non-goals: lexer, parser, language semantics, LLVM/GPU/native lowering, network services, package managers, and placeholder subsystem trees.

Validation commands:

```text
cargo fmt --all -- --check
cargo clippy --locked --all-targets -- -D warnings
cargo test --locked
cargo test --locked --release
cargo build --locked
# after dependencies are provisioned:
cargo test --locked --offline
```

Positive acceptance: all gates pass from a clean checkout on the recorded stable toolchain, with library and binary tests executing in Debug and Release. Negative acceptance: a deliberately failing test is observed to fail, Clippy warnings fail the build, and an unsupported compiler (below declared rust-version) or unresolved lockfile is reported clearly. A newer compatible compiler is not falsely claimed to be unsupported merely because CI pins a specific release. Stop when the clean checkout reproduces the locked Cargo gates and offline check after dependency provisioning.
