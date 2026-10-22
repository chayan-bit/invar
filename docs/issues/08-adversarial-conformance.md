# M0-08 — Adversarial conformance, bounded corpus, and Rust tooling coverage

Objective: harden the accepted bootstrap behavior with a bounded adversarial corpus and Rust debug/release checks.

Prerequisites: M0-07. Own conformance fixtures, corpus runner, and Rust validation configuration; do not invent a native backend or fake differential oracle.

Requirements: cover malformed UTF-8, lexical boundaries, deep nesting, oversized literals/arrays, duplicate names, all required diagnostics, arithmetic/index traps, fuel boundaries, every ownership/loan conflict, cleanup paths, and JSON stdout discipline. Bound corpus count, source size, nesting, allocation, and runtime. Keep bootstrap code safe Rust with `#![forbid(unsafe_code)]`; run the stable Cargo gates and optional bounded Miri for pure-core tests only when a pinned nightly is available. Compare repeated reference runs for deterministic observations only.

Non-goals: differential testing against native code, fuzzing without limits, proof of termination, performance claims, LLVM/GPU, or broad random infrastructure.

Validation commands: `cargo fmt --all -- --check`; `cargo clippy --locked --all-targets -- -D warnings`; `cargo test --locked`; `cargo test --locked --release`; and, only when the documented pinned nightly exists, `cargo +<pinned-nightly> miri test --locked --lib` for the bounded pure core.

Positive acceptance: bounded valid corpus passes identically across repeated runs and debug/release builds. Negative acceptance: malformed and limit-exhausting inputs terminate with documented tool/source statuses without host panics; any configured Miri check reports no violations. Stop when the corpus is checked in, bounded, and its summary is reproducible without claiming differential evidence.
