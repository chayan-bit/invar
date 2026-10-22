# M0-09 — Final M0 acceptance and reproducible evidence

Objective: freeze and accept the offline reference-core milestone with auditable evidence.

Prerequisites: M0-01 through M0-08. Own the acceptance report and issue cross-links; do not add features or rewrite earlier contracts.

Requirements: run the canonical Rust fmt, locked Clippy, Debug test, Release test, locked build, and provisioned-dependency offline checks, plus optional pinned-nightly Miri evidence from M0-08. Record commit, stable toolchain, `rust-version`, OS, commands, exit codes, test counts, artifact paths, and any unsupported environment. Reconcile implementation behavior against `spec/bootstrap.md` and `spec/grammar.ebnf`; document every clarification as a regression case. Keep all later architecture claims clearly out of M0 and follow `docs/rust-engineering.md`.

Non-goals: native lowering acceptance, optimization legality, parallel speedup, GPU, networking, formal proof, or release packaging.

Validation commands: from a clean checkout run `cargo fmt --all -- --check`, `cargo clippy --locked --all-targets -- -D warnings`, `cargo test --locked`, `cargo test --locked --release`, `cargo build --locked`, and then `cargo test --locked --offline` after dependencies are provisioned; run bounded corpus and CLI checks, and optional pinned-nightly Miri if available.

Positive acceptance: a fresh environment reproduces the recorded Cargo gates and all required statuses/diagnostics, with no host panic or overflow standing in for Invar behavior. Negative acceptance: any mismatch, flaky test, missing artifact, unpinned toolchain, or unsupported claim leaves the acceptance evidence incomplete with a concrete failure record. Stop when the evidence is complete, reproducible, and review-ready for M0 sign-off; issue closure may follow the normal merge workflow.
