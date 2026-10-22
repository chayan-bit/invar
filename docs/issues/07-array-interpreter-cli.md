# M0-07 — Full array interpreter and CLI/JSON integration

Objective: integrate the complete bootstrap checker and reference interpreter behind `invar check` and `invar run`.

Prerequisites: M0-06. Own CLI wiring, file/encoding handling, JSON serialization, and end-to-end fixtures; reuse the existing parser, checker, and reference engine.

Requirements: enforce exactly one `fn main() -> i32` for `run`, permit harness files for `check`, support `--engine reference`, positive decimal `--fuel`, and `--format json`. Emit one JSON document on stdout with schema version, status, diagnostics, result/trap/fuel fields as applicable; keep logs off stdout. Match exit codes 0/1/2/3/4 for success/source/tool/trap/evaluation-limit. Distinguish invalid UTF-8, resource exhaustion, and source errors. Apply documented parser, call-depth, allocation, and fuel limits.

Non-goals: native backend, differential oracle, networked service, extra syntax, or changing required JSON meanings.

Validation commands: `cargo test --locked`; invoke the built `invar check` and `invar run` on valid examples, no-main harnesses, traps, invalid UTF-8, bad arguments, and fuel exhaustion; parse stdout as exactly one JSON document.

Positive acceptance: end-to-end array programs return expected values and statuses, including EX-01: `examples/sum.inv` produces `10` and `examples/ownership.inv` produces `41`. Negative acceptance: `run` without main returns `source_error` with exit code 1, banners never pollute JSON, and main's i32 value does not become process exit status. Stop when the documented commands work from a clean checkout.
