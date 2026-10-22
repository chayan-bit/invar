# L1-05 — Practical standard library, formatter, and interface tooling

Problem/result: make the bounded language usable with a small standard library and stable authoring tools rather than promising a complete ecosystem.

Dependencies: L1-04.

Contract and implementation: select a minimal offline library surface for arrays, bytes/strings, Result handling, formatting, and deterministic collection helpers. Each item receives grammar/API documentation where needed, source oracle behavior, checker/interface metadata, native serial implementation, limits, and diagnostics. Add a formatter and machine-readable interface dump that reuse the authoritative parser/typed representation; formatting must not change semantics or spans used in diagnostics without a documented mapping.

Positive cases: documented examples, stable formatting, interface output, errors, ownership, and reference/native agreement pass under bounded inputs. Negative cases: unsupported API, invalid UTF-8, oversized formatting, ambiguous interface, changed public visibility, stale interface version, and non-deterministic output reject clearly.

Acceptance interface and evidence: add `issue_l1_05_stdlib_tooling`; run `cargo test --locked issue_l1_05_stdlib_tooling` and require a nonzero test count (zero-test success fails). Record API/spec revision, formatter idempotence, interface schema, oracle/native fixtures, diagnostics, and capability status. Verify old source and issue corpus remain valid.

Exclusions/stop: no package manager, network service, LSP server, broad collection hierarchy, or hidden allocation policy. Stop when the small offline surface and formatter/interface artifacts are reproducible and documented.

Bound API items, formatting output, interface size, allocations, and recursion. A missing library capability is an explicit error, not a fallback to host behavior.

Tool output remains offline, deterministic, versioned, and safe to consume by external coding tools.

Each library addition must carry a small usage example and a negative boundary case before it is advertised as supported.
