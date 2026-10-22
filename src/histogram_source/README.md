# histogram_source

Histogram recognition, bounded execution and host admission.
The module root is ../histogram_source.rs.
Candidate, Contract, Budget and Cost describe source-derived work.
bounded.rs seals runtime inputs; host.rs owns the execution boundary.

See [the architecture overview](../../docs/architecture.md) for the full pipeline and [validation](../../docs/validation.md) for the checks.
Preserve diagnostics, source order, deterministic observations and receipt identities.
Run focused checks through the required host wrapper before the full suite.
