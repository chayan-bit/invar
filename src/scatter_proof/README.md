# scatter_proof

Scatter legality certificates and bounded host execution.
The module root is ../scatter_proof.rs.
Certificate and Contract retain checked source facts.
host::prepare and host::run_source enforce the admitted plan.

See [the architecture overview](../../docs/architecture.md) for the full pipeline and [validation](../../docs/validation.md) for the checks.
Preserve diagnostics, source order, deterministic observations and receipt identities.
Run focused checks through the required host wrapper before the full suite.
