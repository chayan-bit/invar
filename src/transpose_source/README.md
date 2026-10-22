# transpose_source

Transpose source recognition, contracts and emission.
The module root is ../transpose_source.rs.
Contract and Emission retain recognized source/layout facts.
host::contracts, provide and run_source handle execution inputs and completion.

See [the architecture overview](../../docs/architecture.md) for the full pipeline and [validation](../../docs/validation.md) for the checks.
Preserve diagnostics, source order, deterministic observations and receipt identities.
Run focused checks through the required host wrapper before the full suite.
