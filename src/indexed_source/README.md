# indexed_source

Indexed gather/scatter/frontier source families and host execution.
The module root is ../indexed_source.rs.
Family, Candidate and Contract represent recognized source behavior.
host::prepare and host::run_source validate and execute admitted work.

See [the architecture overview](../../docs/architecture.md) for the full pipeline and [validation](../../docs/validation.md) for the checks.
Preserve diagnostics, source order, deterministic observations and receipt identities.
Run focused checks through the required host wrapper before the full suite.
