# exports

Checked scalar/array signatures and compiled export artifacts.
The module root is ../exports.rs.
compile_exports returns CompiledExports or ExportError.
ExportSignature, ExportParameter and ExportEngine describe the boundary.

See [the architecture overview](../../docs/architecture.md) for the full pipeline and [validation](../../docs/validation.md) for the checks.
Preserve diagnostics, source order, deterministic observations and receipt identities.
Run focused checks through the required host wrapper before the full suite.
