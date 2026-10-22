# stencil_source

Stencil source recognition, contracts and device emission.
The module root is ../stencil_source.rs.
Contract and Emission bind recognized work to generated code.
host.rs handles execution; device_u32.rs handles wrapping-u32 device paths.

See [the architecture overview](../../docs/architecture.md) for the full pipeline and [validation](../../docs/validation.md) for the checks.
Preserve diagnostics, source order, deterministic observations and receipt identities.
Run focused checks through the required host wrapper before the full suite.
