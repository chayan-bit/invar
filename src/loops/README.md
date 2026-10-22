# loops

Ordinary-loop recognition and legal map fusion.
The module root is ../loops.rs.
LiveMap, Capture and Fused preserve source-derived inputs.
single, fuse_pair and chain::fuse_chain are recognition/extraction entry points.

See [the architecture overview](../../docs/architecture.md) for the full pipeline and [validation](../../docs/validation.md) for the checks.
Preserve diagnostics, source order, deterministic observations and receipt identities.
Run focused checks through the required host wrapper before the full suite.
