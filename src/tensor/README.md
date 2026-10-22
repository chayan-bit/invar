# tensor

Tensor contraction contracts, physical layouts and emission.
The module root is ../tensor.rs.
annotate and verify establish Contract; emit_cuda and emit_msl emit code.
host::run_source executes through CUDA, Metal or the explicit CPU model.

See [the architecture overview](../../docs/architecture.md) for the full pipeline and [validation](../../docs/validation.md) for the checks.
Preserve diagnostics, source order, deterministic observations and receipt identities.
Run focused checks through the required host wrapper before the full suite.
