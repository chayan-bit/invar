# check

Type checking, units, modules, generics and closure desugaring.
The module root is ../check.rs.
check(program, require_main) returns CheckedProgram or a source diagnostic.
ownership::check performs the separate affine ownership pass.

See [the architecture overview](../../docs/architecture.md) for the full pipeline and [validation](../../docs/validation.md) for the checks.
Preserve diagnostics, source order, deterministic observations and receipt identities.
Run focused checks through the required host wrapper before the full suite.
