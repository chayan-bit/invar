# eval

The source reference evaluator and observable language semantics.
The module root is ../eval.rs.
Value, Outcome, Observation, TrapCategory and CleanupEvent define results.
observe and named::observe_named execute bounded source programs.

See [the architecture overview](../../docs/architecture.md) for the full pipeline and [validation](../../docs/validation.md) for the checks.
Preserve diagnostics, source order, deterministic observations and receipt identities.
Run focused checks through the required host wrapper before the full suite.
