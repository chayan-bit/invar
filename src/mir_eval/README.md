# mir_eval

MIR execution and bounded bulk-family observations.
The module root is ../mir_eval.rs.
run and observe execute MIR; observe_with_scheduler adds scheduling.
Family modules preserve source failure, fuel and publication behavior.

See [the architecture overview](../../docs/architecture.md) for the full pipeline and [validation](../../docs/validation.md) for the checks.
Preserve diagnostics, source order, deterministic observations and receipt identities.
Run focused checks through the required host wrapper before the full suite.
