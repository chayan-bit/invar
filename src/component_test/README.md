# component_test

The component test CLI and planner subprocess integration.
The module root is ../component_test.rs.
command receives CLI arguments and returns an exit status.
planner::run executes the planner-specific test path.

See [the architecture overview](../../docs/architecture.md) for the full pipeline and [validation](../../docs/validation.md) for the checks.
Preserve diagnostics, source order, deterministic observations and receipt identities.
Run focused checks through the required host wrapper before the full suite.
