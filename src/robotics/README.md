# robotics

Robot-specific source types, layouts and planner interfaces.
The module root is ../robotics.rs.
JointStateWire and JointStateConversionError define joint-state conversion.
planner.rs provides port/config layouts and generated headers.

See [the architecture overview](../../docs/architecture.md) for the full pipeline and [validation](../../docs/validation.md) for the checks.
Preserve diagnostics, source order, deterministic observations and receipt identities.
Run focused checks through the required host wrapper before the full suite.
