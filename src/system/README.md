# system

System IR, foreign components, robots and generated integrations.
The module root is ../system.rs and defines SystemIr.
import_mjcf and import_urdf ingest models; generation entry points emit artifacts.
graph, foreign, ros2, sequencer and tool_projection implement subsystem boundaries.

See [the architecture overview](../../docs/architecture.md) for the full pipeline and [validation](../../docs/validation.md) for the checks.
Preserve diagnostics, source order, deterministic observations and receipt identities.
Run focused checks through the required host wrapper before the full suite.
