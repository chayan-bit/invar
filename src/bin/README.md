# bin

Additional command binaries.
invar-sequencer.rs runs system sequencing.
invar-bench.rs dispatches the benchmark harness in bench/.
invar-nvrtc.rs and invar-m3-03-cold.rs require the cuda feature.

See [the architecture overview](../../docs/architecture.md) for the full pipeline and [validation](../../docs/validation.md) for the checks.
Preserve diagnostics, source order, deterministic observations and receipt identities.
Run focused checks through the required host wrapper before the full suite.
