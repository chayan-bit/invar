# bin

Runtime helper binaries of the compiler.
invar-sequencer.rs runs system sequencing.
invar-nvrtc.rs compiles CUDA source out of process for the CUDA engine and requires the cuda feature.

The acceptance, qualification and benchmark binaries (invar-bench, invar-heldout, invar-performance-report, invar-qualify, invar-pathsum-gate, invar-m3-03-cold) live in [tools](../../tools/README.md).

See [the architecture overview](../../docs/architecture.md) for the full pipeline and [validation](../../docs/validation.md) for the checks.
Preserve diagnostics, source order, deterministic observations and receipt identities.
Run focused checks through the required host wrapper before the full suite.
