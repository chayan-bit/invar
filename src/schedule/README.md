# schedule

Source execution scheduling across CPU, CUDA and Metal.
mod.rs defines Run and Executed and exposes run_source.
workers.rs handles bounded CPU work; families.rs dispatches source families.
cuda.rs and metal.rs enforce device selection and completion paths.
device.rs holds the run records and auto bookkeeping they share.

See [the architecture overview](../../docs/architecture.md) for the full pipeline and [validation](../../docs/validation.md) for the checks.
Preserve diagnostics, source order, deterministic observations and receipt identities.
Run focused checks through the required host wrapper before the full suite.
