# general_net

Checked executable interaction nets and compiled execution ABIs.
mod.rs defines Net, Limits, Proof and stable error categories.
Net::lower and verification admit extraction; oracle provides independent observations.
compiled/ emits code; host modules manage validated sessions and receipts.

See [the architecture overview](../../docs/architecture.md) for the full pipeline and [validation](../../docs/validation.md) for the checks.
Preserve diagnostics, source order, deterministic observations and receipt identities.
Run focused checks through the required host wrapper before the full suite.
