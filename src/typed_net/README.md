# typed_net

Demand-ordered typed graphs, checked rewrites and extraction.
mod.rs defines Net, Node, Type, Effect, Proof and schema/rule identifiers.
verify.rs checks connectivity; rewrite.rs checks transformations; extract.rs produces plans.
This production contract is distinct from the independent three-agent reducer.

See [the architecture overview](../../docs/architecture.md) for the full pipeline and [validation](../../docs/validation.md) for the checks.
Preserve diagnostics, source order, deterministic observations and receipt identities.
Run focused checks through the required host wrapper before the full suite.
