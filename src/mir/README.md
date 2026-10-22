# mir

Typed control-flow IR, lowering, dominance and verification.
The module root is ../mir.rs.
Program, Function, BasicBlock, Instruction and Terminator define the IR.
lower constructs MIR; builder.rs and verify.rs enforce its contract.

See [the architecture overview](../../docs/architecture.md) for the full pipeline and [validation](../../docs/validation.md) for the checks.
Preserve diagnostics, source order, deterministic observations and receipt identities.
Run focused checks through the required host wrapper before the full suite.
