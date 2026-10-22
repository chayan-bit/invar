# M1-04 finite term oracle and typed graph comparison

Status: the direct evaluator in `src/term.rs` is the independent side of the
typed source/net comparison. The M1-04 revision uses `src/typed_net.rs` for
typed connectivity, verification, independent demand execution and decoding.
It does not claim a source-to-three-agent translation.

The internal term fragment has `Unit`, `Bool`, `Nat(u32)`, products, variables
with stable binder IDs, affine lambdas, application, `let`, `if`, and named
calls. `Fail(label)` is an internal provenance fixture, not Invar source
syntax. Function position precedes argument evaluation; named-call arguments
run left to right. An unselected branch is not demanded. Lambdas and products
containing them are affine. A closure captures only free binder IDs, moving
affine values out of the surrounding environment. The oracle reports attempted
demands and completed terms, traps, missing/moved binders, and explicit node,
depth, or step limits. A limit is a tool outcome, not a proof of divergence.

The fixed `issue_m1_04_term_oracle` fixtures use at most 64 source nodes,
depth 16, and 32 evaluation steps. They show that a discarded divergent
argument is demanded, an unselected divergent branch is not, function-side
failure precedes argument failure, named-call arguments stop at the first
trap, and affine closure duplication/capture moves are visible. The fixtures
also check bounded recursive calls.

The graph encoder uses explicit ordered child connections and supplied type
annotations without running this oracle first. `tests/issue_m1_04.rs`
compares decoded structure and exact demand/completion/result observations for
discarded divergent arguments, unselected branches, and first failures.
The [typed graph contract](typed-interaction-v1.md) freezes the M1 scope;
primitive expansion remains the separate
[research gap](research/m1-04-encoding-gate.md).
