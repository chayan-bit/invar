# M1-04 — Typed demand/ownership net verifier and source encoding

Decision revision: 2026-09-23. M1 requires a real typed interaction graph and
checked extraction. It does **not** require translation of source programs into
the three-agent Lafont reference reducer. That source-to-primitive derivation
is a separate research gate and must not be claimed from this issue.

Dependencies: M1-01, M1-03.

Contract: encode the bounded, typed internal term fragment into connected,
ordered input ports. A node has a source ID, result type, effect fact, and
explicit input connections. Verify arity, one parent per non-root node,
reachability, cycles, type compatibility, affine binder uses, effect facts,
schema, and resource bounds. Decode the actual graph and compare its demand,
completion, value, first trap, and tool-limit observations with the independent
term evaluator. Untyped or unsupported source does not acquire an implicit
type or optimization permission.

For public bootstrap programs, encode already verified typed MIR into a graph
whose typed wires own every value-operand connection. Extract MIR from those
wires, then rerun the MIR verifier. This retains MIR's dominance, ownership,
loan, cleanup, type, source-span, and control-flow checks; it does not invent a
second source semantics. A removed, duplicated, mismatched, or swapped wire
must reject or affect extracted execution. Preserve direct MIR lowering as an
experimental comparison path.

Positive cases: term graph/source agreement for a discarded divergent
call-by-value argument, an unselected divergent branch, and first-failure
ordering; source/MIR/graph-extracted MIR agreement for public scalar and array
programs. Negative cases: shared/dangling ports, invalid type or effect,
duplicated affine closure, stale proof/schema, malformed public MIR, missing
cleanup, and bounded resources reject explicitly.

Acceptance: `cargo test --locked issue_m1_04_typed_net_verifier -- --list`
must select nonzero tests; run the filter in Debug and Release. Record schema
and rule versions, fixture counts, diagnostics, source identities, and the
matching or differing observations. Run full milestone gates from
`docs/validation.md`. See `docs/typed-interaction-v1.md` for the frozen M1
node/port and extraction rules.

Exclusions: no arbitrary source-to-Lafont encoding, GPU, automatic loop
recognition, serialized proof trust, or general-purpose graph theorem.
M1-03 remains an independent untyped reference calculus.
