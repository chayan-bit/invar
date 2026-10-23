# Typed interaction graph v1 (M1)

This is the accepted M1 typed graph contract after the 2026-09-23 design
revision. No Invar source-to-primitive-agent encoding is claimed.

## Finite term graph

`src/typed_net.rs` stores nodes by bounded index. Each live node has exactly
one typed result port, a source ID, a conservative effect fact, and ordered
input connections. The entry and named function bodies are roots. Every other
node has exactly one parent; unreachable nodes, cycles, invalid IDs, duplicate
source IDs, missing inputs, incompatible types, and excess limits reject.
Types are supplied explicitly for every internal source term ID and checked
against each node rule. This is a test language, not public Invar syntax.

| Node | Input ports in demand order | Result and obligation |
| --- | --- | --- |
| `Unit`, `Bool`, `Nat` | none | Corresponding scalar |
| `Pair` | left, right | Product; both inputs demanded |
| `Var` | none | Binder type; affine value moves on use |
| `Lambda` | body | Function; body deferred, free affine captures move when closure is made |
| `Apply` | function, argument | Function result; function and argument demanded before body |
| `Let` | producer, body | Body result; producer demanded even when binder unused |
| `Select` | condition, yes, no | Equal branch types; only selected branch demanded |
| `Call` | arguments left to right | Declared result; unknown effect prevents current rewrite admission |
| `Fail` | none | Declared result type; explicit trap fixture |
| `ArrayU32` | none | Bounded affine array, wrapping-u32 numeric policy |
| `Map` | array | Fresh array, one total wrapping add or multiply per element |
| `FusedMap` | array | Fresh array, ordered list of total wrapping operations per element |

The verifier counts uses of each affine binder. Sequential inputs add their
counts; alternative `Select` branches use the maximum because only one branch
is demanded. A function or array value cannot be used twice. Effect facts are
recomputed from connectivity; unknown or possibly failing work cannot satisfy
the current total-rule guard. Term evaluation follows these connections and
records demand/completion events independently of `term::observe`. A bounded
step/depth/node exhaustion is a tool result, not source divergence.

## Public MIR connections

`src/mir_net.rs` takes a verified public MIR program, removes each operation
and terminator operand from a retained MIR skeleton, and stores the connection
as a typed `(use site, producer value)` wire. Extraction fills those ports
from wires and reruns `mir::verify`. MIR's existing verifier enforces
dominance, types, ownership/loans, cleanup, source spans, and control-flow
rules. The MIR block graph and operation order remain in the skeleton; M1 does
not rewrite public source loops. A swapped operand can produce a different
verified result, while a missing/duplicate/type-invalid wire rejects.

## Rewrites and extraction

`affine-beta-v1` changes a *closed, total root* `Apply(Lambda, argument)` into a
`Let(argument, body)` connection. It declines captured closures or uncertain
effects. `total-u32-map-fusion-v1` changes two adjacent total map nodes into
one fused node with the original operation order. The rule has a maximum
operation count. Both reverify a cloned candidate before committing it.
Successful rewrites increment the graph revision; old proof tokens fail.

Extraction decodes term connectivity into a serial term plan, or a bulk chain
into serial loop stages. Fusion changes two stages to one. Public MIR
extraction reconstructs verified MIR, which the existing LLVM CPU backend can
compile. Direct MIR lowering remains a labeled comparison path. Native CPU
does not automatically run interaction rewrites in M1 because the public
bootstrap language has no recognized bulk region or lambda syntax. M2 owns
recognition, admission, scheduling, and the first automatic source-to-bulk
route.

Proof tokens and source identities use process-local hashes for stale-data
checks. They are not serialized trust artifacts or cryptographic digests.
Verification must run again after loading or rebuilding any future region.
