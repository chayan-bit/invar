# Minimal interaction reducer v1

Status: M1-03 reference implementation, landed with the initial Rust bootstrap. This is an untyped
pure graph calculus. It provides no source encoding, demand, ownership, or
automatic parallelization admission.

The graph has `gamma` and `delta` agents with principal port `0` and auxiliary
ports `1,2`, and `epsilon` with only principal port `0`. Boundary ports have
stable integer IDs. Every live port is connected to exactly one other port;
connections are reciprocal by construction. A principal-to-principal wire is
an active pair. The reducer chooses the pair with the lowest sorted agent IDs.
Agent IDs are never reused. `Limits` bounds total allocated agent IDs, live
ports plus boundaries, and completed rewrite steps. Invalid graphs and budget
exhaustion return explicit errors; a step-limit result leaves a valid partial
net and does not assert divergence.

Rule revision 1 is exactly six rules: crossed gamma-gamma,
aligned delta-delta, epsilon-epsilon annihilation, epsilon-gamma and
epsilon-delta erasure (two fresh erasers), and gamma-delta commutation (two
fresh agents of each kind, with the stated eight wires). The implementation
glues rule interfaces through existing auxiliary wires, including internal
wires and cyclic wires with no ports, then validates every port and boundary after each
rewrite. It does not silently apply symmetric HVM annihilation.

The fixed, hand-derived corpus in `issue_m1_03_minimal_reducer` uses limit
`(64 agents, 200 live ports, 16 steps)`:

| Fixture ID | Steps | Expected normal result |
| --- | ---: | --- |
| GG-1 | 1 | boundary pairs `(0,3),(1,2)` |
| DD-1 | 1 | boundary pairs `(0,2),(1,3)` |
| EE-1 | 1 | empty |
| EG-1, ED-1 | 1 | two boundary-to-eraser principal wires |
| GD-1 | 1 | four new agents and the eight specified wires |
| GG-2 | 2 | composed boundary pairs `(0,2),(1,3)` |
| GG-CLOSED | 1 | no agents or boundaries, one cyclic wire |
| GD-LOOP | 2, then limit | still active; explicit step exhaustion |

The order-sensitive GG-1 witness distinguishes crossed `(0,3),(1,2)` from
aligned `(0,2),(1,3)`. A valid wrong-port fixture mutation changes the
boundary result. The suite enumerates all 24 assignments of four boundary
ports for each same-kind binary rule, and tests both orientations of each
eraser rule. Tests also reject dangling, duplicate, invalid agent and
boundary ports, unsupported agent names, and agent/port budgets. These finite
checks are not a proof of confluence, termination, or source-language
equivalence.

Lafont's [original net definition](https://dcreager.net/pdf/Lafont1997.pdf)
explicitly permits cyclic wires with no ports; the reducer retains their
count rather than dropping them.
