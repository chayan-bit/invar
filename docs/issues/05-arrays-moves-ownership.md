# M0-05 — Arrays, moves, and control-flow ownership

Objective: add fixed owned arrays and local ownership analysis through bindings, assignments, indexing, joins, and loops.

Prerequisites: M0-04. Own array typing/runtime and local ownership facts; preserve the existing evaluator and avoid cloning or tracing GC.

Requirements: support contextual `Array<i32>`/`Array<u32>` literals, fixed lengths, scalar element reads/stores, `len`, filled intrinsics, exact index typing, bounds traps, declaration-order cleanup, and moved-state diagnostics. Enforce `let`/`var`, explicit `move`, no bare owner copy, RHS-before-destination evaluation, self-move rejection, no reinitialization of moved bindings, and path joins where only all continuing paths contribute. Reject moves of outer bindings from loops; allow fresh per-iteration values. Require explicit expected array-literal types including assignment context; reject inferred array literals. Destination-index evaluation may read its owner but may not mutably borrow or consume it.

Non-goals: call loans, slices/views, arrays of arrays, resizing, raw pointers, implicit copies, or native code.

Validation commands: `cargo test --locked`; run valid array fill/index/update/return fixtures and invalid use-after-move, copy-without-move, immutable store, wrong index, join, and loop-move fixtures.

Positive acceptance: valid local-array programs preserve element order and cleanup behavior. Negative acceptance: out-of-bounds and moved reads are rejected/trapped at the specified phase; no hidden deep copy occurs; call paths whose loan rules are not implemented yet are rejected as unsupported rather than executed unchecked. Stop when ownership facts are checked at every array access and exit edge, ready for M0-06 call semantics.
