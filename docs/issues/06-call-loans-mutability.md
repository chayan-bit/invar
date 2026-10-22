# M0-06 — Calls, loans, mutability, and conflict checking

Objective: implement array parameter modes and immediate, non-escaping call-loan checking.

Prerequisites: M0-05. Own parameter mode checking, loan state, and call argument validation; do not add general lifetime inference.

Requirements: require `read`, `mut`, or `take` for array parameters; accept only the specified available owners/borrowed parameters; allow nested reborrows with restoration; acquire each loan immediately after its argument evaluation and retain it through the outer call. Reject mutable aliasing, borrow/consume conflicts, incompatible repeated actuals, temporary/index borrow sources, return of borrowed arrays, and movement while loans live. Preserve left-to-right argument evaluation and callee mutation.

Non-goals: escaping references, stored loans, slices, closures, async suspension, lifetime/range solvers, scalar modes, or implicit cloning.

Validation commands: `cargo test --locked`; exercise valid read/read, mut reborrow, and take calls plus invalid `f(xs, g(xs))` conflicts, double mut, borrow-then-take, temporary borrow, and borrowed return fixtures.

Positive acceptance: legal nested calls complete and restore caller access. Negative acceptance: conflicts are rejected even when a callee body does not use both aliases; loan state cannot escape. Stop when every call mode and conflict example has a runnable diagnostic.
