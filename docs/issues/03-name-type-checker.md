# M0-03 — Names, types, and scalar control flow

Objective: type-check parsed programs for names, scalar expressions, declarations, and conservative return coverage.

Prerequisites: M0-02. Own semantic analysis and typed metadata adjacent to the parser; reuse one authoritative representation and do not implement execution.

Requirements: resolve forward function calls and recursion; reject duplicate/hiding names, unknown names, wrong arity, invalid modes on scalars, mismatched integer/Bool/Unit operations, invalid comparisons, bad conditions, bad `for` bounds, and invalid `main`. Check literal ranges even in unreachable code. Implement block scope, immutable `let`, `var` rebinding, and all-path non-Unit returns; Unit may fall through. Preserve expression and statement order in typed output. Check uncalled/unreachable source, exclude dead paths from flow joins, apply the specified conservative definite-return rules, and reject Unit outside return types. Use ENTRYPOINT_MISSING and ENTRYPOINT_SIGNATURE for entrypoint errors.

Non-goals: array ownership, call loans, runtime traps, native lowering, implicit conversions, constant-folding semantics, or truthiness.

Validation commands: `cargo test --locked`; run the built `invar check` fixtures for valid forward calls/recursion and invalid duplicate, unknown, mismatched, missing-return, and unreachable out-of-range cases.

Positive acceptance: valid scalar fixtures type-check with stable typed metadata. Negative acceptance: no numeric promotion or unreachable-code exemption is accepted; a Unit fallthrough passes while a missing i32 return fails. Stop when all scalar and control-flow rules have executable positive and negative fixtures.
