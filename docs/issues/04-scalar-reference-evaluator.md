# M0-04 — Scalar reference evaluator with fuel and checked arithmetic

Objective: execute the checked scalar subset with the normative evaluation order, traps, cleanup, and bounded fuel.

Prerequisites: M0-03. Own the reference evaluator and scalar runtime; keep it separate from any future native lowering.

Requirements: evaluate left-to-right, short-circuit `and`/`or`, selected `if` arms, `while`, counted `for`, calls, returns, and Unit statements. Charge one fuel unit before every expression and statement dispatch, including while-condition reevaluation and loop-control steps. Implement checked i32/u32 arithmetic, division/remainder edge cases, source-span trap categories, recursion/call-depth limits, and process/JSON status distinctions. Never expose host signed overflow or invalid division. Exactly N charged steps may execute with fuel N; the next step reports evaluation_limit with steps_used N. Static checking consumes no fuel.

Non-goals: arrays, ownership, LLVM, optimization, native differential comparison, or treating fuel exhaustion as proof of divergence.

Validation commands: `cargo test --locked`; run reference fixtures for short-circuiting, branch traps, signed minimum cases, division by zero, loop fuel boundaries, and recursive call limits.

Positive acceptance: completed scalar programs return deterministic results and exact trap spans/statuses. Negative acceptance: unselected traps do not fire; `MIN / -1` traps as overflow, `MIN % -1` returns `0`, zero divisor, other overflow, and fuel exhaustion produce their specified distinct outcomes. Stop when a small runnable corpus pins every fuel and arithmetic boundary.
