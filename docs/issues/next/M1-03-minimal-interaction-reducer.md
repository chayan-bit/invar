# M1-03 — Minimal interaction-combinator reducer

Problem/result: establish the required executable interaction foundation for the raw untyped calculus without pretending that a general graph is automatically an interaction net.

Dependencies: M0-09.

Contract and implementation: freeze a minimal combinator syntax, principal-port matching rule, erasing/duplicating behavior, bounded reduction steps, and result/error model. Implement the untyped pure reference reducer in safe Rust with checked graph storage. Hand-built net fixtures carry fixture spans/IDs. Source translation, typed affine permissions, CBV demand and their acceptance belong entirely to M1-04; untyped delta does not itself enforce source ownership. Use the interaction research/design evidence when present; do not invent universal confluence, termination, or hardware guarantees.

Positive cases: hand-built annihilation, commutation, erasure and composed nets reduce to specified boundary connections. Negative cases: malformed port degree, duplicate wiring, invalid boundary ports, unsupported agents, reduction-step exhaustion, and nontermination are rejected or bounded. A permutation of independent-looking nodes is not accepted unless the frozen rule semantics justify it.

Acceptance interface and evidence: add `issue_m1_03_minimal_reducer`; run `cargo test --locked issue_m1_03_minimal_reducer`. Record the grammar/rule revision, hand-built net corpus, step limits, normal forms, and minimized failures. Include a hand-checked order-sensitive witness and state that finite tests are not a proof.

Exclusions/stop: no production compiler IR, arrays at arbitrary scale, LLVM, GPU, effects, scheduler, or source-language extension. Stop when the minimal calculus and its hand-built net fixtures are independently executable and documented.

Keep the reducer's maximum nodes, ports, and steps explicit in every test. Record normal-form or limit outcomes and compare boundary connections with hand-derived expectations; do not use reducer agreement alone as a semantic proof.

## Concrete integration requirements

Use the exact six port-rewrite rules in docs/research/interaction-design.md, including the crossed gamma-gamma rule. Validate reciprocal wiring and boundary preservation before/after each rewrite; do not substitute HVM symmetric rules silently. Include annihilation, commutation and erasure examples with explicit expected external connections plus a wrong-port mutation that fails.

Follow the shared acceptance interface in `docs/implementation-plan.md`: list and verify a nonzero test selection, run the issue suite in Debug and Release, and retain full milestone Cargo gates. Commands are required deliverables, not existing functionality in the planning checkout.
