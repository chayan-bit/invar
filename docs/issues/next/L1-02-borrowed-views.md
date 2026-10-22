# L1-02 — Precise borrowed views and lifetime-return rules

Problem/result: add non-owning array views with precise bounds and lifetime-return rules without weakening M0 whole-array ownership.

Dependencies: L1-01, M2-01.

Contract and implementation: specify view element type, range/stride representation, mutability, alias relation, reborrow duration, return eligibility, and cleanup obligations in grammar/spec and typed MIR. Checker facts must prove bounds and physical access; a view cannot outlive its owner, escape a call without a declared lifetime rule, or be used after a mutable conflict. Independent evaluator, native serial lowering, diagnostics, and source oracle must implement the same rules. Reject unknown alias/layout facts.

Positive cases: bounded read/mut subviews, nested reborrows, nonempty/empty ranges, view-return under the permitted lifetime rule, and disjoint proven slices agree across reference and native paths. Negative cases: out-of-bounds ranges, stride overflow, owner move/drop while borrowed, overlapping mut views, escaping temporaries, invalid returned lifetime, and unsupported source layout reject in every engine. A well-typed view unsupported by a parallel backend may use the serial path; fallback cannot bypass lifetime or alias safety.

Acceptance interface and evidence: add `issue_l1_02_borrowed_views`; run `cargo test --locked issue_l1_02_borrowed_views` and require a nonzero test count (zero-test success fails). Record spec/grammar changes, lifetime fact traces, oracle/native outputs, mutation tests, and physical access evidence. Keep M0 whole-array tests as regression cases.

Exclusions/stop: no general lifetime solver, unsafe raw pointers, implicit materialization, arbitrary strided GPU support, or closure capture. Stop when the minimum view forms are complete across grammar, checker, evaluator, and native serial semantics.

Bound view count, range arithmetic, nesting, and diagnostic size. Unknown lifetime or layout information is rejection, never an inferred permission.

Every accepted view operation records its owner, interval, mutability, and return boundary for review.
