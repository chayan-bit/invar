# L1-03 — Bounded generics and affine closures

Problem/result: add a small monomorphized generic subset and affine closures sufficient for reusable typed operations without opening an unconstrained type-system project.

Dependencies: L1-02.

Contract and implementation: freeze generic parameter kinds, bounds, inference limits, monomorphization identity, closure capture modes, call syntax, ownership transfer, and escape rules. Update grammar/spec, checker, independent evaluator, typed MIR, native serial lowering, diagnostics, and source oracle. Closures capture explicitly by copy/borrow/take, cannot outlive captured owners, and cannot hide effects or cleanup. Reject recursive/unbounded instantiation and ambiguous inference.

Positive cases: one- and two-parameter generic functions, explicit/limited inferred arguments, affine closure calls, borrowed captures, owned captures, and monomorphized native/reference agreement pass. Negative cases: unconstrained inference, type mismatch, incompatible instantiation identity, escaping borrow, double move, hidden effect, recursive expansion, and unsupported higher-kinded/trait syntax reject.

Acceptance interface and evidence: add `issue_l1_03_bounded_generics_closures`; run `cargo test --locked issue_l1_03_bounded_generics_closures` and require a nonzero test count (zero-test success fails). Record grammar/spec version, instantiation count, closure capture traces, oracle/native outcomes, and minimized rejection fixtures.

Exclusions/stop: no general trait solver, dynamic dispatch, macro system, unrestricted closures, or GPU closure kernels. Stop when the bounded subset has complete language and serial implementation evidence and all resource limits are enforced.

Bound generic depth, monomorphizations, closure size, capture count, call depth, and diagnostics. Native code generation must use the same typed contract as the evaluator.

Instantiation identities are stable and include source identity and substituted types; stale instances are rebuilt rather than reused.

Keep closure invocation order explicit. Repeated uses of the same concrete generic instance reuse it; they are not errors. An invalid affine capture is a source error in every backend. Only optimization-specific unknown legality may fall back to a well-typed ordinary serial implementation.
