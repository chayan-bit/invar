# L1-01 - Modules, records, variants, Result, and pattern ownership

Problem/result: extend the useful language beyond the M0 scalar/array subset with a deliberately small module, record, variant, `Result`, and ownership-aware pattern subset.

Dependencies: M1-05, AI-01.

Contract and implementation: update grammar, parser spans, checker diagnostics, independent evaluator, typed MIR, native serial lowering, and capability documentation together. Specify module visibility/import resolution, record field ownership, variant construction/matching, exhaustive patterns, `Result<T,E>` propagation, move/borrow behavior through patterns, and stable source-order evaluation. Keep error values distinct from source traps and tool errors. Add no implicit clone, hidden cleanup, or host panic semantics.

Positive cases: private/public names, forward module references within the declared boundary, moved fields, borrowed patterns, exhaustive variants, nested Result success/error, and native/reference agreement pass. Negative cases: duplicate exports, cycles outside the defined policy, missing fields/arms, non-exhaustive matches, wrong ownership, invalid visibility, malformed imports, and implicit conversion reject with spans.

Acceptance interface and evidence: add `issue_l1_01_language_core`; run `cargo test --locked issue_l1_01_language_core` and require a nonzero test count (zero-test success fails). Record grammar/spec revision, oracle fixtures, checker diagnostics, MIR/native results, and capability status. Include a positive and negative fixture for every ownership rule and ensure old M0 examples remain unchanged.

Exclusions/stop: no unrestricted generics, traits, macros, async, foreign ABI, or large standard library. Stop when the minimum subset has complete syntax/spec/oracle/checker/native coverage and unsupported forms fail explicitly.

Bound module count, nesting, record fields, variant arms, match depth, and diagnostic output. Preserve source spans through lowering and keep cycle/resource errors typed.

Keep module resolution deterministic and keep the reference engine usable during development. Forced native execution reports unsupported lowering rather than invoking the interpreter silently; completion requires native support for every declared feature.
