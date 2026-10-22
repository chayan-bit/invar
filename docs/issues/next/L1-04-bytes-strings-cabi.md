# L1-04 — Bytes, strings, and explicit C ABI

Problem/result: provide bounded byte/string values and a narrow explicit C ABI for practical integration while keeping ownership and encoding visible.

Dependencies: L1-03.

Contract and implementation: specify UTF-8 validity, byte-string distinction, indexing/slicing, allocation limits, normalization policy, error values, ABI scalar/layout rules, ownership transfer, null/length checks, and thread/lifetime restrictions. Update grammar/spec, checker, evaluator, MIR/native serial lowering, diagnostics, and independent host oracle. C calls cross a reviewed unsafe boundary with typed wrappers; foreign code remains outside Invar soundness claims. Unsupported ABI/layout rejects.

Positive cases: valid UTF-8, arbitrary bytes, empty/large-bounded values, explicit conversion errors, C-owned and Invar-owned buffers under the declared transfer rules, and reference/native agreement pass. Negative cases: invalid encoding, unterminated foreign input, overflowed lengths, use-after-transfer, borrowed return, wrong ABI layout, callback escape, and foreign failure reject or return typed error.

Acceptance interface and evidence: add `issue_l1_04_bytes_strings_cabi`; run `cargo test --locked issue_l1_04_bytes_strings_cabi` and require a nonzero test count (zero-test success fails). Record grammar/spec revision, encoding cases, ABI header/layout digest, wrapper invariants, oracle/native results, and platform gaps.

Exclusions/stop: no general string library, implicit NUL termination, arbitrary FFI, exceptions, or unsafe code in safe source. Stop when the narrow subset is complete and foreign failures remain explicit.

Bound lengths, conversions, call depth, foreign allocation, and diagnostic payloads. Do not claim the C ABI makes arbitrary native dependencies safe.

The ABI test fixture must exercise both success and every documented ownership transfer direction.

ABI failures remain typed environmental outcomes and never become host panics or silent truncation.
