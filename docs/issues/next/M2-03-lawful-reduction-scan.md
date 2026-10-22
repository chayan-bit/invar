# M2-03 — Lawful reduction and ordered scan algebra

Problem/result: add only algebraically justified modular reductions and ordered scans without reassociating checked source arithmetic.

Dependencies: M2-02.

Contract and implementation: freeze built-in identities and operation laws for wrapping `u32` reductions and ordered affine-prefix scans. Preserve operand order unless a law explicitly permits reassociation; represent scan summaries with the proven ordered composition contract. Require total, nontrapping, effect-free regions and exact empty/singleton/odd/ragged behavior. Keep checked arithmetic and floating-point regions on serial fallback.

Positive cases: modular sum/product identities, the bounded affine scan, and all supported lengths match the reference and serial extracted plans. Negative cases: checked signed overflow regrouping, strict floating reassociation, noncommutative operand reversal, wrong identity, aliasing, and failed proof/guard reject or fall back.

Acceptance interface and evidence: add `issue_m2_03_lawful_reduction_scan`; run `cargo test --locked issue_m2_03_lawful_reduction_scan`. Include exhaustive bounded algebra checks, deterministic thousands-case differential results, combine count/depth evidence, and the existing summary-law witness. Finite tests establish neither universal algebra nor GPU speedup.

Exclusions/stop: no arbitrary user law inference, floating fast-math, scheduler, or device lowering. Stop when named built-ins have source-order and fallback evidence.

Pin the algebra rule revision and identity in each candidate record. Empty inputs, neutral-only inputs, and the smallest nontrivial shapes must be separate fixtures because an apparently valid law can still mishandle initialization or publication order.

The reference result remains authoritative for completed runs, and unsupported numeric policies retain the serial path.

Every candidate records its identity, operation order, empty-input behavior, and fallback reason. The implementation must not infer commutativity from observed equal outputs, and it must not substitute a faster tree shape when the source contract requires ordered evaluation or trap provenance.

## Concrete integration requirements

Define left-then-right composition and identity for each sealed built-in law. Implement an ordered work-efficient tree, not a quadratic prefix implementation or commutative-only tree. Test empty/singleton/odd/ragged sizes, identity padding and a noncommutative affine-pair witness. Checked i32 sums and strict floating sums must not borrow the modular law. Source sequential loops remain the public input; no user proof flag is trusted.

Follow the shared acceptance interface in `docs/implementation-plan.md`: list and verify a nonzero test selection, run the issue suite in Debug and Release, and retain full milestone Cargo gates. Commands are required deliverables, not existing functionality in the planning checkout.
