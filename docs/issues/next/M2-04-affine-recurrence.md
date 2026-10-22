# M2-04 — Compact affine recurrence inference

Problem/result: recognize one compact affine recurrence summary for eligible `u32` loops, reducing representation cost while retaining exact ordered semantics.

Dependencies: M2-03.

Contract and implementation: infer only the frozen recurrence shape and emit `(a,b)` summaries with modulus and identity explicitly typed. Check bounds, initialization, recurrence purity, no alias/effect/cleanup changes, and source-order observation requirements. Validate summary application against the reference for the complete bounded loop; reject if any coefficient, modulus, or state dependency is unknown. Keep a serial fallback and record inference rule/version.

Positive cases: full-width wrapping affine recurrences, zero iterations, one iteration, ragged lengths, and non-power-of-two counts match sequential states. Negative cases: checked arithmetic, floating values, non-affine updates, hidden calls, state-dependent effects, overflow policy mismatch, and reversed composition reject.

Acceptance interface and evidence: add `issue_m2_04_affine_recurrence`; run `cargo test --locked issue_m2_04_affine_recurrence`. Record inferred coefficients, proof inputs, state-by-state comparisons, operation counts, and rejected near-misses. Reuse the research probe as evidence, not as a compiler oracle.

Exclusions/stop: no general recurrence solver, symbolic theorem claims, scheduler, or GPU. Stop when the one recurrence family is compact, bounded, and fallback-correct.

Bound inference work and coefficient width, and retain the original loop for debugging and fallback. A compact summary is an admitted implementation candidate only after state observations and trap policy match; coefficient inference by itself is not evidence of legality.

Store rejected near misses with their reason codes so future inference changes cannot turn an unknown case into an unexplained admission.

Keep inference deterministic across debug and release builds.

## Concrete integration requirements

Infer coefficients from wrapping_add/wrapping_mul MIR of s=a_i*s+b_i; a_i and b_i must be independent of the evolving state. Support final state and every-prefix output separately with nonidentity seed. Reuse research/summary-laws as independent evidence, not production implementation/oracle shared code. Reject s*s, coefficients computed from s, checked arithmetic without a separate proof, and unknown effects. Compare all prefix states and linear-work/logarithmic-depth accounting.

Follow the shared acceptance interface in `docs/implementation-plan.md`: list and verify a nonzero test selection, run the issue suite in Debug and Release, and retain full milestone Cargo gates. Commands are required deliverables, not existing functionality in the planning checkout.
