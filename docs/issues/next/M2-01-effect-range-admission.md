# M2-01 — Dependence, effect, range analysis, and admission records

Problem/result: decide which verified regions may enter automatic optimization using explicit facts rather than optimistic metadata.

Dependencies: M1-05.

Contract and implementation: derive typed read/write footprints, ownership/loan state, effect order, trap/failure summary, loop ranges, termination state, physical alias assumptions, and cleanup requirements from one authoritative representation. Produce immutable versioned admission records with evidence category, rule revision, target, policy, and fallback. Unknown is ineligible; a false runtime guard selects serial/reference fallback. Any transformation declares invalidated facts.

Positive cases: independent total wrapping maps with provable bounds receive complete records; records survive serialization/version checks and select a fallback when guards are false. Negative cases: hidden aliasing, unknown calls, possible visible effects, unbounded ranges, failure-capable writes, stale source identity, and incomplete evidence are rejected.

Acceptance interface and evidence: add `issue_m2_01_admission_records`; run `cargo test --locked issue_m2_01_admission_records`. Record fact provenance, accepted/rejected reasons, mutation tests for every obligation, and fallback observations. Evidence must distinguish checked invariant, trusted rule, runtime guard, and unknown.

Exclusions/stop: no loop fusion, scheduler, GPU, or theorem/proof claims. Stop when records are deterministic, fail closed, and auditable for the M1 corpus.

Bound analysis iterations, range width, serialized record size, and generated candidates. Preserve an explicit unknown value when a bound cannot be established. Acceptance evidence must show that changing source identity or rule revision invalidates prior records rather than silently recomputing permission.

The record is an admission input, not a proof artifact; later code must still verify its assumptions at the boundary where they matter.

## Concrete integration requirements

Implement the explain JSON interface in docs/implementation-plan.md. Start with fixed counted-loop domains, affine one-dimensional accesses, explicit unknown calls, immutable read captures and injective fresh writes. Distinguish read/read sharing from read/write overlap. Hash immutable region/rule/numeric identities and invalidate facts after changes. A guard checks stable inputs before any side effect; a cost estimate cannot discharge a legality obligation.

Follow the shared acceptance interface in `docs/implementation-plan.md`: list and verify a nonzero test selection, run the issue suite in Debug and Release, and retain full milestone Cargo gates. Commands are required deliverables, not existing functionality in the planning checkout.
