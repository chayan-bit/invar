# Ordered affine-pair reduction

This source form implements the frozen reduction-table requirement: "Modular sum and a noncommutative associative pair composition".
M2-03 requires "left-then-right composition and identity for each sealed built-in law" and a "noncommutative affine-pair witness".
The all-family held-out generators identify the unavailable class as "ordered pair-reduction recognition".
This follows the existing `laws::Affine` contract rather than inventing an argmin or floating policy.

## Law and source

For wrapping `u32`, `(a,b) then (c,d) = (c*a,c*b+d)` modulo 2^32.
The identity is `(1,0)`.
The pair denotes `x -> a*x+b`; `then` applies the left transform before the right transform.
Associativity follows from function composition over the ring of integers modulo 2^32.
Direct expansion gives `(e*c*a, e*c*b+e*d+f)` under both parenthesizations.
The identity works on both sides.
The operation is noncommutative: `(2,3) then (5,7) = (10,22)`, while reversal gives `(10,17)`.
Finite tests corroborate this sealed algebra; they do not infer laws for user operators.

The ordinary source form uses two mutable scalar components and two read coefficient arrays.
See [`examples/pair-reduction.inv`](../examples/pair-reduction.inv).
The loop is zero-based with an invariant checked `u32` trip count.
Each step performs `a = wrapping_mul(aa[i],a)` and `b = wrapping_add(wrapping_mul(aa[i],b),bb[i])`.
Commuting operands inside wrapping add/multiply is legal; swapping the two state stores is legal because neither observes the other state.
Both updates must use the same multiplier array.
Read/read coefficient aliasing is legal.
Only this direct sealed form is recognized; temporary coefficient bindings and arbitrary tuples/operators retain source/MIR serial execution.
No user law declaration or finite-sample inference is trusted.

Both state components are read live at loop entry.
An empty fold preserves that seed, including the identity seed `(1,0)`.
The full two-component result is observable, for example by returning `[a,b]`.
There is no min/argmin selection or tie-breaking rule: repeated equal pairs still compose in source order.
No `f32` reduction is admitted, so NaN handling is inapplicable to this law.
Checked and floating operators retain the existing source numeric policy and serial path.
The explicit extraction API refuses non-lawful pair candidates with a typed `NUMERIC_POLICY` or `SHAPE` diagnostic.

## Execution and bounds

Verified MIR supplies the source recognition boundary.
A sealed `PairReduce` typed graph extracts scalar seed slots 0/1 and array slots 0/1.
Graph rule changes, stale proofs and malformed slots are refused.
CPU chunks fold privately from identity and merge in increasing source-range order.
Both live scalar seeds are validated before either component is published.
Bounds/fuel guards retain source-order failure and cleanup behavior on serial fallback.

CUDA and Metal reuse the existing ordered affine block summaries and propagation.
The final device observation writes both aggregate components into an exact two-word output.
The host composes the live pair seed with that completed aggregate using the same sealed law.
This costs one collective execution, not two scalar reductions.
Temporary lifetimes, barriers, allocation caps and completion fences reuse M3 checks.
CPU candidate length is bounded to 1,000,000 elements.
The existing device collective limit is 67,108,864 elements.
Larger or malformed device plans reject before allocation.
These are correctness bounds, not a performance or full general-GPU acceptance claim.

## Checks and held-out generator follow-up

The default test target is `issue_v01_pair_reduction`.
The `pair_reduction` modules of the `gpu_cuda` and `gpu_metal` targets require their backend features and are ignored unless explicitly selected.
The source reference, serial MIR and scheduled CPU/device observations compare both results, source fuel and cleanup at boundary sizes.
Independent widened-integer host expectations compare complete raw pairs through the maximum device length.
Operator-order and non-associative-admission mutations must fail their relevant checks.
Argmin tie-break and floating NaN mutations are inapplicable to this affine-only contract.

After this merges, the held-out generators should generate identity and nonidentity-seeded affine-pair sources, validate their full pair against an independent host oracle, and require `pair_reduction::extract` plus `Execution::PairReduction` admission on the merged revision.
It should move only ordered pair reduction out of the unavailable list, retain the other three missing forms, and regenerate the proposal sources/counts/hash through review.
Its frozen manifest and held-out entries must remain unchanged until the separate reviewed version bump.
This branch does not edit the held-out generators or authorize permanent freeze.
