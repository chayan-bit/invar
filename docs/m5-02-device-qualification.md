# M5-02 device qualification adapters

Issue #34, stacked on M5-01 draft #121 with CPU draft #112 merged at
`fe2583e8`. Implementation: `38ac19f0`. This adds criterion (b) adapters;
it does not close #34, criterion (c), or unblock M5-03.

The harness prepares 379 bounded cases for each target across all ten manifest
families. It reuses the read-only M4 and pair-reduction corpora and the existing
independent baseline validators. It adds no family implementation, manifest,
calibration, placement or timing change.

Exact preflight requires Apple M4 / Apple9 / MSL 3.1, or NVIDIA GeForce RTX
5070 Ti / sm_120 / NVRTC 12.9. A different name or capability tuple refuses
before qualification. CUDA additionally requires an absolute existing NVRTC
helper. A missing feature, host, runner or numeric contract never qualifies.

Each row retains target, family, track, status, typed refusal, invocation
digests and output digest. Source rows compare result, trap, fuel and cleanup
exactly. Other family rows explicitly identify source-bound device-plan scope;
the rank-1 physical f32 corpus identifies verified-IR scope. Reached-source
f32 uses the existing M4-01 source publisher. Strict f32 follows D1, including
its NaN equivalence; all other non-TC words compare bit-exact. TC rows use the
unchanged D5 exact-sum bound through `invar_device::tc::verify`.

The corpus includes empty and zero-work cases. Their exact boundary results
do not establish a kernel launch. Source cases that publish no region remain
unavailable. Subnormal CPU fallbacks remain unavailable device rows, with
their reason retained. Qualified rows are bounded case results, not complete
family or frozen-manifest acceptance.

The qualification CLI accepts `--device cuda|metal`, `--nvrtc ABSOLUTE_PATH`,
and `--libraries ABSOLUTE_DIRECTORY`. Runner names are `metal-u32`,
`metal-reduce`, `metal-affine`, `metal-gemm`, `metal-families`, `cuda-cub`, and
`cuda-gemm`; these are compiled from the already committed baseline fixtures.
Independent fixture inputs are checked in full by their original validators;
they cannot qualify a different source invocation or subtype. Handwritten
controls retain their kind. The existing matrix baseline's provisional
tolerance stays unavailable, never promoted to strict-f32 or D5 acceptance.

Reports publish only after all requested comparisons complete, with exclusive
creation. A disagreement, unexpected refusal or `--require-gpu` failure
publishes no report and preserves earlier reports. The GPU requirement keeps
every family/target/track in its denominator and refuses pending prerequisite
freeze/full-shape gates even if all bounded rows pass.

CPU checks run with bounded memory and runtime on a Linux host, and Metal
hardware checks run on the Apple M4, one command at a time. CUDA hardware qualification runs only on the pinned RTX 5070 Ti;
any other device is refused with `M5_PREFLIGHT_MISMATCH` before any work.

The original device receipts (#125, 2026-10-14) are not retained in this repository.
Criterion (c), CUDA hardware, full frozen shapes, maximum-size physical sweeps,
independent library coverage of every subtype and hardware sanitizers remain
pending. Device-plan success does not establish reached-source publication
for every later family; those interfaces remain with their family and M4-06 work.
