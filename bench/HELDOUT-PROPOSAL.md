# M5-01 candidate held-out proposal

Status: **proposal-pending-review**.
This document and `heldout-v1-proposal.json` do not freeze anything.
The committed v1 manifest, old corpus, old held-out hash and pending v2 proposal retain their bytes.
No tuning or GPU job is part of this change.

## Contract, quoted verbatim

From `docs/issues/next/M5-01-benchmark-manifest.md`:

> A deterministic seeded generator produces at least 200 random eligible programs across every supported family, varying chain length 1 to 16, 1 to 4 inputs, 1 to 4 outputs and mixed operations.
> The generated programs are split into a tuning set and a held-out set.
> The held-out set's content hash is committed before any tuning work (M4-06) and is never edited.
> Per target, the held-out gate is: median bandwidth fraction at least (family gate - 0.05) and 10th percentile at least (family gate - 0.15).
> The report lists the worst 5 held-out programs with a profiler diagnosis for each.

> Exclusions/stop: no tuning, result cherry-picking, benchmark-name special cases, or universal performance claim.

From `docs/performance-acceptance.md`:

> The harness must include the following families.
> Exact random arrays use fixed seeds, plus independent adversarial inputs.
> CPU-sized correctness samples and larger performance samples are separate; the source interpreter need not execute billion-element cases to establish an oracle.

> Keep old results; never remove failing cases to pass a milestone.

Neither contract specifies a numeric held-out fraction.
**PROPOSAL:** retain one quarter of the distinct programs per family, the old generator's 25% ratio.
The split unit is a distinct program: sources that are identical up to the generated entry name (degenerate shapes and a few legacy scans repeat) form one unit and always share a split, so no held-out program is also a tuning program.
The first member of a unit in generation order carries the split; later identical members are recorded with `split: duplicate` and `duplicate_of`, outside both sets, so a degenerate workload is weighted once.
The candidate split reserves predeclared subtype units and selects the remaining units with a separate seeded shuffle.
It depends on neither correctness outcomes nor measurements.
All source-generation failures stop the command, with the family and ordinal; none are discarded or resampled.
The old modulo split is retained only in the unchanged old corpus.
The new split avoids its correlation between reduction-law parity and held-out membership.

## Generate and check

Run CPU commands with a private target directory and bounded memory, swap and runtime.

```sh
cargo run --locked --bin invar-heldout -- --write-proposal bench/heldout-v1-proposal.json
cargo run --locked --bin invar-heldout -- --check-proposal bench/heldout-v1-proposal.json
cargo test --locked --test issue_m5_01_heldout_generators
cargo test --locked --release --test issue_m5_01_heldout_generators
```

The write command exclusively creates a new file and refuses overwrite.
The check command requires byte equality with complete regeneration, including status, seed, sources, source hashes, split, shape proposals and held-out hash.
Input and generated output are bounded to 4 MiB.
The committed proposal stores only the 139 held-out sources and stays below 500 KiB.
All 608 programs, including the 431 tuning sources and the 38 recorded duplicates, are regenerated and validated before selecting the stored subset.
The candidate held-out SHA-256 is `d69a2607c312e9a608648143eade5eff6d912f51164f5eed4f411e74f95dfbbc`, and the proposal file SHA-256 is `602f5ea1f9141025d0b18e1fe027e9248aa182fe91c6f86da93416e6bb51c67b`.
There is no freeze command.
The generator uses the existing xorshift64 implementation and seed `0x5a17202609010001`.
Its nondefault-seed API exists for determinism and seed-sensitivity checks, not postmeasurement corpus selection.

Every generated source is parsed, checked and MIR verified.
Every positive oracle program compares all output elements, not just a checksum, against independently computed expectations and runs through the source reference evaluator.
New-family programs also compare reference and MIR results, original fuel accounting and cleanup observations.
The family source recognizer must accept each eligible oracle program; a refusal aborts generation.
Compaction and segmented CPU candidates, indexed gather/scatter/histogram CPU candidates, and the frontier CPU observer are separately compared with independent host results.
Float oracle samples use exact small integers, with separate ordered operations; they do not establish NaN, subnormal or signed-zero acceptance.
Three separate bounds-negative controls require source/MIR `INDEX_OUT_OF_BOUNDS` agreement and the typed CPU candidate bounds refusal.
A pointer-chasing control requires serial fallback and records its rejected obligation.
These controls are outside the eligible tuning/held-out denominator.

## Exact table shapes and proposed class sizes

The machine-readable `shape_proposals` is part of the proposal, never part of `manifest-v1.json`.
Exact values from the contract remain exact values.
Every value below for a class without exact sizes is **PROPOSED**, requiring review.
Performance recipes are separate from CPU oracle invocations; they are not claims that these large source invocations have executed or passed resource admission.
The original map/reduction/scan programs retain their three-element correctness invocations.
Additional witnesses actually execute every exact boundary size through 4093 on CPU, retaining the old corpus separately.
The added parameterized functions accept explicit shapes for later performance instantiation.

| Family | Exact contract sizes | PROPOSED unsized classes |
| --- | --- | --- |
| Map | 0, 1, 31, 32, 33, 255, 256, 257, 4093 | Powers of two through the budget, sweep exponents 10, 12, ..., 28 |
| Reduction | Empty and singleton | Odd 33; ragged 257; multi-block 4093 |
| Scan | Same boundary sizes | Long sequence 65537; nonidentity seed 7 |
| Compaction | 0%, 1%, 50%, 99%, 100% | 10000 elements |
| Transpose | 31x33, 127x257, 1024x1024 | Larger bounded 2048x1025; strided view 31x33, stride 2 |
| Stencil | No numeric dimensions in the table | Tiny 0x0, 1x1, 1x3; halo 2x2, 3x5; odd 31x33; wide/tall 3x257, 257x3; square 1024x1024; 1, 2, 3, 16 steps |
| Matrix | Square 256, 1024, 4096 where budget permits | Rectangular/odd 31x33x17; one-sized 1x1x1, 1x33x17, 31x1x17, 31x33x1; zero inner extent 31x0x17 |
| Segmented | No numeric sizes in the table | Empty lengths 0,0; uniform 4,4,4,4; skewed 0,1,2,13; dominant 0,0,15,1; ragged 3,5,9 |
| Gather/scatter | No numeric sizes in the table | 0, 1, 31, 33, 257, 4093; sorted/random/duplicate indices; injective scatter and modular histogram |
| Frontier | No numeric sizes in the table | Chains 1,5,16; stars 5,17; balanced 7,15; skewed 8,17; sparse larger 257 vertices; vertex-count round bound |

Oversized source or device invocations require explicit resource refusals, never substitution of a smaller measured shape.
The exact 1024x1024 transpose exceeds the current interpreter's one-million-element array limit.
The 4096-class matrix and larger proposed shapes are subject to the contract's memory budget.
The source recognizers and their metadata caps may impose additional limits; CPU sample acceptance does not override them.

## Review and eventual freeze

Review the 25% ratio, subtype split, exact recipes, proposed class sizes, seed and generator/source/oracle revisions.
After the M4-02..05 device paths merge, rerun all CPU checks and family recognition against the merged revision.
The merged workstreams now provide ordered affine-pair reduction (#115), per-segment map (#100) and explicit Tensor Core `tc-f16-f32-v1` contraction (#110), and the candidate includes all three forms.
Strided-view transpose stays unavailable, because the M4-02 recognizer (#104) admits direct row-major transpose only; it is never approximated by direct transpose.
Numeric adversarial and large-invocation acceptance remains separate and unexecuted.
The candidate is family-complete for the eligible oracle forms, with these broader contract gates pending.
Only then may review authorize a manifest version bump that binds the candidate content hash permanently before M4-06 tuning.
The existing `manifest-v2-proposal.json` remains pending its own review; this proposal neither adopts nor rejects it.
