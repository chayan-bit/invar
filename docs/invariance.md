# Worker-count and batch-size invariance

This report is the evidence for #209: at C0 and C1, a result does not depend on how many CPU workers ran it, how the input was split between them, or how many other rows shared its batch.
It is checked, not claimed: `invar run --check-workers` compares every run with the reference evaluator and records what it compared, and `tests/numeric_tiers/worker_invariance.rs` repeats the comparison run by run on every engine the build has.
The rule is `docs/spec/invar-v0.2-axis-n.md` NA7.13; the oracle it checks is N0's (NA7.4).

## What is compared

For each listed worker count, the program runs on `cpu-parallel` and on the reference evaluator.
The runs must agree on the result bits, the trap category and its source position, and the fuel used.
A difference is the tool error `INVARIANCE_MISMATCH`, naming the first worker count that differs.
On agreement, the receipt's `invariance` field records:

| Field | Meaning |
| --- | --- |
| `engines` | `reference` and `cpu-parallel` |
| `workers` | the counts that ran, as given |
| `compared` | `result`, `trap` and `fuel` |
| `fuel_used` | the fuel every run used |
| `runs` | per count: regions published, worker threads joined, and the start index of each chunk of the last region the workers split |
| `levels` | the axis levels the program declares, as in `numeric_contracts.levels` of the explain receipt |

A trap or fuel limit that every run reproduces is also checked evidence, so the trap document carries `invariance` too.

## Example receipt

`demo/issue-209-invariance.inv` has an admitted wrapping map and reduction over 100,000 elements and a `#[region(contract = "f32-tree-v2")]` fold whose terms make the summation order visible (a strict fold returns 12, the fixed tree 14).

```text
$ invar run demo/issue-209-invariance.inv --engine cpu-parallel --check-workers 1,2,7,64 --fuel 100000000
Result: 14
Invariance: result, trap and fuel identical on the reference evaluator and on cpu-parallel at workers 1, 2, 7, 64
```

With `--format json`, the receipt (the 64-worker split shortened here) is:

```json
{
  "result": {"type": "i32", "value": 14},
  "invariance": {
    "engines": ["reference", "cpu-parallel"],
    "workers": [1, 2, 7, 64],
    "compared": ["result", "trap", "fuel"],
    "fuel_used": 1900188,
    "runs": [
      {"workers": 1, "published_regions": 2, "worker_joins": 0, "last_split": [0]},
      {"workers": 2, "published_regions": 2, "worker_joins": 4, "last_split": [0, 50000]},
      {"workers": 7, "published_regions": 2, "worker_joins": 14, "last_split": [0, 14286, 28572, 42858, 57144, 71430, 85716]},
      {"workers": 64, "published_regions": 2, "worker_joins": 128, "last_split": [0, 1563, 3126, "...", 98469]}
    ],
    "levels": {"c": ["c0", "c1"], "s": ["s0"], "n": ["n0"], "t": ["t0"]}
  }
}
```

## The test binary

`tests/numeric_tiers/worker_invariance.rs` (in the `numeric_tiers` binary) fails on any bit difference at C0 or C1:

- Worker counts 1, 2, 7 and 64 on `cpu-parallel`, the MIR evaluator and the reference evaluator, compared on result, trap and fuel, for seeded programs over each admitted plan family (wrapping map, wrapping sum and product, select loop, checked `i32` sum with an overflow trap) and each C0/C1 float fold (strict, `f32-tree-v2`, `f32-exact-v2`, `sum_tree`, `sum_exact`).
  Each program also runs one fuel step short of its need, so every engine must stop at the same limit or trap at the same place.
- The same programs through `invar run --check-workers 1,2,7,64`, whose receipt must validate against `schema/invar-diagnostics-v3.schema.json` and name exactly those counts, comparisons and C levels.
- With `--features native`, each program that returns also runs on `native-serial` and must return the same result.
  CI's native job runs only `tier_engines` from this binary, so this row is not yet CI evidence.
- Batch sizes 1, 8 and 1024 with the same per-row inputs: each row is an order-sensitive strict `f32` sum, and an admitted map runs over the batch; rows 0 and 0 to 7 give the same bits in every batch, on every engine and worker count, and the 1024-row batch is split across 7 workers.

## Contrast: expected variation

The claim is specific to Invar's admitted plans, not a property of parallel float sums.
The test computes one binary32 sum of 2^24 and 63 ones the way an engine that combines per-worker partial sums would: one partial per chunk, then their sum.
That sum takes different bits at different worker counts, and the test asserts that it does: this is expected variation, not a failure.
Invar's strict fold of the same terms keeps the source order and gives the same bits at every count.
The issue's original contrast, the same computation at C4, cannot run: C4 is now N4 (NA3.8), which this engine refuses with `CONTRACT_UNAVAILABLE` (NA7.11).

## Revision, host and toolchain

- Revision: the commit that adds this report, on top of `main` at `fe6cb92`.
- Host: aarch64 Linux (NixOS), 16 cores.
- Toolchain: rustc 1.95.0 (`rust-toolchain.toml`), cargo-nextest 0.9.146, debug profile.
- CI runs the same test binary in the default-feature suites on Linux (x86_64 and aarch64) and Windows.

## Holes

- CUDA and Metal are not in this binary: they need their hardware, and their suites (`tests/gpu_cuda`, `tests/gpu_metal`) already compare device bits with the reference per family.
  A device row of this report waits for a hardware run of the same programs.
- `last_split` records the last region the workers split; per-region splits would need the CPU scheduler to keep them.
