# M5-01 - Rust benchmark manifest, independent library baselines, and anti-overfitting harness

Problem/result: define a reproducible benchmark harness before any performance claim or schedule tuning, with independent library baselines, a frozen workload manifest, a frozen held-out program set, and a size sweep, so that later M4 schedules are measured against NVIDIA and Apple libraries rather than against agent-written code.

Dependencies: M3-05, M4-01.

Order: M5-01 lands immediately after M4-01 and before M4-02; every later M4 issue measures its gate through this harness.

Contract and implementation: add a versioned manifest `bench/manifest-v1.json` covering every family in `docs/performance-acceptance.md`, required shapes, seeds, numeric policy, generated and library-assisted tracks, timing boundaries, pinned baseline versions and expected nonzero case counts. M5-01 creates and freezes the manifest; each later M4 issue appends its family and its independent baseline through a reviewed manifest version bump before its own gate runs, and never edits a frozen held-out entry. Add the new exact CPU-runnable interface:

```sh
cargo run --locked --bin invar-bench -- --manifest bench/manifest-v1.json --track generated --validate-only --report target/m5-manifest-validation.json
```

The validate-only command must reject missing/duplicate cases and write a versioned JSON report explicitly marked manifest validation with no GPU performance result. It must not pretend to run generated kernels on CPU. Baseline adapters are independent implementations: handwritten CUDA is an external comparison fixture, never compiler/runtime implementation; library-assisted baselines are reported separately from generated kernels. GPU execution uses the M3-established `cuda` and `metal` features and a distinct preflight command, not a silent CPU substitute.

## Independent library baselines

Use independent library baselines, never agent-written baselines, wherever a library exists.
CUDA baselines: CUB `DeviceReduce`, `DeviceScan`, `DeviceSelect` (compaction), `DeviceSegmentedReduce` and `DeviceHistogram`; Thrust only where CUB lacks the primitive; cuBLAS and CUTLASS for GEMM (the existing library-assisted track); the NVIDIA reference tiled shared-memory transpose for transpose.
Metal baselines: Metal Performance Shaders (`MPSMatrixMultiplication` and any MPS reduce or scan equivalents); where no Apple library exists, a reviewed handwritten Metal kernel pinned by content hash.
The existing agent-written CUDA fixtures remain only as a secondary "same-structure" comparison and never set a gate.

## Library-ratio gate

Generated median over the best independent baseline median is at most 1.25 for every family on the RTX 5070 Ti (sm_120), including scan versus CUB `DeviceScan`.
Matching CUB scan may require a single-pass decoupled look-back scan; that schedule is in scope for the family that needs it.

## Held-out programs

A deterministic seeded generator produces at least 200 random eligible programs across every supported family, varying chain length 1 to 16, 1 to 4 inputs, 1 to 4 outputs and mixed operations.
The generated programs are split into a tuning set and a held-out set.
The held-out set's content hash is committed before any tuning work (M4-06) and is never edited.
Per target, the held-out gate is: median bandwidth fraction at least (family gate - 0.05) and 10th percentile at least (family gate - 0.15).
The report lists the worst 5 held-out programs with a profiler diagnosis for each.

## Size sweep

Every family runs at 2^10, 2^12, ..., 2^28 elements.
Device-resident, host-to-host and cold-process times are reported separately at each size, together with the CPU/GPU crossover size.
Gate: automatic dispatch at each size is within 10 percent of the faster of forced-CPU and forced-GPU, or the program records the typed reason it cannot be.

## Devices

Gates must pass on the RTX 5070 Ti (sm_120) and on the Apple M4 (local Apple Metal).
An NVIDIA L4 run is optional.
There are no per-device hand-written constants: launch geometry comes from capability queries or the M4-06 tuner.

## Adversarial shapes

Every family covers n in {0, 1, 31, 33, primes just below and just above each power of two in the sweep}, long chains (16 stages), multi-output programs (4 outputs), two-input zips, and inputs whose length is not a multiple of the block or warp size.

## Evidence

Commit raw sample logs, the generator seed, the held-out hash and baseline library versions (CUDA toolkit, CUB, cuBLAS, CUTLASS commit, macOS and Metal version), with hostnames, IP addresses and private paths scrubbed.

Positive cases: a CPU-only run validates manifest schema, all required case IDs, exact small-input oracles, nonzero case counts, held-out hash, generator determinism and report fields. Negative cases: missing families, zero cases, altered seeds, an edited held-out entry or hash, wrong arithmetic policy, mismatched track, an agent-written baseline used as a gate, malformed report, unavailable CUDA or Metal, and baseline semantic mismatch fail or remain explicitly pending.

Acceptance interface and evidence: add `issue_m5_01_benchmark_manifest`; run `cargo test --locked issue_m5_01_benchmark_manifest` and require a nonzero test count (zero-test success fails), then run the exact command above. Record manifest digest, held-out hash, generator seed, source/generated baseline revisions, library versions, toolchain, case counts and report schema. GPU qualification is separately recorded with `cargo run --locked --features cuda --bin invar-bench -- --manifest bench/manifest-v1.json --track generated --require-gpu --report target/m5-generated-cuda.json` and the matching `--features metal` run; missing hardware is pending, never pass.

Done criteria: the manifest and held-out hash are committed and frozen; every family present at M5-01 time has a pinned independent library baseline on both targets; CPU validation is reproducible; the size sweep, held-out and library-ratio reports run on the RTX 5070 Ti and the Apple M4 with an explicit status per family.

Exclusions/stop: no tuning, result cherry-picking, benchmark-name special cases, or universal performance claim. Stop when the manifest is frozen, both tracks are distinct, CPU validation is reproducible, and GPU qualification has an explicit status.

Bound manifest size, input memory, report size, subprocess time, and baseline process lifetime. Preserve old reports after a manifest revision and require both sides to rerun.

## Baseline and target freeze before tuning

The manifest binds the named GPU models (RTX 5070 Ti sm_120 and the Apple M4), driver/toolkit/compiler flags, exact large-case shapes, numerical policies, workspace/launch/reuse policy and target ratios before M4-06 starts. Select targets through authorized preflight; do not choose them afterward based on favorable ratios. An optional L4 target is a separately versioned qualification, not a replacement for a failing one. Baseline acceptance requires independent correctness plus measured quality evidence: compare applicable semantically matched CUB/cuBLAS/CUTLASS/MPS implementations, use the fastest correct eligible reference, and preserve all timings. Where a vendor contract cannot match (for example strict ordered FP32), document the mismatch and a profiler/resource review against achievable bandwidth or arithmetic throughput, checking coalescing, materialization, occupancy and synchronization. Pin baseline source/configurations and record reviewer rationale before Invar schedule tuning; an intentionally weak or untuned control cannot establish CUDA-class performance. Same-schedule backend smoke controls remain separate from these competitive baselines.

Amended 2026-10-14: the CUDA performance gates in this contract are measured on an NVIDIA RTX 5070 Ti (sm_120, driver 580.178.04) and, for Metal, on the Apple M4, replacing the earlier RTX A6000 target. No threshold value changes. Every fraction of peak is taken against a reference measured on that same GPU (device-to-device copy bandwidth, or the measured compute peak where this contract defines one), never a datasheet number.
The `targets` entries of the frozen `bench/manifest-v1.json` still name the RTX A6000 (sm_86); replacing them needs a reviewed manifest version bump with both sides rerun, which this docs amendment does not perform.
