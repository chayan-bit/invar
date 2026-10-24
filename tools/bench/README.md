# Provisional M5-01 benchmark inputs

`manifest-v1.json` and `corpus-v1.json` are committed preparation from the
pre-M4-01 u32 compiler. They are **not** the final M5-01 freeze. The verifier
regenerates all 240 programs, parses/checks/lowers them, checks every loop's
admitted plan, runs exact small-input oracles, and compares the full JSON data.
The 60 held-out programs have a separate SHA-256 digest in the manifest.

The current v1 revision is still provisional (`freeze_status` unchanged).
It appends the seven remaining families of `docs/performance-acceptance.md` (compaction, transpose, stencil, matrix, segmented, gather/scatter, frontier).
Their cases record only the shapes of that frozen table, and every case, baseline (`baselines.cuda|metal.<family>`) and per-family target (`provisional_families.<family>.targets`) is `unavailable` with a stated reason.
The validator refuses any such entry as a measured result: `--generated-case` on an unavailable case and `--associate-generated` with one both fail.
`held_out_sha256` is unchanged; `held_out_freeze` records that the permanent held-out freeze happens after the M4-02..M4-05 device paths exist, by a reviewed version bump before M4-06 tuning.
`manifest-v2-proposal.json` stays pending M5 review: it is derived from the map/reduction/scan base only and is neither adopted nor rejected by this revision.

```sh
cargo run --locked -p invar-tools --bin invar-bench -- --manifest tools/bench/manifest-v1.json --track generated --validate-only --report target/m5-manifest-validation.json
```

The report's kind is `manifest_validation`; its `gpu_performance` is `not_run`.
The separately versioned `manifest-v2-proposal.json` also validates with the
same command and writes report schema 2. It changes reduction required bytes to
`4n+4` and pins a separate MPSGraph reduction fixture using the generated u32
input recipe and seed 17. It remains provisional. `--write-v2-proposal` creates
it from a validated v1 manifest at a new path and refuses overwrite. The one
matched Metal n33 run's reports are kept in
`tools/tests/fixtures/archived-evidence/parallel-benchmark-baselines/`; the ratio gates
remain pending. V2 baseline execution requires `--baseline-build-record`; saved
v2 report association requires `--generated-build-record`. Both records are
checked against report hashes and retain their unreviewed status.
The v2 reduction baseline parent also runs 10 warmup and 30 measured fresh
child processes. Each child uses one library call, writes a full-output-checked
report, and leaves a status and stderr artifact. Its parent-wall boundary is
reported separately from the fixture's first-call timer. A matching generated
parent-wall label does not grant cold ratio eligibility: process workload scope
still needs M5 review.
The clean n33 cold-process pair's raw reports are in
`tools/tests/fixtures/archived-evidence/parallel-benchmark-baselines/v2-clean-n33-25c5083/`.
Generated timing currently measures a parameterized device plan extracted from a
checked three-element corpus source. The measured length, LCG inputs and collective seed
are separate from that source's literal invocation. New reports identify the
corpus hash as a template hash and bind the actual plan invocation to its own
digest; exact output checks cover that invocation. Source-program acceptance
and all ratios remain pending until a matching ordinary source path and cold
workload contract are reviewed. Every report, including validation and
preflight, uses exclusive creation: choose a new path for each run so earlier
reports and raw attempts are preserved. `--write-artifacts` also requires both
corpus and manifest output paths to be new; regenerate into a fresh directory.
Every report records the source checkout's Git HEAD and whether that worktree
was clean when the command started; unavailable Git state is explicit. This
invocation state alone does not prove which commit built the benchmark binary.
Read-only GPU availability has a separate command and never times a kernel:

```sh
cargo run --locked --features metal -p invar-tools --bin invar-bench -- --manifest tools/bench/manifest-v1.json --track generated --preflight --report target/m5-metal-preflight.json
```

Run the same command with `--features cuda` on an authorized CUDA host. A
missing backend feature or device writes a pending preflight report and exits
nonzero; an available device writes `status: available` with
`gpu_performance: not_run`. This checks device access only; target toolkit,
occupancy, memory headroom, and site policy still need review before a job.
`--require-gpu` without an available generated adapter writes a pending report
and exits nonzero. An existing report path is refused regardless of manifest
digest, so prior results survive repeated runs and manifest revisions.
`--write-artifacts` deterministically creates provisional files at fresh paths
and must not be used to edit a final frozen
held-out set.

On an Apple GPU, a separate forced-Metal correctness smoke runs all 60 held-out
source programs against the independent reference evaluator. It records exact
result/step/cleanup agreement and confirms a GPU region was executed; it does
not time kernels or satisfy a performance gate:

```sh
cargo run --locked --features metal -p invar-tools --bin invar-bench -- --manifest tools/bench/manifest-v1.json --track generated --require-gpu --metal-held-out-smoke --report target/m5-metal-held-out-smoke.json
```

## External library comparisons

The Rust harness bounds each baseline process to 30 seconds, reads at most
4 MiB of output and 64 KiB of logs, requires 10 warmups and 30 raw samples,
checks byte accounting, and compares the entire u32 output with an independent
host oracle. Timing summaries include median, IQR, and a two-sided exact
order-statistic interval for the median (at least 95% coverage under independent
continuous samples). A baseline report never becomes a generated-track ratio.

CUDA fixture (run only on a host where uploading this repository source is
authorized):

```sh
nvcc -std=c++17 -O2 -arch=sm_86 tools/bench/baselines/cuda-cub.cu -o cuda-cub
cargo run --locked -p invar-tools --bin invar-bench -- --manifest tools/bench/manifest-v1.json --track library-assisted --require-gpu --baseline-executable ./cuda-cub --baseline-family affine --baseline-elements 4093 --report target/cuda-cub-affine-n4093.json
```

The fixture calls CUB `DeviceReduce` and `DeviceScan` for u32 and f32,
including a noncommutative affine operator. For maps it uses CUB
`DeviceTransform` when present (CUDA Toolkit 12.9 or newer), and Thrust
`transform` on older supported toolkits. The f32 output is compared against
an ordered host loop, but the arithmetic-compatibility verdict remains
pending M4-01. Baseline source SHA-256 values are in the manifest; an
executable's build provenance and vendor/toolkit pin still require review.
The CUB fixture reports its allocated vendor temporary bytes and explicit
intermediate bytes separately; `total_scratch_bytes` and `workspace_bytes`
are their checked sum. The affine scan's extra pair buffer is included.
For cuBLAS and MPS, vendor-internal scratch is unknown, so the Rust report
keeps `total_scratch_bytes` null and labels `workspace_bytes` as explicit
fixture scratch only. The earlier L4 raw reports retain their original source
hash and byte fields; the corrected fixture has not been rerun on a GPU.
Each baseline report names a candidate manifest case where its family, law,
and size match. This is a review link, not a gate result. The plain modular
sum scan and current f32 comparisons have no matching manifest case.
All baseline runs currently exit nonzero after writing their measurement report
because semantic, baseline-quality, and build-provenance review is pending.

The external cuBLAS FP32 GEMM path uses pedantic math mode, but cuBLAS still
reassociates or fuses dot products relative to strict ordered source loops:

```sh
nvcc -std=c++17 -O2 -arch=sm_86 tools/bench/baselines/cuda-cublas-gemm.cu -lcublas -o cuda-cublas-gemm
cargo run --locked -p invar-tools --bin invar-bench -- --manifest tools/bench/manifest-v1.json --track library-assisted --require-gpu --baseline-executable ./cuda-cublas-gemm --baseline-family f32gemm --baseline-elements 16 --report target/cuda-cublas-f32gemm-n16.json
```

Local Apple MPS comparison:

```sh
xcrun clang++ -std=c++17 -O2 -fobjc-arc tools/bench/baselines/metal-mpsgraph-u32.mm -framework Foundation -framework Metal -framework MetalPerformanceShaders -framework MetalPerformanceShadersGraph -o metal-mpsgraph-u32
cargo run --locked --features metal -p invar-tools --bin invar-bench -- --manifest tools/bench/manifest-v1.json --track library-assisted --require-gpu --baseline-executable ./metal-mpsgraph-u32 --baseline-family map --baseline-elements 4093 --report target/metal-mpsgraph-map-n4093.json
cargo run --locked --features metal -p invar-tools --bin invar-bench -- --manifest tools/bench/manifest-v1.json --track library-assisted --require-gpu --baseline-executable ./metal-mpsgraph-u32 --baseline-family sum --baseline-elements 1048576 --report target/metal-mpsgraph-sum-n1048576.json
cargo run --locked --features metal -p invar-tools --bin invar-bench -- --manifest tools/bench/manifest-v1.json --track library-assisted --require-gpu --baseline-executable ./metal-mpsgraph-u32 --baseline-family affine --baseline-elements 4093 --report target/metal-mpsgraph-affine-n4093.json
```

MPSGraph `UInt32` elementwise map and reduction are independent Apple library
comparisons with full-output checks. The affine prefix graph composes pairs
with MPSGraph operators in logarithmic stages and matches the manifest law;
its quality against a competent handwritten scan remains unreviewed. Its
native cumulative sum supports modular sum scan, which is outside the affine
manifest law. Framework-internal workspace use and baseline quality remain
unreviewed, so these measurements do not set a gate.

A separate handwritten Metal affine scan supplies a quality control for that
review. It scans 256-element blocks, scans block totals recursively, then
applies preceding-block prefixes and the nonidentity seed. The harness checks
the entire u32 output and validates its explicit workspace and launch counts:

```sh
xcrun clang++ -std=c++17 -O2 -fobjc-arc tools/bench/baselines/metal-affine-handwritten.mm -framework Foundation -framework Metal -o metal-affine-handwritten
cargo run --locked --features metal -p invar-tools --bin invar-bench -- --manifest tools/bench/manifest-v1.json --track library-assisted --require-gpu --baseline-executable ./metal-affine-handwritten --baseline-family affine --baseline-elements 4093 --report target/metal-affine-handwritten-n4093.json
```

Its report kind is `handwritten_quality_control` and gate status is
`quality_control_only`. Separate, noninterleaved sample runs do not establish
which implementation is the fastest correct eligible baseline.

```sh
xcrun clang++ -std=c++17 -O2 -fobjc-arc tools/bench/baselines/metal-mps-gemm.mm -framework Foundation -framework Metal -framework MetalPerformanceShaders -o metal-mps-gemm
cargo run --locked --features metal -p invar-tools --bin invar-bench -- --manifest tools/bench/manifest-v1.json --track library-assisted --require-gpu --baseline-executable ./metal-mps-gemm --baseline-family f32gemm --baseline-elements 16 --report target/metal-mps-f32gemm-n16.json
```

MPS GEMM is an external f32 baseline; strict ordered f32 acceptance is pending
M4-01. MPSImageReduceRowSum and MPSImageIntegral provide external f32
reduction and scan comparisons:

```sh
xcrun clang++ -std=c++17 -O2 -fobjc-arc tools/bench/baselines/metal-mps-image.mm -framework Foundation -framework Metal -framework MetalPerformanceShaders -o metal-mps-image
cargo run --locked --features metal -p invar-tools --bin invar-bench -- --manifest tools/bench/manifest-v1.json --track library-assisted --require-gpu --baseline-executable ./metal-mps-image --baseline-family f32sum --baseline-elements 33 --report target/metal-mps-f32sum-n33.json
cargo run --locked --features metal -p invar-tools --bin invar-bench -- --manifest tools/bench/manifest-v1.json --track library-assisted --require-gpu --baseline-executable ./metal-mps-image --baseline-family f32scan --baseline-elements 33 --report target/metal-mps-f32scan-n33.json
```

These float baselines do not establish a u32 comparison. The MPSGraph u32
baseline review remains pending.

## All-family held-out candidate, not frozen

`invar-heldout` creates and verifies `heldout-v1-proposal.json` separately from the v1 manifest and old corpus.
The proposal carries all-family CPU oracle programs, per-family generated/valid/recognized counts, its own candidate content hash, a proposed 25% split and proposed sizes for unsized classes.
It has `status: proposal-pending-review` and no freeze interface.
See [the contract quotes, sizes and remaining review gates](HELDOUT-PROPOSAL.md).
Neither the v1 frozen-status field and held-out hash nor the pending v2 proposal changes.

The candidate also generates the forms that became available after the all-family held-out generators: ordered affine pair reduction, per-segment map and the explicit Tensor Core `tc_mul_add` contract with its strict twin (M4-03).
They use their own PRNG stream, so the earlier programs keep their bytes and order.
`HeldOutIndex::from_programs` builds an exclusion index from the held-out programs, and `HeldOutIndex::admit` refuses a tuning or calibration source that is, or embeds, a held-out source exactly, after renaming or after reformatting, or that carries a held-out input array.
Every tuning path must pass its candidate programs through `admit` before use.
## Family baseline fixtures

`baselines/metal-families.mm` and `baselines/cuda-families.cu` are the independent family baselines for transpose, gather, stencil, segmented sum, stable compaction and frontier. The CUDA fixture also covers the `*_lcg` map, reduction and scan families.
A vendor library call reports `independent_library_baseline`: Apple MPSGraph, or NVIDIA CUB and Thrust.
A ported vendor sample or a handwritten kernel reports `reviewed_handwritten_baseline`, pinned by the fixture source hash. One example is the NVIDIA tiled transpose port.
CUDA has no separate handwritten gather, because Thrust gather is the contract's library.

`invar-bench --track library-assisted --baseline-executable FIXTURE --baseline-family FAMILY --baseline-elements N` runs one fixture as an untrusted child process:
- the child is bounded to 60 seconds;
- every protocol field is validated (schema 3, 10 warmups, 30 samples, the input recipe and the vendor versions);
- the entire output is compared with the independent Rust references in `family_references.rs`.

A single wrong word makes the run `exact_full_output_fail`, and no timing from it is used.
MPSGraph gather on `UInt32` rounds values through float32 and fails exact output. It is therefore ineligible, and `gather_hw` is the Metal gather baseline.

```sh
xcrun clang++ -std=c++17 -O2 -fobjc-arc tools/bench/baselines/metal-families.mm -framework Foundation -framework Metal -framework MetalPerformanceShaders -framework MetalPerformanceShadersGraph -o metal-families
nvcc -std=c++17 -O2 -arch=sm_XX tools/bench/baselines/cuda-families.cu -o cuda-families
cargo run --locked --release -p invar-tools --bin invar-bench -- --manifest tools/bench/manifest-v1.json --track library-assisted --require-gpu --baseline-executable ./FIXTURE --baseline-family stencil --baseline-elements 65536 --report NEW_REPORT.json
```

Timings from these fixtures are baseline samples only. Performance runs and their measurement conditions are tracked in the M5 GitHub issues.
