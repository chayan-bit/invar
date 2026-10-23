# Stable sort comparisons

Build on the authorized CPU host; run physical device binaries under that
host's GPU lock. No source allocation cap is changed. Every power from 16
through 26 uses `2^power` u32 keys plus original u32 IDs, seed
`0x05eeda11 ^ power`, one warmup, and five measured samples. Every output is
checked for permutation, exact key/ID mapping and stable equal-key IDs after
the timer; a full SHA-256 over little-endian key/ID pairs must agree everywhere.

```sh
cargo bench --locked -p invar-tools --bench sort --no-run
# Run the printed executable directly, not cargo's default bench arguments:
<sort-executable> cpu 26 5
<cuda-sort-executable> cuda 26 5
<metal-sort-executable> metal 26 5
cargo build --locked --release -p invar-metal --example sort-baseline
<sort-baseline-executable> 26 5
cargo run --locked --release -p invar-tools --example sort-report -- <benchmark-report-dir>
```

The CUDA build needs `--features cuda` and `INVAR_NVRTC_COMPILER` pointing to
the real `invar-nvrtc` helper; select matching NVRTC shared libraries. The Metal
build needs `--features metal` and a physical 32-wide SIMD Apple GPU. The
standalone Metal example lives in the existing Metal package and uses its
pinned objc2 dependencies; no second host framework is imported.

The CUB baseline is a foreign nvcc program, built against the pinned CCCL
headers in [NOTICE.md](NOTICE.md), `-O3 -std=c++17 -arch=sm_120`, a compatible
host C++ compiler, CUDA shared runtime and OpenSSL libcrypto. It takes `26 5`.
No CUB guard or NVRTC include restriction is suppressed.

CPU timers cover the sort call, including scratch allocation, excluding input
cloning, verification and the baseline's final unzip. Rust stable sort and
Rayon stable parallel sort use `(key,id)` pairs; Invar uses split key/payload
buffers and the existing four-worker scheduler. This layout difference is
explicit. Rayon is a development-only dependency.

GPU-event timers cover every sorting kernel, excluding compilation,
allocation, input/output copies and host verification. Invar additionally
records total call wall time including those costs. It compiles per call;
these are not cached, precompiled steady-state API timings. ArrowMetal
pipelines and CUB code are compiled outside their timed calls. Their host wall
boundaries differ, so no cross-baseline wall-time ratio is reported. Raw timing
samples retain noise rather than replacing it with an estimated throughput.

A source copy and its unstable-scatter mutant must use **different Cargo
target directories**. Relocated workspaces sharing a target can reuse stale
artifacts when old mtimes survive copying. Keep the independent comparison
oracle and verify an all-equal original-ID sentinel before broader cases.

C13/C14 and compute-shift rules: calibration id is `sort.sort_by_key.u32`.
New CPU builds/tests run without GPU exclusivity; physical CUDA runs
additionally hold an exclusive GPU lock. Each warmup
and measured call records one-minute host load before and after; both must
be under 8. Runs stop rather than accept contended samples. `sort-report`
rejects legacy/missing/at-least-8 receipts. Historical rows are preserved,
but cannot support current performance claims. Large sizes use these raw
bench binaries, never `invar run`; source allocation caps are unchanged.

Load receipts are part of `invar-sort-benchmark-v2`; legacy v1 timings are rejected.
