# M5-03 - CUDA-class performance acceptance and report

Problem/result: measure the frozen qualified pipeline against competent independent CUDA baselines under the published performance protocol.

Dependencies: M5-02.

Contract and implementation: use the frozen M5-01 manifest and M5-02 qualified cases. Report generated and library-assisted tracks separately, plus device-resident steady state, host-to-host warm call, and cold process boundaries. Warm at least 10 times and collect at least 30 independent samples with fixed randomized ordering; report median, IQR, uncertainty, transfer bytes, launches, memory, and compilation cost. Apply the aspirational geometric-mean ≤1.20 and per-large-case ≤1.50 ratios only to included generated cases whose semantics and baseline contract match.

Positive cases: every required family has nonzero qualified cases, exact integer correctness, declared floating-policy correctness, complete report packets, and target-specific ratios. Negative cases: excluded slow cases, missing reports, zero/nonqualified counts, uncertainty crossing thresholds, incorrect output, unsafe lifetime, hidden special cases, silently relaxed numerics, or unavailable hardware leave acceptance incomplete. A missing GPU is pending, never a pass.

Acceptance interface and evidence: add `issue_m5_03_cuda_performance_acceptance`; run `cargo test --locked issue_m5_03_cuda_performance_acceptance` and require a nonzero test count (zero-test success fails), then the authorized M5-01 `--features cuda --require-gpu` benchmark command with the final report path. Record manifest/repository/generated-code digests, hardware/toolchain/driver, seeds, shape/layout, timing boundaries, baseline revisions, ratios, uncertainty, and regressions. Publish partial capability by family when a gate fails.

Exclusions/stop: no universal performance guarantee, benchmark-specific compiler hints, global clock changes, result deletion, or substitution of library-assisted results for generated-track acceptance. Complete only when every frozen correctness, per-case, aggregate and automatic-dispatch target passes with a reproducible review-ready report. A report recording incomplete acceptance is a checkpoint, not issue completion.

Bound variants to 32 per region, memory/workspace, tuning time, compilation time, and additional samples. Performance cannot override semantic or safety failure.

Amended 2026-10-14: the CUDA performance gates in this contract are measured on an NVIDIA RTX 5070 Ti (sm_120, driver 580.178.04) and, for Metal, on the Apple M4, replacing the earlier RTX A6000 target. No threshold value changes. Every fraction of peak is taken against a reference measured on that same GPU (device-to-device copy bandwidth, or the measured compute peak where this contract defines one), never a datasheet number.
The `--require-gpu` benchmark command and the final report therefore run on that RTX 5070 Ti and on the Apple M4, and the report records the actual model, compute capability, driver, toolkit and NVRTC versions of each.
