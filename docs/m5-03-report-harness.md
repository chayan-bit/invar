# M5-03 report preparation, no performance claims

The following acceptance paragraphs are quoted verbatim from the local issue contract and issue #10, which has no comments.
Each paragraph contains preparation and execution obligations; the classification identifies both without claiming completion.

| Verbatim acceptance criterion | Classification |
| --- | --- |
| Contract and implementation: use the frozen M5-01 manifest and M5-02 qualified cases. | (a) immutable-input consumption; (c) actual frozen M5-01 inputs and qualified M5-02 cases. |
| Report generated and library-assisted tracks separately, plus device-resident steady state, host-to-host warm call, and cold process boundaries. | (a) separate track/scope reports; (c) recorded values on real named targets. |
| Warm at least 10 times and collect at least 30 independent samples with fixed randomized ordering; report median, IQR, uncertainty, transfer bytes, launches, memory, and compilation cost. | (a) warmup/sample validation and statistics; (c) independent hardware sessions and collection. |
| Apply the aspirational geometric-mean ≤1.20 and per-large-case ≤1.50 ratios only to included generated cases whose semantics and baseline contract match. | (a) diagnostic geomean and per-case arithmetic, strengthened by the parity amendment; (c) measured eligible ratios. |
| Positive cases: every required family has nonzero qualified cases, exact integer correctness, declared floating-policy correctness, complete report packets, and target-specific ratios. | (a) nonzero counts and report checks; (b) source/device family adapters; (c) full numerical/hardware qualification. |
| Negative cases: excluded slow cases, missing reports, zero/nonqualified counts, uncertainty crossing thresholds, incorrect output, unsafe lifetime, hidden special cases, silently relaxed numerics, or unavailable hardware leave acceptance incomplete. | (a) fail-closed refusals; (b) real device fault/semantic adapters; (c) hardware safety and numerical evidence. |
| A missing GPU is pending, never a pass. | (a) missing-device rows and pending verdict; (c) real GPU availability. |
| Acceptance interface and evidence: add `issue_m5_03_cuda_performance_acceptance`; run `cargo test --locked issue_m5_03_cuda_performance_acceptance` and require a nonzero test count (zero-test success fails), then the authorized M5-01 `--features cuda --require-gpu` benchmark command with the final report path. | (a) named suite and nonzero CPU checks; (c) frozen authorized CUDA benchmark execution. |
| Record manifest/repository/generated-code digests, hardware/toolchain/driver, seeds, shape/layout, timing boundaries, baseline revisions, ratios, uncertainty, and regressions. | (a) provenance fields and ratio/statistical reports; (b) real generated-code packet adapters; (c) recorded hardware and baseline lineage. |
| Publish partial capability by family when a gate fails. | (a) per-family capability reports; (c) publication of actual target-specific outcomes. |
| Exclusions/stop: no universal performance guarantee, benchmark-specific compiler hints, global clock changes, result deletion, or substitution of library-assisted results for generated-track acceptance. | (a) track separation and incomplete verdicts; (c) enforcement of execution exclusions during collection. |
| Complete only when every frozen correctness, per-case, aggregate and automatic-dispatch target passes with a reproducible review-ready report. | (a) aggregate completion guard; (c) all frozen correctness, per-case and dispatch gates on hardware. |
| A report recording incomplete acceptance is a checkpoint, not issue completion. | (a) explicit checkpoint verdict; (c) issue completion remains dependent on real acceptance. |
| Bound variants to 32 per region, memory/workspace, tuning time, compilation time, and additional samples. | (a) budget validation; (b) actual variant/workspace counters; (c) frozen caps and measured compilation/tuning/sample budgets. |
| Performance cannot override semantic or safety failure. | (a) semantic/safety veto; (b) real qualified observations; (c) hardware safety acceptance. |

No clause forbids report code before the freeze.
The prerequisite is for qualification and issue completion.
M4-06 explicitly says: "The M5-01 held-out hash must be committed before any M4-06 tuning work starts, and tuning uses only the tuning set."
This lane performs no tuning or timing.

The parity amendment supersedes aggregate CUDA acceptance with every scored resident and warm case having median ratio <=1.00 and one-sided upper 95% bound <=1.05.
A point estimate above 1.00 fails; uncertainty above 1.05 is pending.
The historical geomean is reported as a diagnostic and cannot hide failed cases.
Existing library, bandwidth, compute, held-out, dispatch, cold and resource gates remain required.
The reviewed target amendment landed with the v0.1 performance gate measurements: CUDA uses RTX 5070 Ti (sm_120), and Metal uses Apple M4. Thresholds are unchanged; the M5-01 freeze must bind these target facts before collection.

The report plan is a separate proposed interface, not an M5-01 freeze.
The freeze owner must review and commit its digest, exact expected rows and session/bootstrap procedure before timing.
Existing parameterized-plan and single-session bench results remain pending.
The M5-02 CPU qualification has unavailable generated/library CUDA/Metal rows and is CPU preparation, not completed qualification.

## Offline execution interface

```sh
cargo test --locked issue_m5_03_cuda_performance_acceptance
cargo test --locked --release issue_m5_03_cuda_performance_acceptance
cargo run --locked --bin invar-performance-report -- \
  --manifest bench/manifest-v1.json \
  --plan frozen-report-plan.json \
  --results retained-session-packets.json \
  --report new-acceptance.json --markdown new-acceptance.md
```

The report command performs no timing or device access.
Exit 0 means complete acceptance; exit 2 means pending, failed, synthetic or malformed input.
Both output paths must be new, and existing evidence is never overwritten.
Inputs are bounded to 32 MiB each; expected rows to 20,000; sessions to 32; raw samples to 1,000 per series.
The JSON retains every input packet, baseline candidate and negative result.
The Markdown gives row verdicts and counts, with packet metrics and provenance in the paired JSON.

`plan.status` is `frozen` only after review, with `manifest_sha256`, `bootstrap_seed`, `statistics: paired-session-cluster-bootstrap-v1` and `expected` rows.
Each expected row binds `id`, `case_id`, named `target`, `scope` and `gate`.
Scopes are `device_resident`, `host_to_host` and `cold_process`.
Gates are `parity`, `library_ratio`, `tc_library_ratio`, `bandwidth`, `compute`, `held_out_median`, `held_out_p10`, `dispatch`, `dispatch_small`, `cold` and `resources`.
Resource rows additionally freeze `workspace_max`, `compile_ms_max` and `tuning_ms_max`; variant count is bounded to 32.
No resource cap is guessed from the future freeze.

The bundle carries `synthetic`, M5-02 `qualification` with manifest digest and explicit qualified/unavailable/failed counts, prerequisite statuses and `rows`.
Each result row has `id`, `track`, `status` and `sessions`.
Each session has a distinct `id`, `ordering_seed`, `measurement_protocol: randomized-interleaved-paired-blocks`, a complete saved `generated` invar-bench packet and `baselines` containing `{kind: handwritten|library, report: <saved packet>}`.
Record unavailable rows explicitly; omitting a row has the same denominator and unavailable outcome.
Library-assisted rows cannot satisfy generated-track gates.
Library candidates implementing a mismatched contract must be retained in a separate informational packet, not supplied as eligible denominators.
If any eligible candidate fails review, the row remains pending rather than silently ignoring that baseline.

Current bench packets lack final acceptance evidence and remain pending unchanged.
Future reviewed adapters must populate `environment` on every packet with exact named `host`, `device_id`, `compute_capability`, driver/toolchain, CPU/OS, power/clock policy, occupancy snapshot, temperature, memory preflight and runtime bounds.
Manifest targets bind the same host and device identity.
Generated results bind ordinary checked source, full output, qualification, safety, numerical policy, source/code/executable digests, reviewed clean build record, shape/layout, required bytes, seed, allocations, launches and compilation/cache/memory costs.
Baseline results additionally bind independent quality/semantic review, source revision and matched input/output digests and allocation/timing policies.
Cold packets require reviewed scope, matching boundary and fresh-process raw samples.

Ratios are ratios of pooled medians, with linear-interpolated quartiles and IQR.
The deterministic one-sided 95% percentile bound resamples paired independent process sessions in 4,096 draws using the frozen seed.
Launches inside a session are never counted as independent experiments.
At least three sessions, ten warmups and thirty positive finite raw samples per scope per session are mandatory.
This proposed procedure needs freeze-owner review before collection; its synthetic tests are not evidence that real sessions are independent.
For library gates use the fastest correct eligible measured candidate in each retained session; all candidates remain in the report.
The historical geometric mean is informational and never affects the acceptance verdict.

`generated_result.acceptance_metrics` records independently measured copy bandwidth (`copy_bytes_per_ms`) or numeric-policy-specific measured compute peak (`measured_peak_ops_per_ms`).
Bandwidth is required bytes / median milliseconds / copy bytes per millisecond; logical rereads and workspace are not credited.
Compute is frozen required operations / median milliseconds / measured matched peak.
Held-out fractions require the frozen held-out hash and five retained profiler diagnoses; linear-interpolated median and p10 are compared with the family floor minus 0.05 and 0.15.
Automatic dispatch is median automatic / min(median forced CPU, median forced GPU), with a 1.10 ceiling.
Small-dispatch overhead has a 0.100 ms ceiling.
Metric uncertainty uses conservative session extrema, which enclose the percentile session-bootstrap interval rather than averaging away a failed session.
Cold timing is reported separately without inventing a cold ratio ceiling.

## Remaining execution boundary

M5-01 must freeze and review the exact schemas, profitable bands, held-out inputs, expected row plan, statistics, caps, target facts and baseline lineage.
Merge the device families and integrate source-derived runtime-input adapters with M5-02's currently unavailable device rows.
Complete M4-06 and GNET prerequisites without exposing held-out configurations to tuning.
Run exact full-output/numerical/safety qualification first, then the authorized frozen M5-01 `--features cuda --require-gpu` command with a new final report path.
Collect resident, warm and cold packets on every frozen named target, with occupancy and independent sessions recorded, forced-engine crossover controls, bandwidth/peak calibration, held-out profiling and all compilation/resource costs.
Retain incomplete or failed reports, diagnose by family, and use only the frozen bounded follow-up sampling policy.
Target execution remains pending on the reviewed RTX 5070 Ti and Apple M4 targets; the timing runs are listed in [remaining work](remaining-work.md) and use only an exclusive GPU timing wrapper.
The timing wrapper records foreign GPU activity during each timed phase; a report prints that record next to every timing with its conditions (for example "shared GPU, max foreign util X%"), and calls a number exclusive-GPU only when the record shows no foreign activity.

A fractional or ratio gate within 0.05 of its threshold also requires at least ten consecutive-run metrics under `acceptance_metrics.consecutive_runs[gate]`.
Their minimum and median are retained; a failed run cannot be averaged away.
Missing near-threshold evidence leaves the row pending.

`bundle.library_assisted` separately retains library-assisted packets keyed by `target` and `case_id`, with their complete saved baseline `report`.
That track has its own reported/unavailable/pending denominator for every manifest case and target.
Reported library-assisted timings always have zero generated-track acceptance credit.

Frozen plan `baselines` pin case, target, kind, source SHA-256, executable SHA-256 and revision; swapping an unpinned reference is refused.
Each frozen manifest target also binds `backend: cuda|metal`, driver, toolchain, compiler flags and stream policy, matched exactly by each packet's environment.
Every timed packet carries `environment.gpu_foreign_activity` (`record`, `max_foreign_util_percent`, `foreign_processes`) from the timing wrapper; a packet without it is refused.
Each result row reports `gpu_conditions`: "exclusive GPU (no foreign activity recorded)" only when every packet records no foreign activity, otherwise "shared GPU, max foreign util X%", and the Markdown report prints it beside the row.
These fields are an explicit adapter/freeze input rather than inferred from current provisional reports.

Manifest cases bind their seed and `numeric_class` separately from the exact `numeric_policy`.
Section-10 coverage requires both strict FP32 and Tensor Core 4096x4096x4096 compute rows, and preflight-approved large f32 map/reassociated reduction/scan bandwidth rows.
`large_case_preflight: accepted` is the freeze owner's evidence that the mandated 64 Mi-element or largest-fitting shape was selected before scoring.
A u32 smoke or small matrix cannot close these gates.

For a frozen `performance_class: placement_control` only, a dispatch row may record a typed exception admitted by the plan's `admitted_typed_reasons`.
Every session requires accepted review, evidence digest and `placement_status: passed`.
Such a row is counted as `exception`, not `passed`, and has zero parity credit.
An unknown reason, a scored parity row or inconsistent session reasons remain pending.
