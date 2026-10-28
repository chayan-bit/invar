# Placement profile v1

`invar run PATH --engine auto [--profile PROFILE] [--host-class CLASS] [--trust-provisional]`
and `invar explain PATH --engine auto --format json` consume the same placement
receipt. Explain accepts the same placement options, plus `--artifact` for Metal.
The implicit path is `calibration/<host-class>.json`; the default class is
`<os>-<arch>`, matching calibrate. Profile identity must exactly match the live
host fingerprint, CPU, device, driver, compiler build and linked LLVM version.
Malformed or missing explicit files are errors. A valid profile for a different
identity is refused with `TUNING_HINT_INVALID` and retains CPU execution.
Unknown frozen profiles are refused with `TUNING_FREEZE_REQUIRED` until M5-01
provides an accepted manifest. Provisional profiles influence placement only
with `--trust-provisional`; D1=(a) therefore retains the CPU default.

The only family placement prices is the single-input wrapping-u32 add-one map.
Native calibration also records CPU rows for the nine other M5-01 table families (#49), under their own family IDs; no selection reads those rows, so they never change placement.
Other laws, fused variants, float contracts and sizes outside the recorded interval
have unknown costs and remain CPU. Matching CPU maps use the same compiled LLVM
host-array export as calibration; the surrounding program uses checked MIR control
and the existing atomic publication/fuel/cleanup guards. With no matching trusted
profile, native-enabled builds run the whole program with LLVM; builds without native
retain checked MIR CPU execution and cannot admit a measured device choice.
Metal uses its matching compiled-ahead `--artifact` by default; the existing explicit
`INVAR_METAL_RUNTIME_SOURCE=1` developer opt-in remains available for debugging.
Ordinary auto never invokes calibration. The hardware CLI test removes the developer
opt-in and arms the source-compiler tripwire. Existing forced engines retain their checks.

Versioned JSON run receipts use `invar-diagnostics-v3` and carry `placement`: profile id/schema version, resource identity,
trust, the engine that actually published, and reached-loop decisions. Each decision
records family/numeric contract and region identity, page/residency observation,
verified byte/work/launch facts, fixed and work costs, page-in, transfers, launch and
total costs, 5% margin, legality and stable refusal codes. Unknown costs are null.
Auto explain (`invar-explain-v5`) includes that same object, and retains the established `device_execution`
schema alongside it. CPU replay is permitted only after confirmed device quiescence;
unknown completion remains a tool error. Rejected publication records CPU and
`TUNING_GUARD_FALSE` in the receipt.

`invar calibrate [--out NEW_PATH] [--family placement.wrapping_add1.u32] [--host-class CLASS]` writes a new
profile. The default path is `calibration/<os>-<arch>.json`; its parent must exist.
Output paths must be UTF-8 so the JSON receipt can represent them; rejection happens
before measurement or publication. Duplicate/unknown options and unknown families are errors. Output is one JSON object;
exit 0 means a valid provisional profile was published, including explicit refusals.
It does not mean every engine was measured or any performance gate passed.
Existing paths, including symlinks, are never replaced. Publication links a fully
written and synced same-directory temporary file, then removes that temporary name.

The JSON schema is `schema/invar-placement-v1.schema.json`. Files are limited to
4 MiB (M5-01, #49; earlier builds capped them at 1 MiB), 128 measurement rows, 16 increasing sample sizes per row and exactly five
passes. Unknown schema/probe versions, nonfinite/negative figures, zero bandwidth,
summaries inconsistent with the five passes are errors. Identity mismatches are typed loader refusals; ordinary auto retains CPU and records the refusal.
Missing coefficients are unknown. Frozen profiles are refused until M5-01 provides
a trusted freeze manifest. Absence of an implicit default is a CPU fallback;
an explicitly supplied invalid/missing file is an error.

A profile may declare the M5-01 measurement protocol (#49) with a top-level `statistics` block, which must equal the loader's (10 warmups, 30 samples, median of order statistics 15 and 16, 95% interval from order statistics 10 and 21, quartiles from 8 and 23), together with `coverage`.
Every row then carries `table_family` and `points`.
Its table family must be the one registered for its probe family ID, numeric contract, engine boundary and family-level region.
Its points are exactly one cold `first_call` (one sample, null elements) plus 30 integer-nanosecond samples for every row size.
Each summary must equal the one recomputed from its samples.
`coverage` lists the ten frozen-table families, and every (engine, family) pair is either measured or pending, exactly once.
Declaring only part of the protocol is an error; profiles without it load as before.
Only a profile loaded against a committed SHA-256 (size, then digest, before JSON) can claim a calibrated family, because recomputation does not prove that `passes` and `best` came from the samples.

A coefficient row binds family, numeric contract and execution boundary; optional
region rows additionally bind source identity, graph revision and region identity.
Under C14 (robotics-language roadmap section 4.4, `a41b6c7f`), family IDs have exactly the form
`<lane>.<op>.<type>`; this lane's probe is `placement.wrapping_add1.u32`.
The old `map` CLI filter and `map-wrapping-add1` profile ID are rejected.
Under C4, every row and decision receipt records `numeric_contract_version`, and
matching requires that version as well as the numeric domain. This integer-only
probe reuses the existing integer explain policy ID `wrapping-u32-v1`; it does
not measure floating math or claim the
`invar-math-cr-v1` operations. Future float rows must carry their
owning lane's actual version.
Region rows take precedence. Selection stays within the recorded size interval.
The scalar work unit of the current probe is one wrapping-add-one map element.
Its data cannot price other maps, reductions, scans or new intrinsics.
The M5-01 family rows (`placement.wrapping_sum.u32`, `placement.affine_scan.u32`, `placement.select_below_half.u32`, `placement.transpose.u32`, `placement.five_point_stencil.u32`, `placement.square_matmul.u32`, `placement.segmented_sum16.u32`, `placement.permuted_gather.u32`, `placement.dense_bfs16.u32`) count one input element per work unit and describe only their own probe kernel.

The selector compares legal engines using fixed plus per-work costs, transfer
bandwidth/latency, actual launch counts, capacity and actual input/output residency.
The device wins only for `device_ms < 0.95 * cpu_ms`; equality retains CPU.
Equal device totals use CUDA then Metal. Unknown CPU cost retains CPU. The receipt
is the decision's JSON object; unknown figures are null. It is not an execution or
publication permit. Callers must still use verified plans and completion guards.
The same decision object is the explain payload.
Its reason codes, and calibration refusal codes, are registered in AI-01's
`schema/diagnostic-codes-v1.json`. Probe failure details are separate from their
stable `calibration-probe-failed` reason.

Host pages are observed through the borrowed live slice without first reading it.
Linux/macOS mincore queries align and round with checked arithmetic and bounded
metadata. Only the resident bit is read. Failure/unsupported platforms return unknown.
The observation is racy and affects costs only. Pageable inputs require independently
measured page-in costs for both CPU reads and device uploads; cache eviction supplies
no page-in measurement.

With `native`, the CPU probe compiles the ordinary map through LLVM host-array
exports outside the five steady-state passes. Each of the nine other M5-01 families gets its own LLVM host-array export under the same protocol, and every timed output must equal an independent Rust reference; a mismatch refuses that family, which stays pending. Calls include their real private host-array
protocol and rollback snapshot. Every output word is checked. Without `native`, CPU
calibration is refused explicitly. Device probes run the corresponding verified
wrapping map, warm compilation outside timing, check every word (including wrapping extrema) and retain five
host-to-host sample pairs. The device probes also measure the n=1 shape and retain kernel and transfer phases.
CUDA uses its existing ordered event clocks. Metal measures the actual host copies
into and out of shared buffers, with allocation outside those copy clocks. The
remaining API wall time remains in compute overhead. A nonnegative envelope covers
all three sizes in each pass: the minimal real launch/copy supplies the fixed floors,
and the largest observed incremental rate supplies the slope. This conservatively
counts the minimal probe's tiny work twice. It is a bounded provisional model, not
a fitted claim outside the measured interval. Unknown/inconsistent phases or an
unresolved positive transfer slope produce a refusal, with raw samples retained.
With `cuda` or `metal` (macOS), each of the nine other families also runs as an ordinary-source `main` on that device (boundary `source-forced-device-v1`); its row counts only when the device ran a region inside the family kernel and every output word equals the Rust reference (embedded in the source) on every call. A family whose kernel admission plans serially stays pending as `device-plan-ineligible`; any other failure leaves it pending as `calibration-probe-failed`.
`source-forced-device-v1` rows time the whole program host to host (host input construction, transfers and launches included), so they carry no link bandwidth, price nothing separately, and are never read by selection.
A live mincore observation must establish resident input immediately before each
timed sample; unknown or nonresident input refuses this resident-input probe.
No page-in measurement is claimed.

Calibration checks a 540-second deadline between bounded probes; invoking recipes
also enforce a 600-second process deadline. A device completion error is a refusal,
and its runtime retains unresolved work under its existing ownership contract.
Profiles are always provisional. Runtime auto never invokes this calibration module.

C13 is satisfied by the probe harness: large CPU shapes invoke compiled LLVM
host-array exports directly; device shapes invoke verified `run_map` adapters.
Neither path evaluates the generated program through `invar run`. This keeps
benchmarking independent of the pre-B2 evaluator array/fuel limits.
