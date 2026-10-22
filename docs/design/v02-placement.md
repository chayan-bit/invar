# Calibrated placement for v0.2

Status: the profile/selector/mincore foundation, explicit provisional calibration and ordinary-source auto/explain integration are implemented. Integration verification is in progress; full performance acceptance remains open. See [the implemented contract](../../spec/placement-v1.md).
No acceptance criterion is closed by this document.

Binding PR #169 section 4.4 (`a41b6c7f`) applies: C14 names the calibration
probe `placement.wrapping_add1.u32` and registers emitted reasons in AI-01;
C4 adds exact `numeric_contract_version` matching and receipt fields;
C13 keeps large probes on LLVM host-array/device harnesses, never `invar run`.
See the implemented contract for the integer-only probe's version and remaining
auto/explain integration boundary. Historical profiles retain their old format.

## Fixed decisions and boundaries

`auto` compares every legal region variant with its CPU implementation, including actual residency, transfers, launch cost, and host page state.
A device must be more than 5% faster than the CPU to win.
Missing measurements, a missing profile, or an untrusted provisional profile retain the CPU.
Forced modes retain their coverage role and their existing legality checks.
Placement must preserve values bit for bit, first source trap, cleanup, and fuel behavior.
The workstream scope resolves D1 to CPU default before the freeze and D2 to LLVM native.
This continuation wires the ordinary CLI here: missing or untrusted profiles use LLVM whole-program CPU execution when available. Matched wrapping maps use the existing LLVM host-array export, so the coefficients describe the implementation that actually executes. Unmeasured work retains checked serial CPU execution. No changes are made to the #120 branch.

The initial base is `main` at `85d38988`.
The local prerequisite tips `ae70c26b` (M4-06, #153) and `4ba88400` (P8, #147) were merged normally, in that order.
Neither tip was an ancestor of the cached base.
Imported historical evidence is evidence for its recorded revision only.

## Reuse survey

| Candidate | License and maintenance | Decision |
| --- | --- | --- |
| P8 `general_net::law_collective::cost` and `tests/support/law_bench.rs` | Existing project code, maintained on #147 | Reuse the recorded-pass validation, best-of-five protocol, 5% band, and determinism tests; the v0.2 strict CPU tie rule remains authoritative. |
| M4-06 `schedule::tuning` and `tuning_resident` | Existing project code, maintained on #153 | Reuse residency, verified-plan facts, transfer accounting, and completion/quiescence rules. |
| `serde_json` 1.0.151 | MIT OR Apache-2.0; existing direct locked dependency | Reuse for bounded profile parsing and output; do not write another JSON parser. |
| `inkwell` 0.8.0 with LLVM 21 | Apache-2.0; existing optional locked dependency | Reuse the actual LLVM CPU execution path; do not label a Rust fold or MIR interpreter as compiled Invar CPU. |
| Rust `Instant` and `black_box` | Rust standard library, MIT OR Apache-2.0 | Use existing timing helpers for bounded probes; no timing dependency. |
| Criterion 0.8.2 | MIT OR Apache-2.0; published maintained benchmark library | Not in the lockfile; a statistical benchmark framework is unnecessary for this fixed best-of-five command. |
| `libc` 0.2.189 and platform `mincore` | MIT OR Apache-2.0; already in the lockfile transitively | Use a narrow reviewed existing FFI boundary for a safe slice-based page-state query; core retains `forbid(unsafe_code)`. |

Survey sources: [Rust monotonic timing](https://doc.rust-lang.org/std/time/struct.Instant.html), [Criterion package](https://docs.rs/crate/criterion/0.8.2), [Darwin libc interface](https://docs.rs/libc/latest/x86_64-apple-darwin/libc/), and [Linux mincore](https://man7.org/linux/man-pages/man2/mincore.2.html).
The local lockfile and the prerequisite implementations were inspected directly.
GitHub CLI queries failed with `Network is unreachable`; no claim of a completed live repository search or current PR status follows.

Live follow-up on 18 October 2026 completed `gh search repos` for Criterion and `gh search code` for `mincore` in `rust-lang/libc`.
Criterion's [original repository](https://github.com/bheisler/criterion.rs) directs active development to [criterion-rs/criterion.rs](https://github.com/criterion-rs/criterion.rs).
The latter and [Inkwell](https://github.com/TheDan64/inkwell) are unarchived with recent upstream pushes; this refresh does not change the locked dependencies or the decision to use the fixed existing probe protocol.
The lockfile still contains serde_json 1.0.151, Inkwell 0.8.0 and libc 0.2.189, with no Criterion dependency.

## Profile contract to implement

The default location is `calibration/<host-class>.json`.
Profiles need a numeric schema version, host class, CPU/device identity and relevant driver/toolchain identity, provisional flag, probe version, units, sample sizes, all five measured passes, source revision, and host load/swap observations before and after calibration.
Validate a bounded file before using any numbers.
Refuse unknown versions, malformed fields, negative/nonfinite costs, zero bandwidth, identity mismatch, or best-pass summaries that disagree with the recorded samples.
An explicitly supplied invalid file is a typed error, not a silent missing-profile fallback.
No measured M4 or Linux CUDA profile is fabricated or promoted to frozen status.
Frozen profiles must come from M5-01; provisional profiles require `--trust-provisional` before they can affect placement.

Each family coefficient set is keyed by the verified family and numeric contract.
A region-specific set additionally binds source identity, graph revision, and region identity; a function name alone is insufficient.
Region-specific measurements take precedence over matching family measurements.
An unmeasured family or intrinsic has unknown cost and remains CPU.
Device coefficients need fixed cost and per-work-unit cost, link bandwidth/latency, launch floor, and the residency boundary they actually measured.
CPU fold coefficients must distinguish resident input from page-in when a valid page-in measurement exists.
Unknown page-in cost must remain unknown; cache eviction is not page eviction.

## Selection and receipts

Reuse M4-06's `Residency { inputs, output }` and verified-plan byte/work/launch facts.
For each legal engine, compute execution cost plus transfers required by its actual API boundary.
Host-to-host adapters must not claim retained device residency.
Device-to-device residency removes uploads/downloads only when buffers are genuinely retained; CPU execution over device-resident input includes its download.
The device estimate includes link latency per required transfer and a launch floor for every actual launch.
Include temporary storage and capacity admission independently of estimated speed.

For CPU time `c` and device time `d`, select a device only when both are known, finite, legal, and `d < 0.95 * c`.
Equality at `0.95 * c` retains CPU.
Choose the cheapest qualifying legal device; exact device ties use a stable order.
Keep this distinction from P8's historical worst-pass tie-break, which can select a collective inside its marginal band.
The specified v0.2 rule takes precedence.

Receipts and explain must use the same decision object and include profile identity/version/provisional trust, family/region identity, legality/refusal, CPU time, device compute time, transfer time, launch time, total time, margin, residency, and host page observation.
Unknown figures are represented explicitly, never as zero.
Execution failures and publication-guard failures update the receipt to the engine that actually published.
Unknown device completion forbids CPU replay, as in M4-06.

## Host page observation

The P8 D5 record shows why a cache-cold fold is not a model of a paged-out input.
Use `mincore` on Linux and macOS over the borrowed live input, without reading or copying the input first.
Align the queried range down to an OS page and round its length up with checked arithmetic.
Hold the slice borrow through the syscall; inspect only the residency bit in each returned byte.
Use bounded metadata allocation, and treat syscall failure or an unsupported OS as unknown residency.
The result is a racy advisory snapshot, never an ownership or exactness guarantee.
It changes only the cost calculation, never legality, array contents, or trap order.
The existing CUDA/FFI crate is a possible boundary; a new crate solely for this one syscall is unnecessary.

## Execution integration

Ordinary `run --engine auto` and `explain --engine auto` use `auto_command` and
`schedule::profiled`, which load profiles once and select at reached loop heads.
They do not invoke runtime calibration. The legacy model/fault APIs remain available
for their existing explicit tests; ordinary auto no longer uses them.

The measured add-one family uses the same compiled LLVM host-array map as the
calibration probe. Device results reuse the existing family adapters and publication
guards. Metal requires its source-matched AOT artifact. Unmeasured laws, fusion,
float contracts, unknown costs and untrusted profiles cannot select a device.
The same decision object feeds run receipts and explain. Runtime failures update
the publishing engine; unknown completion forbids CPU replay.
New v0.2 intrinsic family IDs, exactness contracts, and measured work units must
come from their owning workstreams.

The calibration command must benchmark the same implementation/boundary it models.
Compile outside timed steady-state probes; record cold compilation separately.
Use fixed bounded shapes, best of five passes, actual output checks, and a total deadline below ten minutes.
Unknown/unavailable engines produce explicit refusals and no invented coefficients.
Publish the completed profile without overwriting an existing file accidentally or leaving a partial profile after failure.
Do not run calibration implicitly during `auto`.

## Required acceptance evidence

1. Failing test first, then a synthetic size/residency grid whose analytic optimum includes the exact 5% boundary, zero work, launch floors, transfers, unknown values, device ties, and provisional gating.
2. Profile parser tests for malformed versions/identities/figures, bounded input, and disagreement between five passes and their summaries.
3. Real CLI tests for `calibrate`, `--out`, family filtering, profile loading, trust, receipts, and explain; at least 80% new-code coverage.
4. Bitwise auto/reference comparison for every registered region and intrinsic under CPU, CUDA, and Metal choices, including float bit patterns, traps, and publication failure paths.
5. Mutation tests that kill dropped transfers, ignored margin/provisional gates, ignored host page state, and changed device results.
6. Linux CUDA calibration and exclusive Mac Metal calibration, each under ten minutes with load/swap and baseline figures recorded.
7. Spec/help updates only alongside the implemented CLI contract; source grammar changes only if source syntax actually changes.
8. A recorded calibration/auto/explain demo from both actual hosts, linked from the draft PR.

On 18 October 2026 the then-current `main` was `85d38988`. This continuation normally merges Metal AOT at `b4efbe2db` in `ddd5469b8`. The bounded CPU/CUDA calibration completed on the Linux CUDA host.

The cost selector remains advisory: it cannot bypass device legality or publication.
The ordinary CLI now consumes it through the compiled-map boundary described above.
Full all-family exactness and frozen performance acceptance remain open. New calibration
runs wait for a quiet window on the measuring host.
Historical device profiles retain their measured phases, source identity and provisional
status; they are not silently rebound to this changed compiler build.

The 18 October correctness follow-up recorded real Apple M4 Metal/AOT, Linux CUDA, CPU/mutation/Clippy, and post-merge oracle passes, with failed attempts retained. Main `8c4d509b` was merged once in `be1d0206` after the hardware gates; fresh two-host timing and the refreshed demonstration still await a quiet window.
