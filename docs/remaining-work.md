# Remaining work for v0.1

This is the v0.1 acceptance ledger, reconciled against `main` at 7634aff8 on 20 October 2026.
It lists every open v0.1 item: the open GitHub issues labelled `release:v0.1`, plus the GNET slices that [general-gpu-v01.md](general-gpu-v01.md) places before final M5 qualification and release acceptance (#47 and #45).

## Open v0.1 items

"Evidence on main" names what is merged, not what is qualified.
"Remaining gate" is the acceptance condition still unmet, taken from the issue contract.

| Item | Evidence on main | Remaining gate |
| --- | --- | --- |
| #1 M4-01 numeric and layout contracts | `issue_m4_01_numeric_layout_contracts` and the decimal, oracle, layout, tensor and NVRTC-mode tests; device targets `issue_m4_01_gpu` and `issue_m4_01_metal_gpu`; [numeric-v1](numeric-v1.md), [layout-v1](layout-v1.md), [m4-source](m4-source.md) | f32 family gates through the M5-01 harness against the frozen manifest and held-out set, on the RTX 5070 Ti (sm_120) and the Apple M4 |
| #2 M4-02 transpose and stencil | `issue_m4_02_transpose_stencil_schedules` and the source, host, runtime and schedule tests; `issue_m4_02_gpu` | Transpose and stencil family gates through the M5-01 harness on the named targets |
| #3 M4-03 contraction and Tensor Core | `issue_m4_03_matrix_contraction_tensor_core` (plus `_device`, `_tc`); `issue_m4_03_gpu`, `issue_m4_03_metal_gpu` | Matrix family gate through the M5-01 harness, generated and library-assisted tracks reported apart |
| #4 M4-04 segmented work and stable compaction | `issue_m4_04_segmented_flattening_compaction`, compaction and segmented device IR tests; `issue_m4_04_gpu`, `issue_m4_04_metal_gpu` | Segmented family gate through the M5-01 harness, including the held-out set and the CUDA shared-budget gate |
| #5 M4-05 gather, scatter and frontier | `issue_m4_05_gather_scatter_frontier` and the gather, histogram, scatter-proof and frontier tests; `issue_m4_05_gpu`; [indexed-device-v1](indexed-device-v1.md) | Gather/scatter and frontier family gates through the M5-01 harness |
| #6 M4-07 bounded summaries | `issue_m4_07_bounded_summary_families`, summary source, typed scan and device tests; `issue_m4_07_gpu`, `issue_m4_07_metal_gpu` | Summary family gate through the M5-01 harness; CUDA re-verification of the M4-07 device scans on the merged head |
| #8 M4-06 schedule tuning and residency | `issue_m4_06_bounded_schedule_tuning`; [bounded-schedule-dispatch](bounded-schedule-dispatch.md) | Tuning on the tuning set only, after the held-out hash is committed; steady-state, warm and cold GPU boundaries |
| #7 M5-01 manifest, baselines and anti-overfitting | `bench/manifest-v1.json`, `issue_m5_01_*` tests, the held-out proposal and exclusion index, the CUDA generated-map adapter, family baseline fixtures with an independent-reference runner | The permanent held-out freeze (#52), baseline timing, calibration in `invar-placement-v1` (#49) and the manifest bump binding all three |
| #52 M5-01 held-out freeze checklist | `bench/heldout-v1-proposal.json`: 608 programs, 139 held out, status `proposal-pending-review`; tuning paths admit programs only through `HeldOutIndex::admit` | Five of seven checklist items: split review, baseline timings, calibration, manifest bump, full suite green at the freeze commit |
| M5 timing runs | None on main; earlier timing grids ran beside another GPU tenant and are historical only | CUDA baselines 2^10 to 2^28 on the RTX 5070 Ti, Metal baselines at the final commit, placement calibration at one clean commit, M5-03 packets |
| #49 M5-01 calibration in `invar-placement-v1` | `invar calibrate` and `invar-placement-v1` carry the M5-01 protocol (10 warmups and 30 integer-ns samples with loader-recomputed order statistics, the 4 MiB loader, measured-xor-pending coverage, calibration claims only under a pinned digest) and measure all ten table families on the native CPU; CUDA and Metal probe the nine families beyond map as forced-device source on the `source-forced-device-v1` boundary (no link priced, never selected): `reduction` and `scan` are measured on the H100 and the Apple M4, and the other seven are `pending` with `device-plan-ineligible` because admission plans their kernels serially; [placement-v1](../spec/placement-v1.md) | Timing at a clean commit; the seven serially planned families stay `pending` on devices unless their kernels are rewritten for the device |
| #9 M5-02 full-pipeline qualification | `issue_m5_02_full_pipeline_qualification`; CPU qualification in [m5-02-cpu-qualification](m5-02-cpu-qualification.md); bounded device qualification adapters (`qualification/device.rs`), not yet run on the pinned target; every code `qualification/` emits is registered and the registry scan covers all of it | Device qualification on the pinned target; it also depends on M4-06, M4-07 and M5-01 |
| #10 M5-03 performance acceptance | `issue_m5_03_cuda_performance_acceptance`; [report harness](m5-03-report-harness.md) | Resident, warm and cold packets on every frozen target; per scored case a median ratio at most 1.00 and a 95% upper bound at most 1.05 ([parity contract](cuda-parity-v01.md)) |
| #14 L1-06 Python binding | `crates/invar-python`, `issue_l1_06_*` tests; [private wheel](python-native-wheel.md) qualified on Linux x86_64, glibc 2.41, CPython 3.12 only; the CI Python binding job runs the crate tests and pytest on every main push (Tier 2); `run_main(artifact=)` takes a compiled-ahead Metal library path, checked against the program on every `metal` and `auto` route | Platform matrix beyond the private wheel: the macOS arm64 run and the Nix dev-shell record on both platforms |
| S1-04 ROS 2 Lyrical target | Generator, golden files and `issue_s1_04_ros2_target`; the ignored `issue_s1_04_ros2_target_env` passed 1/1 on ARM64 with mock hardware (28 September) | The issue's own condition: the runtime smoke on the S1-06 cell with `mujoco_ros2_control` passes in S1-06 before S1-04 is reported done |
| #29 S1-06 integration-start acceptance | `issue_s1_06_integration_start` and the planner, sequencer and bridge tests; `issue_s1_06_integration_start_env` is ignored | The recorded two-arm reach-and-place run in the pinned ROS 2 Lyrical, MuJoCo and cuRobo environment |
| #15 AI-02 spec, examples and docs | [The v0.1 specification](spec/invar-v0.1.md), `examples/index.json`, `issue_ai_02_spec_examples` | Re-verify the specification after each of L1-06, S1-06 and M5-03 closes |
| #47 GNET-04 part 2, automatic placement for general nets | Part 1 (device fault injection, the `gnet-fault-injection` hook, the parallel corpus slice) landed; `issue_gnet_04_*` tests; D1(a) and D2(a): the legal engine set with `policy: legal-set-cpu`, `reason: uncalibrated` and `native-feature-unavailable` in placement receipts and explain, and the host route label | Calibrated dispatch after the M5-01 freeze, plus the issue's other items: legal specialization, residency across calls, measured overheads and the full dispatch explain |
| #45 GNET-05 device census and bounded Metal pipeline compiles | Robotics corpus and census tests (`issue_gnet_05_robotics_array`, ignored by default); Metal pipeline creation bounded by an asynchronous 900 s `PIPELINE_DEADLINE` with the typed refusal `metal_pipeline_deadline` | Six Metal census variants without an observation; a real deep-device count (`INVAR_GNET05_DEEP_DEVICE=1`); the Metal census runs only on a `gpu.yml` dispatch with `census_run` |
| #54 GNET P6/P7, the parallel normal form in `invar run` | P6(a): `cpu-parallel`, and `auto` in builds without a device feature, run a program that no existing parallel path claims through the checked normal form (`src/schedule/normal_form.rs`, certificates re-checked by `certificate::check_all`), with serial MIR as the fallback and an optional `normal_form` receipt object; `v01_normal_form_run` compares it with `eval::run`. `gnet-device` and `invar build --target cuda\|metal --entry` use `Limits::certified()`, 8 speculative slots. P7: `v01_gnet05_run` runs all 72 GNET-05 programs through the `invar run` path, equal to `eval::run` in 334 Debug and 364 Release cases | The scope addition of 20 October, not started: one dispatcher, so `run`, `auto`, `explain`, `calibrate` and `build` take regions only from the checked normal form, not the recognizer loop in `src/schedule/families.rs`; each family module becomes a lowering of normal-form nodes, landed one family per pull request (Core, Indexed, Segmented, Grid, Tensor) under selection parity with the recognizers, which become test oracles; acceptance in `tests/issue_54_family_lowerings.rs`. P7 device parity: correctness on the H100 and the Apple M4 (the scope in #54; the design table at `docs/design/gnet-parallel-v01.md:286` predates it and names the RTX 5070 Ti), and timed parity on the RTX 5070 Ti. The nightly tests `issue_gnet_05_corpus_executes_on_cuda_exactly` and `issue_gnet_05_corpus_executes_on_metal_exactly` check the general-net device route, planned directly through `device_host` as `gnet-device` plans, not `invar run`, whose normal-form route is CPU only; #54 records whether they meet P7. P6(b) calibrated placement |

Related issues that are not v0.1 items themselves:
- Device work merged while the sm_120 host was unavailable still needs its sm_120-only CUDA re-verification.
- #46 (Python Metal AOT artifact) belongs to the L1-06 row: `run_main(artifact=)` added the path form, refusing in-memory bytes with `TYPE_MISMATCH`, and the issue is still open.
- `auto` no longer skips the library check on its CPU-only sort and spatial branch.
- The M5-02 qualification diagnostics are registered: the registry holds the codes `qualification/mod.rs` emits, and the registry scan covers all of `qualification/`.
- The GNET-05 robotics kernels over the 300 s CUDA compile deadline were decided on 20 October: the deadline stays, and the typed `compile_deadline` and `compile_error` census rows are the v0.1 outcome. Kernel size, compiled-ahead images and a route-specific compile budget move to the engine program (#62, compile time as a first-class cost). NVRTC 12.9 and sm_120 compile times are unmeasured.
- #51 (GNET-05 CUDA census compile time) met its gate: a nightly GPU run on `main` at 2234446e completed `issue_gnet_05_robotics_array`, compile-ahead census and device census on the H100 included, in 10400 s (2 h 53 min), inside the 3 h per-binary bound with 400 s to spare. The census still compiles each plan once, ahead, in its bounded NVRTC pool, and its compile-deadline refusals are the v0.1 outcome (the 20 October decision above).

Evidence that a row takes from a GPU, native or Release run holds on `main` only once that run is green on `main`.

## ABI11 CPU owned-numeric continuations (qualification record)

Private `cpu-owned-numeric-continuation-v1`: strict-f32-v1, CPU only, direct `Array<i32>`/`Array<u32>`/`Array<f32>` owners, scalar external arguments, the unchanged ABI9 task/frame layout and the ABI10 ten-word terminal packet. Scope includes the compiler entry, provenance receipt, source owner observer, foreign loader and loaded host. ABI1–10 are unchanged. Status: **implemented and focused-qualified**, not qualified for v0.1.

- Focused Debug/Release suites, fmt and strict default-feature Clippy pass. There is no clean final full-workspace run at the published tree. An earlier full run had one stale test failure, since fixed, and one environment failure.
- Native, LLVM 21 on Linux aarch64: the 19 native-gated targets first failed 3 standalone C-call tests (149/3/1). After the one-line `src/native.rs` packaging fix they pass **152, with 0 failed and 1 ignored**, in each of Debug and Release. Focused native gives 140 and default-feature affected targets give 74. The private wheel passes 253 on Linux aarch64, glibc 2.41, CPython 3.13.5. Its `manylinux_2_39` tag has no glibc 2.39 runtime qualification.
- The existing aarch64 `--all-features` strict-Clippy failure (`unnecessary_cast` on NVRTC `c_char` logs in `crates/invar-cuda`) is fixed by a bit-preserving byte helper. It now passes in Debug and Release on aarch64, the `invar-cuda` runtime host tests pass 27 in each profile, and x86_64 is type-checked only. There was no GPU, CUDA driver or x86_64 runtime execution.
- Open: the sealed fixed-PATH `cc` lineage test (`issue_m5_01_build_lineage`) still needs a working system C toolchain. CUDA/Metal execution was not run.
- Accepted for this milestone: if the host cannot allocate its snapshot before a native advance, it fails closed and closes the plan, rather than offering ABI10-style retry. The host API is unchanged.
- Hardware, public bindings (L1-06), M4 families, M5 qualification and robotics S1 gaps are unaffected.

## v0.2 robotics-language program

The v0.2 goal is a robotics stack written only in Invar; the architecture, the Phase A integration (with cross-workstream conflicts C1-C16) and the Phase B decisions are in [design/v02-robotics-language.md](design/v02-robotics-language.md).
Phase A workstreams: `sort`, `fsum`, `f64math`, `spatial`, `placement` (branches `feat/v02-<name>`).
The v0.2 exit gate is stack v0 (#43): Panda reach-and-place in MuJoCo with every stack component in Invar.

| Item | Issue | Release |
| --- | --- | --- |
| B1 FFI v2 | #36 | v0.2 |
| B2 Per-tick budgets, byte budgets, bounded `Vec<T>` | #37 | v0.2 |
| B3 I/O, time and component executor | #38 | v0.2 |
| B4 Multi-file modules and packages | #39 | v0.2 |
| B5 Fixed-size linear algebra | #17 (re-scoped) | v0.2 |
| B6 AD, forward / reverse | #18 (re-scoped) / #40 | v0.2 / v0.3 |
| B7 Real-time profile, measured WCET | #20 (re-scoped) | v0.2 |
| B8 Determinism and record/replay | #41 | v0.2 |
| B9 LSP, MCP, formatter | #22 (re-scoped; formatter from #11) | v0.2 |
| B10 Jetson Orin | #21 (re-scoped, hardware-gated) | v0.2 |
| B11 CPU worker cap | #42 | v0.2 |

#13, #23 and #25 keep their v0.3 scope; their bodies now record which minimum moved into v0.2.

## Rules for updating status

- **Implemented** means source exists and named local checks ran
- **Integrated** means the source is present in the named integration commit
- **Qualified** requires the original acceptance conditions and exact platform/test evidence
- **Closed** is the GitHub issue state, not a synonym for any of the above
- Record unsupported, filtered, ignored and unavailable cases; do not remove them from denominators
- Hardware evidence remains pinned to its original code, toolchain and device
- Do not close issues in bulk or infer release readiness from one green suite
- Update this ledger in the same pull request that changes an item's evidence, gate or blocker

Do not create duplicate implementation issues merely to restate this index.
Add evidence to the existing contract and close only after its actual completion/merge policy is satisfied.
