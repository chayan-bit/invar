# General-net v0.1 benchmark preparation

`manifest.json` pins 15 ordinary Invar source fixtures for reference-engine checking. `parity-spec.json` records the proposed general GPU and handwritten-performance obligations. Neither is a device result or an accepted release manifest. The scope authority is [the GNET contract](../../src/general_net/CONTRACT.md#semantic-obligations) and [ordinary-source parity](../../docs/performance-acceptance.md#ordinary-source-parity); the original 2026-09-28 amendments are `docs/general-gpu-v01.md` and `docs/cuda-parity-v01.md` at commit `85dc53f`. `docs/spec/invar-v0.1.md` governs the executable current-source subset.

## Owned arrays and resumable CPU continuation

The additive [strict-f32 ABI11 corpus](NUMERIC.md) retains every prior row and adds
96 hand-derived observations. It executes the published CPU numeric emission,
records source/oracle/compiler exclusions, and now executes all 250 valid source
rows through the private numeric named reference. The unchanged 254-row
denominator retains four frontend rejections and all compiler/oracle exclusions.

The additive [ABI4/5 child corpus](CHILDREN.md) exercises runtime child extraction,
source-order commit, continuation and private failure paths. It preserves all
119 earlier rows and adds39 source-derived calls. ABI4 runs emitted CPU code;
ABI5 runs explicitly labeled CUDA/Metal CPU workgroup models. Hardware execution
and driver fences remain pending.

The [ABI2/3 continuation](RESUMABLE.md) retains all prior sources and inputs,
expanding to52 ordinary sources and67 scalar runtime vectors. Its real CPU
runner exercises arrays/cleanup, three returned-owner roots, runtime changing
quanta, yielded/resumed worklists, fair independent roots, and older source
nontermination before a later trap. All source/backend exclusions remain in the
119-row combined denominator.

## CPU corpus expansion

The [2026-10-02 expansion](EXPANSION.md) preserves this complete 15-case manifest
and adds `expanded-manifest.json` with 47 ordinary sources and
`runtime-vectors.json` with 54 independent runtime inputs. Its real source/MIR,
GNET oracle and emitted CPU evidence covers owner cleanup/fuel, views,
records/variants/Result payloads, loops, recursion, traps and tool limits. All
31 structural MIR rules appear; execution exclusions remain in the denominator.
The original parity gates remain unchanged.

## Current coverage

| Boundary | Current cases | Required observation |
| --- | --- | --- |
| Irregular recursion with runtime-shaped pivot | `divide_irregular_a/b` | Different trees and exact return values |
| Input-dependent branch tree | `branch_tree_a/b` | Only selected branch demanded; different input structure |
| Dynamically discovered graph frontier | `frontier_graph_a/b` | Reachability for two graph/start variants |
| Cross-form composition | `composed_split_leaf` | Recursive split, branch and leaf loop together |
| First ordered trap | `ordered_trap_first/later` | First source-ordered category and span |
| Untaken bad work | `unselected_trap_diverge` | Normal result; trap and unbounded loop remain unselected |
| Earlier divergent prefix | `divergent_prefix` | Fuel exhaustion before later trap; no divergence proof |
| Ownership and ordered mutation | `alias_conflict`, `ordered_write` | Static borrow rejection; write then read value |
| Limited parallelism | `serial_chain`, `tiny_control` | Correct serial/small automatic placement, no invented speedup |

The paired source-variant fixtures are distinct **source literals**, because the current `main() -> i32` CLI cannot receive external input buffers. They check reference behavior and source-form admission only. GPU acceptance requires source-derived compiled entry points with input arrays, varied depths/shapes/graphs at runtime, structural holdouts, and no filename/hash matching. The current 5-node dense graph is a semantic smoke; the proposed large frontier uses a bounded sparse representation. Large recursive arithmetic must use an explicit wrapping or checked policy so it never gains speed by silently changing overflow behavior.

The future source-encoding matrix belongs to main's versioned net/compiled contract. Every actual v0.1 computational grammar/MIR form needs checked net IR semantics and compositional compiled lowering, including loops, recursion, owners, arrays/views, and later records/variants/closures/numeric forms when admitted. CPU net evaluation remains a semantic oracle. Eligible pure forms need **forced compiled device** evidence with real runtime branches, loops, frames or worklists beyond recognized templates. Host FFI/IO needs ordered host-boundary evidence. A serial compiled CPU fallback may be correct automatic placement; it does not prove device coverage. Record separately: (1) source/net/compiled semantic agreement, (2) measured compiled device work/frontier width in forced mode, and (3) per-case performance.

## Proposed GPU and robotics cases

`parity-spec.json.workloads` is the machine-readable coverage list. It includes three general dynamic families and five robotics rows: Panda/UR5e FK, full tool Jacobian and CRBA mass matrix; collision/distance sparse geometry; rollout/cost with separately bounded search; point-cloud transform/filter/compaction/neighborhood; and a complete kinematics-to-collision-to-rollout pipeline. Each row names proposed size sweeps, distributions, output/reference lineage and missing source primitives. Robotics rows remain pending because bootstrap source has no `f32`/`f64`, trigonometric or square-root operations, runtime numeric buffers, or frozen model/geometry/planner contracts. Those are v0.1 prerequisites, not exclusions. The linked robotics inventory identifies candidates; independent full-output oracles are still required before executable acceptance cases can be frozen.

The robotics CUDA parity reference (commit `52c363f`, not retained) identifies the pinned Menagerie Panda/UR5e MJCFs and eight Pinocchio FK tool-pose goldens. Those support small CPU correctness checks only. It also records MuJoCo model-load/step observations, which are not Jacobian, dynamics, geometry-distance or planner goldens. Panda vendor collision spheres under-cover the original meshes; UR5e spheres do not give original-geometry exact distance. No source-written rollout or point-cloud fixture, complete pipeline oracle, reviewed handwritten CUDA comparator, or real cuRobo GPU result is present. `parity-spec.json.reference_gaps` keeps each missing gate explicit. cuRobo belongs to the separately labeled application/library track.

The robotics workstream froze an internal Jacobian and CRBA mass-matrix **CPU oracle contract** before generation in `tests/fixtures/foreign-components/gpu-parity-oracles/contract.json` (commit `65c75d7`, SHA-256 `85362985e3729e6f40eac401efc3772d27fb90f93238f3e0174a01517bd2b0b6`). It pins model hashes, active joints/fixed fingers, eight deterministic configurations per arm, base-aligned 6×N Jacobian and full N×N mass outputs, source armature treatment, tolerances, and full-output artifact format. The full CPU result at `tests/fixtures/foreign-components/gpu-parity-oracles/reference.json@d313d41` (SHA-256 `8a4cef46ff9aeec67ab51f5c426b219e0d467eb16d0921679586ebc6cceb64c4`) passes Jacobian and finite-difference checks for both arms and UR5e active mass in all eight cases; Panda active mass **fails** the frozen threshold in six of eight. The reported Panda link1 imported-inertia difference needs resolution before its mass values can be an agreed golden. This internal fixture contract does not freeze the scored GPU floating acceptance policy.

The 2026-09-30 integration imports the reviewed all-body FK contract and result from
robotics head `49254e9`. Their SHA-256 values are recorded in the current spec's
`all_body_fk_cpu_evidence`: 144 body-origin transforms, zero historical CPU
comparison failures. Only the stale `all_link_FK_goldens` missing-evidence entry
is removed. Panda mass acceptance, generated-code agreement, device coverage and
performance remain pending. Earlier reports retain their original bytes and spec
digests; they are not evidence for the updated spec. The oracle-lineage regression
test pins the five reviewed artifacts and prevents this reconciliation from
promoting the failed mass comparison or unmeasured GPU track.

For every family, freeze at least two materially different input structures/compositions and a separate structural holdout set before tuning. The listed sweeps are proposals subject to approved target memory and reviewed input generation; their exact generator revision, seeds and hashes are not yet frozen. Keep every exposed failure when a holdout moves into tuning. Current fixtures cannot substitute for these later runtime-input cases.

## Baseline and timing contract

The scored route starts from ordinary Invar source and uses checked typed net IR, verified rewrites/dependence analysis, program-specific compiled CPU/GPU control and dataflow, native CPU code and CUDA/Metal kernels with compiled orchestration, and automatic residency-aware placement. Dynamic branches, loops, recursion frames and bounded worklists execute as compiled control; their queues, allocations and launches remain charged. A generic runtime net interpreter is not a production prerequisite. Prefer AOT native/device artifacts for declared targets and report loading, pipeline creation and any driver/PTX JIT honestly. No hidden runtime language compiler is needed for the default route. No user CUDA, manual schedule, whole-operation library delegation, or parameterized-plan timing counts for that track. Same-algorithm, best-application, and library-assisted comparisons are separately labeled diagnostics. Baselines must be independently reviewed for exact numerical, geometry, algorithm and output contracts; pin source/artifact digest, revision, license, flags, distribution, seeds, reuse policy and quality evidence before timing. Retain a stronger compatible baseline and score against it. A mismatched or weak reference has no scored denominator.

Choose the GPU-profitable size band from reviewed CPU/GPU **baseline-only** crossover evidence, before viewing Invar timing. Keep small and serial controls for correct placement and the declared 1.10 dispatch ratio gate. A missing profitable band leaves GPU parity pending; it cannot remove semantic coverage or failed cases.

Score each workload/size/device separately in both device-resident complete-operation and warm host-input-to-host-output scopes. Let `R = median(Invar) / median(reviewed baseline)`. Passing requires `R <= 1.00` **and** its one-sided 95% upper confidence bound `<= 1.05`; report “matched within the declared 5% uncertainty margin.” Claim “faster” only if the upper bound is below 1.00. A point estimate over 1.00 fails; an upper bound over 1.05 with a passing point estimate remains pending. No family mean can hide a failed row. Compare Metal with native Metal/MPS on the same Apple hardware as a separate target, never with absolute NVIDIA latency.

Use at least three independent process sessions; in each, warm each series at least 10 times and retain at least 30 raw samples per series in randomized, paired, interleaved blocks. Resample sessions then paired blocks with the pinned seeds and 10,000 bootstrap replicates in `parity-spec.json`. If the point passes but the bound is inconclusive, collect at most two additional batches of ten pairs per session under the same conditions; retain all valid samples and negative results. Report cold first result, JIT/tuning and amortization separately, plus p50/p95/p99, jitter, contention and peak memory. A timing result is invalid until full-output or explicitly frozen floating/stochastic correctness passes. Do not infer hard real-time or physical safety from these measurements.

The proposed `ablation_design` requires B net-derived compiled production code, D an independently reviewed handwritten CUDA baseline, and a CPU placement control. C direct structured lowering at B's schedule is diagnostic where available. A generic GPU net reducer is **optional** and currently `not_built`; A/B becomes an optional diagnostic, never a release prerequisite or invented measurement. B/C isolates the compiler net-layer cost; **B/D tests generated-code parity** under the unchanged per-case gate. For certified extracted region interiors, separately instrument zero dynamic net allocation and zero scalar tag dispatch; retain compiler net-IR provenance and count actual worklist queue bytes/allocations, tasks, atomics, launches, compiled dispatch, transfers, spills, occupancy, residual dynamic work, code growth and compilation. Instrumentation runs do not supply scored timings; attribute overlapping work along the measured critical path. Proposed controls include generic fixed-topology strict-order FK-like composition and balanced through 99:1 skewed frontiers, with held-out syntax/layout/runtime inputs. Full joint-angle pipelines charge `sin`/`cos` and geometry math under one frozen numeric policy. Identical precomputed-transform inputs form a separately labeled compute-island control. No domain/name specialization is allowed. Optional BendRT comparison requires a pinned common contract; the inspected BendRT paper uses no interaction nets and reported no CUDA timing, so it supplies no speed claim or mandatory baseline.

## Reference validation and backend integration

The additive Rust CLI validates every source hash and runs fresh `invar check` and `invar run --engine reference` observations, including expected negative exits. A validation pass covers source-literal corpus semantics only. Build `invar` and `invar-bench` together (`cargo build -p invar -p invar-tools`) and use a new output path each time. `--source-revision` is caller supplied; the CLI records the executable SHA-256 but cannot verify that the binary came from that revision. Reports mark `source_revision_provenance: caller_supplied_unverified` and `executable_source_binding: unverified`:

```sh
target/debug/invar-bench --gnet-validate --manifest bench/general-net-v01/manifest.json --spec bench/general-net-v01/parity-spec.json --invar target/debug/invar --source-revision "$REVISION" --report "$NEW_REFERENCE_REPORT"
target/debug/invar-bench --gnet-evaluate --manifest bench/general-net-v01/manifest.json --spec bench/general-net-v01/parity-spec.json --raw "$RAW_PAIRS" --source-revision "$REVISION" --report "$NEW_EVALUATION_REPORT"
```

`--gnet-evaluate` accepts bounded `gnet_paired_raw_v1` JSON: top-level manifest/spec SHA-256 and `cases`; each case has `case_id`, `size`, `work_items` (1–100,000,000), `device`, one of the two scored `scope` values, `track: "B/D"`, identical `binding`/`invar_binding`/`baseline_binding`, `evidence`, and 3–5 sessions. Bindings contain manifest, spec, source, input, baseline, device, numeric-contract and scope SHA-256 values; the scope digest is SHA-256 of its exact name. Each session records a distinct `id`, both warmup counts ≥10, the spec's `order_seed`, and 30/40/50 paired blocks with unique `block_id`, interleaved `order`, and finite positive `invar_ms`/`baseline_ms`. Both scopes are required for each case/size/device. The evaluator reports the median ratio and deterministic hierarchical session/block bootstrap upper bound; synthetic fixture data receives `qualification_status: pending` even if its statistical gate passes. The current source-literal manifest and proposed spec also keep qualification pending. Evaluation performs no hardware measurement and cannot turn CPU observations into GPU acceptance. Reports use `create_new` and refuse overwrite.

A fresh reference report with caller-supplied revision `f264d05` and the amended spec digest recorded all 15 source cases passing while GPU acceptance stays pending; its executable-to-revision binding remains unverified. The current corpus requires no GPU.

Main's backend integration should consume these source files by path and content, compile from source through the checked net IR, and attach a report with source/net-IR/manifest digests, compiler revision, input generator seed/digest, AOT artifact and driver-JIT status, device and driver, actual compiled device work/frontier and worklist statistics, placement reason, result agreement, reviewed baseline provenance and raw per-scope timings. The existing M5 harness does not yet consume this new manifest or parity spec. Do not reclassify a source literal, parser success, generated code, or parameterized plan as an ordinary-source GPU run.
