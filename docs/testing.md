# Testing guide

This page explains how the test suite is organized, how to run the part you need, and what each test file covers.
Many files are named after internal issue labels such as `issue_m4_05_*`; the tables below translate each name into plain words.
For the meaning of the labels see the [glossary](glossary.md#issue-and-milestone-labels), and for the required validation gates see [validation](validation.md).

## How the suite is laid out

| Location | What it holds | How to run it |
| --- | --- | --- |
| `src/**` (`#[cfg(test)]` modules) | Unit tests next to the code, for example [src/schedule/tests.rs](../src/schedule/tests.rs) | `cargo nextest run --locked --lib` |
| `tests/<target>/main.rs` | One integration-test binary per directory; each file in the directory is a module of that binary | `cargo nextest run --locked --test <target>` |
| `tests/*.rs` | Four single-file integration binaries | `cargo nextest run --locked --test <file stem>` |
| `tests/support/`, [test-support/](../test-support/README.md) | Shared helpers included by `#[path]`, not tests themselves | - |
| `tests/fixtures/`, `tests/*_fixtures/`, `tests/a1_tensor/` | Input programs and data | - |
| [tools/tests/](../tools/README.md) | Tests of the benchmark and qualification binaries (package `invar-tools`) | `cargo nextest run --locked -p invar-tools` |
| `crates/*/tests/` | Tests of the device, CUDA, Metal and foreign crates | `cargo nextest run --locked -p invar-device` |
| [crates/invar-python/tests/](../crates/invar-python/README.md) | Python binding tests (pytest) | `maturin develop --locked --extras test` then `pytest`, as the Build (python) CI job does |
| Doctests | Examples in Rust doc comments | `cargo test --locked --doc` |

A test's full name in nextest output is `<binary> <module>::<function>`, for example `invar::language arithmetic_fuel_and_demand::arithmetic_fuel_and_demand_fuel_and_call_depth`.
By convention a function name starts with its module name.

## Running a subset

```sh
just test-quick                                              # the fast loop: lib tests, the language and gnet_net binaries
cargo nextest run --locked --test schedule_cpu              # one binary
cargo nextest run --locked --test language -E 'test(~fuel)' # tests whose name contains "fuel"
cargo test --locked --test language arithmetic_fuel_and_demand             # the same without nextest, by name filter
```

Environment variables that change how the suite runs:

| Variable | Effect |
| --- | --- |
| `INVAR_FULL_CORPUS=1` | Run every program of the corpus sweeps; without it, tests run a deterministic 1-in-8 sample (see [test-support/README.md](../test-support/README.md#corpus-sample)). `just test` and CI on `main` set it |
| `INVAR_TEST_TIMEOUT_SCALE=N` | Multiply wall-clock deadlines of child processes by N (1 to 64, default 1); CI uses 8 |
| `NEXTEST_TEST_THREADS=N` | Number of parallel test processes |
| `INVAR_CI_PROFILE=debug` | Set in CI so [tests/limits/ci_profile_canary.rs](../tests/limits/ci_profile_canary.rs) checks that debug assertions are on |

Nextest profiles live in [.config/nextest.toml](../.config/nextest.toml): `ci` (used by CI and `just test`) and `quick` (used by `just test-quick`).
Its per-test overrides name tests that need several threads or must run first, so a renamed test must be renamed there too.

## Tests that need more than a default build

- **GPU tests.** [tests/gpu_cuda](../tests/gpu_cuda/main.rs) needs `--features cuda` and an NVIDIA GPU; [tests/gpu_metal](../tests/gpu_metal/main.rs) needs `--features metal` on Apple Silicon.
  Each test in them is marked `#[ignore]`, so a default run skips them, and CI only compiles them.
  Without the hardware you do not need to run them; say so in your pull request.
  With it: `just test-gpu-cuda` or `just test-metal`, or one family with `cargo test --features cuda --test gpu_cuda -- --ignored m3_01::`.
- **Native tests** need `--features native` and LLVM 21; several `language` and `mir` tests exercise the native backend only in such a build.
- **`_env` robotics tests** need external tools such as ROS 2 or MuJoCo and are described in their file headers.
- **Compiler-heavy tests** (the `gnet_*` binaries) compile generated C++ and need a C++17 compiler on `PATH`; CI uses the system compiler on Linux and llvm-mingw on Windows.

## Reading the issue-label names

A module named `issue_<label>_<topic>` was written for one issue-tracker item; the label says which.

| Label in a name | Meaning |
| --- | --- |
| `m0` to `m5` | Milestones M0 (language core) to M5 (benchmarks, qualification, placement) |
| `gnet_nn`, `gnet_pNN` | General-net work items and phases |
| `eng_nn` | Engineering work items (laws prover, fold tiers, contraction emitter, byte budgets, native SIMD) |
| `l1_nn`, `r1_nn`, `s1_nn` | Language extensions, units and frames, and system integration |
| `a1`, `v01`, `v02` | The family adapters, the v0.1 acceptance work and the v0.2 features |
| `ai_nn`, `dx_nn` | Tooling items: machine diagnostics and spec examples, debugger and fix tooling |
| `issue_50_`, `issue_54_` | GitHub issue numbers (#50 auto placement receipts, #54 family lowerings) |

The tables name each module by its current file name and say what it covers.
Renaming them to plain topic names is a possible follow-up; until then, use the tables to find the right file.

## What each integration binary covers

### `language`

The source language itself: syntax, names, types, arithmetic, fuel, ownership, loans and the CLI contract, with each case run on the reference evaluator, MIR and (with the `native` feature) native code.

| Module | Covers |
| --- | --- |
| `engines` | Helper: one check that every engine agrees on a program. |
| `syntax_and_limits` | Grammar, precedence, lexical spans, JSON diagnostics and the source-size and nesting limits. |
| `names_types_and_calls` | Name and type checking, forward calls, recursion, `check --format json` and its exit status. |
| `arithmetic_fuel_and_demand` | Evaluation order, demand (unselected branches), integer boundaries, traps, fuel and the call-depth limit. |
| `arrays_ownership_and_cleanup` | Array values, indexing, `filled_*`, ownership errors, cleanup order and the allocation bound. |
| `call_loans` | `read`, `mut` and `take` call loans: accepted combinations and conflicts at the call boundary. |
| `cli_examples_and_exit_codes` | The real `invar` binary on the bootstrap examples: results, statuses and usage errors. |
| `reference_corpus_and_json` | A deterministic reference corpus, adversarial sources and the one-JSON-document rule. |
| `owner_return_fuel` | Fuel charged when a function returns an owner by bare name or explicit move. |
| `records_variants_result` | Records, variants, `match` and `Result`: propagation, moves of fields and ownership errors. |
| `borrowed_views` | Borrowed views: reborrow chains, range and stride traps, and rejected escapes; all engines agree. |
| `generics_and_closures` | Bounded generics, monomorphization, affine closures and their resource limits. |
| `bytes_strings_and_c_abi` | `Bytes`, `String`, UTF-8 conversion and the C ABI shim checked against a foreign C oracle. |
| `units_and_frames` | Quantities, dimensions and frames: erasure is byte-identical, and transform chains agree across engines. |

### `mir`

MIR, the typed net and the native backend: the independent MIR evaluator, the typed-net verifier, checked rewrites, and LLVM lowering (native tests need the `native` feature).

| Module | Covers |
| --- | --- |
| `corpus` | Helper: programs shared by the MIR differential suites. |
| `mir_verifier_and_evaluator` | MIR lowering, the MIR verifier's rejections and the independent MIR evaluator against the reference. |
| `mir_owner_return_fuel` | Fuel for explicit move and borrowed returns is unchanged by MIR lowering. |
| `native_cpu_trap_abi` | Native CPU lowering against the reference: trap ABI, CLI build and run, and refusal of unproved LLVM promises. |
| `typed_net_verifier` | Typed net built from public MIR: ports, ownership, demand order and failure provenance. |
| `typed_net_rewrite_extraction` | Checked rewrites and extraction: beta and fusion change the extracted plan, and results match the reference. |
| `native_parallel_regions` | Compiled `cpu-parallel` map and reduce regions. |
| `native_simd_and_targets` | The native optimization pipeline, CPU target selection and SIMD. |

### `schedule_cpu`

The CPU scheduler: admission records, fused maps, lawful reductions and scans, worker counts and the serial fallback.

| Module | Covers |
| --- | --- |
| `cpu_workers` | Worker-count selection and measurement. |
| `recognition_corpus` | Generated corpus for the scheduler's recognition and fallback. |
| `admission_records` | Admission records: round trip, invalidation, rejection reasons and `explain`. |
| `fused_map_extraction` | Fusing adjacent maps through the typed graph, and when fusion must be refused. |
| `lawful_reductions_and_scans` | Reductions and scans that rely on a sealed law, with exhaustive bounded and seeded checks. |
| `affine_recurrence_scan` | Inference of an affine wrapping recurrence and its prefix scan, and near misses with reasons. |
| `cpu_scheduler_fallback` | Every family, shape and worker count equals the serial result; false guards fall back before any write. |

### `cli`

The `invar` binary as a user runs it.

| Module | Covers |
| --- | --- |
| `cli_smoke` | Help text and typed refusals from the real binary. |
| `placement_receipts` | Placement runs record float-reduction receipts and intrinsic sites. |
| `auto_explain_schema` | Every `--engine auto` explain carries one schema version and `auto_path`. |
| `source_device_command` | `invar source-device` on real source, labeling CPU-model completion. |

### `byte_budgets`

The byte-budget mechanism that bounds admission (host, device and analysis bytes) in each family.

| Module | Covers |
| --- | --- |
| `admission` | Admission bounded by bytes, not region counts. |
| `array_limit` | The per-array language limit after the old element caps were removed. |
| `budget_core` | The single `ByteBudget` mechanism and its refusal shape. |
| `cpu_map` | The CPU map pass is bounded by host bytes. |
| `frontier` | Dense-mask frontier admission by bytes. |
| `histogram` | Weighted histogram admission by bytes. |
| `indexed_scatter` | Gather, injective scatter and affine scatter admission by bytes. |
| `loops` | Typed-graph limits come from the analysis byte budget. |
| `summary` | Summary matcher has no fixed read-size cap. |

### `contraction_emitter`

The generic matrix-contraction emitter (CUDA and Metal source) checked without a GPU.

| Module | Covers |
| --- | --- |
| `emitter` | The move sequence of the generic emitter. |
| `layout` | Strided layouts: rank, extents and strides. |
| `moves` | Physical schedule moves over a generated grid. |
| `pipeline` | Staged pipelines lowered as asynchronous copies. |
| `targets` | Target description files parse and validate. |
| `tc` | Matrix-atom (tensor core) schedules through the emitter. |
| `wiring` | The shipped `gemm` and `tc` schedules emit through the generic path. |

### `device`

Device plans checked on the CPU: the shared device IR, its verifier, CUDA and Metal emission, compiled-ahead container formats and mock-device runtime.

| Module | Covers |
| --- | --- |
| `a1_grid` | Run-path evidence for the grid (spatial) adapters on the CPU boundary. |
| `a1_tensor_dispatch` | Tensor dispatch through the CLI and owning devices. |
| `issue_a1_indexed` | Ordinary indexed-source routing. |
| `issue_a1_segmented` | CLI refusals for forced engines on programs with no device candidate. |
| `issue_gnet_cuda_aot_format` | Every field of the compiled-ahead CUDA container is validated on load. |
| `issue_metal_aot_format` | Every field of the compiled-ahead Metal container is validated on load. |
| `issue_m3_01_cuda_runtime` | The CUDA runtime driver against a mock device: prepare, submit, complete, read. |
| `issue_m3_02_device_ir_verifier` | Device IR for admitted maps: determinism, malformed plans and no device fallback. |
| `issue_m3_03_cuda_fused_maps` | Fused map chains and zip DAGs lowered to device plans with CPU parity. |
| `issue_m3_05_guarded_gpu_integration` | Verified-graph plans, forged cache artifacts and the measured-cost rule. |
| `issue_m4_01_metal_emission` | Metal source emission shares the verified IR with CUDA. |
| `issue_v01_pair_reduction` | Ordered affine-pair reduction cost facts, reference, MIR and chunked CPU. |
| `issue_v01_shared_device_cap` | Shared map, reduction and scan admission against the device byte cap. |
| `v01_placement_calibration` | The `invar-placement-v1` calibration protocol and profile loader. |

### `diagnostics`

Machine-readable diagnostics, source fixes and source-level debugging.

| Module | Covers |
| --- | --- |
| `issue_ai_01_machine_diagnostics` | The diagnostics schema, the code registry, fixtures and CLI documents (a file plus a directory of submodules). |
| `issue_dx_20_debug_command` | `invar debug` builds with debug info in a private directory. |
| `issue_dx_20_debug_info` | DWARF 5 emitted by the native backend, read back and checked. |
| `issue_dx_20_debugger_helpers` | The gdb helper script in a live debugger session. |
| `issue_dx_22_fix_coverage` | `check --fix`: each fix kind, rounds, the diff output and `--write` (a directory of submodules). |

### `exports`

Selected scalar and array exports, host array arguments and the Python boundary policy.

| Module | Covers |
| --- | --- |
| `issue_l1_06_export_failure_contract` | End-to-end failure behavior of the scalar export API. |
| `issue_l1_06_export_lineage` | Exports keep their source lineage across engines. |
| `issue_l1_06_native_arrays` | Array parameters of exports: admission and copying entry. |
| `issue_l1_06_native_f32_arrays` | `f32` array parameters of native exports. |
| `issue_l1_06_python_binding` | Core-independent policy gates of the Python binding. |
| `issue_l1_06_scalar_export_core` | Admission, identity, visibility and limits of scalar exports. |
| `issue_l1_06_scalar_export_corpus` | Scalar source corpus for exports. |
| `issue_numeric_host_mut_reference` | `mut` host array arguments on the copying reference engine. |
| `issue_numeric_host_read_reference` | Read-only host array arguments. |

### `gnet05_corpus`

The GNET-05 acceptance corpus under `bench/general-net-v01/gnet05`.

| Module | Covers |
| --- | --- |
| `issue_gnet_05_corpus` | Source-derived corpus: every program against the oracles. |
| `issue_gnet_05_robotics` | Robotics programs of the corpus. |
| `issue_gnet_05_robotics_array` | Host-array variants and the device admission census (a file plus a directory). |

### `gnet_net`

The general interaction net: lowering a checked program to a net and comparing the net oracle with the reference evaluator and MIR.

| Module | Covers |
| --- | --- |
| `issue_gnet_01` | The checked-source boundary, wire mutations, forged annotations and version separation. |
| `issue_gnet_02` | Arrays in the net: moves, traps, cleanup prefixes, fuel prefixes and live-array budgets. |
| `issue_gnet_02_aggregates` | Records and variants in the net: results, moves, borrows and cleanup. |
| `issue_gnet_02_corpus` | Every runnable fixture and example agrees across engines. |
| `issue_gnet_02_generated` | Seeded random well-typed programs. |
| `issue_gnet_02_native` | Compiled native CPU lowering of the net against the oracle. |
| `issue_gnet_net_oracle_gaps` | The exact corpus gaps recorded by qualification. |

### `gnet_compiled`

Compiled general-net code on the CPU: scalar, owned-array and aggregate ABIs, the device host and speculation.

| Module | Covers |
| --- | --- |
| `owned_driver` | Helper: the C++ driver shared by the owned and aggregate suites. |
| `issue_gnet_03` | Compiled CPU code for scalar programs against the oracle. |
| `issue_gnet_03_owned` | Compiled CPU code with owned arrays and resumable continuations. |
| `issue_gnet_04_host` | The host driver: runtime inputs, continuations, fuel prefixes and caps. |
| `issue_gnet_04_host_cli` | `invar gnet-device` and its CPU-model receipts. |
| `issue_gnet_04_host_failure` | Host-side failure and poison handling. |
| `issue_gnet_04_m4_admission` | Numeric forms are checked and never reinterpreted as integers. |
| `issue_gnet_04_speculation` | Independent recursive children are extracted and committed in source order. |
| `issue_gnet_06_aggregates` | Compiled CPU code for records and variants. |
| `issue_gnet_06_m4_admission` | Current f32 scalar, array and record forms fail closed. |

### `gnet_device_model`

Emitted device kernels compiled as C++ and run on the CPU (the CPU model) against the oracle.

| Module | Covers |
| --- | --- |
| `epoch_model` | Helper: the shared CPU model of per-root continuation kernels. |
| `issue_gnet_03_device` | Ordinary-source device admission and the corpus CPU-model run (the long corpus test is its own CI job). |
| `issue_gnet_03_injection` | Device fault injection on the CPU group model. |
| `issue_gnet_04_device` | Group execution of recursive children and exact prefixes. |
| `issue_gnet_12_cuda_owned_numeric_model` | CPU model of the CUDA owned-numeric continuation (ABI12). |
| `issue_gnet_resume_device_model` | CPU model of the resumable CUDA/Metal continuation (ABI3), every epoch. |

### `gnet_numeric_host`

Loaded numeric (`f32`) continuations, ABI11 hosts and named-source numeric entry points.

| Module | Covers |
| --- | --- |
| `issue_gnet_08_f32` | `f32` storage, emission and resumable batches. |
| `issue_gnet_08_named_source` | Private numeric named-source entry: bit copies and every fuel prefix. |
| `issue_gnet_10_f32_arrays` | `f32` arrays: source, MIR, oracle and native agree on every fuel prefix. |
| `issue_gnet_11_numeric_host` | Loaded ABI11 owned numeric continuations. |
| `issue_gnet_11_numeric_owned_state` | ABI11 emitted C++ under UBSan against the oracle. |
| `issue_gnet_f32_helpers_flag_independence` | Scaffolding for the strict-f32 owned-continuation contract. |

### `gnet_provenance`

Receipts that bind compiled artifacts to the exact source, bytes and budgets they came from.

| Module | Covers |
| --- | --- |
| `issue_gnet_05_provenance` | Reproducible, byte-bound provenance of compiled artifacts. |
| `issue_gnet_05_provenance_owned` | Provenance of owned and continuation layouts (a file plus a directory). |
| `issue_gnet_10_provenance` | Provenance of `f32` array layouts. |
| `issue_gnet_11_provenance_source` | Source-issued ABI11 receipt checks. |

### `gnet_slices`

Small vertical slices of the general-net work: arrays and records, device entry, exactness.

| Module | Covers |
| --- | --- |
| `gnet_p2` | The P2 slice: arrays, copies and limits agree with the oracle. |
| `issue_gnet_p0b_entry` | The device entry ABI: CPU group models, typed refusals, pending work. |
| `issue_gnet_p5b` | Reachability, numeric helpers and returned owners against the oracle. |
| `issue_gnet_p5b_robotics` | Two robotics programs on the device against the oracle. |

### `normal_form`

The law-directed normal form of the net: regions, laws, certificates, family admission and the programs that exercise them.

| Module | Covers |
| --- | --- |
| `law_programs` | Helper: one source per law case. |
| `net_regions` | Region partition of the verified net. |
| `net_normal_form_annotations` | Lowering annotations: re-derived demand order and more. |
| `net_normal_form_corpus` | Classification of the GNET-05 and held-out corpus. |
| `net_normal_form_followup` | Follow-up cases: mixed checked updates stay serial. |
| `net_normal_form_laws` | Randomized law executor against the net oracle; every law has a caught mutation (a file plus a directory). |
| `net_task_region` | The residual fork-join task region against the oracle. |
| `normal_form_residue_seed` | The residue ring behind the affine laws. |
| `pnf_family_admission` | Family admission from the normal form (a file plus a directory). |
| `gnet_p4a_path_sum` | The path-sum lowering against the oracle. |
| `v01_gnet05_run` | `invar run` on the GNET-05 corpus. |
| `v01_gnet_auto_placement` | Placement runs record the legal engine set. |
| `v01_normal_form_run` | `invar run` executes serial-only programs through the checked normal form. |

### `laws_prover`

The in-repo prover for law formulas and lifting certificates.

| Module | Covers |
| --- | --- |
| `certificate` | Lifting certificates (the Kadane example). |
| `lift` | A loop iteration re-derived from the net as law formulas. |
| `lifting_laws` | Laws beyond the sealed catalogue. |
| `prove` | The decision procedure for law formulas. |

### `law_collective`

Certificate-bound ordered collectives on the CPU.

| Module | Covers |
| --- | --- |
| `net_law_collective` | Certificate boundary, layout and emitted sums against the oracle. |
| `net_law_collective_bench` | CPU statistics for the calibration. |

### `limits`

Limits, deadlines and the test harness itself.

| Module | Covers |
| --- | --- |
| `child_deadlines` | Scaled child-process deadlines and scratch directories. |
| `ci_profile_canary` | The CI profile keeps debug assertions. |
| `eval_call_depth_stack` | The 64-call limit is reached without exhausting the host stack. |
| `exec_fixtures` | Generated executables survive parallel forks. |
| `robustness_limits` | MIR size limits and live-array caps. |

### `numeric_layout`

`f32` source semantics and the logical-to-byte layout contract.

| Module | Covers |
| --- | --- |
| `decimal_literals` | Exact decimal-to-binary32 literals, ties and subnormals. |
| `layout_oracle` | Contiguous, transposed and holed layouts against the layout oracle. |
| `numeric_layout_contracts` | Numeric and layout contracts, rank-1 source parity and mutation order. |
| `numeric_oracle` | The software binary32 oracle and NaN classes. |
| `source_f32` | Strict `f32` arithmetic, comparisons and rejected conversions in source. |
| `tensor_cache` | Tensor cache keys are independent of storage. |
| `tensor_host` | Host-side tensor contraction parity. |

### `transpose_stencil`

Transpose and five-point stencil families.

| Module | Covers |
| --- | --- |
| `combined_runtime` | One run publishing transpose and stencil loops together. |
| `device_schedules` | Device plans for transpose and stencil, with mutations that must be refused. |
| `stencil_runtime` | Bounded CPU stencil publication. |
| `stencil_source` | Recognizing the stencil source form. |
| `transpose_host` | Transpose publication keeps typed results, fuel and cleanup. |
| `transpose_source` | Recognizing the transpose source form. |
| `schedules` | The transpose contract entry point. |

### `contraction`

Tiled matrix contraction and the explicit `tc_mul_add` form.

| Module | Covers |
| --- | --- |
| `matrix_contraction` | Contraction source family and contract. |
| `matrix_contraction_device` | Tiled strict-f32 device schedule checked without a GPU. |
| `matrix_contraction_tc` | The `tc-f16-f32-v1` explicit `tc_mul_add` source without a device. |

### `segmented`

Segmented reductions and stable compaction.

| Module | Covers |
| --- | --- |
| `compaction_allocation` | Compaction allocation sites and budgets. |
| `segmented_device_ir` | Segmented-map recognizer and device IR on the CPU. |
| `segmented_flattening_compaction` | Flattening and compaction through the checked source. |
| `stable_compaction_source` | Stable compaction source forms. |

### `indexed`

Gather, scatter, histogram and frontier families.

| Module | Covers |
| --- | --- |
| `device` | Device IR and emitters on the shared corpus; mutations are refused. |
| `frontier_source` | Dense-mask frontier source form. |
| `gather_preflight` | Gather preflight costs and recipes. |
| `gather_runtime` | Gather runtime outputs, aliasing and exact prefixes. |
| `gather_scatter_frontier` | Gather, scatter and frontier contracts. |
| `histogram_runtime` | Weighted histogram runtime, traps and fuel prefixes. |
| `histogram_source` | Histogram source recognition. |
| `static_scatter_proof` | Static injectivity and bounds certificates for scatter. |
| `static_scatter_runtime` | Scatter runtime with a certificate. |

### `summary`

Bounded summaries (finite-state and matrix scans).

| Module | Covers |
| --- | --- |
| `bounded_summary_device` | Device lowering and reason codes. |
| `bounded_summary_families` | Summary families and tables. |
| `summary_source` | Summary source recognition and preparation. |
| `typed_bounded_scan` | The typed bounded-scan graph. |

### `numeric_tiers`

Floating-point fold tiers and storage-only types.

| Module | Covers |
| --- | --- |
| `fold_receipts` | Fold tiers recorded in placement and explain documents. |
| `numeric_tiers` | Numeric policy version 2 and its oracle. |
| `storage_arrays` | `Array<f16|bf16|i8|u8>` on every CPU engine. |
| `tier_engines` | Tree and Exact folds on every CPU engine. |

### `robotics`

The system dialect, foreign components, generated ROS 2 and MuJoCo targets, the planner boundary and sequencer. Tests ending in `_env` need external tools and are described in their headers.

| Module | Covers |
| --- | --- |
| `file_hash` | Helper: SHA-256 of a file. |
| `issue_s1_01_system_dialect` | Parsing and canonical IR of the system dialect. |
| `issue_s1_02_foreign_manifests` | `extern component` manifests and typed configuration. |
| `issue_s1_03_graph_checks` | Rates, latency, frames and resources of a system graph. |
| `issue_s1_04_ros2_target` | Deterministic ROS 2 workspace generation. |
| `issue_s1_04_ros2_target_env` | Building and launching the generated ROS 2 workspace. |
| `issue_s1_05_robot_cell_mujoco` | Typed robot cells and generated MJCF. |
| `issue_s1_05_robot_cell_mujoco_env` | Generated models load and step in MuJoCo. |
| `issue_s1_06_controller_bridge` | The generated controller bridge. |
| `issue_s1_06_grasp_v10` | Explicit grasp semantics. |
| `issue_s1_06_integration_start` | A one-robot reference cell and its imported evidence. |
| `issue_s1_06_integration_start_env` | Recorded task acceptance in simulation. |
| `issue_s1_06_mount_exclusion` | The cell mount relation. |
| `issue_s1_06_panda_inertia` | Vendored Panda inertia in principal axes. |
| `issue_s1_06_planner_boundary` | Planner configuration, header and wire types. |
| `issue_s1_06_planner_contract` | Public CLI with CPU planner doubles. |
| `issue_s1_06_planner_requests` | Generator of the pinned planner requests. |
| `issue_s1_06_planner_ros_build` | CPU preparation and the explicit ROS build gate. |
| `issue_s1_06_planner_ros_messages` | ROS message bounds and joint counts. |
| `issue_s1_06_planner_ros_node` | The separate planner ROS node. |
| `issue_s1_06_planner_runtime` | Planner runtime lifecycle and untrusted output. |
| `issue_s1_06_planner_transport` | The planner transport with a CPU child double. |
| `issue_s1_06_sequencer_binding` | The generated sequencer binding and its Rust process. |
| `issue_s1_06_sequencer_preparation` | Sequencer preparation on synthetic telemetry. |
| `issue_s1_06_tool_projection` | CPU tool projection adapter. |
| `issue_s1_11_robotics_vocabulary` | The shipped robotics vocabulary module. |
| `substep_admission_source` | Substep admission source checks and refusals. |

### `v02`

Features added after v0.1: deterministic float reductions, binary64, the cr-v1 math library, stable sort and spatial grid queries.

| Module | Covers |
| --- | --- |
| `issue_c4_cr_default` | `f32` `sin`, `cos` and `sqrt` use cr-v1 on every engine. |
| `issue_v02_f64_oracle` | The exact-rational binary64 oracle. |
| `issue_v02_f64_source` | Executable `f64` and the math library from ordinary source. |
| `issue_v02_fk_panda` | Panda forward kinematics in `f64` reproduces the reference. |
| `issue_v02_math_differential` | cr-v1 against an independent build of CORE-MATH. |
| `issue_v02_math_engines` | Every library function on every engine. |
| `issue_v02_math_msl` | cr-v1 binary32 for Metal. |
| `sort_fsum` | Programs that also sort or query space keep their float reductions. |
| `v02_fsum_cli` | Deterministic reductions through the real CLI. |
| `v02_fsum_oracle` | The fast reduction code against its reference semantics. |
| `v02_placement` | Placement for the v0.2 families. |
| `v02_sort_contract` | Sort contract, explain schema and oracle. |
| `v02_sort_device` | Sort device admission and CUDA/Metal differential. |
| `v02_sort_lowering` | The radix sort core and its boundaries. |
| `v02_spatial_checker` | Spatial checking and opaque affine ownership. |
| `v02_spatial_demo` | A point-cloud radius demo against an independent oracle. |
| `v02_spatial_oracle` | The all-pairs spatial oracle. |
| `v02_spatial_placement` | Spatial explain families and unmeasured placement. |
| `v02_spatial_reference` | Source-to-reference differential for spatial queries. |

### `gpu_cuda`

Every CUDA hardware test, one module per family. Each is `#[ignore]`d unless run with `--include-ignored`, and the binary is built only with `--features cuda`.

| Module | Covers |
| --- | --- |
| `a1_grid` | Grid adapters on a real CUDA device. |
| `a1_indexed` | Indexed source routing on CUDA. |
| `a1_segmented` | Segmented publication on CUDA. |
| `a1_tensor_dispatch` | Tensor dispatch on CUDA. |
| `eng_03_tier` | Fold-tier device test shared with Metal. |
| `eng_06_pipeline` | Asynchronous-copy pipelines on a GPU of compute capability 8.0 or newer. |
| `gnet_03` | The ordinary-source general-net ABI5 on CUDA. |
| `gnet_03_nvrtc_pipes` | The bounded NVRTC subprocess keeps only a capped log prefix. |
| `gnet_cuda_aot` | Compiled-ahead CUDA on the actual device. |
| `gnet_p0b` | General-net entry corpus against the CPU oracle. |
| `law_bench` | Law-collective calibration and sessions. |
| `law_collective` | Law-collective emitters on CUDA. |
| `m3_01` | A live CUDA round trip. |
| `m3_02` | Generated map and verifier rejections. |
| `m3_03` | Fused maps. |
| `m3_04` | Reductions, scans, sanitizer and cold-process runs. |
| `m3_05` | Forced and auto runs, module cache, hints, faults and CLI. |
| `m4_01` | Strict binary32 contraction. |
| `m4_01_nvrtc_modes` | NVRTC private modes and bounded warning logs. |
| `m4_02` | Transpose and stencil kernels. |
| `m4_03` | Shared tiled-contraction tests plus CUDA-only ones. |
| `m4_04` | Segmented tests plus CUDA admission refusals. |
| `m4_05` | Indexed families, fault paths and 2^28-element runs. |
| `m4_07` | Summary families and the largest scans. |
| `pair_reduction` | Pair-reduction tests shared with Metal. |
| `sort_fsum` | Sorting programs with float reductions. |
| `task_region` | The emitted fork-join task region. |
| `v02_fsum` | v0.2 reductions on a live device. |
| `v02_math` | cr-v1 bit for bit against independent oracles. |
| `v02_spatial` | Spatial tests shared with Metal and size lists. |

### `gpu_metal`

Every Metal hardware test, one module per family; `#[ignore]`d unless run with `--include-ignored` on Apple Silicon with `--features metal`.

| Module | Covers |
| --- | --- |
| `a1_grid` | Grid adapters on a real Metal device. |
| `a1_indexed` | Indexed source routing on Metal. |
| `a1_segmented` | Segmented publication on Metal. |
| `a1_tensor_dispatch` | Tensor dispatch on Metal. |
| `eng_03_tier` | Fold-tier device test shared with CUDA. |
| `gnet_03` | The ordinary-source general-net ABI5 on Metal. |
| `gnet_p0b` | General-net entry corpus against the CPU oracle. |
| `law_bench` | Law-collective pilot and calibration. |
| `law_collective` | Law-collective emitters on Metal. |
| `m3_06` | Live MIR dispatch, auto choice, CLI explain and device kernels (a directory). |
| `m4_01` | Strict binary32 contraction. |
| `m4_03` | Tiled contraction on Metal. |
| `m4_04` | Segmented tests plus Metal refusals. |
| `m4_07` | Summary families and look-back schedules. |
| `metal_aot` | Fresh processes run from built containers. |
| `metal_archives` | Fresh-process backend archive qualification. |
| `pair_reduction` | Pair-reduction tests shared with CUDA. |
| `sort_fsum` | Sorting programs with float reductions. |
| `task_region` | The emitted fork-join task region. |
| `v02_fsum` | v0.2 reductions on a live device. |
| `v02_math_msl` | cr-v1 binary32 on a real GPU against CORE-MATH. |
| `v02_spatial` | Spatial tests shared with CUDA and size lists. |

### Binaries outside the directory layout

| File | Covers |
| --- | --- |
| `tests/issue_54_family_lowerings.rs` | Families are lowerings of checked normal-form nodes: every recognizer choice on the corpus is the dispatcher's choice or a named refusal, and a forged form is refused before any step. |
| `tests/issue_ai_02_spec_examples.rs` | Runs every spec code block, checks every normative citation, validates `examples/index.json`, runs every example and checks documentation links. |
| `tests/issue_m3_04_cuda_reductions_scans.rs` | Reductions and scans lowered to the shared device IR, verifier rejections and the CPU shape corpus. |
| `tests/issue_m4_06_bounded_schedule_tuning.rs` | Bounded device schedule selection, hints and cost models. |
| `tests/issue_m3_06_metal/main.rs` | Metal emission on the CPU, and the missing-device error off Apple Silicon. |

### Tool and crate tests

| File | Covers |
| --- | --- |
| `tools/tests/bench_tooling/issue_eng_10_domain_suites.rs` | Benchmark domain suites. |
| `tools/tests/bench_tooling/issue_gnet_05_parity_report.rs` | The GNET parity report. |
| `tools/tests/bench_tooling/issue_m3_03_cold_process.rs` | Fresh-process CUDA timing control. |
| `tools/tests/bench_tooling/issue_m5_01_benchmark_manifest.rs` | Benchmark manifest validation and rejections. |
| `tools/tests/bench_tooling/issue_m5_01_build_lineage.rs` | Build lineage and bounded child builds. |
| `tools/tests/bench_tooling/issue_m5_01_heldout_generators.rs` | Held-out corpus generators and the exclusion from tuning. |
| `tools/tests/bench_tooling/issue_m5_02_device_qualification.rs` | Device rows of full-pipeline qualification. |
| `tools/tests/bench_tooling/issue_m5_02_full_pipeline_qualification.rs` | CPU semantics, denominators, fuzzing and mutation detection. |
| `tools/tests/bench_tooling/issue_m5_03_cuda_performance_acceptance.rs` | The performance gate arithmetic and refusal mutations. |
| `crates/invar-device/tests/m4_01_floating.rs` | Floating device contract versions and provenance. |
| `crates/invar-device/tests/m4_02_stencil.rs` | Stencil schedule physical mutations. |
| `crates/invar-device/tests/m4_02_transpose.rs` | Transpose version, provenance and capability mutations. |
| `crates/invar-metal/tests/issue_m4_02_metal_gpu.rs` | Transpose and stencil on a real Metal GPU. |
| `crates/invar-metal/tests/issue_m4_05_metal_gpu.rs` | Indexed families on a real Metal GPU. |

The remaining files in `tools/tests/bench_tooling/` (`qualification_faults.rs`, `qualify_cli.rs`) and the Python tests are named for what they test already.
