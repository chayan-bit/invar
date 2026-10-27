# Explain JSON

`invar explain PATH --format json` prints one `invar-explain-v5` document, with `schema_version: 5`, describing how each region of a checked program is admitted.
With `--engine cuda|metal|auto` the document also carries `device_execution`, whose `device_schema` is `invar-explain-device-v5`.
The strict schemas are `schema/explain-v5.schema.json` and `schema/invar-explain-device-v5.schema.json`.

## Static report

`invar explain INPUT --format json` reads and checks one Invar source file, lowers it to verified MIR, then prints one JSON document.
Successful analysis exits 0 even when every region is rejected.
Source and tool errors use the existing `source_error`/`tool_error` documents and exit codes 1/2.

The success document has `schema_version: 5`, `status: "ok"`, `diagnostics: []`, `source_identity`, and `regions` in source/MIR order.
`source_identity` is `fnv1a64:` followed by 16 lowercase hexadecimal digits from FNV-1a (64-bit offset `0xcbf29ce484222325`, multiplier `0x100000001b3`) over exact source bytes.
This checksum establishes identity, not authenticity.

Each region has `function`, byte `span`, `candidate_identity`, `status` (`admitted` or `rejected`), `first_rejected_obligation` (null on admission), `rule_revision`, `rule_identities`, `numeric_policy`, `target`, `chosen_plan`, `fallback_plan`, `runtime_guard`, `facts`, and ordered `obligations`.
An obligation has `name`, `evidence`, and `detail`; evidence is `checked_invariant`, `trusted_rule`, `runtime_guard`, or `unknown`.
Unknown evidence rejects admission.
`candidate_identity` is the same FNV-1a over region source bytes followed by UTF-8 rule revision and `wrapping-u32-v1/cpu`.

The initial candidate is a zero-based counted wrapping `u32` map with one or two ordered reads and one injective fresh-output store.
`facts` records typed logical reads/writes, ownership, loans, effect order, trap/termination summaries, physical alias assumption, and cleanup policy.
For rejected regions, unestablished facts and the guard are null.
The guard contains `upper` (a constant or named array length), ordered `inputs`, and `output`.
It requires the upper bound to be at most every current array length and the map's peak host footprint to fit the host byte budget (inputs plus three more `u32` arrays of that length, #57); it is evaluated before any candidate write while the protected arrays remain stable.
False or missing guard inputs select `mir-serial`.

Persisted documents are descriptive admission inputs.
`admission::validate_serialized` bounds their size to 1,000,000 JSON bytes and re-analyzes current verified MIR and source bytes; any changed source, rule, policy, fact, or obligation rejects the record.
A checksum or successful validation does not prove semantic equivalence by itself.
Later extraction and execution must recheck the obligations at their own boundary.

`f64` programs add the optional `numeric_contracts` object and the device reason `metal-no-f64`.

## Explicit sorts

A report with a sort site lists it in `explicit_sorts`.
Each checked `explicit_sorts` site adds a required `calibration_id` with the C14 family spelling `sort.<op>.<type>`, such as `sort.argsort.f32` or `sort.sort_by_key.u32`.
Operations remain overloaded by their key array type.
These ids identify calibration families; they do not assert measured costs.

`sort-cost-unmeasured` is registered in `explain_reason_codes` in `schema/diagnostic-codes-v1.json`.
Automatic execution still selects CPU with no cost estimate.

## Spatial sites

`explicit_spatial` lists checked sites in MIR order, including unreached branches.
Each site records its function, operation, source span, `cpu-sorted-grid` default, `explicit-spatial-v1` rule, and registered `spatial-cost-unmeasured` reason.
The calibration identities are:

| Operation | Family |
| --- | --- |
| `voxel_grid_build` | `spatial.voxel_grid_build.f32` |
| `radius_neighbors` | `spatial.radius_neighbors.f32` |
| `knn` | `spatial.knn.f32` |
| `grid_excluded` | `spatial.grid_excluded.f32` |

The strict schema rejects unknown fields, operations, reasons, and mismatched operation/family pairs.
Checked sites are not execution or calibration samples.

With `--engine cuda|metal|auto`, reached spatial choices carry the same family identity.
Forced kernels retain their device backend, implementation, and `forced-device` reason.
Host metadata, empty operations without a launch, and explicit host sorts report `cpu-serial` with `spatial-host-work` instead of a device launch.
Unmeasured automatic spatial execution stays on CPU with no invented cost, even if the generic bandwidth model favors the GPU.
Independent benchmark qualification and the 1.10x gate remain prerequisites for automatic device placement.

## Auto engine

Under `--engine auto`, the explain document and the `invar run PATH --engine auto --format json` receipt carry `auto_path`, which names the path that ran the auto engine: `"placement"` or `"established"`.
Other explains have no `auto_path`.
Placement owns it only when the user asks for it: `--profile`, `--host-class` or `--trust-provisional`, or a calibration profile for the live host at `calibration/<host-class>.json`.
A matching provisional profile at the default path never switches auto by itself.
A frozen, mismatched or malformed one does, so placement reports its refusal or typed error instead of skipping it.

Two objects are required when `auto_path` is `"placement"` and absent when it is `"established"`:
- `placement`: the shared placement receipt ([placement-v1](../spec/placement-v1.md)), with the profile identity, the CPU and device costs, and each region's choice.
- `execution`: the run document of the same execution, an `invar-diagnostics-v3` receipt.

On the reference route (`placement.route.host` is `"reference"`), the receipt also lists `placement.reductions`, the reached float reductions, and `placement.intrinsics`, the reached sort and spatial sites.
Both hold the values the established auto path records in `device_execution.choices`, and the explain lists the same entries in its own device choices (#50).
The native route runs compiled code that keeps no per-call receipts, so its receipt carries neither list.
A run that stops with an error (`placement.engine` is `"none"`) carries neither list either; both schemas enforce these rules by route.
`explicit_sorts`, `explicit_spatial` and `numeric_contracts` stay optional and unchanged, so the intrinsic sites stay visible under placement.
`numeric_contracts` may also carry `fold`, which the top-level explain document and the `invar-diagnostics-v3` receipt of `invar run --engine auto` add: each declared fold tier a lowered fold used, to the algorithm that ran ([numeric contracts](numeric.md#receipts)).
The receipt embedded as `execution` carries no `numeric_contracts`.
The device execution may carry the placement reasons `measured-cpu-cost` and `measured-device-cost` and the optional `normal_form` route.

Every placement document, the explain and its `execution` receipt alike, records `auto_path: "placement"`.
Every established document records `auto_path: "established"` and has no `placement`, and neither does its run receipt.

## Device execution

`invar explain PATH --engine cuda|metal|auto --format json` extends the static report with `device_execution`.
It runs the same guarded MIR plan once with the default 1,000,000 fuel limit and reports choices actually applied in that run.
Invar source has no external I/O.
The plain `explain PATH --format json` interface keeps its static report.

`device_execution.requested_engine` is `cuda`, `metal`, or `auto`; `result` is the reached `main` i32.
Each `choices` entry names the function and source span, backend, reason, and (for `auto`) estimated CPU/device milliseconds.
A forced device choice is `cuda` or `metal` only after a completed result passes the live MIR publication guard.
`auto` may choose `cpu-parallel` from the measured cost model, or `mir-serial` when a live guard is false.
The model measures CPU cost from the real cpu-parallel element executor on a one-operation 1,000,000-element map, device fixed cost from the submit-to-completion time of a trivial one-thread kernel, and transfer bandwidth from a host/device buffer round trip.
It then compares transfer bytes, launch count and operation count with a 10% device-win margin.
Compilation is excluded because compiled modules are reused in-process.
Metal caches measured models per CPU worker count.
CPU workers use available parallelism, preferring the number of performance cores on Apple Silicon, with a hard bound of 64.
`INVAR_CPU_WORKERS=1..64` overrides that preference within the OS affinity/quota limit; invalid values return `CPU_WORKERS`.
Explicit `cpu-parallel --workers` requests accept 1..64.
Calibration measures the actual selected CPU width.
`module_cache_hits`, `schedule_hint_hits` and `schedule_hint_invalids` are counts for this process.

The execution path always reconstructs the region via `plan::decide`, lowers the graph-extracted plan into the shared device IR, and reruns the verifier.
The CUDA module cache contains at most eight live-device entries keyed by exact verified IR, capabilities, generated source and kernel names.
The Metal pipeline cache is one bounded LRU per device, shared by the map, scan and general-net routes.
It holds at most 16 pipelines keyed by the SHA-256 of the exact generated MSL and the function name, so two different kernels never share an entry and the three scan passes share one library.
Persistent schedule hints contain only bounded geometry and descriptive identity: exact source FNV-1a identity, typed contract, rule revisions, pinned toolchain, device capability, layout, numeric policy, ABI and generated source hash.
Untrusted hints are compared with a freshly regenerated record before use.
No persisted source, PTX, CUBIN, Metal library or module is loaded from a hint.

Metal kernels are compiled ahead.
`invar build PATH --target metal --output FILE` runs Apple's offline compiler (`xcrun -sdk macosx metal -std=metal3.1 -fno-fast-math -mmacosx-version-min=14.0 -c`, then `metallib`) over every Metal region kernel and the calibration probe.
It writes a private, atomically published `INVGMETL` container; the general-net route adds `--entry NAME`.
`run`, `explain`, `gnet-device` and tensor `source-device ... metal` take `--artifact FILE` and load its libraries with `newLibraryWithData`; production never calls `newLibraryWithSource`.
The tensor route builds with `invar build PATH --target metal --entry ENTRY --family tensor --fuel FUEL --output FILE`: an exact CPU run on this Mac's Metal capabilities records the D1 probe and every shape-specific contraction the source reaches within `FUEL`.
A contraction missing from that container (another Mac's capabilities, or more fuel at run time) is a recorded `tensor_submit` refusal and that loop runs on the CPU.
Without an artifact, a forced Metal run fails with `METAL_UNAVAILABLE` and `auto` records `device-unavailable` and runs on the CPU, as on a machine without a GPU.
A corrupt, oversized, stale, wrong-route, wrong-entry, group- or world-writable or toolchain-rejected container, fails with `DEVICE_TOOLCHAIN_MISMATCH` under forced Metal and `auto` alike, and never compiles source.
A kernel missing from an accepted container is refused the same way when forced; under `auto` it is the recorded `metal-failed-before-publication` CPU result.
A build without Metal refuses `--artifact` with `METAL_UNAVAILABLE` before choosing a backend.
The container is opened once without following a final symlink or blocking on a FIFO, and every check and the read use that descriptor.
`INVAR_METAL_RUNTIME_SOURCE=1` is the developer-only runtime source path, off by default.
The container digest gives integrity, not authenticity.

A region whose Metal kernel would bind more than 31 buffer arguments (the Apple GPU limit) is Metal-ineligible at planning time: `auto` records reason `metal-buffer-slot-limit` and runs it on CPU.
Metal has no binary64, and Invar does not emulate it: under `--engine metal` or `auto` on Metal, every function that computes an `f64` value is listed once, at its first such instruction, with backend `mir-serial` and reason `metal-no-f64`, and runs on the CPU.
Forced CUDA or Metal fails with `CUDA_UNAVAILABLE` or `METAL_UNAVAILABLE` for missing hardware, ineligible regions or false live guards; compilation/verification mismatch uses `DEVICE_TOOLCHAIN_MISMATCH`.
Unknown completion uses `CUDA_PENDING` or `METAL_PENDING` and permits no CPU replay.
`auto` can fall back to a fresh CPU result only before device submission or after confirmed quiescence.
All failures use the existing `tool_error` document and exit code 2.

### Python binding

The CLI and Python binding share `invar::engine` for guarded whole-program selection.
`compile_source(..., engine="auto"|"cuda"|"metal").run_main(observe=True)` returns a receipt from that completed invocation.
`receipt()["backends"]` lists published region backends, or `mir-serial` when no region was published.
`explain()` includes the `device_execution` document after a successful device/auto run and does not execute again.
Before execution it returns the static explanation without a device execution claim.
HA-1.2 NumPy host exports keep their separate reference/native-serial boundary.
