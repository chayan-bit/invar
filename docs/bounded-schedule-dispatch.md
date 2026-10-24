# Bounded schedule selection and residency (M4-06)

This branch implements the class (a) machinery on existing M3 paths. It does
not qualify performance. M5-01's permanent held-out freeze, tuning measurements,
and the five M4 family adapters remain separate prerequisites. No family adapter,
recognizer or MIR region publication semantics are changed.

## Admission and selection

`src/schedule/tuning.rs` owns the backend-independent `Schedule`, `CostRecord`,
`Decision`, `Refusal` and budget types. GNET P6 may consume these records without
matching family names, filenames or benchmark IDs. `enumerate` consumes at most
33 iterator entries and refuses the entire request above 32, before lowering or
compilation. Input order is deterministic, with fixed seed, 10 warmups and 30
samples recorded for the later offline driver. There is no runtime variant search.

Every variant starts from the current typed region extraction and passes the
physical device verifier, including numeric, layout, alias, geometry, barrier
and completion contracts. M3 admits scalar vectors and 1D geometry. Nontrivial
tiles, vector widths, fusion cutoffs and frontier chunks explicitly refuse until
the owning family supplies its implementation; they are not approximated.
`select` compares admitted estimates and breaks equal costs by enumeration order.
The existing analytic model does not distinguish equal-work geometry variants.

The ordinary CUDA/Metal providers attach M3 cost records to their Rust `Choice`
and use the common decision function. The CLI explain schema retains its
compact `cost_ms` summary from that same estimate. Full records are available
through `CostRecord::to_json` and the Rust receipt API; the CLI schema is unchanged. Their existing
family registry cost hooks remain available for new A1 families.
An unavailable full record stays absent, rather than inventing workspace/layout
facts. Ordinary MIR owners remain host buffers. The existing measured host model
calibration is retained; calibration is not schedule search or a frozen tuning
result. CPU/serial source fallbacks, forced-device errors, staged consumers and
completion/publication guards retain their existing behavior.

## Records and real residency

A cost record describes logical shape/layout and numeric policy, input/output
residency and bytes, model coefficients, element/operation counts, transfer bytes, actual planned launch count, temporary
memory, device completion, generated/library-assisted track and three optional
timing boundaries. Estimates and measurements are separate. Invalid/missing
facts produce a typed inconclusive result. Revalidation binds the reported model
and work counts to the estimates, so a stale or forged descriptive record cannot
silently change the decision. The M3 map/DAG record has one launch for nonempty work and only
Metal's four-byte length buffer as temporary device storage (CUDA passes it by value). Collective workspace and launch counts
come from checked shared physical-IR resource facts in `invar-device`; Metal's
allocation helper uses that same workspace formula. Chunked Metal scan counts
three launches per chunk. M4 dimensions are not inferred from iteration count.

Host execution charges input upload and output retrieval. Retained device input
and retained device output charge neither within that boundary; a CPU alternative
must retrieve device input. `CostRecord::validate` rejects transfer accounting
that disagrees with residency. Both costs include the same CPU computation.

`invar-metal` and `invar-cuda` expose opaque `ResidentMap` handles: prepare once,
execute on the retained allocations, then read completed output explicitly.
These are real GPU buffers, not a residency flag on a host `Vec` runner.
CUDA keeps initialized pinned staging and consumes the handle for transfers as
well as launches. Each transfer has a bounded completion event; uncertainty
quarantines the handle and every pinned/device allocation. Setup
upload/compilation and final retrieval are outside a resident launch. CUDA binds
the handle to its original stream; Metal checks the actual device. Handles expose
no device pointer or mutable storage. Unknown completion invalidates output access
and uses the existing quarantine to retain every buffer/module until quiescence.

`src/schedule/tuning_resident.rs` connects common placement to these handles.
Auto selects a profitable admitted GPU plan; forced CPU/serial evaluates the
same typed graph after retrieving inputs. Host results and resident results are
explicit alternatives. They are raw completed arrays, not source-publication
permits: a future GNET caller must still use its source/MIR publication protocol.
The cost is the considered device candidate; CPU selection leaves that candidate
`NotSubmitted`. Only actual device completion changes it to `Completed`.

A raw event interval is reported separately as `kernel_ms`; it does not fill
steady-state, warm host-to-host or cold-process fields. Those remain unmeasured
until the frozen protocol runs. Hardware correctness tests use a declared
synthetic model to make the residency transition deterministic; they establish
execution and accounting, not a measured crossover or performance qualification.

## One existing hint cache

`src/device_cache.rs` extends the M3 cache and retains its file/byte quotas,
invalid-hint state, atomic replacement and deterministic eviction. No second cache
or persistent executable is added. Keys now use SHA-256 over the full identity:
exact source bytes, structural typed/physical contract, interaction rule IDs,
extraction rule revision and extraction/IR schemas, numeric and layout policies, target capabilities, pinned
toolchain/ABI, generated artifact hash, schedule version/index/geometry and
options (tile, vector width, fusion cutoff, frontier chunk, threshold and fixed
protocol). The path contains the derived key only. Older hints miss naturally.

The existing core path reconstructs and verifies its plan before every lookup.
For enumerated schedules, `revalidate` reparses/checks/lowers current source,
rebuilds admission at the current function/span, re-extracts and physically
verifies the selected schedule, regenerates the artifact/identity and compares
the entire hint. A matching hash alone is insufficient. Corrupt, stale, oversized
or forged records never carry executable authority.

## Refusals and remaining gates

Stable `TUNING_*` refusals cover excess/empty variants, unsupported axes,
admission/numeric failure, stale/forged hints, unknown costs, false guards,
missing devices, transfer/launch failures, unknown completion, visible production
output, runtime tuning, missing freeze and exhausted budgets. Auto uses lawful
CPU/serial fallback only before submission or after confirmed quiescence. Forced
GPU refuses; partial completion is fatal and cannot replay on CPU.

`measurement_gate` currently refuses all measurement requests. Production-visible
and runtime uses have their own refusals; offline use requires the permanent
reviewed M5 freeze to be wired into this boundary by a later change. An arbitrary
runtime flag or caller-supplied hash cannot bypass it. `Budget::check` provides
compile/time/workspace limits for that future driver; no actual tuner runs now.

Tests live in `tests/issue_m4_06_bounded_schedule_tuning.rs`. CUDA hardware checks
are feature-gated, ignored and named `issue_m4_06_gpu`; Metal uses
`issue_m4_06_metal_gpu`. Explicit invocation requires real hardware and fails
when it is absent. Synthetic choices and ignored tests never close a hardware or
performance gate. The classification table records the remaining (b)/(c) work.
