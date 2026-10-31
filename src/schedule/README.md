# schedule

Execution of admitted regions on CPU workers, CUDA and Metal, with staged publication and a serial fallback.

## Where to start

- `run_source` and `run_source_with_hooks` in [mod.rs](mod.rs) run the MIR serially and publish an admitted region's result when its loop is reached with true guards.
- [families.rs](families.rs) is the ordered registry that offers each region to the family adapters in [families/](families/) (`core`, `tensor`, `grid`, `segmented`, `indexed`).
- [lowering.rs](lowering.rs) is the single dispatcher for `run`, `auto`, `explain`, `calibrate` and `build`, and `regions` builds the explain report; [normal_form.rs](normal_form.rs) routes serial-only programs through the general-net normal form.
- [workers.rs](workers.rs) has bounded scoped workers (`MAX_WORKERS`, `parallel_chunks`); [folds.rs](folds.rs) and [reductions.rs](reductions.rs) run reductions, pair reductions and scans.
- [cuda.rs](cuda.rs) and [metal.rs](metal.rs) run forced and auto device execution; [device.rs](device.rs) holds the run records both share; [tuning.rs](tuning.rs) and [tuning_resident.rs](tuning_resident.rs) select bounded device schedules; [profiled.rs](profiled.rs) applies calibration profiles.

## Position in the pipeline

After [admission](../admission/README.md) and [plan](../../src/plan.rs); driven by `engine::run_mir` in [../engine.rs](../engine.rs). See the code tour step 7 in the [architecture code tour](../../docs/architecture.md#code-tour-one-program-end-to-end).

## Tests

- `cargo nextest run --locked --test schedule_cpu` (workers, fallback, every family and worker count).
- `cargo nextest run --locked --test device` and `--test family_lowerings` for device plans and family dispatch.
- [tests.rs](tests.rs) holds unit tests; CUDA and Metal hardware suites are `tests/gpu_cuda` and `tests/gpu_metal` (ignored by default).

## Rules to keep

- A false runtime guard selects the serial plan before any visible write.
- Forced device engines never fall back to the CPU silently; unknown completion permits no CPU replay.

## Read next

- [bounded schedule selection](../../docs/bounded-schedule-dispatch.md)
- [CUDA target](../../docs/cuda-target-v1.md)
- [specification section 11](../../docs/spec/invar-v0.1.md#11-execution-engines)
