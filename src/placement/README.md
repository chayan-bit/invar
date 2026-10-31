# placement

Measured placement hints: calibration profiles and the cost decisions of `--engine auto`.

## Where to start

- `calibrate`, `host_identity` and `live_identity` in [calibration.rs](calibration.rs) run the explicit offline probes; [../placement.rs](../placement.rs) validates and loads `invar-placement-v1` profiles (`MAX_PROFILE_BYTES`, `FAMILIES`).
- The CLI is [../calibrate_command.rs](../calibrate_command.rs) and [../auto_command.rs](../auto_command.rs).

## Position in the pipeline

Read by [schedule](../schedule/README.md) (`profiled.rs`); `--engine auto` consults it.

## Tests

- `cargo nextest run --locked --test device` (`placement_calibration`), `--test cli` (`issue_50_*`), `--test fsum_math_sort_spatial` and `--test normal_form`.

## Rules to keep

- A decision describes costs only; it never grants execution or publication authority, and unmeasured families stay CPU.

## Read next

- [placement profile v1](../../spec/placement-v1.md)
