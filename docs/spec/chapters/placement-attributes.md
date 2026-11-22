### B.5 Placement profiles

[NB.5.1] `invar calibrate [--out NEW_PATH] [--family placement.wrapping_add1.u32] [--host-class CLASS]` measures the single family `placement.wrapping_add1.u32` and publishes a new provisional `invar-placement-v1` profile, by default at `calibration/<os>-<arch>.json`, and never replaces an existing path. `tests/fsum_math_sort_spatial/placement.rs::calibrate_cli_exports_a_valid_provisional_profile` `tests/fsum_math_sort_spatial/placement.rs::placement_bounded_loader_atomic_publish_and_no_clobber`
[NB.5.2] `run` and `explain` with `--engine auto` accept `--profile PROFILE`, `--host-class CLASS` and `--trust-provisional`, and a provisional profile influences placement only with `--trust-provisional`. `src/lib.rs::--trust-provisional` `spec/placement-v1.md`
[NB.5.3] Placement owns `--engine auto` only when one of those flags is given or the live host's default profile is not a matching provisional one, and every other auto run keeps the established path of N11.5.6, whose documents take the versions of NB.5.6. `src/auto_command.rs::placement_requested` `tests/fsum_math_sort_spatial/placement.rs::auto_stays_on_the_established_path_unless_placement_is_asked_for`
[NB.5.4] A profile whose identity differs from the live host, CPU, device, driver, compiler build or LLVM version is refused with `TUNING_HINT_INVALID`, a frozen profile is refused with `TUNING_FREEZE_REQUIRED` until M5-01 provides a trusted freeze manifest, and both keep the CPU. `spec/placement-v1.md` `tests/fsum_math_sort_spatial/placement.rs::real_cli_profile_trust_identity_contract_and_cpu_decision`
[NB.5.5] A device is chosen only when its total cost is below 0.95 of the CPU cost, an unknown cost keeps the CPU, and only the measured family can leave the CPU. `src/placement.rs::select` `spec/placement-v1.md`
[NB.5.6] A placement run receipt carries a `placement` object, and `explain --engine auto` under placement adds `placement` and `execution`, both recording `auto_path` as `placement`; an established auto run has no `placement` and records `auto_path` as `established`. `docs/explain.md` `schema/explain-v6.schema.json`

### B.6 Attributes, contract axes and receipts

Note: the attribute registry, the contract axes and their receipts are specified in the v0.2 chapter `docs/spec/invar-v0.2-axes.md`, which has its own size record and budget.

