# Explain JSON v5

`invar explain PATH --engine auto` emits `invar-explain-v5` and `schema_version: 5`, and `invar run PATH --engine auto --format json` emits `invar-diagnostics-v3`, whichever path runs the auto engine (#50).
`auto_path` names that path: `"placement"` or `"established"`.
Placement owns it only when the user asks for it: `--profile`, `--host-class` or `--trust-provisional`, or a calibration profile for the live host at `calibration/<host-class>.json`.
A matching provisional profile at the default path never switches auto by itself.
A frozen, mismatched or malformed one does, so placement reports its refusal or typed error instead of skipping it.

v5 is v4 plus two objects, required when `auto_path` is `"placement"` and absent when it is `"established"`:
- `placement`: the shared placement receipt ([placement-v1](../spec/placement-v1.md)), with the profile identity, the CPU and device costs, and each region's choice.
- `execution`: the run document of the same execution, an `invar-diagnostics-v3` receipt.

`explicit_sorts`, `explicit_spatial` and `numeric_contracts` are optional and appear exactly as in v3, v4 and v1, so the intrinsic sites stay visible under placement.
`numeric_contracts` may also carry `fold`, which the top-level v5 document and the `invar-diagnostics-v3` receipt of `invar run --engine auto` add: each declared fold tier a lowered fold used, to the algorithm that ran ([numeric-v2](numeric-v2.md#receipts)). The receipt embedded as `execution` carries no `numeric_contracts`, as before.
The device execution follows `invar-explain-device-v5`, which is device v4 plus the placement reasons `measured-cpu-cost` and `measured-device-cost`, and the optional `normal_form` route of device v1 (#54).

Every placement document, the v5 explain and its `execution` receipt alike, records `auto_path: "placement"`.
Every established document records `auto_path: "established"`: it is the v1, v3 or v4 document of the same run with only the version fields moved to v5 (`schema`, `schema_version`, `device_schema`), and its run receipt is `invar-diagnostics-v3` without `placement`.
Explains without `--engine` and forced CUDA or Metal explains keep v1, v3 and v4, and archived documents of every version stay readable against their own schema.

The strict schemas are `schema/explain-v5.schema.json` and `schema/invar-explain-device-v5.schema.json`.
