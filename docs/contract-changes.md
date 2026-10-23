# Frozen contract changes

The v0.1 contracts (schemas, code registries and the v0.1 benchmark corpus) are frozen.
A frozen contract changes only with a maintainer decision, recorded here with the files, the exact change, the reason, the effect on existing documents and the tests that pin the new bytes.
Without such an entry, a change ships under a new schema version instead.

## 2026-10-23: tracker references and the integration-start range owner

Decided 2026-10-23: schema descriptions cite this repository's issue tracker, and the S1 milestone is named integration-start.

Reason: the descriptions named issues by their numbers in the previous tracker, which the public repository does not have; the S1-06 milestone and its acceptance file are renamed to integration-start.

| File | Change | FNV-1a 64 (pinned) |
| --- | --- | --- |
| `schema/invar-diagnostics-v1.schema.json` | the `normal_form` description cites `#54` for the normal-form route | `0xdfa28d616759b86d` to `0x879d475c646c15bf` |
| `schema/explain-v1.schema.json` | the `device_execution.normal_form` description, the same | `0xd64e62b004f4341c` to `0x9583d18ffbf72f5e` |
| `schema/invar-explain-device-v1.schema.json` | the `normal_form` description, the same | `0x3be8b64f24eb50c9` to `0xbe4976c7c7b31a61` |
| `schema/invar-diagnostics-v3.schema.json`, `schema/explain-v5.schema.json`, `schema/invar-explain-device-v5.schema.json` | the same in their copies of `normal_form`; the `numeric_contracts.fold` description drops its tracker reference (diagnostics v3 and explain v5 only) | not pinned |
| `schema/diagnostic-codes-v1.json` | the `S1800`-`S1899` range owner is renamed to `integration-start acceptance`; the retired-code justification cites `#57` | not pinned |

Effect on existing documents: none; descriptions and range owners constrain no document, and no code, range bound or category changes.

## 2026-10-22: schema `$id` URLs move to the public repository

Decided 2026-10-22: every schema's `$id` now names its file in the public repository, `https://github.com/chayan-bit/invar/blob/main/schema/<file>`.

Reason: the previous `$id` host is not part of the public repository.

| File | Change | FNV-1a 64 (pinned) |
| --- | --- | --- |
| `schema/invar-diagnostics-v1.schema.json` | `$id` only | `0x17a9ac9464b5db80` to `0xdfa28d616759b86d` |
| `schema/explain-v1.schema.json` | `$id` only | `0x07cc9fed38031263` to `0xd64e62b004f4341c` |
| `schema/invar-explain-device-v1.schema.json` | `$id` only | `0xada00bfdc734e9b9` to `0x3be8b64f24eb50c9` |
| every other `schema/*.json` and `examples/index.schema.json` | `$id` only | not pinned |

Effect on existing documents: none; `$id` identifies the schema and constrains no document.

## 2026-10-21: fold tiers in `numeric_contracts` (ENG-03 step 7)

Decided 2026-10-21 (D12): the optional `fold` key goes only into the newest schemas that carry `numeric_contracts`, and the frozen older versions are unchanged.

Reason: numeric policy v2 lets a function declare how its f32 folds associate ([numeric contracts](numeric.md)), and the checker lowers each tiered fold to `sum_tree` or `sum_exact`. Relaxed folds, and the `reassoc_add` folds of a Strict function, take the Tree lowering (ENG-03 step 5'). A receipt must say which algorithm ran.

| File | Change | FNV-1a 64 (pinned) |
| --- | --- | --- |
| `schema/explain-v5.schema.json` | `numeric_contracts` gains the optional object `fold` | not pinned |
| `schema/invar-diagnostics-v3.schema.json` | `numeric_contracts` gains the optional object `fold` | not pinned |

`fold` maps each declared tier that a lowered fold used to the algorithm that ran: `tree` to `"tree"`, `exact` to `"exact"`, and `relaxed` to `"tree"` (the schema also admits `"exact"`, for a later cost-based choice). A Strict function's `reassoc_add` folds record as `relaxed`.
It appears only in placement documents, and only for a program with at least one lowered fold: the `invar-diagnostics-v3` receipt of `invar run --engine auto` and the top-level `invar-explain-v5` document of `invar explain --engine auto`, when placement owns auto.
The run receipt embedded in v5 as `execution` carries no `numeric_contracts` (unchanged); the top-level object describes the same program.

Effect on existing documents:
- The change only adds an optional key, so a document valid under the old bytes stays valid.
- Documents for programs without a lowered fold are byte-identical to before.
- Explain v1, v3 and v4 and diagnostics v1 and v2 never carry `fold`; their schemas are unchanged.
- `numeric_contracts` may now hold `fold` as its only key.

Tests that pin the new contract:
- `tests/issue_eng_03_fold_receipts.rs`: the receipt for each tier and their mix, none without a lowered fold; placement run and explain documents carry it and validate, and the established documents do not.

## 2026-10-21: `INVAR_NO_NORMAL_FORM` removed (#54, step 6)

Approved 2026-10-21 under #54's scope (after the five family lowerings, the opt-out is removed).

Reason: every family is now a lowering of the checked normal form (`src/schedule/families.rs`), so there is no second region source to fall back to; the variable only hid the `normal_form` object.
No schema changes and no digest moves.

Effect on existing documents:
- The environment variable is ignored. A `cpu-parallel` run receipt (and the explain device document built from it) carries the same `normal_form` object with or without it, so a document that the two entries below say `INVAR_NO_NORMAL_FORM=1` omits now gains the field as described there.
- Those entries are left as written; their opt-out sentences describe the behaviour before this date.

Tests that pin the new contract:
- `tests/v01_normal_form_run.rs`, `run_receipt_records_the_law_and_the_removed_opt_out_changes_nothing`: the receipt with `INVAR_NO_NORMAL_FORM=1` set equals the receipt without it.

## 2026-10-20: `no-lowering` fallback reason (#54, families as lowerings)

Approved 2026-10-20 under #54's scope (a certified loop that no family lowers records a typed fallback in the normal-form route receipt, with no new field).

Reason: `invar run`, `auto`, `explain`, `calibrate` and `build` now take regions only from the checked normal form, and each family is a lowering of its admissions (`src/schedule/lowering.rs`).
A loop the form certifies as parallel but no family lowers runs on the existing CPU path, and the run receipt names it.

| File | Change | FNV-1a 64 (pinned) |
| --- | --- | --- |
| `schema/invar-diagnostics-v1.schema.json` | `normal_form.reason` also accepts `^no-lowering:[a-z]+/[a-z]+(-[a-z]+)*$` | `0x9ef0ddb3fff29617` to `0x17a9ac9464b5db80` |
| `schema/explain-v1.schema.json` | `device_execution.normal_form.reason`, same alternative | `0x666c0e12c62fe194` to `0x07cc9fed38031263` |
| `schema/invar-explain-device-v1.schema.json` | `normal_form.reason`, same alternative | `0x1005fe866174dcb8` to `0xada00bfdc734e9b9` |
| `schema/invar-diagnostics-v3.schema.json`, `schema/explain-v5.schema.json`, `schema/invar-explain-device-v5.schema.json` | the same alternative in their copies of `normal_form` (#50, added after those schemas accepted `auto_path: "established"`) | not pinned |

The reason is `no-lowering:<shape>/<rep>`: `<shape>` is the loop's normal-form shape (`map`, `reduce`, `scan`, `scatter`, `frontier`, `first`) and `<rep>` the representation of its first carried value, else of its first array law, else `none` (for example `no-lowering:reduce/lifted-sum`).

Effect on existing documents:
- The enum only gains an alternative, so a document valid under the old bytes stays valid.
- A `cpu-parallel` run receipt (and the explain device document built from it) that had no `normal_form` object gains `{"status": "fallback", "reason": "no-lowering:..."}` when the program has such a loop, naming the first one. A program with an explicit sort or spatial intrinsic never gains it, since its explain documents are v3/v4, which have no `normal_form`. Every other document is unchanged. `INVAR_NO_NORMAL_FORM=1` still omits it.

Tests that pin the new contract:
- `tests/v01_normal_form_run.rs`, `issue_54_a_loop_no_family_lowers_records_a_typed_fallback`: the receipt names the gap and validates; `issue_54_a_sort_program_keeps_its_versioned_explain_document`: a sort program's explain v3 document stays valid.

## 2026-10-20: normal-form route in run receipts (#54, P6(a))

Approved 2026-10-20 (v0.1 scope decision for #54: one optional receipt field naming the law or the fallback reason).

Reason: `invar run --engine cpu-parallel` (and `--engine auto` on builds without a device) now executes a program that no family, f32, sort, spatial or reduction path claims through its checked parallel normal form (`src/schedule/normal_form.rs`).
The run receipt and the explain device receipt record whether that happened, with the applied laws and the certificate identity, or why the serial path ran.

| File | Change | FNV-1a 64 (pinned) |
| --- | --- | --- |
| `schema/invar-diagnostics-v1.schema.json` | New optional top-level `normal_form` object | `0x4ac8b2108563ce2f` to `0x9ef0ddb3fff29617` |
| `schema/explain-v1.schema.json` | New optional `device_execution.normal_form` object | `0xd8d709c96419c060` to `0x666c0e12c62fe194` |
| `schema/invar-explain-device-v1.schema.json` | New optional top-level `normal_form` object | `0x932dc53db127f944` to `0x1005fe866174dcb8` |

`normal_form` is closed and has one of two shapes:
- `{"status": "routed", "laws", "certificates", "fingerprint", "certificate_digest", "lanes"}`: `laws` lists the laws the law executor applied (`"L3"`), `certificates` counts the certificates the independent checker accepted, `fingerprint` is the compiled-MIR contract digest they are bound to and `certificate_digest` the SHA-256 of the checked certificate set (both lowercase hex);
- `{"status": "fallback", "reason"}`, with `reason` one of `recursion-law-not-routed`, `certificate-rejected`, `residue-law-not-routed`, `executor-refused`, `no-law-applied`, `law-state-violation`, `call-depth-unproven`, `fuel-unproven`, `live-elements-unproven`, `executor-tool-limit`.

Effect on existing documents:
- The change only adds an optional field, so a document valid under the old bytes stays valid.
- `normal_form` appears only for a program that has at least one certified law and reaches the route; every other document is byte-identical to before.
- `INVAR_NO_NORMAL_FORM=1` disables the route, and with it the field.

Tests that pin the new contract:
- `tests/v01_normal_form_run.rs`: routed receipts and explain documents validate, the opt-out and law-free programs carry no field.

## 2026-10-19: strict-f64-v1 and invar-math-cr-v1

Approved 2026-10-19.

Reason: v0.2 makes `f64` executable under `strict-f64-v1` and adds the correctly rounded `invar-math-cr-v1` library.
Explain and run documents must record which numeric contracts a program used, and Metal explain must say why `f64` code stays on the CPU.
The maintainers chose to extend the v1 documents in place rather than emit these fields only under new schema versions.

| File | Change | FNV-1a 64 (pinned) |
| --- | --- | --- |
| `schema/explain-v1.schema.json` | New optional top-level `numeric_contracts` object (keys `binary64`, `math`, `trig`); device choice `reason` enum gains `metal-no-f64` | `0x793b64fd10b13796` to `0xd8d709c96419c060` |
| `schema/invar-explain-device-v1.schema.json` | Choice `reason` enum gains `metal-no-f64` | `0xff5e8378b7374b32` to `0x932dc53db127f944` |
| `schema/invar-diagnostics-v1.schema.json` | New optional top-level `numeric_contracts` object (keys `binary64`, `math`, `trig`) | `0x22e00df7bbdad02d` to `0x4ac8b2108563ce2f` |
| `schema/invar-diagnostics-v2.schema.json` | New optional top-level `numeric_contracts` object | not pinned |
| `schema/explain-v3.schema.json` | Same two additions as v1, so sorting programs that use `f64` or a library function validate | not pinned |
| `schema/invar-explain-device-v3.schema.json` | Choice `reason` enum gains `metal-no-f64` | not pinned |
| `schema/diagnostic-codes-v1.json` | Reason registry gains `metal-no-f64` | not pinned |
| `schema/diagnostic-codes-system-v1.json` | `S1008` ("floating arithmetic is unavailable before M4-01") becomes `deprecated: true`; float literals are valid source, so system parsing no longer emits it | not pinned |
| `bench/general-net-v01/numeric-vectors.json` | Case `variant_f32_source_rejected` is replaced by `variant_f32_array_source_rejected` because `f32` and `f64` variant payloads now check; see [NUMERIC.md](../bench/general-net-v01/NUMERIC.md) | source SHA-256 in the vector |
| Source names (`src/check.rs`) | The 29 `invar-math-cr-v1` function names (`sqrt` through `max`) are reserved for functions: a user `fn` with one is rejected with `NAME_DUPLICATE` naming the library function; a variable may share one, and a call such as `round(x)` still calls the library (ruling, 2026-10-19) | not pinned |

`numeric_contracts` has at least one and at most three keys: `binary64: "strict-f64-v1"`, `math: "invar-math-cr-v1"` and `trig: "invar-trig-f32-v1"`.
It appears only when a program uses `f64` or a library function.
`trig` records main's `invar-trig-f32-v1`, which f32 `sin`, `cos` and `sqrt` keep under the C4 interim ruling (2026-10-19): `invar-math-cr-v1` applies to `f64` and every other library function until the tracked C4 follow-up makes it the f32 default.
The reason was spelled `metal_no_f64` during development and renamed to `metal-no-f64` before merge to match every other reason in the enum (decision, 2026-10-19); no released document ever carried the old spelling.

Breaking for v0.1 source: a user function named like a library function no longer checks.
On main at the time of the change, two test programs did so and were renamed: `round` to `nudge` in `tests/issue_m4_01_source_f32.rs`, and `round` to `expand_round` in `tests/net_normal_form_laws.rs` (L14).
No example, corpus program or spec sample defines one, and variables named `round`, `min`, `max` or `abs` (the GNET-05 frontier corpus uses `var round`) keep working.

Effect on existing documents:
- Every change only adds an optional field, an enum value or a deprecation, so a document valid under the old bytes stays valid.
- Documents for programs that use neither `f64` nor a library function are byte-identical to before.
- Readers with an exhaustive reason enum must accept `metal-no-f64`.
- `S1008` is no longer emitted.

Tests that pin the new contract:
- `tests/issue_ai_01_machine_diagnostics.rs`, `source_codes`: deprecated codes are no longer required to be reachable from source.
- `tests/fixtures/s1_01/negative/S1008-f64-arithmetic.inv` is removed, because the program it rejected is now valid source.
- `bench/general-net-v01/corpus_cpu.rs`: the integer/owner source corpus excludes the `F64` rule beside `F32`.
