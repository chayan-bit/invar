# admission

Admission analysis: finding counted loops in MIR and deciding, obligation by obligation, whether each may run as a parallel plan.
Its report is the data behind `invar explain`.

## Where to start

- `analyze(source, &program)` in [../admission.rs](../admission.rs) returns a `Report` of `Region`s; `Report::to_json` writes the `invar-explain-v5` document; `validate_serialized` re-analyzes a stored record.
- `RULE_REVISION` names the map rule; `Guard`, `MapShape` and `Status` describe the result; `SCHEMA` and `DEVICE_SCHEMA` name the receipt schemas.
- [fold_hint.rs](fold_hint.rs) adds the v0.2 hint for serial binary32 folds to rejected regions.

## Position in the pipeline

After [mir](../mir/README.md); `plan::decide` in [../plan.rs](../plan.rs) classifies the admitted regions. See the code tour step 5 in the [architecture code tour](../../docs/architecture.md#code-tour-one-program-end-to-end).

## Tests

- `cargo nextest run --locked --test schedule_cpu` (`issue_m2_01` covers admission records and `explain`).
- `cargo nextest run --locked --test byte_budgets` covers admission by bytes.
- `cargo nextest run --locked --test cli` covers `explain` receipts.

## Rules to keep

- An unknown obligation rejects the region; unknown evidence kinds and schema versions fail closed.
- Evidence binds to the exact source bytes, rule revision and policy.

## Read next

- [explain JSON](../../docs/explain.md)
- [glossary: evidence kinds](../../docs/glossary.md#e)
