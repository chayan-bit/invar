# check

Name resolution, type checking, units and frames, modules, generics and closure desugaring.
It turns a parsed `Program` into a `CheckedProgram` or a source diagnostic.

## Where to start

- `check(program, require_main)` in [../check.rs](../check.rs) is the entry point; it ends by running `ownership::check` ([../ownership.rs](../ownership.rs)) and `views::check`.
- [modules.rs](modules.rs), [generics.rs](generics.rs), [closures.rs](closures.rs) and [record_generics.rs](record_generics.rs) handle imports, monomorphization and affine closures.
- [units.rs](units.rs), [unit_math.rs](unit_math.rs) and [frames.rs](frames.rs) check physical quantities and coordinate frames; their tags are erased before execution.
- [numeric.rs](numeric.rs), [storage.rs](storage.rs), [reductions.rs](reductions.rs) and [literals.rs](literals.rs) implement the numeric policy declarations, storage-only types, deterministic float reductions and literal typing.
- [views.rs](views.rs) computes lexical view lifetimes.

## Position in the pipeline

Second stage, after [front](../front/README.md). Its output feeds [eval](../eval/README.md) and [mir](../mir/README.md).

## Tests

- `cargo nextest run --locked --test language` covers checking, ownership, views, generics, closures, units and bytes.
- `cargo nextest run --locked --test numeric_tiers` and `--test fsum_math_sort_spatial` cover the numeric and reduction checks.

## Rules to keep

- Preserve stable diagnostic codes and spans.
- No implicit conversions and no implicit copies.

## Read next

- [specification sections 3 to 7](../../docs/spec/invar-v0.1.md)
- [numeric contracts](../../docs/numeric.md)
