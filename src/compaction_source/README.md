# compaction_source

Recognition of ordinary-source stable `u32` compaction, with a bounded local CPU candidate and allocation accounting.

## Where to start

- `Predicate`, `Contract`, `Descriptor` and `Prepared` in [../compaction_source.rs](../compaction_source.rs) describe admitted work; [allocation.rs](allocation.rs) diagnoses allocation sites and reported budgets.

## Position in the pipeline

A source family: the registry in [schedule](../schedule/README.md) offers it each region (`schedule/families/segmented.rs`).

## Tests

- `cargo nextest run --locked --test m4_segmented`.

## Rules to keep

- Recognition never changes execution; a near match is rejected.
