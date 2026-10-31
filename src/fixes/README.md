# fixes

Validated source fixes for diagnostics, used by `invar check PATH --fix [--write]`.

## Where to start

- `generate` and `apply` in [../fixes.rs](../fixes.rs): a generator proposes ordered candidate edits for one diagnostic, and a candidate is accepted only if the edited source re-checks clean.
- One file per fix family: [moves.rs](moves.rs) (use after move), [names.rs](names.rs) (type mismatch and unknown name), [ownership.rs](ownership.rs), [hoist.rs](hoist.rs) and [builtin.rs](builtin.rs).
- [rounds.rs](rounds.rs) applies fixes in rounds (`MAX_ROUNDS`) and shows the diff; [scope.rs](scope.rs) holds the shared source lookups. The CLI is [../fix_cli.rs](../fix_cli.rs).

## Position in the pipeline

Runs on the diagnostics of [check](../check/README.md).

## Tests

- `cargo nextest run --locked --test diagnostics` (`check_fix` has one module per fix kind).

## Rules to keep

- A fix that does not re-check clean is never offered.
- `--write` rewrites a file only if it is unchanged since it was read.
