# Checker

The checker accepts a parsed program only when names, types, ownership, call loans, and view lifetimes hold. Rejections are source diagnostics with stable codes. Check does not execute the program.

## Sub-features

- `checker.ownership-sum` accepts `examples/sum.inv` through ownership, call loans, and view lifetimes.
- `checker.view-overlap` rejects two overlapping mutable views.
- `checker.view-return` rejects returning a view of a local owner.
- `checker.foreign-callback` rejects a callback in a foreign signature.

## How to get to it (user POV)

- Run `invar check PATH --format json`.
- Open a view example under `examples/views/` or a foreign example under `examples/bytes/`.
- `invar check PATH --fix` prints suggested edits as a unified diff. `invar check PATH --fix --write` rewrites the file. The harness never passes `--write`.

## Driving it with invar-verify

Preconditions:

- `invar-verify doctor --format json` reports `ready: true`.
- The example index expects `OWN_BORROW_CONFLICT`, `OWN_RETURN_BORROW`, and `TYPE_MISMATCH` for the three negative files.

- **Ownership of the sum.** Run `invar-verify drive checker --format json`. `checker.ownership-sum` runs `invar check examples/sum.inv --format json`. Exit 0 and `checks_completed` contains `local_ownership`, `call_loans`, and `view_lifetimes`.
- **Overlapping views.** The same drive runs `invar check examples/views/overlap.inv --format json`. Exit 1, `status` is `source_error`, code `OWN_BORROW_CONFLICT`.
- **Returned view.** The drive runs `invar check examples/views/invalid-return.inv --format json`. Exit 1, code `OWN_RETURN_BORROW`.
- **Foreign callback.** The drive runs `invar check examples/bytes/foreign-callback.inv --format json`. Exit 1, code `TYPE_MISMATCH`.

## Gotchas

- A green `check` is not a run. The reference feature is the execution proof.
- `invar check PATH --fix` alone is the dry run: it prints a unified diff and writes nothing. `--fix --write` rewrites source. Do not pass `--write` from a verification drive. `--fix` must be the third argument and does not combine with `--format json`. It refuses system files with exit 2.
- Foreign programs that typecheck still need an absolute `--c-library` before `invar run`. The reference engine executes the Invar oracle. It does not load the library. That is not a C ABI proof.
- View diagnostics are ownership codes (`OWN_...`), not parser codes.
