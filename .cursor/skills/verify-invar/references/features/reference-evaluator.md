# Reference evaluator

The reference evaluator runs a checked program in source order and prints one JSON result. It is the oracle for values and traps. Users select it with `--engine reference`.

## Sub-features

- `reference.sum` returns 10 for `examples/sum.inv`.
- `reference.ownership` returns 41 for `examples/ownership.inv`.
- `reference.recursion` returns 55 for `examples/language/recursion.inv`.
- `reference.matrix` returns 7 for `examples/families/matrix-f32.inv`.
- `reference.transpose` returns 7 for `examples/families/transpose.inv`.
- `reference.checked-sum-traps` traps with `INTEGER_OVERFLOW` on `examples/parallel/reduction-checked-sum.inv`.

## How to get to it (user POV)

- Run `invar run PATH --engine reference --format json`.
- Add `--fuel N` for a positive fuel budget. The default is 1000000.
- Human output is `Result: N` on stdout. JSON is the proof form.

## Driving it with invar-verify

Preconditions:

- `invar-verify doctor --format json` reports `ready: true`.
- The expected values are the ones in `examples/index.json`.

- **Run the bundle.** Run `invar-verify drive reference --format json`. Each value check exits 0 with `status` `ok` and `result.value` equal to the index.
- **Sum.** The drive runs `invar run examples/sum.inv --engine reference --format json`. `result.value` is 10.
- **Ownership.** The drive runs `invar run examples/ownership.inv --engine reference --format json`. `result.value` is 41.
- **Trap.** The drive runs `invar run examples/parallel/reduction-checked-sum.inv --engine reference --format json`. Exit 3, `status` is `trap`, and `trap.category` is `INTEGER_OVERFLOW`.

## Gotchas

- A trap exits 3. A source error exits 1. A tool error exits 2. A usage error exits 2 and prints help on stderr, often with no JSON.
- Fuel exhaustion is `status` `evaluation_limit` and exit 4. It is not a source trap and it is not a proof that the program diverges.
- `main` must return `i32`. The JSON result type is `i32`.
- `--engine reference` does not admit parallel plans. Use the parallelism feature for `cpu-parallel` and `explain`.
- Passing any absolute file as `--c-library` satisfies the reference CLI for a foreign program. The number that comes back is the oracle, not the dynamic library.
