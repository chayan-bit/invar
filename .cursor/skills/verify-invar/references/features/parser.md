# Parser

The parser accepts an Invar source file and rejects malformed tokens with a source diagnostic. `invar check` is the user-facing entry. A successful language check reports the passes it completed, starting with syntax.

## Sub-features

- `parser.accept-sum` accepts `examples/sum.inv` through syntax, names, and types.
- `parser.reject-token` rejects an unclosed parameter list (`fn main( -> ...`) where a parameter name is expected.

## How to get to it (user POV)

- Run `invar check PATH --format json` on a `.inv` file.
- Run `invar check PATH` for the one-line human acceptance message.
- A later `invar run` parses as well. The parser proof is `check`, because it stops before execution.

## Driving it with invar-verify

Preconditions:

- `invar-verify doctor --format json` reports `ready: true`.
- `examples/sum.inv` is the checked sum from the example index.

- **Accept a program.** Run `invar-verify drive parser --format json`. The check `parser.accept-sum` runs `invar check examples/sum.inv --format json`. Exit 0, `status` is `ok`, and `checks_completed` contains `syntax`, `names`, and `types`.
- **Reject a token.** The same drive writes a scratch file `fn main( -> i32 { return 1; }` and runs `invar check` on it with `--format json`. Exit 1, `status` is `source_error`, and the diagnostic code is `PARSE_EXPECTED_TOKEN`.
- **One step, without the bundle.** Run `invar-verify run -- check examples/sum.inv --format json`. The captured document matches the accept check above.

## Gotchas

- The parser accepts at most 1 MiB of UTF-8 and nesting depth 32. A larger file is a tool limit, not a syntax error.
- `invar check` on a file that contains a system declaration (`component`, `system`, `connect`) routes to the system checker. Do not use `examples/system/` to prove the language parser. See the CLI feature.
- JSON spans are UTF-8 bytes. A column in an editor can disagree with `start` and `end`.
- `--format json` must follow the path. The CLI does not accept a free flag order.
