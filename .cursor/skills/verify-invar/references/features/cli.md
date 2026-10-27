# CLI

The `invar` binary is the user interface. Help and usage are text. Checks, runs, explains, and generators that take `--format json` print one JSON document on stdout. Usage errors print the help text on stderr and exit 2.

## Sub-features

- `cli.help` prints the command list and exits 0.
- `cli.usage` rejects an empty argument list with exit 2.
- `cli.missing-file` reports `FILE_READ` for a path that is not there.
- `cli.system-file-is-not-a-language-check` shows that `invar check` on a system file returns the system document.

## How to get to it (user POV)

- Run `invar --help` or `invar -h`.
- Run `invar` with no arguments.
- Run `invar check PATH --format json` on a language file, a system file, or a missing path.
- The command list in the help text is the supported surface: `check`, `run`, `explain`, `build`, `system`, `component`, `import`, `source-device`, `gnet-device`, `calibrate`, `debug`.

## Driving it with invar-verify

Preconditions:

- `invar-verify doctor --format json` reports `ready: true`.
- The missing-file path is under scratch and is not created.

- **Help.** Run `invar-verify drive cli --format json`. `cli.help` runs `invar --help`. Exit 0. Stdout starts with `Usage: invar --help` and contains the `cpu-parallel`, `system gen ros2`, `source-device`, and `gnet-device` lines.
- **Usage.** The drive runs `invar` with no arguments. Exit 2. Stderr contains `Usage: invar --help`.
- **Missing file.** The drive runs `invar check <scratch>/missing-program.inv --format json`. Exit 2, `status` is `tool_error`, code `FILE_READ`.
- **System routing.** The drive runs `invar check examples/system/two-components.inv --format json`. Exit 0 and `status` is `ok`. That document is the system checker, not the language `checks_completed` list.

## Gotchas

- Many commands parse flags as fixed pairs after the path. `--format json` before the path, or a stray flag, is a usage error rather than a partial run.
- `--help` writes stdout. Usage writes stderr. Do not assert on the wrong stream.
- JSON commands print one object and no progress log. If stdout is not JSON, the process took the usage path.
- `invar check` looks at the source. A system declaration changes which checker runs. The language parser feature uses `examples/sum.inv` for that reason.
- The binary does not print a version line. Doctor records `rustc` and the binary path instead.
