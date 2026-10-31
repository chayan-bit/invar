# Tests

The repository's own checks are `scripts/check-file-size`, `cargo test`, and the nextest profiles in the justfile. The default verification drive runs the ratchet and one binary test that does not need LLVM or a GPU. The full corpus stays a separate command.

## Sub-features

- `tests.file-size` runs `scripts/check-file-size` and requires 0 violations.
- `tests.focused-bin` runs the `machine_diagnostics_document_size_limit` test in the `invar` binary.
- `tests.full-nextest` is the CI suite. The default drive skips it and names the command.

## How to get to it (user POV)

- Run `scripts/check-file-size` from the checkout. It reads tracked `.rs` files and does not build.
- Run `cargo test --locked --bin invar machine_diagnostics_document_size_limit`.
- Run `just test-quick` for the fast nextest profile.
- Run `just test` for Debug nextest, Release nextest, and doctests, with `INVAR_FULL_CORPUS=1`.
- Run `just test-gpu-cuda` or `just test-metal` only on matching hardware. Those skips are on the codegen feature.

## Driving it with invar-verify

Preconditions:

- `git` works in the checkout. The ratchet uses `git ls-files`.
- `cargo` is the 1.95 toolchain. A second cargo process is not using `target/`.
- `invar-verify doctor` has recorded whether `cargo-nextest` and `just` exist.

- **Ratchet.** Run `invar-verify drive tests --format json`. `tests.file-size` runs `scripts/check-file-size`. Exit 0 and stdout contains `0 violations`. The drive does not pass `--update` or `--regenerate`. Those rewrite `scripts/file-size-allowlist.txt`.
- **One compiler test.** The drive runs `cargo test --locked --bin invar --jobs 2 machine_diagnostics_document_size_limit`. Exit 0 and the output contains `1 passed`. This compiles the test profile of the binary. It is not the dev binary from `invar-verify build`.
- **Full suite.** `tests.full-nextest` is skipped when `cargo-nextest` is absent. The recorded command is `cargo nextest run --locked --profile ci -p invar -p invar-tools`. To run the real gate on a machine that has `just` and nextest: `invar-verify test --profile full`. That command is not part of `invar-verify suite`.

## Gotchas

- `cargo test` uses a different profile from `cargo build`, so the focused test rebuilds the package. Give it the cargo lock. Do not start it beside `invar-verify build`.
- A filter that matches nothing still exits 0 with `0 passed`. The drive fails unless it sees `1 passed`.
- `just test-quick` omits `general_net`, `schedule`, and `system` library tests. It is not a substitute for `just test`.
- Pull-request CI samples the corpus 1-in-8. `just test` and pushes to main set `INVAR_FULL_CORPUS=1`. Quoting a sampled run as the full corpus is wrong.
- `scripts/check-file-size --update` and `--regenerate` rewrite the allowlist. `--dry-run` on `invar-verify` does not pass those flags, and the drive never calls them.
- GPU tests are `#[ignore]` unless the feature is on and the runner passes `--include-ignored`. A default `cargo test` that exits 0 did not run them.
