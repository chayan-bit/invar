# Contributing to Invar

Thank you for your interest in Invar.
Invar is a pre-release development snapshot, and contributions of all sizes are welcome: bug reports, documentation fixes, tests and implementation work.
Please follow the [Code of Conduct](CODE_OF_CONDUCT.md) in every project space.

## Before you start

Read the [v0.1 language specification](docs/spec/invar-v0.1.md), the [grammar](spec/grammar.ebnf) and [Rust engineering](docs/rust-engineering.md).
Larger work items are tracked in [GitHub issues](https://github.com/chayan-bit/invar/issues).
For a substantial change, open an issue first so the design can be agreed before you invest in it.

## Reporting bugs

Open a bug report using the issue form on GitHub.
Include the Invar commit, your operating system and architecture, the exact command, a minimal `.inv` program or input that reproduces the problem, and the full output.
For a wrong result, say what you expected and why.
Do not report security problems in a public issue; follow [SECURITY.md](SECURITY.md) instead.

## Setting up

Install [rustup](https://rustup.rs/).
The toolchain is pinned in [rust-toolchain.toml](rust-toolchain.toml) (Rust 1.95.0) and installs itself on first use.
Optional tools:

- [just](https://github.com/casey/just) runs the recipes in the [justfile](justfile).
- [cargo-nextest](https://nexte.st/) 0.9.146 runs the suite in parallel with the profiles in [nextest.toml](.config/nextest.toml).

On Windows (MSVC or `x86_64-pc-windows-gnullvm`):

- MSVC builds compile the vendored CORE-MATH sources with clang-cl: install the Visual Studio component "C++ Clang Compiler for Windows", or set `CC=clang-cl`.
- Tests that build C++ harnesses need a Clang `c++` on PATH; CI uses [llvm-mingw](https://github.com/mstorsjo/llvm-mingw) 20261006 (ucrt, x86_64).
- Tests that bound child processes need GNU `timeout` on PATH ahead of `System32`; Git for Windows provides it in `usr\bin`.
- Clone with `git -c core.symlinks=true clone ...` (Developer Mode or an elevated shell), since some fixtures are symlinks.
- A few tests drive POSIX-only programs and are ignored on Windows with the reason in their `ignore` attribute.

## Making a change

1. Implement the smallest complete behavior, including diagnostics and input limits.
2. Add focused positive, negative and boundary tests.
3. Update the specification and grammar together when a change affects language semantics, and explain any changed user-visible behavior.
4. Never weaken a test oracle to make a failing test pass without a justified semantic change.
5. Prefer the standard library; a new dependency needs a concrete need and review.

## Checks before opening a pull request

Run these from the repository root:

| Command | What it does |
| --- | --- |
| `just check` | Formatting, Clippy, the full Debug and Release suites and doctests |
| `just test-quick` | Fast inner loop; a subset of the suite |
| `just test` | The full Debug and Release suites and doctests |
| `just fmt` | Checks formatting without rewriting files |
| `just lint` | Clippy with warnings denied |
| `just size` | The file-size ratchet for `.rs` files over 800 lines |

Without `just`, the equivalent commands are:

```sh
cargo fmt --all -- --check
cargo clippy --locked --all-targets -- -D warnings
INVAR_FULL_CORPUS=1 cargo nextest run --locked --profile ci
INVAR_FULL_CORPUS=1 cargo nextest run --locked --profile ci --release
cargo test --locked --doc
```

The full suite is large and slow on a laptop; `just test-quick` is a good first pass, and CI runs the rest.

## Continuous integration

Every pull request runs [ci.yml](.github/workflows/ci.yml) on GitHub-hosted Linux and Windows runners, on CPU only:

- **Lint**: `cargo fmt` and Clippy with warnings denied.
- **Check**: `cargo check --all-targets`.
- **Hygiene**: the file-size ratchet, the native host bundle check and the script self-tests.
- **Test shards 1 to 4**: the full suite split four ways, plus doctests.
- **Test (CPU-model corpus)**: the device CPU-model corpus test, run alone.
- **Windows (msvc 1/4 to 4/4, gnullvm build)**: the default-feature suite on `windows-latest` with MSVC in four shards, a gnullvm build of every test target, and the CORE-MATH tests on both.
  It always runs the pull-request corpus sample.
  One test is excluded there: the `system gen ros2` tree hash for `cell-ur5e.inv` depends on the platform libm's `asin`, which differs from glibc by one ulp on Windows.

Pull requests run a deterministic 1-in-8 sample of the corpus sweeps that keeps every corpus directory; pushes to main and `just test` run the whole corpus (`INVAR_FULL_CORPUS=1`).

CI must pass before review.
CUDA and Metal suites are not run in CI.

## Running the GPU tests locally

GPU suites are ignored by default and need matching hardware, so contributors run them by hand when a change touches a device path.

- CUDA needs a Linux host with an NVIDIA GPU and the CUDA driver: `just test-gpu-cuda` (this runs `cargo test --locked --features cuda -- --include-ignored`).
- Metal needs an Apple Silicon Mac: `just test-metal` (this runs `cargo test --locked --features metal -- --include-ignored`).

Run them on an otherwise idle GPU, since some tests check timing or free device memory.
In the pull request, say which GPU suites you ran, on which hardware, and which were unavailable.
Never label a check you did not run as passing.

## Commits and pull requests

Use [Conventional Commits](https://www.conventionalcommits.org/): `<type>: <description>`, where the type is one of `feat`, `fix`, `refactor`, `docs`, `test`, `chore`, `perf` or `ci`.
Keep commits small and focused.

Keep each pull request focused on one change and link the issue it addresses.
Fill in the pull request template: describe what changed, how you tested it and any remaining limitations.

## License of contributions

Invar is licensed under the [Apache License, Version 2.0](LICENSE), and contributions use the same terms (inbound equals outbound).
Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion in Invar by you, as defined in the Apache-2.0 license, is licensed under Apache-2.0, without any additional terms or conditions.
No separate contributor agreement or sign-off is required.
