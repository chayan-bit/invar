# Contributing to Invar

Thank you for your interest in Invar.
Invar is a pre-release development snapshot, and contributions of all sizes are welcome: bug reports, documentation fixes, tests and implementation work.
Please follow the [Code of Conduct](CODE_OF_CONDUCT.md) in every project space.

## Before you start

Read the [v0.1 language specification](docs/spec/invar-v0.1.md), the [grammar](spec/grammar.ebnf) and [Rust engineering](docs/rust-engineering.md).
Larger work items are tracked in [GitHub issues](https://github.com/chayan-bit/invar/issues).
For a substantial change, open an issue first so the design can be agreed before you invest in it.

To find your way around the code, read these in order:

1. The [README](README.md), for what Invar is.
2. The [glossary](docs/glossary.md), for words like admission, oracle, family and receipt.
3. The [architecture code tour](docs/architecture.md#code-tour-one-program-end-to-end), which follows one program through the compiler and names the entry function at each step.
4. The README in the `src/` area you will touch; each lists its entry points, its place in the pipeline and its tests.
5. The [testing guide](docs/testing.md), which says what each test binary covers and how to run one.

The [documentation index](docs/README.md) lists everything else.

## Finding something to work on

Issues labelled [good first issue](https://github.com/chayan-bit/invar/labels/good%20first%20issue) are sized for a first contribution.
The other labels say the area (`area:*`), the size (`size:*`), the risk (`risk:*`) and the kind of change (`type:*`).
Comment on the issue before you start so two people do not do the same work.

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

## First build

```sh
git clone https://github.com/chayan-bit/invar.git
cd invar
cargo build --locked
cargo run --locked --bin invar -- run examples/sum.inv --engine reference
```

The program returns 10.
The default build needs only Rust and a C compiler (the vendored CORE-MATH sources are C).
The `native` feature additionally needs LLVM 21 ([native toolchain](docs/native-toolchain-v1.md)), and the `cuda` and `metal` features need their own hardware to run, though they compile without it.

To run one test binary or one test, use nextest directly; the [testing guide](docs/testing.md) lists the binaries:

```sh
cargo nextest run --locked --test language
cargo nextest run --locked --test language -E 'test(/arithmetic_fuel_and_demand/)'
```

## Making a change

1. Implement the smallest complete behavior, including diagnostics and input limits.
2. Add focused positive, negative and boundary tests, in the test binary that already covers the area (see the [testing guide](docs/testing.md)).
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
- **Test (CPU-model corpus 1/2, 2/2)**: the device CPU-model corpus test, run alone, each job on one half of the corpus programs (`INVAR_CORPUS_PART`).
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
Fill in the [pull request template](.github/pull_request_template.md): describe what changed, how you tested it and any remaining limitations.
Name the exact commands you ran and the result, and list any check you could not run, such as a GPU suite.

## License of contributions

Invar is licensed under the [Apache License, Version 2.0](LICENSE), and contributions use the same terms (inbound equals outbound).
Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion in Invar by you, as defined in the Apache-2.0 license, is licensed under Apache-2.0, without any additional terms or conditions.
No separate contributor agreement or sign-off is required.
