# Contributing

Thank you for your interest in Invar.
Contributions of all sizes are welcome: bug reports, documentation fixes, tests and implementation work.

## Before you start

Read the [bootstrap specification](spec/bootstrap.md), the [v0.1 language specification](docs/spec/invar-v0.1.md) and [Rust engineering](docs/rust-engineering.md).
Larger work items are described in [implementation issues](docs/issues/README.md) and [remaining work](docs/remaining-work.md).
For a substantial change, open an issue first so the design can be agreed before you invest in it.

## Making a change

1. Implement the smallest complete behavior, including diagnostics and input limits.
2. Add focused positive, negative and boundary tests.
3. Update the specification and grammar together when a change affects language semantics, and explain any changed user-visible behavior.
4. Never weaken a test oracle to make a failing test pass without a justified semantic change.
5. Prefer the standard library; a new dependency needs a concrete need and review.

## Checks before opening a pull request

Run these from the repository root with the pinned toolchain (`rust-toolchain.toml`):

```sh
cargo fmt --all -- --check
cargo clippy --locked --all-targets -- -D warnings
cargo test --locked
```

GPU suites are ignored by default; run them with `--features cuda` or `--features metal` on matching hardware when your change touches a device path, and say in the pull request which checks you ran and which were unavailable.

## Pull requests

Keep each pull request focused on one change and link the issue it addresses.
Describe what changed, how you tested it and any remaining limitations.
CI must pass before review.

## Developer Certificate of Origin

Every commit must be signed off under the [Developer Certificate of Origin 1.1](https://developercertificate.org/), certifying that you have the right to submit the work under this project's license.
Add the sign-off with `git commit -s`, which appends a line such as:

```text
Signed-off-by: Your Name <your.email@example.com>
```

The name and email must match the commit author.

## License of contributions

Invar is licensed under the [Apache License, Version 2.0](LICENSE).
Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion in Invar by you, as defined in the Apache-2.0 license, shall be licensed under Apache-2.0, without any additional terms or conditions.
