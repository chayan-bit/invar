# Security policy

## Supported versions

Invar is a pre-release development snapshot.
Only the latest commit on the `main` branch is supported; fixes land on `main` and there are no backports.

## Reporting a vulnerability

Please do not open a public issue for a security problem.
Use GitHub private vulnerability reporting instead:

1. Open the [Security tab](https://github.com/chayan-bit/invar/security) of the repository.
2. Choose "Report a vulnerability".
3. Describe the problem, the affected commit, and a minimal reproduction.

Reports are handled on a best-effort basis by the maintainers.
We will acknowledge a report, keep you informed of progress, and credit you in the fix if you wish.
Please allow time for a fix before any public disclosure.

## Scope

Examples of in-scope reports: memory-safety or soundness bugs in the safe core, a way to make the checker admit an unsound program, a way to make the toolchain execute code the user did not ask it to run, or unbounded resource use on untrusted source despite the documented limits.
Foreign code loaded through the C ABI is outside the safety boundary and is documented as such in the [architecture overview](docs/architecture.md).
