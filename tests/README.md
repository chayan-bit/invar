# Integration tests

Each directory here is one nextest binary (`tests/<name>/main.rs`) whose modules are named after the issue they cover, plus a few single-file binaries.
The [testing guide](../docs/testing.md) says what every binary covers, which ones need a GPU, and how to run one.

```sh
cargo nextest run --locked --test language
```

Shared helpers are in [test-support](../test-support/README.md), and checked-in inputs are under `tests/fixtures`.
