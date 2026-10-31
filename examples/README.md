# Invar examples

Every `.inv` file under this directory is a verified example.
[index.json](index.json) lists each one exactly once with its intent, family, specification section, the engines it runs on and its expected result.
The index is validated by [index.schema.json](index.schema.json).
`tests/spec_examples.rs` runs every example through the real toolchain and fails if an output, a hash or a listing changes.

Changing an expected result needs a semantic explanation in the commit, not a refreshed snapshot.

## Layout

| Directory | Contents | Specification |
| --- | --- | --- |
| [sum.inv](sum.inv), [ownership.inv](ownership.inv), [generated-map.inv](generated-map.inv) | The original bootstrap examples | 3, 4 |
| [pair-reduction.inv](pair-reduction.inv), [metal-tensor-two-shapes.inv](metal-tensor-two-shapes.inv) | Feature examples that other tests read by path | 11.4 |
| [fsum.inv](fsum.inv), [sort.inv](sort.inv), [fk_panda.inv](fk_panda.inv) | v0.2 float reductions, sort intrinsics and binary64 on main | Appendix B |
| `language/` | Arithmetic, traps, fuel, loans, moves and modules | 3, 4, 5.1 |
| `types/` | Records, variants, `Result` and strict `f32` | 5.2 to 5.4, 5.8 |
| `views/` | Borrowed views and their rejections | 5.5 |
| `generics/` | Bounded generics and affine closures | 5.6, 5.7 |
| `bytes/` | Bytes, strings, UTF-8 and the C ABI oracle | 7 |
| `units/` | Quantities, frames and transforms | 6 |
| `parallel/` | Planner-admitted map, reduction and scan shapes | 11.3 |
| `families/` | Source forms the M4 recognizers accept | 11.4 |
| `system/` | System dialect and graph-check cases | 8, 9, 10 |
| `cells/` | Robot cells, imported manifests and generated targets | 8, 10.3 |

## Running one

```sh
invar run examples/language/recursion.inv --engine reference
invar check examples/views/overlap.inv --format json
invar system check examples/system/twelve-components.inv --format json
```

The `native-serial` engine needs the `native` Cargo feature and LLVM 21.
Runnable examples list `native-serial` and the test runs that engine only in a build with the `native` feature.

## Adding one

Add the file, add its entry to `index.json`, and run `cargo test --locked --test spec_examples`.
A family with no example yet goes under `pending_families` with a reason.
