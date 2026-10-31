# numeric_contract

The software oracle for the strict floating-point contracts: `strict-f32-v1`, `strict-f64-v1`, the fold tiers and the storage-only narrow types of numeric policy v2.

## Where to start

- `NumericPolicy`, `F32Bits` and `round_rational` in [../numeric_contract.rs](../numeric_contract.rs); [binary64.rs](binary64.rs) holds `F64Bits`; [ieee.rs](ieee.rs) has format-generic exact IEEE operations.
- [fold.rs](fold.rs) defines `FoldTier` and the fold oracle; [storage.rs](storage.rs) defines the narrow storage conversions.

## Position in the pipeline

Used by [check](../check/README.md), [eval](../eval/README.md) and every test that compares float bits.

## Tests

- `cargo nextest run --locked --test m4_numeric_layout`, `--test numeric_tiers` and `--test fsum_math_sort_spatial`.

## Rules to keep

- The oracle uses exact rationals and one rounding per operation; it must not call the code under test.

## Read next

- [numeric contracts](../../docs/numeric.md)
