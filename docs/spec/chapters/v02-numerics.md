## Appendix B. Beyond v0.1: v0.2 features on main

Note: everything in this appendix is on `main` but is not part of the normative v0.1 sections above, and a v0.1 conforming implementation need not provide it.
Note: the statements here are checked and their blocks executed exactly like the rest of this file.

### B.1 Deterministic float reductions

[NB.1.1] `sum_tree(xs)`, `dot_tree(a, b)`, `sum_exact(xs)` and `dot_exact(a, b)` read-borrow bare named `Array<f32>` operands for the call and return `f32`. `src/fsum.rs`
[NB.1.2] `sum_tree` and `dot_tree` use one fixed tree of 8-element leaves that depends only on the length, so every engine and worker count returns the same bits, which generally differ from a sequential fold. `src/fsum.rs`
[NB.1.3] `sum_exact` and `dot_exact` return the exact real sum rounded once to nearest even, and every NaN result is the quiet NaN `0x7fc00000`. `src/fsum.rs`
[NB.1.4] Dot operands of different lengths trap with `INDEX_OUT_OF_BOUNDS` before any arithmetic, and `--engine auto` places each call by the measured crossover in `calibration/fsum-v1.json`. `calibration/fsum-v1.json`
[NB.1.5] A source `for` loop that folds floats is never reassociated, and `invar explain` only hints at these intrinsics with the obligation `FLOAT_FOLD_HINT`. `src/fsum.rs`
[NB.1.6] For `n = len(xs)`, `n = 0` gives `+0`, leaf `k` folds `xs[8k .. min(8k+8, n))` left to right from its first element, node `(L, j)` adds nodes `(L-1, 2j)` and `(L-1, 2j+1)` or keeps the first when the second is absent, and the result is node `(H, 0)` for the least `H` with `2^H` at least the leaf count. `src/fsum.rs`
[NB.1.7] `dot_tree` rounds each product separately, and for `sum_exact` and `dot_exact` an empty input is `+0`, a NaN, an infinity times zero or both infinities is NaN, another infinity wins, and a zero sum is `-0` only when every term is `-0`. `src/fsum.rs`
[NB.1.8] A call charges one step plus one per operand, forced `cuda` and `metal` run every reached nonempty call on the device, and every reached device or `auto` call leaves a receipt with its backend and a registered reason. `src/fsum.rs` `calibration/fsum-v1.json`
[NB.1.9] A device that flushes subnormals discards every result that could differ and lets the CPU produce it, and the CPU picks one worker, reported as backend `cpu-serial`, when the measured cost does not repay more. `src/fsum.rs` `src/schedule/reductions.rs::SERIAL_BACKEND`
[NB.1.10] `sum_exact` and `dot_exact` admit any partition: integer partials merge exactly and round once (NB.1.3, NB.1.7), so every worker or block count gives the same bits. `src/fsum.rs::Exact`
[NB.1.11] A float reduction split over workers or a device names its receipt law, `fsum-exact-merge-v1` or `fsum-aligned-tree-v1`, and a one-worker call names none. `src/schedule/reductions.rs::EXACT_LAW`

```invar outcome=value:0
fn main() -> i32 {
    let xs: Array<f32> = [16777216.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
    if sum_tree(xs) != 16777218.0 { return 2; }
    if sum_exact(xs) != 16777224.0 { return 3; }
    return 0;
}
```

```invar outcome=trap:INDEX_OUT_OF_BOUNDS
fn main() -> i32 {
    let a: Array<f32> = [1.0, 2.0];
    let b: Array<f32> = [1.0];
    if dot_exact(a, b) == 0.0 { return 1; }
    return 0;
}
```

### B.2 Stable sort intrinsics

[NB.2.1] `sort(xs)`, `sort_by_key(keys, values)`, `argsort(xs)` and `segmented_sort(xs, offsets)` are stable, sort keys of `i32`, `u32` or `f32`, and `argsort` returns an owned `Array<u32>` of original indices. `src/eval/sorting.rs`
[NB.2.2] Float keys use IEEE totalOrder after each NaN becomes `0x7fc00000`, signed zeros stay distinct, and payloads, which may also be `Array<Bool>`, keep their bits. `src/eval/sorting.rs`
[NB.2.3] A key and value length mismatch or invalid segment offsets trap with `INDEX_OUT_OF_BOUNDS` at the call span before any write. `src/eval/sorting.rs`
[NB.2.5] Segment offsets have `segments + 1` entries, first 0, last `len(xs)` and non-decreasing, `[0]` is valid only for an empty input, duplicate boundaries are empty segments, and no source loop is implicitly replaced by a sort. `src/eval/sorting.rs`
[NB.2.6] `sort(xs)` takes a call-scoped mutable loan, `sort_by_key(keys, values)` takes call-scoped mutable loans of two distinct named arrays and has no closure form, `argsort(xs)` takes a read loan and leaves the key bits unchanged, `segmented_sort(xs, offsets)` takes a mutable key loan and a read `Array<u32>` offset loan, and every sort except `argsort` returns `Unit`. `src/check.rs::sort_modes`
[NB.2.4] A report with an explicit sort site names each site's calibration family `sort.<op>.<type>`, and automatic sort placement still selects the CPU. `docs/explain.md`

```invar outcome=value:7
fn main() -> i32 {
    var keys: Array<u32> = [3u32, 1u32, 3u32, 0u32];
    var payload: Array<u32> = [0u32, 1u32, 2u32, 3u32];
    sort_by_key(keys, payload);
    if payload[0u32] == 3u32 and payload[1u32] == 1u32 and payload[2u32] == 0u32 {
        return 7;
    }
    return 0;
}
```

```invar outcome=trap:INDEX_OUT_OF_BOUNDS
fn main() -> i32 {
    var keys: Array<u32> = [2u32, 1u32];
    var values: Array<u32> = [0u32];
    sort_by_key(keys, values);
    return 0;
}
```

### B.3 Spatial grid queries

[NB.3.1] `voxel_grid_build(xs, ys, zs, cell)` copies read-borrowed `Array<f32>` coordinates into an opaque affine `VoxelGrid`, and a point with any non-finite coordinate is excluded and counted by `grid_excluded(grid)`. `src/spatial/runtime.rs`
[NB.3.2] `radius_neighbors(grid, qx, qy, qz, radius, max_k)` returns `Neighbors { counts, idx }` with true counts that are never clipped to the stride, using strict f32 `d2 <= radius * radius`. `src/spatial.rs`
[NB.3.3] `knn(grid, qx, qy, qz, k)` returns `Knn { idx, d2 }` ordered by squared distance and then original index, and pads missing slots with `u32::MAX` and positive infinity. `src/spatial.rs`
[NB.3.4] A nonpositive or non-finite cell or an out-of-range cell index traps with `INTEGER_OVERFLOW`, and coordinate arrays of different lengths trap with `INDEX_OUT_OF_BOUNDS`. `src/spatial.rs`
[NB.3.5] A `VoxelGrid` parameter is `read` or `take` and never `mut`, and query result arrays leave their record only through `move result.field`. `src/spatial.rs`
[NB.3.6] A report with a spatial site names the calibration family `spatial.<op>.f32` and the reason `spatial-cost-unmeasured`. `docs/explain.md`
[NB.3.7] Squared distances are strict f32 left to right, a positive infinite radius is allowed, a query count times stride beyond `u32` traps with `INTEGER_OVERFLOW`, and read loans hold through all scalar arguments. `src/spatial.rs`
[NB.3.8] A negative or NaN radius matches nothing, a query with any non-finite coordinate has no neighbours, cell indices are the exact `floor(coord / cell)` converted to `i32`, and query records bind to a name because tuple binding covers only pairs of 32-bit scalars (NB.7.12). `src/spatial/runtime.rs::finite`

```invar outcome=value:3
fn main() -> i32 {
    let xs: Array<f32> = [0.0, 1.0, -1.0];
    let ys: Array<f32> = [0.0, 0.0, 0.0];
    let zs: Array<f32> = [0.0, 0.0, 0.0];
    let grid: VoxelGrid = voxel_grid_build(xs, ys, zs, 1.0);
    let near: Neighbors = radius_neighbors(grid, xs, ys, zs, 1.0, 2u32);
    let nearest: Knn = knn(grid, xs, ys, zs, 4u32);
    let counts = move near.counts;
    let idx = move nearest.idx;
    if counts[0u32] == 3u32 and counts[1u32] == 2u32 and idx[0u32] == 0u32 and idx[3u32] == 4294967295u32 {
        return 3;
    }
    return 0;
}
```

```invar outcome=trap:INTEGER_OVERFLOW
fn main() -> i32 {
    let xs: Array<f32> = [0.0];
    let grid: VoxelGrid = voxel_grid_build(xs, xs, xs, 0.0);
    return 0;
}
```

### B.4 Binary64 and the correctly rounded math library

[NB.4.1] `f64` is a primitive beside `f32` under `strict-f64-v1`, with `Array<f64>`, record and payload fields, round to nearest even, gradual underflow and no contraction except an explicit `fma`. `src/numeric_contract/binary64.rs`
[NB.4.2] A decimal literal may carry an `f32` or `f64` suffix, and an unsuffixed literal takes its float type from context and otherwise defaults to `f32`. `spec/grammar.ebnf`
[NB.4.3] Conversions are explicit intrinsics such as `f64_from_f32`, `f32_from_f64` and `i32_from_f64`, `filled_f64(n, x)` builds an owned array of `n` copies, and a float to integer conversion of NaN or an out-of-range value traps with `INTEGER_OVERFLOW`. `src/eval/floating.rs::i32_from_f64`
[NB.4.4] `invar-math-cr-v1` provides `sqrt`, `cbrt`, `hypot`, the trigonometric, hyperbolic, exponential and logarithmic functions, `pow`, `floor`, `ceil`, `trunc`, `round`, `abs`, `copysign`, `min` and `max`, overloaded on `f32` and `f64` and correctly rounded. `src/math.rs`
[NB.4.5] Every NaN that leaves a library function or a float to float conversion is the canonical quiet NaN, and `min` and `max` ignore a single NaN operand. `src/math.rs`
[NB.4.6] Metal has no binary64, so a program that uses `f64` runs on the CPU with the reason `metal-no-f64`. `src/engine.rs::metal-no-f64`
[NB.4.7] An `extern c fn` of N7.3.1 also accepts `f64` as a value argument, the result `T` stays `i32`, `u32`, `Bytes` or `String`, and an `f64` result or any `f32` argument or result is `TYPE_MISMATCH`. `tests/language/bytes_strings_and_c_abi.rs::f64_foreign_scalar_argument_matches_its_oracle` `src/check.rs::valid_result`
[NB.4.8] An `f64` argument has tag `INV_TAG_F64_V1` (5) and mode `INV_MODE_VALUE_V1`, and its `scalar` holds the binary64 encoding, which C reads with `memcpy` into a `double`. `include/invar_foreign_v1.h` `tests/fixtures/l1_04/foreign.c`
[NB.4.9] `round` rounds half away from zero, `copysign` reads a NaN sign source as positive, `min` and `max` are IEEE 754-2019 minimumNumber and maximumNumber with `-0 < +0`, and every library name is reserved. `src/math.rs`
Note: the binary64 CUDA kernels whose CORE-MATH sources declare an anonymous struct inside a union (`cos`, `tan`, `atan2`, `log`, `log10` and `pow`) need NVRTC 12.9 or newer, every other CUDA unit keeps the 12.4 floor of N11.5.2.

```invar outcome=value:4
fn main() -> i32 {
    let x: f64 = 2.0f64;
    let y: f64 = x * x + 0.5f64;
    if sqrt(16.0f64) == 4.0f64 and floor(y) == 4.0f64 and exp(0.0f64) == 1.0f64 {
        return i32_from_f64(y);
    }
    return 0;
}
```

```invar outcome=trap:INTEGER_OVERFLOW
fn main() -> i32 {
    let big: f64 = 3000000000.0f64;
    return i32_from_f64(big);
}
```

