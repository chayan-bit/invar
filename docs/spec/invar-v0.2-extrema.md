# Invar v0.2: array extrema

Status: DRAFT; section 8 of the v0.2 chapter `docs/spec/invar-v0.2-axes.md`, kept in its own file.
Counts: statements=9 citations=10
Size record: tokens=1004 estimator=bytes/3 limit=6000
## 8. Array extrema

[NA8.1] This file is section 8 of `docs/spec/invar-v0.2-axes.md`, follows its section 1 conventions and has its own 6,000-token budget. `tests/spec_examples.rs::E_SPEC`
[NA8.2] `min_of(a)`, `max_of(a)`, `argmin(a)` and `argmax(a)` read-borrow one named `Array<i32>`, `Array<u32>`, `Array<f32>` or `Array<f64>` for the call, `min_of` and `max_of` return its element type, `argmin` and `argmax` return `u32`, and the four names are reserved. `src/check/reductions.rs`
[NA8.3] `min_of(a)` is the left fold of the library `min` over `a` and `max_of(a)` that of `max`, so integers are exact and floats follow NB.4.9: minimumNumber and maximumNumber with `-0 < +0`, a NaN element is ignored, and an all-NaN input gives the canonical quiet NaN. `src/extremum/fold.rs`
[NA8.4] `argmin(a)` is the lowest index whose element has the bits of `min_of(a)` and `argmax(a)` that of `max_of(a)`, so ties go to the lowest index, and an all-NaN input gives 0. `src/extremum/fold.rs::step`
[NA8.5] On an empty input each of the four traps `INDEX_OUT_OF_BOUNDS` at the call span, as reading `a[0]` would, and every engine charges a call the same fuel. `tests/schedule_cpu/extremum_intrinsics.rs`
[NA8.6] The fold is associative and keeps the first of equal elements, so `cpu-parallel` evaluates a reached call by each chunk's extremum and an ordered combine, with the same bits, index and trap at every worker count. `src/schedule/extrema.rs`
[NA8.7] The reference evaluator, MIR, `cpu-parallel` and `native-serial` implement the four, the GNET CPU oracle refuses them as a pending intrinsic with `ToolFailure::Unsupported`, and no device lowering exists yet. `tests/schedule_cpu/extremum_intrinsics.rs`
[NA8.8] The select loop of NA3.13 keeps its own compare, which differs from these on floats: it treats `-0.0` and `+0.0` as equal, keeps a NaN seed, and with a non-strict compare takes the last of equal values. `src/select_fold.rs`
[NA8.9] The CUDA and Metal engines have no lowering of the four: a program whose only device candidate is such a call is refused with `CUDA_UNAVAILABLE` or `METAL_UNAVAILABLE`, and beside a device region the call runs serially on the host with the reference value, trap, fuel and cleanup. `tests/support/gpu_extremum.rs`

Note: the value fixture: `-0.0` is less than `+0.0` and the NaN is ignored, so the minimum is the first `-0.0`, at index 3.

```invar outcome=value:3
fn main() -> i32 {
    let nan: f32 = 0.0 / 0.0;
    let a: Array<f32> = [2.0, nan, 0.0, -0.0, -0.0, 5.0];
    let m = min_of(a);
    if argmax(a) != 5u32 { return 1; }
    if 1.0 / m > 0.0 { return 2; }
    return i32_from_f32(f32_from_u32(argmin(a)));
}
```

Note: the trap fixture: an empty input has no extremum.

```invar outcome=trap:INDEX_OUT_OF_BOUNDS
fn main() -> i32 {
    let a = filled_i32(0u32, 0);
    return max_of(a);
}
```
