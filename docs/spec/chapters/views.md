### 5.5 Borrowed views

[N5.5.1] `View<T>` and `MutView<T>` over `i32` and `u32` are non-owning views made by `view(owner, start, end[, stride])` and `view_mut(...)` from a bare array or view name. `tests/language/borrowed_views.rs::borrowed_views_reference_mir_native_agree`
[N5.5.2] The length is `ceil((end-start)/stride)`, the stride is a positive `u32`, and range, stride and composition are checked in `u32` arithmetic. `tests/language/borrowed_views.rs::borrowed_views_range_and_stride_trap_identically`
[N5.5.3] An invalid range or stride traps as `INDEX_OUT_OF_BOUNDS`, identically in the reference, MIR and native engines. `tests/language/borrowed_views.rs::borrowed_views_range_and_stride_trap_identically`
[N5.5.4] An owner cannot be moved, dropped or mutated while a view of it is live, and mutable views of one owner need provably disjoint intervals. `tests/language/borrowed_views.rs::borrowed_views_public_check_and_negative_sources`
[N5.5.5] A function returns a view only with `-> View<T> from xs` where `xs` is a parameter, and a view of a local owner cannot be returned. `tests/language/borrowed_views.rs::borrowed_views_read_return_from_mut_parameter`
[N5.5.6] There are at most 4096 view facts per program, and views have no view parameters, no view fields and no `f32` element type. `src/check/views.rs`
[N5.5.7] Views have lexical scope and no cleanup event, element `i` of a nested view is at its parent physical origin plus `start + i*stride` parent strides, two read views may overlap, and unknown bounds never prove disjointness. `src/check/views.rs`
[N5.5.8] A mutable view suspends direct reads of its owner, a `MutView` return needs a `mut` parameter, the caller owner stays borrowed while the returned view lives, and copying a view needs an explicit reborrow. `tests/language/borrowed_views.rs`

```invar outcome=value:65
fn first(xs: read Array<i32>) -> View<i32> from xs {
    return view(xs, 1u32, 5u32, 2u32);
}

fn main() -> i32 {
    var xs: Array<i32> = [1i32, 2i32, 3i32, 4i32, 5i32, 6i32];
    if true {
        let a = view_mut(xs, 0u32, 2u32);
        let b = view_mut(xs, 2u32, 6u32, 2u32);
        a[1u32] = 10i32;
        b[1u32] = 40i32;
        if true {
            let c = view_mut(a, 0u32, 1u32);
            c[0u32] = 9i32;
        }
        a[0u32] = 11i32;
    }
    let empty = view(xs, 6u32, 6u32);
    let sample = first(xs);
    if len(empty) == 0u32 {
        return xs[0u32] + xs[1u32] + xs[4u32] + sample[1u32];
    } else {
        return 0i32;
    }
}
```

```invar outcome=diag:OWN_BORROW_CONFLICT
fn main() -> i32 {
    var xs: Array<i32> = [1i32, 2i32, 3i32];
    let a = view_mut(xs, 0u32, 2u32);
    let b = view_mut(xs, 1u32, 3u32);
    return a[0u32];
}
```

