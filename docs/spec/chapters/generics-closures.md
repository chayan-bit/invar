### 5.6 Generics

[N5.6.1] A generic function has one or two parameters bounded by `Value`, which stands for any complete concrete non-view value type and implies no operations. `tests/language/generics_and_closures.rs::generics_and_closures_monomorphized_functions_agree`
[N5.6.2] Type arguments are inferred only from exact argument types, never from the expected result, or are given with `f::<T, U>(...)`. `tests/language/generics_and_closures.rs::generics_and_closures_public_check_rejects_invalid_sources`
[N5.6.3] An instance is identified by the qualified function name and its concrete type arguments, is built once and reused, and recursive instantiation is rejected. `tests/language/generics_and_closures.rs::generics_and_closures_module_identity_and_unused_take_cleanup`
[N5.6.4] Expansion is limited to 16 generic frames and 128 lowered functions, and traits and higher-kinded syntax are unsupported. `src/check/generics.rs::MAX_GENERIC_DEPTH`
[N5.6.5] A parameter or result may be a bare type parameter, or `Array<T>` when `T` is `i32` or `u32`, and missing or conflicting type arguments are source errors. `src/check/generics.rs`

```invar outcome=value:12
fn identity<T: Value>(x: T) -> T { return x; }
fn choose<T: Value, U: Value>(x: T, y: U) -> T { return x; }

fn main() -> i32 {
    let u: u32 = identity(3u32);
    if u == 3u32 {
        return identity::<i32>(choose(7i32, true)) + identity(5i32);
    } else {
        return 0i32;
    }
}
```

```invar outcome=diag:TYPE_MISMATCH
fn same<T: Value>(x: T, y: T) -> T { return x; }
fn main() -> i32 { return same(1i32, 2u32); }
```

### 5.7 Affine closures

[N5.7.1] A local closure lists explicit captures `copy`, `read` or `take`, has at most 8 captures, 8 parameters and 65,536 source bytes, and is called at most once in its block. `src/check/closures.rs::MAX_CAPTURES`
[N5.7.2] `read` borrows the owner until the call or scope exit, `take` moves the owner at creation and cleans it up if the closure is never called. `tests/language/generics_and_closures.rs::generics_and_closures_module_identity_and_unused_take_cleanup`
[N5.7.3] A closure cannot escape, be assigned, passed or returned, and its body may use only its captures, parameters and global functions. `tests/language/generics_and_closures.rs::generics_and_closures_public_check_rejects_invalid_sources`
[N5.7.4] Captures are named and typed, `copy` snapshots an integer or Bool at creation, and while a `read` capture lives the captured owner cannot be used directly. `src/check/closures.rs`
[N5.7.5] A closure call is a straight-line statement outside any condition, captures are set up left to right at creation, and a call delivers captures in declaration order and then evaluates its arguments left to right. `tests/language/generics_and_closures.rs`

```invar outcome=value:10
fn main() -> i32 {
    let xs: Array<i32> = [1i32, 8i32];
    let ys: Array<i32> = [2i32];
    let n: i32 = 3i32;
    let c = closure [copy n: i32, read xs: Array<i32>, take ys: Array<i32>] (x: i32) -> i32 {
        return n + xs[0u32] + ys[0u32] + x;
    };
    let answer: i32 = c(4i32);
    return answer;
}
```

```invar outcome=diag:UNSUPPORTED_FEATURE
fn main() -> i32 {
    let n: i32 = 1i32;
    let c = closure [copy n: i32]() -> i32 { return n; };
    return c;
}
```

### 5.8 f32 and decimal literals

[N5.8.1] The compiler accepts `f32`, `Array<f32>`, bounded exact decimal literals, unary negation, `+ - * /` and IEEE comparisons. `docs/numeric.md`
[N5.8.2] The numeric policy is `strict-f32-v1`: IEEE 754 binary32, round to nearest even, no reassociation or contraction, and subnormals preserved. `src/numeric_contract.rs`
[N5.8.3] Only an explicit `fma(a, b, c)` fuses a multiply and an add, and `f32_from_i32`, `f32_from_u32` and `filled_f32` are the conversions and constructor. `tests/numeric_layout/source_f32.rs::source_f32_strict_arithmetic_and_ieee_predicates`
[N5.8.4] An integer is not implicitly an `f32`, `%` is not defined on `f32`, and `main` still returns `i32`. `tests/numeric_layout/source_f32.rs::source_f32_rejects_implicit_conversion_and_remainder`
[N5.8.5] `i32_from_f32(x)` truncates a finite `f32` toward zero and traps with `INTEGER_OVERFLOW` when the value is not finite or outside the `i32` range. `docs/numeric.md::i32_from_f32`
[N5.8.6] On `f32`, `sqrt`, `sin` and `cos` are `invar-math-cr-v1` like every library function, and because no device lowers `sin` or `cos` yet, a CUDA or Metal request keeps each function that calls them on the CPU with the reason `math-cr-v1-device-unavailable`. `src/math.rs::cr_v1_device_sites` `docs/numeric.md`

```invar outcome=value:7
fn nudge(x: f32) -> f32 { return x + 0.2; }

fn main() -> i32 {
    if nudge(0.1) == 0.3 { return 7; }
    return 0;
}
```

```invar outcome=diag:TYPE_MISMATCH
fn main() -> i32 {
    let x: f32 = 1;
    return 0;
}
```

```invar outcome=value:9
fn main() -> i32 {
    let root: f32 = sqrt(2.25);
    if root == 1.5 and sin(0.0) == 0.0 and cos(0.0) == 1.0 {
        return i32_from_f32(9.75);
    }
    return 0;
}
```

```invar outcome=trap:INTEGER_OVERFLOW
fn main() -> i32 {
    let big: f32 = 3000000000.0;
    return i32_from_f32(big);
}
```

