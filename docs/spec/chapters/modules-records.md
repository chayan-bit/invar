## 5. Language extensions

### 5.1 Modules and imports

[N5.1.1] A `module` has `pub` and private items, `import m::name;` brings a public item into scope, and forward references inside the file are allowed. `tests/language/records_variants_result.rs::records_variants_result_forward_module_resolution_agrees_across_engines`
[N5.1.2] Duplicate names, invalid visibility, malformed imports and import cycles are source errors, and there are at most 32 modules and 256 imports. `src/check/modules.rs::MAX_MODULES`
[N5.1.3] Module resolution is deterministic and the whole program is one file. `tests/language/records_variants_result.rs`

```invar outcome=value:21
import math::triple;
fn main() -> i32 { return triple(7); }
module math {
    pub fn triple(x: i32) -> i32 { return helper(x) + x; }
    fn helper(x: i32) -> i32 { return x + x; }
}
```

```invar outcome=diag:MODULE_IMPORT_CYCLE
module alpha {
    import beta::other;
    pub fn first() -> i32 { return other(); }
}
module beta {
    import alpha::first;
    pub fn other() -> i32 { return first(); }
}
fn main() -> i32 { return 0; }
```

### 5.2 Records

[N5.2.1] A `struct` has 1 to 16 fields of scalar, array, quantity or nested record types and is built with every field named exactly once, evaluated in source order. `src/check/modules.rs::MAX_RECORD_FIELDS`
[N5.2.2] Reading a scalar field copies it, and an array field needs `move owner.field`, which makes that field unavailable. `tests/language/records_variants_result.rs::records_variants_result_moves_record_array_field_once`
[N5.2.3] A moved record cannot be used again and a missing field is rejected. `tests/language/records_variants_result.rs::records_variants_result_rejects_record_use_after_move`

```invar outcome=value:9
struct Bundle { items: Array<i32>, code: i32 }

fn main() -> i32 {
    let bundle: Bundle = Bundle { items: [3i32, 4i32], code: 5i32 };
    let moved: Array<i32> = move bundle.items;
    return moved[1u32] + bundle.code;
}
```

```invar outcome=diag:OWN_USE_AFTER_MOVE
struct Pair { left: i32 }

fn main() -> i32 {
    let p: Pair = Pair { left: 3i32 };
    let q: Pair = move p;
    return p.left;
}
```

### 5.3 Variants and match

[N5.3.1] An `enum` has 1 to 16 arms with zero or one scalar payload, and a `match` must cover every arm. `src/check/modules.rs::MAX_VARIANT_ARMS`
[N5.3.2] `match owner` borrows the payload and `match move owner` consumes the owner and gives the selected payload to the arm. `tests/language/records_variants_result.rs::records_variants_result_owned_and_borrowed_patterns`
[N5.3.3] Only the selected arm is evaluated, and a missing arm is `MATCH_NONEXHAUSTIVE`. `tests/language/records_variants_result.rs::records_variants_result_rejects_nonexhaustive_variant_match`

```invar outcome=value:11
enum Choice { Low, High(i32) }

fn main() -> i32 {
    let choice: Choice = Choice::High(11i32);
    return match choice {
        Choice::Low => 1i32,
        Choice::High(value) => value,
    };
}
```

```invar outcome=diag:MATCH_NONEXHAUSTIVE
enum Choice { Low, High(i32) }

fn main() -> i32 {
    let choice: Choice = Choice::Low();
    return match choice { Choice::Low => 1i32 };
}
```

### 5.4 Result

[N5.4.1] `Result<T,E>` is a built-in variant with `Ok(T)` and `Err(E)`, nests to depth four, and its constructors need an expected `Result` type. `tests/language/records_variants_result.rs::records_variants_result_rejects_wrong_ownership_and_conversion`
[N5.4.2] Postfix `?` yields the `Ok` payload or returns `Err(E)` from the enclosing `Result<U,E>` function after normal cleanup, and the error type must match exactly. `tests/language/records_variants_result.rs::records_variants_result_nested_result_and_propagation`
[N5.4.3] An `Err` is an ordinary value, distinct from a source trap, an evaluation limit and a tool failure. `tests/language/records_variants_result.rs`

```invar outcome=value:28
fn leaf(ok: Bool) -> Result<i32, u32> {
    if ok { return Result::Ok(7i32); }
    return Result::Err(4u32);
}

fn wrap(ok: Bool) -> Result<i32, u32> {
    let scratch: Array<i32> = [1i32, 2i32];
    let value: i32 = leaf(ok)?;
    return Result::Ok(value + 1i32);
}

fn nested(ok: Bool) -> Result<Result<i32, u32>, u32> {
    let value: Result<i32, u32> = leaf(ok);
    return Result::Ok(move value);
}

fn outer_error() -> Result<Result<i32, u32>, u32> { return Result::Err(9u32); }

fn inspect(ok: Bool) -> i32 {
    let packet: Result<Result<i32, u32>, u32> = nested(ok);
    return match move packet {
        Result::Ok(inner) => match move inner { Result::Ok(number) => number, Result::Err(inner_error) => 0i32 },
        Result::Err(outer_error_code) => 0i32,
    };
}

fn inspect_outer_error() -> i32 {
    let packet: Result<Result<i32, u32>, u32> = outer_error();
    return match move packet {
        Result::Ok(inner_unused) => 0i32,
        Result::Err(code) => 9i32,
    };
}

fn main() -> i32 {
    let good: Result<i32, u32> = wrap(true);
    let bad: Result<i32, u32> = wrap(false);
    let value: i32 = match move good { Result::Ok(x) => x, Result::Err(e) => 0i32 };
    let error: i32 = match move bad { Result::Ok(unused) => 0i32, Result::Err(error_code) => 4i32 };
    return value + error + inspect(true) + inspect(false) + inspect_outer_error();
}
```

