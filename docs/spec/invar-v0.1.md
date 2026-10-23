# Invar v0.1 compact language specification (draft for AI-02)

Status: DRAFT covering behaviour present on `main` at 99b8ffb7, with Appendix A reconciled at 7634aff8, B.4 and B.5 at df9bf05c, and the f64 foreign argument in B.4 at 6437321f; AI-02 is formally the last v0.1 issue, so open v0.1 gaps are listed only in Appendix A and v0.2 features already on `main` only in Appendix B.
Counts: statements=266 citations=282
Size record: tokens=25271 estimator=bytes/3 limit=30000

## 1. Conventions and scope

### 1.1 Normative statements, notes and citations

Note: this file is machine checked by `tests/issue_ai_02_spec_examples.rs`; the checks are described in 1.2.
[N1.1.1] A line starting with `[N` and an identifier is a normative statement and is a requirement on conforming implementations of Invar v0.1. `tests/issue_ai_02_spec_examples.rs::check_spec_text`
[N1.1.2] Every normative statement cites at least one repository file, and a `path::name` citation names text that occurs in that file. `tests/issue_ai_02_spec_examples.rs::issue_ai_02_spec_examples_normative_statements_cite_existing_sources`
[N1.1.3] A line starting with `Note:` is informative and carries no requirement. `tests/issue_ai_02_spec_examples.rs`
[N1.1.4] Where this specification and the implementation disagree, the test fails and the disagreement is fixed or escalated, never papered over. `tests/issue_ai_02_spec_examples.rs`
[N1.1.5] Planned work appears only in clearly labelled roadmap text, open v0.1 gaps appear only in Appendix A, and v0.2 features that are on `main` appear only in Appendix B. `tests/issue_ai_02_spec_examples.rs`

### 1.2 Executable blocks and outcome tags

[N1.2.1] Every fenced block carries an `outcome=` tag and an untagged block fails the test suite. `tests/issue_ai_02_spec_examples.rs::issue_ai_02_spec_examples_rejects_an_untagged_spec_block`
[N1.2.2] Tag `value:N` means `invar run --engine reference` completes and `main` returns the i32 N. `tests/issue_ai_02_spec_examples.rs::run_block`
[N1.2.3] Tag `trap:CATEGORY` means the same command reports a source trap of that category. `tests/issue_ai_02_spec_examples.rs::run_block`
[N1.2.4] Tag `diag:CODE` means `invar check` rejects the source and the first diagnostic has that code. `tests/issue_ai_02_spec_examples.rs::run_block`
[N1.2.5] Tag `limit` with `fuel=N` means the run ends with `evaluation_limit`, and tag `check` means `invar check` accepts the file. `tests/issue_ai_02_spec_examples.rs::run_block`
[N1.2.6] Tag `system:ok` or `system:CODE` runs `invar system check` and requires success or that first diagnostic code. `tests/issue_ai_02_spec_examples.rs::run_block`
[N1.2.7] The flag `foreign` supplies an unused `--c-library` path, because the reference engine runs the Invar oracle and never loads it. `tests/issue_ai_02_spec_examples.rs::run_block`
[N1.2.8] A block that is not Invar source carries `outcome=none`, and an Invar block may not. `tests/issue_ai_02_spec_examples.rs::issue_ai_02_spec_examples_rejects_a_block_whose_outcome_differs`
[N1.2.9] A block whose real outcome differs from its tag fails the suite. `tests/issue_ai_02_spec_examples.rs::issue_ai_02_spec_examples_rejects_a_block_whose_outcome_differs`

### 1.3 Precedence and companion documents

[N1.3.1] The grammar is `spec/grammar.ebnf` and the detailed semantic text is `spec/bootstrap.md`; this file is the compact normative summary of both and of the later contracts. `spec/bootstrap.md`
[N1.3.2] Examples are indexed in `examples/index.json`, validated by `examples/index.schema.json`, and run by the same test binary. `tests/issue_ai_02_spec_examples.rs::issue_ai_02_spec_examples_index_validates_and_every_example_matches_its_expected_output`
[N1.3.3] No safety theorem, native performance result, parallel speedup or AI-authoring advantage is claimed by this specification. `README.md`

## 2. Source text

### 2.1 Lexical rules

[N2.1.1] A source file is one valid UTF-8 text of at most 1,048,576 bytes with syntactic nesting at most 32. `src/front.rs::MAX_SOURCE_BYTES`
[N2.1.2] Invalid UTF-8 is the tool error `INVALID_UTF8`, and lexically invalid UTF-8 text is a source error. `spec/bootstrap.md`
[N2.1.3] Source paths must end in the extension .inv. `src/main.rs::read_source`
[N2.1.4] Comments are `//` to end of line or nested `/* */`, and an unterminated comment is a lexical error. `tests/issue_m0_02.rs::issue_m0_02_lexical_spans_and_json`
[N2.1.5] Spans are zero-based half-open UTF-8 byte intervals, and CRLF does not change byte offsets. `tests/issue_m0_02.rs::issue_m0_02_lexical_spans_and_json`

```invar outcome=value:3
/* a /* nested */ comment */
fn main() -> i32 { return 1 + 2; } // trailing
```

### 2.2 Literals and reserved words

[N2.2.1] A decimal integer literal has an optional `i32` or `u32` suffix, and an unsuffixed literal takes its type from context and otherwise defaults to i32. `spec/bootstrap.md`
[N2.2.2] An identifier character directly after a numeric token is a malformed literal and is not split into two tokens. `tests/issue_m0_02.rs::issue_m0_02_lexical_spans_and_json`
[N2.2.3] The spelling `-2147483648` is a valid i32 and the positive `2147483648i32` is rejected. `spec/bootstrap.md`
[N2.2.4] The words `async await scope break continue total effects requires ensures` are reserved and rejected as `UNSUPPORTED_FEATURE`. `src/front/lexer.rs::unsupported`
[N2.2.5] `break` and `continue` are deferred and not implemented. `spec/bootstrap.md`

```invar outcome=diag:LEX_INVALID_LITERAL
fn main() -> i32 { return 12oops; }
```

```invar outcome=value:-2147483648
fn main() -> i32 { return -2147483648; }
```

```invar outcome=diag:UNSUPPORTED_FEATURE
fn main() -> i32 {
    let total: i32 = 1;
    return total;
}
```

## 3. Core language

### 3.1 Types and values

[N3.1.1] The scalar types are `Bool`, `i32`, `u32` and, as a source increment, `f32`; `Unit` is legal only as a function return type. `spec/bootstrap.md`
[N3.1.2] Scalars are Copy; an `Array<T>` is a non-Copy owner of a fixed-length, fully initialized sequence. `spec/bootstrap.md`
[N3.1.3] Lengths and indices are `u32`, and there is no numeric promotion or implicit conversion between integer types. `spec/bootstrap.md`
[N3.1.4] Array literals, empty or not, need an expected `Array<T>` from an annotation, parameter, return type or assignment destination. `spec/bootstrap.md`
[N3.1.5] `main` is exactly `fn main() -> i32` for `run`; `check` accepts library files without `main`. `tests/issue_m0_03.rs::issue_m0_03_cli_check_json_and_status`
[N3.1.6] A `main` with the wrong signature is the source error `ENTRYPOINT_SIGNATURE` and a missing `main` under `run` is `ENTRYPOINT_MISSING`. `tests/issue_m0_07.rs::issue_m0_07_real_cli_examples_and_statuses`

```invar outcome=diag:TYPE_MISMATCH
fn main() -> i32 {
    let a: i32 = 1;
    let b: u32 = 2u32;
    return a + b;
}
```

### 3.2 Bindings and mutability

[N3.2.1] `let` prevents rebinding and element mutation, and `var` permits both when ownership allows. `spec/bootstrap.md`
[N3.2.2] Local and parameter names are block-scoped, unique within a function, and may not hide a function or an intrinsic. `spec/bootstrap.md`
[N3.2.3] Every non-Unit function must return on all conservatively analysed paths, and a loop never establishes definite return. `tests/issue_m0_03.rs::issue_m0_03_rejects_names_types_and_returns`

```invar outcome=diag:RETURN_MISSING
fn spin(n: i32) -> i32 {
    while n < 0 {
        return n;
    }
}

fn main() -> i32 { return spin(1); }
```

### 3.3 Evaluation order and control flow

[N3.3.1] Expressions and call arguments evaluate left to right, and `and` and `or` short-circuit. `tests/issue_m0_04.rs::issue_m0_04_scalar_order_demand_and_calls`
[N3.3.2] Only the selected `if` arm runs, and conditions must be `Bool` with no truthiness conversion. `spec/bootstrap.md`
[N3.3.3] `for i in lo .. hi` evaluates both `u32` bounds once, iterates the half-open range, and binds an immutable induction variable. `spec/bootstrap.md`
[N3.3.4] An assignment evaluates its right side first, then the destination index, then bounds, then stores. `spec/bootstrap.md`
[N3.3.5] Direct recursion and forward calls are allowed, and recursion is not assumed total. `tests/issue_m0_03.rs::issue_m0_03_forward_calls_recursion_and_typed_order`

```invar outcome=value:2
fn boom() -> Bool {
    let zero: i32 = 0;
    return 1 / zero == 0;
}

fn main() -> i32 {
    if false and boom() { return 1; }
    if true or boom() { return 2; }
    return 3;
}
```

```invar outcome=value:43
fn main() -> i32 {
    var n: i32 = 0;
    var steps: i32 = 0;
    while n < 10 {
        n = n + 3;
        steps = steps + 1;
    }
    var count: i32 = 0;
    for i in 2u32 .. 5u32 {
        count = count + 1;
    }
    return steps * 10 + count;
}
```

### 3.4 Integer arithmetic and traps

[N3.4.1] `+ - *` on `i32` and `u32` and unary `-` on `i32` check representability and trap with `INTEGER_OVERFLOW`, and unary `-` on `u32` is rejected. `tests/issue_m0_04.rs::issue_m0_04_arithmetic_boundaries`
[N3.4.2] Division and remainder by zero trap with `DIVISION_BY_ZERO`, and `i32::MIN / -1` traps with `INTEGER_OVERFLOW`. `tests/issue_m0_04.rs::issue_m0_04_arithmetic_boundaries`
[N3.4.3] Signed division truncates toward zero, the remainder has the dividend's sign, and `MIN % -1` is `0`. `spec/bootstrap.md`
[N3.4.4] `wrapping_add` and `wrapping_mul` take two `u32` operands and compute modulo 2^32. `spec/bootstrap.md`
[N3.4.5] Arithmetic behaves identically in debug and release builds. `spec/bootstrap.md`
[N3.4.6] The source trap categories are `INTEGER_OVERFLOW`, `DIVISION_BY_ZERO`, `INDEX_OUT_OF_BOUNDS` and the unit trap `UNIT_CONVERSION_INEXACT` (6.1.5), and a trap records the operator or index span of the first failing operation in evaluation order. `spec/bootstrap.md`
[N3.4.7] A trap performs no language-level cleanup, and it is distinct from an evaluation limit and from a tool error. `docs/validation.md`

```invar outcome=trap:INTEGER_OVERFLOW
fn main() -> i32 {
    let largest: i32 = 2147483647;
    return largest + 1;
}
```

```invar outcome=diag:TYPE_MISMATCH
fn main() -> i32 {
    let x: u32 = 1u32;
    let y: u32 = -x;
    return 0;
}
```

```invar outcome=trap:DIVISION_BY_ZERO
fn main() -> i32 {
    let zero: i32 = 0;
    return 10 / zero;
}
```

```invar outcome=value:-310
fn main() -> i32 {
    let minimum: i32 = -2147483648;
    let q: i32 = -7 / 2;
    let r: i32 = -7 % 2;
    let edge: i32 = minimum % -1;
    return q * 100 + r * 10 + edge;
}
```

```invar outcome=value:5
fn main() -> i32 {
    let wrapped: u32 = wrapping_add(4294967295u32, 2u32);
    if wrapped == 1u32 { return 5; }
    return 0;
}
```

### 3.5 Arrays and intrinsics

[N3.5.1] `len(xs)` returns the `u32` length, and `filled_i32(n, x)` and `filled_u32(n, x)` build owned arrays of `n` copies. `spec/bootstrap.md`
[N3.5.2] Indexing needs a `u32` index and an out-of-range index traps with `INDEX_OUT_OF_BOUNDS`. `tests/issue_m0_05.rs::issue_m0_05_array_values_indexing_and_filling`
[N3.5.3] Array equality and array arithmetic are unsupported. `spec/bootstrap.md`

```invar outcome=trap:INDEX_OUT_OF_BOUNDS
fn main() -> i32 {
    let xs: Array<i32> = [1, 2, 3];
    return xs[3u32];
}
```

### 3.6 Fuel

[N3.6.1] Each expression evaluation and statement dispatch consumes one unit of fuel, and the default limit is 1,000,000. `spec/bootstrap.md`
[N3.6.2] Exhausting fuel reports `evaluation_limit` with `steps_used` equal to the limit and is never a proof of divergence. `tests/issue_m0_04.rs::issue_m0_04_fuel_and_call_depth`

```invar outcome=limit fuel=1000
fn main() -> i32 {
    var n: i32 = 0;
    while true { n = 1; }
    return n;
}
```

## 4. Ownership

### 4.1 Moves

[N4.1.1] An array owner moves only with `move name`, from a fresh expression, or by `return name`, and a bare owner-to-owner binding is rejected rather than copied. `spec/bootstrap.md`
[N4.1.2] Reading a moved owner is `OWN_USE_AFTER_MOVE`, and a move of an outer binding inside a loop is rejected. `tests/issue_m0_05.rs::issue_m0_05_rejects_invalid_ownership`
[N4.1.3] At a join of continuing `if` paths an owner is usable only if usable on every continuing path. `spec/bootstrap.md`

```invar outcome=diag:OWN_USE_AFTER_MOVE
fn take_it(xs: take Array<i32>) -> i32 { return xs[0u32]; }

fn main() -> i32 {
    var xs: Array<i32> = [4, 5];
    let first: i32 = take_it(move xs);
    return first + xs[1u32];
}
```

### 4.2 Call loans

[N4.2.1] An array parameter has exactly one mode: `read`, `mut` or `take`, and scalar parameters take no mode. `spec/bootstrap.md`
[N4.2.2] Borrowed arguments are bare names, read loans may coexist, and a mutable loan excludes every other access to that array. `tests/issue_m0_06.rs::issue_m0_06_rejects_conflicts_at_call_boundary`
[N4.2.3] A loan is acquired after its argument is evaluated and held until the outer call completes, with no two-phase borrow. `spec/bootstrap.md`
[N4.2.4] A `read` or `mut` parameter cannot be returned as an owner, and a borrower cannot escape its call. `tests/issue_m0_06.rs::issue_m0_06_rejects_conflicts_at_call_boundary`
[N4.2.5] Rejected borrows never trigger implicit cloning or reference counting. `docs/architecture.md`

```invar outcome=diag:OWN_BORROW_CONFLICT
fn both(a: mut Array<i32>, b: read Array<i32>) -> i32 { return a[0u32] + b[0u32]; }

fn main() -> i32 {
    var xs: Array<i32> = [1, 2];
    return both(xs, xs);
}
```

### 4.3 Cleanup

[N4.3.1] Normal exits release still-owned arrays in reverse declaration order for each exited scope, and a move suppresses release. `tests/issue_m0_05.rs::issue_m0_05_cleanup_and_failure_order`
[N4.3.2] The reference evaluator records a cleanup trace, which is a test observation and not a source feature. `docs/validation.md`

## 5. Language extensions

### 5.1 Modules and imports

[N5.1.1] A `module` has `pub` and private items, `import m::name;` brings a public item into scope, and forward references inside the file are allowed. `tests/issue_l1_01_language_core.rs::issue_l1_01_language_core_forward_module_resolution_agrees_across_engines`
[N5.1.2] Duplicate names, invalid visibility, malformed imports and import cycles are source errors, and there are at most 32 modules and 256 imports. `src/check/modules.rs::MAX_MODULES`
[N5.1.3] Module resolution is deterministic and the whole program is one file. `tests/issue_l1_01_language_core.rs`

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
[N5.2.2] Reading a scalar field copies it, and an array field needs `move owner.field`, which makes that field unavailable. `tests/issue_l1_01_language_core.rs::issue_l1_01_language_core_moves_record_array_field_once`
[N5.2.3] A moved record cannot be used again and a missing field is rejected. `tests/issue_l1_01_language_core.rs::issue_l1_01_language_core_rejects_record_use_after_move`

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
[N5.3.2] `match owner` borrows the payload and `match move owner` consumes the owner and gives the selected payload to the arm. `tests/issue_l1_01_language_core.rs::issue_l1_01_language_core_owned_and_borrowed_patterns`
[N5.3.3] Only the selected arm is evaluated, and a missing arm is `MATCH_NONEXHAUSTIVE`. `tests/issue_l1_01_language_core.rs::issue_l1_01_language_core_rejects_nonexhaustive_variant_match`

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

[N5.4.1] `Result<T,E>` is a built-in variant with `Ok(T)` and `Err(E)`, nests to depth four, and its constructors need an expected `Result` type. `tests/issue_l1_01_language_core.rs::issue_l1_01_language_core_result_rejects_wrong_ownership_and_conversion`
[N5.4.2] Postfix `?` yields the `Ok` payload or returns `Err(E)` from the enclosing `Result<U,E>` function after normal cleanup, and the error type must match exactly. `tests/issue_l1_01_language_core.rs::issue_l1_01_language_core_nested_result_and_propagation`
[N5.4.3] An `Err` is an ordinary value, distinct from a source trap, an evaluation limit and a tool failure. `spec/bootstrap.md`

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

### 5.5 Borrowed views

[N5.5.1] `View<T>` and `MutView<T>` over `i32` and `u32` are non-owning views made by `view(owner, start, end[, stride])` and `view_mut(...)` from a bare array or view name. `tests/issue_l1_02_borrowed_views.rs::issue_l1_02_borrowed_views_reference_mir_native_agree`
[N5.5.2] The length is `ceil((end-start)/stride)`, the stride is a positive `u32`, and range, stride and composition are checked in `u32` arithmetic. `tests/issue_l1_02_borrowed_views.rs::issue_l1_02_borrowed_views_range_and_stride_trap_identically`
[N5.5.3] An invalid range or stride traps as `INDEX_OUT_OF_BOUNDS`, identically in the reference, MIR and native engines. `tests/issue_l1_02_borrowed_views.rs::issue_l1_02_borrowed_views_range_and_stride_trap_identically`
[N5.5.4] An owner cannot be moved, dropped or mutated while a view of it is live, and mutable views of one owner need provably disjoint intervals. `tests/issue_l1_02_borrowed_views.rs::issue_l1_02_borrowed_views_public_check_and_negative_sources`
[N5.5.5] A function returns a view only with `-> View<T> from xs` where `xs` is a parameter, and a view of a local owner cannot be returned. `tests/issue_l1_02_borrowed_views.rs::issue_l1_02_borrowed_views_read_return_from_mut_parameter`
[N5.5.6] There are at most 4096 view facts per program, and views have no view parameters, no view fields and no `f32` element type. `src/check/views.rs`

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

### 5.6 Generics

[N5.6.1] A generic function has one or two parameters bounded by `Value`, which stands for any complete concrete non-view value type and implies no operations. `tests/issue_l1_03_bounded_generics_closures.rs::issue_l1_03_bounded_generics_closures_monomorphized_functions_agree`
[N5.6.2] Type arguments are inferred only from exact argument types, never from the expected result, or are given with `f::<T, U>(...)`. `tests/issue_l1_03_bounded_generics_closures.rs::issue_l1_03_bounded_generics_closures_public_check_rejects_invalid_sources`
[N5.6.3] An instance is identified by the qualified function name and its concrete type arguments, is built once and reused, and recursive instantiation is rejected. `tests/issue_l1_03_bounded_generics_closures.rs::issue_l1_03_bounded_generics_closures_module_identity_and_unused_take_cleanup`
[N5.6.4] Expansion is limited to 16 generic frames and 128 lowered functions, and traits and higher-kinded syntax are unsupported. `src/check/generics.rs::MAX_GENERIC_DEPTH`

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
[N5.7.2] `read` borrows the owner until the call or scope exit, `take` moves the owner at creation and cleans it up if the closure is never called. `tests/issue_l1_03_bounded_generics_closures.rs::issue_l1_03_bounded_generics_closures_module_identity_and_unused_take_cleanup`
[N5.7.3] A closure cannot escape, be assigned, passed or returned, and its body may use only its captures, parameters and global functions. `tests/issue_l1_03_bounded_generics_closures.rs::issue_l1_03_bounded_generics_closures_public_check_rejects_invalid_sources`

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
[N5.8.3] Only an explicit `fma(a, b, c)` fuses a multiply and an add, and `f32_from_i32`, `f32_from_u32` and `filled_f32` are the conversions and constructor. `tests/issue_m4_01_source_f32.rs::issue_m4_01_source_f32_strict_arithmetic_and_ieee_predicates`
[N5.8.4] An integer is not implicitly an `f32`, `%` is not defined on `f32`, and `main` still returns `i32`. `tests/issue_m4_01_source_f32.rs::issue_m4_01_source_f32_rejects_implicit_conversion_and_remainder`
[N5.8.5] `i32_from_f32(x)` truncates a finite `f32` toward zero and traps with `INTEGER_OVERFLOW` when the value is not finite or outside the `i32` range. `docs/numeric.md::i32_from_f32`
[N5.8.6] `sqrt`, `sin` and `cos` on `f32` are correctly rounded under the default math contract `invar-math-cr-v1` (Appendix B.4), which supersedes `invar-trig-f32-v1`, and every engine returns the same 32-bit word. `src/math.rs`

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

## 6. Units and frames

### 6.1 Quantities and dimension algebra

[N6.1.1] `Quantity<T, Unit>` carries a unit in its type, built by `quantity::<unit>(x)`, and the unit is erased before MIR so generated code equals the unit-free program. `tests/issue_r1_01_units_frames.rs::issue_r1_01_units_frames_quantity_erasure_is_byte_identical`
[N6.1.2] The known units are `one m mm um kg g s ms us ns A K mol cd rad urad deg Hz N Nm px`, with dimension product, quotient and integer power in the type. `src/check/unit_math.rs`
[N6.1.3] `+`, `-` and comparison need identical units, `*` and `/` combine dimensions, and `1 / s` is `Hz`. `tests/issue_r1_01_units_frames.rs::issue_r1_01_units_frames_dimension_products_quotients_and_rates`
[N6.1.4] `to::<unit>(q)` is the only conversion between scales, there is no implicit conversion, and a unit is not a value. `tests/issue_r1_01_units_frames.rs::issue_r1_01_units_frames_quantity_source_errors`
[N6.1.5] A scale conversion that is not exact traps with `UNIT_CONVERSION_INEXACT`. `tests/issue_ai_01_machine_diagnostics/indexed_and_versions.rs::issue_ai_01_machine_diagnostics_conversion_trap_is_versioned`
[N6.1.6] `rad` and `deg` conversions use a certified bound on pi and round to nearest even. `src/angle.rs`
[N6.1.7] `px` is a nominal camera unit distinct from `one` and every SI unit. `spec/grammar.ebnf`

```invar outcome=value:1
fn main() -> i32 {
    let force: Quantity<i32, N> = quantity::<N>(2i32);
    let arm: Quantity<i32, m> = quantity::<m>(3i32);
    let torque: Quantity<i32, Nm> = force * arm;
    let second: Quantity<i32, s> = quantity::<s>(1i32);
    let frequency: Quantity<i32, Hz> = 1i32 / second;
    if torque == quantity::<Nm>(6i32) and frequency == 1i32 / second { return 1i32; }
    return 0i32;
}
```

```invar outcome=diag:UNIT_MISMATCH
fn main() -> i32 {
    let x: Quantity<i32, mm> = quantity::<mm>(1i32);
    let y: Quantity<i32, m> = quantity::<m>(1i32);
    let z: Quantity<i32, mm> = x + y;
    return 0i32;
}
```

```invar outcome=trap:UNIT_CONVERSION_INEXACT
fn main() -> i32 {
    let mm: Quantity<i32, mm> = quantity::<mm>(1i32);
    let metres: Quantity<i32, m> = to::<m>(mm);
    return 0i32;
}
```

### 6.2 Frames and transforms

[N6.2.1] `frame Name;` declares a root and `frame Name: Parent;` a child, and the tree needs exactly one root, no cycle, no duplicate and no unknown parent. `tests/issue_r1_01_units_frames.rs::issue_r1_01_units_frames_nominal_tree_public_check`
[N6.2.2] `Vector3<T,unit,frame>`, `Point<T,frame>` and `Transform<From,To>` are nominal and are made by `vector3`, `point` and `transform`. `tests/issue_r1_01_units_frames.rs::issue_r1_01_units_frames_transform_chain_agrees`
[N6.2.3] `Transform<A,B> * Transform<B,C>` is `Transform<A,C>`, any other composition is `FRAME_MISMATCH`, and a transform also applies to a point or vector in its source frame. `tests/issue_r1_01_units_frames.rs::issue_r1_01_units_frames_transform_chain_agrees`
[N6.2.4] A transform rotation is a constant proper signed permutation, and Point and translation coordinates are integer micrometres. `spec/grammar.ebnf`
[N6.2.5] There are at most 128 frames. `src/check/frames.rs::MAX_FRAMES`

```invar outcome=value:1
frame world;
frame base: world;
frame tool: base;

fn main() -> i32 {
    let v: Vector3<i32,um,world> = vector3::<um,world>(1000, 0, 0);
    let p: Point<i32,world> = point::<world>(1000, 0, 0);
    let a: Transform<world,base> = transform::<world,base>(1, 2, 3, 1,0,0, 0,1,0, 0,0,1);
    let b: Transform<base,tool> = transform::<base,tool>(0, 0, 0, 1,0,0, 0,1,0, 0,0,1);
    let composed: Transform<world,tool> = a * b;
    let target: Point<i32,tool> = composed * p;
    let shift: Vector3<i32,um,tool> = composed * v;
    let moved: Point<i32,tool> = target + shift;
    if moved[0u32] == quantity::<um>(2001) { return 1; }
    return 0;
}
```

```invar outcome=diag:FRAME_MISMATCH
frame world;
frame base: world;

fn main() -> i32 {
    let a: Vector3<i32,m,world> = vector3::<m,world>(1, 0, 0);
    let b: Vector3<i32,m,base> = vector3::<m,base>(1, 0, 0);
    let c: Vector3<i32,m,world> = a + b;
    return 0;
}
```

```invar outcome=diag:FRAME_CYCLE
frame world;
frame a: b;
frame b: a;
fn main() -> i32 { return 0i32; }
```

### 6.3 Robotics vocabulary

[N6.3.1] `std::robotics` is the shipped qualified module of generic record types for poses, twists, wrenches, joint vectors, stamps and camera models, with C layouts and ROS mappings. `tests/issue_s1_11_robotics_vocabulary.rs::issue_s1_11_robotics_vocabulary_standard_module_checks`
[N6.3.2] `f32` and `f64` are accepted as wire types of ports, and `f64` is a type only with no arithmetic. `spec/grammar.ebnf`
[N6.3.3] Camera intrinsics use pixel units, and a focal length in metres is `UNIT_MISMATCH`. `tests/issue_s1_11_robotics_vocabulary.rs::issue_s1_11_robotics_vocabulary_pixel_is_nominal`
[N6.3.4] Joint-vector types carry their count, and a count mismatch is rejected. `tests/issue_s1_11_robotics_vocabulary.rs::issue_s1_11_robotics_vocabulary_rejects_mismatched_units_and_joint_counts`

## 7. Bytes, strings and the C ABI

### 7.1 Bytes and String

[N7.1.1] `Bytes` and `String` are distinct immutable owned values of at most 1,048,576 bytes, and `len` is the byte length. `tests/issue_l1_04_bytes_strings_cabi.rs::issue_l1_04_bytes_strings_cabi_large_bounded_value`
[N7.1.2] `b"..."` allows `\xNN` escapes and arbitrary bytes, `"..."` must be valid UTF-8 without `\xNN`, and there is no normalization or implicit conversion. `tests/issue_l1_04_bytes_strings_cabi.rs::issue_l1_04_bytes_strings_cabi_utf8_and_byte_values_agree`
[N7.1.3] `Bytes` indexing yields a `u32`, an out-of-range index traps with `INDEX_OUT_OF_BOUNDS`, and `String` has no indexing. `tests/issue_l1_04_bytes_strings_cabi.rs::issue_l1_04_bytes_strings_cabi_public_failures`

```invar outcome=value:7
fn main() -> i32 {
    let x: Bytes = b"\xFF";
    let s: String = "é";
    if x[0u32] == 255u32 and len(s) == 2u32 { return 7; }
    return 0;
}
```

```invar outcome=trap:INDEX_OUT_OF_BOUNDS
fn main() -> i32 {
    let bytes: Bytes = b"a";
    if bytes[1u32] == 0u32 { return 1; }
    return 0;
}
```

### 7.2 UTF-8 conversions

[N7.2.1] `bytes_slice(owner, a, b)` copies a range into `Bytes`, `utf8_slice` returns `Result<String,Utf8Error>`, `utf8_decode` validates and `utf8_encode` copies. `tests/issue_l1_04_bytes_strings_cabi.rs::issue_l1_04_bytes_strings_cabi_utf8_and_byte_values_agree`
[N7.2.2] These borrow a bare named owner, a reversed or out-of-range range traps, and malformed UTF-8 or a split code point is an ordinary `Err(Utf8Error)` value. `tests/issue_l1_04_bytes_strings_cabi.rs::issue_l1_04_bytes_strings_cabi_public_failures`

```invar outcome=value:1
fn main() -> i32 {
    let raw: Bytes = b"\xFF";
    let result = utf8_decode(raw);
    return match move result { Result::Ok(text) => 0i32, Result::Err(error) => 1i32 };
}
```

### 7.3 Foreign C declarations

[N7.3.1] A top-level `extern c fn symbol(args) -> Result<T,ForeignError> oracle name;` declares at most 8 arguments from `i32`, `u32`, `read` or `take` `Bytes` and `String`. `tests/issue_l1_04_bytes_strings_cabi.rs::issue_l1_04_bytes_strings_cabi_unsupported_signatures_reject_publicly`
[N7.3.2] The named oracle is an ordinary Invar function with the exact signature, and `check` never loads a library. `tests/issue_l1_04_bytes_strings_cabi.rs::issue_l1_04_bytes_strings_cabi_foreign_oracle_declares_and_checks`
[N7.3.3] Reference and MIR execution run the oracle, and native execution needs an explicit absolute `--c-library`. `tests/issue_l1_04_bytes_strings_cabi.rs::issue_l1_04_bytes_strings_cabi_foreign_buffers_and_failures_agree`
[N7.3.4] A `take` argument is consumed even when C reports an error, so a later use is `OWN_USE_AFTER_MOVE`. `tests/fixtures/l1_04/foreign-use-after-take.inv`
[N7.3.5] Callbacks, borrowed returns, mutable buffers, C strings and general FFI are rejected. `tests/issue_l1_04_bytes_strings_cabi.rs::issue_l1_04_bytes_strings_cabi_unsupported_signatures_reject_publicly`
[N7.3.6] Foreign code is outside the safe-language claims, and the C ABI does not make native dependencies safe. `docs/architecture.md`

```invar outcome=value:7 foreign
extern c fn c_add(left: i32, right: i32) -> Result<i32, ForeignError> oracle add_oracle;

fn add_oracle(left: i32, right: i32) -> Result<i32, ForeignError> {
    return Result::Ok(left + right);
}

fn main() -> i32 {
    let answer: Result<i32, ForeignError> = c_add(3i32, 4i32);
    return match move answer { Result::Ok(value) => value, Result::Err(error) => 0i32 };
}
```

### 7.4 The C ABI shim

[N7.4.1] The C shim is `int32_t symbol(const inv_arg_v1 *args, uint32_t argc, inv_result_v1 *out)` with tags, modes and 64-bit lengths frozen in `include/invar_foreign_v1.h`. `include/invar_foreign_v1.h`
[N7.4.2] The library must export `inv_abi_version_v1() == 1`, and a wrong version or missing symbol becomes `Err(ForeignError::Failure())`. `tests/issue_l1_04_bytes_strings_cabi.rs::issue_l1_04_bytes_strings_cabi_detectable_foreign_failures_are_typed`
[N7.4.3] Buffer arguments are valid only during the call, and a C-owned result buffer must supply a release callback that runs exactly once after validation and copy. `tests/fixtures/l1_04/foreign.c`
[N7.4.4] A result buffer over 1 MiB, bad tag or version, a null with nonzero length or invalid UTF-8 is a typed error and never truncates. `tests/issue_l1_04_bytes_strings_cabi.rs::issue_l1_04_bytes_strings_cabi_detectable_foreign_failures_are_typed`

## 8. System dialect

### 8.1 Declarations

[N8.1.1] A system description is ordinary Invar plus `component`, `extern component`, `connect`, `flow`, `frame`, `product` and `system`. `docs/system-integration-v1.md`
[N8.1.2] A `component` has `port`, `param`, `resource`, `license`, `isolation`, `wcet`, `deadline` and `impl` clauses. `spec/grammar.ebnf`
[N8.1.3] A port is `port in|out name: Type @ RATE [latency L] [layout c "name"] [bus name] [optional];`, and a kernel signature and a port type are the same type. `spec/grammar.ebnf`
[N8.1.4] `connect a.p -> b.q policy P;` joins an output to an input where the policy is `sample-and-hold`, `queue(n)` or `decimate(n)`. `docs/system-integration-v1.md`
[N8.1.5] Constants are exact decimal literals with units, evaluated as exact rationals and serialized as canonical decimal strings. `docs/system-integration-v1.md`
[N8.1.6] A source needs exactly one `product` and one `system`, and an instance is `name: Component;` inside the system. `tests/fixtures/s1_01/negative/S1005-no-product.inv`
[N8.1.7] An optional input or an unconnected output is the warning `S1011`, and other well-formedness failures are `S1000` to `S1010` errors. `tests/issue_s1_01_system_dialect.rs::issue_s1_01_system_dialect_optional_input_and_output_warning`
[N8.1.8] `connect` requires equal type, unit and frame at both ends, else `S1002`. `tests/issue_s1_01_system_dialect.rs::issue_s1_01_system_dialect_negative_fixtures_have_registered_spans`
[N8.1.9] A system has no runtime semantics, and it is evaluated at compile time within the usual fuel and size limits. `docs/system-integration-v1.md`

```invar outcome=system:ok
frame world;

component Sensor {
    port out reading: Quantity<i32,m> @ 10 Hz;
    resource cpu_cores = 1;
    license "MIT" modified false;
    isolation in-process;
    wcet 1 ms;
    deadline 100 ms;
    impl invar read_sensor;
}

component Logger {
    port in reading: Quantity<i32,m> @ 10 Hz;
    resource cpu_cores = 1;
    license "MIT" modified false;
    isolation in-process;
    wcet 1 ms;
    deadline 100 ms;
    impl invar log_reading;
}

fn read_sensor() -> i32 { return 0; }
fn log_reading() -> i32 { return 0; }

connect sensor.reading -> logger.reading policy queue(4);

product Bench {
    distribution internal;
    commercial false;
    robot menagerie::franka_emika_panda;
    capacity cpu_cores = 4;
}

system cell for Bench { sensor: Sensor; logger: Logger; }
```

```invar outcome=system:S1002
frame world;

component Sensor {
    port out reading: Quantity<i32,m> @ 10 Hz;
    resource cpu_cores = 1;
    license "MIT" modified false;
    isolation in-process;
    wcet 1 ms;
    deadline 100 ms;
    impl invar read_sensor;
}

component Logger {
    port in reading: Quantity<i32,mm> @ 10 Hz;
    resource cpu_cores = 1;
    license "MIT" modified false;
    isolation in-process;
    wcet 1 ms;
    deadline 100 ms;
    impl invar log_reading;
}

fn read_sensor() -> i32 { return 0; }
fn log_reading() -> i32 { return 0; }

connect sensor.reading -> logger.reading policy queue(4);

product Bench {
    distribution internal;
    commercial false;
    robot menagerie::franka_emika_panda;
    capacity cpu_cores = 4;
}

system cell for Bench { sensor: Sensor; logger: Logger; }
```

### 8.2 The invar-system IR

[N8.2.1] A checked system is serialized as canonical JSON named `invar-system-vN`, byte-identical for identical input, with one schema file per version in `schema/`. `tests/issue_s1_01_system_dialect.rs::issue_s1_01_system_dialect_emits_checked_canonical_ir`
[N8.2.2] The checker emits the lowest version that can represent the source: v2 when a bus, identity transform or license detail is used, v3 for typed config, input paths or golden tolerances, v4 for parameter ranges, v5 for robot and cell models, v6 for tool and dynamic objects, v7 and v8 for planner records and config, v9 and v10 for the planner frame policy and grasp, v11 for an actuator `hold` command and v12 for a cell mount surface. `src/system/compiler.rs::use_v2`
[N8.2.3] An unknown version is rejected, and relabelling a document does not make it valid. `tests/issue_s1_03_graph_checks.rs::issue_s1_03_graph_checks_twelve_component_cell_and_golden_report`
[N8.2.4] Permuting module order of the same system gives byte-identical IR. `tests/issue_s1_01_system_dialect.rs::issue_s1_01_system_dialect_module_permutation_is_byte_identical`
[N8.2.5] Deserializing an IR does not establish a source check, and only a source-compiled model can pass the graph checks. `tests/issue_s1_03_graph_checks.rs::issue_s1_03_graph_checks_twelve_component_cell_and_golden_report`

### 8.3 Generators

[N8.3.1] Every generator is a pure function of the checked IR with byte-identical output, and it reads no source text, environment, clock or network. `docs/system-integration-v1.md`
[N8.3.2] `invar system gen ros2|mjcf|urdf PATH --out NEW_DIRECTORY` writes a new directory and refuses an existing one. `tests/issue_s1_04_ros2_target.rs::issue_s1_04_ros2_target_public_errors_leave_no_output`
[N8.3.3] The ROS 2 target needs a graph-checked system, else `S1500`, and its output is deterministic. `tests/issue_s1_04_ros2_target.rs::issue_s1_04_ros2_target_public_generation_is_deterministic`
[N8.3.4] A system with a robot and cell model also generates MJCF and, where it can be preserved, URDF, and unsupported URDF features are `S1607`. `docs/system-integration-v1.md`
[N8.3.5] A generated ROS 2 workspace is mock-hardware accepted on ARM64 only, and a live simulation run is not claimed. `README.md`

## 9. Graph checks

[N9.0.1] Graph checks run when the IR is v2 or later, and a v1-shaped source gets only S1-01 checks and reports `checks_pending` with `system-v2-graph-checks`. `tests/issue_s1_03_graph_checks.rs::issue_s1_03_graph_checks_v1_remains_readable_with_graph_work_pending`
[N9.0.2] Diagnostics are ordered by family and then span, and the same input gives the same output. `tests/issue_s1_03_graph_checks.rs::issue_s1_03_graph_checks_diagnostic_families_keep_stable_order`
[N9.0.3] The checker proves structure only, and deadlines and WCET are declared budgets and not measurements. `docs/system-integration-v1.md`

### 9.1 Rates and queues

[N9.1.1] Under `queue(n)` the producer rate may not exceed the consumer rate, else `S1100`. `src/system/graph.rs`
[N9.1.2] A component deadline may not exceed the period of its ports, else `S1101`, and a WCET above its deadline is `S1101`. `tests/fixtures/s1_03/negative/S1101-wcet-over-deadline.inv`
[N9.1.3] `decimate(n)` needs an integral producer to consumer ratio, else `S1100`. `tests/fixtures/s1_03/negative/S1100-decimate-ratio.inv`

```invar outcome=system:S1101
frame world;

component Sensor {
    port out reading: Quantity<i32,m> @ 10 Hz;
    resource cpu_cores = 1;
    license "MIT" modified false;
    isolation in-process;
    wcet 200 ms;
    deadline 100 ms;
    impl invar read_sensor;
}

component Logger {
    port in reading: Quantity<i32,m> @ 10 Hz;
    resource cpu_cores = 1;
    license "MIT" modified false;
    isolation in-process;
    wcet 1 ms;
    deadline 100 ms;
    impl invar log_reading;
}

fn read_sensor() -> i32 { return 0; }
fn log_reading() -> i32 { return 0; }

connect sensor.reading -> logger.reading policy queue(4);

product Bench {
    distribution internal;
    commercial false;
    robot menagerie::franka_emika_panda;
    capacity cpu_cores = 4;
}

system cell for Bench { sensor: Sensor; logger: Logger; }
```

### 9.2 Latency, flows and budgets

[N9.2.1] A flow `flow f: a.p -> b.q within T;` sums the WCET of each distinct instance on it, each hop's source-port latency, and the consumer period for `sample-and-hold` and `decimate` hops, and the sum may not exceed `T`, else `S1102`. `src/system/graph.rs`
[N9.2.2] A flow hop that is not a connection is `S1102`. `tests/fixtures/s1_03/negative/S1102-disconnected-flow.inv`

```invar outcome=system:S1102
frame world;

component Sensor {
    port out reading: Quantity<i32,m> @ 10 Hz;
    resource cpu_cores = 1;
    license "MIT" modified false;
    isolation in-process;
    wcet 1 ms;
    deadline 100 ms;
    impl invar read_sensor;
}

component Logger {
    port in reading: Quantity<i32,m> @ 10 Hz;
    resource cpu_cores = 1;
    license "MIT" modified false;
    isolation in-process;
    wcet 1 ms;
    deadline 100 ms;
    impl invar log_reading;
}

fn read_sensor() -> i32 { return 0; }
fn log_reading() -> i32 { return 0; }

connect sensor.reading -> logger.reading policy queue(4);
flow loop: sensor.reading -> logger.reading within 1 ms;

product Bench {
    distribution internal;
    commercial false;
    robot menagerie::franka_emika_panda;
    capacity cpu_cores = 4;
}

system cell for Bench { sensor: Sensor; logger: Logger; }
```

### 9.3 Frame-tree closure

[N9.3.1] Every frame used by a port or transform must be reachable from exactly one root, and an unreachable frame or a second root is `S1200`. `tests/fixtures/s1_03/negative/S1200-unreachable-frame.inv`
[N9.3.2] Every tree edge used by a transform needs one declared transform source, else `S1201`, and two sources are `S1202`. `tests/fixtures/s1_03/negative/S1201-missing-transform-source.inv`

### 9.4 Resources

[N9.4.1] The sum of each instance's `cpu_cores`, `memory`, `gpu_memory` and per-bus bandwidth may not exceed the product `capacity`, else `S1250`. `tests/fixtures/s1_03/negative/S1250-cpu-over.inv`
[N9.4.2] Bus bandwidth is declared port rate times canonical C layout bytes times 8, and it excludes serialization and network overhead. `src/system.rs::report`
[N9.4.3] A bus not declared by the product is `S1251`, and a port type with no canonical layout is `S1252`. `tests/fixtures/s1_03/negative/S1251-undeclared-bus.inv`

```invar outcome=system:S1250
frame world;

component Sensor {
    port out reading: Quantity<i32,m> @ 10 Hz;
    resource cpu_cores = 1;
    license "MIT" modified false;
    isolation in-process;
    wcet 1 ms;
    deadline 100 ms;
    impl invar read_sensor;
}

component Logger {
    port in reading: Quantity<i32,m> @ 10 Hz;
    resource cpu_cores = 2;
    license "MIT" modified false;
    isolation in-process;
    wcet 1 ms;
    deadline 100 ms;
    impl invar log_reading;
}

fn read_sensor() -> i32 { return 0; }
fn log_reading() -> i32 { return 0; }

connect sensor.reading -> logger.reading policy queue(4);

product Bench {
    distribution internal;
    commercial false;
    robot menagerie::franka_emika_panda;
    capacity cpu_cores = 1;
}

system cell for Bench { sensor: Sensor; logger: Logger; }
```

### 9.5 License and isolation

[N9.5.1] Each component carries an SPDX expression and one isolation of `in-process`, `separate-process` or `separate-partition`, and an unknown or unparsable license is rejected. `tests/fixtures/s1_03/negative/S1305-unknown-no-grant.inv`
[N9.5.2] For `distribution proprietary-binary`, GPL must be a separate unmodified process and AGPL also may not be network exposed. `docs/system-integration-v1.md`
[N9.5.3] LGPL in process must be dynamic and replaceable, MPL-2.0 in process must be unmodified, and non-commercial terms are rejected when `commercial true`. `docs/system-integration-v1.md`
[N9.5.4] Proprietary or unknown third-party terms need a declared grant id, and these checks are an engineering check and not legal advice. `docs/system-integration-v1.md`
[N9.5.5] The v0.1 check covers components, and the check over the Nix image closure is not part of v0.1. `docs/system-integration-v1.md`

```invar outcome=system:S1300
frame world;

component Sensor {
    port out reading: Quantity<i32,m> @ 10 Hz;
    resource cpu_cores = 1;
    license "MIT" modified false;
    isolation in-process;
    wcet 1 ms;
    deadline 100 ms;
    impl invar read_sensor;
}

component Logger {
    port in reading: Quantity<i32,m> @ 10 Hz;
    resource cpu_cores = 1;
    license "GPL-3.0-only" modified false;
    isolation in-process;
    wcet 1 ms;
    deadline 100 ms;
    impl invar log_reading;
}

fn read_sensor() -> i32 { return 0; }
fn log_reading() -> i32 { return 0; }

connect sensor.reading -> logger.reading policy queue(4);

product Bench {
    distribution proprietary-binary;
    commercial true;
    robot menagerie::franka_emika_panda;
    capacity cpu_cores = 4;
}

system cell for Bench { sensor: Sensor; logger: Logger; }
```

## 10. Foreign component manifests

### 10.1 Manifest clauses

[N10.1.1] An `extern component` declares `abi c "prefix"`, `lifecycle init, configure, step, shutdown`, typed ports with C layouts, resources, an SPDX license, isolation, `golden`, `property` and `shim` clauses. `tests/fixtures/s1_02/manifest-valid.inv`
[N10.1.2] `config c "name"` names a registered typed layout and `input "path"` declares an extra dependency, and only the initial layouts `inv_fk_config_v1` and `inv_planar2_config_v1` and the planner config are registered. `spec/bootstrap.md`
[N10.1.3] A `golden` clause may state up to eight nonnegative exact-decimal absolute tolerances, and an unsupported layout or property text is `S1400`. `tests/issue_s1_02_foreign_manifests.rs::issue_s1_02_foreign_manifests_typed_config_and_tolerance_are_checked`
[N10.1.4] The foreign-component diagnostics are `S1400` to `S1499`. `docs/system-integration-v1.md`

### 10.2 Contract tests and evidence

[N10.2.1] `invar component generate MANIFEST --output NEW_DIRECTORY` writes a deterministic C header and binding. `tests/issue_s1_02_foreign_manifests.rs::issue_s1_02_foreign_manifests_header_is_deterministic_and_c11_layout_checked`
[N10.2.2] `invar component test MANIFEST` runs the generated contract-test harness and records evidence bound to a hash of the declared transitive inputs. `tests/issue_s1_02_foreign_manifests.rs::issue_s1_02_foreign_manifests_hash_tracks_declared_transitive_inputs`
[N10.2.3] The harness checks layout round trip, lifecycle order and golden vectors, and process crashes and malformed output are typed failures. `tests/issue_s1_02_foreign_manifests.rs::issue_s1_02_foreign_manifests_env_process_crash_is_typed`
[N10.2.4] The checker proves that the binding matches the manifest and that recorded tests passed, and it does not measure the library beyond its goldens. `docs/system-integration-v1.md`

### 10.3 Cell imports

[N10.3.1] `import component "relative/path/manifest.inv" [for robot menagerie::name];` inserts one checked manifest from beneath the source directory into the IR. `tests/issue_s1_06_integration_start.rs`
[N10.3.2] The import needs a current passing contract-test record, and a missing or changed record is `S1405` before generation. `tests/issue_s1_06_integration_start.rs`
[N10.3.3] A path that escapes the cell directory is `S1400`, and a project root is the nearest ancestor holding `.git`. `src/system.rs`
[N10.3.4] An ordinary port may use `std::robotics::JointPositions<f64,robot_dof>`, resolved from the selected robot to a seven or six joint type. `spec/bootstrap.md`

## 11. Execution engines

### 11.1 CLI and exit codes

[N11.1.1] The commands are `check`, `run`, `explain`, `build`, `system`, `component`, `import` and the device commands `source-device` and `gnet-device`, and `build` targets `native` or `metal`. `src/lib.rs::help`
[N11.1.2] The exit codes are 0 ok, 1 source error, 2 tool error, 3 trap and 4 evaluation limit, and a completed `main` is 0 whatever it returns. `spec/bootstrap.md`
[N11.1.3] With `--format json` exactly one JSON document is printed on standard output with no logs, and a document over 1,000,000 bytes is the tool error `DIAGNOSTIC_DOCUMENT_LIMIT`. `src/main.rs::print_run_json`
[N11.1.4] The engines are `reference`, `native-serial`, `cpu-parallel --workers 1..64`, `cuda`, `metal` and `auto`, and an unavailable engine is a typed error and never a silent substitute. `src/lib.rs::help`
[N11.1.5] `invar build PATH --target metal` writes a Metal library ahead of time, and `--artifact METAL_LIBRARY` makes `run`, `explain` and the device commands load it instead of compiling Metal source at run time. `src/lib.rs::help`

### 11.2 Reference, MIR and native serial

[N11.2.1] The reference evaluator interprets checked source and is the oracle, and the typed MIR evaluator is an independent block interpreter that must agree on value, trap, steps and cleanup. `docs/mir-v1.md`
[N11.2.2] MIR is verified at lowering and again at evaluation, and it is an internal schema and not a serialized format. `docs/mir-v1.md`
[N11.2.3] The `native-serial` engine needs the `native` Cargo feature and LLVM 21, and otherwise reports `NATIVE_UNAVAILABLE`. `docs/native-toolchain-v1.md`
[N11.2.4] Native arithmetic is checked against the exact i32 and u32 range, wrapping builtins truncate, and the LLVM IR carries no `nsw`, `nuw`, `noalias`, `inbounds` or `mustprogress`. `docs/native-toolchain-v1.md`
[N11.2.5] Native and reference runs may use different fuel and are compared only for completed runs. `spec/bootstrap.md`

### 11.3 CPU parallel and admitted regions

[N11.3.1] The planner admits a region only when the obligations hold, an unknown obligation rejects it, and a rejected region runs the serial MIR plan. `docs/explain.md`
[N11.3.2] The admitted CPU shapes are a zero-based counted wrapping `u32` map, a lawful modular reduction and a prefix scan of an affine wrapping recurrence. `tests/issue_m2_05.rs::issue_m2_05_cpu_scheduler_fallback_all_families_shapes_workers_and_cleanup`
[N11.3.3] A checked `i32` sum is not reassociated, because regrouping can change which addition traps. `tests/issue_m2_03.rs::issue_m2_03_lawful_reduction_scan_rejects_unproved_source_laws`
[N11.3.4] A false runtime guard selects the serial plan before any write, and `cpu-parallel` results equal the reference for every worker count from 1 to 64. `tests/issue_m2_05.rs::issue_m2_05_cpu_scheduler_fallback_false_guard_is_serial_before_workers`
[N11.3.5] The worker count is bounded by `MAX_WORKERS` (64) and by the available hardware. `src/schedule/workers.rs::MAX_WORKERS`

```invar outcome=value:7
fn main() -> i32 {
    let xs: Array<u32> = [4294967295u32, 1u32, 7u32];
    var acc: u32 = 0u32;
    for i in 0u32 .. len(xs) {
        acc = wrapping_add(acc, xs[i]);
    }
    if acc == 7u32 { return 7; }
    return 0;
}
```

### 11.4 Source families

[N11.4.1] The library recognizes ordinary source for transpose, five-point stencil, segmented reduction and map, stable compaction, gather, histogram, ordered affine pair reduction, bounded summary, matrix contraction and bounded frontier families, each through a `from_checked` or `from_source` contract that rejects near matches. `tests/issue_m4_02_stencil_source.rs`
[N11.4.2] Ordinary `invar run` offers every reached loop to a family registry in precedence order `core`, `tensor`, `grid`, `segmented` and `indexed`, the first match at a span wins, and a loop no family takes runs as serial MIR. `src/schedule/families.rs::families`
[N11.4.3] A recognizer reparses exact source bytes and compares annotations, and a stale or forged checked program is refused. `tests/issue_m4_05_gather_preflight.rs`
[N11.4.4] `invar source-device PATH tensor|transpose ENTRY cpu-model|cuda|metal FUEL` runs the matrix or transpose source and labels the CPU model completion. `tests/issue_m4_source_device_cli.rs::source_device_command_runs_real_source_and_labels_model_completion`
[N11.4.5] The ordered affine pair law is `(a,b) then (c,d) = (c*a, c*b+d)` modulo 2^32 with identity `(1,0)`, applied left then right. `docs/pair-reduction-v1.md`

### 11.5 GPU targets

[N11.5.1] Forced `cuda` or `metal` fails with `CUDA_UNAVAILABLE` or `METAL_UNAVAILABLE` for missing hardware, an ineligible region or a false guard, and never returns a CPU result as a device pass. `docs/explain.md`
[N11.5.2] CUDA admits NVRTC 12.4 or newer in the 12.x series and compute capability 5.0 or newer, compiled for the explicit `sm` of device zero. `docs/cuda-target-v1.md`
[N11.5.3] Metal needs Apple7 or newer with MSL 3.1, a kernel needing more than 31 buffer arguments is Metal-ineligible, and the general-net task cascade additionally needs MSL 3.2 and otherwise uses its staged form. `docs/gnet-task-v01.md`
[N11.5.4] The device IR admits `u32` maps (fused chains of 2 to 16 stages), reductions and affine scans through one verifier, and only a verified plan reaches code emission. `tests/issue_m3_04_cuda_reductions_scans.rs::issue_m3_04_cuda_reductions_scans_graph_plans_lower_to_shared_ir`
[N11.5.5] Unknown completion is `CUDA_PENDING` or `METAL_PENDING`, permits no CPU replay, and keeps loans and quarantine. `tests/issue_m3_05_guarded_gpu_integration.rs::issue_m3_05_gpu_failure_prefix_and_quarantine`
[N11.5.6] `auto` chooses by a once-per-process measured cost model with a 10 percent device win margin and may fall back to the CPU only before submission or after confirmed quiescence. `docs/explain.md`
[N11.5.7] Hardware tests are ignored in the default suite, and device results are pinned to their recorded hosts. `docs/validation.md`
[N11.5.8] Map DAGs, wrapping reductions and affine scans admit up to 2^28 `u32` elements on the device. `docs/cuda-target-v1.md`
[N11.5.9] A device schedule is selected from at most 32 enumerated bounded candidates, each passing the physical device verifier, and the existing measured cost model breaks equal costs by enumeration order. `docs/bounded-schedule-dispatch.md`

### 11.6 General interaction-net compilation

[N11.6.1] The interaction net of a checked program is the oracle, and P1 partitions it into regions with law certificates that every lowering phase re-checks before use. `docs/net-normal-form-v01.md`
[N11.6.2] A recursion with a checked L9 or L10 certificate lowers to one program-specific path-sum kernel, and a residual fork-join lowers to the task region `gnet-task-v1`. `docs/path-sum-v01.md`
[N11.6.3] A task region admits only pure functions over `i32`, `u32` and `bool` scalars whose only intrinsics are `wrapping_add` and `wrapping_mul`, and each refusal has a stable reason code. `docs/gnet-task-v01.md`
[N11.6.4] `invar gnet-device PATH ENTRY cpu-model-cuda|cpu-model-metal|cuda|metal ARGS_JSON CONFIG_JSON` runs the general-net compiled route for one entry. `src/lib.rs::help`

## 12. Limits

[N12.1.1] The reference limits are call depth 64, array length 2^24 elements, 2^28 live elements and default fuel 1,000,000, and they are tool limits and not source semantics. `src/array_limits.rs::MAX_LIVE_ELEMENTS`
[N12.1.2] The checker limits are 32 modules, 256 imports, 128 records of 1 to 16 fields, 128 variants of 1 to 16 arms, 128 frames and 128 lowered functions. `src/check/modules.rs::MAX_RECORDS`
[N12.1.3] The MIR limits are 128 functions, 4,096 blocks and 8,192 locals per function, and 100,000 instructions. `src/mir.rs::MAX_INSTRUCTIONS`
[N12.1.4] Limit exhaustion is a typed tool or resource outcome and never a source trap or a proof of divergence. `docs/native-toolchain-v1.md`

## 13. Diagnostics and explain schemas

### 13.1 Diagnostics

[N13.1.1] Diagnostics use the schema `invar-diagnostics-v3` with `schema`, `schema_version`, `status` and `diagnostics`, and a document carries at most 64 diagnostics. `schema/invar-diagnostics-v3.schema.json`
[N13.1.2] A diagnostic has a stable `code`, a `severity`, a message, a primary span with byte offsets and one-based line and column, secondary spans, notes and suggested fixes. `tests/issue_ai_01_machine_diagnostics/contracts.rs::issue_ai_01_machine_diagnostics_positions_and_fix`
[N13.1.3] Codes, span semantics and fix edits are the contract and messages may change wording. `schema/diagnostic-codes-v1.json`
[N13.1.4] The status is one of `ok`, `source_error`, `tool_error`, `trap` and `evaluation_limit`, with exit codes 0, 1, 2, 3 and 4. `schema/invar-diagnostics-v3.schema.json`

### 13.2 Registry

[N13.2.1] Every emitted code is in a committed registry, `schema/diagnostic-codes-v1.json` plus the extension files, and a test scans the source in both directions. `tests/issue_ai_01_machine_diagnostics/registry.rs::issue_ai_01_machine_diagnostics_registry_covers_emitted_literals`
[N13.2.2] Each code declares `reachability` of `source`, `invariant` or `external`, and checked source never emits an `invariant` code. `tests/issue_ai_01_machine_diagnostics/documents.rs::issue_ai_01_machine_diagnostics_corpus_has_no_invariant_codes`
[N13.2.3] Only `source` codes carry fixes. `tests/issue_ai_01_machine_diagnostics/registry.rs::issue_ai_01_machine_diagnostics_registry_covers_emitted_literals`
[N13.2.4] The `S1xxx` ranges are 1000 system dialect, 1100 to 1399 graph checks, 1400 foreign manifests, 1500 ROS 2, 1600 robot and cell, 1700 vocabulary, 1800 acceptance and 1900 reserved. `schema/diagnostic-codes-v1.json`

### 13.3 Explain

[N13.3.1] `invar explain PATH --format json` emits `invar-explain-v5` with `source_identity` and `regions`, and success is exit 0 even when every region is rejected. `docs/explain.md`
[N13.3.2] Each region has `function`, `span`, `status`, `first_rejected_obligation`, `rule_revision`, `numeric_policy`, `chosen_plan`, `fallback_plan`, `runtime_guard`, `facts`, `execution` and ordered `obligations`. `schema/explain-v5.schema.json`
[N13.3.3] Obligation evidence is `checked_invariant`, `trusted_rule`, `runtime_guard` or `unknown`, and unknown evidence rejects admission. `docs/explain.md`
[N13.3.4] `explain --engine cuda|metal|auto` adds `device_execution` described by `invar-explain-device-v5`. `schema/invar-explain-device-v5.schema.json`
[N13.3.5] Persisted admission documents are descriptive, and validation re-analyses current MIR and source bytes. `docs/explain.md`
[N13.3.6] `source_identity` is `fnv1a64:` over the source bytes and establishes identity and not authenticity. `docs/explain.md`

```invar outcome=value:0
fn main() -> i32 { return 0; }
```

## 14. Python binding surface

[N14.1.1] The extension module `invar_python` exposes `compile_source`, `validate_array`, `validate_buffers`, `capabilities`, the class `CheckedModule` and exceptions rooted at `InvarError`. `crates/invar-python/src/lib.rs::invar_python`
[N14.1.2] `compile_source(source, path, engine, exports)` checks a complete source, and `native-serial` needs explicit exports and a native-enabled core build. `crates/invar-python/src/lib.rs::compile_source`
[N14.1.3] Selected exports have at most eight scalar parameters of `i32` or `u32`, and calls validate names, types, arity and fuel before any source work. `docs/scalar-exports-v1.md`
[N14.1.4] `capabilities()` reports what this build supports, and it is the authority for the binding surface. `crates/invar-python/src/lib.rs::capabilities`
[N14.1.5] The reference array entry copies, the GIL is not released, and external array owners, scheduled host arrays and `take` buffers are unavailable. `crates/invar-python/src/lib.rs::external_array_owners`
[N14.1.6] Exports accept `read` and `mut` `Array<i32>`, `Array<u32>` and `Array<f32>` parameters, which `native-serial` passes zero-copy at the NumPy address with rollback on failure and the reference path copies. `crates/invar-python/src/lib.rs::array_export_paths`
[N14.1.8] `CheckedModule.run_main` runs `main` on any engine of `reference`, `native-serial`, `cpu-parallel`, `auto`, `cuda` and `metal`, and `receipt()` returns the receipt of the last successful `run_main`. `crates/invar-python/src/module.rs::run_main`
[N14.1.7] The private wheel qualification is Linux x86_64 with glibc 2.41 and CPython 3.12 only. `crates/invar-python/README.md`

## 15. Example library and coverage

[N15.1.1] Every Invar source file under `examples/` is listed exactly once in `examples/index.json` with its path, intent, family, tags, spec section and expected result. `tests/issue_ai_02_spec_examples.rs::validate_index`
[N15.1.2] Each family is either covered by at least one example or listed under `pending_families` with a reason. `tests/issue_ai_02_spec_examples.rs::validate_index`
[N15.1.3] Each runnable example runs on the reference evaluator and on every engine it declares, check and system examples run their declared check commands, and `native-serial` runs only in a build with the `native` feature. `tests/issue_ai_02_spec_examples.rs::run_example`
[N15.1.4] System examples carry their expected diagnostics and their IR or generated-file hashes, checked twice for byte identity. `tests/issue_ai_02_spec_examples.rs::tree_hash`

## Appendix A. Open v0.1 gaps (not current behaviour)

Note: the open content in this appendix is not current behaviour, and no statement above relies on it.
Note: AI-02 depends on L1-06, S1-06 and M5-03, none of which is closed, so the spec must be re-verified after each of them closes.
Note: the open set is every open `release:v0.1` issue plus the GNET slices #47, #45 and #51, and each GitHub issue records its evidence, remaining gate and blocker.

| Issue | Open content | Spec sections that change when it closes |
| --- | --- | --- |
| #1 M4-01 | f32 family gates through the M5-01 harness on the named targets | 5.8, 11.4 |
| #2 M4-02 | Transpose and stencil family gates | 11.4 |
| #3 M4-03 | Matrix family gate, generated and library-assisted tracks | 11.4 |
| #4 M4-04 | Segmented family gate, including the CUDA shared-budget gate | 11.4 |
| #5 M4-05 | Gather/scatter and frontier family gates | 11.4 |
| #6 M4-07 | Summary family gate | 11.4 |
| #8 M4-06 | Schedule tuning on the tuning set after the held-out freeze | 11.5 |
| #7 M5-01 | Permanent held-out freeze (#52), baseline timing, calibration in `invar-placement-v1` (#49) | 11.5, B.5 |
| #9 M5-02 | Device qualification of the full pipeline | 11.5 |
| #10 M5-03 | CUDA-class performance acceptance and report | 11.5 |
| #14 L1-06 | Python binding platform matrix beyond the private wheel, starting with macOS arm64 | 14 |
| S1-04 | Runtime smoke on the S1-06 cell with `mujoco_ros2_control` | 8.3 |
| #29 S1-06 | Recorded two-arm reach-and-place run | 8.3, 10.3 |
| #47 GNET-04 | Calibrated automatic dispatch for general nets after the M5-01 freeze, plus specialization and residency (D1(a) and D2(a) have landed) | 11.6, 13.3, B.5 |
| #45 GNET-05 | Six Metal census observations and a deep-device count | 11.6 |
| #51 GNET-05 | One completed nightly with the CUDA census compile wall time inside the 3 h bound; its compile-deadline refusals are the v0.1 outcome | 11.6 |
| #54 GNET P6/P7 | One normal-form dispatcher with the families as its lowerings, GNET-05 device parity (correctness on the H100 and the Apple M4, timing on the RTX 5070 Ti) and calibrated placement (P6(b)) | 11.3, 11.4, 11.6, 13.3 |

Note: `cpu-parallel`, and `auto` in builds without a device feature, run a program that no existing parallel path claims through the checked normal form, so the serial MIR fallback of N11.4.2 applies only where that route refuses; the sections listed for #54 are re-verified when it closes.
Note: hardware results in this specification stay pinned to their recorded hosts.

## Appendix B. Beyond v0.1: v0.2 features on main

Note: everything in this appendix is on `main` but is not part of the normative v0.1 sections above, and a v0.1 conforming implementation need not provide it.
Note: the statements here are checked and their blocks executed exactly like the rest of this file.

### B.1 Deterministic float reductions

[NB.1.1] `sum_tree(xs)`, `dot_tree(a, b)`, `sum_exact(xs)` and `dot_exact(a, b)` read-borrow bare named `Array<f32>` operands for the call and return `f32`. `spec/bootstrap.md`
[NB.1.2] `sum_tree` and `dot_tree` use one fixed tree of 8-element leaves that depends only on the length, so every engine and worker count returns the same bits, which generally differ from a sequential fold. `spec/bootstrap.md`
[NB.1.3] `sum_exact` and `dot_exact` return the exact real sum rounded once to nearest even, and every NaN result is the quiet NaN `0x7fc00000`. `spec/bootstrap.md`
[NB.1.4] Dot operands of different lengths trap with `INDEX_OUT_OF_BOUNDS` before any arithmetic, and `--engine auto` places each call by the measured crossover in `calibration/fsum-v1.json`. `calibration/fsum-v1.json`
[NB.1.5] A source `for` loop that folds floats is never reassociated, and `invar explain` only hints at these intrinsics with the obligation `FLOAT_FOLD_HINT`. `spec/bootstrap.md`

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

[NB.2.1] `sort(xs)`, `sort_by_key(keys, values)`, `argsort(xs)` and `segmented_sort(xs, offsets)` are stable, sort keys of `i32`, `u32` or `f32`, and `argsort` returns an owned `Array<u32>` of original indices. `spec/bootstrap.md`
[NB.2.2] Float keys use IEEE totalOrder after each NaN becomes `0x7fc00000`, signed zeros stay distinct, and payloads, which may also be `Array<Bool>`, keep their bits. `spec/bootstrap.md`
[NB.2.3] A key and value length mismatch or invalid segment offsets trap with `INDEX_OUT_OF_BOUNDS` at the call span before any write. `spec/bootstrap.md`
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
[NB.3.2] `radius_neighbors(grid, qx, qy, qz, radius, max_k)` returns `Neighbors { counts, idx }` with true counts that are never clipped to the stride, using strict f32 `d2 <= radius * radius`. `spec/bootstrap.md`
[NB.3.3] `knn(grid, qx, qy, qz, k)` returns `Knn { idx, d2 }` ordered by squared distance and then original index, and pads missing slots with `u32::MAX` and positive infinity. `spec/bootstrap.md`
[NB.3.4] A nonpositive or non-finite cell or an out-of-range cell index traps with `INTEGER_OVERFLOW`, and coordinate arrays of different lengths trap with `INDEX_OUT_OF_BOUNDS`. `spec/bootstrap.md`
[NB.3.5] A `VoxelGrid` parameter is `read` or `take` and never `mut`, and query result arrays leave their record only through `move result.field`. `spec/bootstrap.md`
[NB.3.6] A report with a spatial site names the calibration family `spatial.<op>.f32` and the reason `spatial-cost-unmeasured`. `docs/explain.md`

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
[NB.4.3] Conversions are explicit intrinsics such as `f64_from_f32`, `f32_from_f64` and `i32_from_f64`, and a float to integer conversion of NaN or an out-of-range value traps with `INTEGER_OVERFLOW`. `src/eval/floating.rs::i32_from_f64`
[NB.4.4] `invar-math-cr-v1` provides `sqrt`, `cbrt`, `hypot`, the trigonometric, hyperbolic, exponential and logarithmic functions, `pow`, `floor`, `ceil`, `trunc`, `round`, `abs`, `copysign`, `min` and `max`, overloaded on `f32` and `f64` and correctly rounded. `src/math.rs`
[NB.4.5] Every NaN that leaves a library function or a float to float conversion is the canonical quiet NaN, and `min` and `max` ignore a single NaN operand. `src/math.rs`
[NB.4.6] Metal has no binary64, so a program that uses `f64` runs on the CPU with the reason `metal-no-f64`. `src/engine.rs::metal-no-f64`
[NB.4.7] An `extern c fn` of N7.3.1 also accepts `f64` as a value argument, the result `T` stays `i32`, `u32`, `Bytes` or `String`, and an `f64` result or any `f32` argument or result is `TYPE_MISMATCH`. `tests/issue_l1_04_bytes_strings_cabi.rs::v02_f64_foreign_scalar_argument_matches_its_oracle` `src/check.rs::valid_result`
[NB.4.8] An `f64` argument has tag `INV_TAG_F64_V1` (5) and mode `INV_MODE_VALUE_V1`, and its `scalar` holds the binary64 encoding, which C reads with `memcpy` into a `double`. `include/invar_foreign_v1.h` `tests/fixtures/l1_04/foreign.c`
Note: the binary64 CUDA kernels whose CORE-MATH sources declare an anonymous struct inside a union (`cos`, `tan`, `atan2`, `log`, `log10` and `pow`) need NVRTC 12.9 or newer, every other CUDA unit keeps the 12.4 floor of N11.5.2, and a rewrite for NVRTC 12.4 and 12.6 is deferred to the Jetson Orin target (#21).

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

### B.5 Placement profiles

[NB.5.1] `invar calibrate [--out NEW_PATH] [--family placement.wrapping_add1.u32] [--host-class CLASS]` measures the single family `placement.wrapping_add1.u32` and publishes a new provisional `invar-placement-v1` profile, by default at `calibration/<os>-<arch>.json`, and never replaces an existing path. `tests/v02_placement.rs::calibrate_cli_exports_a_valid_provisional_profile` `tests/v02_placement.rs::issue_v02_placement_bounded_loader_atomic_publish_and_no_clobber`
[NB.5.2] `run` and `explain` with `--engine auto` accept `--profile PROFILE`, `--host-class CLASS` and `--trust-provisional`, and a provisional profile influences placement only with `--trust-provisional`. `src/lib.rs::--trust-provisional` `spec/placement-v1.md`
[NB.5.3] Placement owns `--engine auto` only when one of those flags is given or the live host's default profile is not a matching provisional one, and every other auto run keeps the established path of N11.5.6, whose documents take the versions of NB.5.6. `src/auto_command.rs::placement_requested` `tests/v02_placement.rs::auto_stays_on_the_established_path_unless_placement_is_asked_for`
[NB.5.4] A profile whose identity differs from the live host, CPU, device, driver, compiler build or LLVM version is refused with `TUNING_HINT_INVALID`, a frozen profile is refused with `TUNING_FREEZE_REQUIRED` until M5-01 provides a trusted freeze manifest, and both keep the CPU. `spec/placement-v1.md` `tests/v02_placement.rs::real_cli_profile_trust_identity_contract_and_cpu_decision`
[NB.5.5] A device is chosen only when its total cost is below 0.95 of the CPU cost, an unknown cost keeps the CPU, and only the measured family can leave the CPU. `src/placement.rs::select` `spec/placement-v1.md`
[NB.5.6] A placement run receipt carries a `placement` object, and `explain --engine auto` under placement adds `placement` and `execution`, both recording `auto_path` as `placement`; an established auto run has no `placement` and records `auto_path` as `established`. `docs/explain.md` `schema/explain-v5.schema.json`
