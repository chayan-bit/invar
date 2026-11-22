# Invar v0.1 language specification

Status: DRAFT; open v0.1 work is tracked in GitHub issues, and v0.2 features already on `main` appear only in Appendix B.
Counts: statements=319 citations=351
Size record: tokens=30283 estimator=bytes/3 limit=60000
## 1. Conventions and scope

### 1.1 Normative statements, notes and citations

Note: this file is machine checked by `tests/spec_examples.rs`; the checks are described in 1.2.
[N1.1.1] A line starting with `[N` and an identifier is a normative statement and is a requirement on conforming implementations of Invar v0.1. `tests/spec_examples.rs::check_spec_text`
[N1.1.2] Every normative statement cites at least one repository file, and a `path::name` citation names text that occurs in that file. `tests/spec_examples.rs::spec_examples_normative_statements_cite_existing_sources`
[N1.1.3] A line starting with `Note:` is informative and carries no requirement. `tests/spec_examples.rs`
[N1.1.4] Where this specification and the implementation disagree, the test fails and the disagreement is fixed or escalated, never papered over. `tests/spec_examples.rs`
[N1.1.5] Planned work and open gaps are tracked in GitHub issues and are not normative text here, and v0.2 features that are on `main` appear only in Appendix B. `CONTRIBUTING.md`

### 1.2 Executable blocks and outcome tags

[N1.2.1] Every fenced block carries an `outcome=` tag and an untagged block fails the test suite. `tests/spec_examples.rs::spec_examples_rejects_an_untagged_spec_block`
[N1.2.2] Tag `value:N` means `invar run --engine reference` completes and `main` returns the i32 N. `tests/spec_examples.rs::run_block`
[N1.2.3] Tag `trap:CATEGORY` means the same command reports a source trap of that category. `tests/spec_examples.rs::run_block`
[N1.2.4] Tag `diag:CODE` means `invar check` rejects the source and the first diagnostic has that code. `tests/spec_examples.rs::run_block`
[N1.2.5] Tag `limit` with `fuel=N` means the run ends with `evaluation_limit`, and tag `check` means `invar check` accepts the file. `tests/spec_examples.rs::run_block`
[N1.2.6] Tag `system:ok` or `system:CODE` runs `invar system check` and requires success or that first diagnostic code. `tests/spec_examples.rs::run_block`
[N1.2.7] The flag `foreign` supplies an unused `--c-library` path, because the reference engine runs the Invar oracle and never loads it. `tests/spec_examples.rs::run_block`
[N1.2.8] A block that is not Invar source carries `outcome=none`, and an Invar block may not. `tests/spec_examples.rs::spec_examples_rejects_a_block_whose_outcome_differs`
[N1.2.9] A block whose real outcome differs from its tag fails the suite. `tests/spec_examples.rs::spec_examples_rejects_a_block_whose_outcome_differs`

### 1.3 Precedence and companion documents

[N1.3.1] The grammar is `spec/grammar.ebnf`, and this file is the normative semantic text of the language and of the later contracts. `spec/grammar.ebnf`
[N1.3.2] Examples are indexed in `examples/index.json`, validated by `examples/index.schema.json`, and run by the same test binary. `tests/spec_examples.rs::spec_examples_index_validates_and_every_example_matches_its_expected_output`
[N1.3.3] No safety theorem, native performance result, parallel speedup or AI-authoring advantage is claimed by this specification. `README.md`

## 2. Source text

### 2.1 Lexical rules

[N2.1.1] A source file is one valid UTF-8 text of at most 1,048,576 bytes with syntactic nesting at most 32. `src/front.rs::MAX_SOURCE_BYTES`
[N2.1.2] Invalid UTF-8 is the tool error `INVALID_UTF8`, and lexically invalid UTF-8 text is a source error. `tests/language/syntax_and_limits.rs::syntax_and_limits_tool_limits_and_encoding`
[N2.1.3] Source paths must end in the extension .inv. `src/main.rs::read_source`
[N2.1.4] Comments are `//` to end of line or nested `/* */`, and an unterminated comment is a lexical error. `tests/language/syntax_and_limits.rs::syntax_and_limits_lexical_spans_and_json`
[N2.1.5] Spans are zero-based half-open UTF-8 byte intervals, and CRLF does not change byte offsets. `tests/language/syntax_and_limits.rs::syntax_and_limits_lexical_spans_and_json`
[N2.1.6] Identifiers are `[A-Za-z_][A-Za-z0-9_]*` and keywords are reserved. `src/front/lexer.rs`

```invar outcome=value:3
/* a /* nested */ comment */
fn main() -> i32 { return 1 + 2; } // trailing
```

### 2.2 Literals and reserved words

[N2.2.1] A decimal integer literal has an optional `i32` or `u32` suffix, and an unsuffixed literal takes its type from context and otherwise defaults to i32. `src/check.rs`
[N2.2.2] An identifier character directly after a numeric token is a malformed literal and is not split into two tokens. `tests/language/syntax_and_limits.rs::syntax_and_limits_lexical_spans_and_json`
[N2.2.3] The spelling `-2147483648` is a valid i32, the positive `2147483648i32` is rejected, and an out-of-range literal is rejected at compile time even in unreachable code. `tests/language/names_types_and_calls.rs::names_types_and_calls_rejects_names_types_and_returns`
[N2.2.4] The words `async await scope total effects requires ensures` are reserved and rejected as `UNSUPPORTED_FEATURE`. `src/front/lexer.rs::unsupported`

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

[N3.1.1] The scalar types are `Bool`, `i32`, `u32` and, as a source increment, `f32`; `Unit` is legal only as a function return type. `tests/language/names_types_and_calls.rs::names_types_and_calls_rejects_names_types_and_returns`
[N3.1.2] Scalars are Copy; an `Array<T>` is a non-Copy owner of a fixed-length, fully initialized sequence. `src/eval/value.rs`
[N3.1.3] Lengths and indices are `u32`, and there is no numeric promotion or implicit conversion between integer types. `tests/language/names_types_and_calls.rs::names_types_and_calls_rejects_names_types_and_returns`
[N3.1.4] Array literals, empty or not, need an expected `Array<T>` from an annotation, parameter, return type or assignment destination. `tests/language/names_types_and_calls.rs::names_types_and_calls_rejects_names_types_and_returns`
[N3.1.5] `main` is exactly `fn main() -> i32` for `run`; `check` accepts library files without `main`. `tests/language/names_types_and_calls.rs::names_types_and_calls_cli_check_json_and_status`
[N3.1.6] A `main` with the wrong signature is the source error `ENTRYPOINT_SIGNATURE` and a missing `main` under `run` is `ENTRYPOINT_MISSING`. `tests/language/cli_examples_and_exit_codes.rs::cli_examples_and_exit_codes_real_cli_examples_and_statuses`

```invar outcome=diag:TYPE_MISMATCH
fn main() -> i32 {
    let a: i32 = 1;
    let b: u32 = 2u32;
    return a + b;
}
```

### 3.2 Bindings and mutability

[N3.2.1] `let` prevents rebinding and element mutation, and `var` permits both when ownership allows. `tests/language/names_types_and_calls.rs::names_types_and_calls_rejects_names_types_and_returns`
[N3.2.2] Local and parameter names are block-scoped, unique within a function, and may not hide a function or an intrinsic. `tests/language/names_types_and_calls.rs::names_types_and_calls_rejects_names_types_and_returns`
[N3.2.3] Every non-Unit function must return on all conservatively analysed paths, and a loop never establishes definite return. `tests/language/names_types_and_calls.rs::names_types_and_calls_rejects_names_types_and_returns`
[N3.2.4] A `Unit` function may fall through, an `if` ends its path only when it has an `else` and both arms end it, and every function and statement is checked, including uncalled functions and unreachable code. `tests/language/names_types_and_calls.rs::names_types_and_calls_rejects_names_types_and_returns`
[N3.2.5] Functions resolve across the whole file, duplicate functions, wrong arity, unknown names and redefined intrinsics are errors, and functions are not values. `tests/language/names_types_and_calls.rs::names_types_and_calls_rejects_names_types_and_returns`
[N3.2.6] A `mut` parameter grants element mutation and a `read` parameter never does, parameters cannot be rebound, and a `take` parameter is immutable unless moved into a local `var`. `src/ownership.rs`

```invar outcome=diag:RETURN_MISSING
fn spin(n: i32) -> i32 {
    while n < 0 {
        return n;
    }
}

fn main() -> i32 { return spin(1); }
```

### 3.3 Evaluation order and control flow

[N3.3.1] Expressions and call arguments evaluate left to right, and `and` and `or` short-circuit. `tests/language/arithmetic_fuel_and_demand.rs::arithmetic_fuel_and_demand_scalar_order_demand_and_calls`
[N3.3.2] Only the selected `if` arm runs, and conditions and the operands of `not`, `and` and `or` must be `Bool` with no truthiness conversion. `tests/language/names_types_and_calls.rs::names_types_and_calls_rejects_names_types_and_returns`
[N3.3.3] `for i in lo .. hi` evaluates both `u32` bounds once, iterates the half-open range, binds an immutable induction variable and increments it only after a completed body, so reaching `hi` never overflows. `src/eval/engine.rs`
[N3.3.4] An assignment evaluates its right side first, then the destination index, then bounds, then stores. `src/eval/engine.rs`
[N3.3.5] Direct recursion and forward calls are allowed, and recursion is not assumed total. `tests/language/names_types_and_calls.rs::names_types_and_calls_forward_calls_recursion_and_typed_order`
[N3.3.6] The destination index is evaluated once and may read but not mutably borrow or consume the destination owner, and a right side that consumes the destination owner is rejected. `src/ownership.rs`
[N3.3.7] Whole-owner assignment needs an available unborrowed local `var` and a fresh or moved array, a moved binding is never reinitialized, and `return;` is allowed only in a `Unit` function. `src/ownership.rs`

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

[N3.4.1] `+ - *` on `i32` and `u32` and unary `-` on `i32` check representability and trap with `INTEGER_OVERFLOW`, and unary `-` on `u32` is rejected. `tests/language/arithmetic_fuel_and_demand.rs::arithmetic_fuel_and_demand_arithmetic_boundaries`
[N3.4.2] Division and remainder by zero trap with `DIVISION_BY_ZERO`, and `i32::MIN / -1` traps with `INTEGER_OVERFLOW`. `tests/language/arithmetic_fuel_and_demand.rs::arithmetic_fuel_and_demand_arithmetic_boundaries`
[N3.4.3] Signed division truncates toward zero, the remainder has the dividend's sign, and `MIN % -1` is `0`. `tests/language/arithmetic_fuel_and_demand.rs::arithmetic_fuel_and_demand_arithmetic_boundaries`
[N3.4.4] `wrapping_add` and `wrapping_mul` take two `u32` operands and compute modulo 2^32. `tests/language/arithmetic_fuel_and_demand.rs::arithmetic_fuel_and_demand_arithmetic_boundaries`
[N3.4.5] Arithmetic behaves identically in debug and release builds. `docs/rust-engineering.md`
[N3.4.6] The source trap categories are `INTEGER_OVERFLOW`, `DIVISION_BY_ZERO`, `INDEX_OUT_OF_BOUNDS` and the unit trap `UNIT_CONVERSION_INEXACT` (6.1.5), and a trap records the operator or index span of the first failing operation in evaluation order. `src/eval/error.rs::UNIT_CONVERSION_INEXACT`
[N3.4.7] A trap performs no language-level cleanup, and it is distinct from an evaluation limit and from a tool error. `docs/validation.md`
[N3.4.8] `u32` division floors, its remainder lies in `[0, divisor)`, and `u32` subtraction below zero traps with `INTEGER_OVERFLOW`. `tests/language/arithmetic_fuel_and_demand.rs::arithmetic_fuel_and_demand_arithmetic_boundaries`

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

[N3.5.1] `len(xs)` returns the `u32` length, and `filled_i32(n, x)` and `filled_u32(n, x)` evaluate `n` then `x` once and build owned arrays of `n` copies. `tests/language/arrays_ownership_and_cleanup.rs::arrays_ownership_and_cleanup_array_values_indexing_and_filling`
[N3.5.2] Indexing needs a `u32` index and an out-of-range index traps with `INDEX_OUT_OF_BOUNDS`. `tests/language/arrays_ownership_and_cleanup.rs::arrays_ownership_and_cleanup_array_values_indexing_and_filling`
[N3.5.3] Array equality and array arithmetic are unsupported. `tests/language/names_types_and_calls.rs::names_types_and_calls_rejects_names_types_and_returns`
[N3.5.4] Allocation failure is a tool error and not a program result, reads copy scalar elements, and stores need exclusive access. `tests/language/arrays_ownership_and_cleanup.rs::arrays_ownership_and_cleanup_bounded_array_allocation`

```invar outcome=trap:INDEX_OUT_OF_BOUNDS
fn main() -> i32 {
    let xs: Array<i32> = [1, 2, 3];
    return xs[3u32];
}
```

### 3.6 Fuel

[N3.6.1] Each expression evaluation and statement dispatch consumes one unit of fuel, and the default limit is 1,000,000. `src/eval/engine.rs::charge`
[N3.6.2] Exhausting fuel reports `evaluation_limit` with `steps_used` equal to the limit and is never a proof of divergence. `tests/language/arithmetic_fuel_and_demand.rs::arithmetic_fuel_and_demand_fuel_and_call_depth`
[N3.6.3] A literal, name or intrinsic is an expression step, a call adds no step for entering its function, and a block or parentheses add no step of their own. `src/eval/engine.rs`
[N3.6.4] A `for` is charged once on entry and once before each iteration, a `while` condition on every evaluation, and resource bookkeeping and static checking consume no fuel. `tests/language/arithmetic_fuel_and_demand.rs::arithmetic_fuel_and_demand_fuel_and_call_depth`
[N3.6.5] With fuel `N` exactly `N` steps run, `--fuel N` takes a positive decimal, and fuel is a tool limit and never an optimization fact. `tests/language/arithmetic_fuel_and_demand.rs::arithmetic_fuel_and_demand_fuel_and_call_depth`

```invar outcome=limit fuel=1000
fn main() -> i32 {
    var n: i32 = 0;
    while true { n = 1; }
    return n;
}
```

