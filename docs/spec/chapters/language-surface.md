### B.7 Language surface

[NB.7.1] An integer literal may separate digits with single `_` characters and take a `0x`, `0o` or `0b` radix prefix, and its token value is the decimal literal it denotes, so the range checks of N2.2.3 and every later stage are unchanged (#253). `src/front/lexer.rs::integer_literal` `tests/language/surface_sugar.rs::literal_spellings_give_identical_mir_outcomes_and_receipts`
[NB.7.2] A leading, trailing or doubled `_`, a digit outside the radix, a radix value above 128 bits, or a separator or prefix before a decimal point is `LEX_INVALID_LITERAL`. `tests/language/surface_sugar.rs::literal_separators_and_radix_prefixes_lex_to_the_same_token_value`

```invar outcome=value:1
fn main() -> i32 {
    let mask: u32 = 0xFFFF_0000u32;
    if mask == 4_294_901_760u32 and 0b1010 == 10 { return 1; }
    return 0;
}
```

```invar outcome=diag:LEX_INVALID_LITERAL
fn main() -> i32 { return 1__0; }
```
[NB.7.3] `p op= e;` with `op` one of `+ - * / %` is exactly `p = p op (e);`, desugared before checking and family recognition, except that an index other than a name or literal is first bound to a hidden immutable `u32` local, so it evaluates once and before `e` (#254). `src/front/parser/statements.rs::compound_assignment` `tests/language/surface_sugar.rs::compound_assignment_evaluates_a_computed_index_once`

```invar outcome=trap:INDEX_OUT_OF_BOUNDS
fn main() -> i32 {
    var ys: Array<u32> = [1u32, 2u32];
    ys[0u32] *= 3u32;
    ys[ys[0u32]] += 1u32;
    return 0;
}
```
[NB.7.4] `assert(c, "m")` takes a `Bool` condition and a string-literal message and returns `Unit`, costs one step plus its condition, never evaluates the message, and on a false condition traps `ASSERTION_FAILED` at the call span, which the CLI reports with the message (#258). `src/eval/engine.rs` `tests/language/assertions.rs::the_cli_reports_the_assertion_message_and_span`
[NB.7.5] The reference, MIR, net oracle, native and cpu-parallel engines agree on that trap and its fuel prefix, a loop holding an assert is not admitted, so its first failing iteration in source order traps, and the compiled net route refuses it as `unsupported_call`. `tests/language/assertions.rs::a_false_assert_traps_identically_on_every_engine` `tests/language/assertions.rs::a_loop_with_an_assert_is_not_admitted_and_runs_serially`

```invar outcome=trap:ASSERTION_FAILED
fn main() -> i32 {
    assert(2 + 2 == 5, "arithmetic");
    return 0;
}
```
[NB.7.6] `for x in xs { b }` over a named array, `View` or `MutView` is `for i in 0u32 .. len(xs) { let x = xs[i]; b }` with a hidden `i`, desugared before checking and family recognition, except that a body assigning `x` names `i` as `x`, reserving it alike, and makes every `x` the place `xs[i]`, so loans, traps and fuel are the counted loop's; such an `x` cannot be moved or captured (`OWNERSHIP`) (#255). `src/front/parser/loops.rs` `tests/language/for_each.rs`
[NB.7.7] `for i in lo .. hi step k` binds `lo`, `hi` and `k` once, in order, to hidden `u32` locals and runs the counted loop over `0 .. ceil((hi - lo) / k)` trips, none when `hi <= lo`, with `i = lo + j * k`, the view stride rule of N5.5.2; a literal zero `k`, in any parentheses, is `PARSE_ZERO_STEP`, and a zero `k` traps `DIVISION_BY_ZERO` over the header only on a non-empty range (#256). `tests/language/stepped_ranges.rs`
[NB.7.8] A `///` line comment documents the item whose first token follows it and a `//!` line comment the file or module it opens before any item; both are trivia kept with that item's span in `Program::docs`, never reach checking, MIR or plans, and anywhere else are `PARSE_DOC_COMMENT_MISPLACED` (#257). `src/front/parser/docs.rs` `tests/language/doc_comments.rs`
[NB.7.9] `break;` leaves the innermost enclosing loop and `continue;` starts its next trip, `name: while` or `name: for` labels a loop for `break name;` and `continue name;`, and each costs one step, as `return;` does (#261). `tests/language/loop_control.rs::an_early_exit_costs_and_cleans_up_like_return`
[NB.7.10] An exit drops the owners of every scope it leaves, innermost first, a loop never counts as returning, and an exit outside a loop, to an unknown label or under a duplicate label, including from a closure body, is `PARSE_LOOP_CONTROL`. `tests/language/loop_control.rs::loop_control_outside_its_loop_is_refused`
[NB.7.11] All engines agree, an innermost `continue` takes the laws of its guarded form, any other exit keeps its loop serial without L15, and family recognizers and tiered folds refuse a loop holding an exit. `tests/language/loop_control.rs` `tests/gnet_device_model/loop_control_model.rs`

```invar outcome=value:4
fn main() -> i32 {
    var n = 0;
    rows: for i in 0u32 .. 4u32 {
        for j in 0u32 .. 4u32 {
            if j > i { continue rows; }
            if i + j == 3u32 { break rows; }
            n += 1;
        }
    }
    return n;
}
```

[NB.7.12] `(A, B)` with `A` and `B` each `i32`, `u32`, `f32` or `Bool` is a Copy pair type with value `(a, b)` and fields `p.0` and `p.1`, and `let (x, y) = e;` binds `e` once to a hidden local, then copies its fields in order; any other arity, element type or field index is `TUPLE_SHAPE`, and pairs are not compared, captured, written by field, or held in records, variants, views or generics (#262). `src/front/parser/tuples.rs` `tests/language/tuples.rs::malformed_pairs_are_refused_with_exact_codes`
[NB.7.13] The reference, MIR, cpu-parallel and native engines agree on pairs, native packing element 0 in the low half of one 64-bit slot; the net oracle, device compilers and C exports refuse them, and a loop reading one has no admitted candidate, so it runs serially. `tests/language/tuples.rs`

```invar outcome=value:8
fn divmod(a: i32, b: i32) -> (i32, i32) { return (a / b, a % b); }
fn main() -> i32 {
    let (q, r) = divmod(23, 4);
    return q + r;
}
```
