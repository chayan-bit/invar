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

[NB.7.14] A match arm may carry a guard, `pattern if g => value`: arms are tried in source order, and the guard of an arm whose case matches runs once every earlier arm has failed, with the payload bound, charging its own steps and trapping at its own operator, and a false guard falls through to the next arm (#260). `src/eval/matching.rs` `tests/language/match_guards.rs::each_evaluated_guard_costs_its_own_steps_and_an_unreached_one_none` `tests/language/match_guards.rs::a_trapping_guard_traps_at_its_operator_with_the_reference_fuel_prefix`
[NB.7.15] A guarded arm never covers its case, so every case still needs an unguarded arm; an arm after its case's unguarded arm is `NAME_DUPLICATE`, a guard must be `Bool`, and an arm with an owned payload takes no guard (`UNSUPPORTED_FEATURE`); a false guard continues at the next arm of its own case, so a move in a guard reaches only its arm and that case's later arms. `src/check/matching.rs` `src/ownership.rs` `tests/language/match_guards.rs::guarded_arms_never_cover_their_case_and_refusals_have_exact_codes` `tests/language/match_guards.rs::a_guard_that_moves_an_owner_moves_it_only_on_its_own_case`
[NB.7.16] The reference, MIR, net oracle, native and cpu-parallel engines agree on a guarded match's value, trap, fuel prefix and cleanup, because a guard lowers to a MIR branch between its arm's binding and its value. `src/mir/builder.rs` `tests/language/match_guards.rs::guards_pick_the_first_passing_arm_identically_on_every_engine`

```invar outcome=value:9
enum Reading { Celsius(i32), Missing }

fn band(r: read Reading) -> i32 {
    return match r {
        Reading::Celsius(t) if t < 0 => 1,
        Reading::Celsius(u) if u < 30 => 2,
        Reading::Celsius(v) => 3,
        Reading::Missing => 0,
    };
}

fn main() -> i32 {
    let cold: Reading = Reading::Celsius(-4);
    let warm: Reading = Reading::Celsius(21);
    let hot: Reading = Reading::Celsius(35);
    return band(cold) + band(warm) * 1 + band(hot) * 2;
}
```

[NB.7.17] `let V::P(x) = s else { B };`, with `s` a named owner or `move` of one, is exactly the hidden-name statements `let ok: Bool = match s { V::P(c) => true, ... => false }; if not ok { B } let x: T = match s { V::P(v) => v, ... => z };`, where the extraction keeps any `move`, `T` is the scalar payload type and `z` its zero literal, which never runs; without a payload binding only the first two remain (#260). `src/front/parser/let_else.rs` `src/check/let_else.rs::complete` `tests/language/match_guards.rs::let_else_is_the_check_the_diverging_block_and_the_extraction`
[NB.7.18] The else block must end in a diverging statement, which is `return`, `break`, `continue`, `assert(false, m)` with a literal `false`, or an `if` with an `else` whose blocks all end in one, and any other block is `PARSE_LET_ELSE_FALLTHROUGH`. `src/front/parser/let_else.rs::diverging` `tests/language/match_guards.rs::let_else_refusals_have_exact_codes_and_spans`
[NB.7.19] A let-else pattern with an owned payload is `UNSUPPORTED_FEATURE`, a `Result` pattern needs an owner whose type a parameter, an annotation, a call's result or a `move` of such an owner declares (else `TYPE_MISMATCH`), any other scrutinee is `PARSE_EXPECTED_TOKEN`, and an unknown arm is reported by the match check. `src/check/let_else.rs` `tests/language/match_guards.rs::let_else_refusals_have_exact_codes_and_spans`

```invar outcome=value:5
fn half(x: u32) -> Result<u32, u32> {
    if x % 2u32 == 0u32 { return Result::Ok(x / 2u32); }
    return Result::Err(x);
}

fn main() -> i32 {
    var kept = 0;
    for i in 0u32 .. 10u32 {
        let r = half(i);
        let Result::Ok(h) = r else { continue; };
        kept += 1;
    }
    return kept;
}
```

```invar outcome=diag:PARSE_LET_ELSE_FALLTHROUGH
enum Pick { A(u32), B(u32) }

fn main() -> i32 {
    let p: Pick = Pick::B(1u32);
    let Pick::A(v) = p else { let w: u32 = 0u32; };
    return 0;
}
```

[NB.7.20] A pair local does not cost its function's other loops their admission: the region tree tracks it as its two scalar components, and a loop that reads or writes one stays serial as NB.7.13 requires; a pair parameter or result, a field of a pair that is not a local, or a store whose second component reads the first still leaves its function without a tree, so all of that function's loops stay serial (#362). `src/general_net/regions/tree/pairs.rs` `src/general_net/normal_form/gate.rs` `tests/schedule_cpu/pair_admission.rs::issue_362_a_pair_outside_the_loop_keeps_its_admission` `tests/schedule_cpu/pair_admission.rs::issue_362_a_loop_touching_a_pair_component_has_no_normal_form_law`

[NB.7.21] `if c { a } else { b }` in expression position, where `else if` nests another, is a scalar expression: `c` is `Bool`, and both arms are single expressions of one type among `i32`, `u32`, `f32`, `f64` and `Bool`, taken from the expected type, else from the first arm whose type is known; another condition or arm type is `TYPE_MISMATCH`, another result type `UNSUPPORTED_FEATURE`, a missing `else` `PARSE_IF_WITHOUT_ELSE`, and an `if` starting a statement stays the statement (#361). `src/front/parser/expressions.rs::if_expr` `src/check/matching.rs::if_expr` `tests/language/if_expressions.rs::malformed_if_expressions_are_refused_with_exact_codes`
[NB.7.22] It costs one step, then its condition, then only the selected arm, so a trap in either is reported at its own operator and the other arm never runs; an owner either arm moves is moved after it, as for match arms. `src/eval/engine.rs` `src/ownership.rs` `tests/language/if_expressions.rs::an_if_expression_costs_one_step_its_condition_and_its_selected_arm` `tests/language/if_expressions.rs::a_trap_is_reported_at_its_operator_and_the_unselected_arm_never_runs` `tests/language/if_expressions.rs::an_owner_moved_in_either_arm_is_moved_after_the_expression`
[NB.7.23] Its MIR is a two-arm match without the tag test, so every engine agrees on its value, trap, fuel prefix and cleanup, and a loop holding one gets the equivalent match's receipt. `src/mir/builder.rs` `tests/language/if_expressions.rs::the_mir_of_an_if_expression_is_a_two_arm_match_without_its_tag_test` `tests/language/if_expressions.rs::a_loop_holding_an_if_expression_has_the_receipt_of_the_equivalent_match`

```invar outcome=value:4322
fn grade(x: i32) -> i32 { return if x > 90 { 4 } else if x > 75 { 3 } else if x > 50 { 2 } else { 1 }; }
fn main() -> i32 {
    let a = grade(95) * 1000 + grade(80) * 100 + grade(60) * 10 + grade(3);
    return if a == 4321 { a + 1 } else { 0 };
}
```

```invar outcome=trap:DIVISION_BY_ZERO
fn main() -> i32 { let z = 0; return if 10 / z > 1 { 1 } else { 2 }; }
```

```invar outcome=trap:DIVISION_BY_ZERO
fn main() -> i32 { let z = 0; return if z == 0 { 10 / z } else { 10 / 1 }; }
```

```invar outcome=diag:TYPE_MISMATCH
fn main() -> i32 { return if 1 { 2 } else { 3 }; }
```

```invar outcome=diag:TYPE_MISMATCH
fn main() -> i32 { let v = if true { 2i32 } else { 3u32 }; return 0; }
```

```invar outcome=diag:PARSE_IF_WITHOUT_ELSE
fn main() -> i32 { return if true { 1 }; }
```
