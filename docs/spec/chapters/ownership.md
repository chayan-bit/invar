## 4. Ownership

### 4.1 Moves

[N4.1.1] An array owner moves only with `move name`, from a fresh expression, or by `return name` or `return move name`, a bare owner-to-owner binding is rejected rather than copied, and `move` on a scalar is rejected. `tests/language/arrays_ownership_and_cleanup.rs::arrays_ownership_and_cleanup_rejects_invalid_ownership`
[N4.1.2] Reading a moved owner is `OWN_USE_AFTER_MOVE`, and a move of an outer binding inside a loop is rejected. `tests/language/arrays_ownership_and_cleanup.rs::arrays_ownership_and_cleanup_rejects_invalid_ownership`
[N4.1.3] At a join of continuing `if` paths an owner is usable only if usable on every continuing path, and a returning branch takes no part in the join. `src/ownership.rs`

```invar outcome=diag:OWN_USE_AFTER_MOVE
fn take_it(xs: take Array<i32>) -> i32 { return xs[0u32]; }

fn main() -> i32 {
    var xs: Array<i32> = [4, 5];
    let first: i32 = take_it(move xs);
    return first + xs[1u32];
}
```

### 4.2 Call loans

[N4.2.1] An array parameter has exactly one mode: `read`, `mut` or `take`, and scalar parameters take no mode. `tests/language/names_types_and_calls.rs::names_types_and_calls_rejects_names_types_and_returns`
[N4.2.2] Borrowed arguments are bare names, read loans may coexist, and a mutable loan excludes every other access to that array. `tests/language/call_loans.rs::call_loans_rejects_conflicts_at_call_boundary`
[N4.2.3] A loan is acquired after its argument is evaluated and held until the outer call completes, with no two-phase borrow. `tests/language/call_loans.rs::call_loans_rejects_conflicts_at_call_boundary`
[N4.2.4] A `read` or `mut` parameter cannot be returned as an owner, and a borrower cannot escape its call. `tests/language/call_loans.rs::call_loans_rejects_conflicts_at_call_boundary`
[N4.2.5] Rejected borrows never trigger implicit cloning or reference counting. `docs/architecture.md`
[N4.2.6] `read` accepts an owner or a compatible borrowed parameter, `mut` a mutable local owner or a `mut` parameter, and `take` only an owned array. `tests/language/call_loans.rs::call_loans_read_read_and_take`
[N4.2.7] In `f(xs, g(xs))` a mutable first loan bars `g` from reading `xs`, a read loan bars `g` from mutating or consuming it, and one actual passed twice mutably or both borrowed and consumed is rejected. `tests/language/call_loans.rs::call_loans_rejects_conflicts_at_call_boundary`
[N4.2.8] A nested call may reborrow a `mut` parameter as `read` or `mut`, suspending the parent access until it returns. `tests/language/call_loans.rs::call_loans_mut_reborrow_restores_caller_access`

```invar outcome=diag:OWN_BORROW_CONFLICT
fn both(a: mut Array<i32>, b: read Array<i32>) -> i32 { return a[0u32] + b[0u32]; }

fn main() -> i32 {
    var xs: Array<i32> = [1, 2];
    return both(xs, xs);
}
```

### 4.3 Cleanup

[N4.3.1] Normal exits release still-owned arrays in reverse declaration order for each exited scope, and a move suppresses release. `tests/language/arrays_ownership_and_cleanup.rs::arrays_ownership_and_cleanup_and_failure_order`
[N4.3.2] The reference evaluator records a cleanup trace, which is a test observation and not a source feature. `docs/validation.md`

