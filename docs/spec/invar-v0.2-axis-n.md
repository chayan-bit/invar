# Invar v0.2: nondeterminism axis N

Status: DRAFT; section 7 of the v0.2 chapter `docs/spec/invar-v0.2-axes.md`, kept in its own file.
Counts: statements=12 citations=13
Size record: tokens=1565 estimator=bytes/3 limit=6000
## 7. Nondeterminism axis N

[NA7.1] This file is section 7 of `docs/spec/invar-v0.2-axes.md`, follows its section 1 conventions and has its own 6,000-token budget. `tests/spec_examples.rs::N_SPEC`
[NA7.2] The N axis governs only the order in which a region produces and combines intermediate results, and float freedom is the C axis, so an N level and a C level combine without interacting. `src/axes.rs::N_AXIS`
[NA7.3] A region that declares no N level is N0, and an undeclared program's receipts are its v5 receipts plus the safe axis fields of section 6. `tests/numeric_tiers/explain_v6.rs::undeclared_documents_project_to_the_committed_v5_goldens`
[NA7.4] N0 fixes a deterministic order, and its oracle is bit equality with the serial reference evaluator, with the same trap, trap position and fuel. `src/eval.rs`
[NA7.5] N1 frees the hardware order of a combine only when the combine is a proven exact commutative monoid, such as the sealed folds of law L2 and the scatter-combine of law L11, and its oracle is bit equality with the serial reference. `docs/net-normal-form-v01.md`
[NA7.6] N2 lets intermediate results arrive in any order and then applies a canonicalisation the compiler inserts, such as a stable sort by source index, and its oracle is bit equality after that canonicalisation. `docs/net-normal-form-v01.md`
[NA7.7] N3 makes the result any member of a set that a checkable predicate defines, needs a source construct naming that predicate, and its oracle is predicate membership, with an optional recorded order for replay. `src/axes.rs::N_AXIS`
[NA7.8] N4 admits bounded-error float results that may vary from run to run, is never chosen by the compiler, and its oracle is the declared error bound over a seeded corpus. `src/numeric_contract/fold.rs::relaxed_admits`
[NA7.9] N0, N1 and N2 keep two runs bit-identical and record `deterministic: true`, while N4, and N3 when its predicate admits several results, record `deterministic: false`. `src/axes/receipt.rs::receipt`
[NA7.10] A receipt names the rule behind its level in `n_justification`: `serial-reference-bits` for an N1 or N2 plan that already meets N0, a monoid proof for an N1 plan that frees the hardware order, the canonicalisation kind for an N2 plan that inserts one, the predicate name at N3 and the bound at N4. `schema/explain-v6.schema.json`
[NA7.11] This engine honours N1 and N2 by admitting only plans that already meet N0, so their receipts name `serial-reference-bits`, and it refuses N3 and N4 with `CONTRACT_UNAVAILABLE`. `src/axes/receipt.rs::N_JUSTIFICATION`
[NA7.12] The table below is the conformance column for N, a level without a passing test stays a hole, and the conformance suite of #208 will generate this column with its evidence. `tests/numeric_tiers/region_axes.rs`

| Level | Oracle | Fixture | Conformance today |
| --- | --- | --- | --- |
| N0 | bit equality with the serial reference | `value:6` | every admitted plan, every engine |
| N1 | bit equality, exact commutative monoid | `value:6` | N0 plans only; hole: no hardware-order plan |
| N2 | bit equality after canonicalisation | `value:3` | N0 plans only; hole: no canonicalisation pass |
| N3 | predicate membership | `diag:CONTRACT_UNAVAILABLE` | hole: refused, no predicate construct |
| N4 | error bound over a seeded corpus | `diag:CONTRACT_UNAVAILABLE` | hole: refused |

```invar outcome=value:6
fn add_all(xs: read Array<u32>) -> u32 {
    var acc: u32 = 0u32;
    for i in 0u32..len(xs) { acc = wrapping_add(acc, xs[i]); }
    return acc;
}
fn main() -> i32 {
    let xs: Array<u32> = [1u32, 2u32, 3u32];
    if add_all(xs) == 6u32 { return 6; }
    return 0;
}
```

```invar outcome=value:6
#[region(n = "n1")]
fn add_all(xs: read Array<u32>) -> u32 {
    var acc: u32 = 0u32;
    for i in 0u32..len(xs) { acc = wrapping_add(acc, xs[i]); }
    return acc;
}
fn main() -> i32 {
    let xs: Array<u32> = [1u32, 2u32, 3u32];
    if add_all(xs) == 6u32 { return 6; }
    return 0;
}
```

```invar outcome=value:3
#[region(n = "n2")]
fn keep_odd(xs: read Array<u32>) -> u32 {
    var kept: u32 = 0u32;
    for i in 0u32..len(xs) { if xs[i] % 2u32 == 1u32 { kept = wrapping_add(kept, 1u32); } }
    return kept;
}
fn main() -> i32 {
    let xs: Array<u32> = [1u32, 2u32, 3u32, 5u32];
    if keep_odd(xs) == 3u32 { return 3; }
    return 0;
}
```

```invar outcome=diag:CONTRACT_UNAVAILABLE
#[region(n = "n3")]
fn main() -> i32 { return 0; }
```

```invar outcome=diag:CONTRACT_UNAVAILABLE
#[region(n = "n4")]
fn main() -> i32 { return 0; }
```
