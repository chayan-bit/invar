# Invar v0.2: attributes, axes and receipts

Status: DRAFT; a v0.2 chapter on `main`, beside Appendix B of the v0.1 specification.
Counts: statements=20 citations=21
Size record: tokens=2944 estimator=bytes/3 limit=6000
## 1. Conventions

[NA1.1] This chapter extends `docs/spec/invar-v0.1.md`, and its statements, notes, citations and fenced blocks follow v0.1 sections 1.1 and 1.2 and are checked by the same test. `tests/spec_examples.rs::AXES_SPEC`
[NA1.2] The chapter carries its own counts and size record, and its size stays below 6,000 tokens by the v0.1 estimator. `tests/spec_examples.rs::SPECS`
[NA1.3] Statement ids in this chapter start with `NA` and are never reused by the v0.1 file. `tests/spec_examples.rs`

## 2. Attributes

[NA2.1] An attribute is `#[name(key = "value", ...)]` with at least one key, and every value is a string literal. `spec/grammar.ebnf`
[NA2.2] One registry lists every attribute name, the positions it may take, its keys and the closed set of values of each key, and every attribute is accepted or refused there and nowhere else. `src/front/items.rs::REGISTRY`
[NA2.3] An unknown name, an unknown key, a repeated key or an attribute in a position it does not take is `UNSUPPORTED_FEATURE`. `src/front/items.rs::check_position`
[NA2.4] A value that is not a string literal is `PARSE_EXPECTED_TOKEN`, and a string outside the key's set is the code the registry gives that key. `src/front/items.rs::attribute_value`
[NA2.5] The only position is a function: the attribute precedes `fn` or `pub fn`, and before a record or variant it is refused. `src/front/items.rs::item_prefix`

| Attribute | Positions | Key | Values | Unknown value |
| --- | --- | --- | --- | --- |
| `numeric` | function | `fold` | `strict`, `tree`, `exact`, `relaxed` | `NUMERIC_POLICY` |

Note: `#[numeric(fold = "...")]` and its tiers are specified in `docs/numeric.md`; the registry only owns its spelling.

```invar outcome=value:7
#[numeric(fold = "tree")]
fn fold_sum(xs: read Array<f32>) -> f32 {
    var acc: f32 = 0.0;
    for i in 0u32..len(xs) { acc = acc + xs[i]; }
    return acc;
}
fn main() -> i32 {
    let xs: Array<f32> = [1.0, 2.0, 4.0];
    if fold_sum(xs) == 7.0 { return 7; }
    return 0;
}
```

```invar outcome=diag:NUMERIC_POLICY
#[numeric(fold = "fast")]
fn main() -> i32 { return 0; }
```

```invar outcome=diag:UNSUPPORTED_FEATURE
#[inline]
fn main() -> i32 { return 0; }
```

```invar outcome=diag:UNSUPPORTED_FEATURE
#[numeric(fold = "tree")]
struct Point { x: f32 }
fn main() -> i32 { return 0; }
```

## 3. Contract axis C

Note: a region picks one level on each of six independent axes: C numeric contract, S schedule, N nondeterminism, T timing, D distribution and A algorithm substitution.
Note: this section defines C; schedule freedom and ordering freedom are not C levels, so a region can be C2 with N0, or C0 with N2, without the choices interacting.
[NA3.1] A region that declares no C level is C0, and C0 is the behaviour of every program without a declaration, so no existing program changes bits, fuel or trap position. `src/axes.rs::CLevel`
[NA3.2] A C level selects which algebraic laws (L1 to L20 of the net normal form) and which hardware operations the typed-net rewrite and extraction may use for the region; it never selects another engine, and every rule it admits still yields its certificate. `docs/net-normal-form-v01.md`
[NA3.3] C0 admits exact laws only, and its oracle is bit equality with the serial reference evaluator, NaN class aside, with the same trap, trap position and fuel. `docs/numeric.md`
[NA3.4] C1 admits one deterministic float rule named by its contract, a fixed reduction tree or one exact rounding, and its oracle is bit equality with that contract's reference on every engine and worker count. `src/numeric_contract/fold.rs::reference`
[NA3.5] C2 admits bounded-error rules, and its oracle is that a result lies within its contract's stated bound of the exact or strict reference and repeats bit for bit on one device. `src/numeric_contract/fold.rs::relaxed_admits`
[NA3.6] C5 pairs a trusted foreign device kernel with an Invar reference function, and its oracle is a differential test against that function whose digest the receipt records. `src/axes.rs::C5`
[NA3.7] At every level the source trap order, fuel prefixes, affine ownership and cleanup are kept, and only float bits may vary, within the level's oracle. `docs/numeric.md`
[NA3.8] The C axis has no C3 or C4, and older prose that names them means S1 or S2 and N4. `src/axes.rs::level_names_round_trip_and_c3_c4_are_not_levels`
[NA3.9] Each numeric contract on `main` maps to exactly one level by the table below, and a test checks that the mapping is total and that the table matches the code. `src/axes.rs::every_contract_maps_to_exactly_one_level_as_the_spec_says`
[NA3.10] An intrinsic that names its contract, such as `sum_tree` or `tc_mul_add`, applies it at that call, and a fold tier applies it to that function's folds; neither raises the level of the rest of the region. `src/axes.rs::CONTRACTS`
[NA3.11] A C0 checked fold may be evaluated by any associative summary that reproduces its value and its first trap position, and `cpu-parallel` evaluates a counted `s = s + a[i]` over `i32` or `u32` by the exact (total, minimum prefix, maximum prefix) carry in 64 bits, rerunning only the chunk that holds the first overflow. `src/checked_fold.rs::Carry`
[NA3.12] The value, trap category, trap span, fuel prefix and cleanup of such a fold equal the serial fold at every worker count, and `explain` names the plan `checked-sum-carry-if-guard` and the rule `pnf-l3-checked-sum-carry-v1`. `tests/schedule_cpu/checked_fold_carry.rs`

| Level | May vary | Must not vary | Evidence | Minimum receipt |
| --- | --- | --- | --- | --- |
| C0 | schedule and worker count | bits (NaN class aside), traps, fuel | `checked_invariant`, `trusted_rule` or `runtime_guard` per law | `c_level` |
| C1 | association, fixed by the contract | bits on every engine and worker count, traps, fuel | the contract's reference | `c_level`, `contract_name` |
| C2 | float bits, within the bound | bits on one device, traps, fuel | the bound check | `c_level`, `contract_name`, `bound` |
| C5 | the kernel | agreement with the reference on the recorded corpus | `trusted_rule` | `c_level`, `contract_name`, `trusted_rule`, reference function, test digest |

| Contract | Selected by | Level |
| --- | --- | --- |
| `strict-f32-v1` | every `f32` operation by default, `fold = "strict"` | c0 |
| `strict-f64-v1` | every `f64` operation | c0 |
| `invar-math-cr-v1` | `sin`, `cos`, `sqrt` | c0 |
| `f32-tree-v2` | `fold = "tree"`, `sum_tree`, `dot_tree` | c1 |
| `f32-exact-v2` | `fold = "exact"`, `sum_exact`, `dot_exact` | c1 |
| `f32-relaxed-v2` | `fold = "relaxed"`, `reassoc_add` | c2 |
| `tc-f16-f32-v1` | `tc_mul_add` | c2 |

Note: the C2 bounds are the Relaxed bound of `docs/numeric.md` and the amended D5 bound of `tc-f16-f32-v1`.
Note: C5 has no implementation or fixture until #201; section 4 says how a declaration of it is refused.

Note: the C0 fixture: a strict fold is the source-order left fold, so the 1.0 is absorbed and the result is 0.5.

```invar outcome=value:5
fn fold_sum(xs: read Array<f32>) -> f32 {
    var acc: f32 = 0.0;
    for i in 0u32..len(xs) { acc = acc + xs[i]; }
    return acc;
}
fn main() -> i32 {
    let xs: Array<f32> = [100000000.0, 1.0, -100000000.0, 0.5];
    if fold_sum(xs) == 0.5 { return 5; }
    return 0;
}
```

Note: the C1 fixture: `f32-tree-v2` and `f32-exact-v2` each give one result on every engine, and here they differ from the left fold and from each other.

```invar outcome=value:1
#[numeric(fold = "exact")]
fn exact_sum(xs: read Array<f32>) -> f32 {
    var acc: f32 = 0.0;
    for i in 0u32..len(xs) { acc = acc + xs[i]; }
    return acc;
}
fn main() -> i32 {
    let xs: Array<f32> = [16777216.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
    if sum_tree(xs) != 16777218.0 { return 2; }
    if exact_sum(xs) != 16777224.0 { return 3; }
    return 1;
}
```

Note: the C2 fixture: `f32-relaxed-v2` admits any result inside its bound and takes the Tree lowering today, which lies inside it.

```invar outcome=value:2
#[numeric(fold = "relaxed")]
fn relaxed_sum(xs: read Array<f32>) -> f32 {
    var acc: f32 = 0.0;
    for i in 0u32..len(xs) { acc = acc + xs[i]; }
    return acc;
}
fn main() -> i32 {
    let xs: Array<f32> = [16777216.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
    if relaxed_sum(xs) == 16777218.0 { return 2; }
    return 0;
}
```

Note: the C0 checked-fold fixture: the first prefix leaves the `i32` range at the second element, so the run traps there although the full sum fits.

```invar outcome=trap:INTEGER_OVERFLOW
fn main() -> i32 {
    let a: Array<i32> = [2147483647, 1, -5];
    var s: i32 = 0;
    for i in 0u32..len(a) { s = s + a[i]; }
    return s;
}
```
