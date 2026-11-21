# Invar v0.2: attributes, axes and receipts

Status: DRAFT; a v0.2 chapter on `main`, beside Appendix B of the v0.1 specification.
Counts: statements=8 citations=9
Size record: tokens=829 estimator=bytes/3 limit=6000
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
