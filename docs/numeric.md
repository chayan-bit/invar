# Numeric contracts

This document is normative for the f32 source semantics (`strict-f32-v1`) and for the declared fold tiers and storage types of numeric policy version 2.
The f64 contract `strict-f64-v1` and the math library `invar-math-cr-v1` are specified in Appendix B of [the specification](spec/invar-v0.1.md); the f32 `sin`, `cos` and `sqrt` sequence is [`invar-trig-f32-v1`](trig-f32-v1.md).

## Policy `strict-f32-v1`

The default policy is `{ version: 1, kind: "strict-f32" }`, named `strict-f32-v1`.
An unknown policy version or kind is refused with a typed diagnostic rather than approximated.

### Source grammar

`f32` is an IEEE 754 binary32 scalar and `Array<f32>` is an affine owned array.
A decimal literal has grammar `digit+ "." digit+` and is rounded once from its exact decimal value to binary32 using round to nearest, ties to even.
Decimal literals contain at most 256 digits and unary negation is separate, including for negative zero.
`filled_f32(length: u32, value: f32) -> Array<f32>` constructs an array under the existing allocation limit.
`+`, `-`, `*`, `/`, unary `-`, and the six comparison operators accept matching f32 operands without implicit integer conversion.
`fma(a: f32, b: f32, c: f32) -> f32` computes the exact product plus addend with one rounding.
`f32_from_u32` and `f32_from_i32` round the exact integer to nearest even.
`sin`, `cos` and `sqrt` (each `f32 -> f32`) follow the frozen [`invar-trig-f32-v1`](trig-f32-v1.md) contract: `sqrt` is correctly rounded, and `sin` and `cos` are a fixed sequence of the operations above that is bit-identical on every engine but not correctly rounded.
`i32_from_f32(value: f32) -> i32` first requires a finite value in the closed interval `[i32::MIN, i32::MAX]`, then truncates toward zero.
An invalid `i32_from_f32` traps `INTEGER_OVERFLOW` at the source call, before publication of any subsequent result.
`Quantity<f32, Unit>` checks units before erasing the unit tag to f32 and must generate identical code to the equivalent bare f32 program.

### Strict semantics

Every arithmetic operation rounds independently to nearest even.
Subnormal inputs and results are preserved without flush to zero or denormals as zero.
Signed zeros are preserved according to IEEE arithmetic, and exact cancellation under nearest even produces positive zero except addition of two negative zeros.
Overflow produces the signed infinity and invalid floating arithmetic produces NaN rather than an integer arithmetic trap.
NaN payloads and NaN signs are unobservable in M4 and every NaN result compares as the same NaN class for engine agreement.
Source comparisons are IEEE unordered for NaNs, so every ordered comparison and equality is false and inequality is true.
Positive and negative zero compare equal.
Strict engine agreement requires identical non-NaN words and NaN-class equality.
Implicit FMA, contraction, reassociation, and silent fast math are forbidden.
A strict reduction is a source-order left fold from its specified initial value with one rounding per addition.
A strict inclusive scan emits each prefix of that same left fold, including the initial value in the first addition.
An empty reduction returns its initial value and an empty scan has no output elements.
A strict rank-2 contraction initializes each cell to positive zero and visits the inner axis in increasing source order with separate multiply and add roundings.
Parallelism may distribute independent cells or map elements but must preserve the ordered operations within each cell or prefix.

### Explicit relaxed contracts

Explicit `fma` changes only that operation's rounding and does not relax adjacent operations.
`reassoc_add(acc, x)` explicitly permits reassociation of its reduction, with error bounded by `(n - 1) * 2^-24 * sum(abs(x_i))` against the source-order reference for the finite domain covered by that contract.
`tc_mul_add` selects the separate `tc-f16-f32-v1` contract and does not change ordinary f32 operations.
Ordinary checking admits it only as the direct-cell step of a recognized fresh-zero matrix contraction.
Reference, serial MIR, the CPU net oracle and CPU/auto fallback round inputs to binary16, retain source control flow/fuel/bounds/cleanup, and return the once-rounded exact sum of products.
Nonfinite or overflowing reached inputs report `TC_INPUT_RANGE`; an unreached contraction does not inspect its inputs.
The CPU net oracle executes its own activations, fuel, traps and cleanup; only binary16 rounding and exact D5 arithmetic are shared. Native executables use the same pure integer D5 oracle at successful function completion; native library export remains explicitly unsupported for TC.
Device completion may return any result within the amended D5 bound. A quiescent device refusal uses the exact serial path; unknown completion still forbids replay or publication.
Unsupported relaxed syntax or policies must be refused and cannot silently become strict or fast arithmetic.
M4-01 does not implement a Tensor Core schedule or claim performance acceptance.

### Device admission and fallback

Every process probes the live device under the pinned strict compiler options before admitting arithmetic, comparisons, or conversions.
CUDA requires disabled FMA contraction and FTZ and explicit precise division and square-root options.
Metal requires disabled fast math and `#pragma METAL fp contract(off)`.
Division remains CPU-only unless device admission establishes its required rounding and subnormal behavior.
Source `sin`, `cos` and `sqrt` run on a device only through the `invar-trig-f32-v1` helpers ([`trig-f32-v1.md`](trig-f32-v1.md)), which never read or produce a subnormal; they are admitted when every probe mismatch of the operations they use is a subnormal-class case.
A flushing Metal device may execute an operation only behind a sound dispatch guard covering every possible divergence under the documented device model.
A flagged dispatch is discarded after confirmed completion and rerun on the exact CPU path with an explicit fallback reason.
Unknown completion quarantines work and forbids publication and CPU replay.
An operation whose guard cannot be justified is refused before submission.
Policy, IR version, layout version, target capability, compiler options, and generated source all participate in artifact and cache identity.

### Semantic fixtures

The committed `issue_m4_01_numeric_oracle`, `issue_m4_01_decimal_literals`, and `issue_m4_01_source_f32` fixtures establish literal rounding, signed zeros, infinities, NaNs, cancellation, conversion traps, and independent exact CPU outcomes.
The committed `issue_m4_01_device/corpus.rs` establishes zero, one, odd, rectangular, and subnormal contraction cases.
Device or mock-only outcomes do not establish native, CPU-parallel, or hardware acceptance for a missing engine.

### Implementation

The source and MIR evaluators use the integer/rational binary32 oracle.
Native LLVM carries f32 bits in internal scalar words, emits strict floating operations without reassociation/contraction flags, and preserves subnormals.
Only explicit `fma` fuses multiplication/addition.
Source `main` remains i32.
MIR carries the strict numeric policy and layout version 1; verification rejects stale layout versions, and MIR/net/provenance/export identities bind both contracts.
The production layout checker validates bounded physical byte maps.
Ordinary ordered rank-2 matrix loops produce sealed source annotations bound to the exact verified MIR function and stable net identity.
Concrete dimensions and input lengths bind bounded row-major byte maps; unknown forms keep CPU execution.
`tensor::emit_cuda` checks actual source, ownership/net contracts and layouts, then emits a strict program-specific contraction and a canonical SHA-256 cache identity from those exact inputs, options and emitted bytes.
Required CUDA numeric options prohibit FMA contraction and FTZ.
A matching content key remains an untrusted hint; it grants no pointer lease, source-fuel/trap commit or device-fence/launch permission.
`tensor::host::run_source` admits only reached contraction loops, checks the full remaining source fuel, and holds live read loans through private staging and confirmed completion.
Packets bind the artifact key, dimensions, lengths, unique call generation and input-byte digest.
Publication updates the fresh source output atomically and charges its exact source ticks.
Pending, failed, stale, malformed or nonfinite work keeps the ordered source fallback with the original spans and cleanup, and unconfirmed work stays quarantined.

## Policy version 2: fold tiers and storage types

Version 2 lets a function declare how its binary32 folds associate, and adds storage-only narrow types.
Every rule of `strict-f32-v1` still holds for a function without a declaration.

### Policy

A v2 policy is `{ version: 2, kind: "<base>-<fold>" }`.
`base` is `f32` or `f64`.
`fold` is `strict`, `tree`, `exact` or `relaxed`.

| Kind | Identity | Meaning |
| --- | --- | --- |
| `f32-strict` | `strict-f32-v1` | The v1 source-order left fold. |
| `f32-tree` | `f32-tree-v2` | The fixed `sum_tree` shape. |
| `f32-exact` | `f32-exact-v2` | The correctly rounded sum. |
| `f32-relaxed` | `f32-relaxed-v2` | Any order within the `reassoc_add` bound. |
| `f64-strict` | `strict-f64-v1` | The v1 binary64 contract. |

- The strict v2 spellings resolve to the v1 policies `strict-f32-v1` and `strict-f64-v1`.
  - A strict program therefore has one identity, and frozen receipt ids that embed the policy stay byte-identical.
  - Only the three non-strict tiers serialize as version 2.
- Refused with a typed error, never approximated:
  - `fast`, `f32-fast` and every other unknown kind;
  - `tree`, `exact` and `relaxed` on `f64`, which wait for `Array<f64>` reductions;
  - a v2 kind under version 1, and every version other than 1 and 2.
- `tc-f16-f32-v1` is unchanged: no serialized field selects it.
- Code: `NumericPolicy::from_parts`, `NumericPolicy::fold` and `FoldTier` in `src/numeric_contract.rs`.

### Tiered folds

- A tiered fold is one update `acc = acc + term` (or `acc = term + acc`) directly in the body of a counted `for i in lower..upper` loop, where:
  - `acc` is declared `f32`, by annotation, a float literal or a parameter;
  - `acc` is declared at the loop's own depth;
  - the loop reads or writes `acc` nowhere else.
- Its summands are `init` (the accumulator's value before the loop) and one slot per iteration: slot `j` holds the term of iteration `lower + j`. Each iteration reaches its update at most once, and a slot whose update is not reached holds `-0`, the exact additive identity.
- With `slots` the M = `upper - lower` slots (none when `upper <= lower`) and `N = M + 1`:

| Tier | Result |
| --- | --- |
| Strict | `((init + s0) + s1) + ...`, one rounding per addition (v1). |
| Tree | `init` if the loop runs no iteration, otherwise `init + sum_tree(slots)`. |
| Exact | `sum_exact([init, s0, s1, ...])`, the correctly rounded sum of all N summands. |
| Relaxed | Any result the bound below admits. |

- Tree and Exact are bit-identical on every engine and every worker count, because `sum_tree` and `sum_exact` are (Appendix B of [the specification](spec/invar-v0.1.md)).
- With `init` = `+0`, Tree equals `0.0 + sum_tree(slots)`, and Exact equals `sum_exact(slots)` except that an all-`-0` input gives `+0`, as the strict fold does.
- Code: `FoldTier::reference` in `src/numeric_contract/fold.rs`.

#### Lowering on the CPU engines

- In a Tree or Exact function, the checker rewrites each tiered fold before typing (`src/check/numeric/fold_tiers.rs`). A Relaxed function's folds, and the `reassoc_add` folds of a Strict function, take the Tree rewrite, a fixed deterministic choice that the Relaxed contract admits (ENG-03 step 5'):
  - the bounds are evaluated once, in order;
  - an `f32` slot buffer `filled_f32(M, -0.0)` is allocated (M + 1 slots for Exact, slot 0 holding `init`);
  - the update becomes the slot write;
  - after the loop comes one `sum_tree` or `sum_exact` call.
- The reference evaluator, MIR, cpu-parallel and native-serial all run this one program, so they agree in bits and fuel.
- The buffer counts against the array allocation limit, so a fold over more iterations than the limit allows traps `ARRAY_SIZE_LIMIT` before its first iteration (Exact saturates its `M + 1` slot count, so it never overflows first).
- In a Tree or Exact function, any other `f32` sum inside a loop that reads its own target, declared outside the loop, is refused with `NUMERIC_POLICY` and never runs strict. Under Relaxed it runs strict, which the contract admits. That covers:
  - a second update or another read in the loop;
  - an update under `if`;
  - a fold over nested loops;
  - a `while` loop;
  - `acc - term` or `acc + a + b`;
  - an accumulator whose `f32` type is only inferred.
- An accumulator declared inside the loop body restarts each iteration and is not a fold of that loop.

#### On the devices

- A lowered fold needs no device law of its own: the slot store is an ordinary store loop, and the reduction is the same `sum_tree` or `sum_exact` call that forced CUDA and Metal already run on their fsum kernels, which are bit-identical to the CPU reduction (Appendix B of [the specification](spec/invar-v0.1.md)).
- Tests: `tests/gpu_cuda/eng_03_tier.rs` (the H100, `--features cuda`) and `tests/gpu_metal/eng_03_tier.rs` (the Mac, `--features metal`), over one shared body in `tests/support/gpu_eng_03_tier.rs`, run Tree and Exact folds of 1, 4,096 and 100,003 cancelling terms forced on the device. They require the oracle's bits, and the fold's reduction receipt to name the device.

### Declaration

- A function declares its tier with one attribute before `fn` (or `pub fn`): `#[numeric(fold = "strict" | "tree" | "exact" | "relaxed")]`. Without it, a function is strict.
  - An unknown tier string, including `fast`, is a `NUMERIC_POLICY` compile error.
  - Any other attribute or key, or the attribute on a record or variant, is `UNSUPPORTED_FEATURE`.
- The tier belongs to that function alone.
  - A callee keeps its own tier, whatever its caller declares.
  - A closure lifted from the function is strict.
- `reassoc_add(acc, x)` relaxes one fold in any function, and is accepted only as the whole update `acc = reassoc_add(acc, x)` of an `f32` accumulator inside a loop that uses `acc` nowhere else: not in `x`, another statement, a `while` condition or the `for` bounds. The innermost enclosing loop counts, so a read of `acc` after an inner loop ends is allowed. An update under `if` is a fold of its loop.
  - It becomes `acc + x`.
  - Every other use is a `NUMERIC_POLICY` error: a product or any other first argument, a nested call, a use inside an expression, or a non-`f32` fold.
  - The name is reserved.
- Execution:
  - A `reassoc_add` fold folds by its function's tier, and as Tree in a Strict function; the rest of a Strict function stays strict.
  - Tree, Exact and Relaxed lower as below; a Relaxed fold gives the Tree result with the Tree program's fuel.
- Code: `numeric_attribute` in `src/front/parser.rs`, `src/check/numeric.rs`.

### The Relaxed bound

Let `S` be the sum of `|x_i|` over the finite summands, and `B = (N - 1) * 2^-24 * S`.

1. Any NaN summand, or `+inf` together with `-inf`, gives NaN.
2. If `S + B > 2^127`, a partial sum of some order may overflow, and any IEEE value is admitted.
3. Otherwise, infinities of one sign give that infinity.
4. Otherwise, the result is finite and within `B` of the exact rational sum.
5. A zero result is `-0` exactly when every summand is `-0`.

- `B` is the bound of Jeannerod and Rump, which holds for every summation tree.
  - So Strict, Tree and Exact all satisfy Relaxed, and a Relaxed lowering may choose either of the deterministic algorithms.
  - This is the v1 `reassoc_add` bound, stated against the exact sum rather than the source-order fold.
- Code: `fold::relaxed_admits`, computed in exact rationals.

### Storage conversions

`f16` (IEEE binary16), `bf16` (bfloat16), `i8` and `u8` are storage-only: no arithmetic is defined on them.

| Type | Load | Store |
| --- | --- | --- |
| `f16`, `bf16` | Exact widening to `f32`; a NaN loads as the NaN class. | One round to nearest even from `f32`. Overflow gives the signed infinity, subnormals are kept, NaN stays NaN. |
| `i8` | Exact widening to `i32`. | Traps `INTEGER_OVERFLOW` outside `[-128, 127]`. |
| `u8` | Exact widening to `u32`. | Traps `INTEGER_OVERFLOW` above 255. |

- The rational oracle performs every rounding, with binary16 and bfloat16 as instances of its IEEE format, and never uses host floats.
- Code: `src/numeric_contract/storage.rs`.
- `half` 2.7.1 is a test-only cross-check (a dev-dependency).

#### Storage arrays in source

`Array<f16>`, `Array<bf16>`, `Array<i8>` and `Array<u8>` hold storage elements. There are no scalar narrow types.
- `filled_f16(n, x)`, `filled_bf16(n, x)` (with `x: f32`), `filled_i8(n, x)` (`x: i32`) and `filled_u8(n, x)` (`x: u32`) make one; the fill value is stored once, so it rounds or traps like any store.
- `a[i]` loads the widened value (`f32`, `i32` or `u32`), so arithmetic only ever sees widened values. `len(a)` works as on any array.
- `a[i] = v` takes a value of the widened type and stores it by the table above. An i8 or u8 store out of range traps `INTEGER_OVERFLOW` at `v`, so the first trap in source order is the one reported.
- No other array converts to or from a storage array.
- Refused with `UNSUPPORTED_FEATURE` for now:
  - a storage array as a parameter (a closure capture included), result, record field or variant payload, so no caller or host hands one an unconverted value;
  - `view` and `view_mut` of a storage array.
- On the CPU engines, after typing, a storage array runs as its widened array, and every store passes through one generated conversion (`src/check/storage.rs`). The reference evaluator, MIR, cpu-parallel and native-serial run that one program, so they agree in bits, traps and fuel. A store also spends the fuel of its conversion call.
- The elements are held widened, so a storage array does not yet save memory; packed storage is a later change with the same results.

### Receipts

- Placement documents record the lowered folds in `numeric_contracts.fold` (D12): the top-level explain-v5 document and the invar-diagnostics-v3 receipt of `invar run --engine auto`. Each declared tier a lowered fold used maps to the algorithm that ran. The run receipt embedded in v5 as `execution` carries no `numeric_contracts`.
  - Example: `{"exact": "exact", "relaxed": "tree"}`.
  - A Strict function's `reassoc_add` folds record as `relaxed`.
  - Relaxed records `tree` today; the schema also admits `exact`, for a later cost-based choice.
- A program without a lowered fold carries no `fold`, and older document versions never do.
- Code: `math::fold_receipt`; the checker collects the record in `fold_tiers::rewrite`.

### Not yet in this contract

- Updates under `if` and folds over nested loops (refused today).
- A streaming Tree lowering without the slot buffer. (The parallel normal form needs no float-sum law: a lowered fold reaches it as a slot store and one `sum_tree` or `sum_exact` call, and a Strict f32 fold keeps its refusal.)
- Storage arrays across function boundaries, views of them, packed storage, and reductions (`sum_tree` and the folds) over them.

### Tests

`tests/issue_eng_03_numeric_tiers.rs`:
- the policy accept and refuse sets, and the unchanged v1 `Debug` text;
- the tier references, including a fixture where the tree shape differs from the left fold;
- 10,000 seeded fold inputs (cancellation, subnormals, signed zeros, infinities, the NaN class, and any finite word). On each, Strict, Tree and Exact stay inside the Relaxed bound, and Exact equals the once-rounded rational sum;
- results that the Relaxed contract refuses;
- the declaration accept and refuse sets, `reassoc_add` misuse, and the callee and closure tiers;
- Relaxed folds, and `reassoc_add` folds in Strict and Relaxed functions, give the same result and fuel as the Tree program, at every fuel prefix on the source evaluator and MIR;
- all 65,536 f16 and bf16 patterns round-trip and widen as `half` does;
- every f32-to-f16 and f32-to-bf16 rounding boundary, including subnormals and the overflow threshold: the midpoint goes to the even neighbour and one f32 ulp either side goes to the nearer one, for both signs. 100,000 seeded words agree with `half`;
- i8 and u8 narrowing traps;
- a strict f32 fold (no attribute, and `fold = "strict"`) stays `Serial` in the parallel normal form with the v1 reason.

`tests/issue_eng_03_tier_engines.rs`:
- 80 seeded folds under Tree, Exact and Strict (infinities, NaN, signed zeros, subnormals, cancellation, any finite word) equal `FoldTier::reference` on the reference evaluator, MIR, cpu-parallel (2 workers, and 1, 2, 7 and 64 on every tenth) and native-serial under `--features native`;
- a fixture where Tree and Exact differ, and empty loops that keep a `-0` init;
- 100,003 generated terms under Tree and Exact at 1, 2, 7 and 64 workers;
- every fold shape above that is refused, and the shapes that are not folds;
- fuel prefixes on the reference evaluator and MIR.

`tests/issue_eng_03_storage_arrays.rs`, on the reference evaluator, MIR, cpu-parallel (1, 2 and 7 workers) and native-serial under `--features native`:
- f16 and bf16 stores of signed zeros, infinities, NaN, the overflow threshold, subnormals, ties both ways and seeded words equal the oracle's rounding; fill values round too;
- i8 and u8 loads widen and in-range stores keep their value;
- i8 and u8 narrowing traps at the first trap in source order, in a fill, a store and a loop;
- a storage store inside a loop, with fuel prefixes on the reference evaluator and MIR;
- the refused crossings, views and type mismatches.

`tests/issue_eng_03_fold_receipts.rs`:
- the fold receipt for Tree, Exact, Relaxed and a Strict function's `reassoc_add` fold, and for all of them together; none for a strict fold or a tier without a fold;
- placement run and explain documents carry it and validate against their schemas; the established documents do not.
