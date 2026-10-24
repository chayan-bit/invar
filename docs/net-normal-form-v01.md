# Net normal form v0.1 (P1a regions, P1b laws and certificates)

This is the net normal form for the v0.1 law set.
This document is the contract that the lowering phases (P2, P3, P4a, P4b and P8) consume.
Lowering itself is out of scope here.

## 1. Layers

The interaction net stays the oracle (M1).
Every layer below is derived from `Net::compiled_mir(proof)` and is bound to the compiled-MIR contract digest, so any change to the net invalidates it, exactly as it invalidates the proof.

| Layer | Module | Entry points |
| --- | --- | --- |
| P1a regions | `general_net::regions` | `partition(net, proof)`, `verify(net, proof, &partition)` |
| P1b normal form | `general_net::normal_form` | `extract(net, proof)`, `from_partition(&partition)` |
| Certificates | `general_net::normal_form::certificate` | `check`, `check_with`, `check_all` |
| Reference semantics | `general_net::normal_form::exec` | `run(net, proof, &form, entry, args, Mutation::None)` |

No layer looks at file names, hashes of sources or benchmark ids.

## 2. Regions (P1a)

`regions::partition` decompiles each compiled-MIR function into a structured statement tree (`regions::tree`), or records why it stays an opaque serial frame (`Unstructured`).
A region is a function body or a natural loop, keyed by `RegionId { function, head }`, where `head` is the loop-head block (`None` for the body).
Each `Region` records its blocks, exits, live-in and live-out locals, array reads and writes, effects (host calls, possible traps, allocation, early exit, callees, `mut` loans) and ownership (declared locals, moved owners).
Effects come from the net's own agents and call annotations, and purity (`FunctionRegions::pure`) is transitive over callees.
`regions::verify` re-derives the whole partition from the current net and rejects any difference.
Loops are classified by `Counting`: `For`, `CountedWhile` (`while i < E { ...; i = i + 1; }` with `i` and the names of `E` otherwise unchanged), or `Uncounted`.

## 3. Normal form API (P1b)

`normal_form::extract` returns a `NormalForm { version, fingerprint, functions }`.
Each `FunctionForm` carries:

- `opaque`: why the function stays a serial frame, if it does;
- `recursion`: `Some(Recursion::PathSum { modulus })` (L9) or `Some(Recursion::RangeSum)` (L10);
- `loops`: one `LoopForm { region, node, certificates }` per loop in source pre-order;
- `forks`: residual L20 fork sites (`ForkSite { start, siblings }`);
- `certificates`: the function-level certificates (L9, L10, L20).

A `LoopForm.node` is `Node::Serial { reason }` (the missing law is named) or `Node::Parallel(LoopNode)`.
A `LoopNode` has:

- `counting`: the counted shape the lanes are enumerated from;
- `carried`: carried scalar state in evaluation order, each `Carried { var, law, rep, scan }`;
  a representation may read variables earlier in the list (successive scans), and `scan` says the prefixes are read outside the variable's own updates;
- `arrays`: the law of every array the loop writes (`ArrayLaw::{Own, Combine { wrapping }, Ordered, Compaction { counter }, Monotone { own_writer }}`);
- `first_exit`: the loop has an early `return` (L15);
- `fused_with`: the adjacent loop executed in the same lanes (L19).

`LoopNode::shape()` gives the PNF node kind of normal-form design section 3 (`map`, `reduce`, `scan`, `scatter`, `frontier`, `first`).
`NormalForm::is_flat()` is true when no residual fork-join remains, which selects D5 tier 1.

### Representations

A `Rep` is a law representation `R` with identity, combine and apply:

| `Rep` | Law | Words | Meaning |
| --- | --- | --- | --- |
| `Sealed(WrappingSum or WrappingProduct)` | L2 | 1 | `(Z/2^32, +, 0)` or `(Z/2^32, *, 1)` |
| `LiftedSum(ty)` | L3 | 3 | total, minimum and maximum prefix, exact in 64 bits |
| `SaturatedProduct(ty)` | L3 | 1 | signed magnitude saturated above the integer range |
| `Affine(ring)` | L4 | 2 | `s -> a s + b` over `ring` |
| `Matrix { ring, vars }` | L5 | `(k+1)^2` | `k <= 4` jointly linear variables |
| `Tropical { order, ty }` | L6 | 2 | `s -> max(s + a, b)` or `min` |
| `Select { order, strict, elem, companions }` | L7 | 1 + companions | leftmost (strict) or rightmost extremum; f32 NaN is the identity |
| `Last` | L12 | 1 | last value written in source order |
| `Counter(ty)` | L13 | 1 | exclusive counter scan, prefixes are store positions |

A `Ring` is `Wrapping32`, `Checked64(ty)` (checked arithmetic regrouped in `Z/2^64`, exact before the first violation, Lemma A) or `Residue { modulus, ty }` (`Z/P`, exact for non-negative values, Lemma B).

### Laws and guards

| Law | Applies to | Runtime guards (false selects the serial fallback for the whole region) |
| --- | --- | --- |
| L1 map | iterations, owned arrays | `StrideNonZero(k)` for a store index `i * k + r` with an invariant local stride |
| L2 sealed fold | `wrapping_add`, `wrapping_mul` folds | none |
| L3 trap-lifted fold | checked sums and products | none (first trap located by per-lane replay) |
| L4 affine | `s = a s + b`, resets | residues: `ReducedSeed(s)` (`0 <= s < P`), `NonNegativeCoefficients(s)` |
| L5 matrix | jointly linear groups | residues: both guards for every member |
| L6 tropical | `max(s + a, b)` shapes (Kadane) | none |
| L7 select | argmin and argmax with companions | none |
| L9 path sum | `(sum c_j r_j + k) % P` recursion | `TableRange { modulus }`, `NonNegativeConstants`, `NoOverflowCombine` |
| L10 range sum | `r_1 + ... + r_m` recursion | none |
| L11 scatter-combine | `a[k] = a[k] + e` (checked or wrapping) | none |
| L12 ordered write | write-only arrays, `Last` scalars | none |
| L13 compaction | stores at a certified counter | none |
| L14 monotone frontier | constant flag stores, own-writer reads | none |
| L15 first exit | early `return` | none |
| L19 map fusion | adjacent equal-range maps | none |
| L20 fork-join | independent sibling calls to pure scalar functions | none |

The `NoOverflowCombine` bound is static: `sum_j c_j (P - 1) < 2^31 - 2^20`, so only the runtime constant needs checking.

## 4. Certificate format

```rust
pub struct Certificate {
    pub version: u32,                 // normal_form::VERSION (1)
    pub fingerprint: [u8; 32],        // compiled-MIR contract digest
    pub region: RegionId,
    pub law: LawId,                   // L1 .. L20 of the v0.1 set
    pub subject: Subject,             // Iterations, Var, Array, Exit, Recursion, Fused, Fork
    pub rep: Option<Rep>,
    pub guards: Vec<Guard>,
    pub facts: Vec<Fact>,             // per-update affine/reset/select facts, stores, combines, sibling count
    pub annotations: Vec<Annotated>,  // lowering annotations, none emitted in v1
}
```

Every applied law yields one certificate; a parallel loop carries one `L1 Iterations` certificate, one per carried group, one per written array, one for an early exit and one for a fusion.

### Lowering annotations (P4a iteration 2)

P4a iteration 2 found that a law-directed flat form matches or beats handwritten Metal on 31 of 32 gate cells once the normal form carries the right facts.
`normal_form::annotation` gives those facts a typed place on the certificate, and the checker validates them; range inference itself comes later.

| P4a profile item | Where it lives | Checker in v1 |
| --- | --- | --- |
| 1. Bounds and signedness of every intermediate, and the guards that license narrow 32-bit unchecked arithmetic with an exact checked fallback | `Annotation::Bounds { target, low, high, signedness }` and `Annotation::Narrow { target, bits, signedness, license: Vec<RangeGuard> }` over `Target::{Var, Array, Edge, PathWeight, Op(Span)}` | well-formedness, then verified by the named `Analysis` that produced it, else `Rejection::Unverified` |
| 2. Exact operator spans, demand order and checked-fallback state | `Target::Op(Span)`, `Annotation::Demand(Vec<Span>)`, `Annotation::Fallback { vars, arrays }` | re-derived from the region and compared exactly |
| 3. Combine laws, coefficients and modulus boundaries | `Rep`, `Fact::Affine`, `Fact::Combine` (existing) and `Annotation::Modulus { target, modulus }` | facts re-derived as before; each modulus boundary re-derived from its `% P` site, `P >= 2` |
| 4. Enumeration costs (padding, duplicate leaves, tail work, imbalance) | cost-model input of the region, not a certificate fact | not typed in P1 |
| 5. Target fusion thresholds, launch and layout choices, calibrated coefficients | cost-model input of the region and target, not a certificate fact | not typed in P1 |
| 6. Provenance, with legality, profitability, planning cost and device acceptance kept apart | `Annotated { annotation, provenance: Provenance { evidence: Evidence, source: Source } }` | only `Evidence::Legality` is allowed on a certificate; `Source::Analysis` needs its named `Analysis` |

`Annotation::Narrow` means the target may run as unchecked `bits`-bit arithmetic when every `RangeGuard { var, low, high }` holds at region entry, and otherwise the exact checked fallback runs and reports the first source-order trap.
`Annotation::Demand` lists every operator, index and store site of the region in evaluation order (statements in source pre-order, operands before their operator, a `while` condition first), which is the order that decides which trap is first.
`Annotation::Fallback` names the outer scalars and arrays the region writes, which the serial fallback must restore before it re-runs the region.
`certificate::rederive(&partition, region)` returns the demand order, fallback state and modulus boundaries of a region as `Source::Rederived` legality annotations, so a lowering can attach them to any certificate of that region.
A modulus boundary is a `% P` site with a literal `P >= 2`, the same rule as `Ring::Residue` and `Recursion::PathSum`; `% 1` stays an ordinary demand site and is never emitted or accepted as a boundary.
P1 derives no bounds or narrowing itself.
A `Source::Analysis { name, version }` annotation is verified only by a caller-supplied `certificate::Analysis` with that name and version (`check_with_analyses`, `check_all_with_analyses`), which must re-derive exactly that annotation for the region from the partition alone; a different value is `Rejection::Annotation`, and a missing analysis is `Rejection::Unverified`.
A `Source::Rederived` bound or narrowing is always unverified, because P1 has nothing to re-derive it from.
P4a's `path_sum::rederive` fits this hook by implementing `Analysis`.
The extractor emits no annotations in v1.

Well-formedness means: the target exists in the region (a scalar local, an array the region reads or writes, an edge between region blocks, a site span of the region, or a path weight of an L9 certificate); `low <= high`; unsigned bounds are non-negative; signedness agrees with a variable's declared type; narrowing targets an operator or a scalar at 8, 16 or 32 bits; each licence guard names a distinct live-in with ordered bounds; demand spans are distinct; fallback names are sorted and unique; and a modulus boundary sits on an operator with `modulus >= 2`.

Items 4 and 5 are inputs the P3 and P4b cost model attaches to a region after planning, alongside the size and shape inputs (trip counts, array lengths, recursion fan-out and depth).
They carry profitability, planning-cost or device-acceptance evidence, so they never enter a certificate, and P1 does not type them.

## 5. Checker

`certificate::check(net, proof, &cert)` re-derives the partition from the net and accepts a certificate only when every stated fact holds on the current statement tree:

- version, fingerprint and region exist, and every annotation is well formed, legality evidence and equal to the one re-derived by P1 or by the named analysis that produced it (otherwise unverified);
- the subject exists in the region and has the right kind;
- the representation matches the re-derived updates in the certificate's ring, using the polynomial normal form of `normal_form::poly` (atoms compared without spans), not the extractor's syntactic linear forms;
- the guard set is exactly the set the law requires (order-insensitive, no duplicates);
- the facts equal the re-derived per-update or per-store facts, in source order.

The checker keeps its own view of a loop (`certificate::view`): body-local definitions are substituted in program order, and a definition captured before a name it reads is reassigned is marked stale.
Closed-composition laws (sums, products, affine and matrix maps) hold whichever version of a variable a value reads, so stale reads are allowed there.
Comparisons are not, so select shapes require exact (non-stale) values and guards.
Companions must be assigned only inside the select's own `if` arms, from values that read neither the select variable nor another companion.
L1 legality re-checks the region's effects: no host call, no `mut` loan, no outer owner moved, only pure callees, and an induction the body never assigns.

`certificate::check_all(net, proof, &form)` checks every certificate and the completeness of the form: each parallel loop's certificates cover exactly its carried variables (members and companions included, each once), its written arrays (with the right `ArrayLaw` parameters), its exit and its fusion, in a dependency-respecting order, and a non-`scan` group is never read outside its own updates.
`Rejection` names the failed check (`Fingerprint`, `Region`, `Subject`, `Law`, `Rep`, `Guards`, `Facts`, `Annotation`, `Unverified`, `Incomplete`).

## 6. Reference semantics (`exec`)

`exec::run` executes a program through its normal form on the CPU and returns the result with a `Receipt` (laws applied, guard fallbacks, violations, lanes).
It is the specification the lowerings must match under observation contract v2 (value plus first source-order trap):

1. A certified loop is cut into lanes, one per iteration.
2. For each carried group in `carried` order, every lane runs symbolically to produce its element of `R`; a balanced tree scan gives exclusive prefixes; lanes whose prefix is unrepresentable are skipped, since an earlier lane has the first event.
3. L11 arrays get a contribution phase (targets read as zero) and a per-target running table.
4. Every lane is replayed concretely from its prefixes; the minimum lane with an event (trap or `return`) decides the outcome, and only lanes up to it publish writes, merged in source order.
5. Recursion with L9 or L10 runs in path mode: self-call results become linear holes, trap order is the lexicographic path order, and a failed guard falls back to serial evaluation of the function.
6. L20 evaluates siblings in reverse and commits them in source order, keeping the first source-order trap.

A false guard, an unrepresentable state or an unknown control decision falls back to serial execution of the whole region and is recorded in the receipt.
`Mutation::{LastTrap, DropGuards, WrongCombine}` inject soundness bugs on purpose; the tests require each law to have a mutation the oracle catches.

## 7. What the lowering phases consume

- **P2** (body map M, host frame, `first`, frontier rounds): `LoopNode` with `shape()` in `map`, `first` or `frontier`, the `ArrayLaw::Own` and `Monotone` footprints, `counting` for the lane range, `fused_with` for L19, and `Node::Serial { reason }` for the explain output.
- **P3** (family admission predicates): `LoopNode::shape()`, the `Rep` of each carried group and the `ArrayLaw` of each array; a family admits a `(node, R, body op set)` instead of matching MIR.
- **P8** (generic reduce and scan templates): `Rep` gives the word count, identity, combine and apply; `Ring` gives the arithmetic; `Guard` lists the entry checks; `Carried::scan` selects scan over reduce.
- **P4a** (path sums): `Recursion::PathSum { modulus }` and the L9 certificate's `Fact::Combine` coefficients and constants, with the three L9 guards; `Recursion::RangeSum` for L10; `certificate::rederive` for demand order, fallback state and modulus boundaries; and the `Bounds` and `Narrow` annotation types for the range facts its inference will produce.
- **P4b** (residual fork-join T): `FunctionForm::forks` and `NormalForm::is_flat()`.

Every consumer should call `certificate::check_all` (or `check_with` on its own partition) before trusting a form it did not extract itself.

## 8. Corpus classification against the net-laws prototype

`tests/normal_form/net_normal_form_corpus.rs` classifies every GNET-05 program (60) and every M5-01 held-out candidate (140) and compares each loop's class and law list, and each function's recursion law and modulus, with `tests/net_normal_form_fixtures/prototype.json`.
All 200 programs match the prototype with zero differences, and the checker accepts every normal form in full (`check_all`).

## 9. Tests

- `tests/normal_form/net_normal_form_laws.rs`: differential tests against the CPU net oracle, at least 3,000 randomised cases per law with trapping cases, a caught mutation per law, and forged-certificate rejection.
- `tests/normal_form/net_normal_form_corpus.rs`: corpus classification and checker acceptance.
- Unit tests in `general_net::normal_form::poly` and `general_net::regions`.
