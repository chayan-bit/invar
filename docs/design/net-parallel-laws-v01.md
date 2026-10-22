# Law-directed parallelism for the GNET net (design, v0.1)

Status: adopted in part; P1b and P8 on main, P10 in #281 (ENG-02) and #279 (ENG-04).
The design was adopted on 2026-10-15 as recorded in section 12, and it supersedes the P1, P3 and P4 plan of #132.
The float contract of sections 0, 2, 4, 5, 7 and 9 to 12 was amended by the maintainer rulings of 2026-10-21 (#321), summarised in "Amendment A" below.
The measurements in section 7 are a prototype's, not a performance claim of any shipped code.
It extends `docs/design/gnet-parallel-v01.md` (#132, called "the region design" below) and changes no contract by itself; the contract changes it needs are listed as decisions in section 12.
Authority: `docs/general-gpu-v01.md`, `docs/cuda-parity-v01.md`, `src/general_net/CONTRACT.md` and the float contract of Amendment A govern.

### Amendment A: the default float contract (maintainer rulings of 2026-10-21, #321)

This document was written under strict-f32-v1, in which source-order f32 accumulation is the only legal order.
The maintainers replaced that default; the rulings below win over any older sentence in this document, and each affected passage carries a pointer back here.
- **Default tier, `deterministic-tree`.**
  - A reduction follows one language-defined, fixed reduction tree.
  - The tree's shape depends only on the problem size, never on thread count, block size, backend or GPU model.
  - Every backend and both reference interpreters (`eval`, `mir_eval`) compute exactly this tree, so results are bit-identical across CPU, CUDA and Metal.
  - `sum_tree` of section 2 (L17) is this tree; it is now the meaning of an ordinary f32 fold, not a separate opt-in operation.
- **Multiply-add.** `a * b + c` lowers to IEEE fused multiply-add (one rounding) on every backend and in the reference interpreters.
- **`fsum`.** The correctly rounded mode (`sum_exact`, L18; #173) stays, and is order-independent by construction.
- **`strict-sequential`.** An explicit opt-in tier for code that must bit-match scalar source order.
  Under it a parallel f32 reduction is refused, as strict-f32-v1 did.
- **`unordered`.** The only non-deterministic tier and an explicit opt-in.
  It is required for float atomics and order-free reductions; float atomics are banned everywhere else.
- **Reduced precision and approximate transcendentals** (TF32, BF16, FP8, approximate math) are explicit types or functions (#328), never silent rewrites.
- **Fixtures and oracles** that encode sequential f32 order are migrated to the tree order (a semantics change; the oracle implements the new definition exactly).
  The migration, the canonical tree shape and the cross-backend bit-identity test are tracked in #280.
- **State on main.** `docs/numeric-v2.md` still defaults a function to strict until the #280 migration lands, so the code, not this document, is the record of what runs today.
- **Not decided here.** The rulings define reductions.
  Scans over f32 affine, matrix and tropical updates (L4 to L6 over f32) have no ruling; they stay refused as laws until one defines a scan tree.

## 0. The model in one paragraph

The span of a net is a property of the net, not of the scheduler: for a strongly confluent net every reduction performs the same interactions with the same causal order, so no runtime (HVM2-style or ours) can finish faster than the longest causal chain of the net as written.
More parallelism therefore has to come from rewriting the net into an observationally equivalent net with a shorter chain, and the only rewrites that do that are **algebraic laws**: associativity and identity (a fold becomes a tree), closure of a function space under composition (a loop-carried update becomes a prefix scan over finitely represented functions: affine maps over a ring, tropical maps, small matrices, finite state maps), **lifting** (extra state that makes a non-homomorphism a homomorphism, including the state that locates the first checked-integer trap), linearity over a semiring (argument-determined recursion becomes a weighted sum over root-to-leaf paths), and order-insensitive writes (idempotent, monotone or last-writer scatters).
The proposal is a **parallel normal form (PNF)**, a small algebra of map, tree reduce, prefix scan over a law representation, segmented operation, scatter-combine, frontier round, path-sum and residual fork-join, which the compiler derives from the verified net by checking each law's proof obligation, records as a certificate next to the region certificate of #132, and lowers to the existing family emitters or the two GNET kernels.
Every law is exact under the integer contract: checked arithmetic is reassociated only in a wider exact ring with an explicit search for the first source-order violation, so the trap index and value are unchanged.
Strict f32 was the hard limit when this document was written: reassociating an f32 fold changed the result in 1,897 of 2,000 random arrays, so under strict-f32-v1 a source-order f32 accumulation keeps its order.
Amendment A removes that limit as the default: an f32 reduction now means the fixed deterministic tree (a definition, so no law reassociates anything), f32 parallelism also comes from independent accumulators (batches) and from laws that are exact in f32 (min, max, argmin with NaN lifting, compare and select), and the correctly rounded `sum_exact` stays available.
`strict-sequential` keeps the source-order behaviour for code that opts in.
On the corpora, the law-directed model reaches a flat normal form (no task frames, no join continuations) for 48 of 60 GNET-05 programs where the region design needs fork-join tasks for 43, keeps all 140 M5-01 held-out candidates at their family span, and changes nothing for batched robotics, where the batch map already saturates the device.
The interaction net stays the semantic oracle and the IR of record; the PNF is an analysis and lowering layer beside it (section 8 compares the alternatives).

## 1. Mathematical model

### 1.1 Dependency structure, work and span

Let `N` be a verified GNET net and `x` an input.
The CPU oracle executes `N` on `x` as a sequence of agent activations `a_1, a_2, ..., a_W`, each identified by `(frame, block activation, token)` (CONTRACT.md "Activation and ordered observations").
**Work** `W(N, x)` is the number of non-Tick activations (the prototype counts one unit per operator, index, call, store, bind, branch and loop step).

The **dependency DAG** `D(N, x)` has the activations as nodes and an edge `a_i -> a_j` (with `i < j`) when one of these holds:
1. data: `a_j` consumes a wire value produced by `a_i`;
2. memory: `a_i` and `a_j` access the same array place and at least one writes (true, anti and output dependences);
3. effect: both carry a `Host` or `HostBoundary` effect token;
4. control: `a_j` exists only because a branch or loop test `a_i` selected its block activation.

The block-position token of CONTRACT.md is **not** an edge of `D`; it is the sequential order the oracle uses to define observations.
**Span** `S(N, x)` is the number of nodes on the longest path of `D`.
For `p` processors, Brent's bound gives `max(W/p, S) <= T_p <= W/p + S` for a greedy schedule; `W/S` is the average parallelism.

Three spans matter:
- `S_tok = W`: the token order, which is what today's oracle and ABI5 root continuation execute.
- `S_dep(N)`: the span of `D`, the best any scheduler can do on the net as written.
- `S_law(N) = min over law-equivalent nets N' of S_dep(N')`: what this proposal targets, approximated by a bounded search.

### 1.2 Why a scheduler alone cannot get there

Lafont's interaction nets (POPL 1990) and interaction combinators (Information and Computation 1997) have exactly one rule per active pair, and rules are local, which gives strong confluence (one-step diamond).
Strong confluence implies that every complete reduction of a net performs the same multiset of interactions, so work is schedule-invariant; the HVM2 paper states the same consequence for its runtime ("the number of reductions is invariant to the order", HVM2 paper §"Interaction Table").
The causal relation between interactions is also schedule-invariant, so the span of the interaction DAG is an invariant of the net.

**Proposition 1 (scheduler bound).**
For every schedule of `N` on any number of processors, the makespan is at least `S_dep(N)` interaction steps.
*Argument:* each edge of `D` is a true dependence (a value, a place, an effect or a control decision) and must be respected by any execution that produces the oracle's observations; a chain of `S` dependent activations takes `S` steps regardless of processors.

The corollary is the design principle: once a runtime exposes all of `S_dep`, the only remaining lever is to change the net.
HVM2 shows the other half of the cost: it reaches 74,000 million interactions per second on an RTX 4090 with near-ideal scaling (HVM2 paper abstract), but each interaction moves 64-bit nodes through global memory with atomic links, and the paper states that single-threaded HVM2 is about 5x slower than GHC and up to 100x slower on loops and mutable arrays (HVM2 paper, "Single Core Inefficiency").
A flat GPU sum reads 4 bytes per element at memory bandwidth and does no allocation, so per element it does one to two orders of magnitude less memory traffic than a reduction that allocates a recursive stack of nodes.
The region design's spike 1 measured the same gap inside Invar: a structure-preserving fork-join with path replay is 1.9x to 2.7x of the flat kernel resident, and 56x when level-synchronous (#132 §8).
The gap is in the **form** (frames, joins, dispatches per level), not only in the span.

### 1.3 Laws

A **law** is a typed rewrite `N => N'` together with:
- an algebraic statement over the values the rewritten agents compute;
- a **proof obligation**, discharged either statically (from the net: types, literals, footprints, intervals) or by a **runtime guard** evaluated before any visible effect, with a compiled fallback to the unrewritten region;
- a **certificate**: the law id, the representation it uses, the guard, and the facts it relied on, re-derived and checked by the region verifier against the current net fingerprint exactly as `RegionCertificate` in #132 §3;
- a span effect and a work effect.

Seven algebraic shapes cover every law in section 2:
1. **Monoid fold.** `fold(op, e, xs)` with `op` associative and `e` an identity equals any bracketing, so it is a tree reduction of depth `ceil(log2 n)` (Bird 1987, list homomorphisms; Blelloch 1990, prefix sums).
2. **Closed composition.** A loop-carried update `s := f_i(s)` with every `f_i` in a family `F` that is closed under composition and has a finite representation `rep(f)` turns into an exclusive scan of `rep(f_1) ; rep(f_2) ; ...`, then a map `s_i = apply(prefix_i, s_0)`: affine maps over a ring (Kogge and Stone 1973; Blelloch 1990), matrices over a semiring (Sato and Iwasaki, PLDI 2011), tropical maps `s -> max(s + a, b)`, and maps on a finite state set (Mytkowicz, Musuvathi and Schulte, ASPLOS 2014).
3. **Lifting.** Adding auxiliary state turns a non-homomorphism into a homomorphism (Farzan and Nicolet, PLDI 2017; modular nested-loop version PLDI 2019): argmin carries the index, maximum segment sum carries a tropical state, and a checked sum carries its minimum and maximum prefix.
4. **Linear recursion over a semiring.** If a recursive function's arguments to its self calls depend only on its own arguments, and its result is `(sum_j c_j * r_j + k) op P` in the self-call results `r_j`, the result equals the sum, over all root-to-node paths, of the product of edge coefficients times the node's constant or leaf value (a tree homomorphism; Morihata et al., POPL 2009, "The third homomorphism theorem on trees"; Miller and Reif 1985 tree contraction for the general case).
5. **Order-insensitive writes.** A scatter whose combine is associative and commutative (sum, min, max, or), idempotent (set to a constant), or ordered last-writer (the right-projection semigroup `x . y = y`, implemented as a per-target maximum writer index) is a parallel scatter-combine.
6. **Monotone frontier.** Writes that only set flags to one constant, with writers determined by round-start state, let every source-order read be reconstructed in parallel: a reader at iteration `u` sees the flag set iff the round-start flag is set or a writer `v <= u` was active this round.
7. **First exit.** A loop that returns at the first iteration satisfying a predicate equals a min-index reduction over the predicate (the same rule #132 uses for the first trap).

### 1.4 Two exactness lemmas

**Lemma A (wide ring, first violation).**
Let a checked integer operation be the restriction to `i32` of a ring operation of `Z`, and let `phi: Z -> Z/2^64` be the ring homomorphism.
Consider a source loop whose state follows `s_j = g_j(s_{j-1})` with each `g_j` a polynomial of degree one in `s` with `i32` coefficients, trapping at the first `j` where an operand or result leaves `i32`.
Compute the closed-composition scan in `Z/2^64` (wrapping 64-bit arithmetic, in any bracketing).
Then for every `j` up to and including the first violation, the 64-bit value read as signed equals the exact `s_j`, because `s_{j-1}` is in `i32` and one step from an `i32` state with `i32` coefficients stays below `2^63` in magnitude; so the first violating index computed from the scan output equals the source trap index, and the value equals the source value when no violation exists.
For sums the cheaper lifted monoid `(total, min prefix, max prefix)` in 64-bit decides "no trap" in a single reduction; only when it reports a violation does a second pass (a 64-bit prefix scan and a min-index reduction) locate it.
Tropical maps (`max` and `+`) satisfy the same argument with exact 64-bit integers for `n < 2^32`.

**Lemma B (residue ring).**
On non-negative operands the language's truncated remainder equals the Euclidean residue, so `x -> x mod P` is a ring homomorphism from the non-negative integers to `Z/P`.
If the seed and every coefficient of an affine (or linear) update with an outer `% P` are non-negative, induction shows every source state is non-negative, and the recurrence may be evaluated in `Z/P` in any bracketing with 64-bit products.
The obligation is a static interval fact or an `O(1)`-per-scan runtime guard on seeds and coefficients; it also requires that the source combine cannot overflow, which is a static bound `sum_j |c_j| * max(child) + k < 2^31` from interval facts.
A root that resolves to a leaf through tail calls returns the raw representative, not its residue; the path-sum form keeps that case (section 2, L9).

### 1.5 Work and span effect per shape

| Shape | Span before | Span after | Work after |
| --- | --- | --- | --- |
| monoid fold over `n` | `n * c_body` | `c_body + ceil(log2 n)` | `W` |
| closed composition scan | `n * c_body` | `c_body + 2 ceil(log2 n)` | `W + n * |rep|` (2 for affine, 3 for lifted sums, `|S|` for an FSM over `S` states, `k^2` for `k x k` matrices) |
| path-sum over a tree of depth `d` with `L` leaves | `d * (c_pro + c_comb)` with frames and joins | `d * c_pro + c_leaf + ceil(log2 L)` | `W` plus the replayed prologue: each leaf lane re-executes its root-to-leaf prologue, so `c_pro` is paid `sum over leaves of depth` times (`L d` for a balanced tree, about `d / 2` times the `~2 L` node evaluations of the source); no frames |
| scatter-combine | `n` (if conflicting) | `ceil(log2 n)` (or atomics) | `W` |
| monotone frontier round | `n * c_body` | `c_body + ceil(log2 n)` | `W + n` |

Span is not the objective by itself: once `W/p` dominates, a law that adds work makes the program slower.
The cost model in section 6.4 therefore uses `W/p + S + L * c_launch + B/bandwidth`, with `L` dependent dispatches and `B` bytes moved.
For L9 the work term `W` includes the replayed prologue of the path-sum row above, so the model can prefer the residual fork-join form, which shares each prefix, when `d` is large and `W/p` dominates; no mechanism that shares the prefix across lanes is proposed here.

## 2. Law catalogue

"Static" obligations are checked from the net; "guard" obligations are runtime checks before the region publishes anything, with the unrewritten region as the fallback.
Every law certificate names its law id, representation and guard, and the verifier re-derives it from the net.
The trap column says how source-order traps and the GNET observation contract v2 (D3) are preserved; the f32 column says what happens to f32 under the default float contract and under `strict-sequential` (Amendment A).
"v0.1" means in the proposed v0.1 cut (section 10).

| Law | Algebraic statement | Proof obligation and certificate | Net rewrite | Span effect | Traps / f32 | Cut |
| --- | --- | --- | --- | --- | --- | --- |
| L1 map | iterations with disjoint footprints commute | affine footprints: every store index iteration-injective (`i`, `a*i + r` with loop-invariant `a != 0`, or block ownership `i*K + r` with `0 <= r < K` from inner extents); reads of stored arrays stay in the iteration's block; no carried scalar | `ForNext` loop region to `map(i, body)` | `n c -> c + 1` | first trap = min failing index (#132 rule); f32 exact (no reassociation) | v0.1 (#132) |
| L2 sealed wrapping fold | `(Z/2^32, +, 0)` and `(Z/2^32, *, 1)` are commutative monoids | static (op is `wrapping_add` / `wrapping_mul`, existing `laws::Law`) | fold to `reduce(law)` | `n -> log n` | no traps; not f32 | v0.1 (exists) |
| L3 trap-lifted checked fold | checked `+`/`*` on `i32`/`u32` is the restriction of `Z`; lift to `(total, min prefix, max prefix)` in 64-bit (sum) or saturated magnitude with zero flag (product) | static op and type; Lemma A; certificate records the lifted representation | fold to `reduce(lift)`, then a rare-path `scan + min-index` | `n -> log n` (twice on the trap path) | exact value and first trap index (lawcheck: 3,000 cases each, 0 mismatches); not f32 | v0.1 |
| L4 affine recurrence | `s -> a s + b` closed under composition over a ring | ring by op: wrapping `Z/2^32` (existing M2-04), checked via Lemma A, or residue `Z/P` via Lemma B (guard: non-negative seed and coefficients) | carried update to `scan(affine) ; map(apply)` | `n -> 2 log n` | exact trap index and value (Lemma A); f32 affine is **not** a law (rounding; no ruling defines an f32 scan, Amendment A) | v0.1 |
| L5 matrix recurrence | `k` jointly linear carried variables compose as `k x k` matrices over the ring | strongly connected carried set whose updates are jointly linear with loop-invariant or data coefficients; same ring for all | `scan(matrix)` | `n -> 2 log n`, work `x k^2` | as L4; f32 not exact | v0.1 for `k <= 4` integer, later otherwise |
| L6 tropical | `s -> max(s + a, b)` closed under composition | update statements are `+ data`, `max`/`min` with data, or reset; guards independent of `s` | `scan(tropical)` | `n -> 2 log n` | exact via 64-bit (Lemma A); f32 `+` inside is not exact | v0.1 integer |
| L7 min / max / argmin | leftmost minimum under a strict weak order is associative; NaN elements are the identity; the seed is applied once on the left | static compare-and-select shape; companion variables assigned under the same guard become the argument | `reduce(argmin)` | `n -> log n` | integer exact; **f32 exact** with NaN lifting and signed-zero preservation (lawcheck: 3,000 cases, 0 mismatches) | v0.1 |
| L8 finite-state map | functions `[S] -> [S]` compose; the update `s := T[s*k + c]` is such a function | guard: every table entry `< |S|` (one map over the table); `|S|` bound for work | `scan(fsm vector)` | `n -> 2 log n`, work `x |S|` | index traps checked by the guard; not f32 | later (cost model needed) |
| L9 path-sum tree | for argument-determined recursion with combine `(sum_j c_j r_j + k) % P`, the result is the path-weighted sum of node constants and leaves | static: self-call arguments independent of self results, constant `c_j`, one modulus, no-overflow bound; guard: non-negative leaves and arguments (Lemma B); root-chain case returns the raw leaf | recursive function to `paths(prologue) ; map(leaf * weight) ; reduce(+ mod P)` | `d (c_pro + c_comb) -> d c_pro + log L`, no frames or joins; work grows by the replayed prologue (section 1.5) | exact (lawcheck: 3,000 random trees, data-dependent splits, 0 mismatches); trap order: the source combine cannot trap (static bound), prologue traps resolved by min path in source order | v0.1 |
| L10 range-split sum (#132 D4) | special case of L9 with `c_j = 1`, no modulus, sealed or trap-lifted `+` | static | as L9 | as L9 | as L3 | v0.1 (D4) |
| L11 scatter-combine | AC combine (`wrapping_add`, checked `+` via L3, min, max, or) commutes across writers | static update shape `A[k] = A[k] op e`; no other access | `scatter_reduce(op)` | conflicts `n -> log n` | checked `+` per target via L3; f32 `+` is a float atomic, allowed only under `unordered` (Amendment A) | v0.1 (histogram family) |
| L12 ordered scatter | last writer in source order = maximum writer index per target | static: no reads of the target in the loop | `scatter_max_index ; gather` | `n -> log n` | exact; f32 values are moved, never combined | v0.1 |
| L13 compaction | store index is a counter that increments exactly when the store happens | static (counter class, same guard) | `scan(+) ; scatter` | `n -> 2 log n` | counter cannot overflow (bounded by trip count) | v0.1 (M4-04) |
| L14 monotone frontier | flags set to one constant, writers determined by round-start state: reader at `u` sees `old or (active(v) and v <= u)` | static: constant stores, writer predicate reads only round-start state | round to `map(active) ; map(reconstruct)` | per round `n -> log n` | reconstructs the exact `next` array and the source step count (lawcheck: 3,000 rounds, 0 mismatches) | v0.1 |
| L15 first exit | early return at the first satisfying iteration = min index | static: body otherwise a map | `map(pred) ; reduce(min index)` | `n -> log n` | traps in iterations before the exit still win in index order | v0.1 |
| L16 f32 integer-valued fold | f32 `+` is exact on integers while every partial sum is below `2^24` in magnitude | guard: all inputs integral with `n * max|x| < 2^24` | as L2 | `n -> log n` | exact, else fallback to source order; only meaningful under `strict-sequential`, since the default tier already gives the fold its tree meaning (Amendment A) | later |
| L17 f32 deterministic tree | **not a law**: the default meaning of an f32 fold, the pairwise tree of `sum_tree` fixed over index order, with a shape that depends only on the length (Amendment A) | default tier; the oracle computes the same tree (`FoldTier::reference`, `docs/numeric-v2.md`) | `reduce(f32 +, pinned shape)` | `n -> log n` | bit-exact against the oracle on every backend; differs from a source-order loop, which `strict-sequential` keeps | decision F1 as amended (#321 ruling 1) |
| L18 exact f32 sum | **not a law**: `sum_exact`, the correctly rounded mode of `fsum` (#173), which stays available (Amendment A) | user opts in (the `exact` tier or the call); superaccumulator or binned representation | `reduce(superacc)` | `n -> log n` | order-independent by construction (lawcheck: 2,000 shuffles, 0 changes) | decision F1 as amended |
| L19 map fusion | `map f . map g = map (f . g)` when footprints allow | static | fuse adjacent regions | removes one dispatch and one round trip of the intermediate | exact | v0.1 (M3-03 laws) |
| L20 fork-join (residual) | independent pure calls commute | `speculation_analysis` facts | region T of #132 | `sum -> max` of siblings | ABI5 source-ordered commit | v0.1 (#132) |

What no law in the catalogue covers is listed in section 5.

## 3. Parallel normal form and lowering

The PNF is the grammar the law pass produces; every node carries its certificate.

```text
P ::= serial(region)                     -- the frame, compiled, no parallel claim
    | map(i in [a,b), P)                  -- L1, L19
    | reduce(R, i in [a,b), P)            -- L2, L3, L7, L16-L18
    | scan(R, i in [a,b), P) ; P         -- L4, L5, L6, L8, L13
    | segmented(R, offsets, P)           -- reduce/scan per segment
    | scatter(C, idx, P)                  -- L11, L12
    | frontier(round: map ; reconstruct)  -- L14 inside a serial round loop
    | paths(f, depth) ; reduce(R)         -- L9, L10
    | first(i in [a,b), pred)             -- L15
    | forkjoin(P, ..., P)                 -- L20, residual
R ::= a law representation: identity, combine on k words, apply, guard
```

| PNF node | Lowering | Owner |
| --- | --- | --- |
| `map` with a body in a family op set | family map emitter | M4 families via stage "a" |
| `map` with an arbitrary compiled body | body map **M** | GNET (#132 P2) |
| `reduce`/`scan` with `R` a sealed u32 law | family collective | M4-02 |
| `reduce`/`scan` with any other `R` | one generic block-reduce and decoupled look-back scan template, parameterised by `R`'s word count, combine and apply (Merrill and Garland 2016 single-pass scan) | new P8 (section 10) |
| `segmented` | family segmented (M4-04) when `R` is sealed, else the generic template per segment | M4-04 / P8 |
| `scatter` | family histogram / injective scatter; ordered scatter as a `max` writer-index scatter then a gather | M4-07 / P8 |
| `frontier` | family frontier (M4-05) for the dense form, else two maps per round in the host frame | M4-05 / P2 |
| `paths ; reduce` | one path-replay kernel (lane = digit string, as #132's path-replay T, but without the continuation machine; each lane replays its prologue, so work grows as in section 1.5) plus a generic reduction | new P4a |
| `first` | a map writing per-lane predicate status plus a min-index reduction (the existing trap-resolution kernel) | P2 |
| `forkjoin` | task region **T** | #132 P4 (now P4b) |
| `serial` | host frame (D2) or device frame | #132 P5 |

This replaces "one MIR-region recognizer per family" (#132 P3) with "one PNF extractor plus a per-family admission predicate on PNF nodes": a family declares which `(node, R, body op set)` it accepts.

## 4. Soundness

**Claim.** If every law certificate in a PNF verifies against the current net, and every runtime guard either holds or selects the fallback before any publication, then for every input the parallel program's observation under contract v2 (D3: status, value, first source-order trap and its span, source steps, cleanup prefix, and the Demand/Complete sequence reconstructed in verification mode) equals the CPU oracle's observation of the original net.

**Argument**, by induction over the PNF tree, with each node's obligation:
1. *Value.* Each law's algebraic statement gives equality of the computed values for all inputs that satisfy its obligation (sections 1.3 and 1.4); guards make the obligation hold or route to the fallback, which is the unrewritten region.
2. *First trap.* Laws that regroup checked operations (L3 to L6, L9 to L11) compute in a wider exact ring and search for the first violation in source index order (Lemma A), or prove statically that no violation exists (L9's combine bound, L13's counter bound); maps and paths resolve traps by the minimum source-ordered index, the same rule as #132 §3.
   Because the source evaluator stops at the first trap and the parallel form never publishes beyond the minimum failing index, the published prefix is the source prefix.
3. *Source steps.* Source fuel is additive over iterations and paths; each lane records its own step count, and the region's step total is a sum reduction, exact in 64-bit.
   L14 reconstructs data-dependent branch outcomes (which `next` stores the source executed), so its step count is exact too, not an upper bound (lawcheck verifies steps, not only values).
   Fuel cut inside a region follows #132: prefix sums locate the iteration where fuel ends, and only that iteration replays serially.
4. *Cleanup and effects.* Laws apply only to regions whose certificate shows no `Host` effect and no owner escaping an iteration (the #132 region legality); cleanup happens in the frame at the region boundary in source order.
5. *Demand/Complete sequence.* In verification mode every PNF node keeps the map from its parallel index to the source position (iteration index, or path digits in lexicographic order for L9 and L20), so the source-order event sequence is reconstructed after the fence, as #132 §7 requires; production runs are trace-off variants compared against traced ones.
6. *f32.* No law reassociates f32 arithmetic; L7 only selects among existing values; L16 is a guarded exactness fact under `strict-sequential`; L17 and L18 give f32 folds a defined meaning (the default tree and the correctly rounded sum) with their own oracle semantics, so a tree lowering is the program's meaning and not a rewrite of source order.
   `a * b + c` is one fused operation in the oracle and on every backend (Amendment A).

The prototype's `lawcheck.py` exercises points 1, 2 and 3 differentially for L3, L4 (residue and checked), L6, L7, L8, L9, L12 and L14 on 3,000 random cases each, with trap-heavy inputs (918 trapping sums, 887 trapping Kadane runs, 2,023 trapping affine runs) and 0 mismatches; a mutation that reports the last instead of the first violation is detected in 500 of 500 cases.
This is evidence for the algebra, not a proof of an implementation; section 10 adds the certificate verifier, mutation and fuzzing obligations for the real pass.

## 5. Completeness limits: what no law parallelizes

1. **Non-closed state updates.** If the carried update's function space is not closed under composition with a bounded representation, no scan exists: the Collatz-style `v := v even ? v/2 : 3v+1` (GNET-05 serial family), polynomial updates of degree above one (`s := s*s + x` doubles the representation per composition), and updates whose branch depends on the state.
   Only speculation can help those, and only when the reachable state set is small (Prabhu, Ramalingam and Vaswani, PLDI 2010; Mytkowicz et al. 2014 enumerate all states); otherwise they stay sequential, and P-completeness results for iterated function evaluation suggest no general escape.
2. **Dependent call chains.** `f(g(h(x)))` with data flowing through every call has span equal to its work; this is the serial family's chain shape.
3. **Recursion whose control reads child results**, or whose combine is non-linear in the children (`r1 * r2 % P`, `if r1 > r2 { recurse }`), has no path-sum; it keeps fork-join T, with its 2x to 3x form cost (#132 spike 1).
4. **`strict-sequential` f32 folds**: source-order accumulation has exactly one bit-exact order, so under that opt-in tier it parallelizes across independent accumulators only; the default tree tier has no such limit (Amendment A).
5. **Algorithmic substitutions**: laws preserve the algorithm's operation multiset up to regrouping; they never turn a quadratic neighbour scan into a grid sort, a linear search into a BVH, or a dense frontier into a work-efficient sparse one.
   Handwritten baselines that use a different algorithm (allowed by `docs/cuda-parity-v01.md`) are reachable only through new families or user-level algorithms.
6. **Effects and ownership**: regions with `Host` effects, owner escape or aliasing `mut` arguments are excluded by the region legality of #132 before any law is considered.
7. **Work-inefficient laws**: L5 with large `k` and L8 with large `|S|` are legal but multiply work by `k^2` or `|S|`; they are only chosen when `W/p` is not the bottleneck, so some legal laws are deliberately not used.

## 6. Inference: how laws are found

### 6.1 Primitive operations (automatic, sealed)

A closed table maps each MIR operation and intrinsic to its algebraic properties: `wrapping_add` (commutative group on `Z/2^32`), `wrapping_mul` (commutative monoid), checked `+`, `-`, `*` (restrictions of `Z`, liftable by Lemma A), `%` with a literal modulus (ring homomorphism on non-negatives, Lemma B), compare-and-select (lattice operations), `and`/`or` (Boolean semiring).
These are sealed in the compiler with proofs in the repository (the existing `laws.rs` pattern, extended with rule revisions); nothing is inferred about them at compile time.

### 6.2 Loop-carried state (automatic, symbolic)

The pass works on the MIR of a counted region (counted `for` and `while i < E { ...; i = i + 1; }` with invariant `E`), as the prototype does:
1. Inline body-local single-definition lets, so updates are expressed over carried variables; a local reassigned under carried-dependent control makes the update opaque.
2. For each carried variable, classify every update statement by **linear-form extraction**: `s := c*s + d` with `c`, `d` free of carried state (or the guarded compare-and-select of L7), and classify its guards (they must not read the carried state except as a min/max test).
3. Join statement classes in a small lattice (`const < add < affine`, `max + add = tropical`, ...), per ring (wrapping, checked, residue, f32).
4. Build the dependency graph among carried variables; strongly connected sets must be jointly linear (L5), acyclic dependencies become successive scans (Kadane's `best` reads `s`'s scan output), and argmin companions attach to their min variable.
5. Classify every array by footprint: iteration-injective, block-owned, AC-combined, constant-monotone, counter-indexed or last-writer.

This is the SYMPLE idea (Raychev, Musuvathi and Mytkowicz, SOSP 2015: symbolic summaries of a chunk with unknown input state) restricted to summaries that are linear forms, which keeps it decidable and linear-time in the region size.
Prior systems that detect such recurrences: Fisher and Ghuloum (PLDI 1994), Sato and Iwasaki (PLDI 2011, semiring matrices including max-plus), and the function-reconstruction approach of Jiang and Agrawal (PPoPP 2018) and Xia et al. (CC 2019).

### 6.3 User functions and lifting

Three sources of laws for user-defined combines, in order of preference:
1. **Inferred from the body** when it is built from sealed primitives (section 6.2 applies to recursive functions too: L9 is linear-form extraction over self-call results).
2. **Synthesized lifting**, offline and certified: for a non-homomorphic loop, search for auxiliary state that makes the join well defined, as Farzan and Nicolet do with syntax-guided synthesis (PLDI 2017, PLDI 2019; phased synthesis PLDI 2021) and Morita et al. derive list homomorphisms from a forward and a backward program via the third homomorphism theorem (Gibbons 1996; PLDI 2007).
   The synthesized join is a candidate, not a proof: it becomes a law only after an SMT check of associativity and of `join(h(xs), h(ys)) = h(xs ++ ys)` over the bit-precise types (bounded for f32, exact for integers), and its certificate stores the checked formula.
   This runs in the toolchain, not in the compile path of v0.1.
3. **Declared sealed laws**: a user or library declares `law assoc(f)`, `law identity(f, e)`, `law linear(f)` on a function, with a machine-checked proof obligation discharged by the same SMT check, failing compilation if it does not hold.
   This is how libraries (quaternions, spatial transforms over exact types, interval arithmetic) get scans without inference.

### 6.4 Bounded search with a cost model

Most regions have one or two lawful forms, so v0.1 needs no search: a deterministic rule order (L1, then the cheapest representation that applies) with the cost model below picks one.
An e-graph becomes useful when laws interact (fusion against reassociation, a scan inside a map against a segmented scan, different modulus factorings).
The plan for later is an egglog program (Zhang et al., PLDI 2023) over PNF terms: rewrite rules are the catalogue, e-class analyses carry interval and type facts (egg's e-class analyses, Willsey et al., POPL 2021), and extraction minimises

`cost = W/p + c_step * S + c_launch * L + B / bandwidth + c_frame * F`

with `L` dependent dispatches, `B` bytes moved and `F` task frames, calibrated by M5-01 (#121) per target.
Equality saturation is bounded by node and iteration limits (the M4-06 compile budget), AC rules are applied as rewrites with a limit on reorderings rather than by AC matching, and proof certificates come from egg-style explanations (Flatt et al., "Small Proofs from Congruence Closure", FMCAD 2022), so each extracted form carries the chain of laws that justify it.
Cranelift's acyclic e-graph mid-end shows the production pattern: e-graphs as a bounded optimisation layer, not the IR of record.
The search is step 2 of #279: a bounded e-graph search beside the net (decisions E1 and M1), where the interaction net stays the IR of record and every extracted form is re-certified by the independent checker.

## 7. Empirical analysis

### 7.1 Method

A throwaway prototype outside the repository (`$HOME/.cache/invar-net-laws/`, mode 700, about 2,100 lines of Python):
- `lang.py` parses the Invar subset the corpora use; `engine.py` executes it with checked `i32`/`u32`, wrapping intrinsics and arrays, and agrees with the GNET-05 reference values on **240 of 240** recorded inputs and with all **140** M5-01 held-out oracle programs (each returns 0).
- `laws.py` is the static inference of section 6.2 (counted loops, linear forms, class lattice, carried-variable SCCs, footprints) plus the L9/L10 recursion classifier; it reads only the AST, never file names or ids.
- The engine runs each program once and keeps one strand clock per model, so all models see identical semantics:
  - **ser**: source order, span = work.
  - **a, today's GNET**: serial root continuation, loops serial, independent same-block call siblings offered to 8 speculative slots (the test configuration; the CLI allows 2), children never fork.
  - **b, the region design (#132)**: maps and nested maps, sealed wrapping folds, M2-04 wrapping affine scans, checked counters with constant increments (static guard), the families it lists (compaction, histogram, injective scatter with a runtime injectivity check, dense frontier, and the min/argmin reductions #132 §8 claims), fork-join T for independent calls, and the D4 range-split sum.
  - **c, this proposal**: b plus L3 to L15.
  - **ct**: c plus a pinned f32 tree (L17), to show what the f32 contract costs.
    After Amendment A, ct is the figure for the default float contract, and a, b and c are the figures under `strict-sequential`; nothing was re-measured.
- A parallel construct restarts each branch from its start time and ends at the latest branch plus its combine depth (`ceil(log2 n)` per reduced variable, twice for scans), so spans are series-parallel spans in unit agent steps.
- Under c, sibling self-calls inside a path-sum tree are timed like a fork because each path is an independent lane of the flat replay; the combine is moved off the path (one weight multiply per edge) and the root pays one reduction of depth `ceil(log2 L)`.
  A separate check confirmed that every fork c executes on the GNET-05 probes is inside a path-sum tree, so no c run needs a task frame or join continuation.
- Inputs: the 4 recorded inputs per GNET-05 program plus one larger off-corpus probe input per family (marked "probe"), the 140 M5-01 held-out candidates at their oracle size and at a scaled size (4,093 elements, or the larger shapes in section 7.3), eight robotics sketches written in the same subset (with `sin`, `cos`, `sqrt` standing for the P0a intrinsics at 20 agents each) and twelve law probes.

Limits of the method: spans are unit-cost agent counts, not time; the a model ignores quanta and submission overhead (it flatters today's GNET); the engine does not model memory traffic or launch latency, which is why section 7.6 discusses form separately; the engine's `W` is source-order work and does not model the prologue replay of section 1.5, so `W/S` for path-sum programs is relative to source work; and the robotics sketches are mine, not the reviewed GNET-05 robotics workloads, which do not exist yet.

### 7.2 GNET-05 (60 programs, 240 recorded inputs plus 60 probes)

Average parallelism `W/S` (geometric mean; higher is better):

| family | inputs | median W | today (a) | #132 (b) | laws (c) | probe median W | probe a / b / c | path-sum (c) | needs T (b) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| divide | 48 | 277 | 1.70 | 2.96 | 3.38 | 396,993 | 2.5 / 1,005 / 1,121 | 12/12 | 12/12 |
| branch | 48 | 183 | 1.34 | 1.64 | 1.95 | 1,335 | 1.7 / 8.9 / 10.3 | 12/12 | 12/12 |
| frontier | 48 | 219 | 1.13 | 1.98 | 2.06 | 586 | 1.2 / 12.5 / 13.2 | 8/12 | 7/12 |
| compose | 48 | 488 | 1.81 | 3.18 | 4.18 | 7,254 | 2.0 / 27.9 / 34.6 | 12/12 | 12/12 |
| serial | 36 | 111 | 1.00 | 1.00 | 1.08 | 122 | 1.0 / 1.0 / 1.1 | 0/12 | 0/12 |

Probe inputs: divide `(20000, 30)`, branch `(12, 517)`, compose `(3000, 14)`, frontier BFS `(4999, 21)` (5,004 vertices, 5 rounds) and recursive discovery `(40, 23)` (depth 6, the generator's maximum), serial `(900, 3)`.

Findings:
1. **The recorded inputs are too small to show device parallelism.** Median work is 111 to 488 agents and the best average parallelism on any family is 4.2; at these sizes placement must choose the CPU, and the GNET-05 timing gate cannot be about GPU parallelism unless the inputs grow.
2. **The region design (b) already exposes most of the span** of divide, branch and compose: at the probe inputs it reaches parallelism 1,005 on divide against 2.5 for today's GNET.
   The law model (c) cuts span by a further 1.11x (divide), 1.15x (branch), 1.06x (frontier) and 1.24x (compose) in geometric mean, up to 2.07x on single programs.
3. **The real difference is form, not span.** Under b, every divide, branch and compose program and 7 of 12 frontier programs need fork-join task frames and join continuations (median 14,842 task frames per probe run for divide, up to 27,586); under c, 44 programs (all divide, branch and compose, and the 8 recursive-discovery frontier programs) are **path-sum trees** with no frames and no joins, and the 4 BFS programs are monotone frontier rounds.
   That is the form #132 spike 1 identified as the parity gap (path-replay T at 1.9x to 2.7x of a flat kernel, level-synchronous T at 3.4x to 56x).
   Every GNET-05 combine is linear modulo a literal prime (`(a*5 + b + c*2) % 100003`, `(v0*7 + v1 + v2) % 10007`, `(w + v) % 10007`), and every split point and branch selector depends only on arguments, so L9 applies to all of them.
4. **Programs where only laws help** (c at least 20% below b on some input): 16 programs, all through L9 (path-sum trees whose combines dominate the path) or L4 over `Z/P` (the `while i < x % 17 { s = (s * 12 + i) % 10007; ... }` leaf loops of branch, compose and two serial programs).
   No GNET-05 program needs speculation.
5. **Programs that stay sequential under every model**: 10 of the 12 serial programs (Collatz-style data-dependent `while` loops and dependent call chains, completeness limits 1 and 2), and the 4 recursive-discovery frontier programs whose probe trees are chains (depth at most 6, so their work is tiny, not inherently sequential).
   Serial-03 and serial-10 gain only on a leaf loop of at most 8 iterations.
6. **Generator bias.** Because every combine in the corpus is linear over a residue ring, the corpus contains no case that keeps fork-join T after all laws (completeness limit 3), which is the case the D5 amendment is about; section 11 recommends adding such cases.

The per-program table at the probe input is in Appendix B.

### 7.3 M5-01 held-out candidates (140 programs)

Scaled shapes: map, reduction and scan at 4,093 elements; compaction 10,000; transpose 127 x 257; stencil 129 x 129; matrix 64 x 64 x 64; segmented 64 segments of 64; gather, injective scatter and modular histogram at 4,093; frontier at its oracle size (the vertex count is a source literal).

| family | programs | median W | a | b | c | ct | b and c differ |
| --- | --- | --- | --- | --- | --- | --- | --- |
| map | 28 | 176,071 | 1.0 | 283 | 643 | 643 | 7 |
| reduction | 28 | 16,382 | 1.0 | 292 | 292 | 292 | 0 |
| scan | 28 | 28,676 | 1.0 | 181 | 216 | 216 | 4 |
| compaction | 8 | 110,651 | 1.0 | 1,702 | 1,702 | 1,702 | 0 |
| transpose | 8 | 228,734 | 1.0 | 14,296 | 14,296 | 14,296 | 0 |
| stencil | 8 | 748,594 | 1.0 | 13,849 | 13,849 | 13,849 | 0 |
| matrix | 8 | 2,646,151 | 1.0 | 4,040 | 4,040 | 85,360 | 0 |
| segmented | 8 | 16,838 | 1.0 | 842 | 842 | 842 | 0 |
| gather / scatter | 8 | 16,380 | 1.0 | 1,135 | 1,135 | 1,135 | 0 |
| frontier | 8 | 421 | 1.0 | 2.3 | 2.3 | 2.3 | 0 |

Findings:
1. **Every kernel in the held-out set reaches the same span under b and c.** The 11 programs where they differ are boundary witnesses whose verification code in `main` contains `for q in ... { if answer[q] != wanted[q] { return 1; } }`; c parallelizes that check with L15 (first exit), b keeps it serial (61 such loops in the set).
2. **Today's GNET path runs all of them serially** (it parallelizes only call siblings); the M4 families, not GNET, carry this corpus.
3. **Matrix is the f32 boundary.** Under strict-f32-v1, GEMM keeps each output's inner product in source order, so b and c reach 4,040; a pinned tree reaches 85,360 (21x lower span) but computes a different function.
   After Amendment A the pinned tree is the default meaning, so 85,360 is the default-contract figure and 4,040 is the `strict-sequential` one.
4. The M5-01 set is family-shaped by construction, so it cannot distinguish a law-directed extractor from per-family recognizers; its value here is as a regression check that the law pass finds every family form (it does, for 140 of 140).

### 7.4 Robotics sketches (batch 1,024 and 8,192)

| case | batch | W at 8,192 | a | S b | S c | S ct | laws (c) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| R1 FK, 6 joints | 1,024 / 8,192 | 8,617,987 | refused (f32) | 1,055 / 1,055 | 1,055 / 1,055 | 292 | map; joint chain is an f32 matrix product (ct only) |
| R2 Jacobian | 1,024 / 8,192 | 13,688,835 | refused | 1,279 / 1,279 | 1,279 / 1,279 | 366 | map, block-owned scratch frames |
| R3 CRBA-shaped mass matrix | 1,024 / 8,192 | 27,344,899 | refused | 999 / 1,002 | 999 / 1,002 | 189 | map; backward pass is f32 affine (ct only) |
| R4 sphere distance, min | 1,024 / 8,192 | 22,630,434 | refused | 87 / 93 | 87 / 93 | 93 | map, f32 min (L7) |
| R5 rollout cost, argmin | 1,024 / 8,192 | 10,567,704 | refused | 1,329 / 1,338 | 1,329 / 1,338 | 1,338 | map; horizon non-linear (sequential); argmin (L7) |
| R6 cloud transform, filter, compact | 1,024 / 8,192 | 252,074 | refused | 113 / 122 | 113 / 122 | 122 | map, counter, compaction |
| R7 neighbourhood (512 points, quadratic) | 512 | 5,247,915 | refused | 38 | 38 | 38 | map, counter |
| R8 composed FK, collision, rollout | 1,024 / 8,192 | 41,816,130 | refused | 2,476 / 2,491 | 2,476 / 2,491 | 1,728 | all of the above |

Findings:
1. **Batch maps saturate the device**: span is constant in the batch (1,055 at 1,024 and at 8,192 configurations for FK), so the region design's body map M is the whole story, and laws add nothing to b except exactness for the f32 reductions.
2. **The region design's §8 claims two f32 reductions it cannot have under strict-f32-v1**: "family reduction for cost" sums f32 costs (a source-order fold, sequential per accumulator), and "family min-reduction" over f32 needs L7's NaN lifting to be exact.
   Per-rollout cost sums stay sequential inside each lane (harmless, the batch is parallel); a sum across rollouts would need L17 or L18.
   Under Amendment A that sum is a reduction on the default tree, so it is a family reduction again.
3. The joint chain and the composite-inertia pass are f32 matrix and affine recurrences; reassociating them (ct) shortens span 3.5x to 5.4x but changes rounding, and with batches of thousands it would only add work.
   Handwritten FK kernels do not reassociate the chain either (cuRobo parallelizes across links of a batch element for the transforms, then multiplies in order).
   The Amendment A rulings define reductions, not these recurrences, so the chain and the backward pass stay serial per lane.
4. The neighbourhood case stays quadratic: a grid or sort is an algorithmic substitution (completeness limit 5), consistent with #132's "at risk" rating.

### 7.5 Law probes (n = 4,096 and 65,536)

| probe | law | S a at 65,536 | S b at 65,536 | S c at 65,536 | S ct |
| --- | --- | --- | --- | --- | --- |
| P1 checked i32 sum | L3 | 262,149 | 262,149 | 24 | 24 |
| P2 argmin, first index | L7 | 262,171 | 60 | 60 | 60 |
| P3 maximum segment sum (Kadane) | L6 + L7 | 524,929 | 524,929 | 78 | 78 |
| P4 rolling hash mod P | L4 over `Z/P` | 393,221 | 393,221 | 26 | 26 |
| P5 longest collision-free run | L4 (reset affine) + L7 | 491,439 | 491,439 | 78 | 78 |
| P6 contact FSM, 16 states | L8 | 655,365 | 655,365 | 30 | 30 |
| P7 scatter with duplicate targets | L12 | 262,148 | 262,148 (injectivity guard fails) | 23 | 23 |
| P8 IIR filter, f32 | none (f32 affine) | 393,221 | 393,221 | 393,221 | 42 |
| P9 Fibonacci pair mod P | L5 (2 x 2) | 458,757 | 458,757 | 76 | 76 |
| P10 checked prefix sum | L3 scan | 327,685 | 327,685 | 41 | 41 |
| P11 f32 dot product | none | 393,221 | 393,221 | 393,221 | 26 |
| P12 find first | L15 | 262,149 | 262,149 | 8 | 8 |

These are the shapes where only lifting, closed composition or ordered-write laws give parallelism, and where the region design stays sequential.
They are common in robotics streams (single long trajectories, sensor logs, contact detection), not in batched kernels.

### 7.6 What the numbers do and do not say

- For batched robotics and family-shaped kernels, b and c have the same span; the law pass is a generalisation of the family recognizers, not a speed-up.
- For recursive divide-and-conquer and branch trees, b and c have similar span, but c removes all task frames and joins; whether that closes #132's measured 1.9x to 2.7x form gap is **not measured here**.
  The decisive measurement is a Metal and CUDA spike of the L9 lowering (one path-replay kernel plus a weighted reduction) against the flat handwritten kernel on GNET-05 divide and compose shapes at large inputs; it is the first gate of P4a (section 10) and was not run in this pass.
- For single long sequences with carried state, c is the only model with logarithmic span.
- Under strict-f32-v1, strict f32 folds bounded everything: no model parallelized P8 or P11 without a contract change.
  Amendment A is that contract change for folds: P11 (a dot product) is a reduction and takes the default tree, while P8 (an IIR filter) is an f32 affine recurrence that no ruling covers.

## 8. Beyond interaction nets: alternative models

The question is whether another mathematical model would induce general parallelism more easily or more effectively than the interaction net.
Each row is judged on what it parallelizes automatically, the cost and decidability of its analysis, irregular and dynamic control, exactness under checked traps and strict f32, published evidence of near-handwritten GPU performance, and fit with Invar.

| Model | Parallelism exposed automatically | Analysis | Irregular / dynamic control | Traps and strict f32 | GPU evidence near handwritten | Fit for Invar |
| --- | --- | --- | --- | --- | --- | --- |
| Interaction nets (Lafont 1990, 1997; HVM2) | all independent redexes (`S_dep`), schedule-free | none needed; strong confluence | natural for recursion and sharing; no arrays or loops in the model | traps and effects need the token order GNET adds; f32 ordered | no: HVM2 74 G interactions/s with up to 100x single-core overhead on loops and arrays | keep as semantic oracle and IR of record (contract v1) |
| SSA and program dependence graphs (Ferrante, Ottenstein and Warren, TOPLAS 1987; Cytron et al., TOPLAS 1991); DSWP (Ottoni et al., MICRO 2005) | dataflow and pipeline parallelism between strongly connected components | polynomial; memory dependences need alias analysis | control dependence explicit; recursion opaque; loops pipelined, not data-parallel | exact if traps are dependence edges | CPU pipelines; no GPU near-handwritten results | already present: the net is a typed dependence graph; use it, nothing to adopt |
| Polyhedral model (Feautrier 1991, 1992; Pluto, Bondhugula et al., PLDI 2008; PPCG, Verdoolaege et al., TACO 2013) | exact affine dependences, parallel loops, tiling, fusion, skewing, GPU mapping | integer linear programming (NP-hard in general, practical for small nests); only static-control parts | none for recursion; data-dependent control only by over-approximation | reorders checked operations freely unless traps are dependences; f32 order kept per statement, reductions need separate detection (Doerfert et al., Polly reductions) | competitive with hand-tuned code on PolyBench stencils and linear algebra; below vendor libraries on GEMM without extra tuning | **beside the net**, as a dependence oracle for affine loop nests (L1 legality, tiling inside family emitters); a full scheduler is a later option for stencils and dense nests |
| Kahn process networks (Kahn 1974); synchronous dataflow (Lee and Messerschmitt 1987); StreamIt (Thies et al., CC 2002) | pipeline and task parallelism between actors; SDF schedules statically | SDF: balance equations, decidable and cheap; general KPN: boundedness undecidable | no recursion; data-dependent rates break SDF | exact per actor | StreamIt on GPUs (Udupa et al., CGO 2009) for streaming kernels | **beside**, at the system layer: Invar's components, ports and flows already are a KPN; not a kernel model |
| Petri nets | concurrency semantics, reachability | reachability is decidable but Ackermann-complete (Czerwinski and Orlikowski 2021; Leroux 2021) | expresses dynamic control, analysis cost prohibitive | no arithmetic exactness model | none for compute | no |
| Recursion schemes and Bird-Meertens (Meijer, Fokkinga and Paterson 1991; Bird 1987; Gibbons 1996; Hu, Iwasaki and Takeichi, TOPLAS 1997) | catamorphisms that are homomorphisms become reductions and scans; hylomorphism fusion removes intermediates; tree homomorphisms become tree contraction | needs the homomorphism (inferred or synthesised; sections 6.2 and 6.3) | the only family here that handles recursion and data-dependent splits (L9) | exact when laws are exact; this proposal's Lemmas A and B | indirectly: once a homomorphism is found, the kernel is a standard reduction or scan | **adopted**: this is the law catalogue and the PNF |
| Monoidal categories and string diagrams | parallel composition is the tensor; interaction nets are a special case | equational, not algorithmic | as the underlying calculus | as the underlying calculus | none | a semantics for laws (laws are 2-cells), no algorithm to adopt |
| Semiring and tensor algebra (einsum; TACO, Kjolstad et al., OOPSLA 2017; MLIR linalg; TVM, Chen et al., OSDI 2018) | dense and sparse contractions, reductions over semirings, fusion, layout | decidable on index expressions; schedule search is the hard part | none (sparse formats cover irregular data, not control) | reassociates f32 by default; exact modes must be pinned | TVM and Ansor near vendor libraries on dense kernels; TACO GPU competitive on sparse kernels | **beside**, through the existing M4 GEMM and tensor families; L5 is its semiring core for recurrences |
| Relational algebra and Datalog (semi-naive evaluation; Souffle, Jordan et al., CAV 2016; GPU Datalog work) | set-at-a-time joins, level-synchronous fixpoints | semi-naive evaluation is polynomial for Datalog | frontiers and graph closure natively; no general recursion over values | exact on sets; aggregates need laws | GPU Datalog engines report large speedups over CPU engines on graph analytics | **borrow the idea**: L14 is semi-naive evaluation of a monotone frontier; no Datalog IR |
| Array and combinator calculi (APL; Futhark, Henriksen et al., PLDI 2017; Dex, Paszke et al., ICFP 2021; Lift, Steuwer et al., ICFP 2015; RISE and ELEVATE, Hagedorn et al., ICFP 2020) | map, reduce, scan, scatter with fusion and flattening; Lift explores rewrite rules to pick GPU-specific forms | type-directed, cheap; Lift's rewrite search is bounded exploration | Futhark and Dex refuse recursion; nested parallelism via flattening (incremental flattening, PPoPP 2019) | Futhark requires the user to promise associativity; reassociates f32 inside `reduce` | Lift reports performance on par with hand-tuned OpenCL on several kernels; Futhark is competitive on many benchmarks | **adopted as the PNF's target vocabulary**; Invar derives the SOACs from laws instead of requiring the user to write them |
| Halide algorithm/schedule separation (Ragan-Kelley et al., PLDI 2013; autoschedulers) | any schedule the user or autoscheduler picks for a fixed algorithm | schedules are legal by construction for its stencil-like algebra | none | f32 order fixed by the algorithm except declared reductions | near hand-tuned image pipelines | **adopted as a principle**: certificates (algorithm) separate from placement and variant choice (schedule), already in #132 |
| Rewriting logic (Maude, Meseguer 1992) | concurrent rewriting, AC matching | AC matching is NP-complete in general | general | as the rules | none for GPUs | no: the e-graph covers search; AC is handled as bounded rewrites |
| E-graphs as the IR (egg; egglog; Tensat, Yang et al., MLSys 2021; Cranelift's acyclic e-graph) | all law-equivalent forms at once, extraction by cost | saturation can blow up; bounded by node and iteration limits; extraction is NP-hard in general | as rich as the rules; e-graphs do not represent effects or control well | exact if rules are | Tensat improves tensor graphs; no general near-handwritten claim | **beside, as the search layer** (section 6.4), never the IR of record |
| Heartbeat and lazy task creation (Acar et al., PLDI 2018; Rainey et al., PLDI 2021) | fork-join with bounded overhead by promoting only when needed | runtime technique | the best known technique for residual fork-join | exact (no reordering) | CPU results; GPUs lack the interrupt mechanism | **beside**: the later promotion step of #132's T |
| Graph DSLs with schedule separation (GraphIt, Zhang et al., OOPSLA 2018; GPU backend, Brahmakshatriya et al., CGO 2021); Gunrock | frontier algorithms with direction, layout and load-balance schedules | schedule search | frontiers yes, general recursion no | exact on integers | near Gunrock and hand-tuned graph code | **borrow**: frontier schedules for M4-05 and L14 later |

**Recommendation: a layered design, not a replacement.**
No single model covers Invar's language: polyhedral and tensor models are exact and fast on affine nests but have nothing for recursion; array calculi need users to write the skeletons and promise associativity; recursion schemes handle recursion only once a homomorphism is known; Datalog handles monotone frontiers only; e-graphs search but do not represent effects and traps; and interaction nets express everything but are, as Proposition 1 shows, bounded by the span of the program as written.
Replacing the net would change the v0.1 contract (the CPU net oracle is its semantic reference) and would discard the one model that already encodes Invar's traps, fuel, ownership and source-ordered observations.
The layered design gets each model's benefit without that change:
1. the **net** for semantics, verification and the oracle (unchanged, contract v1, observations v2);
2. **affine footprint analysis** (a polyhedral subset: injectivity, block ownership, affine dependence) for loop-nest legality, with full polyhedral scheduling as a later option inside family emitters (isl through a reviewed FFI boundary is approved in principle, #321 ruling 3, and needs its own PR with a license check and a review);
3. **recursion schemes and homomorphisms** (the law catalogue, lifting, path-sum) for carried state, recursion and irregular control;
4. **semi-naive frontier evaluation** for monotone frontiers (L14);
5. a **bounded e-graph** to choose between law-equivalent PNF forms under the cost model, later than v0.1;
6. **Halide-style separation**: certificates fix the algorithm, placement picks the schedule.
The empirical data supports this split: M5-01 is covered entirely by layer 2 plus sealed folds, the 44 recursive GNET-05 programs need layer 3 (no affine model applies to them), the 4 BFS programs need layer 4, and the 10 sequential serial programs need none and belong on the CPU.

## 9. Exactness: the honest answer

**Checked integers.**
Every integer law in the catalogue preserves the source value and the source-order first trap exactly.
The price is a lifted representation (three 64-bit words for a sum instead of one 32-bit word, two for an affine map) and, only when a trap happens, a second pass to locate it; the input is still read once, so memory-bound reductions keep their traffic, and the extra cost is registers and arithmetic.
Static no-overflow guards (#132's approach) remain the fast path when interval facts prove them; trap lifting removes the need for them, so checked folds over arbitrary data (P1, P10) parallelize without a guard that could fail.

**Floating point (as amended by Amendment A).**
Under strict-f32-v1 a source loop `acc = acc + x[i]` in f32 had exactly one bit-exact implementation: left-to-right per accumulator.
The prototype found a pairwise tree differing from source order in 1,897 of 2,000 random arrays, so this was not a corner case, and it is why the maintainers replaced the default instead of asking each program to opt out.
The invariant kept is deterministic, specified output, not source-order evaluation.
The ways to get parallel f32, in the order a program meets them:
1. **The default, `deterministic-tree`.** A fold is the fixed pairwise tree over index order (`sum_tree`, L17), whose shape depends only on the length; the oracle computes the same tree, and CPU, CUDA and Metal agree bitwise.
   GPU cost is close to a native reduction because block reductions can follow a fixed pairwise shape, but the result differs from the source-order loop by definition.
2. **Independent accumulators** (batch and per-element maps): bit-exact under every tier, and the dominant robotics case (FK, Jacobian, mass matrix, collision, rollouts are all batch maps).
3. **Exact f32 laws**: min, max and argmin with NaN lifting and signed-zero preservation (L7), selects and compares, and the integer-valued guard (L16, meaningful under `strict-sequential`).
4. **The correctly rounded sum** (`sum_exact`, L18, the exact mode of `fsum`): order-independent by construction, so any parallel shape is bit-exact (0 changes under 2,000 shuffles in the prototype); cost on GPUs is reported as comparable to standard summation for large memory-bound sums (Collange, Defour, Graillat and Iakymchuk, Parallel Computing 2015, ExBLAS), while binned reproducible summation costs about 9n flops for a 6-word accumulator (Ahrens, Demmel and Nguyen, ACM TOMS 2020); both are cheap where bandwidth dominates and expensive inside compute-bound kernels such as GEMM.
5. **`strict-sequential`**, an explicit opt-in: source order, so a parallel f32 reduction is refused, as strict-f32-v1 did.
6. **`unordered`**, an explicit opt-in and the only non-deterministic tier: order-free reductions and float atomics, which are banned under every other tier.
7. **Reduced precision and approximate math** (TF32, BF16, FP8, approximate transcendentals) are explicit types or functions (#328), never silent rewrites; `docs/cuda-parity-v01.md` already allows a non-exact mode only as a separately frozen approximate contract.

**What parity against a handwritten f32 kernel can honestly mean.**
`docs/cuda-parity-v01.md` already says the baseline must preserve the operation's numerical contract and that a baseline implementing a different contract is informative only.
The scored ML suites run under the default contract (#321 ruling 1), so the scored baseline implements the same pinned tree and fuses `a * b + c` as IEEE fused multiply-add; for `sum_exact` it implements the same correctly rounded sum; under `strict-sequential` it keeps the same per-accumulator order and may parallelize across accumulators.
A cuBLAS-style kernel that reassociates freely implements another contract and is reported as an informative comparison, never as the denominator.
The FK spike reached bitwise-identical parity against unfused strict arithmetic, before the multiply-add ruling made fused `a * b + c` the contract on every backend.
A handwritten kernel's freedom to reassociate is a different contract, not a faster implementation of the same one.

## 10. Revised plan

### What changes in #132

- **P1** (region partition and certificates) splits into P1a (unchanged: regions, footprints, `RegionCertificate`) and **P1b: PNF extraction and law certificates** (sections 2, 3, 6.1, 6.2), which produces the PNF for every legal region.
- **P3** (family bridge) shrinks: instead of a MIR-region recognizer per family, each family declares an admission predicate over PNF nodes; the AST-agreement tests stay.
- **P4** splits into **P4a: path-sum and chain recursion** (L9, L10: path-replay kernel plus generic reduction; its first gate is the Metal and CUDA spike of section 7.6) and **P4b: residual fork-join T** (#132's ABI6 scope, reduced to fixed-width expansion and nested forks, since laws remove most of its use).
- **New P8: generic law-representation collectives**: one block-reduce and one single-pass decoupled look-back scan template, parameterised by a law representation (word count, combine, apply, identity, guard), on CUDA and Metal, plus the trap-lifting rare path.
- **P9, the default float contract (amended 2026-10-21; tracked in #280)**: the fixed tree, fused multiply-add and `sum_exact` as the meaning of f32 folds, with oracle semantics and the fixture migration (decision F1 as amended).
- **New P10: bounded e-graph search** over PNF terms (section 6.4, #279) and offline lifting synthesis (section 6.3, #281), both after the v0.1 cut.

### Phases (estimated new or changed lines, code / tests)

| Phase | Content | Code / tests | Depends on | Cut |
| --- | --- | --- | --- | --- |
| P0a, P0b | as #132 (trig and sqrt contract; f32 and array entry) | 1,800 / 1,400 | D6, HA-1 | v0.1 |
| P1a | regions, footprints, region certificates | 1,200 / 900 | #99, #114 | v0.1 |
| P1b | PNF extractor: counted regions, linear forms, class lattice, carried SCCs, footprint classes, recursion classifier; law certificates and verifier; sealed primitive law table (L1 to L7, L9 to L15) | 2,200 / 1,800 | P1a | v0.1 |
| P2 | body map M and host frame (as #132), plus `first` and frontier rounds | 1,400 / 1,000 | P1a | v0.1 |
| P3 | family admission predicates on PNF nodes; AST-agreement differential tests | 1,100 / 900 | P1b, stage "a" | v0.1 |
| P8 | generic law-representation reduce and scan templates (CUDA, Metal), trap-lifting rare path | 1,000 / 800 | P1b | v0.1 |
| P4a | path-sum lowering: path-replay kernel, weighted reduction, raw-root case; gate: spike against the flat handwritten kernel | 900 / 800 | P1b, P8 | v0.1 |
| P4b | residual fork-join T: nested forks, fixed-width expansion, arena, joins | 1,400 / 1,100 | P1a | v0.1 |
| P5, P5b, P6, P7 | as #132 | 1,900 / 1,200 plus evidence | as #132 | v0.1 |
| P9 | default float contract: tree folds, fused multiply-add, `sum_exact`; oracle, emitters and fixture migration (#280; the line estimate predates the ruling) | 900 / 800 | F1 | amended 2026-10-21: the default contract, tracked in #280 |
| P10 | egglog search over PNF, offline lifting synthesis with SMT certificates, L8 FSM and large-`k` L5 under the cost model | 2,000 / 1,500 | P1b, M5-01 calibration | later; scheduled in #281 and #279 |

v0.1 total: about 12,900 lines of code and 9,900 of tests (against 9,700 and 6,500 in #132), the increase being P1b, P8 and P4a.

### Risks

1. **Inference soundness.** A wrong law silently changes results.
   Mitigations: certificates re-derived by an independent verifier; every law has a differential test against the oracle with trap-heavy generators (the prototype's `lawcheck.py` is the seed); a named mutation per law (drop a guard, take the last violation, wrong representation identity, skip the raw-root case, reorder an f32 fold) that must be killed.
2. **Form cost not yet measured.** Path-sum's advantage over T is argued from spike 1 but not measured; P4a's spike is a gate, and if it fails, D5's T tier applies to those programs.
3. **Work inflation.** Lifted representations and matrix or FSM scans add work, and path-sum replays each prologue along every root-to-leaf path; the cost model must refuse them when `W/p` dominates, which needs M5-01 calibration.
4. **Guard fallbacks.** Residue and path-sum laws need non-negativity guards; negative inputs fall back to T or serial, so inputs can change the form (and the timing) of one program; receipts must report the guard outcome.
5. **Corpus bias.** GNET-05's combines are all linear modulo a prime, so the corpus overstates how often L9 applies.
6. **Compile time.** More variants per region; the M4-06 budget of 32 variants per region still bounds them.

## 11. D5 threshold (adopted)

The tiered D5 gate below was adopted on 2026-10-15.
D5 (accepted direction: amend the parity gate for general dynamic categories) needed a threshold; the data suggests the threshold should depend on the **normal form the compiler certifies**, not on the benchmark family name:
1. **Law-reachable flat forms** (any PNF without `forkjoin`: maps, reductions, scans, path-sum, frontier rounds): **no amendment**; the standard gate applies (point estimate `R <= 1.00`, upper 95% bound `<= 1.05`, both scopes).
   On the current GNET-05 corpus this covers 48 of 60 programs under the law model.
2. **Residual fork-join** (programs whose PNF still contains `forkjoin` after all laws, for a reason the explain output names: non-linear combine, control reading child results, guard failure): an amended gate of **`R <= 3.0` resident and `R <= 2.0` host-to-host** (point estimates, with upper 95% bounds at `3.15` and `2.10`), from #132 spike 1's path-replay T (1.9x to 2.7x resident, 1.1x to 2.1x host-to-host up to 2^24 elements), reviewed again with P4b data.
   Today GNET-05 contains no such program, so this tier needs new cases.
3. **Sequential programs** (`W/S < 2` under the law model at the scored input, such as 10 of the 12 serial programs): not a GPU parity case; placement must choose the CPU, and the gate is CPU-native parity against a handwritten CPU baseline plus the existing dispatch-overhead gate.
4. The tier is decided by the compiler's certificate and explain output, frozen before timing, never chosen after measurement, and a program cannot move to a looser tier by failing a guard on purpose: guard outcomes are part of the scored input set.
5. Add held-out GNET-05 cases (through a reviewed manifest bump) for tier 2: non-linear combines (`a * b % P`), child-result-dependent recursion, and negative inputs that fail the residue guard; and grow the recorded inputs, whose median work of 111 to 488 agents is too small for any GPU gate.

## 12. Decisions (recorded 2026-10-15)

| # | Decision | Outcome |
| --- | --- | --- |
| L1 | Adopt the PNF and law-certificate layer (P1b) as the way regions are recognized, with per-family admission predicates replacing per-family MIR recognizers (P3) | Adopted |
| L2 | Parallelize checked integer folds and recurrences by trap lifting (exact-then-check in a wider ring, Lemma A), not only under static no-overflow guards | Adopted |
| L3 | Admit the residue-ring (L4 over `Z/P`) and path-sum (L9) laws in v0.1, with runtime non-negativity guards and serial or T fallback | Adopted; the P4a spike gates performance, not correctness |
| L4 | Which laws are in v0.1: L1 to L7, L9 to L15 and L19, L20; L8 (FSM), large-`k` L5, L16 and synthesis later | Adopted |
| F1 | Strict-f32 folds keep source order; add declared `sum_tree` (pinned pairwise tree) and `sum_exact` (correctly rounded) operations: in v0.1, later, or never | Original outcome: later (not v0.1); v0.1 parity for f32 compares against same-order baselines. **Superseded 2026-10-21 (#321 ruling 1, Amendment A):** the fixed tree is the default, `strict-sequential` is the opt-in, `sum_exact` stays as the exact mode, and parity compares against baselines of the same contract, as `docs/cuda-parity-v01.md` already requires |
| F2 | Correct #132 §8: f32 cost sums across rollouts are not a family reduction under strict-f32-v1, and f32 min/argmin needs the NaN-lifted law | Adopted; applied to #132 sections 3 and 8. Under the default tree (Amendment A) an f32 cost sum across rollouts is a family reduction again; the NaN-lifted min and argmin law is unchanged |
| D5 | Threshold by certified normal form: standard gate for flat forms; `R <= 3.0` resident and `<= 2.0` host-to-host for residual fork-join; CPU parity for sequential programs | Adopted as the tiered gate of section 11; thresholds revisited with P4b data |
| C1 | Request tier-2 held-out cases (non-linear combines, child-dependent control, guard-failing inputs) and larger recorded inputs, through a reviewed manifest bump | Adopted |
| E1 | E-graph search and lifting synthesis after v0.1; v0.1 uses deterministic rule order plus the cost model | Adopted; the search is scheduled as #279 step 2 and the synthesis as #281 |
| M1 | Keep the interaction net as the semantic oracle and IR of record; add polyhedral-subset footprint analysis, recursion-scheme laws, semi-naive frontiers and a later e-graph beside it; do not replace the core model | Adopted |
| D10 | Metal binary archives (#132 decision D10) | Yes, in v0.1, after kernel shrinking |

## Appendix A. Sources

Interaction nets and optimal reduction:
- Y. Lafont, "Interaction nets", POPL 1990; "Interaction combinators", Information and Computation 137(1), 1997.
- J. Lamping, "An algorithm for optimal lambda calculus reduction", POPL 1990; A. Asperti and S. Guerrini, *The Optimal Implementation of Functional Programming Languages*, Cambridge University Press, 1998; A. Asperti and H. Mairson, "Parallel beta reduction is not elementary recursive", POPL 1998.
- V. Taelin et al., "HVM2: A Parallel Evaluator for Interaction Combinators" (paper in `HigherOrderCO/HVM2`, `paper/HVM2.typst`): strong confluence, 74,000 MIPS on an RTX 4090, single-core inefficiency section.
- E. Jiresch, "Towards a GPU-based implementation of interaction nets", arXiv 1404.0076.

Homomorphisms, scans and tree contraction:
- R. Bird, "An introduction to the theory of lists", 1987; J. Gibbons, "The third homomorphism theorem", JFP 1996; Z. Hu, H. Iwasaki and M. Takeichi, "Formal derivation of efficient parallel programs by construction of list homomorphisms", TOPLAS 1997.
- E. Meijer, M. Fokkinga and R. Paterson, "Functional programming with bananas, lenses, envelopes and barbed wire", FPCA 1991.
- G. Blelloch, "Prefix sums and their applications", 1990; P. Kogge and H. Stone, "A parallel algorithm for the efficient solution of a general class of recurrence equations", IEEE TC 1973; R. Ladner and M. Fischer, "Parallel prefix computation", JACM 1980; R. Brent and H. T. Kung, "A regular layout for parallel adders", IEEE TC 1982; D. Merrill and M. Garland, "Single-pass parallel prefix scan with decoupled look-back", NVIDIA 2016.
- G. Miller and J. Reif, "Parallel tree contraction and its application", FOCS 1985; A. Morihata, K. Matsuzaki, Z. Hu and M. Takeichi, "The third homomorphism theorem on trees", POPL 2009.

Synthesis and inference of parallel reductions:
- A. Farzan and V. Nicolet, "Synthesis of divide and conquer parallelism for loops", PLDI 2017; "Modular divide-and-conquer parallelization of nested loops", PLDI 2019 (extended: arXiv 1904.01031); "Phased synthesis of divide and conquer programs", PLDI 2021.
- V. Raychev, M. Musuvathi and T. Mytkowicz, "Parallelizing user-defined aggregations using symbolic execution" (SYMPLE), SOSP 2015.
- K. Morita, A. Morihata, K. Matsuzaki, Z. Hu and M. Takeichi, "Automatic inversion generates divide-and-conquer parallel programs", PLDI 2007.
- A. Fisher and A. Ghuloum, "Parallelizing complex scans and reductions", PLDI 1994; S. Sato and H. Iwasaki, "Automatic parallelization via matrix multiplication", PLDI 2011; P. Jiang and G. Agrawal, "Revealing parallel scans and reductions in sequential loops through function reconstruction", PPoPP 2018 (poster); Y. Xia, P. Jiang et al., "Enabling prefix sum parallelism pattern for recurrences with principled function reconstruction", CC 2019.
- T. Mytkowicz, M. Musuvathi and W. Schulte, "Data-parallel finite-state machines", ASPLOS 2014; P. Prabhu, G. Ramalingam and K. Vaswani, "Safe programmable speculative parallelism", PLDI 2010.
- L. Rauchwerger and D. Padua, "The LRPD test", PLDI 1995 (speculative run-time dependence tests, background for L12 and L14).

Rewrite search and certificates:
- M. Willsey et al., "egg: Fast and extensible equality saturation", POPL 2021; Y. Zhang et al., "Better together: unifying Datalog and equality saturation" (egglog), PLDI 2023; O. Flatt et al., "Small proofs from congruence closure", FMCAD 2022; Y. Yang et al., "Equality saturation for tensor graph superoptimization" (Tensat), MLSys 2021; J. Meseguer, "Conditional rewriting logic as a unified model of concurrency", TCS 1992.

Loop-nest and array models:
- P. Feautrier, "Dataflow analysis of array and scalar references", IJPP 1991, and "Some efficient solutions to the affine scheduling problem", IJPP 1992; U. Bondhugula et al., "A practical automatic polyhedral parallelizer and locality optimizer" (Pluto), PLDI 2008; S. Verdoolaege et al., "Polyhedral parallel code generation for CUDA" (PPCG), TACO 2013.
- T. Henriksen et al., "Futhark: purely functional GPU-programming with nested parallelism and in-place array updates", PLDI 2017; T. Henriksen, F. Thoren, C. Oancea, "Incremental flattening for nested data parallelism", PPoPP 2019; A. Paszke et al., "Getting to the point: index sets and parallelism-preserving autodiff for pointful array programming" (Dex), ICFP 2021; M. Steuwer et al., "Generating performance portable code using rewrite rules" (Lift), ICFP 2015; B. Hagedorn et al., "Achieving high-performance the functional way" (RISE and ELEVATE), ICFP 2020.
- J. Ragan-Kelley et al., "Halide", PLDI 2013; F. Kjolstad et al., "The tensor algebra compiler" (TACO), OOPSLA 2017; T. Chen et al., "TVM", OSDI 2018.
- J. Ferrante, K. Ottenstein and J. Warren, "The program dependence graph and its use in optimization", TOPLAS 1987; R. Cytron et al., "Efficiently computing static single assignment form", TOPLAS 1991; G. Ottoni et al., "Automatic thread extraction with decoupled software pipelining", MICRO 2005.
- G. Kahn, "The semantics of a simple language for parallel programming", IFIP 1974; E. Lee and D. Messerschmitt, "Synchronous data flow", Proc. IEEE 1987; W. Thies et al., "StreamIt", CC 2002; A. Udupa et al., "Software pipelined execution of stream programs on GPUs", CGO 2009.
- W. Czerwinski and L. Orlikowski, "Reachability in vector addition systems is Ackermann-complete", FOCS 2021; J. Leroux, "The reachability problem for Petri nets is not primitive recursive", FOCS 2021.
- H. Jordan et al., "Souffle: on synthesis of program analyzers", CAV 2016; Y. Zhang et al., "GraphIt", OOPSLA 2018; A. Brahmakshatriya et al., "Compiling graph applications for GPUs with GraphIt", CGO 2021.
- U. Acar et al., "Heartbeat scheduling", PLDI 2018; M. Rainey et al., "Task parallel assembly language for uncompromising parallelism", PLDI 2021.

Floating point:
- S. Collange, D. Defour, S. Graillat and R. Iakymchuk, "Numerical reproducibility for the parallel reduction on multi- and many-core architectures", Parallel Computing 49, 2015 (ExBLAS).
- P. Ahrens, J. Demmel and H. D. Nguyen, "Algorithms for efficient reproducible floating point summation", ACM TOMS 46(3), 2020; J. Demmel and H. D. Nguyen, "Fast reproducible floating-point summation", ARITH 2013; U. Kulisch, *Computer Arithmetic and Validity*, de Gruyter, 2008.

## Appendix B. GNET-05 per program, probe input

`W` is work in agents; `S` columns are spans per model; "b form" and "c form" are the PNF node kinds used.

| program | split | probe input | W | S ser | S a | S b | S c | c laws | b form | c form |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| divide-00 | held_out | (20000, 30) | 216498 | 216498 | 72189 | 282 | 256 | path-sum | fork-join | path-sum |
| divide-01 | held_out | (20000, 30) | 407538 | 407538 | 203802 | 567 | 483 | affine Z/P, path-sum | fork-join | loop, path-sum |
| divide-02 | held_out | (20000, 30) | 376778 | 376778 | 215602 | 474 | 432 | path-sum | fork-join | path-sum |
| divide-03 | held_out | (20000, 30) | 290484 | 290484 | 87501 | 579 | 509 | path-sum | fork-join | path-sum |
| divide-04 | tuning | (20000, 30) | 386448 | 386448 | 183993 | 447 | 397 | path-sum | fork-join | path-sum |
| divide-05 | tuning | (20000, 30) | 198752 | 198752 | 108826 | 428 | 391 | path-sum | fork-join | path-sum |
| divide-06 | tuning | (20000, 30) | 250858 | 250858 | 125442 | 378 | 347 | path-sum | fork-join | path-sum |
| divide-07 | tuning | (20000, 30) | 452526 | 452526 | 151499 | 525 | 473 | affine Z/P, path-sum | fork-join | loop, path-sum |
| divide-08 | held_out | (20000, 30) | 4725359 | 4725359 | 1421330 | 1015 | 945 | path-sum | fork-join | path-sum |
| divide-09 | tuning | (20000, 30) | 921935 | 921935 | 656444 | 675 | 608 | affine Z/P, path-sum | fork-join | loop, path-sum |
| divide-10 | tuning | (20000, 30) | 457752 | 457752 | 146085 | 525 | 435 | affine Z/P, path-sum | fork-join | loop, path-sum |
| divide-11 | held_out | (20000, 30) | 4637684 | 4637684 | 785107 | 1221 | 1146 | path-sum | fork-join | path-sum |
| branch-00 | held_out | (12, 517) | 2683 | 2683 | 2453 | 340 | 208 | affine Z/P, path-sum | fork-join | loop, path-sum |
| branch-01 | tuning | (12, 517) | 781 | 781 | 410 | 410 | 410 | path-sum | fork-join | path-sum |
| branch-02 | tuning | (12, 517) | 124970 | 124970 | 60475 | 833 | 805 | path-sum | fork-join | path-sum |
| branch-03 | held_out | (12, 517) | 9095 | 9095 | 4578 | 826 | 820 | path-sum | fork-join | path-sum |
| branch-04 | tuning | (12, 517) | 198 | 198 | 153 | 153 | 74 | affine Z/P, path-sum | fork-join | loop, path-sum |
| branch-05 | held_out | (12, 517) | 211832 | 211832 | 96876 | 287 | 253 | path-sum | fork-join | path-sum |
| branch-06 | held_out | (12, 517) | 205 | 205 | 93 | 93 | 74 | affine Z/P, path-sum | fork-join | loop, path-sum |
| branch-07 | held_out | (12, 517) | 35 | 35 | 35 | 35 | 36 | affine Z/P, path-sum | serial | loop, path-sum |
| branch-08 | tuning | (12, 517) | 31774 | 31774 | 10617 | 261 | 248 | path-sum | fork-join | path-sum |
| branch-09 | held_out | (12, 517) | 859 | 859 | 582 | 191 | 190 | path-sum | fork-join | path-sum |
| branch-10 | tuning | (12, 517) | 239 | 239 | 206 | 206 | 205 | path-sum | fork-join | path-sum |
| branch-11 | tuning | (12, 517) | 1812 | 1812 | 797 | 233 | 217 | affine Z/P, path-sum | fork-join | loop, path-sum |
| frontier-00 | held_out | (4999, 21) | 202777 | 202777 | 202777 | 448 | 448 | counter, monotone | loop | loop |
| frontier-01 | held_out | (4999, 21) | 200484 | 200484 | 200484 | 444 | 444 | counter, monotone | loop | loop |
| frontier-02 | held_out | (4999, 21) | 80239 | 80239 | 80239 | 189 | 189 | counter, monotone | loop | loop |
| frontier-03 | tuning | (40, 23) | 138 | 138 | 138 | 138 | 140 | path-sum | serial | path-sum |
| frontier-04 | tuning | (40, 23) | 214 | 214 | 214 | 214 | 214 | path-sum | serial | path-sum |
| frontier-05 | tuning | (40, 23) | 299 | 299 | 299 | 299 | 293 | path-sum | serial | path-sum |
| frontier-06 | held_out | (40, 23) | 1978 | 1978 | 979 | 339 | 333 | path-sum | fork-join | path-sum |
| frontier-07 | tuning | (40, 23) | 479 | 479 | 269 | 190 | 112 | affine Z/P, path-sum | fork-join | loop, path-sum |
| frontier-08 | held_out | (40, 23) | 539 | 539 | 253 | 111 | 103 | path-sum | fork-join | path-sum |
| frontier-09 | tuning | (40, 23) | 633 | 633 | 357 | 116 | 106 | path-sum | fork-join | path-sum |
| frontier-10 | tuning | (40, 23) | 26 | 26 | 26 | 26 | 28 | path-sum | serial | path-sum |
| frontier-11 | held_out | (4999, 21) | 200782 | 200782 | 200782 | 446 | 446 | counter, monotone | loop | loop |
| serial-00 | tuning | (900, 3) | 1030 | 1030 | 1030 | 1030 | 1030 | - | serial | serial |
| serial-01 | tuning | (900, 3) | 610 | 610 | 610 | 610 | 610 | - | serial | serial |
| serial-02 | held_out | (900, 3) | 69 | 69 | 69 | 69 | 69 | - | serial | serial |
| serial-03 | held_out | (900, 3) | 186 | 186 | 186 | 186 | 97 | affine Z/P | serial | loop |
| serial-04 | tuning | (900, 3) | 42 | 42 | 42 | 42 | 42 | - | serial | serial |
| serial-05 | held_out | (900, 3) | 38 | 38 | 38 | 38 | 38 | - | serial | serial |
| serial-06 | tuning | (900, 3) | 42 | 42 | 42 | 42 | 42 | - | serial | serial |
| serial-07 | held_out | (900, 3) | 1162 | 1162 | 1162 | 1162 | 1162 | - | serial | serial |
| serial-08 | tuning | (900, 3) | 37 | 37 | 37 | 37 | 37 | - | serial | serial |
| serial-09 | tuning | (900, 3) | 1204 | 1204 | 1204 | 1204 | 1204 | - | serial | serial |
| serial-10 | held_out | (900, 3) | 171 | 171 | 171 | 171 | 90 | affine Z/P | serial | loop |
| serial-11 | held_out | (900, 3) | 74 | 74 | 74 | 74 | 74 | - | serial | serial |
| compose-00 | held_out | (3000, 14) | 313627 | 313627 | 105724 | 372 | 270 | affine Z/P, path-sum | fork-join | loop, path-sum |
| compose-01 | tuning | (3000, 14) | 155437 | 155437 | 53063 | 523 | 402 | affine Z/P, path-sum | fork-join | loop, path-sum |
| compose-02 | tuning | (3000, 14) | 103259 | 103259 | 39202 | 379 | 340 | path-sum | fork-join | path-sum |
| compose-03 | held_out | (3000, 14) | 12106 | 12106 | 8406 | 357 | 235 | affine Z/P, path-sum | fork-join | loop, path-sum |
| compose-04 | tuning | (3000, 14) | 2402 | 2402 | 819 | 243 | 162 | affine Z/P, path-sum | fork-join | loop, path-sum |
| compose-05 | held_out | (3000, 14) | 710 | 710 | 406 | 406 | 399 | path-sum | fork-join | path-sum |
| compose-06 | tuning | (3000, 14) | 16492 | 16492 | 11168 | 377 | 237 | affine Z/P, path-sum | fork-join | loop, path-sum |
| compose-07 | held_out | (3000, 14) | 114 | 114 | 98 | 98 | 96 | path-sum | fork-join | path-sum |
| compose-08 | held_out | (3000, 14) | 253 | 253 | 154 | 121 | 115 | path-sum | fork-join | path-sum |
| compose-09 | tuning | (3000, 14) | 2024 | 2024 | 920 | 525 | 517 | path-sum | fork-join | path-sum |
| compose-10 | tuning | (3000, 14) | 2340567 | 2340567 | 884050 | 1039 | 994 | path-sum | fork-join | path-sum |
| compose-11 | held_out | (3000, 14) | 1379 | 1379 | 837 | 233 | 147 | affine Z/P, path-sum | fork-join | loop, path-sum |

## Appendix C. Prototype

Location `$HOME/.cache/invar-net-laws/` (mode 700, outside the repository; Python 3, no dependencies).

| File | Purpose | sha256 (prefix...suffix) |
| --- | --- | --- |
| `lang.py` | Invar-subset parser | `5c07762e...14f1c662` |
| `laws.py` | law inference (section 6.2) and recursion classifier | `c4df6b40...ea3315ae` |
| `engine.py` | multi-model work/span interpreter | `affca93e...cb7276f5` |
| `probes.py` | robotics sketches and law probes | `e3c1140c...d69b64c1` |
| `lawcheck.py` | randomized differential checks of the laws and the f32 experiment | `a0d13cec...5ed10b75` |
| `run_gnet05.py`, `run_m501.py`, `run_probes.py`, `summarize.py` | drivers and tables | `f444336f...845c9dd7`, `ca942130...abc857a8`, `63a1363c...79e49afe`, `c5d63da6...74d765e1` |

Inputs: `bench/general-net-v01/gnet05` from `origin/feat/v01-gnet-05-corpus` and `bench/heldout-v1-proposal.json` from `origin/feat/v01-m5-01-heldout-generators`, read with `git archive`/`git show`.
Rerun: `python3 run_gnet05.py; python3 run_m501.py; python3 run_probes.py; python3 summarize.py; python3 lawcheck.py` (about five minutes on one core at nice priority; no Rust build, no GPU).
