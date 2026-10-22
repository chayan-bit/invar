# Interaction calculus and typed extraction design

Status: implementation design for M1/M2, 2026-09-22, revised 2026-09-23.
This specifies the small reference calculus and typed-graph obligations.
M1-04 encodes source into the typed graph, not into the three primitive agents.
See `../typed-interaction-v1.md` for the frozen implemented M1 fragment. The
primitive source encoding remains unproved research; no optimal schedule is
claimed.

## Decision and boundaries

The reference calculus is Lafont's standard interaction combinators: three
agent kinds, `gamma` (constructor), `delta` (duplicator), and `epsilon`
(eraser), with one principal port and respectively two, two, and zero
auxiliary ports. A port is connected to at most one other port or to the
region boundary. An active pair is a principal-to-principal wire. This is the
minimal, untyped, pure graph-rewrite reference; it is not Invar's ownership or
effect system.

Invar will also have a richer typed interaction representation. It carries
source types, affine permissions, demand, effects, cleanup, numeric policy,
and structured `Map`, `Fold`, `Reduce`, and `Scan` operations. That layer is a
typed extension with its own rules and correspondence obligations. It must not
be described as Lafont IC with labels added, and a `delta` in the untyped
reference never licenses duplicating an affine array. Every production
automatic-parallel candidate, including a future GPU candidate, is checked in
the typed layer and extracted through it. The compiler may still lower ordinary
scalar code directly and may use `native_only` after interaction validation;
the reference graph is not a required runtime for every scalar operation. A
supported high-level loop or recursive computation can therefore become a
generated CPU or CUDA-class device program without application-authored CUDA;
the device backend remains a later acceptance milestone.

## Minimal port graph

Use indexed storage so the Rust implementation can validate every untrusted
port and wire without cyclic Rust ownership. Let `A[x,y]` mean an agent `A`
whose two auxiliary wires leave to external endpoints `x` and `y`; `p` is its
principal port. Fresh names on a right-hand side are fresh ports or agents.
The six rules below are the normative M1 table; the order of the two agents is
irrelevant up to renaming.

```text
gamma[x,y]  <-> gamma[z,w]  -> wire(x,w), wire(y,z)
delta[x,y]  <-> delta[z,w]  -> wire(x,z), wire(y,w)
epsilon     <-> epsilon     -> empty

epsilon     <-> gamma[x,y]  -> epsilon0.p--x, epsilon1.p--y
epsilon     <-> delta[x,y]  -> epsilon0.p--x, epsilon1.p--y
```

For the commutation rule, consume `gamma[x,y]` and `delta[z,w]`, create
`D0,D1:delta` and `C0,C1:gamma`, and add these wires:

```text
D0.p--x       D1.p--y       C0.p--z       C1.p--w
D0.a1--C0.a1  D0.a2--C1.a1  D1.a1--C0.a2  D1.a2--C1.a2
```

The erasure rules create two fresh `epsilon` agents; they do not silently
drop an arbitrary typed value. The first rule is intentionally crossed. The
standard Lafont calculus differs here from the symmetric interaction
combinators used by some runtimes, where same-label annihilation uses one
uniform wiring. Invar must test this distinction rather than call the
variants equivalent.

The verifier checks agent arity, port ownership, one wire per port, boundary
types, and that every rewrite preserves the external boundary. A deterministic
reference reducer chooses the lowest stable redex ID. A bounded parallel
reducer may process only non-overlapping redexes and must compare its result
with the deterministic reducer. Strong confluence of the pure reference is a
useful property for this comparison; it is not a termination proof and does
not transfer automatically to typed bulk nodes or effectful code.

## Finite source encoding and call-by-value

M1 encodes a bounded, explicitly typed internal fragment: `Unit`, `Bool`,
`Nat`, products, affine lambdas, application, `let`, `if`, named calls, and
failure fixtures. Its nodes have ordered input ports. `Let` demands the
producer even when unused; `Apply` demands function before argument; `Select`
demands only the chosen branch. The typed checker rejects duplicated affine
owners. An independent term evaluator and the graph evaluator compare demand,
completion, value, first failure, and tool-limit outcomes. Node/port templates
and extraction are frozen in `../typed-interaction-v1.md`.

This graph is a typed extension with `Apply` and `Select` nodes; it is not a
Lafont three-agent net. Translating the fragment into gamma/delta/epsilon
while preserving call-by-value demand remains separate research. A finite
graph test is evidence for the implemented typed rules, not a universal
correspondence theorem. Unknown termination keeps the ordinary sequential
implementation.

## Typed bulk extension and extraction

The target production representation is a typed port graph with demand-aware
control and structured bulk nodes. M1 implements only `Apply`, `Select`,
wrapping-u32 `Map`/`FusedMap`, and public MIR data-wire reification. Later
`Fold`, `Reduce`, `Scan`, cleanup/effect tokens, and admission records remain
their assigned milestones. Unknown facts never authorize a rewrite.

Bulk nodes have normative source meanings for every supported finite input.
For example, `Map(f, xs)` is the ordered vector `[f(x_i)]`; `Fold` keeps source
order; `Reduce` may regroup only with an established associative identity and
may permute only with commutativity; `Scan` preserves prefix order. The
M1 implements a serial graph evaluator and extraction into serial loop stages.
It does not expand bulk nodes into the minimal reference calculus or generate
GPU kernels.

The M1 affine beta rule rewires a closed, total `Apply(Lambda, argument)` to
`Let(argument, body)`, preserving demand of the argument and changing the
extracted body. Capturing or uncertain-effect cases remain unchanged. The M1
bulk rewrite is wrapping-u32 map fusion:

```text
Map(f, Map(g, xs)) -> Map(f composed_with g, xs)
```

The M1 callbacks are the finite wrapping add/multiply operations, which are
total and nontrapping. Both map nodes must have total effect facts; otherwise
the rule declines. The test shows two extracted loops becoming one. General
callback legality and footprints remain later obligations.

The later extraction contract covers recursion. A split-and-concatenate
recursive function can become `BulkMap` only when the checker proves complete
disjoint coverage, stable output order, callback suitability, and compatible
ownership. The checker does not unroll runtime-sized input; a failed proof
leaves the recursive source on its ordinary sequential path.

## Purity, effects, traps, and affine scan summaries

Initial automatic candidates require a total, nontrapping, externally pure
region. Clock, randomness, IO, atomics, foreign calls, externally visible
mutation, observable allocation, and resource cleanup exclude a region. A
fresh internal output is allowed only when its lifetime and publication are
checked. A trapping callback, unknown effect, or uncertain branch demand keeps
the serial candidate; parallel workers must not turn a source-ordered trap or
effect prefix into a task-local result.

`Scan` summary inference is a typed law, not a heuristic. Recognize a callback
whose state update has a finite summary `s : X -> X`, prove its composition and
identity under the exact numeric policy, and attach a rule ID to the immutable
region hash. For the modular affine case `s(x) = a*x + b (mod 2^32)`, applying
the left summary then the right summary is:

```text
(a,b) tensor (c,d) = (c*a, c*b+d) mod 2^32
identity = (1,0)
```

This operation is associative but generally noncommutative, so an ordered tree
scan is legal while operand reversal is not. The existing
[bounded affine probe](summary-evidence.md) is useful evidence for this finite
law and its edge cases; it is not a proof or compiler admission. Checked `i32`
arithmetic, floating-point reassociation, data-dependent effects, or an
unproved summary produce `Unknown` and use the sequential scan.

## Milestones, controls, and acceptance

1. **M1 reference port core:** implement the indexed graph, six rules, verifier, bounded
   reducer, and malformed-connectivity tests. Test all port permutations,
   standard gamma-gamma swapping, duplicate/dangling ports, and resource
   limits.
2. **M1 typed source agreement:** test beta on closed pure terms, CBV demand on an
   unselected divergent branch, affine duplicate/drop rejection, bounded
   recursion, and source/net value agreement. Compare traps and limits as
   typed outcomes where the fragment exposes them.
3. **M1 typed extraction:** implement the finite bulk meanings;
   test map fusion and one topology-changing beta rewrite against the source
   interpreter and serial extracted IR. Mutating a port, contract version, or
   rule ID must reject the candidate.
4. **M2 admission:** recognize one counted ordinary loop with disjoint writes,
   route it through the typed interaction representation and extraction, and
   compare serial and coarse CPU execution at worker counts 1, 2, and a bounded
   larger value. Add the modular affine scan law only after the negative order
   and numeric-policy tests pass.
5. **GPU later:** generate device code for an admitted extracted region and
   validate transfer costs, completion lifetimes, and CPU fallback. No GPU
   acceptance follows from a graph evaluator or a benchmark alone.

`direct_structured_control` is an experimental ablation: it may lower the same
program directly to structured IR to measure compile time, graph cost, and
runtime quality. It must be labeled as a control and cannot admit a production
automatic-parallel candidate. `native_only` means no residual graph runtime
after interaction validation and extraction; it is not a bypass of the
typed-graph milestone. The three-agent source encoding is not an M1 gate.

## Pinned references

- Lafont, *Interaction Combinators*, Information and Computation 137 (1997),
  DOI [10.1006/inco.1997.2643](https://doi.org/10.1006/inco.1997.2643).
- HVM2, an extended interaction-calculus runtime, inspected at commit
  [`654276018084b8f44a22b562dd68ab18583bfb5b`](https://github.com/HigherOrderCO/HVM2/tree/654276018084b8f44a22b562dd68ab18583bfb5b).
  Its runtime and symmetric/extended rules are evidence about an implementation
  path, not Invar's minimal-calculus specification.
- Current Bend 2, a distinct design from the earlier Bend/HVM2 language, inspected at commit
  [`a49524265bdfa5753a4bf38e25f0574a705dd868`](https://github.com/bendlang/bend/tree/a49524265bdfa5753a4bf38e25f0574a705dd868).
  Its affine ownership and parallel-call restrictions are comparison points,
  not Invar requirements or evidence that the current compiler uses HVM2.
