# Review of the original proposal

Updated: 2026-09-22. Revision: design 0.3; bootstrap language 0.1. Rust implementation and required interaction foundations supersede the initial 0.2 recommendations.

## Assessment

The proposal already makes several unusually good distinctions: checked integer addition is not associative, pure-looking code may diverge, cancellation does not release borrowed buffers, physical copies are not semantic clones, and plan checking is not a proof of the backend. Keep these. The main weakness is the distance between this careful architecture and something a compiler implementer can finish and independently test.

The proposed region contract is the central hypothesis worth testing. Interaction combinators are now an explicit required foundation for Invar's automatic-parallelization architecture. Their role and proof obligations are defined in [parallelism foundations](parallelism-foundation.md); they are not a prerequisite for GPU compilation in general. A custom dialect, recipe system, effect system, general borrow checker, GPU runtime, and proof engine all arriving together would make failures hard to isolate.

## Changes adopted

| Original section | Problem or gap | Correction |
| --- | --- | --- |
| 16, 22, 30, 33 | Several representations and a large repository structure precede a working semantic core. | One Rust Cargo package with a thin binary/library boundary, one typed representation, and an independent tree interpreter first. Add a representation only when a concrete transformation needs it. |
| 17, 33.3 | Interaction work sits ahead of first multicore value. | Freeze the reference calculus and typed extraction in M1, required before automatic parallelism. Retain an ablation against direct structured optimization to measure implementation choices; do not make the subsystem optional. |
| 4, 19, 21 | An admission equation can sound stronger than the checker actually establishes. | Separate structural validity, legality under named trusted rules, runtime guard checks, and proof artifacts. Unknown and unrecognized evidence reject a candidate. |
| 5.2, 7, 8 | Call modes and owner mutability do not completely fix overlapping argument evaluation. | Specify left-to-right argument evaluation and call-duration loans. Reject `f(xs, xs)` when either argument is mutable/consuming. No implicit two-phase mutable borrowing in the bootstrap. |
| 5.2 | Assignment specifies RHS order but not destination evaluation. | RHS first, then destination index, then bounds/permission validation, then replacement. No mutation if RHS fails. A later destination failure occurs after RHS effects. |
| 8.4 | Last-use borrowing plus views, closures, task captures, and lifetime polymorphism is too large for the first checker. | Bootstrap supports whole-array call borrows only, forbids escaping borrows and partial moves, and uses explicit control-flow joins. Precise views come later with their own rules. |
| 9.3 | A destructor that cannot panic can still diverge or block forever. | Future user cleanup needs separate termination and blocking contracts. Bootstrap arrays contain only trivial scalar elements, so no user destructor loophole exists. |
| 11, 35 | Literal typing, signed remainder, minimum signed integer, parsing limits, and assignment grammar leave implementers decisions that affect meaning. | Freeze a small lexical/numeric contract and full EBNF now. Wider types and floats wait for matching tests. |
| 14.4 | Scope exits beyond `?` need explicit treatment. | Future scope rules cover `return`, `break`, `continue`, and cancellation, including deterministic drainage order and terminal process-trap behavior. No async syntax is accepted before this is specified. |
| 21.3 | A false guard has a fallback, but a malformed or stale proof artifact is a compiler defect/security boundary. | Runtime false selects fallback; invalid internal IR is a hard compiler error. Stale cache entries are discarded and recomputed. Never silently execute unvalidated IR. |
| 22.5 | LLVM attribute guidance omits progress promises. | Preserve divergence; never infer `mustprogress` or `willreturn` merely from memory-effect freedom. Check divide/shift corner cases before issuing operations with undefined/poison behavior. |
| 29.3 | “Divergent computation” is not a terminal machine configuration. | State progress as value, defined trap, or next step; divergence is an infinite execution. Interpreter fuel exhaustion is inconclusive tooling output, not a source trap. |
| 31, 32 | Interpreter agreement alone can reproduce a shared bug; numeric targets have no measured baseline. | Use hand-derived boundary cases and structural lowering checks alongside differential tests; keep performance targets as hypotheses until baselines are frozen. |
| 33 | “Milestone 0” already contains a large fraction of a full language. | First delivery is a deliberately finite integer/array core. Native CPU work is separately gated; no unsupported feature is treated as implemented. |

## Larger architectural improvements

**Semantic identity before extensibility.** An executable semantic operation and its input/output contract have a version. An optimization references that version and a named rule. Do not start with a public recipe DSL or a plugin loader. Two built-in CPU transformations are enough to test the architecture.

**Start with one useful demonstration.** The first parallel milestone is a total wrapping map over independent array elements, with a serial fallback and a lawful modular sum. Test both small and large inputs, forced serial and forced parallel, and all supported worker counts. This isolates the project's promise better than one favorable GPU matrix multiplication.

**Separate program faults from tool limits.** Lexical errors, rejected borrows, source traps, exhausted evaluation fuel, compiler defects, and unavailable hardware must have distinguishable results. None permits undefined source behavior.

**Build authoring support early.** Stable diagnostic codes and byte spans are cheap relative to a language server. They should arrive with the parser. A deterministic compiler-produced interface packet follows typing; a model integration does not.

**Make unimplemented syntax fail clearly.** The full proposal remains valuable, but accepting a spelling with guessed semantics is worse than an explicit unsupported-feature diagnostic. Each grammar expansion requires tests and a documented contract.

## Open decisions deliberately retained

The project has not selected a license, promised a stable ABI, proven soundness, selected a GPU backend release, or measured a profitability threshold. The native milestone must pin a tested LLVM release and target matrix; do not install a guessed LLVM version during documentation bootstrap. Floating-point NaN canonicalization and backend conformance remain a cost/semantics decision before float support, not an excuse to use fast math silently.

The first core is a testable foundation, not a general-purpose replacement for Rust, Go, or C. Records, variants, strings, recoverable `Result` errors, interfaces, views, and interoperability remain necessary for that later claim.

## Source grounding

The original proposal is not retained in this repository (SHA-256 `838e4aeef66887bb7d459a275d4d4898c2d04059e8b47201c5274f17fc4d588b`). Its bibliography is historical; its claim that every source was checked is not repeated here.

The following primary documents were checked for this review:

- [MLIR effects and speculation](https://mlir.llvm.org/docs/Rationale/SideEffectsAndSpeculation/): effect freedom and safe speculation are distinct. This supports conservative motion and termination handling, not an Invar safety proof.
- [LLVM language reference](https://llvm.org/docs/LangRef.html): overflow intrinsics support checked arithmetic; progress attributes and arithmetic flags impose semantic obligations. Invar lowering must establish those obligations.
- [MLIR GPU dialect](https://mlir.llvm.org/docs/Dialects/GPU/): provides GPU operations and lowering infrastructure. It does not discharge Invar's source-level legality, lifetime, or profitability obligations.

These support specific infrastructure choices. The proposed integration remains our engineering hypothesis.

## Rust implementation decision

The compiler, runtime, interaction machinery, and tools use Rust under [the project engineering contract](rust-engineering.md). This replaces the original implementation-language recommendation, not the Invar source semantics. External native backends remain isolated dependencies. The nine M0 issues now use Cargo, safe Rust, explicit errors/arithmetic, and debug/release tests. No code migration was needed because implementation had not begun.
