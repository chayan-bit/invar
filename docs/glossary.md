# Glossary

Terms a newcomer meets in the code, tests, issues and receipts, in alphabetical order.
Each entry says what the word means in this project and links to where it lives.
Milestone labels such as M4-05, GNET-03 or ENG-06 are issue tracker names, listed at the end.

## A

**Admission**
The decision that a loop region may run as a parallel or device plan instead of plain serial code.
A region is admitted only when every obligation has evidence; one `unknown` obligation rejects it, and a rejected region runs the serial MIR plan.
See [src/admission.rs](../src/admission.rs), [explain JSON](explain.md) and [architecture](architecture.md).

**Admitted candidate**
The value the admission code hands to the code generators, so a generator cannot be given a plan that was never checked.
See [architecture](architecture.md#admission-and-trust).

**Affine ownership**
Every owned array can be used at most once: a move consumes it, and a borrow constrains access for a while.
There is no hidden copy and no reference counting.
See [specification section 4](spec/invar-v0.1.md#4-ownership) and [src/ownership.rs](../src/ownership.rs).

**AOT container (compiled-ahead container)**
A file written at build time that holds precompiled device libraries, so a production run never compiles kernel source.
`invar build --target metal` writes the `INVGMETL` container, and the CUDA general-net route has a matching format.
Its digests give integrity, not authenticity, so every field is validated on load.
See [src/metal_aot/](../src/metal_aot/README.md) and [src/general_net/device_host/cuda_aot.rs](../src/general_net/device_host/cuda_aot.rs).

## B

**Byte budget**
The single mechanism that bounds admission: three ceilings in bytes for host memory, device memory and analysis structures.
It replaced older fixed element-count caps.
See [src/budget.rs](../src/budget.rs).

## C

**Candidate**
A recognized source loop together with the facts needed to run it as one plan.
Every source family has its own `Candidate` type, for example in [src/histogram_source.rs](../src/histogram_source.rs).

**Certificate**
A checked record that a rewrite law applies to a region of the net.
Each lowering phase re-verifies the certificate against the current net; it never trusts a name, hash or benchmark id.
See [net normal form](net-normal-form-v01.md) and [src/general_net/normal_form/](../src/general_net/normal_form/mod.rs).

**Checked (arithmetic)**
Arithmetic that traps on overflow, the default for `i32` and `u32`.
It is why a checked sum is never regrouped into a tree reduction.
`wrapping_add` and `wrapping_mul` are the modular alternative that the planner may parallelize.
See [specification section 3.4](spec/invar-v0.1.md#34-integer-arithmetic-and-traps).

**Contract**
A word with three nearby meanings.
A region contract lists what a region reads, writes, owns and can fail on ([architecture](architecture.md#shared-contract-progressively-enriched)).
A family contract (the `Contract` type in each `*_source` module) binds recognized source to the exact behavior an executor must reproduce.
A document called a contract (for example [the GNET core contract](../src/general_net/CONTRACT.md)) is a frozen interface description.

**cr-v1 (`invar-math-cr-v1`)**
The correctly rounded math library for `f32` and `f64`: `sqrt`, `sin`, `cos`, `exp` and the rest return the correctly rounded result on every engine.
It is built on the vendored CORE-MATH sources.
See [specification Appendix B.4](spec/invar-v0.1.md#b4-binary64-and-the-correctly-rounded-math-library), [numeric contracts](numeric.md) and [crates/invar-math/](../crates/invar-math/Cargo.toml).

## D

**Demand**
Which sub-expressions are actually evaluated.
An unselected branch is never demanded, so a trap or infinite loop inside it must not happen.
The typed net records demand order explicitly.
See [typed interaction graph](typed-interaction-v1.md).

**Device IR**
The shared physical description of a device kernel plan, verified once and then emitted as CUDA or Metal source.
Only a verified plan reaches code emission.
See [src/device_ir.rs](../src/device_ir.rs) and [crates/invar-device/](../crates/invar-device/Cargo.toml).

## E

**Engine**
One way to run a program: `reference`, `native-serial`, `cpu-parallel`, `cuda`, `metal` or `auto`.
An engine that is unavailable gives a typed error, never a silent substitute.
See [src/engine.rs](../src/engine.rs) and [specification section 11.1](spec/invar-v0.1.md#111-cli-and-exit-codes).

**Evidence kinds**
The four values an obligation can carry in a receipt.
`checked_invariant`: a verifier checked it, with a stated scope.
`trusted_rule`: a named, reviewed compiler rule establishes it, and the implementation is trusted.
`runtime_guard`: a total, nonobservable predicate is checked at run time before any write.
`unknown`: no evidence, so the region is rejected.
See [explain JSON](explain.md) and [architecture](architecture.md#admission-and-trust).

**Extraction**
See *typed extraction*.

## F

**Fallback plan**
The plan that runs when a runtime guard is false or a region is rejected, always serial MIR.
A false guard selects it before any visible write.
See [explain JSON](explain.md#static-report).

**Family**
A named shape of ordinary source loop that a recognizer accepts and a bounded executor runs: map, reduction, scan, segmented, compaction, gather, histogram, stencil, transpose, tensor contraction, summary and others.
`invar run` offers each reached loop to the family registry in precedence order, and the first match wins.
See [specification section 11.4](spec/invar-v0.1.md#114-source-families), [src/schedule/families.rs](../src/schedule/families.rs) and [examples/families/](../examples/README.md).

**Fold tier**
How a floating-point reduction is grouped: `strict` (source order), `tree` (a fixed tree shape), `exact` (the correctly rounded sum) or `relaxed` (any order within a declared bound).
A non-strict tier is declared in source and recorded in the receipt.
See [numeric contracts](numeric.md#policy-version-2-fold-tiers-and-storage-types).

**Frame, quantity (units)**
Invar checks physical dimensions on numbers (quantities) and coordinate frames on transforms.
See [specification section 6](spec/invar-v0.1.md#6-units-and-frames).

**Fuel**
A step budget for the evaluators; `--fuel N` runs exactly N steps, default 1,000,000.
Running out gives `evaluation_limit`, which is a tool limit and never a proof that the program diverges.
See [specification section 3.6](spec/invar-v0.1.md#36-fuel).

## G

**GNET (general net)**
The general interaction-net route: a checked program is encoded as an executable net, which serves as an independent oracle and as the input for general compiled lowering to CPU models, CUDA and Metal.
See [src/general_net/](../src/general_net/README.md), [the GNET core contract](../src/general_net/CONTRACT.md) and [specification section 11.6](spec/invar-v0.1.md#116-general-interaction-net-compilation).

**Guard (runtime guard)**
A cheap check evaluated before a plan writes anything, for example that the loop bound fits every array length and the footprint fits the byte budget.
False selects the fallback plan.
See [explain JSON](explain.md#static-report).

## I

**Interaction combinators, interaction net**
A graph model of computation where agents connect through ports and local rewrites replace pairs of connected agents.
Invar uses a typed, demand-aware version with bulk operations as the representation through which every automatic parallel plan passes.
It is not one graph node per array element.
See [parallelism foundations](parallelism-foundation.md) and [typed interaction graph](typed-interaction-v1.md).

## L

**Law**
An algebraic identity the compiler may use to regroup work, such as associativity of wrapping addition.
The sealed wrapping-`u32` laws are in [src/laws.rs](../src/laws.rs), and the normal form uses numbered laws L1 to L20, each with a certificate.
See [net normal form](net-normal-form-v01.md) and [law collective interface](law-collective-v01.md).

**Loan**
A temporary borrow of an array for the duration of a call: `read` (shared) or `mut` (exclusive).
See [specification section 4.2](spec/invar-v0.1.md#42-call-loans).

## M

**MIR**
The typed mid-level control-flow IR: functions of basic blocks with typed values and exactly one terminator per block.
It is an in-memory schema, verified when lowered and again when evaluated.
See [typed MIR v1](mir-v1.md) and [src/mir/](../src/mir/README.md).

## N

**Native (engine, feature)**
Optional LLVM 21 code generation for verified MIR, behind the `native` Cargo feature.
See [native toolchain](native-toolchain-v1.md) and [src/native/](../src/native/README.md).

**Normal form (parallel normal form, PNF)**
The law-directed description of a net's regions: which regions are maps, folds, scans or residual fork-joins, each with certificates.
See [net normal form](net-normal-form-v01.md) and [PNF family admission](pnf-family-admission-v01.md).

**NVRTC**
NVIDIA's runtime compiler for CUDA source.
Invar compiles CUDA kernels through it in a separate helper process, `invar-nvrtc`.
See [src/bin/](../src/bin/README.md).

## O

**Obligation**
One item in an admission report, with a name, evidence kind and detail.
All obligations of a region must have evidence other than `unknown`.
See [explain JSON](explain.md#static-report).

**Oracle**
An independent implementation used only to check another one.
Invar has three: the reference evaluator on checked source ([src/eval/](../src/eval/README.md)), the MIR evaluator ([src/mir_eval/](../src/mir_eval/README.md)) and the GNET net oracle ([src/general_net/oracle.rs](../src/general_net/oracle.rs)).
Tests compare value, trap, step count and cleanup across them, and never weaken an oracle to make a test pass.
See [validation](validation.md#independent-oracles).

## P

**Placement**
Choosing where a program runs (CPU or device) from measured costs, using a calibration profile for the live host.
Placement describes costs and never grants permission to run something that admission has not admitted.
See [placement profile](../spec/placement-v1.md) and [src/placement.rs](../src/placement.rs).

**Plan**
The chosen way to execute one region: serial, map, fused map, reduction, scan or a family plan.
`plan::decide` picks it after admission.
See [src/plan.rs](../src/plan.rs).

## R

**Receipt**
A JSON record of what actually happened or was decided: which regions were admitted, which obligations had which evidence, which backend ran.
`invar explain` writes an `invar-explain-v5` document, and `invar run --format json` writes an `invar-diagnostics-v3` document.
See [explain JSON](explain.md) and the [schemas](../schema/explain-v5.schema.json).

**Reference evaluator**
The tree-walking interpreter over checked source that defines what a program means.
Every other engine must agree with it on completed runs.
See [src/eval/](../src/eval/README.md).

**Region**
A unit of analysis: a loop or function body that admission and planning consider as one piece.
In the GNET route a region is a function body or natural loop of the compiled MIR.
See [src/admission.rs](../src/admission.rs) and [net normal form](net-normal-form-v01.md#2-regions-p1a).

**Rule revision**
The version name of the compiler rule that justified a plan, for example `counted-wrapping-map-v1`.
Evidence is bound to a rule revision, so a rule change invalidates old receipts.
See [src/admission.rs](../src/admission.rs).

## S

**Scheduler (schedule)**
The code that runs admitted regions on CPU workers, CUDA or Metal and publishes their results.
See [src/schedule/](../src/schedule/README.md).

**Serial fallback**
Plain execution in source order.
It is always correct, and it is what happens whenever legality is unknown.

**Source identity**
`fnv1a64:` plus 16 hex digits over the exact source bytes.
It identifies a source file in a receipt; it does not prove authenticity.
See [explain JSON](explain.md#static-report).

**Strict f32 (`strict-f32-v1`)**
The floating-point policy for `f32`: every operation rounds independently to nearest even, and implicit fused multiply-add, contraction, reassociation and fast math are forbidden.
See [numeric contracts](numeric.md#policy-strict-f32-v1).

**System dialect**
The part of the language for describing robot systems: `component`, `connect`, `flow`, `frame`, `system`.
`invar system` checks them, and generators turn them into ROS 2 workspaces and MJCF or URDF files.
See [system integration](system-integration-v1.md), [specification section 8](spec/invar-v0.1.md#8-system-dialect) and [src/system/](../src/system/README.md).

## T

**Trap**
A defined runtime failure: overflow, division by zero, index out of bounds.
A trap ends the whole program with a category and a source span, in source order, and no cleanup runs after it.
The CLI exit code for a trap is 3.
See [specification section 3.4](spec/invar-v0.1.md#34-integer-arithmetic-and-traps) and [src/eval/](../src/eval/README.md).

**Typed extraction**
Turning a verified typed net back into a conventional execution plan (loops, tasks, kernels) after checked rewrites.
Every automatic parallel candidate, GPU ones included, goes through it.
See [src/typed_net/extract.rs](../src/typed_net/extract.rs) and [parallelism foundations](parallelism-foundation.md).

## W

**Wrapping**
Modular arithmetic on `u32` (`wrapping_add`, `wrapping_mul`).
It is associative, so reductions and scans over it can be regrouped legally.

## Issue and milestone labels

These are names from the issue tracker that also appear in test and file names.
The [testing guide](testing.md) maps the test files that carry them to what they cover.

| Label | Meaning |
| --- | --- |
| M0 to M5 | Milestones: M0 language core and reference evaluator, M1 MIR, native and typed nets, M2 CPU scheduler, M3 CUDA and Metal maps, reductions and scans, M4 source families (indexed, transpose, stencil, contraction, summary), M5 benchmarks, qualification and placement |
| P1 to P8 | General-net normal form phases: P1 regions and laws, P2 to P4 lowering, P8 law collective |
| GNET-nn | General-net work items (oracle, native lowering, device models, provenance, numeric hosts) |
| ENG-nn | Engineering work items (laws prover, tiers, contraction emitter, byte budgets, native SIMD) |
| L1, R1, S1 | Language extensions, robotics vocabulary and system integration work |
| v01, v02 | Releases: v0.1 is the specified language, v0.2 adds float reductions, sorts, spatial queries and binary64 |
