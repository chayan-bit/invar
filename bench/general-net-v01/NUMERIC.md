# Scalar f32 source corpus and executable ABI8

The additive `numeric-vectors.json` preserves all 158 prior corpus rows and adds
96 independently expected calls from 11 source files. The union has 254 rows,
69 unique source files and 159 named function declarations. Existing manifests,
old expectations, integer/owner helpers and immutable evidence are unchanged.

The new source functions cover ordered arithmetic, explicit fma versus separate
multiply/add, all six IEEE predicates, negation/copy, integer conversions,
recursion, loops, branch selection and earlier integer failure. Expectations pin
rounding ties, subnormal-to-normal carry, underflow, overflow, signed zero,
infinities and NaN classification. Echo/negate/selected rows also pin the actual
NaN payload/sign encodings. Ordinary source exclusions include f32 arrays,
nominal numeric fields, numeric custom variants and an integer owner coexisting
with f32. Their different source/oracle/compiler admission results are retained.

## Published reference and execution boundary

`compiled::emit_numeric` emits standalone C++17 for `Target::Cpu`, ABI8, with
`numeric_policy() == Some(NumericPolicy::STRICT_F32)`. Artifact type metadata and
nonnegative u32 bit carriers remain mandatory. ABI8 has no owner, continuation,
speculation or device layout. CUDA/Metal emission returns `NumericTarget`.
Legacy emission still rejects f32. No public scalar-export ABI is expanded.

The private `eval::named::observe_numeric_named` source API directly executes
the original checked function with exact scalar arguments and its original fuel,
operator spans and cleanup. All 250 source-valid rows now execute; four original
frontend rejections remain. This closes all 85 previously unavailable f32 named
observations. Unknown names/types/signatures or invalid fuel fail the corpus
check rather than becoming unavailable rows. The public `observe_named` export
signature still rejects f32; no wrapper or public wire tag is introduced.

All 244 oracle executions, six oracle exclusions, 179 actual CPU executions
and 71 compiler exclusions remain in the same 254-row denominator. Three private
oracle cap failures (transition, trace, allocation) are explicitly distinct from
source execution: 241 full-row source/oracle comparisons and 177 source/native
comparisons agree semantically. All 179 CPU observations, including backend
failures, still compare exact oracle headers/events. Every one of the 85 numeric
runtime rows additionally compares every positive fuel prefix through its final
source step plus one. The 3,626 environment vectors now compare actual source
as well as oracle and all four generated CPU process environments.

Ordinary `main` programs retain checked-source/MIR/oracle fuel observations,
including earlier integer traps and owner cleanup. The source corpus represents
F32 structurally while preserving the old integer/owner corpus's separate 31/32
structural denominator. ABI10 separately admits direct f32 arrays to the oracle; this ABI8 corpus still retains all compiler exclusions and does not widen nominal/device admission.

Every native comparison checks the eight-word result/trap/fuel/transition/frame/
trace header and every seven-word logical event. Non-NaN results must match exact
encodings; arithmetic NaNs agree by the frozen unobservable class. Selected bit
copy/negation cases additionally assert raw encodings. Source/MIR cleanup is
checked even when ABI8 rejects the owning program. Backend transition and trace
limits remain separate from completing source semantics.

## Reproduction

The integration basis3027091 contains the private named-source API and the
unchanged `tests/issue_gnet_v01_corpus_expansion.rs` adapter. This continuation
changes only the bench-owned numerical comparator and current docs; it reuses
the three existing numeric tests. No root, array/core or test adapter edit occurs.

```sh
INVAR_FULL_CORPUS=1 INVAR_GNET_CORPUS_REPORT_DIR=/tmp/new-numeric-debug \
  cargo test --offline --locked --test issue_gnet_v01_corpus_expansion numeric_ -- --nocapture
INVAR_FULL_CORPUS=1 INVAR_GNET_CORPUS_REPORT_DIR=/tmp/new-numeric-release \
  cargo test --offline --locked --release --test issue_gnet_v01_corpus_expansion numeric_ -- --nocapture
cargo fmt --all -- --check
cargo clippy --offline --locked --all-targets -- -D warnings
```

Use Rust/Cargo1.95, C++17 with UBSan, `CARGO_BUILD_JOBS=2` and
`RUST_TEST_THREADS=2`. The native compiler receives
`-O2 -fsanitize=undefined -fno-sanitize-recover=all`, a 60-second deadline with
two-second kill grace, and a 30-second batch execution deadline. Host rounding
and x86 FTZ/DAZ changes apply only inside each native child process. There is no
GPU execution or timing result.

The original numeric corpus's 85 unavailable source observations describe an
older API scope. No hardware/performance acceptance follows.

## Root integration qualification

On integration source d4e19c6 plus this corpus change, Debug and Release each
passed16 selected tests (10 corpus,3 parity,3 named source); the three older long
corpus tests were explicitly filtered. Strict all-target native,cuda,metal Clippy
and fmt passed. The ABI10 oracle addition changes one historical exclusion into
an execution:244 executed,6 unsupported,4 frontend rejected;241 source/oracle
full-row comparisons and223 source/MIR/oracle fuel observations. All254 rows and
ABI8 compiler expectations remain. The first root attempt failed only the old
243 execution assertion. Earlier receipts describe their pre-ABI10 scope.

This is component qualification, not complete
v0.1 or hardware acceptance.

## strict-f64-v1 scope change (v0.2)

strict-f64-v1 admits `f32` and `f64` variant payloads, so `numeric/variant_f32_rejected.inv` no longer marks the numeric source boundary.
The case `variant_f32_array_source_rejected` (`numeric/variant_f32_array_rejected.inv`, an `Array<f32>` payload) takes its place with the same `UNSUPPORTED_FEATURE` code, so the row totals are unchanged.
The old source stays because the 2026-10-02 receipts hash it.
The integer/owner source corpus excludes the `F64` rule beside `F32`.
