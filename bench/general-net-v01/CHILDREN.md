# Source-derived child extraction and ordered commit

The additive `corpus_children.rs` runner retains the 52 ordinary sources and
67 runtime rows. `child-vectors.json` adds 39 independently expected calls from
six ordinary source files, bringing the execution denominator to 158 rows and
the unique-source inventory to 58 files. Original manifests, source exclusions,
ABI2/3 tests and historical receipts remain intact.

The new fixtures cover runtime-selected branches, recursive wrapping `u32`
frontiers, a three-call fanout, checked `i32` overflow, division traps, an older
divergent call before later bad work, dependent scalar calls, affine mutation,
and private owner replay. Frontier expectations use
`2^depth * (seed + 2*depth)` modulo `2^32`; fanout expectations use
`3 * 2^depth * (seed + 2*depth + 1)` modulo `2^32`. Trap expectations pin the
operator byte, including the division operator at bytes 171..172. Fuel is a
source observation bound, not proof that a divergent source terminates.

## Executed boundary

ABI4 uses `compiled::emit_speculative` and runs its emitted C++ CPU program.
ABI5 uses `compiled::emit_device_speculative` for CUDA and Metal, then executes
the generated control body in an explicitly labeled CPU workgroup model with
32 real threads and barriers. The Metal model removes address-space attributes
and bridges the generated kernel signature. These models do not establish
CUDA/MSL compiler acceptance, GPU execution, hardware work width, driver fence
behavior, or performance. No device job or timing series is part of this corpus.

Every admitted row runs with quantum/capacity schedules `(1, 2)`,
`([3,17], 2)` and `(17, 0)`. Capacity zero tests ordered execution through the
same compiled continuation. The runner compares the complete result header,
trap span, source steps, ordered trace, transition count, ordered cleanup and
root-owner values/handles against the oracle. It checks monotonic counters,
stable committed prefixes and an unpublished result at every yield. Completed
retries must preserve primary continuation, result, trace, cleanup and heap.
ABI5 wrapper bookkeeping may advance on a completed retry.

The focused failure matrix exercises private fuel/transition/trace exhaustion,
missing or aliased ABI4 scratch, stale generations, generation overflow,
checked projection guards, geometry/configuration rejection, owner replay and
primary allocation limits. Each must either preserve ordered reference
behavior or return the explicitly recorded private backend failure. Dependent
and affine calls must serialize without creating a private child. Source-order
fuel prefixes ensure an older divergent call hides a later trap/overflow even
when later work was computed privately. Fanout measures two private children
and actual commit in the CPU/model executions.

## Admission and completion ownership

The workset test verifies all 12 complete ABI5 buffer ranges, the published
`DeviceConfig`, alignment, device/allocation identities, checked range ends and
all pairwise overlaps. Its allocation IDs belong to a bounded CPU registry;
they are not GPU allocation identities.

`FencedExecution` owns the complete CPU model buffer bundle and child-process
handle until OS process completion. One successful completion is observed and
compared with the reference; cancellation publishes no result and reclaims
only after completion. Seven separately labeled fault-model cases exercise
pending/unknown quiescence, quarantined submission, failed quiescent completion,
not-submitted execution and invalid lease budgets. Unknown completion retains
resources. These state-machine checks do not substitute for driver fences.

## Reproduction and integration

The current integration branch already provides
`tests/gnet_slices/main.rs`, which imports `cpu_tests.rs`, and the
public general-net module. This corpus adds four tests inside the owned benchmark
directory; it requires no root adapter edit. Core-only checkouts need main's
production module and existing root adapter before running these tests.

```sh
INVAR_FULL_CORPUS=1 INVAR_GNET_CORPUS_REPORT_DIR=/tmp/new-child-debug \
  cargo test -p invar -p invar-tools --offline --locked --test gnet_slices \
  --test issue_gnet_05_parity_report -- --nocapture
INVAR_FULL_CORPUS=1 INVAR_GNET_CORPUS_REPORT_DIR=/tmp/new-child-release \
  cargo test -p invar -p invar-tools --offline --locked --release --test gnet_slices \
  --test issue_gnet_05_parity_report -- --nocapture
cargo fmt --all -- --check
cargo clippy --offline --locked --all-targets -- -D warnings
```

Rust builds use the existing lockfile. Generated models use C++17 for ABI4 and
C++20 for ABI5, `-O1 -fsanitize=undefined -fno-sanitize-recover=all`, bounded
compilation/execution and finite continuation submission caps. Getter-derived
layout/fork metadata, emitted source and the exact executable harness are
archived with the report. Execution exclusions stay in every denominator:
aggregate/view/blob/UTF8/Result forms unsupported by the published compiler,
ABI5 owned programs, frontend rejections and unrepresented F32 remain explicit.

