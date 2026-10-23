# Owned arrays and resumable CPU corpus, 2026-10-02

The corpus now retains **52 ordinary sources / 103 parsed function occurrences**,
**67 named scalar runtime vectors**, and **3 returned-owner root vectors**. All
47 sources and all 54 runtime vectors from the previous qualification remain
unchanged; five ordinary sources and thirteen scalar runtime vectors were added.
The initial 15-case manifest and parity gates remain unchanged. The sole prior
expectation adjustment is the raw-MIR implicit-return fuel delta, now zero on
the supplied canonical-fix integration base. Historical receipts are unchanged.

| Boundary | Ordinary sources / 52 | Scalar runtime vectors / 67 |
| --- | ---: | ---: |
| Source executed | 49 | 67 |
| Source rejected | 3 | 0 |
| Raw / checked MIR main execution | 49 / 49 | unavailable / unavailable |
| Independent GNET oracle executed | 45 | 66 |
| Independent GNET oracle unsupported | 4 | 1 |
| Scalar ABI1 CPU executed | 13 | 43 |
| Scalar ABI1 compiler unsupported | 36 | 24 |
| Owned ABI2 CPU executed | 40 | 60 |
| Owned ABI2 compiler unsupported | 9 | 7 |
| Resumable ABI3 CPU executed | 40 | 60 |
| Resumable ABI3 compiler unsupported | 9 | 7 |

Every source rejection and backend exclusion remains a row in both the full
ordinary corpus and the combined **119-row** ABI2/3 report. Compiler exclusions
are actual `UnsupportedType` responses, rather than filtered source families.
Any unexpected compiler error fails the runner. Source rejections remain the
borrow conflict, moved-owner reuse, and unsupported array-valued user variant.

The current MIR inventory contains32 rules: this integer/owner corpus represents
31 and explicitly excludes F32, which the current oracle does not support.
All31 integer/owner rules remain represented. These concrete inputs still
have four ordinary exclusions: two view cases, the `utf8_encode` intrinsic, and
an unsupported type in owned UTF-8 Result propagation. One runtime view vector
is also excluded. Whole-program admission counts are 99 function occurrences
for source/MIR/net, 92 for oracle, 29 for scalar compilation, and 85 for owned
and resumable compilation. Admission counts do not imply execution of every
function or acceptance of the full v0.1 language.

## Actual execution and independent expectations

`corpus_owned.rs` extends the existing Rust runner and consumes the actual
`emit_owned` / `emit_resumable` APIs. It compiles immutable emitted C++17 with
UBSan, retaining the existing scalar runner. Array allocation/index/mutation,
read/mut loans, moves, take parameters, owner returns, replacement, normal
cleanup, traps, fuel and depth execute through the emitted programs.

The new source cases cover recursive owner transfer, mutable loans across heap
compaction, runtime-selected independent worklist roots, runtime-selected older
divergence before a later array trap, and unsigned owned-root returns. Their
results and selected cleanup sequences are hand-derived in the source/runtime
manifests. The returned-owner vectors independently expect `[3,5]`, an empty
u32 array, and three `4294967295u32` elements. Named source observation is scalar
only; it is explicitly unavailable for those three root results, while each
producer's original source main and raw/checked MIR execute separately.

Every admitted ordinary/runtime entry is executed under ABI2, then under ABI3
with runtime quantum schedules `[1]`, `[3]`, `[17]`, and `[1,3,17]`. The same
artifact and executable receive different inputs and quanta. No input result,
filename, hash or quantum is specialized into the emitted program. Each slice
checks monotonic source/physical counters, physical transition delta no greater
than the selected quantum, unchanged committed event and cleanup prefixes, and
zero output publication while yielded. A completed-task retry must leave output,
trace, cleanup and the entire heap unchanged.

CPU comparison checks status/result/trap byte spans/source steps/transitions/
maximum frames, every ordered Demand/Complete event, and every cleanup name,
length and originating frame. Stable allocation-ID counts are checked against
actually completed allocation events. Normal scalar results retain no live
owned descriptor. Returned owners retain exactly the live result handle; the
host snapshots its complete array, signedness and descriptor before process
storage is released. Reports include immutable artifact/layout hashes and full
private-word hashes; no GPU fence or hardware admission is inferred.

The seven prior owner programs retain **168 exhaustive positive-fuel prefixes**.
Each is executed once under ABI2 and under all four ABI3 schedules. Three root
array results also run under both ABIs and all schedules. A separate tight
three-element heap executes four temporary allocations through a mutable loan,
checks all six stable allocation IDs, and returns13 after compaction and cleanup.

## Worklist and resource accounting

The four-root fixture takes runtime modes0..3: older divergent loop, value17,
index trap after completed cleanup, and value23 with normal owner cleanup.
The actual CPU worklist:

1. Uses quantum1 and eight submissions: status14, zero roots completed,
   exactly two physical transitions in every persisted task, zero output words.
2. Resumes with quantum1 for512 submissions: status14, three roots completed;
   the divergent root is still yielded and has published no output.
3. Resumes that workset with quantum17: all four roots complete and match their
   independent bounded oracle observations. Previously completed roots' output,
   heap, trace and cleanup remain byte-for-byte unchanged.
4. Repeats with zero submissions: status0, submitted0, completed4 and no state
   or observation changes.

These roots are independent caller entries, not speculative source children.
The separate `ordered(0u32)` fixture proves no logical demand enters `later`
while the older source call exhausts fuel. Every ABI3 quantum schedule preserves
that complete event prefix and the original source's limit observation.

Heap elements, allocation IDs and cleanup-record caps have three distinct
backend statuses11/12/13. Each case executes under both ABIs and is recorded as
an explicit backend-budget failure while its source and unconstrained oracle
complete. The older private-arena vector also retains its requested lower
oracle cap; its matched heap-limit prefix is identified separately from source
semantics. Transition/trace caps retain their own outcomes. No cap is a new
source-language bound, divergence proof, seed restriction or parity result.

Per profile, **1,417 actual CPU/oracle comparisons** comprise56 scalar ABI1,
500 full-denominator ABI2/3 observations,855 owner-prefix/returned-root
observations, and6 worklist/compaction observations. Six additional backend
budget executions intentionally differ from the completing source. CUDA and
Metal ABI2/3 compilation, dispatch and timing are not attempted. The retained
ABI1 corpus only emits13 CUDA and13 Metal sources, without hardware execution.

## Qualification and integration

Final validation uses main
`c1f71cd9cb12805439bf6c0413d205dee3cc9a7c` plus this benchmark-only continuation
and the separate structural-inventory fix `e46a7ecf6ebbb127ecc22bb276f32a5b75800c6a`.
That integration contains the published ABI2/3 core source
`3a055a90045440ead2b4977cdf15593674716a42` and canonical MIR fix. Earlier runs
on main `e949fa9817bc3a8d59f127e98b4a39aef2304f5e` used a published core source patch. The real public
`invar::general_net` module is used. This continuation authors only benchmark files;
root Cargo, core/oracle/compiler, schemas, CI and shared wiring remain untouched.

Final Debug and Release each pass9 focused tests (6 corpus +3 existing parity
regressions),0 failed,0 ignored. CLI replay52/52 each, formatting and strict
all-target Clippy pass. Final semantic/physical/artifact receipts agree exactly.
The receipts are not retained in this repository.

Current main `c1f71cd` already contains the unchanged three-line root adapter
`tests/issue_gnet_v01_corpus_expansion.rs`. Final qualification reuses that
adapter without authoring or committing a shared-root change. Main can import
the two benchmark commits directly on that compatible core and run:

```sh
INVAR_FULL_CORPUS=1 INVAR_GNET_CORPUS_REPORT_DIR=/tmp/gnet-owned-debug-new cargo test -p invar -p invar-tools --offline --locked --test issue_gnet_v01_corpus_expansion --test issue_gnet_05_parity_report -- --nocapture
INVAR_FULL_CORPUS=1 INVAR_GNET_CORPUS_REPORT_DIR=/tmp/gnet-owned-release-new cargo test -p invar -p invar-tools --offline --locked --release --test issue_gnet_v01_corpus_expansion --test issue_gnet_05_parity_report -- --nocapture
cargo fmt --all -- --check
cargo clippy --offline --locked --all-targets -- -D warnings
```

The full denominator requires a few minutes of bounded CPU compilation per
profile; main integration uses a60-second compiler deadline with a2-second
forced termination grace. The original receipts used a 30-second deadline;
one unchanged281-state artifact takes32.11 seconds on the integration host.
Every executable
run a10-second deadline. No full repository suite, GPU/parity qualification,
automatic child extraction/speculation, release, merge or tag is claimed.
