# CPU corpus expansion, 2026-10-02

This records the historical ABI1 qualification at commit `f5ee22f`. Current
owned-array/resumable coverage and the canonical MIR fuel fix are documented
in [RESUMABLE.md](RESUMABLE.md). Historical receipts remain unchanged.

The expanded denominator is **47 ordinary source occurrences, 89 parsed function
occurrences, and 54 separately supplied runtime-input vectors**. All original
15 cases, their bytes and expectations remain unchanged in `manifest.json` and
are included verbatim in `expanded-manifest.json`. The original parity manifest
and thresholds remain unchanged. `expanded-parity-spec.json` differs only in
its source-corpus filename. These two consistency properties are executable
assertions, rather than a selected-subset claim.

`expanded/` adds 32 ordinary source files. Arrays, transfers, borrowed mutation,
replacement, scoped cleanup, views, record fields and owned fields, scalar
variants, owned Result payloads and `?`, UTF-8 blobs, loops, recursive runtime
inputs, ordered traps, checked/wrapping extremes, fuel and call-depth limits all
have concrete cases. The checked net inventory visits all **31/31 structural
MIR rules**; this establishes representation coverage for this corpus only.
It does not establish oracle, compiled-device or full language execution.

`expanded-expectations.json` and `runtime-vectors.json` hold hand-derived scalar
results, operator byte spans, exact small fuel boundaries and selected named
owner cleanup sequences. The owned Result fixture reuses the repository's real
L1-04 source; its success expectation follows five UTF-8 bytes plus a selected
error arm. Expected values are not captured from the implementation under test.
Source, raw MIR and checked-net-extracted MIR are each executed against these
expectations. Named runtime input tests call the original checked functions;
no wrapper-source specialization or synthetic call fuel is introduced.

| Boundary | Ordinary source denominator 47 | Runtime-vector denominator 54 |
| --- | ---: | ---: |
| Source executed | 44 | 54 |
| Source rejected | 3 | 0 |
| Raw / checked MIR main entries executed | 44 / 44 | 0 / 0; main-only shared API |
| GNET CPU oracle executed | 35 | 47 |
| GNET CPU oracle unsupported | 9 | 7 |
| Compiled C++17 CPU executed under UBSan | 13 | 43 |
| Compiled CPU unsupported | 31 | 11 |
| CUDA / Metal source emitted | 13 / 13 | not attempted |
| CUDA / Metal executed | 0 / 0 | 0 / 0 |

Source rejection accounts for `OWN_BORROW_CONFLICT`, `OWN_USE_AFTER_MOVE`, and
the current frontend's `UNSUPPORTED_FEATURE` for an array-valued user variant
payload. That last source remains in the denominator; owned builtin Result
payloads are a separately accepted case. Every compiled exclusion is the actual
`UnsupportedType` returned after admitting the entire program, including unused
functions. Oracle exclusions cover views, records, variants and blobs and record
the real pre-execution refusal. All other compiler errors fail the test rather
than being relabeled as unsupported.

The ordinary source denominator admits 85 function occurrences into checked
MIR/net, 71 into the oracle's whole-program admission, and 29 into compiled
whole-program admission. Those are admission counts, not claims that every
function executed. The oracle supports 21 of the full 31 structural rules at
this core checkpoint. All rows retain their source hashes, function counts,
actual statuses, reasons and observations.

The runtime oracle executes 47 vectors, including three **separate tool-budget
failures**: transition cap, trace cap and private array-arena cap. The source
reference still completes those programs. Forty-four runtime vectors have full
source/oracle semantic equivalence; three have the explicitly expected tool
failure. The emitted CPU matches the complete bounded oracle observation for
all 43 compiled runtime vectors, including the transition and trace limits.
The private arena failure remains a compiled type exclusion. No resource
failure is classified as a source trap or divergence.

There are also **168 exhaustive positive-fuel prefixes** across seven owner
programs. Each compares result/trap/limit, source fuel and completed cleanup
between source, checked MIR and the GNET oracle. Implicit array return completes
in 11 source/checked-net steps, while raw MIR requires 12. The runner explicitly
checks raw MIR exhaustion at fuel 11 and completion at 12. Explicit `move`
requires 12 on every path. No shared MIR fix or weakened source expectation is
included in this corpus.

CPU comparison includes every private output word and every ordered event word:
status, scalar result, trap byte span, source steps, physical transitions, maximum
frames, and Demand/Complete events. Reports store the SHA-256 of the full private
word sequence encoded as little-endian i64, alongside readable observations.
This keeps evidence compact; actual comparisons use all words. C++ compilation
has a 30-second enclosing deadline, execution a 10-second deadline, and uses
`-std=c++17 -O1 -fsanitize=undefined -fno-sanitize-recover=all`.
