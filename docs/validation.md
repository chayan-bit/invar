# Validation contract

Status: M0 and M1-01 landed with the initial Rust bootstrap (M1-03, the untyped reference reducer, was retired and removed in 2026-10), and M1-02 through M1-05 (native LLVM CPU, the revised typed graph/source agreement, checked rewrites, and extraction) and the later increments are on `main`; open items are tracked in GitHub issues. The specification and example gate is `cargo test --locked --test issue_ai_02_spec_examples`. See [native evidence](native-toolchain-v1.md). Automatic source-loop recognition, CPU parallel scheduling, device acceptance, and source-to-primitive-agent encoding are outside M1. The historical document-only bootstrap was not compiler evidence.

## M0 required commands

These Cargo commands are the required gate:

```sh
cargo fmt --all -- --check
cargo clippy --locked --all-targets -- -D warnings
cargo test --locked
cargo test --locked --release
cargo build --locked
# after dependencies are provisioned:
cargo test --locked --offline
```

Cargo must run real behavioral tests; a successful build or zero-test run is not acceptance. Release tests must remain meaningful. Bootstrap code forbids unsafe Rust; optional bounded Miri is only for a pinned nightly and pure-core tests. Unsupported checks are recorded as unavailable, never green.

## Mandatory adversarial cases

| ID | Case | Expected |
| --- | --- | --- |
| LEX-01 | Nested/unterminated comments, CRLF, malformed literal suffix, invalid UTF-8 | Correct spans and distinct lexical/tool errors; no crash. |
| PARSE-01 | Precedence, unary minus, short-circuit, trailing commas, missing delimiter | Exact AST behavior or stable source rejection. |
| TYPE-01 | Mixed integer operands, integer condition, missing return, duplicate names, Unit parameter/local | Reject consistently; no implicit conversions. |
| NUM-01 | i32 extrema for add/sub/mul/negation, u32 underflow/overflow | Defined overflow trap or exact valid result. |
| NUM-02 | Division/remainder by zero; MIN/-1; MIN%-1; signed truncation | Correct category; MIN%-1 is 0. |
| NUM-03 | Wrapping u32 max + 1 and max * max | 0 and 1 respectively. |
| DEM-01 | Unselected branch contains infinite loop or runtime trap | Branch not executed; selected branch result preserved. |
| DEM-02 | Multiple possible failures in evaluated arguments | First source-order trap, with original operator span. |
| TOOL-01 | Fuel at a pinned boundary; recursion/nesting/input/allocation limit | Distinct bounded tool result; never claimed source divergence. |
| ARR-01 | Empty/singleton arrays, filled array, first/last/out-of-bounds access | Correct length/value or index trap; no unchecked host access. |
| ARR-02 | RHS changes state used by destination index; RHS traps; index tries to consume or mutably borrow its destination | RHS-first order; no premature store; invalidating destination-index access rejected. |
| OWN-01 | Move into take/binding/return followed by use | Reject stale owner; no implicit clone or double release. |
| OWN-02 | Move on one if branch; return on another branch | Conservative continuing-path join; terminating branch excluded. |
| OWN-03 | Move outer binding inside a loop vs move per-iteration local | Reject former, accept valid latter. |
| BORROW-01 | Same array passed read/read; mut/read; mut/mut; read/take | Accept only compatible loans. |
| BORROW-02 | First argument borrows, later nested call mutates/consumes | Reject conflicting later evaluation; preserve read/read nesting. |
| BORROW-03 | Return a read/mut parameter; mutate let/read; move borrower | Reject; mut parameter element stores accepted. |
| DROP-01 | Normal exit, nested scopes, return, move, replacement, trap | Exact test-internal lifetime trace; no semantic cleanup after trap. |
| CLI-01 | JSON escaping, filenames with spaces, no main, source error, trap, resource error | One valid JSON object, schema fields/exit code correct, no mixed logs. |
| EX-01 | Both committed examples | sum = 10, ownership = 41. |

Lifetime traces are an internal interpreter-test observation, not a new source IO primitive. The abstract spec controls which bookkeeping events represent language cleanup.

## Independent oracles

Expected arithmetic outcomes derive from mathematical boundaries and the spec, not by invoking the same arithmetic helper under test. Use named hand-derived cases. For small domains, exhaustive enumeration can supplement them. Bounded generated cases record seeds and preserve minimized failures as permanent fixtures. Random tests never replace the listed negative cases.

M0 has only reference execution. Do not label two calls to the same evaluator a differential backend comparison. M1 adds independently implemented native lowering and compares successful values, exact trap category/source, and required traces, with deliberate shared-bug probes. Additional runtime checkers validate the implementation within their scope; they do not prove language soundness.

## Native and optimization gates

M1 must inspect LLVM for accidental `nsw`/`nuw`, unjustified `noalias`/alignment, and progress promises. Arithmetic checks must dominate invalid division and other poisonous operations. Preserve operator source mapping in optimized trap paths. A wrapping operation is not allowed to acquire signed/unsigned no-overflow promises.

M1 also requires the separate typed connectivity verifier, and source/typed-graph/extraction agreement on supported fragments. Test malformed ports, forbidden ownership duplication, branch demand, wrapping-map fusion, beta rewriting, and stale evidence. M1's beta/fusion and connectivity mutation tests change extracted behavior or reject. M2 must route an automatically recognized ordinary source loop through this path; a hand-constructed map fixture alone cannot satisfy that later gate. See [parallelism foundations](parallelism-foundation.md).

M2 must mutate plan inputs/rule IDs/hashes/evidence to prove rejection; unknown information cannot admit a plan. Compare forced serial and chunked runs with worker counts 1, 2, and a bounded larger count. Include zero/tiny/ragged lengths, cutoff boundaries, alias attempts, and numeric extremes. Use available thread sanitizer tests for the scheduler's actual memory accesses. A false runtime guard must exercise the serial fallback; an invalid internal candidate must fail compilation.

## Evidence record

Every milestone acceptance records commit, host/architecture, compiler/toolchain versions, configuration, exact commands, test count, seeds, results, unavailable checks, and remaining limitations. Performance adds algorithms, baselines, sample count, uncertainty, data layout/residency, and timed boundaries. Do not sum cumulative telemetry or call a process exit a scientific acceptance result.

## Continuous integration

GitHub Actions runs CPU-only checks on GitHub-hosted Linux runners: Lint, Check, Hygiene, four Test shards and the CPU-model corpus test.
On pull requests the corpus sweeps (device CPU model, GNET corpus denominators, #54 selection parity and the GNET-02 native corpus and generated programs) run a deterministic 1-in-8 sample by path hash that keeps every corpus directory; pushes to main and `just test` set `INVAR_FULL_CORPUS=1` and run the whole corpus.
The workflow is [ci.yml](../.github/workflows/ci.yml).
CUDA and Metal suites are not run in CI.
Run them locally on a CUDA-capable Linux host or an Apple Silicon Mac when a change touches a device path, and record the hardware and result in the pull request.

`just test-gpu-cuda-remote <ref> [filters]` (`scripts/remote-cuda.sh`) runs a ref's CUDA suites on a remote Windows CUDA host over ssh.
It ships `git archive <ref>` to the host, builds with the per-user Rust `x86_64-pc-windows-gnullvm` toolchain linked by a per-user llvm-mingw, and runs the lib tests plus every test binary that names the `cuda` feature or `CudaDevice`, with `--include-ignored`.
Set `REMOTE_CUDA_HOST` to the host's ssh alias; the script names no host.
On the host, everything lives under `%USERPROFILE%\<REMOTE_CUDA_DIR>` (default `invar-remote-cuda`): `nvrtc\` holds the NVRTC DLLs, and the cargo home, target directory, sources, run logs and the runner lock sit beside it.
The pinned llvm-mingw lives under `%USERPROFILE%\toolchains\` (`REMOTE_CUDA_MINGW` overrides the path).
One run holds the lock at a time, and a run waits for three quiet GPU samples (at most `REMOTE_CUDA_MAX_UTIL` percent busy and `REMOTE_CUDA_MIN_FREE_MIB` free) before it starts; it exits 3 when the GPU stays busy, and never stops another tenant's work.
`REMOTE_CUDA_SHARED=1` runs beside another tenant when memory allows; such runs check correctness only and their timings are not valid.
Exit codes: 0 pass, 1 test failures, no test run, or a filter that matched no test, 2 build failure, 3 busy, 4 timeout, 5 ssh dropped (the run finished remotely; read the log).
The full log lands in `target/remote-cuda/<run>.log`.
The skip list in the script names Windows-only gaps, none of them a CUDA result; `REMOTE_CUDA_NO_SKIP=1` runs them, `REMOTE_CUDA_EXTRA_SKIPS` adds skips and `REMOTE_CUDA_TESTS` replaces the target list.

## Later implementation and hardware acceptance

After M0, use each GitHub issue's acceptance contract. List filtered tests and require a nonzero relevant count before running them in Debug/Release. Default CPU CI and real CUDA qualification are distinct: skipped, mock-only or unavailable-device checks cannot close a hardware issue. [Performance acceptance](performance-acceptance.md) defines independent baselines, numerical policy and the frozen per-family targets; no performance claim follows from this specification or the standalone recurrence probe.

## Ordered affine-pair source form

The [pair-reduction contract](pair-reduction-v1.md) defines the sealed modular law and its source/CPU/device boundaries.
Run `cargo test --locked --test device issue_v01_pair_reduction` in Debug and Release on the authorized CPU host.
Run the ignored `pair_reduction::` tests of the `gpu_cuda` target with `--features cuda` on qualified NVIDIA hardware, and of the `gpu_metal` target with `--features metal` on an Apple GPU.
Require nonzero listed counts and retain profile/host separation.
Pair source/reference/MIR/fuel/cleanup agreement, atomic two-scalar publication, maximum resource bounds and operator-order/non-associative-admission mutation failures are required.
Unsupported min/argmin and floating pair laws are not covered by this affine contract.
