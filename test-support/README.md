# Shared test support

Executable test helpers and fixture inputs belong here. The scalar-export and
lineage inputs were moved here byte-for-byte during v0.09 consolidation. Their consumers are the
scalar-export integration tests and named-evaluator unit tests.

This directory is intentionally outside the source-corpus discovery roots:
the fixtures already have explicit vector coverage, and helper source must not
recursively discover itself. Historical result files remain at their original
paths and are pinned to their original commits.

The original inputs remain recoverable from commit
`164dc101db77bbe6226de1a86f68cabfb425bc58`.

## Child wall deadlines under shared-host load

`child_deadline.rs` is test support, included by path, with no production API.
`INVAR_TEST_TIMEOUT_SCALE` is an explicit integer multiplier in `1..=64`, default
`1`. Invalid, zero, non-UTF-8 and overflowing values fail before spawning the
child. CI selects `8` for the shared runners. For a loaded host, for example:

```sh
INVAR_TEST_TIMEOUT_SCALE=8 cargo test --locked --offline --test gnet_slices
```

Run Cargo through the host's required resource wrapper. This setting adds
scheduling allowance to **wall time**, including OS scheduling delay after
spawn. It does not estimate load average or change semantic fuel, transition
budgets, CPU-seconds limits, memory/log caps or any product watchdog. Wrapper
queueing before the test starts is outside these timers. Compilation and
execution have separate watchdog calls and budgets; a long test containing
many compiles/runs is not one long child execution.

`wall(Duration)` supports Rust poll loops, `seconds(u64)` supports nested
watchdogs, and `command(u64)` builds GNU `timeout` with the scaled base budget
and an unscaled two-second TERM-to-KILL grace. It supervises the child's process
group. Timeout status is 124, forced kill is SIGKILL (shell status 137), and callers must retain
their normal failure checks. Existing typed product timeout tests keep their
exact injected short deadlines. The knob must never be read by product code.

`cxx(flags, source, exe)` compiles a C++ test harness with `$CXX` (else `c++`) under a 120-second base deadline and panics with the compiler's stderr on failure; `CXX17_UBSAN_O1`/`O2` are the emitted-CPU harness flag sets.

Sweep of compiled/process test deadlines:

- `bench/general-net-v01/corpus_{cpu,owned,children,numeric}.rs`: each compiler,
  generated program, capability/rejection probe, fair worklist and child fence.
  The fence observer allows three extra seconds for the child watchdog's
  two-second TERM/KILL grace and reaping.
- `tests/issue_gnet_{04_device,05_provenance_owned,09_provenance,10_f32_arrays,11_numeric_owned_state}.rs`:
  compiler/model/run watchdogs, including the asynchronous group child.
- `tests/issue_gnet_{12_cuda_owned_numeric_model,f32_helpers_flag_independence,resume_device_model}.rs`:
  the existing process-group poll runners. CPU limits remain unchanged.
- `tests/m4_transpose_stencil/issue_m4_02_{stencil_source,transpose_source}.rs`,
  `tests/m4_contraction/issue_m4_03_matrix_contraction_tensor_core.rs`,
  `crates/invar-device/tests/m4_02_transpose.rs`: compiler and CPU-model runs.
  Embedded `alarm()` wall timers now use the parent group watchdog instead;
  stencil's independent CPU limit remains unchanged.
- `crates/invar-cuda/src/cuda/{floating_tests,transpose_tests}.rs`: mock compiler
  children expecting success/diagnostics/caps scale; exact 50 ms typed timeout
  requests stay fixed.
- `tests/support/owner_return_cpu.rs`: nested timeout and parent watchdog agree;
  CPU/address-space/file/descriptor/core limits remain unchanged.
- `src/native.rs`: only the `cfg(test)` elapsed-time assertions around child
  output/draining and executable-writer checks; the real 30-second process
  watchdog and 250 ms executable-writer retry budget remain unchanged.
- `tools/tests/bench_tooling/issue_m5_01_build_lineage.rs`: successful fixture build budgets, executable
  and FIFO probes, and harness elapsed-time assertions. Deliberate 200/250 ms
  timeout requests retain their exact durations and typed `Error::Timeout`.
  Scaled non-timeout requests saturate at the product's existing 600-second
  request maximum, including at scale 64.

- `crates/invar-python/tests/{conftest,test_scalar_artifacts}.py`: existing C
  fixture compile and fresh-process scalar-artifact checks consume the same knob.

Product limits are intentionally unchanged: `src/native.rs`,
`src/component_test/planner.rs`, `src/general_net/{device_host,numeric_owned_resume_host}`,
`src/gnet_device_command.rs`, `tools/bench/{baseline,build_lineage,general_net_cli}.rs`,
and CUDA/Metal driver watchdogs. Tests of these bounds continue to exercise the
real contract; GPU completion polling and semantic timing fixtures are not
compiled-child harness watchdogs.

## Corpus sample

`corpus_sample.rs` walks the source corpus and picks the deterministic sample that pull-request CI runs for the exhaustive corpus sweeps: the device CPU-model corpus, the GNET v0.1 corpus denominators, the GNET-02 native corpus (with one generated program in 8) and the #54 selection-parity passes (which also keep every example, since the examples reach every family).
A program is kept when the 64-bit FNV-1a hash of its path falls in bucket 0 of 8, and each directory category the hash misses keeps its lowest-hash program, which the sampler asserts.
Count floors written for the whole corpus are divided by 8 for the sample; agreement checks are unchanged.
`INVAR_FULL_CORPUS=1` runs every program; `just test` and CI on pushes to main set it.

## Term oracle

`term_oracle.rs` is test support, included by path, with no production API.
It holds the call-by-value term evaluator and the demand-ordered observation of verified `typed_net` graphs that `issue_m1_04`, `issue_m1_05` and `issue_m4_07_typed_bounded_scan` use as independent oracles.
