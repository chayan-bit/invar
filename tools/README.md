# tools

The `invar-tools` package: acceptance, qualification and benchmark tooling.
It depends on the `invar` package as a library; the compiler, the CLI and the helpers the compiler spawns (`invar-nvrtc`, `invar-sequencer`) stay in `invar`.

| Path | Contents |
| --- | --- |
| `src/bin/invar-bench.rs` | Benchmark harness: manifest validation, generated and baseline timings, domain suites, GNET parity reports. |
| `src/bin/invar-heldout.rs` | Held-out corpus proposal generator and checker. |
| `src/bin/invar-performance-report.rs` | M5-03 performance acceptance report. |
| `src/bin/invar-qualify.rs` | M5-02 full-pipeline qualification (CPU semantics, device rows with `--device`). |
| `src/bin/invar-pathsum-gate.rs` | P4a path-sum gate protocol. |
| `src/bin/invar-m3-03-cold.rs` | Fresh-process CUDA timing control (feature `cuda`). |
| `bench/` | Harness modules, manifests, corpora, baselines, domain suites and the `[[bench]]` and `[[example]]` timing targets; see [bench/README.md](bench/README.md). |
| `qualification/` | The M5-02 qualification harness compiled into `invar-qualify` and its tests. |
| `tests/` | Tests that launch these binaries or compile the harness. |

Tests here launch the `invar` package binaries (`invar`, `invar-nvrtc`) from the target directory next to their own executable, so build both packages together: `cargo test -p invar -p invar-tools`.
The features `native`, `cuda` and `metal` forward to `invar`.
`bench/general-net-v01` and `bench/task-region-v01` at the repository root are corpus data that `invar` tests read; the harness code for them lives here.
