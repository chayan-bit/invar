# Domain benchmark suites (ENG-10, #56)

Three suites of ordinary Invar loops: ML (`ml/`), games (`games/`) and robotics (`robotics/`).
Each `<kernel>.inv` is the kernel; `mod.rs` of each suite lists its shape and seed variants and the Rust oracle call that produces the expected values.
`mod.rs` generates a `main` per variant that fills its inputs from a seeded recurrence and returns 0 only when every output equals the oracle bit for bit.
The oracles in `oracle.rs` are plain Rust; the only shared pieces are the language's own `exp`, `sin` and `cos`.

The split unit is the kernel: a seeded 25% of each suite's kernels (rounded up) is held out, with every variant of a held-out kernel.
`manifest-v3-proposal.json` records the programs, hashes and split.
It is a proposal, not a freeze: the freeze is gated by #52.
Regenerate it with `invar-bench --write-domain-proposal bench/manifest-v3-proposal.json` from `tools/`.

`invar-bench --domain-report <ml|games|robotics|all>` prints per-program results, the worst five held-out programs and how many programs reached a parallel lowering on `cpu-parallel`, with the refusal reasons.
It times nothing.
Device timings and baseline comparisons (#109) are pending until real hardware runs them.

Quantized matmul (i8 storage, i32 accumulate) is recorded pending in the proposal until the ENG-03 storage types exist.

The no-keying rule is enforced by `tests/issue_eng_10_domain_suites.rs`: no file under `src/` or `crates/` may name a suite program, kernel, fixture path, held-out hash or spell a domain word (`ml`, `games`, `robotics`) as a string literal, in any position or case.
