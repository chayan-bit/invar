# M1 acceptance record

Scope: M1-01 and M1-03 landed with the initial Rust bootstrap. The M1 native and interaction foundations contain M1-02 and the
2026-09-23 revised M1-04/M1-05 contracts. The revised decision requires a
typed, demand-aware production graph and checked extraction. It does not
claim source translation into Lafont's three-agent reference calculus.

Environment: macOS Darwin arm64; Rust/Cargo 1.95.0; LLVM 21.1.8 from pinned
nixpkgs `e2587caef70cea85dd97d7daab492899902dbf5d`; Inkwell 0.8.0.
The LLVM Nix store path was restored from the same pinned cache after local
garbage collection; no remote compute or paid device was used.

## Reproducible checks

`LLVM_SYS_211_PREFIX` is the dev output path of the pinned LLVM 21.1.8
package (see `native-toolchain-v1.md`). Filtered test listing showed three
nonignored `issue_m1_04_typed_net_verifier` tests and four default/five
native `issue_m1_05_rewrite_extraction_equivalence` tests.

```sh
cargo fmt --all -- --check
cargo clippy --locked --all-targets -- -D warnings
cargo test --locked issue_m1_04_typed_net_verifier -- --list
cargo test --locked issue_m1_05_rewrite_extraction_equivalence -- --list
cargo test --locked issue_m1_04_typed_net_verifier
cargo test --locked --release issue_m1_04_typed_net_verifier
cargo test --locked issue_m1_05_rewrite_extraction_equivalence
cargo test --locked --release issue_m1_05_rewrite_extraction_equivalence
cargo test --locked
cargo test --locked --release
cargo build --locked
cargo test --locked --offline
LLVM_SYS_211_PREFIX=<pinned LLVM dev output> cargo clippy --locked --features native --all-targets -- -D warnings
LLVM_SYS_211_PREFIX=<pinned LLVM dev output> cargo test --locked --features native
LLVM_SYS_211_PREFIX=<pinned LLVM dev output> cargo test --locked --release --features native
```

All listed checks passed on this host. The default full suite selected 43
tests in both Debug and Release. The native feature selected 48 in both
profiles, including the M1-02 linked executable differential suite and the
M1-05 direct-versus-graph-extracted native comparison. The initial native
attempt failed because the pinned LLVM store output had been collected;
after restoring it and cleaning the stale `llvm-sys` build result, the same
native checks passed. `--locked --offline` is recorded separately from
normal locked runs.

## Behavioral evidence

- M1-04: finite typed term encoding/decoding and independent graph evaluation
  agree with the term oracle on demanded recursive arguments, undemanded
  branches, function-before-argument traps, and left-to-right named-call
  failures. Invalid arity/types, affine reuse, stale schema/proof/effect,
  removed/duplicated/typed-wrong MIR wires, missing cleanup, and small
  resource caps reject. The public MIR graph round trip preserves result,
  source step count, and cleanup trace for an owned array program.
- M1-05: `affine-beta-v1` replaces a closed total application with a `Let`
  in the extracted term; captured and failing cases retain the original.
  `total-u32-map-fusion-v1` changes two serial loops to one for empty,
  singleton, and `u32::MAX`-containing arrays, retaining ordered wrapping
  results. Stale proof/rule revisions reject. A swapped subtraction data wire
  changes both public MIR and linked native CPU output from 7 to -7, proving
  that extraction consumes graph connectivity. Direct and graph-extracted MIR agree with the source
  oracle and linked native CPU on the tested scalar, branch, and owned-array
  programs.

Limits: the map examples are internal typed regions. M1 does not automatically
recognize maps in public source; M2-02 owns that integration. The current
typed graph has finite scalar/affine and wrapping-u32 map rules, not general
effectful rewrites, GPU scheduling, or a source-to-three-agent correspondence.
No Linux/Windows native run, Miri run, CUDA hardware check, or performance
qualification is claimed.
