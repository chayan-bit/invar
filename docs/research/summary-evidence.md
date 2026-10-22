# Ordered affine summary probe

This is a bounded mathematical feasibility check, separate from the Rust Cargo M0 implementation. The probe uses pure Rust and no dependencies to test an ordered modular affine summary and a work-efficient prefix scan. It is evidence about this finite model only; it is not a proof, a compiler implementation, a GPU result, or a claim that summary associativity permits permutation.

The summary pair represents `f(x) = a*x+b (mod m)`. For an ordered left summary followed by a right summary:

```text
(a,b) ⊗ (c,d) = (c*a, c*b+d) mod m
identity = (1,0)
```

The tree scan pads only to the next power of two, performs an ordered Blelloch up/down sweep, and compares every inclusive output state with the sequential recurrence. Its combine count is `2*(P-1)+n` for non-empty input (`P = next_power_of_two(n)`), hence linear work with logarithmic tree depth; this is an operation-count model, not wall-clock performance.

The executable checks:

- exhaustive identity and associativity for every affine pair at moduli 4 and 8;
- 1,024 modulus-4 and 32,768 modulus-8 action-law checks against a direct scalar recurrence oracle using independent `u128` arithmetic;
- 5,000 deterministic xorshift-seeded full-width `u32` cases plus zero, singleton, odd, power-of-two, non-power-of-two, and ragged lengths;
- reversed operand order, which is rejected by a concrete noncommutativity witness;
- checked signed overflow regrouping, where `(MAX+1)+(-1)` traps but `MAX+(1+(-1))` succeeds;
- strict floating-point reassociation, where `(1e20 + -1e20) + 3` differs from `1e20 + (-1e20 + 3)`.

## Reproduction

Recorded environment: `rustc 1.94.0 (4a4ef493e 2026-03-02)`, `aarch64-apple-darwin`, LLVM 21.1.8. No permanent tool installation or remote compute was used.

```sh
rustc --edition=2024 -D warnings -C debuginfo=0 research/summary-laws/summary_laws.rs -o /tmp/invar-summary-laws-debug
/tmp/invar-summary-laws-debug
rustc --edition=2024 -D warnings -C opt-level=3 research/summary-laws/summary_laws.rs -o /tmp/invar-summary-laws-release
/tmp/invar-summary-laws-release
diff <(/tmp/invar-summary-laws-debug) <(/tmp/invar-summary-laws-release)
```

Expected machine-readable shape is one `status:"ok"` line with `exhaustive_mod4_checks:4128`, `exhaustive_mod8_checks:262272`, `action_mod4_checks:1024`, `action_mod8_checks:32768`, `generated_cases:5011`, and an identical checksum in both builds. Seed is `0x9e3779b97f4a7c15`; random lengths are 0–96 and explicit shape lengths reach 257. Root reproduction also passed `rustfmt --edition 2024 --check research/summary-laws/summary_laws.rs`. Initial review found the purported sequential oracle shared summary composition and a formatting failure; the final probe uses direct scalar state updates, adds exhaustive action checks and passes formatting.

The run is bounded to 5,011 generated/shape cases, uses far below 256 MiB, and is intended to complete well under one minute. Finite agreement does not establish a general parallel-scan law, source-language legality, strict floating-point transformation, or GPU speedup.
