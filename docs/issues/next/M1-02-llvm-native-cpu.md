# M1-02 — LLVM native CPU and trap ABI

Problem/result: provide a bounded native CPU implementation of verified M1-01 MIR while preserving Invar traps, cleanup, ordering, and host-safe failure behavior.

Dependencies: M1-01.

Contract and implementation: pin a tested LLVM release/toolchain in project evidence; lower only verified MIR to LLVM IR and a documented host executable ABI. Use checked arithmetic and explicit trap runtime calls; never attach unjustified `nsw`, `nuw`, `noalias`, alignment, `mustprogress`, or fast-math promises. Preserve source spans through trap paths and return normal `main` values with the specified process statuses. Keep LLVM behind a narrow Rust boundary and retain the reference evaluator.

Positive cases: native and reference executions agree on completed values, exact trap category/span, cleanup traces, and source-order effects for the bounded M0/M1 corpus. Negative cases: malformed MIR, unsupported target operations, invalid ABI requests, overflow, invalid division, and resource failure reject or report typed errors; generated IR inspection catches forbidden flags and false progress claims.

Acceptance interface and evidence: add `issue_m1_02_native_cpu_trap_abi`; run `cargo test --locked issue_m1_02_native_cpu_trap_abi` with the recorded LLVM environment. Archive target triple, LLVM version, IR snippets/checks, corpus results, and unavailable-platform checks. No native result is accepted without reference comparison or an explicit inconclusive record.

Exclusions/stop: no optimizer admission, interaction extraction, GPU, JIT service, or unbounded native recursion. Stop when the pinned CPU backend is reproducible, trap ABI evidence is review-ready, and unsupported MIR fails closed.

Bound the compile corpus, generated IR size, native call depth, and subprocess deadline. Treat missing LLVM or target support as unavailable evidence, never as a green result or permission to install an unreviewed toolchain.

## Concrete integration requirements

Implement the native build/run interfaces in docs/implementation-plan.md. Pin LLVM major, Rust binding version, target triple/data layout, object/link command and a versioned Rust runtime ABI. Lower checked arithmetic via overflow checks, guard division/remainder corner cases, and avoid unproved nsw/nuw/inbounds/noalias/mustprogress attributes. A terminal source trap reports its category/span and terminates without source unwinding; test in a subprocess. Cross-check completed source/MIR/native results in Debug and optimized builds.

Follow the shared acceptance interface in `docs/implementation-plan.md`: list and verify a nonzero test selection, run the issue suite in Debug and Release, and retain full milestone Cargo gates. Commands are required deliverables, not existing functionality in the planning checkout.
