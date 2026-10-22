# M3-02 — Device IR/verifier and generated CUDA toolchain

Problem/result: define a bounded device representation and verifier that can generate CUDA only from admitted physical plans.

Dependencies: M3-01.

Contract and implementation: freeze one device IR version covering types, layouts, bounds, address spaces, transfers, synchronization, numeric policy, ownership/completion obligations, and source spans. Verify actual accesses, launch dimensions, resource limits, and required toolchain capabilities before code generation. Generated CUDA is allowed; handwritten kernels are not a substitute for verified plans. Unknown layouts, transfers, attributes, or tool versions reject and select CPU fallback.

Positive cases: one admitted map plan verifies, generates, compiles with the pinned toolchain, and carries its contract to M3-01. Negative cases: overreads, aliasing, stale IR, invalid launch geometry, unsupported arithmetic, missing synchronization, unsafe completion, toolchain mismatch, and compilation failure reject cleanly.

Acceptance interface and evidence: add `issue_m3_02_device_ir_verifier`; run `cargo test --locked issue_m3_02_device_ir_verifier`, with a no-CUDA verifier suite in default CI and an explicitly recorded toolchain integration command when available. Archive IR, verifier diagnostics, generated source hash, compiler version, and fallback evidence.

Exclusions/stop: no reductions, transfers optimization, cache, or broad CUDA dialect. Stop when one device plan is fail-closed and reproducible.

Bound IR nodes, generated source size, launch dimensions, and compiler subprocess time. Record the exact generated artifact and toolchain inputs. If a verifier cannot establish layout or lifetime facts, the correct result is CPU fallback or explicit admission failure.

The no-device verifier suite must exercise the same malformed plans used by hardware integration, so hardware availability cannot hide an invalid IR path.

## Concrete integration requirements

Use Rust-emitted CUDA C++ through NVRTC and Driver API per docs/research/backend-options.md. Freeze compiler options, capability, generated entry signature and argument packing; no Rust struct layout in the device ABI. Device IR carries lane/block loops, masks, barriers and physical addresses. Verify uniform block barriers and correct active-lane masks. Add required toolchain compilation plus real GPU smoke evidence under feature cuda; absence leaves acceptance pending.

Follow the shared acceptance interface in `docs/implementation-plan.md`: list and verify a nonzero test selection, run the issue suite in Debug and Release, and retain full milestone Cargo gates. Commands are required deliverables, not existing functionality in the planning checkout.
