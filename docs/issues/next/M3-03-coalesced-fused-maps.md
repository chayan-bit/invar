# M3-03 — Coalesced/fused map kernels and transfers

Problem/result: lower the admitted fused map family to a coalesced CUDA kernel with explicit host/device transfer and lifetime accounting.

Dependencies: M3-02.

Contract and implementation: preserve element order per source semantics, contiguous layout assumptions, bounds, no-overread access, transfer ownership, and completion waits. Fuse only operations already admitted by M2 and verified by M3-02. Keep a CPU path for small inputs, failed guards, unavailable device, transfer failure, and unsupported layout. Measure transfer and kernel boundaries separately.

Positive cases: empty, singleton, odd, ragged, and large arrays agree with serial/reference values and allowed traps; completion traces show storage remains live through the final access. Negative cases: noncoalesced/unknown layout, aliasing, transfer OOM, launch failure, partial completion, and failed profitability guard fall back without stale device use.

Acceptance interface and evidence: add `issue_m3_03_cuda_fused_maps`; run `cargo test --locked issue_m3_03_cuda_fused_maps` with mock/default fallback tests and the pinned hardware command when available. Record generated kernel hash, launch geometry, transfer bytes, timings, outputs, and unavailable-device results.

Exclusions/stop: no reductions, multi-kernel scheduler, persistent cache, or universal CUDA performance claim. Stop when one fused map is correct, bounded, and fallback-safe.

Bound grid dimensions, transfer bytes, temporary allocations, and completion waits. Compare device results only after synchronization and preserve the CPU result for every failure path. Report timing with hardware and residency context; a fast kernel does not override a failed contract.

Keep generated code and launch metadata reviewable, and retain a source span for each operation that contributes to the fused kernel.

Record residency explicitly.

## Concrete integration requirements

Generate a reusable grid-stride fused map schedule with coalesced scalar loads/stores and explicit tail masks. Test lengths 0,1,31,32,33,255,256,257 and 4093, equal/unequal guards, several stage compositions and nonaliasing outputs. Charge copies and allocation to host-to-host timing. Capture generated code and actual forced-GPU execution evidence; passing the CPU fallback is not acceptance of this issue.

Follow the shared acceptance interface in `docs/implementation-plan.md`: list and verify a nonzero test selection, run the issue suite in Debug and Release, and retain full milestone Cargo gates. Commands are required deliverables, not existing functionality in the planning checkout.
