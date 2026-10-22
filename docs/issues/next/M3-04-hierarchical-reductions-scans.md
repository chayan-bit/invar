# M3-04 — Hierarchical reductions and ordered scans

Problem/result: extend one admitted modular reduction and ordered scan to hierarchical device execution without changing algebra or source order.

Dependencies: M3-03.

Contract and implementation: use only M2-03 laws and identities; define block-level summaries, merge order, synchronization, temporary ownership, and completion lifetimes. Ordered scans retain affine composition order; checked or strict floating operations remain on fallback. Validate neutral elements, partial blocks, non-power-of-two lengths, and transfer boundaries. Every device candidate carries numeric policy and a serial plan.

Positive cases: modular reductions and affine scans agree state-by-state or by the specified final observation with serial/reference for all bounded shapes. Negative cases: wrong identity, permutation, overflow-sensitive reassociation, ragged block handling, synchronization omission, temporary lifetime failure, and device error reject or fall back.

Acceptance interface and evidence: add `issue_m3_04_cuda_reductions_scans`; run `cargo test --locked issue_m3_04_cuda_reductions_scans` plus the pinned hardware integration command if available. Record law ID, block geometry, combine counts, scan depth, transfer/completion trace, and exact fallback reason.

Exclusions/stop: no arbitrary algebra inference, floating fast math, dynamic graph reducer, or performance generalization. Stop when the named operations have reproducible serial/device evidence.

Bound hierarchy depth, block count, temporary storage, and device synchronization. Keep law identifiers attached to generated plans and test identity/partial-block behavior independently. If strict numeric behavior cannot be maintained, report an admission miss and execute the CPU candidate.

No result is accepted solely because a device checksum matches one sample; bounded shape coverage and an independent serial oracle remain required.

Record the selected law and identity explicitly.

## Concrete integration requirements

Use separate kernel launches for block aggregates, ordered prefix/aggregate propagation, and final writeback; no global spin barrier. Specify inclusive/exclusive seed placement. Preserve left/right order for affine summaries, including non-power-of-two multi-block arrays. Track all temporary lifetimes and overflow-checked workspace sizes. Run real-device differential and supported sanitizer checks; no-device/mocks keep the hardware gate open.

Follow the shared acceptance interface in `docs/implementation-plan.md`: list and verify a nonzero test selection, run the issue suite in Debug and Release, and retain full milestone Cargo gates. Commands are required deliverables, not existing functionality in the planning checkout.
