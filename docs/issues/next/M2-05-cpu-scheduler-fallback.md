# M2-05 — CPU scheduler and serial fallback

Problem/result: execute admitted CPU candidates with bounded scoped workers and an always-available serial fallback.

Dependencies: M2-04.

Contract and implementation: use a reviewed standard Rust threading facility or narrowly scoped dependency; publish disjoint outputs only after admission proves physical nonoverlap. Define worker count, chunking, join, panic/error propagation, cancellation, cleanup, and allocation limits. Preserve source-order observations and failure prefix; unknown or failed guards run serially. No worker may outlive borrowed storage or continue visible work after terminal failure.

Positive cases: forced worker counts 1, 2, and a bounded larger count agree with serial output for maps/reductions/scans, including empty/tiny/ragged inputs and cleanup. Negative cases: overlapping chunks, worker panic, allocation failure, timeout, failed guard, unavailable platform, and cancellation select a safe fallback or typed tool error without use-after-free.

Acceptance interface and evidence: add `issue_m2_05_cpu_scheduler_fallback`; run `cargo test --locked issue_m2_05_cpu_scheduler_fallback`. Record worker counts, chunk boundaries, joins, errors, fallback decisions, and bounded resource observations. Report performance separately from legality.

Exclusions/stop: no async runtime, GPU, unbounded task pool, or claim of deterministic scheduling for unsupported effects. Stop when admitted CPU plans and serial fallback are reproducible and ownership-safe.

Bound worker count, queue/chunk metadata, join wait, and retained output memory. Exercise the serial path directly in every test so a scheduler regression cannot remove the only legal implementation. Record timing only as operational context, never as a substitute for semantic evidence.

All worker-owned state must be dropped after join or explicit failure acknowledgement, and a canceled operation must not free borrowed storage early.

## Concrete integration requirements

Implement forced cpu-parallel and workers interfaces from docs/implementation-plan.md. Use scoped standard-library workers and contiguous disjoint slices first; no tasks for scalar functions or tiny serial-selected regions. Inject spawn/join failure before publication and account for ownership on all paths. Compare 1, 2 and 4 workers on noncommutative scans and ragged partitions, and test false guards before any worker starts.

Follow the shared acceptance interface in `docs/implementation-plan.md`: list and verify a nonzero test selection, run the issue suite in Debug and Release, and retain full milestone Cargo gates. Commands are required deliverables, not existing functionality in the planning checkout.
