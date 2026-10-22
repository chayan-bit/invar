# S1-03 - Graph checks: timing, frames, resources, license and isolation

Problem/result: prove the structural properties of a whole system graph before anything is generated, so integration mistakes (rate mismatches, budget overruns, missing transforms, over-committed hardware, license violations) are compile errors with stable codes instead of field failures.

Dependencies: S1-01.

Release: v0.1 (milestone S1).

Design: `docs/system-integration-v1.md` sections 2, 4 and 6.

## Scope

Checks over the `invar-system-v1` IR, run by `invar system check` after S1-01 well-formedness, each a pure function of the IR:

1. Timing (codes `S1100` to `S1199`):
   - rate compatibility per connection under its policy: `sample-and-hold` accepts any producer and consumer rates; `decimate(n)` requires `producer_rate = n * consumer_rate` exactly; `queue(d)` requires `producer_rate <= consumer_rate` (queue drained at least as fast as filled) and a finite `d`;
   - every connection has a finite policy (no unbounded queues);
   - per flow, the sum of declared WCET budgets of components on the path, the declared latency budgets of the connections, and one consumer period of sampling delay for every `sample-and-hold` or `decimate` hop, is at most the flow budget;
   - per component, `wcet <= deadline <= 1 / rate`.
2. Frame-tree closure (codes `S1200` to `S1249`): every frame named by a port type or a transform is reachable from the single root; every tree edge used by a transform in any port type has a transform source (an `out` port of type `Transform<Child, Parent>` or `Stamped<Pose<...>>`, or a constant in a component `param`); at most one source per edge.
3. Resource budgets (codes `S1250` to `S1299`): sums per product of declared `cpu_cores`, `memory`, `gpu_memory` and per-bus bandwidth (rate times message size from the port type's C layout, for ports declared on a bus) are at most the product `capacity`; a component on a bus the product does not declare is rejected.
4. License and isolation (codes `S1300` to `S1399`): the lattice of design note section 4 over component SPDX expressions, the product's `distribution` and `commercial`, and each component's `isolation` and linking mode (`impl extern` manifests from S1-02 declare `link static | dynamic | process`); SPDX parsing through a maintained crate selected by a reuse survey (for example `spdx`), with the license list version recorded.

All arithmetic is exact rational arithmetic over the IR's decimals, so no check depends on rounding.
Every report produced from these checks (including the S1-06 evidence report) carries the sentence: "Deadlines and WCET values are declared budgets, not measurements, until R1-06 evidence exists."
`invar system check --format json` emits `invar-diagnostics-v1`, and `invar system report PATH` writes a deterministic Markdown summary of the checks with that sentence.

## Semantics

A check either passes or emits at least one error; warnings never change the result.
The checks are independent of declaration order, and the diagnostic order is deterministic (check family, then source order).
A license check result is an engineering check, not legal advice, and the report says so.

## Acceptance tests

Add the integration test binary `issue_s1_03_graph_checks`.
Run `cargo test --locked issue_s1_03_graph_checks` in Debug and Release and require a nonzero test count.
Positive cases:
- a 12-component fixture graph resembling the S1-06 cell (camera, perception stub, planner, controller, hardware interface, logger) passes every check, and its report matches a golden file;
- each connection policy has a passing fixture at the boundary (exact decimation, queue at equal rates, flow budget met with equality);
- license fixtures: Apache-2.0, BSD-2-Clause, BSD-3-Clause and MIT in-process pass; `LGPL-2.1-or-later` dynamic in-process passes; `GPL-2.0-only` as a separate unmodified process passes; `MPL-2.0` unmodified in-process passes; `internal` distribution with a GPL in-process component passes;
- the same fixture checked twice produces byte-identical diagnostics and report.

## Negative cases

Each is rejected with its registered code: `decimate(3)` between 1000 Hz and 500 Hz; a queue with producer faster than consumer; a flow exceeding its budget by 1 ms; `wcet > deadline`; a frame unreachable from the root; a transform edge with no source; two sources for one edge; CPU, memory, GPU memory or bus bandwidth over capacity by the smallest unit; GPL linked in-process for `proprietary-binary`; AGPL as a network-exposed separate process; LGPL static; MPL declared `modified`; `CC-BY-NC-4.0` or the NVIDIA non-commercial license with `commercial = true`; a proprietary license without a grant id; an unknown `LicenseRef-*` without a declared grant.

## Done criteria

All four check families, the report and the diagnostics are committed with fixtures; the test binary passes in Debug and Release with a nonzero count; every code is registered and has a positive or negative fixture.

## Out of scope

Measured WCET (R1-06), runtime monitors (R1-05), AADL export (S1-10), license analysis of the Nix image closure (S1-08), schedulability of ROS 2 executors.
