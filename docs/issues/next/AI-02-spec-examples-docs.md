# AI-02 - Compact language spec, verified example library, and docs reconciliation

Problem/result: give AI agents and people one compact, executable description of Invar v0.1 and a verified example library they can retrieve from, and bring every document in the repository in line with what v0.1 actually implements.

Dependencies: L1-06, S1-06, M5-03.

Release: v0.1 (milestone A1); this is the final v0.1 issue.

## Scope

Write one compact normative specification `docs/spec/invar-v0.1.md`, small enough to load into an AI agent's context (target under 30k tokens, measured with a stated tokenizer and recorded).
It covers syntax, types, ownership, numeric policy, limits, engines, the C ABI, the Python binding surface, diagnostics and explain schemas, and GPU targets, with normative statements distinguished from notes.
Every code block in the spec is tagged with its expected outcome (value, trap code, or diagnostic code) and is extracted and executed as a test against the reference evaluator.
Add a verified example library under `examples/`: idiomatic programs per family (maps, reductions, scans, transpose, stencil, matrix, segmented, gather/scatter, frontier, summaries, and the L1 language features), each with its expected output checked in CI.
Add a machine-readable index `examples/index.json` listing, for each example, its path, intent (one sentence), family, tags, and the spec section it illustrates, validated by a committed JSON Schema.
Perform the deferred documentation reconciliation: every document under `docs/` is updated to match implemented v0.1 behaviour (explain fields, limits, numeric policy, GPU targets, engines, exit codes), and stale statements are removed or moved to `docs/archive/`.

## Semantics

The spec is normative for v0.1; where the spec and the implementation disagree, the test fails and the disagreement is fixed or escalated, never papered over.
Documents may describe planned v0.2+ work only in a clearly labelled roadmap section.

## Acceptance tests

Add the integration test binary `issue_ai_02_spec_examples`.
Run `cargo test --locked issue_ai_02_spec_examples` in Debug and Release and require a nonzero test count.
Positive cases:
- every spec code block runs and matches its tagged outcome, with the block count reported;
- every example runs on the reference evaluator and every engine it declares, and matches its checked-in expected output;
- `examples/index.json` validates, lists every example exactly once, and every referenced spec section exists;
- the spec token count is below the target and recorded.

## Negative cases

An untagged spec code block, a block whose outcome differs, an example missing from the index, an index entry without a file, a dangling spec-section reference, and a docs link to a removed file each fail a test.

## Done criteria

The spec, example library, index and reconciled docs are committed; the test binary passes with nonzero counts; a reviewed list of every changed doc and every removed stale statement is recorded in the completion comment.

## Out of scope

No new language features, no tutorials beyond the example library, no website, no generated API reference beyond the spec, and no v0.2+ design documents.

## System integration impact (from S1)

Added 2026-09-26 by the integration plan (`docs/system-integration-v1.md`); existing content above is unchanged except where this section says so.

- AI-02 stays the last v0.1 issue; its dependencies now include S1-06 and M5-03 so that it follows both tracks.
- The spec also covers the system dialect (`component`, `extern component`, `connect`, `flow`, `frame`, `product`, `system`), the `invar-system-v1` IR, the graph checks and their proven-versus-measured statements, the foreign-manifest contract, and the ROS 2 and MuJoCo targets, still within the token target.
- `examples/` includes system examples (at least a minimal two-component system, one example per graph-check family, and the S1-06 cell under `examples/cells/`), each checked in CI with its expected diagnostics or generated-file hashes, and listed in `examples/index.json` with family `system`.
- The documentation reconciliation includes `docs/system-integration-v1.md`, which is updated to match what landed.
