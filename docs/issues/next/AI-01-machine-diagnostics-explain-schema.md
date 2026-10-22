# AI-01 - Stable machine-readable diagnostics and versioned explain schema

Problem/result: make every Invar diagnostic and every `explain` document a stable, versioned machine interface, so that AI agents and tools can act on compiler output (read a reason code, jump to a span, apply a suggested fix) without scraping prose.

Dependencies: REN-01 (was M5-03; changed 2026-09-26 when v0.1 was reordered integration-first).

Release: v0.1 (milestone A1).

## Scope

`invar check --format json`, `invar run --format json` and `invar explain --format json` emit diagnostics under the schema `invar-diagnostics-v1`.
Each diagnostic carries a stable reason code taken from a committed registry file `schema/diagnostic-codes-v1.json`.
Every diagnostic the compiler can emit at this commit (front end, checker, ownership, evaluator traps, limits, tool errors, device unavailability, fallback reasons) receives a stable ID in that registry.
Each diagnostic carries a severity (`error`, `warning`, `note`), a primary span, zero or more secondary spans with labels, a message, and zero or more suggested fixes.
A span is byte offsets (`start`, `end`, half-open, UTF-8) plus 1-based `line` and `column` for both ends, and the source path.
A suggested fix is a list of text edits (`span`, `replacement`) plus a short description; edits within one fix never overlap.
The JSON Schema for diagnostics is committed as `schema/invar-diagnostics-v1.schema.json`.
The `explain` document emitted at this commit (including the device execution fields present at this commit; M4 and M5 now land after AI-01, so their explain fields arrive through explain schema version bumps with new schema files, as the semantics below require) is frozen as a versioned JSON Schema file `schema/explain-vN.schema.json`, where N is the explain schema version it emits.

## Semantics

A reason code is never reused for a different meaning and never renamed; retiring a code marks it deprecated in the registry.
Messages may change wording; codes, span semantics and fix edits are the contract.
Output ordering is deterministic (source order, then code).
The existing `schema_version: 1` diagnostic document either becomes `invar-diagnostics-v1` or is documented as its predecessor with a mapping; the exit codes (0 ok, 1 source error, 2 tool error, 3 trap, 4 evaluation limit) do not change.
Any change to the diagnostics or explain schema requires a schema version bump and a new schema file; the old file stays committed.

## Acceptance tests

Add the integration test binary `issue_ai_01_machine_diagnostics` (`tests/issue_ai_01_machine_diagnostics.rs`).
Run `cargo test --locked issue_ai_01_machine_diagnostics` in Debug and Release and require a nonzero test count.
Positive cases:
- every registry entry declares `reachability`: `source` or `invariant` (amended by D35);
- every `source` code has at least one real `.inv` fixture program that produces it through the public `invar` check, run or explain path, and every code the compiler emits appears in the registry (a test scans both directions);
- every `invariant` code (a guard on an internal invariant that checked source cannot violate, such as a MIR, typed-net or device-IR verifier failure or an evaluator type mismatch on a checked program) has a one-sentence `justification` of why checked source cannot reach it and a `fault_test` naming a targeted Rust fault-injection test that builds the malformed internal state through test-only constructors (`#[cfg(test)]` or a non-default `testing` feature, never public release API) and asserts the exact code and a schema-valid document;
- a corpus test runs every checked `.inv` program in the repository (examples, fixtures and the M2/M3 corpus) and fails if any of them emits an `invariant` code, which would mean the classification or the checker is wrong;
- every emitted diagnostics and explain document validates against its committed schema, using a maintained JSON Schema validator crate selected by a reuse survey;
- for every fix-bearing code (only `source` codes may carry fixes), a test applies the fix edits to the fixture source and proves the edited program then checks, and, where it runs, matches the reference evaluator;
- line and column agree with byte offsets for ASCII, multi-byte UTF-8 and CRLF sources;
- a schema compatibility test pins the frozen explain and diagnostics schema digests, so any schema change without a version bump fails.

## Negative cases

An unregistered code, a duplicate code, a registry entry without `reachability`, an `invariant` entry without `justification` or `fault_test`, an `invariant` code carrying a fix, a span outside the source, overlapping fix edits, a fix that does not make the program check, a schema-invalid document, and an explain field added without a version bump each fail a test.
A diagnostic that cannot compute a span emits `span: null` with a registered reason, never a fabricated span.

## Done criteria

The registry, both schema files and the compatibility test are committed; every existing diagnostic has a stable ID; the fix-application test passes for every fix-bearing code; all fixture outputs validate; Debug and Release runs pass with a nonzero count.

## Out of scope

No LSP or MCP server (AI-03), no incremental checking, no natural-language explanation generation, no model calls, and no change to language semantics or exit codes.

Bound diagnostic count per run, fix count per diagnostic, edit size, and document size.

## System integration impact (from S1)

Added 2026-09-26 by the integration plan (`docs/system-integration-v1.md`); existing content above is unchanged except where this section says so.

- Schema names: `invar-diagnostics-v1` for diagnostics and `invar-explain-v1` (with `invar-explain-device-v1` for device fields) for explain documents; REN-01 lands first, so no `invar-` schema name is ever committed.
- The registry `schema/diagnostic-codes-v1.json` reserves the range `S1000` to `S1999` for system integration, with the sub-ranges of design note section 5 (S1-01 `S1000`-`S1099`, S1-03 `S1100`-`S1399`, S1-02 `S1400`-`S1499`, S1-04 `S1500`-`S1599`, S1-05 `S1600`-`S1699`, S1-11 `S1700`-`S1799`, S1-06 `S1800`-`S1899`, reserved `S1900`-`S1999`).
- AI-01 commits the reserved ranges only; each S1 issue registers its codes in its range, and the both-directions registry test covers them.
- Diagnostics may carry an `ir_path` (a JSON pointer into `invar-system-v1`) in addition to, or instead of, a source span when a system check failure is best located in the IR; a diagnostic without a source span still follows the `span: null` rule.
