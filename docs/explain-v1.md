# Explain JSON v1

`invar explain INPUT --format json` reads and checks one Invar source file, lowers it to verified MIR, then prints one JSON document. Successful analysis exits 0 even when every region is rejected. Source and tool errors use the existing `source_error`/`tool_error` documents and exit codes 1/2.

The success document has `schema_version: 1`, `status: "ok"`, `diagnostics: []`, `source_identity`, and `regions` in source/MIR order. `source_identity` is `fnv1a64:` followed by 16 lowercase hexadecimal digits from FNV-1a (64-bit offset `0xcbf29ce484222325`, multiplier `0x100000001b3`) over exact source bytes. This checksum establishes identity, not authenticity.

Each region has `function`, byte `span`, `candidate_identity`, `status` (`admitted` or `rejected`), `first_rejected_obligation` (null on admission), `rule_revision`, `rule_identities`, `numeric_policy`, `target`, `chosen_plan`, `fallback_plan`, `runtime_guard`, `facts`, and ordered `obligations`. An obligation has `name`, `evidence`, and `detail`; evidence is `checked_invariant`, `trusted_rule`, `runtime_guard`, or `unknown`. Unknown evidence rejects admission. `candidate_identity` is the same FNV-1a over region source bytes followed by UTF-8 rule revision and `wrapping-u32-v1/cpu`.

The initial candidate is a zero-based counted wrapping `u32` map with one or two ordered reads and one injective fresh-output store. `facts` records typed logical reads/writes, ownership, loans, effect order, trap/termination summaries, physical alias assumption, and cleanup policy. For rejected regions, unestablished facts and the guard are null. The guard contains `upper` (a constant or named array length), ordered `inputs`, and `output`. It requires the upper bound to be at most every current array length and 1,000,000; it is evaluated before any candidate write while the protected arrays remain stable. False or missing guard inputs select `mir-serial`.

Persisted documents are descriptive admission inputs. `admission::validate_serialized` bounds their size to 1,000,000 JSON bytes and re-analyzes current verified MIR and source bytes; any changed source, rule, policy, fact, or obligation rejects the record. A checksum or successful validation does not prove semantic equivalence by itself. Later extraction and execution must recheck the obligations at their own boundary.

Version 1 remains frozen apart from the approved `f64` extension below. Explicit sort metadata uses [Explain JSON v3](explain-v3.md) (v2 is historical); spatial metadata uses [Explain JSON v4](explain-v4.md).
The v1 schema was extended in place for v0.2 `f64` (optional `numeric_contracts`, reason `metal-no-f64`), approved by owner 2026-10-19; see [frozen contract changes](contract-changes.md).
