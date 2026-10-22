# M0-02 — Lexer, parser, spans, and diagnostics

Objective: parse the exact bootstrap grammar and expose stable source diagnostics with authoritative UTF-8 byte spans.

Prerequisites: M0-01. Own front-end source and focused Cargo tests under the existing project layout; do not create a second AST or parser framework.

Requirements: validate UTF-8, CRLF, nested block comments, reserved keywords, longest-token behavior, malformed numeric literals, and all productions in `spec/grammar.ebnf`. Store zero-based `[start,end)` byte spans. Enforce parser nesting and source-size limits as tool errors. Emit the required diagnostic JSON fields and codes, including null spans only for file-level errors. Keep source errors distinct from CLI/tool errors.

Non-goals: type checking, ownership, evaluation, recovery parsing, formatter, language server, or accepting syntax outside the spec.

Validation commands: `cargo test --locked`; add focused tests that check valid `sum.inv`, invalid `12oops`, unterminated nested comments, invalid UTF-8, and a missing token with exact byte spans.

Positive acceptance: valid examples parse and diagnostics serialize deterministically. Negative acceptance: malformed literals are not split into valid tokens; unsupported later keywords are rejected; excessive nesting/size returns a tool error without unbounded host recursion. Stop when every grammar production and stated lexical edge has a runnable case.
