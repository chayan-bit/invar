# front

Source syntax, spans, lexical errors and AST construction.
The module root is ../front.rs.
parse builds Program or returns FrontError with a Span.
lexer.rs, parser.rs and system.rs handle tokens and source declarations.

See [the architecture overview](../../docs/architecture.md) for the full pipeline and [validation](../../docs/validation.md) for the checks.
Preserve diagnostics, source order, deterministic observations and receipt identities.
Run focused checks through the required host wrapper before the full suite.
