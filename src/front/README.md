# front

The lexer, parser and AST.
It turns source bytes into a `Program` whose nodes carry UTF-8 byte spans, or into a `FrontError` that says where the source is wrong.

## Where to start

- `parse(bytes)` in [parser.rs](parser.rs) is the entry point; it enforces the 1 MiB size and nesting limits declared in [../front.rs](../front.rs).
- [lexer.rs](lexer.rs) produces tokens; [items.rs](items.rs) holds the lookahead for item prefixes such as foreign functions.
- [system.rs](system.rs) parses the system dialect (`component`, `connect`, `system`); `parse_system` in [../front.rs](../front.rs) is its entry.
- [query.rs](query.rs) has small AST queries used by later passes.

## Position in the pipeline

First stage. Its output feeds [check](../check/README.md).

## Tests

- `cargo nextest run --locked --test language` (grammar, precedence, spans and limits are in `syntax_and_limits`).
- `cargo nextest run --locked --test diagnostics` for diagnostic documents.

## Rules to keep

- Spans are byte offsets into the exact source text.
- Unsupported syntax is a clear rejection, never a silent skip.

## Read next

- [grammar](../../spec/grammar.ebnf)
- [specification section 2](../../docs/spec/invar-v0.1.md#2-source-text)
