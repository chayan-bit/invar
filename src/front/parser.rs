include!("parser/cursor.rs");
include!("parser/decl.rs");
include!("parser/types.rs");
include!("parser/statements.rs");
include!("parser/expressions.rs");

fn if_statement(
    start: usize,
    condition: Expr,
    then_block: Block,
    else_arm: Option<ElseArm>,
) -> Result<Stmt, FrontError> {
    let end = match &else_arm {
        Some(ElseArm::If(stmt)) => stmt.span.end,
        Some(ElseArm::Block(block)) => block.span.end,
        None => then_block.span.end,
    };
    Stmt::new(
        StmtKind::If {
            condition,
            then_block,
            else_arm,
        },
        Span { start, end },
    )
}

/// Parse one bounded UTF-8 source file without type checking or execution.
/// Invalid encoding and resource limits are tool errors; malformed source is a source error.
/// Source text is capped at 1 MiB and lexical/syntactic nesting at 32.
/// Parser internals do not panic for accepted bounded input.
pub fn parse(bytes: &[u8]) -> Result<Program, FrontError> {
    if bytes.len() > MAX_SOURCE_BYTES {
        return Err(FrontError::tool(
            "SOURCE_SIZE_LIMIT",
            "source file exceeds 1 MiB",
        ));
    }
    let source = std::str::from_utf8(bytes)
        .map_err(|_| FrontError::tool("INVALID_UTF8", "source file is not valid UTF-8"))?;
    let tokens = lex(source)?;
    Parser {
        tokens,
        pos: 0,
        depth: 0,
        closures: Vec::new(),
    }
    .program()
}

include!("system.rs");
