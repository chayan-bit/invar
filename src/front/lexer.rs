#[derive(Debug, PartialEq, Eq)]
enum TokenKind {
    Name(String),
    Integer(String),
    Decimal(String),
    StringLit(String),
    BytesLit(Vec<u8>),
    Keyword(&'static str),
    Symbol(&'static str),
    Eof,
}

#[derive(Debug, PartialEq, Eq)]
struct Token {
    kind: TokenKind,
    span: Span,
}

fn keyword(word: &str) -> Option<&'static str> {
    Some(match word {
        "fn" => "fn",
        "module" => "module",
        "frame" => "frame",
        "component" => "component",
        "extern" => "extern",
        "connect" => "connect",
        "flow" => "flow",
        "product" => "product",
        "system" => "system",
        "import" => "import",
        "pub" => "pub",
        "struct" => "struct",
        "const" => "const",
        "enum" => "enum",
        "match" => "match",
        "read" => "read",
        "mut" => "mut",
        "take" => "take",
        "copy" => "copy",
        "closure" => "closure",
        "i32" => "i32",
        "u32" => "u32",
        "Bool" => "Bool",
        "Unit" => "Unit",
        "Array" => "Array",
        "View" => "View",
        "MutView" => "MutView",
        "from" => "from",
        "let" => "let",
        "var" => "var",
        "return" => "return",
        "if" => "if",
        "else" => "else",
        "while" => "while",
        "for" => "for",
        "in" => "in",
        "or" => "or",
        "and" => "and",
        "not" => "not",
        "move" => "move",
        "true" => "true",
        "false" => "false",
        _ => return None,
    })
}

fn unsupported(word: &str) -> bool {
    matches!(
        word,
        "async"
            | "await"
            | "scope"
            | "break"
            | "continue"
            | "total"
            | "effects"
            | "requires"
            | "ensures"
    )
}

fn ident_start(byte: u8) -> bool {
    byte.is_ascii_alphabetic() || byte == b'_'
}

fn ident_continue(byte: u8) -> bool {
    ident_start(byte) || byte.is_ascii_digit()
}

fn quoted(source: &str, index: &mut usize, bytes_literal: bool) -> Result<TokenKind, FrontError> {
    let input = source.as_bytes();
    let start = *index;
    *index += if bytes_literal { 2 } else { 1 };
    let mut value = Vec::new();
    while *index < input.len() {
        let at = *index;
        match input[at] {
            b'"' => {
                *index += 1;
                if value.len() > 1_048_576 {
                    return Err(FrontError::tool("RESOURCE_LIMIT", "literal exceeds one MiB"));
                }
                return if bytes_literal {
                    Ok(TokenKind::BytesLit(value))
                } else {
                    String::from_utf8(value)
                        .map(TokenKind::StringLit)
                        .map_err(|_| FrontError::source("LEX_INVALID_LITERAL", "text literal is not UTF-8", Span { start, end: *index }))
                };
            }
            b'\n' | b'\r' => break,
            b'\\' => {
                *index += 1;
                let Some(&escaped) = input.get(*index) else { break };
                match escaped {
                    b'"' | b'\\' => value.push(escaped),
                    b'n' => value.push(b'\n'),
                    b'r' => value.push(b'\r'),
                    b't' => value.push(b'\t'),
                    b'x' if bytes_literal => {
                        let digits = input.get(*index + 1..*index + 3).ok_or_else(|| FrontError::source("LEX_INVALID_LITERAL", "byte escape needs two hex digits", Span { start: at, end: (*index + 3).min(input.len()) }))?;
                        let hex = std::str::from_utf8(digits).map_err(|_| FrontError::source("LEX_INVALID_LITERAL", "byte escape needs two hex digits", Span { start: at, end: *index + 3 }))?;
                        value.push(u8::from_str_radix(hex, 16).map_err(|_| FrontError::source("LEX_INVALID_LITERAL", "byte escape needs two hex digits", Span { start: at, end: *index + 3 }))?);
                        *index += 2;
                    }
                    _ => return Err(FrontError::source("LEX_INVALID_LITERAL", "unsupported string escape", Span { start: at, end: *index + 1 })),
                }
                *index += 1;
            }
            _ => {
                let ch = source[at..].chars().next().ok_or_else(|| FrontError::source("LEX_INVALID_LITERAL", "invalid string", Span { start, end: at }))?;
                value.extend_from_slice(&input[at..at + ch.len_utf8()]);
                *index += ch.len_utf8();
            }
        }
        if value.len() > 1_048_576 {
            return Err(FrontError::tool("RESOURCE_LIMIT", "literal exceeds one MiB"));
        }
    }
    Err(FrontError::source("LEX_INVALID_LITERAL", "unterminated string literal", Span { start, end: *index }))
}

fn lex(source: &str) -> Result<Vec<Token>, FrontError> {
    let bytes = source.as_bytes();
    let mut tokens = Vec::new();
    let mut i = 0;
    while i < bytes.len() {
        if matches!(bytes[i], b' ' | b'\t' | b'\n' | b'\r') {
            i += 1;
            continue;
        }
        let start = i;
        if bytes[i..].starts_with(b"//") {
            i += 2;
            while i < bytes.len() && bytes[i] != b'\n' {
                i += 1;
            }
            continue;
        }
        if bytes[i..].starts_with(b"/*") {
            i += 2;
            let mut depth = 1;
            while i < bytes.len() && depth > 0 {
                if bytes[i..].starts_with(b"/*") {
                    depth += 1;
                    if depth > MAX_NESTING {
                        return Err(FrontError::tool(
                            "SOURCE_NESTING_LIMIT",
                            "comment nesting limit exceeded",
                        ));
                    }
                    i += 2;
                } else if bytes[i..].starts_with(b"*/") {
                    depth -= 1;
                    i += 2;
                } else {
                    i += 1;
                }
            }
            if depth != 0 {
                return Err(FrontError::source(
                    "LEX_INVALID_TOKEN",
                    "unterminated block comment",
                    Span {
                        start,
                        end: bytes.len(),
                    },
                ));
            }
            continue;
        }
        let kind = if bytes[i..].starts_with(b"b\"") {
            quoted(source, &mut i, true)?
        } else if ident_start(bytes[i]) {
            i += 1;
            while i < bytes.len() && ident_continue(bytes[i]) {
                i += 1;
            }
            let word = &source[start..i];
            if unsupported(word) {
                return Err(FrontError::source(
                    "UNSUPPORTED_FEATURE",
                    format!("unsupported reserved keyword '{word}'"),
                    Span { start, end: i },
                ));
            }
            keyword(word).map_or_else(|| TokenKind::Name(word.to_owned()), TokenKind::Keyword)
        } else if bytes[i].is_ascii_digit() {
            i += 1;
            while i < bytes.len() && bytes[i].is_ascii_digit() {
                i += 1;
            }
            let digits_end = i;
            if i + 1 < bytes.len() && bytes[i] == b'.' && bytes[i + 1].is_ascii_digit() {
                i += 2;
                while i < bytes.len() && bytes[i].is_ascii_digit() {
                    i += 1;
                }
                let digits_end = i;
                while i < bytes.len() && ident_continue(bytes[i]) {
                    i += 1;
                }
                if i > digits_end && !matches!(&source[digits_end..i], "f32" | "f64") {
                    return Err(FrontError::source(
                        "LEX_INVALID_LITERAL",
                        "malformed decimal literal",
                        Span {
                            start,
                            end: digits_end + 1,
                        },
                    ));
                }
                tokens.push(Token {
                    kind: TokenKind::Decimal(source[start..i].to_owned()),
                    span: Span { start, end: i },
                });
                continue;
            }
            while i < bytes.len() && ident_continue(bytes[i]) {
                i += 1;
            }
            if i > digits_end && !matches!(&source[digits_end..i], "i32" | "u32") {
                return Err(FrontError::source(
                    "LEX_INVALID_LITERAL",
                    "malformed integer literal",
                    Span { start, end: i },
                ));
            }
            TokenKind::Integer(source[start..i].to_owned())
        } else if bytes[i] == b'"' {
            quoted(source, &mut i, false)?
        } else {
            let two = if i + 1 < bytes.len() {
                &bytes[i..i + 2]
            } else {
                &[]
            };
            let symbol = match two {
                b"->" => Some("->"),
                b"::" => Some("::"),
                b".." => Some(".."),
                b"==" => Some("=="),
                b"!=" => Some("!="),
                b"<=" => Some("<="),
                b">=" => Some(">="),
                b"=>" => Some("=>"),
                _ => None,
            };
            if let Some(symbol) = symbol {
                i += 2;
                TokenKind::Symbol(symbol)
            } else {
                let symbol = match bytes[i] {
                    b'(' => "(",
                    b')' => ")",
                    b'{' => "{",
                    b'}' => "}",
                    b'[' => "[",
                    b']' => "]",
                    b',' => ",",
                    b':' => ":",
                    b'.' => ".",
                    b';' => ";",
                    b'=' => "=",
                    b'<' => "<",
                    b'>' => ">",
                    b'+' => "+",
                    b'-' => "-",
                    b'*' => "*",
                    b'/' => "/",
                    b'?' => "?",
                    b'@' => "@",
                    b'%' => "%",
                    b'#' => "#",
                    _ => {
                        let end = start + source[start..].chars().next().map_or(1, char::len_utf8);
                        return Err(FrontError::source(
                            "LEX_INVALID_TOKEN",
                            "invalid source token",
                            Span { start, end },
                        ));
                    }
                };
                i += 1;
                TokenKind::Symbol(symbol)
            }
        };
        tokens.push(Token {
            kind,
            span: Span { start, end: i },
        });
    }
    tokens.push(Token {
        kind: TokenKind::Eof,
        span: Span { start: i, end: i },
    });
    Ok(tokens)
}

/// Byte spans of the matched `{`...`}` pairs in `source`, in closing order,
/// for native debug scopes (#305). Comments and literals are skipped by the
/// lexer, so their braces never count.
#[cfg(feature = "native")]
pub(crate) fn brace_spans(source: &str) -> Result<Vec<Span>, FrontError> {
    let mut open = Vec::new();
    let mut spans = Vec::new();
    for token in lex(source)? {
        match token.kind {
            TokenKind::Symbol("{") => open.push(token.span.start),
            TokenKind::Symbol("}") => {
                if let Some(start) = open.pop() {
                    spans.push(Span {
                        start,
                        end: token.span.end,
                    });
                }
            }
            _ => {}
        }
    }
    Ok(spans)
}

