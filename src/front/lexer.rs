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

/// A `///` (outer) or `//!` (inner) line comment: trivia kept beside the
/// tokens with the item it documents (#257). It never reaches checking.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DocComment {
    pub inner: bool,
    /// The comment after its marker and one optional space.
    pub text: String,
    pub span: Span,
    /// The documented item: the following item for `///`, and the file or
    /// enclosing module for `//!`.
    pub item: Span,
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

/// The canonical token text of an integer literal (decimal digits plus the
/// optional suffix), or `None` when it is malformed. `_` may appear only
/// between two digits, and `0x`, `0o` and `0b` select the radix (issue #253).
fn integer_literal(word: &str) -> Option<String> {
    let (body, suffix) = match word.len().checked_sub(3).map(|at| word.split_at(at)) {
        Some((body, suffix @ ("i32" | "u32"))) => (body, suffix),
        _ => (word, ""),
    };
    let (radix, digits) = match body.get(..2) {
        Some("0x") => (16, &body[2..]),
        Some("0o") => (8, &body[2..]),
        Some("0b") => (2, &body[2..]),
        _ => (10, body),
    };
    let well_formed = digits
        .split('_')
        .all(|group| !group.is_empty() && group.chars().all(|digit| digit.is_digit(radix)));
    if !well_formed {
        return None;
    }
    let digits = digits.replace('_', "");
    if radix == 10 {
        return Some(digits + suffix);
    }
    // Radix values above u128 are refused here; every integer type is far narrower,
    // and a value in range of u128 reaches the ordinary range check unchanged.
    let value = digits.chars().try_fold(0u128, |value, digit| {
        value
            .checked_mul(u128::from(radix))?
            .checked_add(u128::from(digit.to_digit(radix)?))
    })?;
    Some(format!("{value}{suffix}"))
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
    lex_with_docs(source).map(|(tokens, _)| tokens)
}

fn lex_with_docs(source: &str) -> Result<(Vec<Token>, Vec<DocComment>), FrontError> {
    let bytes = source.as_bytes();
    let mut tokens = Vec::new();
    let mut docs = Vec::new();
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
            let comment = &source[start..i];
            let inner = comment.starts_with("//!");
            if inner || (comment.starts_with("///") && !comment.starts_with("////")) {
                let text = &comment[3..];
                let text = text.strip_prefix(' ').unwrap_or(text).trim_end_matches('\r');
                let (span, item) = (Span { start, end: i }, Span { start, end: start });
                docs.push(DocComment { inner, text: text.to_owned(), span, item });
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
            let span = Span { start, end: i };
            let value = integer_literal(&source[start..i]).ok_or_else(|| {
                FrontError::source("LEX_INVALID_LITERAL", "malformed integer literal", span)
            })?;
            if i > digits_end
                && bytes.get(i) == Some(&b'.')
                && bytes.get(i + 1).is_some_and(u8::is_ascii_digit)
            {
                return Err(FrontError::source(
                    "LEX_INVALID_LITERAL",
                    "a decimal literal takes no separator, radix or suffix before its point",
                    span,
                ));
            }
            TokenKind::Integer(value)
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
                b"+=" => Some("+="),
                b"-=" => Some("-="),
                b"*=" => Some("*="),
                b"/=" => Some("/="),
                b"%=" => Some("%="),
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
    Ok((tokens, docs))
}

/// The value of the last string literal token in `text`, if it lexes.
pub fn last_string_literal(text: &str) -> Option<String> {
    lex(text).ok()?.into_iter().rev().find_map(|token| match token.kind {
        TokenKind::StringLit(value) => Some(value),
        _ => None,
    })
}

/// Byte spans of the matched `{`...`}` pairs in `source`, in closing order,
/// for native debug scopes (DX-20). Comments and literals are skipped by the
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

