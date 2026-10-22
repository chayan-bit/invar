//! MSL translation units for the binary32 library, built from the vendored
//! CORE-MATH C text.
//!
//! Metal has no binary64 and flushes binary32 subnormals, so the prelude
//! (`msl_prelude.h`) defines `double` as `D` and `float` as `F`, encodings
//! whose operations are integer-implemented and correctly rounded. This pass
//! makes the C text acceptable to MSL without changing its meaning:
//! - comments are removed and `crate::cxx` applies the C++ rewrites;
//! - every unsuffixed floating literal becomes `D::raw(<its exact binary64
//!   encoding>)`, so no constant passes through a binary32 parse;
//! - `static const` tables move to program scope in the `constant` address
//!   space (function-local ones are hoisted with the macros their function
//!   defines and renamed; tables whose initializers compute and function-local
//!   `static const` scalars become plain locals);
//! - pointers that view tables are `constant`, pointer parameters that
//!   return values are `thread`, and `unsigned __int128` is `invar_u128`.
//!
//! Without `__METAL_VERSION__` the prelude maps the address spaces to host
//! C++, so the same text is checked on the host before it reaches a GPU.

use super::{Function, source};
use crate::device::{namespace, splice};

/// The prelude every MSL unit starts with.
pub const MSL_PRELUDE: &str = include_str!("msl_prelude.h");

/// Types a pointer parameter may point to in upstream's binary32 files.
const POINTEES: [&str; 8] = [
    "int",
    "double",
    "float",
    "uint32_t",
    "uint64_t",
    "int64_t",
    "fexcept_t",
    "unsigned",
];

/// The qualified entry point of `f`'s binary32 function, e.g.
/// `invar_cm_sin_32::cr_sinf`.
pub fn msl_symbol(f: Function) -> String {
    format!("{}::cr_{}f", namespace(f, false), f.name())
}

/// One MSL (or host C++) translation unit holding the binary32 functions
/// requested, deduplicated. Callers append their kernels.
pub fn msl_source(functions: &[Function]) -> String {
    let mut out = String::from(MSL_PRELUDE);
    let mut seen = Vec::new();
    for &f in functions {
        if seen.contains(&f) {
            continue;
        }
        seen.push(f);
        let (text, headers) = source(f, false);
        let (mut unit, mut defined) = (String::new(), Vec::new());
        splice(text, headers, &mut unit, &mut defined);
        let unit = crate::cxx::rewrite(&crate::cxx::uncommented(&unit));
        out.push_str(&format!("namespace {} {{\n", namespace(f, false)));
        out.push_str(&metal(&unit));
        out.push_str("\n}\n");
        for name in defined {
            out.push_str(&format!("#undef {name}\n"));
        }
    }
    out
}

fn metal(text: &str) -> String {
    let text = text
        .replace(
            "typedef unsigned _BitInt(128) u128;",
            "typedef invar_u128 u128;",
        )
        .replace(
            "typedef unsigned __int128 u128;",
            "typedef invar_u128 u128;",
        );
    let tokens = tokens(&literals(&text));
    let tokens = pointers(tokens);
    hoist(tokens).concat()
}

/// Rewrite every unsuffixed floating literal to its exact binary64 encoding.
fn literals(text: &str) -> String {
    let bytes = text.as_bytes();
    let (mut out, mut i) = (String::new(), 0);
    while i < bytes.len() {
        let starts = bytes[i].is_ascii_digit()
            || (bytes[i] == b'.' && bytes.get(i + 1).is_some_and(u8::is_ascii_digit));
        let glued = i > 0 && (bytes[i - 1].is_ascii_alphanumeric() || bytes[i - 1] == b'_');
        if !starts || glued {
            let width = text[i..].chars().next().map_or(1, char::len_utf8);
            out.push_str(&text[i..i + width]);
            i += width;
            continue;
        }
        let mut end = i + 1;
        while end < bytes.len() {
            let c = bytes[end];
            let signed =
                matches!(c, b'+' | b'-') && matches!(bytes[end - 1], b'e' | b'E' | b'p' | b'P');
            if !(c.is_ascii_alphanumeric() || c == b'.' || c == b'_' || signed) {
                break;
            }
            end += 1;
        }
        out.push_str(&number(&text[i..end]));
        i = end;
    }
    out
}

/// A pp-number, rewritten when it is an unsuffixed floating literal.
fn number(token: &str) -> String {
    let lower = token.to_ascii_lowercase();
    let hex = lower.starts_with("0x");
    let float = if hex {
        lower.contains('p')
    } else {
        lower.contains('.') || lower.contains('e')
    };
    if !float || lower.ends_with('f') {
        return token.to_owned();
    }
    let body = lower.trim_end_matches('l');
    let bits = if hex {
        hex_float(&body[2..])
    } else {
        body.parse::<f64>()
            .unwrap_or_else(|_| panic!("bad literal {token}"))
            .to_bits()
    };
    format!("D::raw({bits:#018x}ul)")
}

/// The binary64 encoding of a hexadecimal floating literal (after `0x`),
/// which must be exactly representable.
fn hex_float(text: &str) -> u64 {
    let (digits, exponent) = text.split_once('p').expect("hex float exponent");
    let mut exponent: i32 = exponent.parse().expect("hex float exponent");
    let mut m: u128 = 0;
    let (whole, fraction) = digits.split_once('.').unwrap_or((digits, ""));
    for (i, c) in whole.chars().chain(fraction.chars()).enumerate() {
        m = (m << 4) | u128::from(c.to_digit(16).expect("hex digit"));
        if i >= whole.len() {
            exponent -= 4;
        }
        assert!(m >> 100 == 0, "hex literal too long: {text}");
    }
    if m == 0 {
        return 0;
    }
    while m & 1 == 0 {
        m >>= 1;
        exponent += 1;
    }
    let length = 128 - m.leading_zeros() as i32;
    assert!(length <= 53, "inexact hex literal {text}");
    let top = exponent + length - 1;
    assert!(top <= 1023, "hex literal overflows: {text}");
    if top >= -1022 {
        let fraction = (m << (52 - (length - 1))) as u64 & ((1 << 52) - 1);
        ((top + 1023) as u64) << 52 | fraction
    } else {
        let shift = exponent + 1074;
        assert!(shift >= 0, "inexact subnormal hex literal {text}");
        (m << shift) as u64
    }
}

/// Split C text into identifier, number, punctuation, whitespace and whole
/// preprocessor-line tokens, which concatenate back to the input.
fn tokens(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut rest = text;
    let mut line_start = true;
    while let Some(c) = rest.chars().next() {
        let width = if line_start && rest.trim_start_matches([' ', '\t']).starts_with('#') {
            let mut end = 0;
            for line in rest.split_inclusive('\n') {
                end += line.len();
                if !line.trim_end().ends_with('\\') {
                    break;
                }
            }
            end
        } else if c.is_ascii_alphanumeric() || c == '_' {
            rest.find(|c: char| !(c.is_ascii_alphanumeric() || c == '_'))
                .unwrap_or(rest.len())
        } else if c.is_whitespace() {
            rest.find(|c: char| !c.is_whitespace())
                .unwrap_or(rest.len())
        } else if rest.starts_with("->") {
            2
        } else {
            c.len_utf8()
        };
        let (token, after) = rest.split_at(width);
        line_start = token.ends_with('\n') || (line_start && token.trim().is_empty());
        out.push(token.to_owned());
        rest = after;
    }
    out
}

fn significant(token: &str) -> bool {
    !token.trim().is_empty() && !token.trim_start().starts_with('#')
}

fn ident(token: &str) -> bool {
    token.starts_with(|c: char| c.is_ascii_alphabetic() || c == '_')
}

/// Address spaces for pointers: views of tables are `constant`, pointer
/// parameters (results returned through them) are `thread`.
fn pointers(mut tokens: Vec<String>) -> Vec<String> {
    let at: Vec<usize> = (0..tokens.len())
        .filter(|&i| significant(&tokens[i]))
        .collect();
    let word =
        |tokens: &[String], k: usize| at.get(k).map_or("", |&i| tokens[i].as_str()).to_owned();
    for k in 0..at.len() {
        let (prev, this) = (
            if k > 0 {
                word(&tokens, k - 1)
            } else {
                String::new()
            },
            word(&tokens, k),
        );
        let next = |n: usize| word(&tokens, k + n);
        let parameter = prev == "(" || prev == ",";
        if this == "const"
            && ident(&next(1))
            && (next(2) == "*" || (parameter && next(3) == "[" && next(4) == "]"))
        {
            tokens[at[k]] = "INVAR_CONSTANT".into();
        } else if parameter
            && POINTEES.contains(&this.as_str())
            && next(1) == "*"
            && ident(&next(2))
            && (next(3) == "," || next(3) == ")")
        {
            tokens[at[k]] = format!("INVAR_THREAD {this}");
        } else if this == "(" && ident(&next(1)) && next(2) == "*" && next(3) == ")" {
            tokens[at[k]] = "(INVAR_CONSTANT ".into();
        }
    }
    tokens
}

/// Move `static const` tables to program scope in the constant address space.
fn hoist(tokens: Vec<String>) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    let (mut depth, mut header, mut body) = (0usize, 0usize, None::<usize>);
    let mut renames: Vec<(String, String)> = Vec::new();
    // Macros a function body defines, which its hoisted tables may use.
    let mut defines: Vec<String> = Vec::new();
    let mut i = 0;
    while i < tokens.len() {
        let token = tokens[i].as_str();
        if token == "static"
            && next_significant(&tokens, i + 1).map(|j| tokens[j].as_str()) == Some("const")
        {
            let after_const = next_significant(&tokens, i + 1).unwrap() + 1;
            let end = statement_end(&tokens, i);
            let declaration = &tokens[after_const..=end];
            if depth == 0 {
                out.push("INVAR_CONSTANT".into());
                out.extend(declaration.iter().cloned());
                header = out.len();
            } else if declaration
                .iter()
                .take_while(|t| *t != "=")
                .any(|t| t == "[")
                && constant_initializer(declaration)
            {
                let name = declared_name(declaration);
                let renamed = format!("invar_t{}_{name}", renames.len());
                let mut hoisted = vec!["\n".to_owned()];
                hoisted.extend(defines.iter().cloned());
                hoisted.push("INVAR_CONSTANT".to_owned());
                hoisted.extend(declaration.iter().map(|t| {
                    if *t == name {
                        renamed.clone()
                    } else {
                        t.clone()
                    }
                }));
                out.splice(header..header, hoisted.iter().cloned());
                header += hoisted.len();
                // A later block may declare the same name; it shadows the first.
                renames.retain(|(earlier, _)| *earlier != name);
                renames.push((name, renamed));
            } else {
                out.push("const".into());
                out.extend(declaration.iter().cloned());
            }
            i = end + 1;
            continue;
        }
        match token {
            "{" => {
                if depth == 0 && previous_significant(&out) == Some(")") {
                    body = Some(out.len());
                }
                depth += 1;
            }
            "}" => depth = depth.checked_sub(1).expect("unbalanced braces"),
            _ => {}
        }
        let mut token = token.to_owned();
        if body.is_some()
            && !matches!(previous_significant(&out), Some("." | "->"))
            && let Some((_, renamed)) = renames.iter().find(|(name, _)| *name == token)
        {
            token = renamed.clone();
        }
        if depth > 0 && token.trim_start().starts_with("#define") {
            defines.push(token.clone());
        }
        let boundary =
            depth == 0 && (token == ";" || token == "}" || token.trim_start().starts_with('#'));
        out.push(token);
        if boundary {
            header = out.len();
            body = None;
            renames.clear();
            defines.clear();
        }
        i += 1;
    }
    assert_eq!(depth, 0, "unbalanced braces");
    out
}

fn next_significant(tokens: &[String], from: usize) -> Option<usize> {
    (from..tokens.len()).find(|&j| significant(&tokens[j]))
}

fn previous_significant(tokens: &[String]) -> Option<&str> {
    tokens
        .iter()
        .rev()
        .find(|t| significant(t))
        .map(String::as_str)
}

/// The index of the `;` ending the declaration that starts at `start`.
fn statement_end(tokens: &[String], start: usize) -> usize {
    let mut depth = 0i32;
    for (j, token) in tokens.iter().enumerate().skip(start) {
        match token.as_str() {
            "{" | "(" | "[" => depth += 1,
            "}" | ")" | "]" => depth -= 1,
            ";" if depth == 0 => return j,
            _ => {}
        }
    }
    panic!("unterminated declaration");
}

/// Whether a table's initializer is a constant expression for MSL: literals,
/// names and unary signs only. Arithmetic on `D` is not `constexpr`, so such
/// tables stay function-local.
fn constant_initializer(declaration: &[String]) -> bool {
    let start = declaration
        .iter()
        .position(|t| t == "=")
        .map_or(declaration.len(), |i| i + 1);
    let mut previous = "=";
    for token in declaration[start..].iter().filter(|t| significant(t)) {
        let unary = matches!(previous, "=" | "{" | "," | "(");
        if matches!(
            token.as_str(),
            "*" | "/" | "+" | "%" | "<" | ">" | "&" | "|" | "^"
        ) || (token == "-" && !unary)
        {
            return false;
        }
        previous = token;
    }
    true
}

/// The declarator name: the last identifier before `=` outside brackets and
/// braces (`double c[]`, `struct {...} st[]`).
fn declared_name(declaration: &[String]) -> String {
    let mut depth = 0i32;
    let mut name = None;
    for token in declaration {
        match token.as_str() {
            "{" | "(" | "[" => depth += 1,
            "}" | ")" | "]" => depth -= 1,
            "=" if depth == 0 => break,
            t if depth == 0 && ident(t) => name = Some(t.to_owned()),
            _ => {}
        }
    }
    name.expect("declarator name")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn literals_become_exact_binary64_encodings() {
        assert_eq!(number("0x1.8p52"), "D::raw(0x4338000000000000ul)");
        assert_eq!(number("1.0"), "D::raw(0x3ff0000000000000ul)");
        assert_eq!(
            number("1.4182e-9"),
            format!("D::raw({:#018x}ul)", 1.4182e-9f64.to_bits())
        );
        assert_eq!(number("0x1p-1074"), "D::raw(0x0000000000000001ul)");
        assert_eq!(
            number("0x1.fffffffffffffp+1023"),
            "D::raw(0x7feffffffffffffful)"
        );
        for kept in ["0x1p-126f", "1.0f", "0xffu", "123", "0x7f800000"] {
            assert_eq!(number(kept), kept);
        }
        assert_eq!(
            literals("x1e5 + a.5 + 2.5"),
            "x1e5 + a.5 + D::raw(0x4004000000000000ul)"
        );
    }

    #[test]
    #[should_panic(expected = "inexact hex literal")]
    fn inexact_hex_literals_are_refused() {
        hex_float("1.00000000000001p0");
    }

    #[test]
    fn local_tables_move_to_program_scope_and_scalars_stay_local() {
        let text = "typedef int T;\nstatic const double g[] = {1};\nfloat f(int *q, const double *c) {\n  static const double t[2] = {1, 2};\n  static const double s = 3;\n  return t[*q] + c[0] + s + x.t;\n}\n";
        let out = hoist(pointers(tokens(text))).concat();
        assert!(out.contains("INVAR_CONSTANT double g[]"), "{out}");
        assert!(out.contains("INVAR_CONSTANT double g[] = {1};\nINVAR_CONSTANT double invar_t0_t[2] = {1, 2};\nfloat f("), "{out}");
        assert!(
            out.contains("INVAR_THREAD int *q, INVAR_CONSTANT double *c"),
            "{out}"
        );
        assert!(out.contains("const double s = 3;"), "{out}");
        assert!(
            out.contains("return invar_t0_t[*q]") && out.contains("x.t"),
            "{out}"
        );
        let text = "float f(int i) {\n#define P 0x1p0\n  static const double a[] = {-P, 1};\n  static const double b[] = {P, 2*P};\n  return a[i] + b[i];\n}\n";
        let out = hoist(tokens(text)).concat();
        assert!(
            out.starts_with("\n#define P 0x1p0\nINVAR_CONSTANT double invar_t0_a[] = {-P, 1};"),
            "{out}"
        );
        assert!(out.contains("  const double b[] = {P, 2*P};"), "{out}");
        let text = "float f(int i) {\n  if (i) {\n    static const double c[] = {1};\n    return c[0];\n  }\n  static const double c[] = {2};\n  return c[0];\n}\n";
        let out = hoist(tokens(text)).concat();
        assert!(
            out.contains("return invar_t0_c[0];\n  }")
                && out.ends_with("return invar_t1_c[0];\n}\n"),
            "{out}"
        );
    }

    #[test]
    fn every_binary32_unit_transforms() {
        let text = msl_source(&Function::ALL);
        let left: Vec<&str> = text
            .lines()
            .filter(|l| l.contains("static const "))
            .collect();
        assert!(left.is_empty(), "{left:?}");
        assert!(!text.contains("__int128 u128"));
        for f in Function::ALL {
            let symbol = msl_symbol(f);
            let (space, _) = symbol.split_once("::").unwrap();
            assert_eq!(text.matches(&format!("namespace {space} {{")).count(), 1);
        }
    }
}
