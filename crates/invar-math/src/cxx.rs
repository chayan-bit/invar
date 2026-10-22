//! The C-to-C++ rewrites the device emitters need for the vendored CORE-MATH.
//!
//! CORE-MATH initializes unions of anonymous structs with C99 designators,
//! for example `dint64_t x = {.hi = a, .lo = b, .ex = c, .sgn = d}` where
//! `hi` and `lo` live in an anonymous struct that shares storage with a
//! `u128`. C++ (NVRTC, MSL) only designates direct members, so this pass
//! moves that struct to the front of its union and makes such initializers
//! positional in the struct's declaration order, which initializes exactly
//! the same bytes. It also resolves the one byte-order conditional upstream
//! uses to its little-endian branch, the order of every supported device.
//! C lets `goto` jump past initialized declarations and C++ does not, so the
//! declarations between each `goto` statement and its label are wrapped in
//! a block the jump skips whole; a variable still used after the label then
//! fails to compile rather than changing meaning.
//! The pass refuses (panics) on any shape it does not recognize, so a new
//! upstream form fails the source-shape test instead of compiling wrongly.

const LITTLE_ENDIAN_IF: &str = "#if __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__";

/// Keep the little-endian branch of every byte-order conditional.
fn little_endian(text: &str) -> String {
    let mut out = String::new();
    // (nesting depth of the byte-order #if, whether its branch is kept)
    let mut state: Option<(usize, bool)> = None;
    let mut depth = 0usize;
    for line in text.lines() {
        let word = line
            .trim_start()
            .strip_prefix('#')
            .map(str::trim_start)
            .unwrap_or("");
        let ours = state.is_some_and(|(at, _)| at == depth);
        let mut keep = state.is_none_or(|(_, kept)| kept);
        if word.starts_with("if") {
            depth += 1;
            if line.trim() == LITTLE_ENDIAN_IF {
                assert!(state.is_none(), "nested byte-order conditionals");
                state = Some((depth, true));
                keep = false;
            }
        } else if ours && word.starts_with("else") {
            state = state.map(|(at, _)| (at, false));
            keep = false;
        } else if ours && word.starts_with("elif") {
            panic!("#elif in a byte-order conditional");
        } else if word.starts_with("endif") {
            if ours {
                state = None;
                keep = false;
            }
            depth = depth.checked_sub(1).expect("unbalanced #endif");
        }
        if keep {
            out.push_str(line);
        }
        out.push('\n');
    }
    assert!(state.is_none(), "unterminated byte-order conditional");
    out
}

/// `text` without C comments (a block comment becomes one space).
pub(crate) fn uncommented(text: &str) -> String {
    let (mut out, mut rest) = (String::new(), text);
    loop {
        let block = rest.find("/*");
        let line = rest.find("//");
        match (block, line) {
            (Some(b), l) if l.is_none_or(|l| b < l) => {
                out.push_str(&rest[..b]);
                out.push(' ');
                rest = rest[b..].split_once("*/").map_or("", |(_, after)| after);
            }
            (_, Some(l)) => {
                out.push_str(&rest[..l]);
                rest = rest[l..].split_once('\n').map_or("", |(_, after)| after);
                out.push('\n');
            }
            _ => {
                out.push_str(rest);
                return out;
            }
        }
    }
}

/// Field names declared by the lines of one struct body.
fn fields(body: &[&str]) -> Vec<String> {
    let mut names = Vec::new();
    for declaration in uncommented(&body.join("\n"))
        .split(';')
        .filter(|d| !d.trim().is_empty())
    {
        for (i, part) in declaration.split(',').enumerate() {
            let words: Vec<&str> = part.split_whitespace().collect();
            let name = if i == 0 { words.last() } else { words.first() };
            let name = name.unwrap_or_else(|| panic!("unparsed field `{declaration}`"));
            assert!(
                name.chars().all(|c| c.is_alphanumeric() || c == '_'),
                "unparsed field `{name}`"
            );
            names.push((*name).to_owned());
        }
    }
    names
}

/// One member of a union body: its source lines and, for an anonymous
/// struct, its field names.
struct Member {
    lines: Vec<String>,
    fields: Option<Vec<String>>,
}

/// The anonymous-struct unions of `text`: (start line, end line, members).
fn unions(lines: &[&str]) -> Vec<(usize, usize, Vec<Member>)> {
    let mut found = Vec::new();
    let mut i = 0;
    while i < lines.len() {
        if lines[i].trim() != "typedef union {" {
            i += 1;
            continue;
        }
        let start = i;
        let mut members = Vec::new();
        i += 1;
        while !lines[i].trim_start().starts_with('}') {
            if lines[i].trim() == "struct {" {
                let open = i;
                while lines[i].trim() != "};" {
                    i += 1;
                }
                let body = &lines[open + 1..i];
                members.push(Member {
                    lines: lines[open..=i].iter().map(|l| (*l).to_owned()).collect(),
                    fields: Some(fields(body)),
                });
            } else {
                members.push(Member {
                    lines: vec![lines[i].to_owned()],
                    fields: None,
                });
            }
            i += 1;
        }
        found.push((start, i, members));
        i += 1;
    }
    found
}

/// Whether `text` declares a union with an anonymous struct member on its
/// little-endian path. NVRTC accepts that only from 12.9.
pub(crate) fn has_anonymous_struct_union(text: &str) -> bool {
    let text = little_endian(text);
    let lines: Vec<&str> = text.lines().collect();
    unions(&lines)
        .iter()
        .any(|(_, _, members)| members.iter().any(|m| m.fields.is_some()))
}

/// Split `{ ... }` content at top-level commas.
fn items(content: &str) -> Vec<&str> {
    let (mut depth, mut from, mut out) = (0i32, 0, Vec::new());
    for (i, c) in content.char_indices() {
        match c {
            '{' | '(' | '[' => depth += 1,
            '}' | ')' | ']' => depth -= 1,
            ',' if depth == 0 => {
                out.push(&content[from..i]);
                from = i + 1;
            }
            _ => {}
        }
    }
    out.push(&content[from..]);
    out.into_iter()
        .filter(|item| !item.trim().is_empty())
        .collect()
}

/// `.name = value` pairs of a designated initializer body, or `None`.
fn designated(content: &str) -> Option<Vec<(&str, &str)>> {
    items(content)
        .into_iter()
        .map(|item| {
            let (name, value) = item.trim().strip_prefix('.')?.split_once('=')?;
            let name = name.trim();
            name.chars()
                .all(|c| c.is_alphanumeric() || c == '_')
                .then_some((name, value.trim()))
        })
        .collect()
}

/// Rewrite designated initializers that target anonymous-struct fields.
/// Returns the text and the field lists that must lead their union.
/// `direct` lists the plain members of every union: a designator naming one
/// (`{.r = x}`) is already valid C++ and stays as it is.
fn initializers(
    text: &str,
    structs: &[Vec<String>],
    direct: &[String],
) -> (String, Vec<Vec<String>>) {
    let (mut out, mut used, mut rest) = (String::new(), Vec::new(), text);
    while let Some(open) = rest.find('{') {
        out.push_str(&rest[..=open]);
        rest = &rest[open + 1..];
        if !rest.trim_start().starts_with('.') {
            continue;
        }
        let mut depth = 1;
        let close = rest
            .char_indices()
            .find(|&(_, c)| {
                depth += i32::from(c == '{') - i32::from(c == '}');
                depth == 0
            })
            .map(|(i, _)| i)
            .expect("unbalanced initializer");
        let Some(pairs) = designated(&rest[..close]) else {
            continue;
        };
        let names: Vec<&str> = pairs.iter().map(|(n, _)| *n).collect();
        if names.iter().all(|n| direct.iter().any(|d| d == n)) {
            continue;
        }
        let owners: Vec<&Vec<String>> = structs
            .iter()
            .filter(|s| names.iter().all(|n| s.iter().any(|f| f == n)))
            .collect();
        let [owner] = owners[..] else {
            assert!(owners.is_empty(), "ambiguous designators {names:?}");
            continue;
        };
        let values: Vec<&str> = owner
            .iter()
            .map(|field| {
                pairs
                    .iter()
                    .find(|(n, _)| n == field)
                    .map_or("0", |(_, v)| v)
            })
            .collect();
        out.push_str(&format!("{{{}}}", values.join(", ")));
        rest = &rest[close..];
        if !used.contains(owner) {
            used.push(owner.clone());
        }
    }
    out.push_str(rest);
    (out, used)
}

/// Net brace depth change of one line, ignoring `//` comments.
fn braces(line: &str) -> i32 {
    let code = line.split("//").next().unwrap();
    code.matches('{').count() as i32 - code.matches('}').count() as i32
}

/// Wrap the statements between each `goto label;` and `label:` in a block.
fn goto_blocks(text: &str) -> String {
    let mut lines: Vec<String> = text.lines().map(str::to_owned).collect();
    let mut i = 0;
    while i < lines.len() {
        let Some(label) = lines[i]
            .trim()
            .strip_prefix("goto ")
            .and_then(|r| r.strip_suffix(';'))
        else {
            i += 1;
            continue;
        };
        let target = format!("{label}:");
        let at = (i + 1..lines.len())
            .find(|&j| lines[j].trim() == target)
            .unwrap_or_else(|| panic!("goto {label} without a later label"));
        let net: i32 = lines[i + 1..at].iter().map(|l| braces(l)).sum();
        assert!(net <= 0, "goto {label} jumps into a block");
        let mut depth = 0;
        let mut open = i + 1;
        if net < 0 {
            open = (i + 1..at)
                .find(|&j| {
                    depth += braces(&lines[j]);
                    depth == net
                })
                .unwrap()
                + 1;
        }
        lines.insert(at, "}".to_owned());
        lines.insert(open, "{".to_owned());
        i = at + 2;
    }
    lines.join("\n")
}

/// Rewrite one CORE-MATH translation unit (headers already spliced) into the
/// C++ subset NVRTC and MSL accept.
pub(crate) fn rewrite(text: &str) -> String {
    let text = goto_blocks(&little_endian(text));
    let lines: Vec<&str> = text.lines().collect();
    let found = unions(&lines);
    let structs: Vec<Vec<String>> = found
        .iter()
        .flat_map(|(_, _, m)| m.iter().filter_map(|m| m.fields.clone()))
        .collect();
    let direct: Vec<String> = found
        .iter()
        .flat_map(|(_, _, m)| {
            let plain: Vec<&str> = m
                .iter()
                .filter(|m| m.fields.is_none())
                .map(|m| m.lines[0].as_str())
                .collect();
            fields(&plain)
        })
        .collect();
    let mut out: Vec<String> = lines.iter().map(|l| (*l).to_owned()).collect();
    let (_, used) = initializers(&text, &structs, &direct);
    for (start, end, members) in found.iter().rev() {
        let lead: Vec<&Member> = members
            .iter()
            .filter(|m| m.fields.as_ref().is_some_and(|f| used.contains(f)))
            .collect();
        assert!(lead.len() <= 1, "two designated structs in one union");
        let Some(lead) = lead.first() else { continue };
        let mut body: Vec<String> = lead.lines.clone();
        for member in members.iter().filter(|m| !std::ptr::eq(*m, *lead)) {
            body.extend(member.lines.iter().cloned());
        }
        out.splice(start + 1..*end, body);
    }
    initializers(&out.join("\n"), &structs, &direct).0
}

#[cfg(test)]
mod tests {
    use super::*;

    const UNIT: &str = "#if __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
typedef union {
  struct {
    u128 r;
    int64_t _ex;
  };
  struct {
    uint64_t lo; // low
    uint64_t hi;
    int64_t ex;
  };
} dint64_t;
#else
typedef union { BIG } dint64_t;
#endif
static const dint64_t A = {.hi = 0x1, .lo = 0x2, .ex = -3};
static const dint64_t B[] = {{.hi = f(1, 2), .ex = 4}};
static const f64_u C = {.f = 1.5};
typedef union {
  u128 r;
  struct {
    uint64_t l;
    uint64_t h;
  };
} uint128_t;
uint128_t D = {.r = x}, E = {.h = a, .l = b};
";

    #[test]
    fn designated_anonymous_struct_initializers_become_positional() {
        let out = rewrite(UNIT);
        assert!(!out.contains("union { BIG }"));
        assert!(out.contains("A = {{0x2, 0x1, -3}}"), "{out}");
        assert!(out.contains("B[] = {{{0, f(1, 2), 4}}}"), "{out}");
        assert!(out.contains("C = {.f = 1.5}"));
        assert!(out.contains("D = {.r = x}, E = {{b, a}}"), "{out}");
        let lo = out.find("uint64_t lo").unwrap();
        assert!(
            lo < out.find("u128 r").unwrap(),
            "designated struct must lead its union"
        );
    }

    #[test]
    fn declarations_a_goto_skips_move_into_a_block() {
        let text = "int f(int a) {\n  int r = 0;\n  if (a) {\n    r = 1;\n    goto end;\n  }\n  int b = a + 1;\n  r = b;\n end:\n  return r;\n}";
        let out = goto_blocks(text);
        assert_eq!(
            out,
            "int f(int a) {\n  int r = 0;\n  if (a) {\n    r = 1;\n    goto end;\n  }\n{\n  int b = a + 1;\n  r = b;\n}\n end:\n  return r;\n}"
        );
    }

    #[test]
    fn multiple_declarators_are_fields() {
        assert_eq!(
            fields(&[
                "uint64_t m, h, l; // order",
                "int64_t ex; /* a",
                "b */ uint64_t sgn;"
            ]),
            ["m", "h", "l", "ex", "sgn"]
        );
    }
}
