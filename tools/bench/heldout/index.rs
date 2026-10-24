//! Held-out index: canonical programs and the anti-overfitting admission check.

use super::*;

/// Anti-overfitting boundary. Tuning, calibration and any later M4-06 search
/// see programs only through `tuning_programs` or `HeldOutIndex::admit`; both
/// refuse held-out programs, renamed copies of them and their data inputs.
pub struct HeldOutIndex {
    sources: std::collections::BTreeSet<String>,
    programs: Vec<std::collections::BTreeSet<String>>,
    inputs: std::collections::BTreeSet<String>,
}

/// Source tokens: identifiers, numeric literals with their suffix, and single
/// punctuation characters. Whitespace and layout never reach a token.
fn tokens(source: &str) -> Vec<&str> {
    let bytes = source.as_bytes();
    let mut out = Vec::new();
    let mut i = 0;
    while i < bytes.len() {
        let c = bytes[i];
        let start = i;
        if c.is_ascii_whitespace() {
            i += 1;
            continue;
        }
        if c.is_ascii_alphanumeric() || c == b'_' {
            while i < bytes.len()
                && (bytes[i].is_ascii_alphanumeric()
                    || bytes[i] == b'_'
                    || (bytes[i] == b'.'
                        && i + 1 < bytes.len()
                        && bytes[i + 1].is_ascii_digit()
                        && bytes[start].is_ascii_digit()))
            {
                i += 1;
            }
        } else {
            i += source[i..].chars().next().map_or(1, char::len_utf8);
        }
        out.push(&source[start..i]);
    }
    out
}

/// Canonical forms of every top-level function: each name the program
/// declares (functions, parameters, `let`, `var` and `for` bindings) is
/// replaced by its order of first use inside that function, so renaming or
/// reformatting a program, or wrapping it in other functions, keeps its
/// functions' identities. Keywords, types and intrinsics are kept.
pub fn canonical_functions(source: &str) -> std::collections::BTreeSet<String> {
    let toks = tokens(source);
    let mut declared = std::collections::BTreeSet::new();
    for (k, t) in toks.iter().enumerate() {
        if matches!(*t, "fn" | "let" | "var" | "for")
            && let Some(name) = toks.get(k + 1)
        {
            declared.insert(*name);
        }
        if matches!(*t, "(" | ",")
            && toks.get(k + 2) == Some(&":")
            && let Some(name) = toks.get(k + 1)
        {
            declared.insert(*name);
        }
    }
    let mut functions = std::collections::BTreeSet::new();
    let mut k = 0;
    while k < toks.len() {
        if toks[k] != "fn" {
            k += 1;
            continue;
        }
        let (mut depth, mut end) = (0usize, k);
        for (j, t) in toks.iter().enumerate().skip(k) {
            match *t {
                "{" => depth += 1,
                "}" => {
                    depth = depth.saturating_sub(1);
                    if depth == 0 {
                        end = j;
                        break;
                    }
                }
                _ => {}
            }
            end = j;
        }
        let mut names: Vec<&str> = Vec::new();
        let canonical: Vec<String> = toks[k..=end]
            .iter()
            .map(|t| {
                if declared.contains(t) {
                    let at = names.iter().position(|n| n == t).unwrap_or_else(|| {
                        names.push(t);
                        names.len() - 1
                    });
                    format!("v{at}")
                } else {
                    (*t).to_owned()
                }
            })
            .collect();
        functions.insert(sha256(canonical.join(" ").as_bytes()));
        k = end + 1;
    }
    functions
}

/// One digest for a whole program from its canonical functions: programs
/// that differ only in names or layout are the same program.
pub fn canonical_program(source: &str) -> String {
    let functions: Vec<String> = canonical_functions(source).into_iter().collect();
    sha256(functions.join(",").as_bytes())
}

/// Distinctive literal data arrays: at least eight elements and three
/// distinct values. Shorter or low-cardinality literals (shape parameters,
/// segment offsets, 0/1 masks, one-cell matrices) occur in both splits by
/// construction and carry no held-out identity; the program itself is
/// protected by `canonical_functions`. This is a second guard only.
fn data_inputs(source: &str) -> Vec<String> {
    let compact: String = source.chars().filter(|c| !c.is_whitespace()).collect();
    compact
        .split('[')
        .skip(1)
        .filter_map(|tail| tail.split(']').next())
        .filter(|body| {
            let items: Vec<&str> = body.split(',').collect();
            let distinct: std::collections::BTreeSet<&str> = items.iter().copied().collect();
            items.len() >= 8
                && distinct.len() >= 3
                && items.iter().all(|x| {
                    !x.is_empty()
                        && x.trim_start_matches('-')
                            .chars()
                            .all(|c| c.is_ascii_digit() || matches!(c, '.' | 'u' | 'f' | '3' | '2'))
                })
        })
        .map(str::to_owned)
        .collect()
}

impl HeldOutIndex {
    pub fn from_programs(programs: &Value) -> Result<Self, String> {
        let mut index = Self {
            sources: Default::default(),
            programs: Vec::new(),
            inputs: Default::default(),
        };
        for p in programs.as_array().ok_or("held-out programs missing")? {
            if p["split"] != "held_out" {
                continue;
            }
            let source = p["source"].as_str().ok_or("held-out source missing")?;
            index.sources.insert(sha256(source.as_bytes()));
            index.programs.push(canonical_functions(source));
            index.inputs.extend(data_inputs(source));
        }
        if index.sources.is_empty() {
            return Err("empty held-out index".into());
        }
        Ok(index)
    }

    pub fn len(&self) -> usize {
        self.sources.len()
    }

    /// Refuse a candidate tuning or calibration program that is, or embeds the
    /// data of, a held-out program.
    pub fn admit(&self, source: &str) -> Result<(), String> {
        if self.sources.contains(&sha256(source.as_bytes())) {
            return Err("held-out program refused on a tuning path".into());
        }
        // Every function of a held-out program, canonically renamed, inside
        // the candidate: the program itself, renamed, reformatted or wrapped.
        let candidate = canonical_functions(source);
        if self.programs.iter().any(|held| held.is_subset(&candidate)) {
            return Err("renamed or embedded held-out program refused on a tuning path".into());
        }
        if data_inputs(source).iter().any(|x| self.inputs.contains(x)) {
            return Err("held-out input data refused on a tuning path".into());
        }
        Ok(())
    }
}

/// The only corpus view a tuning path may use: the tuning split, regenerated
/// from the seed, each program admitted against the held-out index.
pub fn tuning_programs(seed: u64) -> Result<(HeldOutIndex, Vec<Value>), String> {
    let all = generate(seed)?;
    let index = HeldOutIndex::from_programs(&all["programs"])?;
    let mut tuning = Vec::new();
    for p in all["programs"].as_array().ok_or("programs missing")? {
        if p["split"] != "tuning" {
            continue;
        }
        let source = p["source"].as_str().ok_or("tuning source missing")?;
        index
            .admit(source)
            .map_err(|e| format!("{}: {e}", p["id"].as_str().unwrap_or("?")))?;
        tuning.push(p.clone());
    }
    Ok((index, tuning))
}
