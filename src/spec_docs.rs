//! DX-25 (#86): the specification as chapters with token budgets, and topic assembly.
//!
//! `docs/spec/chapters.json` lists every chapter in reading order. The chapters with
//! `part: true` concatenate, byte for byte, into the generated single-file spec
//! (`docs/spec/invar-v0.1.md`), so its links and clause numbers never move.
//! `docs/spec/budget.json` holds every budget. The texts are embedded, so `invar spec`
//! answers the same from any build; `tests/spec_budget.rs` keeps the recorded token
//! counts, the budgets and the generated file honest.

use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet};

const MANIFEST: &str = include_str!("../docs/spec/chapters.json");
const BUDGETS: &str = include_str!("../docs/spec/budget.json");
/// Every chapter text by manifest path; a test checks this covers the manifest exactly.
const TEXTS: [(&str, &str); 19] = [
    (
        "docs/spec/chapters/core.md",
        include_str!("../docs/spec/chapters/core.md"),
    ),
    (
        "docs/spec/chapters/ownership.md",
        include_str!("../docs/spec/chapters/ownership.md"),
    ),
    (
        "docs/spec/chapters/modules-records.md",
        include_str!("../docs/spec/chapters/modules-records.md"),
    ),
    (
        "docs/spec/chapters/views.md",
        include_str!("../docs/spec/chapters/views.md"),
    ),
    (
        "docs/spec/chapters/generics-closures.md",
        include_str!("../docs/spec/chapters/generics-closures.md"),
    ),
    (
        "docs/spec/chapters/units-frames.md",
        include_str!("../docs/spec/chapters/units-frames.md"),
    ),
    (
        "docs/spec/chapters/bytes-cabi.md",
        include_str!("../docs/spec/chapters/bytes-cabi.md"),
    ),
    (
        "docs/spec/chapters/system.md",
        include_str!("../docs/spec/chapters/system.md"),
    ),
    (
        "docs/spec/chapters/graph-checks.md",
        include_str!("../docs/spec/chapters/graph-checks.md"),
    ),
    (
        "docs/spec/chapters/manifests.md",
        include_str!("../docs/spec/chapters/manifests.md"),
    ),
    (
        "docs/spec/chapters/engines.md",
        include_str!("../docs/spec/chapters/engines.md"),
    ),
    (
        "docs/spec/chapters/limits-diagnostics.md",
        include_str!("../docs/spec/chapters/limits-diagnostics.md"),
    ),
    (
        "docs/spec/chapters/v02-numerics.md",
        include_str!("../docs/spec/chapters/v02-numerics.md"),
    ),
    (
        "docs/spec/chapters/placement-attributes.md",
        include_str!("../docs/spec/chapters/placement-attributes.md"),
    ),
    (
        "docs/spec/chapters/language-surface.md",
        include_str!("../docs/spec/chapters/language-surface.md"),
    ),
    (
        "docs/spec/invar-v0.2-axes.md",
        include_str!("../docs/spec/invar-v0.2-axes.md"),
    ),
    (
        "docs/spec/invar-v0.2-axis-n.md",
        include_str!("../docs/spec/invar-v0.2-axis-n.md"),
    ),
    (
        "docs/spec/invar-v0.2-regions.md",
        include_str!("../docs/spec/invar-v0.2-regions.md"),
    ),
    (
        "docs/spec/invar-v0.2-extrema.md",
        include_str!("../docs/spec/invar-v0.2-extrema.md"),
    ),
];
/// A spec line that records the file's own size; no estimator counts it.
pub const SIZE_RECORD_PREFIX: &str = "Size record:";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Tokens {
    pub bytes3: usize,
    pub bpe: usize,
}

impl Tokens {
    /// Every gate uses the larger estimate.
    pub fn gate(self) -> usize {
        self.bytes3.max(self.bpe)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Chapter {
    pub id: String,
    pub path: String,
    pub core: bool,
    pub part: bool,
    pub requires: Vec<String>,
    pub topics: Vec<String>,
    pub tokens: Tokens,
    pub text: String,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Spec {
    pub generated: String,
    pub chapters: Vec<Chapter>,
    pub budgets: Value,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SpecError {
    UnknownTopic(String),
    UnknownChapter(String),
    OverBudget {
        budget: usize,
        tokens: usize,
        largest: Vec<(String, usize)>,
    },
}

impl SpecError {
    pub fn code(&self) -> &'static str {
        match self {
            Self::UnknownTopic(_) => "SPEC_UNKNOWN_TOPIC",
            Self::UnknownChapter(_) => "SPEC_UNKNOWN_CHAPTER",
            Self::OverBudget { .. } => "SPEC_BUDGET_EXCEEDED",
        }
    }
}

impl std::fmt::Display for SpecError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnknownTopic(topic) => write!(f, "topic {topic} has no chapter"),
            Self::UnknownChapter(id) => write!(f, "no chapter {id}"),
            Self::OverBudget {
                budget,
                tokens,
                largest,
            } => write!(
                f,
                "the assembly is {tokens} tokens, {} over the budget {budget}; largest chapters: {}",
                tokens - budget,
                listing(largest)
            ),
        }
    }
}

/// The chapters chosen by `assemble`, in manifest order.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Assembly {
    pub chapters: Vec<(String, String, usize)>,
    pub tokens: usize,
    pub text: String,
}

/// The first estimator: UTF-8 bytes divided by 3, rounded up, without the size-record line.
pub fn bytes3_tokens(text: &str) -> usize {
    counted_text(text).len().div_ceil(3)
}

/// The text an estimator counts: every line but the size record, newline-terminated.
pub fn counted_text(text: &str) -> String {
    text.lines()
        .filter(|line| !line.starts_with(SIZE_RECORD_PREFIX))
        .flat_map(|line| [line, "\n"])
        .collect()
}

fn listing(items: &[(String, usize)]) -> String {
    let parts: Vec<String> = items
        .iter()
        .map(|(name, n)| format!("{name} {n}"))
        .collect();
    parts.join(", ")
}

fn strings(value: &Value, key: &str) -> Result<Vec<String>, String> {
    value[key]
        .as_array()
        .ok_or_else(|| format!("missing array {key}"))?
        .iter()
        .map(|item| {
            item.as_str()
                .map(str::to_owned)
                .ok_or_else(|| format!("{key}: not a string"))
        })
        .collect()
}

fn count(value: &Value, key: &str) -> Result<usize, String> {
    value[key]
        .as_u64()
        .and_then(|n| usize::try_from(n).ok())
        .ok_or_else(|| format!("missing count {key}"))
}

impl Spec {
    /// The spec this binary was built with.
    pub fn embedded() -> Spec {
        let text = |path: &str| {
            TEXTS
                .iter()
                .find(|(p, _)| *p == path)
                .map(|(_, t)| (*t).to_owned())
        };
        Spec::parse(MANIFEST, BUDGETS, &text).expect("the embedded spec manifest parses")
    }

    /// Paths of the embedded texts, so a test can compare them with the manifest.
    pub fn embedded_paths() -> Vec<&'static str> {
        TEXTS.iter().map(|(path, _)| *path).collect()
    }

    pub fn parse(
        manifest: &str,
        budgets: &str,
        text: &dyn Fn(&str) -> Option<String>,
    ) -> Result<Spec, String> {
        let manifest: Value =
            serde_json::from_str(manifest).map_err(|e| format!("chapters.json: {e}"))?;
        let budgets: Value =
            serde_json::from_str(budgets).map_err(|e| format!("budget.json: {e}"))?;
        let mut chapters = Vec::new();
        for entry in manifest["chapters"]
            .as_array()
            .ok_or("chapters.json: missing chapters")?
        {
            let field = |key: &str| {
                entry[key]
                    .as_str()
                    .map(str::to_owned)
                    .ok_or(format!("chapter: missing {key}"))
            };
            let (id, path) = (field("id")?, field("path")?);
            chapters.push(Chapter {
                text: text(&path).ok_or(format!("chapter {id}: no text at {path}"))?,
                core: entry["core"]
                    .as_bool()
                    .ok_or(format!("chapter {id}: missing core"))?,
                part: entry["part"]
                    .as_bool()
                    .ok_or(format!("chapter {id}: missing part"))?,
                requires: strings(entry, "requires")?,
                topics: strings(entry, "topics")?,
                tokens: Tokens {
                    bytes3: count(&entry["tokens"], "bytes3")?,
                    bpe: count(&entry["tokens"], "bpe")?,
                },
                id,
                path,
            });
        }
        let generated = manifest["generated"]
            .as_str()
            .ok_or("chapters.json: missing generated")?;
        Ok(Spec {
            generated: generated.to_owned(),
            chapters,
            budgets,
        })
    }

    fn budget(&self, key: &str) -> usize {
        count(&self.budgets, key).unwrap_or(0)
    }

    pub fn default_budget(&self) -> usize {
        self.budget("assembly_default")
    }

    fn chapter_budget(&self, id: &str) -> usize {
        count(&self.budgets["chapters"], id).unwrap_or_else(|_| self.budget("chapter"))
    }

    fn get(&self, id: &str) -> Option<&Chapter> {
        self.chapters.iter().find(|chapter| chapter.id == id)
    }

    /// The single-file spec: every `part` chapter in manifest order.
    pub fn generated_text(&self) -> String {
        self.chapters
            .iter()
            .filter(|c| c.part)
            .map(|c| c.text.as_str())
            .collect()
    }

    /// The manifest with `tokens` replaced by `measure`, in the canonical layout.
    pub fn manifest_with(&self, measure: &dyn Fn(&str) -> Tokens) -> String {
        let chapters: Vec<Value> = self
            .chapters
            .iter()
            .map(|c| {
                let tokens = measure(&c.text);
                serde_json::json!({
                    "id": c.id, "path": c.path, "core": c.core, "part": c.part,
                    "requires": c.requires, "topics": c.topics,
                    "tokens": {"bytes3": tokens.bytes3, "bpe": tokens.bpe},
                })
            })
            .collect();
        let manifest = serde_json::json!({"generated": self.generated, "chapters": chapters});
        serde_json::to_string_pretty(&manifest).expect("a JSON value serializes") + "\n"
    }

    /// Every broken rule, each with a named message. `bpe` is the second estimator.
    pub fn validate(&self, bpe: &dyn Fn(&str) -> usize) -> Vec<String> {
        let mut errors = Vec::new();
        let mut ids = BTreeSet::new();
        for chapter in &self.chapters {
            let id = &chapter.id;
            if !ids.insert(id.as_str()) {
                errors.push(format!("duplicate chapter id {id}"));
            }
            let measured = Tokens {
                bytes3: bytes3_tokens(&chapter.text),
                bpe: bpe(&counted_text(&chapter.text)),
            };
            if measured != chapter.tokens {
                errors.push(format!(
                    "stale tokens field: chapter {id} records {:?}, measured {measured:?}",
                    chapter.tokens
                ));
            }
            if chapter.topics.is_empty() || chapter.topics.iter().any(|t| t.is_empty()) {
                errors.push(format!(
                    "chapter {id}: every chapter needs non-empty topics"
                ));
            }
            for required in &chapter.requires {
                if self.get(required).is_none() {
                    errors.push(format!("chapter {id} requires unknown chapter {required}"));
                }
            }
            let budget = self.chapter_budget(id);
            if !chapter.core && chapter.tokens.gate() > budget {
                errors.push(self.over("chapter", id, chapter.tokens.gate(), budget, &[chapter]));
            }
        }
        errors.extend(self.cycle());
        let core: Vec<&Chapter> = self.chapters.iter().filter(|c| c.core).collect();
        let core_tokens = core.iter().map(|c| c.tokens.gate()).sum();
        if core_tokens > self.budget("core") {
            errors.push(self.over("core", "core", core_tokens, self.budget("core"), &core));
        }
        let all: Vec<&Chapter> = self.chapters.iter().collect();
        let total = all.iter().map(|c| c.tokens.gate()).sum();
        if total > self.budget("spec") {
            errors.push(self.over("whole spec", "spec", total, self.budget("spec"), &all));
        }
        errors.extend(duplicate_ids(&self.chapters));
        errors
    }

    fn over(
        &self,
        what: &str,
        key: &str,
        tokens: usize,
        budget: usize,
        scope: &[&Chapter],
    ) -> String {
        let mut sections: Vec<(String, usize)> =
            scope.iter().flat_map(|c| sections(&c.text)).collect();
        sections.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
        sections.truncate(5);
        format!(
            "{what} over budget: {key} is {tokens} tokens, {} over {budget} in docs/spec/budget.json; largest sections: {}",
            tokens - budget,
            listing(&sections)
        )
    }

    fn cycle(&self) -> Option<String> {
        fn visit<'a>(
            spec: &'a Spec,
            id: &'a str,
            path: &mut Vec<&'a str>,
            done: &mut BTreeSet<&'a str>,
        ) -> Option<String> {
            if let Some(start) = path.iter().position(|seen| *seen == id) {
                return Some(format!(
                    "requires cycle: {} -> {id}",
                    path[start..].join(" -> ")
                ));
            }
            if !done.insert(id) {
                return None;
            }
            path.push(id);
            let chapter = spec.get(id)?;
            let found = chapter
                .requires
                .iter()
                .find_map(|next| visit(spec, next, path, done));
            path.pop();
            found
        }
        let mut done = BTreeSet::new();
        self.chapters
            .iter()
            .find_map(|c| visit(self, &c.id, &mut Vec::new(), &mut done))
    }

    /// The core chapters, the chapters with any of `topics`, the named `chapters`, and
    /// everything they require, in manifest order; refused above `budget` tokens.
    pub fn assemble(
        &self,
        topics: &[String],
        chapters: &[String],
        budget: usize,
    ) -> Result<Assembly, SpecError> {
        let mut chosen: BTreeSet<&str> = self
            .chapters
            .iter()
            .filter(|c| c.core)
            .map(|c| c.id.as_str())
            .collect();
        for topic in topics {
            let matching: Vec<&str> = self
                .chapters
                .iter()
                .filter(|c| c.topics.contains(topic))
                .map(|c| c.id.as_str())
                .collect();
            if matching.is_empty() {
                return Err(SpecError::UnknownTopic(topic.clone()));
            }
            chosen.extend(matching);
        }
        for id in chapters {
            chosen.insert(
                &self
                    .get(id)
                    .ok_or_else(|| SpecError::UnknownChapter(id.clone()))?
                    .id,
            );
        }
        let mut pending: Vec<&str> = chosen.iter().copied().collect();
        while let Some(id) = pending.pop() {
            for required in self
                .get(id)
                .map(|c| c.requires.as_slice())
                .unwrap_or_default()
            {
                if chosen.insert(required) {
                    pending.push(required);
                }
            }
        }
        let picked: Vec<&Chapter> = self
            .chapters
            .iter()
            .filter(|c| chosen.contains(c.id.as_str()))
            .collect();
        let tokens = picked.iter().map(|c| c.tokens.gate()).sum();
        if tokens > budget {
            let mut largest: Vec<(String, usize)> = picked
                .iter()
                .map(|c| (c.id.clone(), c.tokens.gate()))
                .collect();
            largest.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
            largest.truncate(5);
            return Err(SpecError::OverBudget {
                budget,
                tokens,
                largest,
            });
        }
        Ok(Assembly {
            chapters: picked
                .iter()
                .map(|c| (c.id.clone(), c.path.clone(), c.tokens.gate()))
                .collect(),
            tokens,
            text: picked.iter().map(|c| c.text.as_str()).collect(),
        })
    }
}

/// The `##`/`###` sections of a chapter with their bytes/3 sizes, for budget failures.
fn sections(text: &str) -> Vec<(String, usize)> {
    let mut out: Vec<(String, String)> = Vec::new();
    for line in text.lines() {
        if line.starts_with("## ") || line.starts_with("### ") || out.is_empty() {
            out.push((
                line.trim_start_matches('#').trim().to_owned(),
                String::new(),
            ));
        }
        let last = out.last_mut().expect("a section was pushed");
        last.1.push_str(line);
        last.1.push('\n');
    }
    out.into_iter()
        .map(|(name, body)| (name, bytes3_tokens(&body)))
        .collect()
}

/// Normative ids (`[N...]`, `[NA...]`) must be unique across every chapter.
fn duplicate_ids(chapters: &[Chapter]) -> Vec<String> {
    let mut seen: BTreeMap<String, &str> = BTreeMap::new();
    let mut errors = Vec::new();
    for chapter in chapters {
        let mut fenced = false;
        for line in chapter.text.lines() {
            fenced ^= line.starts_with("```");
            let id = line
                .strip_prefix('[')
                .and_then(|rest| rest.split_once(']'))
                .map(|(id, _)| id);
            let Some(id) = id.filter(|id| !fenced && id.starts_with('N')) else {
                continue;
            };
            if let Some(first) = seen.insert(id.to_owned(), &chapter.id) {
                errors.push(format!(
                    "duplicate normative id {id} in chapters {first} and {}",
                    chapter.id
                ));
            }
        }
    }
    errors
}
