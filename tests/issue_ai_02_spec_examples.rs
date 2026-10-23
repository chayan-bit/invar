//! AI-02: the compact spec, the verified example library and the docs link/size gates.
//!
//! Every fenced block of `docs/spec/invar-v0.1.md` carries an `outcome=` tag that is executed
//! through the real `invar` binary. Every example listed in `examples/index.json` is run on the
//! reference evaluator and on each engine it declares. The validators are plain functions so the
//! negative tests can feed them deliberately broken inputs.
#![forbid(unsafe_code)]

use std::{
    collections::{BTreeMap, BTreeSet},
    path::{Path, PathBuf},
    process::Command,
};

use invar::{check, eval, front, mir, mir_eval};
use serde_json::Value;
use sha2::{Digest, Sha256};

const SPEC: &str = "docs/spec/invar-v0.1.md";
const INDEX: &str = "examples/index.json";
const INDEX_SCHEMA: &str = "examples/index.schema.json";
/// Target from the AI-02 contract. Measured with the stated estimator, see `spec_tokens`.
const TOKEN_LIMIT: usize = 30_000;
const SIZE_RECORD_PREFIX: &str = "Size record:";

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn read(relative: &str) -> String {
    std::fs::read_to_string(root().join(relative)).unwrap_or_else(|e| panic!("{relative}: {e}"))
}

fn cli(args: &[&str]) -> (i32, Value) {
    let output = Command::new(env!("CARGO_BIN_EXE_invar"))
        .args(args)
        .current_dir(root())
        .output()
        .unwrap();
    let document = serde_json::from_slice(&output.stdout).unwrap_or_else(|e| {
        panic!(
            "{args:?}: not JSON ({e}): {}",
            String::from_utf8_lossy(&output.stdout)
        )
    });
    (output.status.code().unwrap(), document)
}

// ---------------------------------------------------------------- spec blocks

#[derive(Debug)]
struct Block {
    line: usize,
    lang: String,
    outcome: String,
    flags: BTreeSet<String>,
    attrs: BTreeMap<String, String>,
    body: String,
}

/// Extract every fenced block; an untagged block is an error.
fn extract_blocks(markdown: &str) -> Result<Vec<Block>, String> {
    let mut blocks = Vec::new();
    let mut open: Option<Block> = None;
    for (index, text) in markdown.lines().enumerate() {
        let line = index + 1;
        if let Some(block) = open.as_mut() {
            if text.trim_end() == "```" {
                blocks.push(open.take().unwrap());
            } else {
                block.body.push_str(text);
                block.body.push('\n');
            }
        } else if let Some(info) = text.strip_prefix("```") {
            let mut words = info.split_whitespace();
            let lang = words.next().unwrap_or("").to_owned();
            let mut outcome = None;
            let (mut flags, mut attrs) = (BTreeSet::new(), BTreeMap::new());
            for word in words {
                match word.split_once('=') {
                    Some(("outcome", value)) => outcome = Some(value.to_owned()),
                    Some((key, value)) => {
                        attrs.insert(key.to_owned(), value.to_owned());
                    }
                    None => {
                        flags.insert(word.to_owned());
                    }
                }
            }
            let outcome = outcome.ok_or_else(|| format!("line {line}: untagged code block"))?;
            open = Some(Block {
                line,
                lang,
                outcome,
                flags,
                attrs,
                body: String::new(),
            });
        }
    }
    match open {
        Some(block) => Err(format!("line {}: unterminated code block", block.line)),
        None => Ok(blocks),
    }
}

fn scratch(name: &str, source: &str) -> PathBuf {
    let path = std::env::temp_dir().join(format!("invar-ai-02-{}-{name}.inv", std::process::id()));
    std::fs::write(&path, source).unwrap();
    path
}

fn first_code(document: &Value) -> String {
    document["diagnostics"][0]["code"]
        .as_str()
        .unwrap_or("<none>")
        .to_owned()
}

/// Run one block and compare with its tagged outcome.
fn run_block(block: &Block) -> Result<(), String> {
    let at = format!("line {} (outcome={})", block.line, block.outcome);
    if block.outcome == "none" {
        return if block.lang == "invar" {
            Err(format!("{at}: an invar block cannot be outcome=none"))
        } else {
            Ok(())
        };
    }
    let path = scratch(&format!("block-{}", block.line), &block.body);
    let file = path.to_str().unwrap();
    let foreign = block.flags.contains("foreign");
    let (kind, arg) = block
        .outcome
        .split_once(':')
        .unwrap_or((block.outcome.as_str(), ""));
    let result = match kind {
        "value" | "trap" | "limit" => {
            let fuel = block.attrs.get("fuel").map_or("1000000", String::as_str);
            let mut args = vec![
                "run",
                file,
                "--engine",
                "reference",
                "--fuel",
                fuel,
                "--format",
                "json",
            ];
            if foreign {
                args.extend(["--c-library", env!("CARGO_BIN_EXE_invar")]);
            }
            let (_, document) = cli(&args);
            match (kind, document["status"].as_str().unwrap_or("")) {
                ("value", "ok")
                    if document["result"]["value"]
                        .as_i64()
                        .map(|v| v.to_string())
                        .as_deref()
                        == Some(arg) =>
                {
                    Ok(())
                }
                ("trap", "trap") if document["trap"]["category"] == arg => Ok(()),
                ("limit", "evaluation_limit") => Ok(()),
                _ => Err(format!("{at}: got {document}")),
            }
        }
        "check" | "diag" => {
            let (_, document) = cli(&["check", file, "--format", "json"]);
            match (kind, document["status"].as_str().unwrap_or("")) {
                ("check", "ok") => Ok(()),
                ("diag", "source_error") if first_code(&document) == arg => Ok(()),
                _ => Err(format!("{at}: got {document}")),
            }
        }
        "system" => {
            let (_, document) = cli(&["system", "check", file, "--format", "json"]);
            let status = document["status"].as_str().unwrap_or("");
            if (arg == "ok" && status == "ok")
                || (status == "source_error" && first_code(&document) == arg)
            {
                Ok(())
            } else {
                Err(format!("{at}: got {document}"))
            }
        }
        other => Err(format!("{at}: unknown outcome kind {other}")),
    };
    let _ = std::fs::remove_file(path);
    result
}

// ------------------------------------------------------------ citations, size

#[derive(Debug, Default, PartialEq)]
struct Stats {
    statements: usize,
    citations: usize,
}

fn is_citation(token: &str) -> bool {
    let path = token.split("::").next().unwrap_or("");
    [
        "rs", "md", "json", "inv", "h", "c", "toml", "py", "txt", "ebnf",
    ]
    .iter()
    .any(|ext| path.ends_with(&format!(".{ext}")))
        && path
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || "_./-".contains(c))
}

/// Check the normative/note discipline. `has` answers whether a repo file exists, `contains`
/// whether a file contains an identifier (for `path::name` citations).
fn check_spec_text(
    markdown: &str,
    has: &dyn Fn(&str) -> bool,
    contains: &dyn Fn(&str, &str) -> bool,
) -> Result<Stats, Vec<String>> {
    let (mut errors, mut ids, mut stats) = (Vec::new(), BTreeSet::new(), Stats::default());
    let mut in_fence = false;
    for (index, text) in markdown.lines().enumerate() {
        let line = index + 1;
        if text.starts_with("```") {
            in_fence = !in_fence;
            continue;
        }
        if in_fence || text.trim().is_empty() || text.starts_with('#') || text.starts_with('|') {
            continue;
        }
        if text.contains('\u{2014}') {
            errors.push(format!("line {line}: em dash"));
        }
        let outside_code: String = text.split('`').step_by(2).collect();
        if outside_code.contains(". ")
            && outside_code
                .split(". ")
                .skip(1)
                .any(|s| s.starts_with(|c: char| c.is_uppercase()))
        {
            errors.push(format!("line {line}: more than one sentence on a line"));
        }
        if let Some(rest) = text.strip_prefix("[N") {
            let Some((id, body)) = rest.split_once(']') else {
                errors.push(format!("line {line}: malformed statement id"));
                continue;
            };
            if !ids.insert(id.to_owned()) {
                errors.push(format!("line {line}: duplicate statement id N{id}"));
            }
            let cites: Vec<&str> = body
                .split('`')
                .skip(1)
                .step_by(2)
                .filter(|t| is_citation(t))
                .collect();
            if cites.is_empty() {
                errors.push(format!("line {line}: statement N{id} has no citation"));
            }
            for cite in cites {
                stats.citations += 1;
                let (path, name) = cite.split_once("::").unwrap_or((cite, ""));
                if !has(path) {
                    errors.push(format!("line {line}: N{id} cites missing file {path}"));
                } else if !name.is_empty() && !contains(path, name) {
                    errors.push(format!(
                        "line {line}: N{id} cites {path}::{name}, not found"
                    ));
                }
            }
            stats.statements += 1;
        } else if !["Note:", "Status:", SIZE_RECORD_PREFIX, "Counts:"]
            .iter()
            .any(|p| text.starts_with(p))
        {
            errors.push(format!(
                "line {line}: prose must be a [N] statement or a Note: ({})",
                text.chars().take(40).collect::<String>()
            ));
        }
    }
    if errors.is_empty() {
        Ok(stats)
    } else {
        Err(errors)
    }
}

/// Estimator: UTF-8 bytes divided by 3, rounded up, excluding the size-record line.
/// For English prose and code this is above the typical 3.5-4 bytes per token of BPE tokenizers,
/// so it errs on the side of reporting more tokens than a real tokenizer would.
fn spec_tokens(markdown: &str) -> usize {
    let bytes: usize = markdown
        .lines()
        .filter(|l| !l.starts_with(SIZE_RECORD_PREFIX))
        .map(|l| l.len() + 1)
        .sum();
    bytes.div_ceil(3)
}

fn recorded_tokens(markdown: &str) -> Option<usize> {
    let line = markdown
        .lines()
        .find(|l| l.starts_with(SIZE_RECORD_PREFIX))?;
    line.split("tokens=")
        .nth(1)?
        .split_whitespace()
        .next()?
        .parse()
        .ok()
}

// --------------------------------------------------------------------- index

fn section_ids(markdown: &str) -> BTreeSet<String> {
    markdown
        .lines()
        .filter(|l| l.starts_with('#'))
        .filter_map(|l| {
            l.trim_start_matches('#')
                .split_whitespace()
                .next()
                .map(|w| w.trim_end_matches('.').to_owned())
        })
        // Numbered sections, and appendix subsections such as `B.1`.
        .filter(|w| {
            let b = w.as_bytes();
            b.first().is_some_and(u8::is_ascii_digit)
                || (b.len() > 2 && b[0].is_ascii_uppercase() && b[1] == b'.')
        })
        .collect()
}

fn example_files() -> BTreeSet<String> {
    fn visit(dir: &Path, out: &mut BTreeSet<String>) {
        for entry in std::fs::read_dir(dir).unwrap() {
            let path = entry.unwrap().path();
            if path.is_dir() {
                visit(&path, out);
            } else if path.extension().is_some_and(|e| e == "inv") {
                out.insert(
                    path.strip_prefix(root())
                        .unwrap()
                        .to_str()
                        .unwrap()
                        .to_owned(),
                );
            }
        }
    }
    let mut out = BTreeSet::new();
    visit(&root().join("examples"), &mut out);
    out
}

/// Validate index shape, uniqueness, file coverage in both directions and spec references.
fn validate_index(
    index: &Value,
    schema: &Value,
    files: &BTreeSet<String>,
    sections: &BTreeSet<String>,
) -> Result<(), Vec<String>> {
    let mut errors = Vec::new();
    let validator = jsonschema::validator_for(schema).unwrap();
    for error in validator.iter_errors(index) {
        errors.push(format!("schema: {error}"));
    }
    if !errors.is_empty() {
        return Err(errors);
    }
    let mut listed = BTreeSet::new();
    let mut covered = BTreeSet::new();
    for entry in index["examples"].as_array().unwrap() {
        let path = entry["path"].as_str().unwrap();
        // A runnable example always runs on the reference oracle; check and
        // system examples run only their declared check commands.
        if entry["expect"]["command"] == "run"
            && !entry["engines"]
                .as_array()
                .unwrap()
                .iter()
                .any(|e| e == "reference")
        {
            errors.push(format!(
                "{path}: runnable example without the reference engine"
            ));
        }
        if !listed.insert(path.to_owned()) {
            errors.push(format!("{path}: listed more than once"));
        }
        if !files.contains(path) {
            errors.push(format!("{path}: index entry without a file"));
        }
        let section = entry["spec_section"].as_str().unwrap();
        if !sections.contains(section) {
            errors.push(format!("{path}: dangling spec section {section}"));
        }
        covered.insert(entry["family"].as_str().unwrap().to_owned());
    }
    for file in files.difference(&listed) {
        errors.push(format!("{file}: example missing from the index"));
    }
    let pending: BTreeSet<String> = index["pending_families"]
        .as_array()
        .unwrap()
        .iter()
        .map(|p| p["family"].as_str().unwrap().to_owned())
        .collect();
    for family in schema["$defs"]["family"]["enum"].as_array().unwrap() {
        let family = family.as_str().unwrap();
        if covered.contains(family) == pending.contains(family) {
            errors.push(format!(
                "family {family}: needs examples xor a pending entry"
            ));
        }
    }
    if errors.is_empty() {
        Ok(())
    } else {
        Err(errors)
    }
}

fn hex(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

/// Check the observation of one engine against the example's expectation.
fn matches_expect(
    expect: &Value,
    status: &str,
    value: Option<i64>,
    category: Option<&str>,
    code: &str,
) -> Result<(), String> {
    let wanted = expect["status"].as_str().unwrap();
    if wanted != status {
        return Err(format!("status {status}, expected {wanted}"));
    }
    if let Some(v) = expect.get("value")
        && value != v.as_i64()
    {
        return Err(format!("value {value:?}, expected {v}"));
    }
    if let Some(c) = expect.get("category")
        && category != c.as_str()
    {
        return Err(format!("trap {category:?}, expected {c}"));
    }
    if let Some(c) = expect.get("code")
        && !code.is_empty()
        && code != c.as_str().unwrap()
    {
        return Err(format!("code {code}, expected {c}"));
    }
    Ok(())
}

fn observe_cli(args: &[&str]) -> (String, Option<i64>, Option<String>, String) {
    let (_, d) = cli(args);
    (
        d["status"].as_str().unwrap_or("").to_owned(),
        d["result"]["value"].as_i64(),
        d["trap"]["category"].as_str().map(str::to_owned),
        first_code(&d),
    )
}

/// File count and SHA-256 over sorted (relative path, length, bytes) of a generated tree.
fn tree_hash(dir: &Path) -> (usize, String) {
    fn visit(base: &Path, dir: &Path, out: &mut Vec<(String, PathBuf)>) {
        for entry in std::fs::read_dir(dir).unwrap() {
            let path = entry.unwrap().path();
            if path.is_dir() {
                visit(base, &path, out);
            } else {
                out.push((
                    path.strip_prefix(base)
                        .unwrap()
                        .to_str()
                        .unwrap()
                        .to_owned(),
                    path,
                ));
            }
        }
    }
    let mut files = Vec::new();
    visit(dir, dir, &mut files);
    files.sort();
    let mut hasher = Sha256::new();
    for (relative, path) in &files {
        let bytes = std::fs::read(path).unwrap();
        hasher.update(format!("{relative}\0{}\0", bytes.len()));
        hasher.update(&bytes);
    }
    (
        files.len(),
        hasher
            .finalize()
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect(),
    )
}

/// Run the named source-family recognizer of the library on checked source.
fn recognize(module: &str, source: &[u8], entry: &str) -> Result<(), String> {
    use invar::{
        compaction_source, contraction_source, frontier_source, histogram_source, indexed_source,
        segmented_source, stencil_source, summary_source, transpose_source,
    };
    let checked = check::check(front::parse(source).map_err(|e| e.code.to_string())?, true)
        .map_err(|e| e.code.to_string())?;
    let outcome = match module {
        "transpose_source" => transpose_source::Contract::from_checked(&checked, entry)
            .map(|_| ())
            .map_err(|e| format!("{e:?}")),
        "stencil_source" => stencil_source::Contract::from_checked(&checked, entry)
            .map(|_| ())
            .map_err(|e| format!("{e:?}")),
        "summary_source" => summary_source::Contract::from_checked(&checked, entry)
            .map(|_| ())
            .map_err(|e| format!("{e:?}")),
        "indexed_source" => indexed_source::Contract::from_checked(source, &checked, entry)
            .map(|_| ())
            .map_err(|e| format!("{e:?}")),
        "histogram_source" => histogram_source::Contract::from_checked(source, &checked, entry)
            .map(|_| ())
            .map_err(|e| format!("{e:?}")),
        "frontier_source" => frontier_source::Contract::from_checked(source, &checked, entry)
            .map(|_| ())
            .map_err(|e| format!("{e:?}")),
        "segmented_source" => segmented_source::Contract::from_source(source, entry)
            .map(|_| ())
            .map_err(|e| format!("{e:?}")),
        "compaction_source" => compaction_source::Contract::from_source(source, entry)
            .map(|_| ())
            .map_err(|e| format!("{e:?}")),
        "contraction_source" => contraction_source::Contract::from_source(source, entry)
            .map(|_| ())
            .map_err(|e| format!("{e:?}")),
        other => Err(format!("unknown recognizer module {other}")),
    };
    outcome.map_err(|e| format!("{module}::{entry} did not recognize the source: {e}"))
}

fn trap_code(category: eval::TrapCategory) -> &'static str {
    match category {
        eval::TrapCategory::IntegerOverflow => "INTEGER_OVERFLOW",
        eval::TrapCategory::DivisionByZero => "DIVISION_BY_ZERO",
        eval::TrapCategory::IndexOutOfBounds => "INDEX_OUT_OF_BOUNDS",
        eval::TrapCategory::UnitConversionInexact => "UNIT_CONVERSION_INEXACT",
    }
}

/// Run one engine of one example; returns a counted result or an error text.
fn run_engine(entry: &Value, engine: &str) -> Result<(), String> {
    let path = entry["path"].as_str().unwrap();
    let full = root().join(path);
    let file = full.to_str().unwrap();
    let expect = &entry["expect"];
    let source = std::fs::read(&full).unwrap();
    let foreign = String::from_utf8_lossy(&source).contains("extern c fn");
    let at = |e: String| format!("{path} [{engine}]: {e}");
    match (expect["command"].as_str().unwrap(), engine) {
        (_, "explain") => {
            let (_, document) = cli(&["explain", file, "--format", "json"]);
            for region in expect["explain"]["regions"].as_array().unwrap() {
                let found = document["regions"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .find(|r| r["function"] == region["function"]);
                let Some(found) = found else {
                    return Err(at(format!("no region {}", region["function"])));
                };
                if found["status"] != region["status"]
                    || found["execution"]["kind"] != region["kind"]
                {
                    return Err(at(format!(
                        "region {} is {} {}",
                        region["function"], found["status"], found["execution"]["kind"]
                    )));
                }
            }
            Ok(())
        }
        (_, "recognize") => {
            let module = expect["recognize"]["module"].as_str().unwrap();
            recognize(
                module,
                &source,
                expect["recognize"]["entry"].as_str().unwrap(),
            )
            .map_err(at)
        }
        ("system", "generate") => {
            for target in expect["generate"].as_array().unwrap() {
                let name = target["target"].as_str().unwrap();
                let mut hashes = Vec::new();
                for attempt in 0..2 {
                    let out = std::env::temp_dir().join(format!(
                        "invar-ai-02-gen-{}-{name}-{attempt}",
                        std::process::id()
                    ));
                    let _ = std::fs::remove_dir_all(&out);
                    let (status, document) =
                        cli(&["system", "gen", name, file, "--out", out.to_str().unwrap()]);
                    if status != 0 {
                        return Err(at(format!("generate {name}: {document}")));
                    }
                    hashes.push(tree_hash(&out));
                    let _ = std::fs::remove_dir_all(&out);
                }
                let (files, hash) = &hashes[0];
                if hashes[0] != hashes[1]
                    || *files != target["files"].as_u64().unwrap() as usize
                    || hash != target["tree_sha256"].as_str().unwrap()
                {
                    return Err(at(format!(
                        "generate {name}: {files} files {hash}, double run equal {}",
                        hashes[0] == hashes[1]
                    )));
                }
            }
            Ok(())
        }
        ("check", "check") => {
            let (s, _, _, c) = observe_cli(&["check", file, "--format", "json"]);
            matches_expect(expect, &s, None, None, &c).map_err(at)
        }
        ("system", "system") => {
            let (_, document) = cli(&["system", "check", file, "--format", "json"]);
            let status = document["status"].as_str().unwrap_or("");
            matches_expect(expect, status, None, None, "").map_err(at)?;
            let observed: Vec<String> = document["diagnostics"]
                .as_array()
                .unwrap()
                .iter()
                .map(|d| {
                    format!(
                        "{}:{}",
                        d["code"].as_str().unwrap(),
                        d["severity"].as_str().unwrap()
                    )
                })
                .collect();
            let wanted: Vec<String> = expect["diagnostics"].as_array().map_or(vec![], |w| {
                w.iter().map(|x| x.as_str().unwrap().to_owned()).collect()
            });
            if observed != wanted {
                return Err(at(format!("diagnostics {observed:?}, expected {wanted:?}")));
            }
            if let Some(hash) = expect["ir_sha256"].as_str() {
                let output = Command::new(env!("CARGO_BIN_EXE_invar"))
                    .args(["system", "ir", file])
                    .current_dir(root())
                    .output()
                    .unwrap();
                let again = Command::new(env!("CARGO_BIN_EXE_invar"))
                    .args(["system", "ir", file])
                    .current_dir(root())
                    .output()
                    .unwrap();
                if output.stdout != again.stdout || hex(&output.stdout) != hash {
                    return Err(at(format!(
                        "system IR hash {} (double run equal: {})",
                        hex(&output.stdout),
                        output.stdout == again.stdout
                    )));
                }
                let ir: Value = serde_json::from_slice(&output.stdout).unwrap();
                let name = ir["schema"].as_str().unwrap();
                let schema: Value =
                    serde_json::from_str(&read(&format!("schema/{name}.schema.json"))).unwrap();
                if !jsonschema::is_valid(&schema, &ir) {
                    return Err(at("system IR does not validate against its schema".into()));
                }
            }
            Ok(())
        }
        ("run", "reference") => {
            let mut args = vec!["run", file, "--engine", "reference", "--format", "json"];
            if foreign {
                args.extend(["--c-library", env!("CARGO_BIN_EXE_invar")]);
            }
            let (s, v, t, c) = observe_cli(&args);
            matches_expect(expect, &s, v, t.as_deref(), &c).map_err(at)
        }
        ("run", "mir") => {
            let checked = check::check(
                front::parse(&source).map_err(|e| at(e.code.to_string()))?,
                true,
            )
            .map_err(|e| at(e.code.to_string()))?;
            let program = mir::lower(&checked).map_err(|e| at(e.code.to_string()))?;
            let reference = eval::observe(&checked, 1_000_000);
            let observed = mir_eval::observe(&program, 1_000_000);
            let same = format!("{:?}", observed.result) == format!("{:?}", reference.result)
                && observed.steps_used == reference.steps_used
                && format!("{:?}", observed.cleanup_trace)
                    == format!("{:?}", reference.cleanup_trace);
            if !same {
                return Err(at(format!(
                    "MIR {:?} differs from reference {:?}",
                    observed.result, reference.result
                )));
            }
            let (status, value, category) = match &observed.result {
                Ok(eval::Value::I32(v)) => ("ok", Some(i64::from(*v)), None),
                Err(eval::EvalError::Trap { category, .. }) => {
                    ("trap", None, Some(trap_code(*category).to_owned()))
                }
                Err(eval::EvalError::EvaluationLimit { .. }) => ("evaluation_limit", None, None),
                other => return Err(at(format!("unexpected {other:?}"))),
            };
            matches_expect(expect, status, value, category.as_deref(), "").map_err(at)
        }
        ("run", "cpu-parallel") => {
            for workers in ["1", "2", "3", "4"] {
                let (s, v, t, c) = observe_cli(&[
                    "run",
                    file,
                    "--engine",
                    "cpu-parallel",
                    "--workers",
                    workers,
                    "--format",
                    "json",
                ]);
                matches_expect(expect, &s, v, t.as_deref(), &c)
                    .map_err(|e| at(format!("workers {workers}: {e}")))?;
            }
            Ok(())
        }
        ("run", "native-serial") => {
            if cfg!(feature = "native") {
                let mut args = vec!["run", file, "--engine", "native-serial", "--format", "json"];
                let library = std::env::var("INVAR_AI02_C_LIBRARY").unwrap_or_default();
                if foreign {
                    args.extend(["--c-library", library.as_str()]);
                }
                let (s, v, t, c) = observe_cli(&args);
                matches_expect(expect, &s, v, t.as_deref(), &c).map_err(at)
            } else {
                Ok(())
            }
        }
        (command, other) => Err(at(format!(
            "engine {other} is not defined for command {command}"
        ))),
    }
}

/// Run every declared engine of one example: (executed, skipped) engine runs.
/// `native-serial` is skipped, and counted as skipped, in a build without `native`.
fn run_example(entry: &Value) -> Result<(usize, usize), String> {
    let (mut executed, mut skipped) = (0, 0);
    for engine in entry["engines"].as_array().unwrap() {
        let engine = engine.as_str().unwrap();
        run_engine(entry, engine)?;
        if engine == "native-serial" && !cfg!(feature = "native") {
            skipped += 1;
        } else {
            executed += 1;
        }
    }
    Ok((executed, skipped))
}

// ------------------------------------------------------------------ doc links

/// Relative markdown links outside code that point to a file that does not exist.
fn broken_links(docs: &[(String, String)], exists: &dyn Fn(&str) -> bool) -> Vec<String> {
    let mut broken = Vec::new();
    for (file, text) in docs {
        let mut in_fence = false;
        for line in text.lines() {
            if line.starts_with("```") {
                in_fence = !in_fence;
            }
            if in_fence {
                continue;
            }
            let mut rest = line;
            while let Some(start) = rest.find("](") {
                let after = &rest[start + 2..];
                let end = after.find(')').unwrap_or(after.len());
                let target = after[..end].split_whitespace().next().unwrap_or("");
                rest = &after[end.min(after.len())..];
                let local = !target.is_empty() && !target.starts_with('#') && !target.contains(':');
                if local {
                    let resolved = Path::new(file)
                        .parent()
                        .unwrap()
                        .join(target.split('#').next().unwrap());
                    let normalized = normalize(&resolved);
                    if !exists(&normalized) {
                        broken.push(format!("{file}: link to missing file {target}"));
                    }
                }
            }
        }
    }
    broken
}

fn normalize(path: &Path) -> String {
    let mut parts: Vec<String> = Vec::new();
    for component in path.components() {
        match component.as_os_str().to_str().unwrap() {
            "." => {}
            ".." => {
                parts.pop();
            }
            other => parts.push(other.to_owned()),
        }
    }
    parts.join("/")
}

fn active_docs() -> Vec<(String, String)> {
    fn visit(dir: &Path, out: &mut Vec<String>) {
        for entry in std::fs::read_dir(dir).unwrap() {
            let path = entry.unwrap().path();
            let relative = path
                .strip_prefix(root())
                .unwrap()
                .to_str()
                .unwrap()
                .to_owned();
            if path.is_dir() {
                visit(&path, out);
            } else if relative.ends_with(".md") {
                out.push(relative);
            }
        }
    }
    let mut files = vec![
        "README.md".to_owned(),
        "CONTRIBUTING.md".to_owned(),
        "examples/README.md".to_owned(),
    ];
    visit(&root().join("docs"), &mut files);
    visit(&root().join("spec"), &mut files);
    files.sort();
    files.into_iter().map(|f| (f.clone(), read(&f))).collect()
}

// ------------------------------------------------------------------- the tests

fn repo_has(path: &str) -> bool {
    root().join(path).is_file()
}

fn repo_contains(path: &str, name: &str) -> bool {
    read(path).contains(name)
}

#[test]
fn issue_ai_02_spec_examples_every_spec_block_runs_with_its_tagged_outcome() {
    let spec = read(SPEC);
    let blocks = extract_blocks(&spec).unwrap();
    let executed: Vec<&Block> = blocks.iter().filter(|b| b.outcome != "none").collect();
    let failures: Vec<String> = executed.iter().filter_map(|b| run_block(b).err()).collect();
    assert!(failures.is_empty(), "{}", failures.join("\n"));
    for block in blocks.iter().filter(|b| b.outcome == "none") {
        run_block(block).unwrap();
    }
    assert!(
        executed.len() >= 40,
        "only {} executable blocks",
        executed.len()
    );
    eprintln!(
        "spec blocks: {} total, {} executed",
        blocks.len(),
        executed.len()
    );
}

#[test]
fn issue_ai_02_spec_examples_normative_statements_cite_existing_sources() {
    let stats = check_spec_text(&read(SPEC), &repo_has, &repo_contains)
        .unwrap_or_else(|e| panic!("{}", e.join("\n")));
    assert!(stats.statements >= 100, "{stats:?}");
    assert!(stats.citations >= stats.statements, "{stats:?}");
    let spec = read(SPEC);
    let counts = spec
        .lines()
        .find(|l| l.starts_with("Counts:"))
        .expect("Counts: record");
    assert!(
        counts.contains(&format!("statements={}", stats.statements)),
        "{counts} vs {stats:?}"
    );
    assert!(
        counts.contains(&format!("citations={}", stats.citations)),
        "{counts} vs {stats:?}"
    );
    eprintln!(
        "spec statements: {}, citations: {}",
        stats.statements, stats.citations
    );
}

#[test]
fn issue_ai_02_spec_examples_token_count_is_below_the_target_and_recorded() {
    let spec = read(SPEC);
    let measured = spec_tokens(&spec);
    assert!(measured < TOKEN_LIMIT, "{measured} tokens");
    assert_eq!(
        recorded_tokens(&spec),
        Some(measured),
        "update the Size record line to tokens={measured}"
    );
    eprintln!("spec tokens (bytes/3 estimator): {measured} of {TOKEN_LIMIT}");
}

#[test]
fn issue_ai_02_spec_examples_index_validates_and_every_example_matches_its_expected_output() {
    let index: Value = serde_json::from_str(&read(INDEX)).unwrap();
    let schema: Value = serde_json::from_str(&read(INDEX_SCHEMA)).unwrap();
    let sections = section_ids(&read(SPEC));
    validate_index(&index, &schema, &example_files(), &sections)
        .unwrap_or_else(|e| panic!("{}", e.join("\n")));
    let (mut runs, mut skipped, mut failures) = (0, 0, Vec::new());
    for entry in index["examples"].as_array().unwrap() {
        match run_example(entry) {
            Ok((executed, not_built)) => {
                runs += executed;
                skipped += not_built;
            }
            Err(error) => failures.push(error),
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
    let examples = index["examples"].as_array().unwrap().len();
    assert!(
        examples >= 60 && runs >= examples,
        "{examples} examples, {runs} engine runs"
    );
    eprintln!(
        "examples: {examples}, engine runs executed: {runs}, native-serial {}",
        if cfg!(feature = "native") {
            "executed".to_owned()
        } else {
            format!("not built (feature off): {skipped} runs skipped")
        }
    );
}

#[test]
fn issue_ai_02_spec_examples_active_docs_have_no_link_to_a_removed_file() {
    let docs = active_docs();
    assert!(docs.len() > 20);
    let broken = broken_links(&docs, &repo_has);
    assert!(broken.is_empty(), "{}", broken.join("\n"));
}

// ------------------------------------------------------------ negative cases

fn block(outcome: &str, body: &str) -> Block {
    Block {
        line: 1,
        lang: "invar".into(),
        outcome: outcome.into(),
        flags: BTreeSet::new(),
        attrs: BTreeMap::new(),
        body: body.into(),
    }
}

#[test]
fn issue_ai_02_spec_examples_rejects_an_untagged_spec_block() {
    assert!(
        extract_blocks("text\n```invar\nfn main() -> i32 { return 1; }\n```\n")
            .unwrap_err()
            .contains("untagged")
    );
    assert!(extract_blocks("```\nplain\n```\n").is_err());
    assert!(
        extract_blocks("```invar outcome=value:1\nfn main() -> i32 { return 1; }\n```\n").is_ok()
    );
    assert!(run_block(&block("none", "fn main() -> i32 { return 1; }")).is_err());
}

#[test]
fn issue_ai_02_spec_examples_rejects_a_block_whose_outcome_differs() {
    let program = "fn main() -> i32 { return 7; }";
    assert!(run_block(&block("value:7", program)).is_ok());
    assert!(run_block(&block("value:8", program)).is_err());
    assert!(run_block(&block("trap:INTEGER_OVERFLOW", program)).is_err());
    assert!(run_block(&block("diag:TYPE_MISMATCH", program)).is_err());
    assert!(
        run_block(&block(
            "diag:OWN_USE_AFTER_MOVE",
            "fn main() -> i32 { return true; }"
        ))
        .is_err()
    );
    assert!(
        run_block(&block(
            "diag:TYPE_MISMATCH",
            "fn main() -> i32 { return true; }"
        ))
        .is_ok()
    );
}

#[test]
fn issue_ai_02_spec_examples_rejects_statements_without_a_valid_citation() {
    let has = |p: &str| p == "src/check.rs";
    let contains = |_: &str, n: &str| n == "check";
    assert!(check_spec_text("[N1.1] Fine. `src/check.rs::check`\n", &has, &contains).is_ok());
    assert!(check_spec_text("[N1.1] No source here.\n", &has, &contains).is_err());
    assert!(check_spec_text("[N1.1] Gone. `src/removed.rs`\n", &has, &contains).is_err());
    assert!(
        check_spec_text(
            "[N1.1] Wrong name. `src/check.rs::nothing`\n",
            &has,
            &contains
        )
        .is_err()
    );
    assert!(check_spec_text("Loose prose without a marker.\n", &has, &contains).is_err());
    assert!(
        check_spec_text(
            "[N1.1] A. `src/check.rs`\n[N1.1] B. `src/check.rs`\n",
            &has,
            &contains
        )
        .is_err()
    );
    assert!(
        check_spec_text(
            "[N1.1] One. Two sentences here. `src/check.rs`\n",
            &has,
            &contains
        )
        .is_err()
    );
}

#[test]
fn issue_ai_02_spec_examples_rejects_an_oversized_or_unrecorded_spec() {
    let big = "x".repeat(TOKEN_LIMIT * 3 + 3);
    assert!(spec_tokens(&big) >= TOKEN_LIMIT);
    assert_eq!(
        recorded_tokens("Size record: tokens=12 estimator=bytes/3\n"),
        Some(12)
    );
    assert_eq!(recorded_tokens("no record\n"), None);
}

fn sample_index(path: &str, section: &str) -> Value {
    serde_json::json!({
        "schema": "invar-examples-index-v1", "spec": SPEC, "pending_families": [],
        "examples": [{"path": path, "intent": "a synthetic example for tests", "family": "core", "tags": [], "spec_section": section,
                      "engines": ["reference"], "expect": {"command": "run", "status": "ok", "value": 1}}]
    })
}

#[test]
fn issue_ai_02_spec_examples_rejects_bad_example_indexes() {
    let mut schema: Value = serde_json::from_str(&read(INDEX_SCHEMA)).unwrap();
    // Reduce the family enum so these synthetic indexes can focus on the other rules.
    schema["$defs"]["family"]["enum"] = serde_json::json!(["core"]);
    let files: BTreeSet<String> = ["examples/a.inv".to_owned()].into();
    let sections: BTreeSet<String> = ["3.1".to_owned()].into();
    assert!(
        validate_index(
            &sample_index("examples/a.inv", "3.1"),
            &schema,
            &files,
            &sections
        )
        .is_ok()
    );
    let missing_from_index = validate_index(
        &sample_index("examples/a.inv", "3.1"),
        &schema,
        &["examples/a.inv".into(), "examples/b.inv".into()].into(),
        &sections,
    );
    assert!(
        missing_from_index
            .unwrap_err()
            .iter()
            .any(|e| e.contains("missing from the index"))
    );
    let no_file = validate_index(
        &sample_index("examples/gone.inv", "3.1"),
        &schema,
        &files,
        &sections,
    );
    assert!(
        no_file
            .unwrap_err()
            .iter()
            .any(|e| e.contains("without a file"))
    );
    let dangling = validate_index(
        &sample_index("examples/a.inv", "9.9"),
        &schema,
        &files,
        &sections,
    );
    assert!(
        dangling
            .unwrap_err()
            .iter()
            .any(|e| e.contains("dangling spec section"))
    );
    let mut twice = sample_index("examples/a.inv", "3.1");
    let entry = twice["examples"][0].clone();
    twice["examples"].as_array_mut().unwrap().push(entry);
    assert!(
        validate_index(&twice, &schema, &files, &sections)
            .unwrap_err()
            .iter()
            .any(|e| e.contains("more than once"))
    );
    let mut invalid = sample_index("examples/a.inv", "3.1");
    invalid["examples"][0]
        .as_object_mut()
        .unwrap()
        .remove("intent");
    assert!(
        validate_index(&invalid, &schema, &files, &sections).unwrap_err()[0].starts_with("schema:")
    );
    let mut no_reference = sample_index("examples/a.inv", "3.1");
    no_reference["examples"][0]["engines"] = serde_json::json!(["mir"]);
    assert!(
        validate_index(&no_reference, &schema, &files, &sections)
            .unwrap_err()
            .iter()
            .any(|e| e.contains("runnable example without the reference engine"))
    );
}

#[test]
fn issue_ai_02_spec_examples_rejects_an_example_whose_output_differs() {
    let entry = serde_json::json!({"path": "examples/sum.inv", "engines": ["reference", "mir"],
        "expect": {"command": "run", "status": "ok", "value": 10}});
    assert_eq!(run_example(&entry).unwrap(), (2, 0));
    let wrong = serde_json::json!({"path": "examples/sum.inv", "engines": ["reference"],
        "expect": {"command": "run", "status": "ok", "value": 11}});
    assert!(run_example(&wrong).is_err());
    let undeclared = serde_json::json!({"path": "examples/sum.inv", "engines": ["system"],
        "expect": {"command": "run", "status": "ok", "value": 10}});
    assert!(run_example(&undeclared).is_err());
}

#[test]
fn issue_ai_02_spec_examples_rejects_a_docs_link_to_a_removed_file() {
    let exists = |p: &str| p == "docs/present.md" || p == "README.md";
    let ok = [("docs/a.md".to_owned(), "[x](present.md) [y](../README.md#top) [z](https://example.org/x.md)\n```\n[c](gone.md)\n```".to_owned())];
    assert!(broken_links(&ok, &exists).is_empty());
    let bad = [("docs/a.md".to_owned(), "see [gone](removed.md)".to_owned())];
    assert_eq!(
        broken_links(&bad, &exists),
        ["docs/a.md: link to missing file removed.md"]
    );
}
