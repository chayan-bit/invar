//! DX-25 (#86): the spec chapters, their token budgets and topic assembly.
//!
//! The chapters in `docs/spec/chapters.json` regenerate `docs/spec/invar-v0.1.md` byte for
//! byte; every recorded token count is fresh under both estimators (bytes/3 and cl100k BPE);
//! the core, every chapter and the whole spec fit `docs/spec/budget.json`; `invar spec`
//! assembles deterministically and refuses an over-budget request with a typed error.
//!
//! After editing a chapter, `INVAR_SPEC_BLESS=1 cargo test --test spec_budget` rewrites the
//! generated file, the recorded counts and the size record, then rerun without it.
#![forbid(unsafe_code)]

use std::{path::PathBuf, process::Command, sync::LazyLock};

use invar::spec_docs::{SIZE_RECORD_PREFIX, Spec, SpecError, Tokens, bytes3_tokens, counted_text};
use serde_json::Value;

const MANIFEST: &str = "docs/spec/chapters.json";
const BUDGETS: &str = "docs/spec/budget.json";

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn read(relative: &str) -> Option<String> {
    std::fs::read_to_string(root().join(relative)).ok()
}

/// The second estimator: the cl100k_base vocabulary, embedded in the pinned `tiktoken-rs`
/// crate (its Cargo.lock checksum pins the vocabulary), so the count is offline.
fn bpe(text: &str) -> usize {
    static BPE: LazyLock<tiktoken_rs::CoreBPE> =
        LazyLock::new(|| tiktoken_rs::cl100k_base().unwrap());
    BPE.encode_ordinary(text).len()
}

fn measure(text: &str) -> Tokens {
    Tokens {
        bytes3: bytes3_tokens(text),
        bpe: bpe(&counted_text(text)),
    }
}

/// The spec as checked in, read from the working tree.
fn on_disk() -> Spec {
    Spec::parse(&read(MANIFEST).unwrap(), &read(BUDGETS).unwrap(), &read).unwrap()
}

fn invar(args: &[&str]) -> (i32, String, String) {
    let output = Command::new(env!("CARGO_BIN_EXE_invar"))
        .args(args)
        .output()
        .unwrap();
    let text = |bytes: &[u8]| String::from_utf8(bytes.to_vec()).unwrap();
    (
        output.status.code().unwrap(),
        text(&output.stdout),
        text(&output.stderr),
    )
}

/// Rewrite the size record of the core chapter for a generated file of `tokens`.
fn with_size_record(text: &str, tokens: usize) -> String {
    text.lines()
        .map(|line| match line.strip_prefix(SIZE_RECORD_PREFIX) {
            Some(rest) => {
                let after = rest.split_once(" estimator=").map_or("", |(_, tail)| tail);
                format!("{SIZE_RECORD_PREFIX} tokens={tokens} estimator={after}\n")
            }
            None => format!("{line}\n"),
        })
        .collect()
}

fn bless() {
    let spec = on_disk();
    let generated = spec.generated_text();
    let core = spec
        .chapters
        .iter()
        .find(|c| c.core && c.text.contains(SIZE_RECORD_PREFIX))
        .unwrap();
    let write = |path: &str, text: &str| std::fs::write(root().join(path), text).unwrap();
    write(
        &core.path,
        &with_size_record(&core.text, bytes3_tokens(&generated)),
    );
    let spec = on_disk();
    write(&spec.generated, &spec.generated_text());
    write(MANIFEST, &spec.manifest_with(&measure));
}

/// `None` when `checked_in` is exactly what the chapters generate.
fn generated_mismatch(spec: &Spec, checked_in: &str) -> Option<String> {
    let expected = spec.generated_text();
    if checked_in == expected {
        return None;
    }
    let same = expected
        .lines()
        .zip(checked_in.lines())
        .take_while(|(a, b)| a == b)
        .count();
    let line = same + 1;
    Some(format!(
        "{} differs from its chapters in {MANIFEST} (first at line {line}): edit the chapter, then run INVAR_SPEC_BLESS=1 cargo test --test spec_budget",
        spec.generated
    ))
}

#[test]
fn spec_budget_chapters_regenerate_the_single_file_spec_byte_for_byte() {
    if std::env::var_os("INVAR_SPEC_BLESS").is_some() {
        bless();
    }
    let spec = on_disk();
    let checked_in = read(&spec.generated).unwrap();
    if let Some(message) = generated_mismatch(&spec, &checked_in) {
        panic!("{message}");
    }
    assert!(
        read(MANIFEST).unwrap() == spec.manifest_with(&measure),
        "{MANIFEST} is stale or not in its canonical layout: run INVAR_SPEC_BLESS=1 cargo test --test spec_budget"
    );
    let limit = format!("limit={}", spec.budgets["spec"]);
    let record = checked_in
        .lines()
        .find(|l| l.starts_with(SIZE_RECORD_PREFIX))
        .unwrap();
    assert!(
        record.ends_with(&limit),
        "{record}: the size record states the whole-spec budget, {limit}"
    );
    let parts = spec.chapters.iter().filter(|c| c.part).count();
    eprintln!(
        "{}: {} bytes from {parts} chapters",
        spec.generated,
        checked_in.len()
    );
}

#[test]
fn spec_budget_rejects_a_hand_edit_of_the_generated_file() {
    let spec = on_disk();
    let edited = spec
        .generated_text()
        .replacen("[N3.4.1] ", "[N3.4.1] Edited. ", 1);
    let message = generated_mismatch(&spec, &edited).expect("a hand edit is refused");
    assert!(
        message.starts_with("docs/spec/invar-v0.1.md differs from its chapters"),
        "{message}"
    );
    let line = spec
        .generated_text()
        .lines()
        .position(|l| l.starts_with("[N3.4.1] "))
        .unwrap()
        + 1;
    assert!(
        message.contains(&format!("(first at line {line})")),
        "{message}"
    );
    assert_eq!(generated_mismatch(&spec, &spec.generated_text()), None);
}

#[test]
fn spec_budget_every_chapter_core_and_the_whole_spec_fit_their_budgets() {
    let spec = on_disk();
    let errors = spec.validate(&bpe);
    assert!(errors.is_empty(), "{}", errors.join("\n"));
    let (mut bytes3, mut tokens) = (0, 0);
    for chapter in &spec.chapters {
        let measured = measure(&chapter.text);
        eprintln!(
            "{:<22} bytes/3 {:>5}  cl100k {:>5}  core {}",
            chapter.id, measured.bytes3, measured.bpe, chapter.core
        );
        bytes3 += measured.bytes3;
        tokens += measured.bpe;
    }
    let ratio = bytes3 as f64 / tokens as f64;
    eprintln!("whole spec: bytes/3 {bytes3}, cl100k {tokens}, ratio {ratio:.3}");
    assert!(
        (0.75..=1.25).contains(&ratio),
        "the estimators disagree by more than 25 percent: {ratio:.3}"
    );
}

#[test]
fn spec_budget_embedded_texts_cover_the_manifest_exactly() {
    let disk = on_disk();
    let mut paths: Vec<&str> = disk.chapters.iter().map(|c| c.path.as_str()).collect();
    let mut embedded = Spec::embedded_paths();
    paths.sort_unstable();
    embedded.sort_unstable();
    assert_eq!(
        paths, embedded,
        "src/spec_docs.rs TEXTS must list every chapter in {MANIFEST}"
    );
    assert_eq!(
        Spec::embedded(),
        disk,
        "rebuild: the embedded spec differs from the working tree"
    );
}

fn broken(edit: impl FnOnce(&mut Spec)) -> String {
    let mut spec = on_disk();
    edit(&mut spec);
    spec.validate(&bpe).join("\n")
}

fn chapter<'a>(spec: &'a mut Spec, id: &str) -> &'a mut invar::spec_docs::Chapter {
    spec.chapters.iter_mut().find(|c| c.id == id).unwrap()
}

#[test]
fn spec_budget_rejects_each_broken_rule_with_a_named_message() {
    let over = broken(|spec| {
        let views = chapter(spec, "views");
        views
            .text
            .push_str(&format!("### 5.9 Padding\n{}\n", "x".repeat(12_000)));
        views.tokens = measure(&views.text);
    });
    assert!(over.contains("chapter over budget: views is "), "{over}");
    assert!(over.contains("largest sections: 5.9 Padding "), "{over}");
    let core = broken(|spec| {
        spec.budgets["core"] = Value::from(100);
    });
    assert!(
        core.contains("core over budget: core is ")
            && core.contains("over 100 in docs/spec/budget.json"),
        "{core}"
    );
    let whole = broken(|spec| spec.budgets["spec"] = Value::from(1000));
    assert!(
        whole.contains("whole spec over budget: spec is "),
        "{whole}"
    );
    let stale = broken(|spec| chapter(spec, "views").tokens.bpe += 1);
    assert!(
        stale.contains("stale tokens field: chapter views records "),
        "{stale}"
    );
    let cycle = broken(|spec| {
        chapter(spec, "ownership")
            .requires
            .push("generics-closures".to_owned())
    });
    assert!(
        cycle.contains("requires cycle: ownership -> generics-closures -> ownership"),
        "{cycle}"
    );
    let unknown = broken(|spec| chapter(spec, "views").requires.push("nowhere".to_owned()));
    assert!(
        unknown.contains("chapter views requires unknown chapter nowhere"),
        "{unknown}"
    );
    let topicless = broken(|spec| chapter(spec, "views").topics.clear());
    assert!(
        topicless.contains("chapter views: every chapter needs non-empty topics"),
        "{topicless}"
    );
    let duplicate = broken(|spec| {
        let views = chapter(spec, "views");
        views.text.push_str("[N3.4.1] A copy. `src/eval.rs`\n");
        views.tokens = measure(&views.text);
    });
    assert!(
        duplicate.contains("duplicate normative id N3.4.1 in chapters core and views"),
        "{duplicate}"
    );
    let topic = on_disk().assemble(&["no-such-topic".to_owned()], &[], 60_000);
    assert_eq!(
        topic,
        Err(SpecError::UnknownTopic("no-such-topic".to_owned()))
    );
    assert_eq!(
        topic.unwrap_err().to_string(),
        "topic no-such-topic has no chapter"
    );
}

fn expected(ids: &[&str]) -> String {
    let spec = on_disk();
    spec.chapters
        .iter()
        .filter(|c| ids.contains(&c.id.as_str()))
        .map(|c| c.text.as_str())
        .collect()
}

#[test]
fn spec_budget_cli_assembles_topics_deterministically() {
    for (args, ids) in [
        (
            &["spec", "--topic", "ownership"][..],
            &["core", "ownership"][..],
        ),
        (
            &["spec", "--topic", "views", "--budget", "6000"],
            &["core", "views"],
        ),
        (
            &["spec", "--topic", "closures"],
            &["core", "ownership", "generics-closures"],
        ),
        (
            &["spec", "--chapter", "axis-n", "--topic", "spatial"],
            &["core", "v02-numerics", "axes", "axis-n"],
        ),
    ] {
        let first = invar(args);
        assert_eq!(first, invar(args), "{args:?} is not deterministic");
        assert_eq!((first.0, first.2.as_str()), (0, ""), "{args:?}");
        assert!(first.1 == expected(ids), "{args:?} did not print {ids:?}");
    }
    let (status, stdout, _) = invar(&["spec", "--topic", "views", "--format", "json"]);
    let document: Value = serde_json::from_str(&stdout).unwrap();
    assert_eq!(
        (status, document["schema"].as_str()),
        (0, Some("invar-spec-assembly-v1"))
    );
    let ids: Vec<&str> = document["chapters"]
        .as_array()
        .unwrap()
        .iter()
        .map(|c| c["id"].as_str().unwrap())
        .collect();
    assert_eq!(ids, ["core", "views"]);
    assert_eq!(document["budget"], 20_000);
    assert_eq!(
        document["text"].as_str(),
        Some(expected(&["core", "views"]).as_str())
    );
}

#[test]
fn spec_budget_cli_refuses_an_over_budget_or_unknown_request() {
    let (status, stdout, stderr) = invar(&["spec", "--topic", "engines", "--budget", "6000"]);
    assert_eq!((status, stdout.as_str()), (2, ""));
    assert!(
        stderr.starts_with("error[SPEC_BUDGET_EXCEEDED]: the assembly is "),
        "{stderr}"
    );
    assert!(
        stderr.contains("over the budget 6000; largest chapters: core "),
        "{stderr}"
    );
    let (status, stdout, _) = invar(&[
        "spec", "--topic", "engines", "--budget", "6000", "--format", "json",
    ]);
    let document: Value = serde_json::from_str(&stdout).unwrap();
    assert_eq!(
        (status, document["status"].as_str()),
        (2, Some("tool_error"))
    );
    assert_eq!(document["diagnostics"][0]["code"], "SPEC_BUDGET_EXCEEDED");
    let (status, _, stderr) = invar(&["spec", "--topic", "nowhere"]);
    assert_eq!(
        (status, stderr.as_str()),
        (
            2,
            "error[SPEC_UNKNOWN_TOPIC]: topic nowhere has no chapter\n"
        )
    );
    let (status, _, stderr) = invar(&["spec", "--chapter", "nowhere"]);
    assert_eq!(
        (status, stderr.as_str()),
        (2, "error[SPEC_UNKNOWN_CHAPTER]: no chapter nowhere\n")
    );
    for args in [
        &["spec", "--budget"][..],
        &["spec", "--budget", "x"],
        &["spec", "--format", "html"],
        &["spec", "--bogus", "1"],
    ] {
        assert_eq!(invar(args).0, 2, "{args:?}");
    }
}
