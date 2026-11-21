//! Issue #263. The TextMate grammar in `editors/vscode` names exactly the words the lexer treats as
//! keywords and as reserved words, every keyword is a terminal of `spec/grammar.ebnf`, and the
//! grammar's intrinsic names are words of that spec grammar. A word added to or removed from the
//! lexer without the grammar (or the reverse) fails here.

use std::{collections::BTreeSet, fs, path::Path};

use serde_json::Value;

const GRAMMAR: &str = "editors/vscode/syntaxes/invar.tmLanguage.json";
const FIX: &str = "Edit editors/vscode/syntaxes/invar.tmLanguage.json (and spec/grammar.ebnf) to match src/front/lexer.rs.";

/// Grammar rules whose words are the lexer's `keyword()` table.
const KEYWORD_RULES: &[&str] = &[
    "keyword-control",
    "keyword-declaration",
    "keyword-modifier",
    "keyword-operator",
    "constant-language",
    "type-keyword",
];

fn read(path: &str) -> String {
    fs::read_to_string(Path::new(env!("CARGO_MANIFEST_DIR")).join(path))
        .unwrap_or_else(|error| panic!("cannot read {path}: {error}"))
}

/// The text of `signature`'s function in the lexer, up to the closing brace in column 0.
fn lexer_function(signature: &str) -> String {
    let source = read("src/front/lexer.rs");
    let start = source.find(signature).unwrap_or_else(|| {
        panic!("src/front/lexer.rs has no `{signature}`; update tests/editor_grammar.rs")
    });
    let rest = &source[start..];
    rest[..rest.find("\n}\n").expect("end of lexer function")].to_owned()
}

/// Every `"word"` literal of identifier characters in `text`.
fn quoted_words(text: &str) -> BTreeSet<String> {
    text.split('"')
        .skip(1)
        .step_by(2)
        .filter(|word| {
            !word.is_empty() && word.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'_')
        })
        .map(str::to_owned)
        .collect()
}

fn grammar() -> Value {
    serde_json::from_str(&read(GRAMMAR)).expect("the TextMate grammar is JSON")
}

/// The alternatives of a `\\b(a|b|c)\\b` regex, optionally followed by a call lookahead.
fn match_words(pattern: &str) -> BTreeSet<String> {
    let inner = pattern
        .strip_prefix("\\b(")
        .and_then(|rest| {
            rest.strip_suffix(")\\b")
                .or_else(|| rest.strip_suffix(")(?=\\s*\\()"))
        })
        .unwrap_or_else(|| panic!("grammar regex is not of the form \\b(a|b)\\b: {pattern}"));
    inner.split('|').map(str::to_owned).collect()
}

fn rule_words(grammar: &Value, rule: &str) -> BTreeSet<String> {
    match_words(
        grammar["repository"][rule]["match"]
            .as_str()
            .unwrap_or_else(|| panic!("grammar rule `{rule}` has no match")),
    )
}

fn lexer_keywords() -> BTreeSet<String> {
    // Each arm is `"word" => "word"`; both sides are the same word.
    let words = quoted_words(&lexer_function("fn keyword("));
    assert!(
        words.len() > 30,
        "found only {} lexer keywords: {words:?}",
        words.len()
    );
    words
}

fn lexer_reserved() -> BTreeSet<String> {
    let words = quoted_words(&lexer_function("fn unsupported("));
    assert!(
        words.len() >= 5,
        "found only {} reserved words: {words:?}",
        words.len()
    );
    words
}

fn whole_word(text: &str, word: &str) -> bool {
    text.match_indices(word).any(|(at, _)| {
        let before = text[..at].chars().next_back();
        let after = text[at + word.len()..].chars().next();
        let part = |c: Option<char>| c.is_some_and(|c| c.is_ascii_alphanumeric() || c == '_');
        !part(before) && !part(after)
    })
}

#[test]
fn editor_grammar_keywords_are_the_lexer_keywords() {
    let grammar = grammar();
    let mut named = BTreeSet::new();
    for rule in KEYWORD_RULES {
        for word in rule_words(&grammar, rule) {
            assert!(
                named.insert(word.clone()),
                "`{word}` is in two keyword rules of the grammar"
            );
        }
    }
    let lexer = lexer_keywords();
    let missing: Vec<_> = lexer.difference(&named).collect();
    let extra: Vec<_> = named.difference(&lexer).collect();
    assert!(
        missing.is_empty() && extra.is_empty(),
        "keyword drift: the lexer has {missing:?} that the grammar lacks; the grammar has {extra:?} that the lexer lacks. {FIX}"
    );
}

#[test]
fn editor_grammar_reserved_words_are_the_lexer_reserved_words() {
    let named = rule_words(&grammar(), "reserved-word");
    let lexer = lexer_reserved();
    let missing: Vec<_> = lexer.difference(&named).collect();
    let extra: Vec<_> = named.difference(&lexer).collect();
    assert!(
        missing.is_empty() && extra.is_empty(),
        "reserved-word drift: the lexer has {missing:?} that the grammar lacks; the grammar has {extra:?} that the lexer lacks. {FIX}"
    );
    let keywords = lexer_keywords();
    let both: Vec<_> = lexer.intersection(&keywords).collect();
    assert!(
        both.is_empty(),
        "{both:?} are both keywords and reserved words in the lexer"
    );
}

#[test]
fn editor_grammar_spec_grammar_lists_every_lexer_word() {
    let spec = read("spec/grammar.ebnf");
    let terminals = quoted_words(&spec);
    let unquoted: Vec<_> = lexer_keywords()
        .into_iter()
        .filter(|word| !terminals.contains(word))
        .collect();
    assert!(
        unquoted.is_empty(),
        "lexer keywords that are not terminals of spec/grammar.ebnf: {unquoted:?}"
    );
    let unlisted: Vec<_> = lexer_reserved()
        .into_iter()
        .filter(|word| !whole_word(&spec, word))
        .collect();
    assert!(
        unlisted.is_empty(),
        "lexer reserved words that spec/grammar.ebnf does not mention: {unlisted:?}"
    );
}

#[test]
fn editor_grammar_builtin_names_are_words_of_the_spec_grammar() {
    let grammar = grammar();
    let spec = read("spec/grammar.ebnf");
    let functions = &grammar["repository"]["functions"]["patterns"][0];
    let types = &grammar["repository"]["types"]["patterns"][0];
    assert_eq!(functions["name"], "support.function.builtin.invar");
    assert_eq!(types["name"], "support.type.builtin.invar");
    let mut words = match_words(functions["match"].as_str().unwrap());
    words.extend(match_words(types["match"].as_str().unwrap()));
    let unknown: Vec<_> = words
        .iter()
        .filter(|word| !whole_word(&spec, word))
        .collect();
    assert!(
        unknown.is_empty(),
        "grammar names that spec/grammar.ebnf never mentions: {unknown:?}"
    );
}

#[test]
fn editor_grammar_extension_manifest_packages_the_grammar() {
    let manifest: Value =
        serde_json::from_str(&read("editors/vscode/package.json")).expect("package.json is JSON");
    let contributes = &manifest["contributes"];
    assert_eq!(contributes["languages"][0]["id"], "invar");
    assert_eq!(contributes["languages"][0]["extensions"][0], ".inv");
    let packaged = &contributes["grammars"][0];
    assert_eq!(packaged["scopeName"], grammar()["scopeName"]);
    for path in [
        packaged["path"].as_str().unwrap(),
        contributes["languages"][0]["configuration"]
            .as_str()
            .unwrap(),
    ] {
        let file = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("editors/vscode")
            .join(path);
        assert!(
            file.is_file(),
            "the extension manifest names {path}, which does not exist"
        );
        serde_json::from_str::<Value>(&fs::read_to_string(&file).unwrap())
            .unwrap_or_else(|e| panic!("{path} is not JSON: {e}"));
    }
}

#[test]
fn editor_grammar_strings_end_at_line_break() {
    // The lexer rejects a literal newline in a string, so an unclosed quote must not swallow later lines.
    let grammar = grammar();
    let rules = grammar["repository"]["strings"]["patterns"]
        .as_array()
        .expect("string rules");
    assert_eq!(rules.len(), 2);
    for rule in rules {
        assert_eq!(
            rule["end"], "\"|$",
            "{} must end at a quote or a line break",
            rule["name"]
        );
    }
}
