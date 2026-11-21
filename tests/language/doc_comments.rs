//! Issue #257: `///` and `//!` doc comments are trivia attached to the item
//! they document and never change checking, MIR, plans or receipts.
use crate::surface_sugar::first_code;
use invar::{check, engine, front, mir};
use std::process::Command;

const DOCUMENTED: &str = "//! Sums of small arrays.
//!
//! Every function here is total.

/// A point on the grid.
struct Point { x: u32, y: u32 }

/// The sum of `xs`.
///
/// Wraps on overflow.
fn sum_all(xs: read Array<u32>) -> u32 {
    var sum: u32 = 0u32;
    for i in 0u32 .. len(xs) {
        sum = wrapping_add(sum, xs[i]);
    }
    return sum;
}

module geometry {
    //! Grid helpers.

    /// Manhattan length.
    pub fn length(x: u32, y: u32) -> u32 { return x + y; }
}

//// Four slashes are an ordinary comment, anywhere.
fn main() -> i32 {
    //// Still ordinary.
    let xs: Array<u32> = [1u32, 2u32, 3u32];
    if sum_all(xs) == 6u32 { return 1; }
    return 0;
}
";

/// The same program with every doc marker turned into an ordinary comment of
/// the same length, so every span is unchanged.
fn undocumented() -> String {
    DOCUMENTED
        .replace("////", "\u{1}")
        .replace("///", "// ")
        .replace("//!", "// ")
        .replace('\u{1}', "////")
}

#[test]
fn doc_comments_attach_to_the_item_they_document() {
    let program = front::parse(DOCUMENTED.as_bytes()).unwrap();
    let at = |text: &str| DOCUMENTED.find(text).unwrap();
    let file = front::Span {
        start: 0,
        end: DOCUMENTED.len(),
    };
    let module_end =
        at("pub fn length") + DOCUMENTED[at("pub fn length")..].find("}\n}").unwrap() + 3;
    let module = front::Span {
        start: at("module geometry"),
        end: module_end,
    };
    let item = |start: &str, end: &str| front::Span {
        start: at(start),
        end: at(end) + end.len(),
    };
    let point = item("struct Point", "y: u32 }");
    let total = item("fn sum_all", "return sum;\n}");
    let length = item("pub fn length", "return x + y; }");
    let attached: Vec<(bool, &str, front::Span)> = program
        .docs
        .iter()
        .map(|doc| (doc.inner, doc.text.as_str(), doc.item))
        .collect();
    assert_eq!(
        attached,
        [
            (true, "Sums of small arrays.", file),
            (true, "", file),
            (true, "Every function here is total.", file),
            (false, "A point on the grid.", point),
            (false, "The sum of `xs`.", total),
            (false, "", total),
            (false, "Wraps on overflow.", total),
            (true, "Grid helpers.", module),
            (false, "Manhattan length.", length),
        ]
    );
    for doc in &program.docs {
        assert!(DOCUMENTED[doc.span.start..doc.span.end].starts_with("//"));
    }
    assert!(
        front::parse(undocumented().as_bytes())
            .unwrap()
            .docs
            .is_empty()
    );
}

#[test]
fn doc_comments_change_no_mir_plan_or_receipt() {
    let lowered = |source: &str| {
        let checked = check::check(front::parse(source.as_bytes()).unwrap(), true).unwrap();
        let program = mir::lower(&checked).unwrap();
        let receipt = engine::run_mir(
            source.as_bytes(),
            &program,
            engine::Engine::CpuParallel,
            1_000_000,
            4,
            None,
        )
        .unwrap();
        // Every function with its spans; locals sorted out of hash-map order.
        let functions: Vec<String> = program
            .functions
            .iter()
            .map(|f| {
                let mut locals = f.locals.iter().collect::<Vec<_>>();
                locals.sort_by_key(|(name, _)| *name);
                let shape = (&f.name, &f.params, &f.result, f.span, f.entry, &f.blocks);
                format!("{shape:?} {locals:?} {:?}", f.declaration_order)
            })
            .collect();
        (functions, receipt)
    };
    let (documented, plain) = (lowered(DOCUMENTED), lowered(&undocumented()));
    assert_eq!(documented.0, plain.0, "MIR, spans included");
    assert_eq!(documented.1, plain.1, "cpu-parallel receipt");
    // The plan, byte for byte except the identity hash of the source bytes,
    // which by contract covers the comments too.
    let explain = |source: &str| {
        let path = std::env::temp_dir().join(format!("invar-docs-{}.inv", std::process::id()));
        std::fs::write(&path, source).unwrap();
        let output = Command::new(env!("CARGO_BIN_EXE_invar"))
            .args(["explain", path.to_str().unwrap(), "--format", "json"])
            .output()
            .unwrap();
        std::fs::remove_file(path).unwrap();
        assert!(output.status.success());
        let text = String::from_utf8(output.stdout).unwrap();
        let identity = source_identities(&text);
        assert!(!identity.is_empty(), "{text}");
        identity
            .iter()
            .fold(text.clone(), |text, hash| text.replace(hash.as_str(), "_"))
    };
    assert_eq!(
        explain(DOCUMENTED),
        explain(&undocumented()),
        "explain plan"
    );
}

/// Every `"source_identity":"..."` value in an explain document.
fn source_identities(text: &str) -> Vec<String> {
    let key = "\"source_identity\":\"";
    text.match_indices(key)
        .map(|(at, _)| {
            let value = &text[at + key.len()..];
            value[..value.find('"').unwrap()].to_owned()
        })
        .collect()
}

#[test]
fn a_doc_comment_that_documents_nothing_is_refused() {
    for source in [
        "fn main() -> i32 {\n    /// Not an item.\n    return 0;\n}\n",
        "fn main() -> i32 { return 0; }\n/// Nothing follows.\n",
        "fn main() -> i32 { return 0; }\n//! Too late for the file.\n",
        "module m {\n    fn f() -> i32 { return 0; }\n    //! Too late for the module.\n}\nfn main() -> i32 { return 0; }\n",
        "module m {\n    fn f() -> i32 { return 0; }\n    /// Nothing follows in the module.\n}\nfn main() -> i32 { return 0; }\n",
        "fn main() -> i32 {\n    let x: i32 = 1; //! Not a module.\n    return x;\n}\n",
        "fn main() -> i32 /// Inside the signature.\n{ return 0; }\n",
    ] {
        assert_eq!(
            first_code(source),
            "PARSE_DOC_COMMENT_MISPLACED",
            "{source}"
        );
        let start = source.find("///").or_else(|| source.find("//!")).unwrap();
        let error = front::parse(source.as_bytes()).unwrap_err();
        assert_eq!(error.span.map(|span| span.start), Some(start), "{source}");
    }
    for source in [
        "//! Only a file doc.\n",
        "/// Documented.\n// An ordinary comment between.\n\n/* and a block */ fn main() -> i32 { return 0; }\n",
    ] {
        let parsed = front::parse(source.as_bytes());
        assert!(parsed.is_ok(), "{source}: {parsed:?}");
    }
    let text = "fn main() -> i32 { let s: String = \"/// not a comment\"; return 0; }\n";
    assert!(front::parse(text.as_bytes()).unwrap().docs.is_empty());
}
