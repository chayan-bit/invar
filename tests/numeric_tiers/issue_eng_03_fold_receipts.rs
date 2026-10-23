//! ENG-03 step 7 (D12): the newest documents that carry `numeric_contracts`
//! (explain-v5 and invar-diagnostics-v3) record each declared fold tier a
//! lowered fold used and the algorithm that ran. Older versions never carry
//! it, and a program without a lowered fold is unchanged.

use invar::{check, front, math, mir};
use serde_json::{Value, json};

/// A function `f{n}` under `attribute` with one fold whose update is `update`.
fn folding(n: usize, attribute: &str, update: &str) -> String {
    format!(
        "{attribute} fn f{n}(xs: read Array<f32>) -> f32 {{ var acc: f32 = 0.0; \
         for i in 0u32..len(xs) {{ acc = {update}; }} return acc; }}"
    )
}

fn program(functions: &[String]) -> String {
    // Each function is `fn f<n>(`; call it by that name.
    let calls: String = functions
        .iter()
        .map(|f| {
            let name = &f[f.find("fn ").unwrap() + 3..];
            format!("s = s + {}(xs); ", &name[..name.find('(').unwrap()])
        })
        .collect();
    format!(
        "{} fn main() -> i32 {{ let xs: Array<f32> = [1.0, 2.0, 3.0]; var s: f32 = 0.0; {calls}\
         if s == {}.0 {{ return 42; }} return 9; }}",
        functions.concat(),
        6 * functions.len()
    )
}

fn receipt(source: &str) -> Option<Value> {
    let checked = check::check(front::parse(source.as_bytes()).unwrap(), true)
        .unwrap_or_else(|error| panic!("{error:?}\n{source}"));
    math::fold_receipt(&mir::lower(&checked).unwrap())
}

#[test]
fn issue_eng_03_fold_receipt_names_each_declared_tier_and_its_algorithm() {
    let tree = folding(0, "#[numeric(fold = \"tree\")]", "acc + xs[i]");
    let exact = folding(1, "#[numeric(fold = \"exact\")]", "acc + xs[i]");
    let relaxed = folding(2, "#[numeric(fold = \"relaxed\")]", "acc + xs[i]");
    let reassoc = folding(3, "", "reassoc_add(acc, xs[i])");
    let cases = [
        (vec![tree.clone()], Some(json!({"tree": "tree"}))),
        (vec![exact.clone()], Some(json!({"exact": "exact"}))),
        (vec![relaxed.clone()], Some(json!({"relaxed": "tree"}))),
        // A Strict function's reassoc_add fold records as relaxed.
        (vec![reassoc.clone()], Some(json!({"relaxed": "tree"}))),
        (
            vec![tree, exact, relaxed, reassoc],
            Some(json!({"tree": "tree", "exact": "exact", "relaxed": "tree"})),
        ),
        // No lowered fold: a strict fold, and a tier with no fold.
        (vec![folding(0, "", "acc + xs[i]")], None),
        (
            vec!["#[numeric(fold = \"tree\")] fn f0(xs: read Array<f32>) -> f32 { return xs[0u32]; }".into()],
            None,
        ),
    ];
    for (functions, expected) in &cases {
        let source = program(functions);
        assert_eq!(receipt(&source), *expected, "{source}");
    }
    assert_eq!(cases.len(), 7);
}

fn schema(name: &str) -> jsonschema::Validator {
    let path = format!("{}/schema/{name}", env!("CARGO_MANIFEST_DIR"));
    let schema: Value = serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
    jsonschema::validator_for(&schema).unwrap()
}

fn cli(dir: &std::path::Path, args: &[&str]) -> (i32, Value) {
    let out = std::process::Command::new(env!("CARGO_BIN_EXE_invar"))
        .current_dir(dir)
        .args(args)
        .output()
        .unwrap();
    let document = serde_json::from_slice(&out.stdout).unwrap_or_else(|error| {
        panic!(
            "{error}: {}\n{}",
            String::from_utf8_lossy(&out.stdout),
            String::from_utf8_lossy(&out.stderr)
        )
    });
    (out.status.code().unwrap(), document)
}

#[test]
fn issue_eng_03_only_placement_documents_carry_the_fold_receipt() {
    let dir = std::env::temp_dir().join(format!("invar-fold-receipts-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let host = format!("{}-{}", std::env::consts::OS, std::env::consts::ARCH);
    let folds = program(&[
        folding(0, "#[numeric(fold = \"exact\")]", "acc + xs[i]"),
        folding(1, "#[numeric(fold = \"relaxed\")]", "acc + xs[i]"),
    ]);
    let plain = program(&[folding(0, "", "acc + xs[i]")]);
    let (explain_v5, run_v3) = (
        schema("explain-v5.schema.json"),
        schema("invar-diagnostics-v3.schema.json"),
    );
    for (source, fold) in [
        (&folds, json!({"exact": "exact", "relaxed": "tree"})),
        (&plain, Value::Null),
    ] {
        std::fs::write(dir.join("folds.inv"), source).unwrap();
        for (command, validator) in [("run", &run_v3), ("explain", &explain_v5)] {
            let placed = [
                "--engine",
                "auto",
                "--format",
                "json",
                "--host-class",
                &host,
            ];
            let (status, document) = cli(&dir, &[&[command, "folds.inv"][..], &placed].concat());
            assert_eq!(status, 0, "{document}");
            assert_eq!(document["auto_path"], "placement", "{document}");
            assert_eq!(document["numeric_contracts"]["fold"], fold, "{document}");
            if fold.is_null() {
                assert!(document.get("numeric_contracts").is_none(), "{document}");
            }
            let errors: Vec<String> = validator
                .iter_errors(&document)
                .map(|e| e.to_string())
                .collect();
            assert!(errors.is_empty(), "{command}: {errors:?}\n{document}");
            // The receipt embedded in v5 carries no numeric_contracts (unchanged).
            assert!(
                document["execution"].get("numeric_contracts").is_none(),
                "{document}"
            );
            // The established documents (explain v1, diagnostics v1) never
            // carry it: their frozen schemas are unchanged (D12).
            let established: &[&str] = if command == "run" {
                &[
                    "run",
                    "folds.inv",
                    "--engine",
                    "reference",
                    "--format",
                    "json",
                ]
            } else {
                &["explain", "folds.inv", "--format", "json"]
            };
            let (status, established) = cli(&dir, established);
            assert_eq!(status, 0, "{established}");
            assert!(
                established.get("numeric_contracts").is_none(),
                "{established}"
            );
        }
    }
    std::fs::remove_dir_all(dir).unwrap();
}
