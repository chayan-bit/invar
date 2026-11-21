//! #198: the `#[region(...)]` declaration of `docs/spec/invar-v0.2-axes.md`
//! section 4. Its accept and refuse sets, its caller, callee and closure
//! rules (as the fold tier tests check them), the unchanged admitted plan of a
//! declared region, and the CLI caps and region report.

use invar::axes::{Axes, C_AXIS, N_AXIS};
use invar::numeric_contract::FoldTier;
use serde_json::Value;
use std::process::Command;

/// A fold whose Tree (16777218), Exact (16777224) and Strict (16777216)
/// results differ, so `main` returns 2, 3 or 1 by which one ran.
fn program(attribute: &str) -> String {
    format!(
        "{attribute} fn fold_sum(xs: read Array<f32>) -> f32 {{ var acc: f32 = 0.0; \
         for i in 0u32..len(xs) {{ acc = acc + xs[i]; }} return acc; }} \
         fn main() -> i32 {{ let xs: Array<f32> = [16777216.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]; \
         let r = fold_sum(xs); if r == 16777218.0 {{ return 2; }} if r == 16777224.0 {{ return 3; }} \
         if r == 16777216.0 {{ return 1; }} return 0; }}"
    )
}

fn code(source: &str) -> &'static str {
    match invar::front::parse(source.as_bytes()) {
        Err(error) => error.code,
        Ok(parsed) => invar::check::check(parsed, true).map_or_else(|error| error.code, |_| "OK"),
    }
}

fn scratch(name: &str, source: &str) -> std::path::PathBuf {
    let path = std::env::temp_dir().join(format!("invar-198-{}-{name}.inv", std::process::id()));
    std::fs::write(&path, source).unwrap();
    path
}

fn cli(args: &[&str]) -> (i32, String) {
    let output = Command::new(env!("CARGO_BIN_EXE_invar"))
        .args(args)
        .output()
        .unwrap();
    (
        output.status.code().unwrap(),
        String::from_utf8(output.stdout).unwrap(),
    )
}

fn run(source: &str, engine: &[&str]) -> i64 {
    let path = scratch("run", source);
    let mut args = vec!["run", path.to_str().unwrap()];
    args.extend(engine);
    args.extend(["--format", "json"]);
    let (_, stdout) = cli(&args);
    let document: Value = serde_json::from_str(&stdout).unwrap();
    document["result"]["value"]
        .as_i64()
        .unwrap_or_else(|| panic!("{document}"))
}

#[test]
fn region_attribute_accepts_honoured_levels_and_refuses_the_rest() {
    let mut refused = 0;
    for (attribute, expected) in [
        ("", "OK"),
        ("#[region(c = \"c0\")]", "OK"),
        (
            "#[region(c = \"c0\", contract = \"strict-f32-v1\", s = \"s0\", n = \"n0\", t = \"t0\")]",
            "OK",
        ),
        ("#[region(c = \"c1\", contract = \"f32-tree-v2\")]", "OK"),
        ("#[region(contract = \"f32-exact-v2\")]", "OK"),
        ("#[region(c = \"c2\", contract = \"f32-relaxed-v2\")]", "OK"),
        ("#[region(c = \"c2\", contract = \"tc-f16-f32-v1\")]", "OK"),
        ("#[region(n = \"n1\")]", "OK"),
        ("#[region(n = \"n2\", s = \"s0\")]", "OK"),
        ("#[numeric(fold = \"tree\")] #[region(n = \"n0\")]", "OK"),
        ("#[region(t = \"t0\")] #[numeric(fold = \"exact\")]", "OK"),
        // Unknown axis key, level or contract.
        ("#[region(d = \"d1\")]", "CONTRACT_UNKNOWN"),
        ("#[region(c = \"c3\")]", "CONTRACT_UNKNOWN"),
        ("#[region(c = \"C1\")]", "CONTRACT_UNKNOWN"),
        ("#[region(n = \"n5\")]", "CONTRACT_UNKNOWN"),
        (
            "#[region(contract = \"tc-bf16-f32-v1\")]",
            "CONTRACT_UNKNOWN",
        ),
        ("#[region(c = \"c0\", c = \"c0\")]", "CONTRACT_UNKNOWN"),
        // Specified levels this engine does not implement.
        ("#[region(c = \"c5\")]", "CONTRACT_UNAVAILABLE"),
        ("#[region(s = \"s1\")]", "CONTRACT_UNAVAILABLE"),
        ("#[region(s = \"s2\")]", "CONTRACT_UNAVAILABLE"),
        ("#[region(n = \"n3\")]", "CONTRACT_UNAVAILABLE"),
        ("#[region(n = \"n4\")]", "CONTRACT_UNAVAILABLE"),
        ("#[region(t = \"t1\")]", "CONTRACT_UNAVAILABLE"),
        ("#[region(t = \"t3\")]", "CONTRACT_UNAVAILABLE"),
        // Level and contract disagree, or the contract is named twice.
        ("#[region(c = \"c1\")]", "CONTRACT_MISMATCH"),
        ("#[region(c = \"c2\")]", "CONTRACT_MISMATCH"),
        (
            "#[region(c = \"c0\", contract = \"f32-tree-v2\")]",
            "CONTRACT_MISMATCH",
        ),
        (
            "#[region(c = \"c1\", contract = \"tc-f16-f32-v1\")]",
            "CONTRACT_MISMATCH",
        ),
        (
            "#[numeric(fold = \"tree\")] #[region(contract = \"f32-tree-v2\")]",
            "CONTRACT_MISMATCH",
        ),
        (
            "#[region(c = \"c0\")] #[numeric(fold = \"strict\")]",
            "CONTRACT_MISMATCH",
        ),
        // Grammar and position.
        ("#[region(c = c1)]", "PARSE_EXPECTED_TOKEN"),
        (
            "#[region(n = \"n0\")] #[region(n = \"n0\")]",
            "UNSUPPORTED_FEATURE",
        ),
    ] {
        assert_eq!(code(&program(attribute)), expected, "{attribute}");
        refused += usize::from(expected != "OK");
    }
    let record = "#[region(c = \"c0\")] struct S { x: f32 } fn main() -> i32 { return 0; }";
    assert_eq!(code(record), "UNSUPPORTED_FEATURE");
    let module =
        "module m { #[region(contract = \"f32-tree-v2\")] pub fn one() -> f32 { return 1.0; } }";
    assert_eq!(
        invar::front::parse(module.as_bytes()).unwrap().functions[0].fold,
        FoldTier::Tree
    );
    assert_eq!(refused, 21);
}

/// The declared contract selects the rule set, so every engine returns its result.
#[test]
fn a_declared_contract_runs_its_reference_on_every_engine() {
    for (attribute, value) in [
        ("", 1),
        ("#[region(c = \"c0\", n = \"n2\")]", 1),
        ("#[region(c = \"c1\", contract = \"f32-tree-v2\")]", 2),
        ("#[region(contract = \"f32-exact-v2\")]", 3),
        ("#[region(contract = \"f32-relaxed-v2\")]", 2),
    ] {
        let source = program(attribute);
        assert_eq!(
            run(&source, &["--engine", "reference"]),
            value,
            "{attribute}"
        );
        let parallel = ["--engine", "cpu-parallel", "--workers", "4"];
        assert_eq!(run(&source, &parallel), value, "{attribute}");
    }
}

#[test]
fn a_caller_region_never_reaches_its_callee_or_closure() {
    let source = "fn callee(x: f32) -> f32 { var acc: f32 = 0.0; acc = acc + x; return acc; } \
        #[region(c = \"c2\", contract = \"f32-relaxed-v2\", n = \"n1\")] fn main() -> i32 { \
        let k: i32 = 1; \
        let c = closure [copy k: i32] (y: f32) -> f32 { return y + f32_from_i32(k); }; \
        let y = callee(c(0.0)); var acc: f32 = 0.0; for j in 0u32..1u32 { acc = acc + y; } \
        if acc == 1.0 { return 7; } return 0; }";
    let checked =
        invar::check::check(invar::front::parse(source.as_bytes()).unwrap(), true).unwrap();
    let functions: std::collections::BTreeMap<_, _> = checked
        .program
        .functions
        .iter()
        .map(|f| (f.name.clone(), (f.fold, f.axes)))
        .collect();
    let (fold, axes) = functions["main"];
    assert_eq!(
        (fold, axes.level(&C_AXIS), axes.level(&N_AXIS)),
        (FoldTier::Relaxed, "c2", "n1")
    );
    assert_eq!(functions["callee"], (FoldTier::Strict, Axes::default()));
    let lifted: Vec<_> = functions
        .iter()
        .filter(|(name, _)| !matches!(name.as_str(), "main" | "callee"))
        .collect();
    assert!(!lifted.is_empty(), "{functions:?}");
    for (_, (fold, axes)) in lifted {
        assert_eq!(
            (*fold, axes.level(&C_AXIS), axes.level(&N_AXIS)),
            (FoldTier::Strict, "c0", "n0")
        );
    }
    let lowered = invar::mir::lower(&checked).unwrap();
    assert_eq!(lowered.region_axes.keys().collect::<Vec<_>>(), ["main"]);
}

/// A region at the default levels, written out or left implicit, admits the
/// same plan: the family recognisers reparse the source bytes (N11.4.3), and
/// the declaration changes neither the region's bytes nor its plan.
#[test]
fn a_declared_region_admits_the_same_plan_as_the_implicit_one() {
    let declared = "#[region(c = \"c0\", s = \"s0\", n = \"n1\", t = \"t0\")]\n\
        fn bump(xs: read Array<u32>) -> Array<u32> { var out: Array<u32> = filled_u32(len(xs), 0u32); \
        for i in 0u32..len(xs) { out[i] = wrapping_add(xs[i], 1u32); } return out; }\n\
        fn main() -> i32 { let xs: Array<u32> = [1u32, 2u32, 3u32]; let ys = bump(xs); \
        if ys[2u32] == 4u32 { return 4; } return 0; }";
    let attribute_end = declared.find('\n').unwrap();
    let implicit = format!(
        "{}{}",
        " ".repeat(attribute_end),
        &declared[attribute_end..]
    );
    let explain = |source: &str| {
        let path = scratch("plan", source);
        let (status, stdout) = cli(&["explain", path.to_str().unwrap(), "--format", "json"]);
        assert_eq!(status, 0, "{stdout}");
        serde_json::from_str::<Value>(&stdout).unwrap()["regions"].clone()
    };
    let regions = explain(declared);
    assert_eq!(regions[0]["status"], "admitted", "{regions}");
    assert!(
        regions
            .as_array()
            .is_some_and(|regions| !regions.is_empty()),
        "{regions}"
    );
    assert_eq!(regions, explain(&implicit));
    assert_eq!(
        run(declared, &["--engine", "cpu-parallel", "--workers", "2"]),
        4
    );
}

#[test]
fn caps_only_narrow_and_the_region_report_lists_every_function() {
    let path = scratch(
        "caps",
        &program("#[region(contract = \"f32-relaxed-v2\", n = \"n2\")]"),
    );
    let file = path.to_str().unwrap();
    let run = ["run", file, "--engine", "reference", "--format", "json"];
    for (caps, status, code) in [
        (&[][..], 0, None),
        (&["--max-c", "c2", "--max-n", "n2"][..], 0, None),
        (&["--max-c", "c1"][..], 1, Some("CONTRACT_CAP")),
        (&["--max-n", "n1"][..], 1, Some("CONTRACT_CAP")),
        (
            &["--max-c", "c0", "--max-n", "n3"][..],
            1,
            Some("CONTRACT_CAP"),
        ),
    ] {
        let args: Vec<&str> = run.iter().chain(caps).copied().collect();
        let (exit, stdout) = cli(&args);
        assert_eq!(exit, status, "{caps:?} {stdout}");
        let document: Value = serde_json::from_str(&stdout).unwrap();
        assert_eq!(
            document["diagnostics"][0]["code"].as_str(),
            code,
            "{caps:?}"
        );
    }
    // A cap never widens: an undeclared program passes the narrowest caps,
    // and a level a cap cannot name is a usage error.
    let plain = scratch("plain", &program(""));
    let (exit, _) = cli(&[
        "check",
        plain.to_str().unwrap(),
        "--max-c",
        "c0",
        "--max-n",
        "n0",
    ]);
    assert_eq!(exit, 0);
    for caps in [
        ["--max-c", "c5"],
        ["--max-n", "n4"],
        ["--max-c", "c3"],
        ["--max-n", "N1"],
    ] {
        let args: Vec<&str> = ["check", file].iter().chain(&caps).copied().collect();
        assert_eq!(cli(&args).0, 2, "{caps:?}");
    }
    assert_eq!(cli(&["check", file, "--max-c", "c1", "--max-c", "c2"]).0, 2);
    let (exit, stdout) = cli(&["check", file, "--region-report", "--format", "json"]);
    assert_eq!(exit, 0);
    let report: Value = serde_json::from_str(&stdout).unwrap();
    assert_eq!(
        report["regions"],
        serde_json::json!([
            {"function": "fold_sum", "c": "c2", "contract": "f32-relaxed-v2", "s": "s0", "n": "n2", "t": "t0"},
            {"function": "main", "c": "c0", "contract": null, "s": "s0", "n": "n0", "t": "t0"}
        ])
    );
    let (exit, text) = cli(&["check", file, "--region-report"]);
    assert_eq!(exit, 0);
    assert!(
        text.lines()
            .nth(1)
            .is_some_and(|line| line.starts_with("fold_sum ")),
        "{text}"
    );
    assert_eq!(
        cli(&["run", file, "--engine", "reference", "--region-report"]).0,
        2
    );
    // The report refuses what `check` refuses, and names a module function once.
    let ill_typed = scratch("ill-typed", "fn main() -> i32 { return true; }");
    let (exit, stdout) = cli(&["check", ill_typed.to_str().unwrap(), "--region-report"]);
    assert_eq!(exit, 1, "{stdout}");
    let source =
        "fn main() -> i32 { return m::one(); }\nmodule m { pub fn one() -> i32 { return 1; } }";
    let module = scratch("module", source);
    let (exit, stdout) = cli(&[
        "check",
        module.to_str().unwrap(),
        "--region-report",
        "--format",
        "json",
    ]);
    assert_eq!(exit, 0, "{stdout}");
    let names: Vec<Value> = serde_json::from_str::<Value>(&stdout).unwrap()["regions"]
        .as_array()
        .unwrap()
        .iter()
        .map(|row| row["function"].clone())
        .collect();
    assert!(names.contains(&"m::one".into()), "{stdout}");
}

/// #210: each contract code carries its note, and a validated fix where one exists.
#[test]
fn contract_codes_carry_notes_and_validated_fixes() {
    for (attribute, code, fixed) in [
        (
            "#[region(n = \"n4\")]",
            "CONTRACT_UNAVAILABLE",
            Some("#[region(n = \"n2\")]"),
        ),
        (
            "#[region(s = \"s2\", t = \"t1\")]",
            "CONTRACT_UNAVAILABLE",
            Some("#[region(s = \"s0\", t = \"t0\")]"),
        ),
        ("#[region(c = \"c5\")]", "CONTRACT_UNAVAILABLE", None),
        (
            "#[region(c = \"c0\", contract = \"f32-exact-v2\")]",
            "CONTRACT_MISMATCH",
            Some("#[region(c = \"c1\", contract = \"f32-exact-v2\")]"),
        ),
        ("#[region(c = \"c1\")]", "CONTRACT_MISMATCH", None),
        (
            "#[region(c = \"C1\", contract = \"f32-tree-v2\")]",
            "CONTRACT_UNKNOWN",
            Some("#[region(c = \"c1\", contract = \"f32-tree-v2\")]"),
        ),
        ("#[region(c = \"c3\")]", "CONTRACT_UNKNOWN", None),
    ] {
        let source = format!("{attribute}\nfn main() -> i32 {{ return 0; }}\n");
        let path = scratch("fix", &source);
        let (status, stdout) = cli(&["check", path.to_str().unwrap(), "--format", "json"]);
        assert_eq!(status, 1, "{attribute}");
        let diagnostic = serde_json::from_str::<Value>(&stdout).unwrap()["diagnostics"][0].clone();
        assert_eq!(diagnostic["code"], code, "{attribute}");
        assert_eq!(
            diagnostic["notes"].as_array().map(Vec::len),
            Some(1),
            "{attribute}"
        );
        let fixes = diagnostic["suggested_fixes"].as_array().unwrap();
        let applied = fixes.first().map(|fix| {
            let mut text = source.clone();
            let mut edits: Vec<_> = fix["edits"].as_array().unwrap().iter().collect();
            edits.sort_by_key(|edit| std::cmp::Reverse(edit["span"]["start"].as_u64()));
            for edit in edits {
                let (start, end) = (
                    edit["span"]["start"].as_u64().unwrap(),
                    edit["span"]["end"].as_u64().unwrap(),
                );
                text.replace_range(
                    start as usize..end as usize,
                    edit["replacement"].as_str().unwrap(),
                );
            }
            text.lines().next().unwrap().to_owned()
        });
        assert_eq!(applied.as_deref(), fixed, "{attribute}");
    }
    let path = scratch(
        "cap-note",
        &program("#[region(contract = \"f32-tree-v2\")]"),
    );
    let (_, stdout) = cli(&[
        "check",
        path.to_str().unwrap(),
        "--max-c",
        "c0",
        "--format",
        "json",
    ]);
    let diagnostic = serde_json::from_str::<Value>(&stdout).unwrap()["diagnostics"][0].clone();
    assert_eq!(
        (
            diagnostic["code"].as_str(),
            diagnostic["notes"].as_array().map(Vec::len)
        ),
        (Some("CONTRACT_CAP"), Some(1))
    );
    assert_eq!(diagnostic["suggested_fixes"], serde_json::json!([]));
}
