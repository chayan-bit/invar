#![forbid(unsafe_code)]
//! P6(a) (#263): `invar run` executes serial-only programs through the checked
//! parallel normal form. Every law program of the P1b differential harness
//! (`tests/net_normal_form_laws.rs`, read from that file so the corpus cannot
//! drift) is wrapped in a `main` with seeded inputs and run through the CPU
//! engine; each result, first trap and fuel exhaustion must equal the
//! reference evaluator (`eval::run`), at the default fuel and at the exact
//! fuel boundary.

use invar::{
    engine::{self, Engine},
    eval::{self, EvalError, Value},
};
use serde_json::Value as Json;
use std::{collections::BTreeMap, path::PathBuf, process::Command};

const LAWS: &str = include_str!("net_normal_form_laws.rs");

/// SplitMix64, as in the law harness.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }
    fn below(&mut self, n: u64) -> u64 {
        self.next() % n.max(1)
    }
    fn i32(&mut self) -> i32 {
        match self.below(10) {
            0 => [i32::MIN, i32::MAX, -1, 0, 1][self.below(5) as usize],
            1 => (self.next() as i32) >> 1,
            _ => self.below(101) as i32 - 50,
        }
    }
}

/// The `const L..: &str = "...";` law programs of the P1b harness.
fn law_programs() -> Vec<(String, String)> {
    LAWS.lines()
        .filter_map(|line| {
            let rest = line.strip_prefix("const L")?;
            let (name, source) = rest.split_once(": &str = \"")?;
            Some((format!("L{name}"), source.strip_suffix("\";")?.to_owned()))
        })
        .collect()
}

struct Signature {
    entry: String,
    params: Vec<(String, String)>,
    result: String,
}

/// The last function of a law program is its entry.
fn signature(source: &str) -> Signature {
    let at = source.rfind("fn ").unwrap() + 3;
    let (entry, rest) = source[at..].split_once('(').unwrap();
    let (params, rest) = rest.split_once(") -> ").unwrap();
    let result = rest.split_once(" {").unwrap().0.to_owned();
    let params = params
        .split(", ")
        .filter(|p| !p.is_empty())
        .map(|p| {
            let (_, ty) = p.split_once(": ").unwrap();
            let mode = ty.split_once(' ').map_or("", |(mode, _)| mode);
            let ty = ty.rsplit(' ').next().unwrap();
            (mode.to_owned(), ty.to_owned())
        })
        .collect();
    Signature {
        entry: entry.to_owned(),
        params,
        result,
    }
}

fn i32_literal(v: i32) -> String {
    match v {
        i32::MIN => "(0 - 2147483647 - 1)".into(),
        v if v < 0 => format!("(0 - {})", -v),
        v => v.to_string(),
    }
}

/// `checksum` folds a u32 into an i32 without a conversion intrinsic.
const CHECKSUM: &str = "fn checksum(r: u32) -> i32 { var out = 0; var bit = 1; var x = r; var k = 0u32; while k < 31u32 { if x % 2u32 == 1u32 { out = out + bit; } x = x / 2u32; if k < 30u32 { bit = bit * 2; } k = k + 1u32; } return out; }";

/// A `main` that calls the entry with seeded arguments and folds its result
/// and every `mut` array into the returned `i32`.
fn wrap(source: &str, rng: &mut Rng) -> Option<String> {
    let sig = signature(source);
    if sig.params.iter().any(|(_, ty)| ty.contains("f32")) || sig.result.contains("f32") {
        return None;
    }
    let n = rng.below(14) as u32;
    let mut lets = String::new();
    let mut args = Vec::new();
    let mut folds = Vec::new();
    for (i, (mode, ty)) in sig.params.iter().enumerate() {
        let name = format!("a{i}");
        match ty.as_str() {
            "Array<i32>" | "Array<u32>" => {
                let len = match rng.below(6) {
                    0 => n.saturating_sub(1),
                    1 => n + 3,
                    _ => n,
                } as usize;
                let u32s = ty == "Array<u32>";
                let items: Vec<String> = (0..len)
                    .map(|_| match (u32s, rng.below(4)) {
                        (true, 0) => format!("{}u32", rng.next() as u32),
                        (true, _) => format!("{}u32", rng.below(n as u64 + 2)),
                        (false, _) => i32_literal(rng.i32()),
                    })
                    .collect();
                let value = if items.is_empty() {
                    format!(
                        "filled_{}(0u32, 0{})",
                        &ty[6..9],
                        if u32s { "u32" } else { "" }
                    )
                } else {
                    format!("[{}]", items.join(", "))
                };
                let binding = if mode == "mut" { "var" } else { "let" };
                lets += &format!("{binding} {name}: {ty} = {value}; ");
                if mode == "mut" {
                    folds.push((name.clone(), u32s));
                }
            }
            "u32" => {
                let v = match rng.below(5) {
                    0 => rng.below(4),
                    _ => u64::from(n),
                };
                lets += &format!("let {name} = {v}u32; ");
            }
            "i32" => lets += &format!("let {name} = {}; ", i32_literal(rng.i32())),
            other => panic!("law parameter type {other}"),
        }
        args.push(name);
    }
    let call = format!("{}({})", sig.entry, args.join(", "));
    let mut body = match sig.result.as_str() {
        "Unit" => format!("{call}; var c = 0; "),
        "i32" => format!("let r = {call}; var c = r % 1000003; "),
        "u32" => format!("let r = {call}; var c = checksum(r % 1000003u32); "),
        "Array<i32>" => {
            folds.push(("r".into(), false));
            format!("let r = {call}; var c = 0; ")
        }
        other => panic!("law result type {other}"),
    };
    for (array, u32s) in folds {
        let element = if u32s {
            format!("checksum({array}[j{array}] % 1000u32)")
        } else {
            format!("{array}[j{array}] % 1000")
        };
        body += &format!(
            "for j{array} in 0u32 .. len({array}) {{ c = (c % 1000003) * 3 + {element}; }} "
        );
    }
    Some(format!(
        "{source} {CHECKSUM} fn main() -> i32 {{ {lets}{body}return c; }}"
    ))
}

fn checked(source: &str) -> invar::check::CheckedProgram {
    invar::front::parse(source.as_bytes())
        .and_then(|program| invar::check::check(program, true))
        .unwrap_or_else(|e| panic!("{source}: {e:?}"))
}

/// What `invar run --engine cpu-parallel` did, as the reference's outcome type.
fn engine_run(source: &str, fuel: u64) -> (Result<Value, EvalError>, Option<Json>) {
    let program = invar::mir::lower(&checked(source)).unwrap();
    match engine::run_mir(
        source.as_bytes(),
        &program,
        Engine::CpuParallel,
        fuel,
        2,
        None,
    ) {
        Ok(receipt) => (
            Ok(Value::I32(receipt["result"].as_i64().unwrap() as i32)),
            receipt.get("normal_form").cloned(),
        ),
        Err(error) => (Err(error), None),
    }
}

fn same(left: &Result<Value, EvalError>, right: &Result<Value, EvalError>) -> bool {
    match (left, right) {
        (Ok(a), Ok(b)) => a == b,
        (Err(EvalError::Tool { code: a, .. }), Err(EvalError::Tool { code: b, .. })) => a == b,
        (Err(a), Err(b)) => a == b,
        _ => false,
    }
}

#[test]
fn law_programs_through_run_equal_the_reference_evaluator() {
    let programs = law_programs();
    assert!(programs.len() >= 25, "law corpus found: {}", programs.len());
    let mut rng = Rng(0x263);
    let mut routed: BTreeMap<String, usize> = BTreeMap::new();
    let mut fallbacks: BTreeMap<String, usize> = BTreeMap::new();
    let mut laws = std::collections::BTreeSet::new();
    let mut cases = 0;
    for (name, law_source) in &programs {
        for _ in 0..30 {
            let Some(source) = wrap(law_source, &mut rng) else {
                continue;
            };
            let program = checked(&source);
            let reference = eval::observe(&program, eval::DEFAULT_FUEL);
            let steps = reference.steps_used;
            let mut fuels = vec![eval::DEFAULT_FUEL, steps, steps.saturating_sub(1)];
            fuels.retain(|&fuel| fuel > 0 && fuel <= eval::DEFAULT_FUEL);
            fuels.dedup();
            for fuel in fuels {
                let expected = eval::run(&program, fuel).map(|outcome| outcome.value);
                let (got, route) = engine_run(&source, fuel);
                assert!(
                    same(&got, &expected),
                    "{name} fuel {fuel}: engine {got:?} reference {expected:?}\n{source}"
                );
                cases += 1;
                match route {
                    Some(route) if route["status"] == "routed" => {
                        *routed.entry(name.clone()).or_default() += 1;
                        for law in route["laws"].as_array().unwrap() {
                            laws.insert(law.as_str().unwrap().to_owned());
                        }
                    }
                    Some(route) => {
                        *fallbacks
                            .entry(route["reason"].as_str().unwrap().to_owned())
                            .or_default() += 1;
                    }
                    None => {}
                }
            }
        }
    }
    eprintln!("P6_RUN_CASES={cases} ROUTED={routed:?} LAWS={laws:?} FALLBACKS={fallbacks:?}");
    // Every v0.1 law runs by law through `invar run` except the recursion laws
    // (L9, L10) and residue rings, which the route refuses on purpose.
    for law in [
        "L1", "L2", "L3", "L4", "L5", "L6", "L7", "L11", "L12", "L13", "L14", "L15", "L19", "L20",
    ] {
        assert!(laws.contains(law), "{law} never routed: {laws:?}");
    }
}

fn write(name: &str, source: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("invar-p6-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join(name);
    std::fs::write(&path, source).unwrap();
    path
}

/// `stale_opt_out` sets the removed `INVAR_NO_NORMAL_FORM=1`, which must
/// change nothing (#263).
fn cli(args: &[&str], stale_opt_out: bool) -> (i32, Json) {
    let mut command = Command::new(env!("CARGO_BIN_EXE_invar"));
    command.args(args).env_remove("INVAR_NO_NORMAL_FORM");
    if stale_opt_out {
        command.env("INVAR_NO_NORMAL_FORM", "1");
    }
    let output = command.output().unwrap();
    (
        output.status.code().unwrap(),
        serde_json::from_slice(&output.stdout).unwrap(),
    )
}

fn schema(name: &str) -> Json {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("schema")
        .join(name);
    serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap()
}

const SUM: &str = "fn f(xs: read Array<i32>, n: u32, s0: i32) -> i32 { var s = s0; for i in 0u32 .. n { s = s + xs[i]; } return s; } fn main() -> i32 { let xs: Array<i32> = [4, 9, (0 - 3), 12]; return f(xs, 4u32, 20); }";

#[test]
fn run_receipt_records_the_law_and_the_removed_opt_out_changes_nothing() {
    let path = write("sum.inv", SUM);
    let path = path.to_str().unwrap();
    let args = [
        "run",
        path,
        "--engine",
        "cpu-parallel",
        "--workers",
        "2",
        "--format",
        "json",
    ];
    let (status, document) = cli(&args, false);
    assert_eq!(status, 0, "{document}");
    assert_eq!(document["result"]["value"], 42);
    let route = &document["normal_form"];
    assert_eq!(route["status"], "routed", "{document}");
    assert!(route["laws"].as_array().unwrap().contains(&"L3".into()));
    assert!(route["certificates"].as_u64().unwrap() >= 2);
    assert_eq!(route["fingerprint"].as_str().unwrap().len(), 64);
    let v1 = schema("invar-diagnostics-v1.schema.json");
    assert!(jsonschema::is_valid(&v1, &document), "{document}");
    assert_eq!(cli(&args, true), (0, document.clone()));
    let (_, reference) = cli(
        &["run", path, "--engine", "reference", "--format", "json"],
        false,
    );
    assert_eq!(reference["result"], document["result"]);
}

/// A core map makes the program not serial-only, so the route never runs; the
/// checked sum L3 certifies has no family lowering and runs on the CPU path.
const GAP: &str = "fn run(a: read Array<u32>, n: u32) -> u32 { var s = 0u32; for i in 0u32..n { s = s + a[i]; } return s; } fn main() -> i32 { let a = filled_u32(8u32, 2u32); var out = filled_u32(8u32, 0u32); for i in 0u32..8u32 { out[i] = wrapping_add(a[i], 3u32); } if run(out, 8u32) != 40u32 { return 1; } return 42; }";

#[test]
fn issue_263_a_loop_no_family_lowers_records_a_typed_fallback() {
    let path = write("gap.inv", GAP);
    let path = path.to_str().unwrap();
    let args = [
        "run",
        path,
        "--engine",
        "cpu-parallel",
        "--workers",
        "2",
        "--format",
        "json",
    ];
    let (status, document) = cli(&args, false);
    assert_eq!(status, 0, "{document}");
    assert_eq!(document["result"]["value"], 42);
    assert_eq!(
        document["normal_form"],
        serde_json::json!({"status": "fallback", "reason": "no-lowering:reduce/lifted-sum"}),
        "{document}"
    );
    let v1 = schema("invar-diagnostics-v1.schema.json");
    assert!(jsonschema::is_valid(&v1, &document), "{document}");
    let mut forged = document.clone();
    forged["normal_form"]["reason"] = "no-lowering:".into();
    assert!(!jsonschema::is_valid(&v1, &forged));
    assert_eq!(cli(&args, true), (0, document));
}

#[test]
fn issue_263_every_normal_form_schema_accepts_the_typed_fallback() {
    let fallback =
        serde_json::json!({"status": "fallback", "reason": "no-lowering:reduce/lifted-sum"});
    let malformed = serde_json::json!({"status": "fallback", "reason": "no-lowering:reduce/"});
    for (name, pointer) in [
        (
            "invar-diagnostics-v1.schema.json",
            "/properties/normal_form",
        ),
        (
            "invar-diagnostics-v3.schema.json",
            "/properties/normal_form",
        ),
        (
            "invar-explain-device-v1.schema.json",
            "/properties/normal_form",
        ),
        (
            "invar-explain-device-v5.schema.json",
            "/properties/normal_form",
        ),
        (
            "explain-v1.schema.json",
            "/properties/device_execution/properties/normal_form",
        ),
        (
            "explain-v5.schema.json",
            "/properties/device_execution/properties/normal_form",
        ),
    ] {
        let object = schema(name).pointer(pointer).unwrap().clone();
        assert!(jsonschema::is_valid(&object, &fallback), "{name}");
        assert!(!jsonschema::is_valid(&object, &malformed), "{name}");
    }
}

#[test]
fn issue_263_a_sort_program_keeps_its_versioned_explain_document() {
    // The engine records no gap for a sort program, whose auto explain was
    // v3 (no `normal_form`) before #216 made every auto explain v5.
    let source = GAP.replace(
        "return 42; }",
        "let xs: Array<u32> = [3u32, 1u32, 2u32]; let ids = argsort(xs); if ids[0u32] != 1u32 { return 2; } return 42; }",
    );
    let path = write("gap-sort.inv", &source);
    let path = path.to_str().unwrap();
    let (status, document) = cli(
        &["explain", path, "--engine", "auto", "--format", "json"],
        false,
    );
    assert_eq!(status, 0, "{document}");
    assert_eq!(document["schema"], "invar-explain-v5", "{document}");
    assert!(document["device_execution"].get("normal_form").is_none());
    assert!(
        jsonschema::is_valid(&schema("explain-v5.schema.json"), &document),
        "{document}"
    );
    assert!(jsonschema::is_valid(
        &schema("invar-explain-device-v5.schema.json"),
        &document["device_execution"]
    ));
}

#[test]
fn programs_without_a_certified_law_keep_their_documents() {
    let path = write("plain.inv", "fn main() -> i32 { return 42; }");
    let path = path.to_str().unwrap();
    let args = [
        "run",
        path,
        "--engine",
        "cpu-parallel",
        "--workers",
        "1",
        "--format",
        "json",
    ];
    let (status, document) = cli(&args, false);
    assert_eq!(status, 0, "{document}");
    assert_eq!((status, document), cli(&args, true));
}

#[test]
fn a_tight_fuel_falls_back_to_the_exact_serial_limit() {
    let path = write("sum-fuel.inv", SUM);
    let path = path.to_str().unwrap();
    let (status, document) = cli(
        &[
            "run",
            path,
            "--engine",
            "cpu-parallel",
            "--workers",
            "2",
            "--fuel",
            "40",
            "--format",
            "json",
        ],
        false,
    );
    let (reference_status, reference) = cli(
        &[
            "run",
            path,
            "--engine",
            "reference",
            "--fuel",
            "40",
            "--format",
            "json",
        ],
        false,
    );
    assert_eq!(status, reference_status, "{document} {reference}");
    assert_eq!(document["status"], reference["status"]);
    assert_eq!(document["status"], "evaluation_limit");
}

#[cfg(not(any(feature = "cuda", all(feature = "metal", target_os = "macos"))))]
#[test]
fn explain_auto_records_the_normal_form_route() {
    let path = write("sum-explain.inv", SUM);
    let (status, document) = cli(
        &[
            "explain",
            path.to_str().unwrap(),
            "--engine",
            "auto",
            "--format",
            "json",
        ],
        false,
    );
    assert_eq!(status, 0, "{document}");
    assert_eq!(document["schema"], "invar-explain-v5");
    assert_eq!(document["auto_path"], "established");
    let device = &document["device_execution"];
    assert_eq!(device["normal_form"]["status"], "routed", "{document}");
    assert_eq!(device["result"], 42);
    assert!(jsonschema::is_valid(
        &schema("explain-v5.schema.json"),
        &document
    ));
    assert!(jsonschema::is_valid(
        &schema("invar-explain-device-v5.schema.json"),
        device
    ));
}

/// 63 live `filled_i32` arrays of 2^22 elements and one of `last`, then the
/// routed SUM law over a 4-element literal: `2^28 - 4 + last` live elements.
fn live_arrays(last: u32) -> String {
    let lets: String = (0..63)
        .map(|k| format!("let a{k}: Array<i32> = filled_i32(4194304u32, 0); "))
        .collect();
    format!(
        "fn f(xs: read Array<i32>, n: u32, s0: i32) -> i32 {{ var s = s0; for i in 0u32 .. n {{ s = s + xs[i]; }} return s; }} \
         fn main() -> i32 {{ {lets}let last: Array<i32> = filled_i32({last}u32, 0); let xs: Array<i32> = [4, 9, (0 - 3), 12]; return f(xs, 4u32, 20); }}"
    )
}

#[test]
fn the_live_array_ceiling_is_enforced_before_publishing() {
    // Exactly the 2^28-element ceiling is routed; one element more is not.
    let inside = live_arrays((1 << 22) - 4);
    let (result, receipt) = engine_run(&inside, eval::DEFAULT_FUEL);
    let reference = eval::run(&checked(&inside), eval::DEFAULT_FUEL).map(|outcome| outcome.value);
    assert!(same(&result, &reference), "{result:?} {reference:?}");
    assert_eq!(result.unwrap(), Value::I32(42));
    assert_eq!(receipt.unwrap()["status"], "routed");

    let past = live_arrays((1 << 22) - 3);
    let (result, _) = engine_run(&past, eval::DEFAULT_FUEL);
    let reference = eval::run(&checked(&past), eval::DEFAULT_FUEL).map(|outcome| outcome.value);
    assert!(
        matches!(
            &reference,
            Err(EvalError::Tool {
                code: "ARRAY_SIZE_LIMIT",
                ..
            })
        ),
        "{reference:?}"
    );
    assert!(same(&result, &reference), "{result:?} {reference:?}");
    let program = invar::mir::lower(&checked(&past)).unwrap();
    assert!(matches!(
        invar::schedule::normal_form::route(&program, eval::DEFAULT_FUEL),
        invar::schedule::normal_form::Route::Fallback("live-elements-unproven")
    ));
}

#[test]
fn invalid_worker_counts_are_rejected_before_routing() {
    let program = invar::mir::lower(&checked(SUM)).unwrap();
    for workers in [0, invar::schedule::MAX_WORKERS + 1] {
        let error = engine::run_mir(
            SUM.as_bytes(),
            &program,
            Engine::CpuParallel,
            eval::DEFAULT_FUEL,
            workers,
            None,
        )
        .unwrap_err();
        assert!(
            matches!(
                error,
                EvalError::Tool {
                    code: "CPU_WORKERS",
                    ..
                }
            ),
            "{workers}: {error:?}"
        );
    }
    let (result, receipt) = engine_run(SUM, eval::DEFAULT_FUEL);
    assert_eq!(result.unwrap(), Value::I32(42));
    assert_eq!(receipt.unwrap()["status"], "routed");
}
