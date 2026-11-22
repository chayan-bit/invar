//! #209: worker-count, engine and batch-size invariance at C0 and C1, checked
//! here run by run, and the `invariance` receipt of `run --check-workers`
//! (`docs/spec/invar-v0.2-axis-n.md` NA7.13, `docs/invariance.md`).

use std::path::PathBuf;
use std::process::Command;
use std::sync::atomic::{AtomicUsize, Ordering};

use invar::{check::check, front::parse, mir, mir_eval, schedule};
use serde_json::{Value, json};

use crate::split_mix::Rng;

const WORKERS: [usize; 4] = [1, 2, 7, 64];
const FUEL: u64 = 50_000_000;

/// `main` returns `i32`: a `u32` leaves through an injective code (low 31
/// bits, the top bit as the sign).
const CODE: &str = "fn code(s: u32) -> i32 { var r: i32 = 0; var p: i32 = 1; var q: u32 = s; \
    for b in 0u32 .. 31u32 { if q % 2u32 == 1u32 { r = r + p; } q = q / 2u32; if b < 30u32 { p = p * 2; } } \
    if q == 1u32 { return -1 - r; } return r; } ";

/// `source` in a temporary file of its own: tests run concurrently in one process.
fn source_file(source: &str) -> PathBuf {
    static NEXT: AtomicUsize = AtomicUsize::new(0);
    let n = NEXT.fetch_add(1, Ordering::Relaxed);
    let path = std::env::temp_dir().join(format!("invar-209-{}-{n}.inv", std::process::id()));
    std::fs::write(&path, source).unwrap();
    path
}

/// `invar run SOURCE ARGS...`: exit status and stdout.
fn invar_run(args: &[&str], source: &str) -> (Option<i32>, Vec<u8>) {
    let path = source_file(source);
    let output = Command::new(env!("CARGO_BIN_EXE_invar"))
        .arg("run")
        .arg(&path)
        .args(args)
        .output()
        .unwrap();
    std::fs::remove_file(&path).unwrap();
    (output.status.code(), output.stdout)
}

fn invar_json(args: &[&str], source: &str) -> (Option<i32>, Value) {
    let (status, stdout) = invar_run(args, source);
    let document = serde_json::from_slice(&stdout)
        .unwrap_or_else(|e| panic!("{e}: {}", String::from_utf8_lossy(&stdout)));
    (status, document)
}

/// The compared outcome of one run: result or error, and the fuel used.
fn outcome(observation: &invar::eval::Observation) -> String {
    format!("{:?}; fuel {}", observation.result, observation.steps_used)
}

/// Every engine this build has and every worker count give the reference
/// evaluator's outcome for `source`; returns that outcome. The `run
/// --check-workers` receipt names exactly the counts and comparisons made.
fn assert_invariant(source: &str, fuel: u64, c_levels: &[&str]) -> String {
    let checked = check(parse(source.as_bytes()).unwrap(), true)
        .unwrap_or_else(|e| panic!("{source}: {e:?}"));
    let program = mir::lower(&checked).unwrap();
    let expected = outcome(&invar::eval::observe(&checked, fuel));
    assert_eq!(
        outcome(&mir_eval::observe(&program, fuel)),
        expected,
        "MIR: {source}"
    );
    for workers in WORKERS {
        let run = schedule::run_source(source.as_bytes(), &program, fuel, workers).unwrap();
        assert_eq!(
            outcome(&run.observation),
            expected,
            "{workers} workers: {source}"
        );
    }
    let fuel_text = fuel.to_string();
    let (status, document) = invar_json(
        &[
            "--engine",
            "cpu-parallel",
            "--check-workers",
            "1,2,7,64",
            "--fuel",
            &fuel_text,
            "--format",
            "json",
        ],
        source,
    );
    let schema: Value = serde_json::from_str(include_str!(
        "../../schema/invar-diagnostics-v3.schema.json"
    ))
    .unwrap();
    assert!(jsonschema::is_valid(&schema, &document), "{document}");
    let invariance = &document["invariance"];
    assert_eq!(invariance["workers"], json!(WORKERS), "{document}");
    assert_eq!(invariance["engines"], json!(["reference", "cpu-parallel"]));
    assert_eq!(invariance["compared"], json!(["result", "trap", "fuel"]));
    assert_eq!(invariance["levels"]["c"], json!(c_levels), "{source}");
    assert!(
        expected.ends_with(&format!("; fuel {}", invariance["fuel_used"])),
        "{expected} {invariance}"
    );
    match status {
        Some(0) => assert!(
            expected.starts_with(&format!("Ok(I32({}))", document["result"]["value"])),
            "{expected} {document}"
        ),
        Some(3) => assert!(
            expected.contains("Trap") && document["status"] == "trap",
            "{document}"
        ),
        Some(4) => assert!(expected.contains("EvaluationLimit"), "{document}"),
        other => panic!("{other:?}: {document}"),
    }
    #[cfg(feature = "native")]
    if status == Some(0) {
        let (_, native) = invar_json(
            &[
                "--engine",
                "native-serial",
                "--fuel",
                &fuel_text,
                "--format",
                "json",
            ],
            source,
        );
        assert_eq!(
            native["result"], document["result"],
            "native-serial: {source}"
        );
    }
    expected
}

fn u32_literals(rng: &mut Rng, n: usize) -> String {
    (0..n)
        .map(|_| format!("{}u32", rng.next() as u32))
        .collect::<Vec<_>>()
        .join(", ")
}

/// An `f32` fold `f` with an optional attribute, over terms whose order
/// matters (2^24 then small integers); `main` returns the sum less 2^24.
fn f32_fold(attribute: &str) -> String {
    format!(
        "{attribute} fn f(xs: read Array<f32>) -> f32 {{ var s: f32 = 0.0; \
         for i in 0u32 .. len(xs) {{ s = s + xs[i]; }} return s; }} \
         fn main() -> i32 {{ let xs: Array<f32> = [{F32_TERMS}]; return i32_from_f32(f(xs) - 16777216.0); }}"
    )
}

const F32_TERMS: &str = "16777216.0, 1.0, 1.0, 3.0, 1.0, 2.0, 1.0, 1.0, 3.0, 1.0, 1.0, 1.0";

/// Seeded programs over each admitted plan family and each C0/C1 fold: every
/// worker count, the MIR evaluator and the reference give the same bits,
/// trap position and fuel, at the full fuel and one step short of it.
#[test]
fn worker_counts_and_engines_keep_bits_trap_and_fuel_at_c0_and_c1() {
    let mut rng = Rng(0x209);
    let xs = u32_literals(&mut rng, 300);
    let cases: Vec<(String, &[&str])> = vec![
        (
            format!(
                "{CODE}fn main() -> i32 {{ let xs: Array<u32> = [{xs}]; var s: u32 = 0u32; \
                 for i in 0u32 .. len(xs) {{ s = wrapping_add(s, xs[i]); }} return code(s); }}"
            ),
            &["c0"],
        ),
        (
            format!(
                "{CODE}fn main() -> i32 {{ let xs: Array<u32> = [{xs}]; \
                 var ys: Array<u32> = filled_u32(len(xs), 0u32); \
                 for i in 0u32 .. len(xs) {{ ys[i] = wrapping_mul(xs[i], 2654435761u32); }} \
                 var s: u32 = 1u32; for j in 0u32 .. len(ys) {{ s = wrapping_mul(s, wrapping_add(ys[j], 1u32)); }} \
                 return code(s); }}"
            ),
            &["c0"],
        ),
        (
            format!(
                "{CODE}fn main() -> i32 {{ let xs: Array<u32> = [{xs}]; var m: u32 = 0u32; var c: u32 = 0u32; \
                 for i in 0u32 .. len(xs) {{ if xs[i] >= m {{ m = xs[i]; c = i; }} }} \
                 return code(wrapping_add(m, c)); }}"
            ),
            &["c0"],
        ),
        // A checked i32 sum that overflows: the trap and its position are compared.
        (
            "fn main() -> i32 { let xs: Array<i32> = [5, 2147483600, 40, 9, -3]; var s: i32 = 0; \
             for i in 0u32 .. len(xs) { s = s + xs[i]; } return s; }"
                .to_owned(),
            &["c0"],
        ),
        (f32_fold(""), &["c0"]),
        (
            f32_fold("#[region(contract = \"f32-tree-v2\")]"),
            &["c0", "c1"],
        ),
        (
            f32_fold("#[region(contract = \"f32-exact-v2\")]"),
            &["c0", "c1"],
        ),
        (
            format!(
                "fn main() -> i32 {{ let xs: Array<f32> = [{F32_TERMS}]; \
                 return i32_from_f32(sum_tree(xs) - 16777216.0) * 100 \
                 + i32_from_f32(sum_exact(xs) - 16777216.0); }}"
            ),
            &["c0"],
        ),
    ];
    for (source, levels) in &cases {
        let expected = assert_invariant(source, FUEL, levels);
        let used: u64 = expected.rsplit(' ').next().unwrap().parse().unwrap();
        // One step short: every engine stops at the same fuel limit, or
        // traps at the same place first.
        let short = assert_invariant(source, used - 1, levels);
        assert!(
            short.contains("EvaluationLimit") || short.contains("Trap"),
            "{short}"
        );
    }
}

/// A batch of `batch` rows whose inputs depend only on the row index: row
/// `r` is a strict binary32 sum of 16 terms whose order matters (2^24 then
/// small integers), and an admitted wrapping map runs over the batch.
/// `main` returns a hash of rows `0 .. rows`.
fn batch_program(batch: u32, rows: u32) -> String {
    format!(
        "{CODE}fn row(xs: read Array<f32>, r: u32) -> f32 {{ var s: f32 = 0.0; \
         for j in 0u32 .. 16u32 {{ s = s + xs[r * 16u32 + j]; }} return s; }} \
         fn keyed(ids: read Array<u32>) -> Array<u32> {{ \
         var keys: Array<u32> = filled_u32(len(ids), 0u32); \
         for k in 0u32 .. len(ids) {{ keys[k] = wrapping_mul(ids[k], 2654435761u32); }} return keys; }} \
         fn main() -> i32 {{ var xs: Array<f32> = filled_f32({batch}u32 * 16u32, 0.0); \
         for i in 0u32 .. len(xs) {{ if i % 16u32 == 0u32 {{ xs[i] = 16777216.0; }} \
             else {{ xs[i] = f32_from_u32((i / 16u32 + i) % 3u32 + 1u32); }} }} \
         var ids: Array<u32> = filled_u32({batch}u32, 0u32); \
         for d in 0u32 .. len(ids) {{ ids[d] = d; }} \
         let keys: Array<u32> = keyed(ids); \
         var out: Array<f32> = filled_f32({batch}u32, 0.0); \
         for r in 0u32 .. {batch}u32 {{ out[r] = row(xs, r); }} \
         var h: u32 = 0u32; \
         for q in 0u32 .. {rows}u32 {{ \
             h = wrapping_add(wrapping_mul(h, 31u32), wrapping_add(u32_from_f32(out[q]), keys[q])); }} \
         return code(h); }}"
    )
}

/// The same rows give the same bits whether they run in a batch of 1, 8 or
/// 1024, on every engine and worker count. A bigger batch does more work, so
/// across batches only the result is compared, not the fuel.
#[test]
fn batch_sizes_keep_every_row() {
    let result = |outcome: String| outcome.split("; fuel").next().unwrap().to_owned();
    for rows in [1, 8] {
        let results: Vec<_> = [1, 8, 1024]
            .into_iter()
            .filter(|batch| *batch >= rows)
            .map(|batch| result(assert_invariant(&batch_program(batch, rows), FUEL, &["c0"])))
            .collect();
        assert!(results[0].starts_with("Ok(I32("), "{results:?}");
        assert!(
            results.iter().all(|r| *r == results[0]),
            "rows {rows}: {results:?}"
        );
    }
    // The batch map is published, so the batch really is split across workers.
    let (_, document) = invar_json(
        &[
            "--engine",
            "cpu-parallel",
            "--check-workers",
            "7",
            "--fuel",
            "50000000",
            "--format",
            "json",
        ],
        &batch_program(1024, 8),
    );
    let run = &document["invariance"]["runs"][0];
    assert!(
        run["published_regions"].as_u64().unwrap() >= 1
            && run["worker_joins"].as_u64().unwrap() >= 7,
        "{document}"
    );
}

/// The contrast the claim is specific to: a combine whose order follows the
/// worker split (per-chunk binary32 partial sums, then their sum) varies with
/// the worker count. This is expected variation, not a failure; Invar's
/// strict fold of the same terms keeps one order at every count.
#[test]
fn a_worker_ordered_float_sum_varies_and_the_strict_fold_does_not() {
    let terms: Vec<f32> = std::iter::once(16_777_216.0).chain([1.0; 63]).collect();
    let chunked = |workers: usize| {
        let size = terms.len().div_ceil(workers);
        let partials = terms
            .chunks(size)
            .map(|chunk| chunk.iter().fold(0.0f32, |s, x| s + x));
        partials.fold(0.0f32, |s, x| s + x).to_bits()
    };
    let bits: std::collections::BTreeSet<_> = WORKERS.into_iter().map(chunked).collect();
    assert!(
        bits.len() > 1,
        "expected variation across worker counts: {bits:?}"
    );
    let literals = terms
        .iter()
        .map(|x| format!("{x:.1}"))
        .collect::<Vec<_>>()
        .join(", ");
    let source = format!(
        "fn main() -> i32 {{ let xs: Array<f32> = [{literals}]; var s: f32 = 0.0; \
         for i in 0u32 .. len(xs) {{ s = s + xs[i]; }} return i32_from_f32(s - 16777216.0); }}"
    );
    assert!(assert_invariant(&source, FUEL, &["c0"]).starts_with("Ok(I32(0))"));
}

/// `--check-workers` is a list of increasing counts in 1..64, replaces
/// `--workers`, and needs `--engine cpu-parallel`.
#[test]
fn check_workers_arguments_are_checked() {
    let source = "fn main() -> i32 { return 1; }";
    for args in [
        &["--engine", "cpu-parallel", "--check-workers", "2,1"][..],
        &["--engine", "cpu-parallel", "--check-workers", "0"],
        &["--engine", "cpu-parallel", "--check-workers", "65"],
        &[
            "--engine",
            "cpu-parallel",
            "--check-workers",
            "2",
            "--workers",
            "2",
        ],
        &["--engine", "reference", "--check-workers", "2"],
    ] {
        assert_eq!(invar_run(args, source).0, Some(2), "{args:?}");
    }
    let (status, stdout) = invar_run(
        &["--engine", "cpu-parallel", "--check-workers", "1,3"],
        source,
    );
    assert_eq!(status, Some(0));
    assert_eq!(
        String::from_utf8(stdout).unwrap(),
        "Result: 1\nInvariance: result, trap and fuel identical on the reference evaluator \
         and on cpu-parallel at workers 1, 3\n"
    );
}
