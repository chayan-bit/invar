//! A program that also sorts (or runs a spatial query) runs its float
//! reductions through the fsum CPU engine, with the same receipts as the
//! program without the sort.

use invar::{check, front, mir, schedule};

const FUEL: u64 = 50_000_000;

const REDUCTIONS: &str = "let a: Array<f32> = filled_f32(70000u32, 0.5); let b: Array<f32> = filled_f32(70000u32, 2.0); let s = sum_tree(a); let d = dot_exact(a, b); let e = sum_exact(b); let t = dot_tree(a, b); if s != 35000.0 or d != 70000.0 or e != 140000.0 or t != 70000.0 { return 1; } return 0; }";
const SORT: &str =
    "var keys: Array<u32> = [3u32, 1u32, 2u32]; sort(keys); if keys[0u32] != 1u32 { return 2; } ";
const SPATIAL: &str = "let p: Array<f32> = [0.0]; let grid = voxel_grid_build(p, p, p, 1.0); let q = filled_f32(129u32, 0.0); let near = radius_neighbors(grid, q, q, q, 1.0, 1u32); ";

fn run(source: &str, workers: usize) -> schedule::Run {
    let checked =
        check::check(front::parse(source.as_bytes()).expect("parse"), true).expect("check");
    let program = mir::lower(&checked).expect("lower");
    schedule::run_source(source.as_bytes(), &program, FUEL, workers).expect("run")
}

/// Receipts without spans, which move with the inserted prefix.
fn receipts(run: &schedule::Run) -> Vec<String> {
    run.reductions
        .iter()
        .map(|r| {
            let (family, backend, reason) = (r.family, r.backend, r.reason);
            format!(
                "{} {family} {} {backend} {reason} {}",
                r.function, r.elements, r.workers
            )
        })
        .collect()
}

#[test]
fn sort_and_spatial_runs_reduce_like_the_fsum_only_program() {
    let alone = format!("fn main() -> i32 {{ {REDUCTIONS}");
    for prefix in [SORT, SPATIAL] {
        let with = format!("fn main() -> i32 {{ {prefix}{REDUCTIONS}");
        for workers in [1, 2, 4] {
            let (expected, actual) = (run(&alone, workers), run(&with, workers));
            assert_eq!(expected.observation.result, actual.observation.result);
            assert_eq!(
                actual.observation.result,
                Ok(invar::eval::Value::I32(0)),
                "{with}"
            );
            assert_eq!(expected.reductions.len(), 4, "{:?}", expected.reductions);
            assert_eq!(
                receipts(&actual),
                receipts(&expected),
                "{prefix} workers {workers}"
            );
            assert!(
                !actual.executed.is_empty(),
                "the sort or query ran: {prefix}"
            );
            if workers > 1 {
                assert!(
                    actual
                        .reductions
                        .iter()
                        .any(|r| r.backend == "cpu-parallel"),
                    "{:?}",
                    actual.reductions
                );
            }
        }
    }
}

#[test]
fn reduction_receipts_keep_their_source_spans() {
    let with = format!("fn main() -> i32 {{ {SORT}{REDUCTIONS}");
    let run = run(&with, 2);
    let start = with.find("sum_tree(a)").expect("call");
    let first = run.reductions.first().expect("receipt");
    assert_eq!(
        (first.span.start, first.span.end),
        (start, start + "sum_tree(a)".len())
    );
}
