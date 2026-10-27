//! Real-CLI contracts for spatial checking and opaque affine ownership.

use std::{io::Write, process::Command};

fn invoke(source: &str, command: &str) -> std::process::Output {
    let path = std::env::temp_dir().join(format!(
        "invar-spatial-checker-{}-{}.inv",
        std::process::id(),
        // `::` in a test name is not a valid Windows file name.
        std::thread::current()
            .name()
            .unwrap_or("test")
            .replace("::", "-")
    ));
    let mut file = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&path)
        .unwrap();
    file.write_all(source.as_bytes()).unwrap();
    drop(file);
    let mut invocation = Command::new(env!("CARGO_BIN_EXE_invar"));
    invocation
        .arg(command)
        .arg(&path)
        .args(["--format", "json"]);
    if command == "run" {
        invocation.args(["--engine", "reference"]);
    }
    let result = invocation.output();
    std::fs::remove_file(path).unwrap();
    result.unwrap()
}

fn check(source: &str) -> std::process::Output {
    invoke(source, "check")
}

const QUERIES: &str = r#"
fn main() -> i32 {
    let xs: Array<f32> = [0.0, 1.0];
    let ys: Array<f32> = [0.0, 0.0];
    let zs: Array<f32> = [0.0, 0.0];
    let grid: VoxelGrid = voxel_grid_build(xs, ys, zs, 1.0);
    let near: Neighbors = radius_neighbors(grid, xs, ys, zs, 1.0, 1u32);
    let counts = move near.counts;
    let radius_idx = move near.idx;
    let excluded: u32 = grid_excluded(grid);
    let nearest: Knn = knn(grid, xs, ys, zs, 3u32);
    let idx = move nearest.idx;
    let d2 = move nearest.d2;
    return 0;
}
"#;

#[test]
fn spatial_checker_accepts_shared_reads_and_owned_record_results() {
    let result = check(QUERIES);
    assert!(
        result.status.success(),
        "{}{}",
        String::from_utf8_lossy(&result.stdout),
        String::from_utf8_lossy(&result.stderr)
    );
}

#[test]
fn spatial_checker_rejects_grid_use_after_move_at_the_ownership_boundary() {
    let source = QUERIES.replace(
        "let nearest: Knn = knn(grid, xs, ys, zs, 3u32);",
        "let moved: VoxelGrid = move grid; let nearest: Knn = knn(grid, xs, ys, zs, 3u32);",
    );
    let result = check(&source);
    assert!(!result.status.success());
    let report: serde_json::Value = serde_json::from_slice(&result.stdout).unwrap();
    assert_eq!(report["diagnostics"][0]["code"], "OWN_USE_AFTER_MOVE");
}

#[test]
fn spatial_checker_checks_intrinsic_types_instead_of_treating_them_as_unknown() {
    let source = QUERIES.replace(
        "voxel_grid_build(xs, ys, zs, 1.0)",
        "voxel_grid_build(xs, ys, zs, 1u32)",
    );
    let result = check(&source);
    assert!(!result.status.success());
    let report: serde_json::Value = serde_json::from_slice(&result.stdout).unwrap();
    assert_eq!(report["diagnostics"][0]["code"], "TYPE_MISMATCH");
}

fn rejects(source: &str, code: &str) {
    let result = check(source);
    assert!(!result.status.success(), "unexpected acceptance: {source}");
    let report: serde_json::Value = serde_json::from_slice(&result.stdout).unwrap();
    assert_eq!(report["diagnostics"][0]["code"], code, "{report}");
}

#[test]
fn spatial_checker_enforces_each_signature_and_grid_opacity() {
    for (before, after) in [
        (
            "voxel_grid_build(xs, ys, zs, 1.0)",
            "voxel_grid_build(xs, ys, zs)",
        ),
        ("radius_neighbors(grid, xs", "radius_neighbors(xs, xs"),
        ("knn(grid, xs, ys, zs, 3u32)", "knn(grid, xs, ys, zs, 3.0)"),
    ] {
        rejects(&QUERIES.replace(before, after), "TYPE_MISMATCH");
    }
    rejects(
        &format!("struct VoxelGrid {{ x: u32 }} {QUERIES}"),
        "NAME_DUPLICATE",
    );
    rejects(
        &format!("fn bad(g: mut VoxelGrid) -> Unit {{}} {QUERIES}"),
        "TYPE_MISMATCH",
    );
    rejects(
        &QUERIES.replace("return 0;", "let x = grid.x; return 0;"),
        "NAME_UNKNOWN",
    );
    rejects(
        &QUERIES.replace(
            "let grid: VoxelGrid = voxel_grid_build(xs, ys, zs, 1.0);",
            "let grid: VoxelGrid = VoxelGrid { x: 0u32 };",
        ),
        "NAME_UNKNOWN",
    );
}

#[test]
fn spatial_checker_holds_all_read_loans_until_scalar_arguments_finish() {
    let source = format!(
        "fn consume(g: take VoxelGrid) -> f32 {{ return 1.0; }} {}",
        QUERIES.replace("zs, 1.0, 1u32", "zs, consume(move grid), 1u32")
    );
    rejects(&source, "OWN_BORROW_CONFLICT");
    rejects(
        &QUERIES.replace("knn(grid, xs", "knn(move grid, xs"),
        "OWN_BORROW_SOURCE",
    );
}

#[test]
fn spatial_reference_and_mir_execute_the_checked_surface() {
    let program = invar::front::parse(QUERIES.as_bytes()).unwrap();
    let checked = invar::check::check(program, true).unwrap();
    let mir = invar::mir::lower(&checked).unwrap();
    assert_eq!(
        invar::mir_eval::run(&mir, 10_000).unwrap().value,
        invar::eval::Value::I32(0)
    );
    assert_eq!(
        invar::eval::run(&checked, 10_000).unwrap().value,
        invar::eval::Value::I32(0)
    );
    let result = invoke(QUERIES, "run");
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stdout)
    );
}

#[test]
fn spatial_record_fields_reuse_affine_array_rules_and_allow_mutable_outputs() {
    let source = QUERIES
        .replace(
            "let counts = move near.counts",
            "var counts = move near.counts",
        )
        .replace("return 0;", "counts[0u32] = 0u32; return 0;");
    assert!(check(&source).status.success());
    rejects(
        &QUERIES.replace("return 0;", "let duplicate = counts; return 0;"),
        "OWN_MOVE_REQUIRED",
    );
    rejects(
        &QUERIES.replace(
            "return 0;",
            "let moved = move counts; let bad = counts[0u32]; return 0;",
        ),
        "OWN_USE_AFTER_MOVE",
    );
    rejects(
        &QUERIES.replace("return 0;", "let again = move near.counts; return 0;"),
        "OWN_USE_AFTER_MOVE",
    );
}

#[test]
fn spatial_checker_rejects_tuples_and_reserved_builtin_types() {
    for declaration in [
        "enum VoxelGrid { Empty }",
        "struct Neighbors { x: u32 }",
        "enum Knn { Empty }",
    ] {
        rejects(&format!("{declaration} {QUERIES}"), "NAME_DUPLICATE");
    }
    rejects(
        &QUERIES.replace("let near: Neighbors", "let (counts, idx)"),
        "PARSE_EXPECTED_TOKEN",
    );
    rejects(
        &QUERIES.replace(
            "voxel_grid_build(xs, ys, zs, 1.0)",
            "voxel_grid_build(move xs, ys, zs, 1.0)",
        ),
        "OWN_BORROW_SOURCE",
    );
    rejects(&QUERIES.replace("VoxelGrid", "Grid"), "NAME_UNKNOWN");
    rejects(
        &QUERIES.replace("grid_excluded(grid)", "grid_excluded(xs)"),
        "TYPE_MISMATCH",
    );
    rejects(
        &QUERIES.replace("grid_excluded(grid)", "grid_excluded(move grid)"),
        "OWN_BORROW_SOURCE",
    );
}

#[test]
fn spatial_records_can_cross_function_boundaries_and_construct_owned_fields() {
    let source = format!(
        "fn pass(n: take Neighbors) -> Neighbors {{ return move n; }} \
         fn read_grid(g: read VoxelGrid) -> u32 {{ return grid_excluded(g); }} {}",
        QUERIES.replace("return 0;", "let owned = Neighbors { counts: move counts, idx: move radius_idx }; let moved = pass(move owned); return 0;")
    );
    assert!(check(&source).status.success());
    rejects(
        "fn wrong(xs: read Array<f64>, ys: read Array<f32>, zs: read Array<f32>) -> VoxelGrid { return voxel_grid_build(xs, ys, zs, 1.0); } fn main() -> i32 { return 0; }",
        "TYPE_MISMATCH",
    );
    rejects(
        &QUERIES.replace("return 0;", "let wrong: Array<f32> = move idx; return 0;"),
        "TYPE_MISMATCH",
    );
    rejects(
        &QUERIES.replace(
            "voxel_grid_build(xs, ys, zs, 1.0)",
            "voxel_grid_build(filled_u32(2u32, 0u32), ys, zs, 1.0)",
        ),
        "TYPE_MISMATCH",
    );
}

#[test]
fn spatial_mir_rejects_forged_layouts_and_loans() {
    use invar::{
        front::{Mode, Type},
        mir::{self, Operation},
    };
    let checked =
        invar::check::check(invar::front::parse(QUERIES.as_bytes()).unwrap(), true).unwrap();
    let program = mir::lower(&checked).unwrap();
    let mut malformed = program.clone();
    let field = &mut malformed
        .records
        .iter_mut()
        .find(|r| r.name == "Neighbors")
        .unwrap()
        .fields[0];
    field.ty = Type::ArrayF32;
    assert!(mir::verify(&malformed).is_err());
    let mut malformed = program.clone();
    malformed
        .records
        .iter_mut()
        .find(|r| r.name == "Neighbors")
        .unwrap()
        .name = "VoxelGrid".into();
    assert!(mir::verify(&malformed).is_err());
    let mut malformed = program.clone();
    let loan = malformed
        .functions
        .iter_mut()
        .flat_map(|f| &mut f.blocks)
        .flat_map(|b| &mut b.instructions)
        .find_map(|i| match &mut i.kind {
            Operation::ArrayRef { loan: mode, .. } if *mode == Some(Mode::Read) => Some(mode),
            _ => None,
        })
        .unwrap();
    *loan = None;
    assert!(mir::verify(&malformed).is_err());
    let mut malformed = program.clone();
    let loan = malformed
        .functions
        .iter_mut()
        .flat_map(|f| &mut f.blocks)
        .flat_map(|b| &mut b.instructions)
        .find_map(|i| match &mut i.kind {
            Operation::RecordRef { loan: mode, .. } if *mode == Some(Mode::Read) => Some(mode),
            _ => None,
        })
        .unwrap();
    *loan = Some(Mode::Mut);
    assert!(mir::verify(&malformed).is_err());
}

#[test]
fn spatial_builtin_records_preserve_the_user_record_limit() {
    let mut source = (0..128)
        .map(|i| format!("struct Record{i} {{ value: u32 }}\n"))
        .collect::<String>();
    source.push_str(QUERIES);
    let checked =
        invar::check::check(invar::front::parse(source.as_bytes()).unwrap(), true).unwrap();
    let program = invar::mir::lower(&checked).unwrap();
    assert_eq!(program.records.len(), 130);
    let mut too_many = program.clone();
    let mut extra = too_many
        .records
        .iter()
        .find(|r| r.name == "Record0")
        .unwrap()
        .clone();
    extra.name = "Record128".into();
    too_many.records.push(extra);
    assert_eq!(
        invar::mir::verify(&too_many).unwrap_err().code,
        "RESOURCE_LIMIT"
    );
}

#[test]
fn spatial_unreached_calls_do_not_claim_execution() {
    let source = "fn latent(xs: read Array<f32>) -> u32 { let grid = voxel_grid_build(xs,xs,xs,1.0); return grid_excluded(grid); } fn main() -> i32 { return 0; }";
    let checked =
        invar::check::check(invar::front::parse(source.as_bytes()).unwrap(), true).unwrap();
    let mir = invar::mir::lower(&checked).unwrap();
    let run = invar::schedule::run_source(source.as_bytes(), &mir, 1000, 4).unwrap();
    assert_eq!(run.plan, invar::plan::PlanKind::Serial);
    assert_eq!(run.observation.result, Ok(invar::eval::Value::I32(0)));
    assert!(run.executed.is_empty());
    assert_eq!(run.joined, 0);
}

#[test]
fn spatial_result_declarations_do_not_pollute_unrelated_programs() {
    let source = b"fn main() -> i32 { let value = 1.0 + 2.0; return 0; }";
    let checked = invar::check::check(invar::front::parse(source).unwrap(), true).unwrap();
    assert!(checked.program.records.is_empty());
    assert!(invar::mir::lower(&checked).unwrap().records.is_empty());
}

#[test]
fn spatial_cpu_worker_failures_join_before_refusing_publication() {
    use invar::{
        eval::{EvalError, Value},
        schedule::{self, Fault, Hooks},
    };
    use std::sync::atomic::Ordering;
    let source = "fn main() -> i32 { let p: Array<f32> = [0.0]; let grid = voxel_grid_build(p,p,p,1.0); let q = filled_f32(129u32,0.0); let near = radius_neighbors(grid,q,q,q,1.0,1u32); return 0; }";
    let checked =
        invar::check::check(invar::front::parse(source.as_bytes()).unwrap(), true).unwrap();
    let mir = invar::mir::lower(&checked).unwrap();
    assert_eq!(
        invar::eval::observe(&checked, 10000).result,
        Ok(Value::I32(0))
    );
    for (fault, code) in [
        (Fault::Spawn, "CPU_SPAWN"),
        (Fault::Join, "CPU_JOIN"),
        (Fault::Panic, "CPU_WORKER_PANIC"),
        (Fault::Allocation, "RESOURCE_LIMIT"),
        (Fault::Overlap, "CPU_CHUNK_OVERLAP"),
        (Fault::Timeout, "CPU_TIMEOUT"),
        (Fault::Cancel, "CPU_CANCELLED"),
    ] {
        let hooks = Hooks {
            fault,
            ..Hooks::default()
        };
        let run =
            schedule::run_source_with_hooks(source.as_bytes(), &mir, 10000, 4, &hooks).unwrap();
        assert!(
            matches!(&run.observation.result,Err(EvalError::Tool {code:actual,..}) if *actual==code)
        );
        assert!(run.observation.cleanup_trace.is_empty());
        assert_eq!(
            run.executed.len(),
            1,
            "failed query must not publish an execution receipt"
        );
        assert_eq!(
            hooks.started.load(Ordering::SeqCst),
            hooks.joined.load(Ordering::SeqCst)
        );
        assert_eq!(run.joined, hooks.joined.load(Ordering::SeqCst));
    }
}
