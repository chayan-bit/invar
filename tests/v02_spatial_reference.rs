//! Source-to-reference differential against the independent all-pairs oracle.
#[path = "support/language_limits.rs"]
mod language_limits;
#[path = "support/spatial_oracle.rs"]
#[allow(dead_code)]
mod oracle;

use invar::{check, eval, front, numeric_contract::F32Bits as F, ownership};
use oracle::Cloud;
use std::process::Command;

fn checked(source: &str) -> check::CheckedProgram {
    let checked = check::check(front::parse(source.as_bytes()).unwrap(), true).unwrap();
    ownership::check(&checked).unwrap();
    checked
}
fn literal(v: F) -> String {
    if v.is_nan() {
        "(0.0 / 0.0)".into()
    } else if v.is_infinite() {
        if v.is_negative() {
            "(-1.0 / 0.0)".into()
        } else {
            "(1.0 / 0.0)".into()
        }
    } else {
        format!("{:.149}", f32::from_bits(v.to_bits()))
    }
}
fn floats(values: &[F]) -> String {
    values
        .iter()
        .map(|&v| literal(v))
        .collect::<Vec<_>>()
        .join(",")
}
fn uints(values: &[u32]) -> String {
    values
        .iter()
        .map(|v| format!("{v}u32"))
        .collect::<Vec<_>>()
        .join(",")
}
fn cloud(points: &[[F; 3]]) -> [Vec<F>; 3] {
    std::array::from_fn(|axis| points.iter().map(|p| p[axis]).collect())
}
fn view(c: &[Vec<F>; 3]) -> Cloud<'_> {
    Cloud::new(&c[0], &c[1], &c[2]).unwrap()
}
fn f(v: f32) -> F {
    F::from_bits(v.to_bits())
}

fn differential(points: &[[F; 3]], queries: &[[F; 3]], cell: F, radius: F, stride: u32) -> String {
    let p = cloud(points);
    let q = cloud(queries);
    oracle::validate_grid(view(&p), cell).unwrap();
    let (counts, radius_idx) =
        oracle::radius_neighbors(view(&p), view(&q), radius, stride).unwrap();
    let (idx, d2) = oracle::knn(view(&p), view(&q), stride).unwrap();
    let mut source = String::from("fn main() -> i32 {\n");
    for (prefix, arrays) in [("p", &p), ("q", &q)] {
        for (axis, values) in arrays.iter().enumerate() {
            source.push_str(&format!(
                "let {prefix}{axis}: Array<f32> = [{}];\n",
                floats(values)
            ));
        }
    }
    source.push_str(&format!(r#"
        let grid = voxel_grid_build(p0,p1,p2,{});
        if grid_excluded(grid) != {}u32 {{ return 1; }}
        let near = radius_neighbors(grid,q0,q1,q2,{},{}u32);
        let counts = move near.counts;
        let radius_idx = move near.idx;
        let expected_counts: Array<u32> = [{}];
        let expected_radius_idx: Array<u32> = [{}];
        for c in 0u32..len(counts) {{ if counts[c] != expected_counts[c] {{ return 2; }} }}
        for r in 0u32..len(radius_idx) {{ if radius_idx[r] != expected_radius_idx[r] {{ return 3; }} }}
        let nearest = knn(grid,q0,q1,q2,{}u32);
        let idx = move nearest.idx;
        let d2 = move nearest.d2;
        let expected_idx: Array<u32> = [{}];
        let expected_d2: Array<f32> = [{}];
        if len(counts) != len(expected_counts) or len(radius_idx) != len(expected_radius_idx)
            or len(idx) != len(expected_idx) or len(d2) != len(expected_d2) {{ return 6; }}
        for k in 0u32..len(idx) {{
            if idx[k] != expected_idx[k] {{ return 4; }}
            if d2[k] != expected_d2[k] {{ return 5; }}
        }}
        return 7;
    }}"#, literal(cell), oracle::grid_excluded(view(&p)), literal(radius), stride,
        uints(&counts), uints(&radius_idx), stride, uints(&idx), floats(&d2)));
    let checked = checked(&source);
    let observation = eval::observe(&checked, 100_000);
    assert_eq!(observation.result, Ok(eval::Value::I32(7)), "{source}");
    let mir = invar::mir::lower(&checked).unwrap();
    let serial = invar::mir_eval::observe(&mir, 100_000);
    assert_eq!(serial.result, observation.result);
    assert_eq!(serial.steps_used, observation.steps_used);
    assert_eq!(serial.cleanup_trace, observation.cleanup_trace);
    for workers in [1, 2, 4] {
        let parallel =
            invar::schedule::run_source(source.as_bytes(), &mir, 100_000, workers).unwrap();
        assert_eq!(
            parallel.observation.result, observation.result,
            "workers={workers}"
        );
        assert_eq!(parallel.observation.steps_used, observation.steps_used);
        assert_eq!(
            parallel.observation.cleanup_trace,
            observation.cleanup_trace
        );
        if queries.len() >= 64 && workers > 1 {
            assert!(parallel.joined > 0);
        }
    }
    source
}

#[test]
fn spatial_reference_oracle_random_clustered_and_duplicate_clouds() {
    let mut state = 0x5350415449414cu64;
    for round in 0..9 {
        let mut next = || {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let scale = if round % 2 == 0 { 0.25 } else { 16.0 };
            f(((state >> 32) as i32 % 200) as f32 * scale)
        };
        let mut points = (0..32)
            .map(|_| [next(), next(), next()])
            .collect::<Vec<_>>();
        points.extend_from_within(..4);
        let queries = (0..8).map(|_| [next(), next(), next()]).collect::<Vec<_>>();
        differential(&points, &queries, f(1.0), f(20.0), 5);
    }
}

#[test]
fn spatial_cpu_ragged_query_rows_join_real_workers() {
    let points = (0..16)
        .map(|i| [f(i as f32), F::POS_ZERO, F::POS_ZERO])
        .collect::<Vec<_>>();
    let queries = (0..129)
        .map(|i| [f((i % 21) as f32), F::POS_ZERO, F::POS_ZERO])
        .collect::<Vec<_>>();
    differential(&points, &queries, f(1.0), f(2.0), 5);
}

#[test]
fn spatial_reference_oracle_filtering_padding_and_radius_rules() {
    let points = [
        [f(1.0), f(0.0), f(0.0)],
        [f(-1.0), f(0.0), f(0.0)],
        [F::NEG_ZERO; 3],
        [F::NAN; 3],
        [F::INFINITY; 3],
        [F::NEG_INFINITY; 3],
        // Whole-point C6 filtering precedes C7, even when x alone would overflow.
        [f(f32::MAX), F::POS_ZERO, F::NAN],
    ];
    let queries = [[F::POS_ZERO; 3], [F::NAN; 3], [F::INFINITY; 3]];
    for radius in [
        F::POS_ZERO,
        F::NEG_ZERO,
        f(1.0),
        f(-1.0),
        F::NAN,
        F::INFINITY,
        F::NEG_INFINITY,
    ] {
        for stride in [0, 1, 8] {
            differential(&points, &queries, f(0.1), radius, stride);
        }
    }
    differential(&[], &queries, f(1.0), f(1.0), 4);
    differential(&points, &[], f(1.0), f(1.0), 4);
    differential(&[[f(1.0); 3]; 5], &[[f(1.0); 3]; 2], f(1.0), f(0.0), 3);
}

#[test]
fn spatial_reference_oracle_strict_rounding_underflow_and_infinite_distances() {
    differential(
        &[[
            F::from_bits(0x3f800001),
            F::from_bits(0x39800000),
            F::POS_ZERO,
        ]],
        &[[F::POS_ZERO; 3]],
        f(1.0),
        f(2.0),
        2,
    );
    differential(
        &[[f(4096.0), f(1.0), f(1.0)], [f(4096.0), f(0.0), f(0.0)]],
        &[[f(0.0); 3]],
        f(1.0),
        f(4096.0),
        2,
    );
    // Squared distance rounds to zero despite a separation of 2^20 cells.
    differential(
        &[[F::from_bits(0x17800000), F::POS_ZERO, F::POS_ZERO]],
        &[[F::POS_ZERO; 3]],
        F::from_bits(0x0d800000),
        F::POS_ZERO,
        2,
    );
    differential(
        &[[F::from_bits(0x1a800000), F::POS_ZERO, F::POS_ZERO]],
        &[[F::POS_ZERO; 3]],
        F::from_bits(0x1a800000),
        F::POS_ZERO,
        2,
    );
    differential(
        &[[f(f32::MAX), F::POS_ZERO, F::POS_ZERO]],
        &[[f(-f32::MAX), F::POS_ZERO, F::POS_ZERO]],
        f(f32::MAX),
        F::INFINITY,
        2,
    );
}

fn trap_source(body: &str, category: eval::TrapCategory, marked: &str) {
    let source = format!("fn main() -> i32 {{ {body} return 0; }}");
    let outcome = eval::observe(&checked(&source), 100_000);
    let start = source.find(marked).unwrap();
    assert_eq!(
        outcome.result,
        Err(eval::EvalError::Trap {
            category,
            span: front::Span {
                start,
                end: start + marked.len()
            }
        })
    );
    assert!(
        outcome.cleanup_trace.is_empty(),
        "trap performs no language cleanup"
    );
    let checked = checked(&source);
    let mir = invar::mir::lower(&checked).unwrap();
    for workers in [1, 2, 4] {
        let actual =
            invar::schedule::run_source(source.as_bytes(), &mir, 100_000, workers).unwrap();
        assert_eq!(actual.observation.result, outcome.result);
        assert_eq!(actual.observation.steps_used, outcome.steps_used);
        assert_eq!(actual.observation.cleanup_trace, outcome.cleanup_trace);
    }
    #[cfg(feature = "native")]
    {
        let (status, document) = invar::native::run_program(&mir, 100_000).unwrap();
        assert_eq!(status, 3, "{document}");
        let expected = outcome.result.as_ref().unwrap_err().to_json("fixture.inv");
        assert_eq!(document["trap"], expected["trap"]);
        assert_eq!(document["cleanup_trace"], serde_json::json!([]));
        assert_eq!(document["steps_used"], outcome.steps_used);
    }
}

#[test]
fn spatial_reference_c7_c9_and_first_source_trap() {
    use eval::TrapCategory::{
        DivisionByZero as DivZero, IndexOutOfBounds as Bounds, IntegerOverflow as Overflow,
    };
    let prefix = "let xs: Array<f32> = [0.0]; let empty: Array<f32> = [];";
    let call = "voxel_grid_build(xs,empty,xs,0.0)";
    trap_source(&format!("{prefix} let g = {call};"), Bounds, call);
    let call = "voxel_grid_build(xs,xs,xs,0.0)";
    trap_source(&format!("{prefix} let g = {call};"), Overflow, call);
    for cell in ["-1.0", "-0.0", "(0.0/0.0)", "(1.0/0.0)"] {
        let call = format!("voxel_grid_build(empty,empty,empty,{cell})");
        trap_source(&format!("{prefix} let g = {call};"), Overflow, &call);
    }
    for coord in ["2147483648.0", "-2147483904.0"] {
        let call = "voxel_grid_build(xs,xs,xs,1.0)";
        trap_source(
            &format!("let xs: Array<f32> = [{coord}]; let g = {call};"),
            Overflow,
            call,
        );
    }
    let prefix =
        format!("{prefix} let g = voxel_grid_build(xs,xs,xs,1.0); let q: Array<f32> = [0.0,0.0];");
    for call in [
        "knn(g,q,xs,q,4294967295u32)",
        "radius_neighbors(g,q,xs,q,0.0,4294967295u32)",
    ] {
        trap_source(&format!("{prefix} let r = {call};"), Bounds, call);
    }
    for call in [
        "knn(g,q,q,q,4294967295u32)",
        "radius_neighbors(g,q,q,q,(0.0/0.0),4294967295u32)",
    ] {
        trap_source(&format!("{prefix} let r = {call};"), Overflow, call);
    }
    // The scalar argument is demanded before the intrinsic validates shapes.
    trap_source(
        &format!("{prefix} let r = knn(g,q,xs,q,1u32/0u32);"),
        DivZero,
        "/",
    );
}

#[test]
fn spatial_reference_snapshot_moves_and_cleanup() {
    let source = r#"
        fn transfer(g: take VoxelGrid) -> VoxelGrid { return g; }
        fn main() -> i32 {
            var xs: Array<f32> = [0.0];
            let g = voxel_grid_build(xs,xs,xs,1.0);
            xs[0u32] = 100.0;
            let g2 = transfer(move g);
            let q: Array<f32> = [0.0];
            let r = radius_neighbors(g2,q,q,q,0.0,1u32);
            let counts = move r.counts;
            if counts[0u32] == 1u32 { return 7; }
            return 0;
        }"#;
    let observation = eval::observe(&checked(source), 10_000);
    assert_eq!(observation.result, Ok(eval::Value::I32(7)));
    assert_eq!(
        observation
            .cleanup_trace
            .iter()
            .filter(|e| e.name == "g2")
            .count(),
        1
    );
    assert!(!observation.cleanup_trace.iter().any(|e| e.name == "g"));
    let mir = invar::mir::lower(&checked(source)).unwrap();
    let serial = invar::mir_eval::observe(&mir, 10_000);
    assert_eq!(serial.result, observation.result);
    assert_eq!(serial.cleanup_trace, observation.cleanup_trace);
    for workers in [1, 2, 4] {
        let run = invar::schedule::run_source(source.as_bytes(), &mir, 10_000, workers).unwrap();
        assert_eq!(run.observation.result, observation.result);
        assert_eq!(run.observation.cleanup_trace, observation.cleanup_trace);
    }
    #[cfg(feature = "native")]
    {
        let (status, document) = invar::native::run_program(&mir, 10_000).unwrap();
        assert_eq!(status, 0, "{document}");
        assert_eq!(document["result"]["value"], 7);
        assert_eq!(document["steps_used"], observation.steps_used);
        let cleanup = observation
            .cleanup_trace
            .iter()
            .map(|e| serde_json::json!({"name":e.name,"length":e.length}))
            .collect::<Vec<_>>();
        assert_eq!(document["cleanup_trace"], serde_json::json!(cleanup));
    }
}

#[test]
fn spatial_reference_real_cli_demo() {
    let mut engines = vec!["reference", "cpu-parallel"];
    if cfg!(feature = "native") {
        engines.push("native-serial");
    }
    for engine in engines {
        let mut command = Command::new(env!("CARGO_BIN_EXE_invar"));
        command.args([
            "run",
            "demo/spatial-checker.inv",
            "--engine",
            engine,
            "--format",
            "json",
        ]);
        if engine == "cpu-parallel" {
            command.args(["--workers", "4"]);
        }
        let output = command.output().unwrap();
        assert!(
            output.status.success(),
            "{engine}: {}{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
        let report: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
        assert_eq!(report["status"], "ok");
        assert_eq!(report["result"]["value"], 3);
    }
}

#[cfg(feature = "native")]
#[test]
fn spatial_native_serial_differential() {
    let zero = [F::POS_ZERO; 3];
    let sources = [
        differential(
            &[
                [f(2.0), f(0.0), f(0.0)],
                [f(-1.0), f(0.0), f(0.0)],
                [f(1.0), f(0.0), f(0.0)],
                zero,
                [F::NAN; 3],
                [f(f32::MAX), F::POS_ZERO, F::NAN],
            ],
            &[zero, [F::INFINITY; 3]],
            f(0.1),
            f(2.0),
            6,
        ),
        differential(
            &[[
                F::from_bits(0x3f800001),
                F::from_bits(0x39800000),
                F::POS_ZERO,
            ]],
            &[zero],
            f(1.0),
            f(2.0),
            2,
        ),
        differential(
            &[[f(4096.0), f(1.0), f(1.0)]],
            &[zero],
            f(1.0),
            f(4096.0),
            2,
        ),
        differential(
            &[[F::from_bits(0x1a800000), F::POS_ZERO, F::POS_ZERO]],
            &[zero],
            F::from_bits(0x1a800000),
            F::POS_ZERO,
            2,
        ),
        differential(
            &[[f(f32::MAX), F::POS_ZERO, F::POS_ZERO]],
            &[[f(-f32::MAX), F::POS_ZERO, F::POS_ZERO]],
            f(f32::MAX),
            F::INFINITY,
            2,
        ),
        differential(&[], &[zero], f(1.0), F::NAN, 0),
        differential(&[zero; 4], &[zero], f(1.0), F::POS_ZERO, 1),
        differential(&[zero; 4], &[zero], f(1.0), f(-1.0), 1),
        differential(
            &[[F::from_bits(0x17800000), F::POS_ZERO, F::POS_ZERO]],
            &[zero],
            F::from_bits(0x0d800000),
            F::POS_ZERO,
            2,
        ),
    ];
    for source in sources {
        let checked = checked(&source);
        let expected = eval::observe(&checked, 100_000);
        let program = invar::mir::lower(&checked).unwrap();
        let (status, document) = invar::native::run_program(&program, 100_000).unwrap();
        assert_eq!(status, 0, "{document}");
        assert_eq!(document["result"]["value"], 7, "{document}");
        assert_eq!(document["steps_used"], expected.steps_used);
        let cleanup = expected
            .cleanup_trace
            .iter()
            .map(|e| serde_json::json!({"name":e.name,"length":e.length}))
            .collect::<Vec<_>>();
        assert_eq!(document["cleanup_trace"], serde_json::json!(cleanup));
    }
}

#[test]
fn spatial_result_limits_remain_tool_errors_before_filtering_or_work() {
    let base = "let p: Array<f32> = [0.0]; let grid = voxel_grid_build(p,p,p,1.0); let q: Array<f32> = [0.0,0.0];";
    // Two queries at this k need one more result slot than an array may hold.
    let over = invar::array_limits::MAX_ARRAY_LEN / 2 + 1;
    let knn_over = format!("knn(grid,q,q,q,{over}u32)");
    let radius_over = format!("radius_neighbors(grid,q,q,q,(0.0/0.0),{over}u32)");
    // Each result array is legal, but the pair does not fit the remaining live budget.
    let ballast = language_limits::padding(invar::array_limits::MAX_LIVE_ELEMENTS - 200_000);
    for (prefix, call) in [
        ("", knn_over.as_str()),
        ("", "knn(grid,p,p,p,4294967295u32)"),
        ("", radius_over.as_str()),
        (ballast.as_str(), "knn(grid,q,q,q,100000u32)"),
    ] {
        let source =
            format!("fn main() -> i32 {{ {prefix} {base} let result = {call}; return 0; }}");
        let program = checked(&source);
        let reference = eval::observe(&program, 10_000);
        assert!(matches!(
            &reference.result,
            Err(eval::EvalError::Tool {
                code: "ARRAY_SIZE_LIMIT",
                ..
            })
        ));
        assert!(reference.cleanup_trace.is_empty());
        let mir = invar::mir::lower(&program).unwrap();
        for workers in [1, 2, 4] {
            let run =
                invar::schedule::run_source(source.as_bytes(), &mir, 10_000, workers).unwrap();
            assert!(matches!(
                &run.observation.result,
                Err(eval::EvalError::Tool {
                    code: "ARRAY_SIZE_LIMIT",
                    ..
                })
            ));
            assert_eq!(run.observation.steps_used, reference.steps_used);
            assert!(run.observation.cleanup_trace.is_empty());
        }
        #[cfg(feature = "native")]
        {
            let (status, document) = invar::native::run_program(&mir, 10_000).unwrap();
            assert_eq!(status, 2, "{document}");
            assert_eq!(document["tool_code"], "ARRAY_SIZE_LIMIT");
            assert_eq!(document["steps_used"], reference.steps_used);
            assert_eq!(document["cleanup_trace"], serde_json::json!([]));
        }
    }
}
