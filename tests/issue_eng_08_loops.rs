//! #57 ENG-08 part B (loops): the typed-graph limits come from the analysis byte budget and the
//! chain and owner-walk caps no longer hide in `loops.rs`. The stack guard, the owner-walk bound
//! and the extractor's fused plan size (`typed_net::MAX_DAG_MAPS`) remain, and `src/loops.rs`
//! says why. Every test asserts a nonzero case count.
use invar::budget::DEFAULT_ANALYSIS_BYTES;
use invar::loops::{GRAPH_LIMITS, MAX_FUSED_STAGES, graph_limits};
use invar::{admission, check::check, eval, front::parse, mir, mir_eval, plan::PlanKind, schedule};

fn chain_source(stages: usize) -> String {
    let mut source = String::from("fn main() -> i32 { let xs: Array<u32> = [1u32,2u32,3u32]; ");
    for stage in 0..stages {
        let input = if stage == 0 {
            "xs".to_owned()
        } else {
            format!("stable{}", stage - 1)
        };
        source.push_str(&format!(
            "var out{stage}: Array<u32> = filled_u32(len({input}),0u32); \
             for j{stage} in 0u32 .. len({input}) {{ out{stage}[j{stage}] = wrapping_add({input}[j{stage}],1u32); }} "
        ));
        if stage + 1 < stages {
            source.push_str(&format!(
                "let stable{stage}: Array<u32> = move out{stage}; "
            ));
        }
    }
    source.push_str(&format!(
        "if out{}[0u32] != {}u32 {{ return 1; }} return 0; }}",
        stages - 1,
        stages + 1
    ));
    source
}

fn program_of(source: &str) -> (invar::check::CheckedProgram, mir::Program) {
    let checked = check(parse(source.as_bytes()).unwrap(), true).unwrap();
    let program = mir::lower(&checked).unwrap();
    (checked, program)
}

#[test]
fn issue_eng_08_graph_limits_follow_the_analysis_budget() {
    let mut cases = 0;
    assert_eq!(GRAPH_LIMITS, graph_limits(DEFAULT_ANALYSIS_BYTES));
    // The old limits were 64 nodes, 128 ports, 16 map operations and 1_000_000 elements.
    let limits = std::hint::black_box(GRAPH_LIMITS);
    assert!(limits.nodes > 64 * 1000 && limits.ports >= 2 * limits.nodes);
    assert!(limits.map_ops > 16 * 1000);
    assert_eq!(limits.array_len, invar::array_limits::MAX_ARRAY_LEN);
    cases += 1;
    let zero = graph_limits(0);
    assert_eq!((zero.nodes, zero.ports, zero.map_ops), (0, 0, 0));
    let mut previous = zero;
    for bytes in [1usize << 10, 1 << 20, DEFAULT_ANALYSIS_BYTES, usize::MAX] {
        let limits = graph_limits(bytes);
        assert!(limits.nodes >= previous.nodes && limits.map_ops >= previous.map_ops);
        assert_eq!(limits.depth, GRAPH_LIMITS.depth);
        previous = limits;
        cases += 1;
    }
    assert_eq!(cases, 5);
}

#[test]
fn issue_eng_08_chains_past_the_fused_plan_size_fuse_in_segments_with_identical_results() {
    let mut cases = 0;
    for stages in [MAX_FUSED_STAGES + 1, 20, 33, 64] {
        let source = chain_source(stages);
        let (checked, program) = program_of(&source);
        let report = admission::analyze(source.as_bytes(), &program).unwrap();
        assert_eq!(report.regions().len(), stages);
        // `extract_dag` holds at most MAX_FUSED_STAGES maps, so a longer run fuses in segments and
        // the remainder stays an ordinary map; no region is left serial.
        let kinds: Vec<_> = report.regions().iter().map(|r| r.plan_kind()).collect();
        assert!(
            kinds[..MAX_FUSED_STAGES]
                .iter()
                .all(|kind| *kind == PlanKind::FusedMap),
            "stages={stages}"
        );
        assert!(
            kinds
                .iter()
                .all(|kind| matches!(kind, PlanKind::FusedMap | PlanKind::Map)),
            "stages={stages}"
        );
        let reference = eval::observe(&checked, 1_000_000);
        assert_eq!(reference.result, Ok(eval::Value::I32(0)));
        let serial = mir_eval::observe(&program, 1_000_000);
        for workers in [1, 2, 7] {
            let run =
                schedule::run_source(source.as_bytes(), &program, 1_000_000, workers).unwrap();
            assert_eq!(run.observation.result, serial.result, "stages={stages}");
            assert_eq!(run.observation.steps_used, serial.steps_used);
            assert_eq!(run.observation.cleanup_trace, serial.cleanup_trace);
            assert_eq!(run.executed.len(), stages, "workers={workers}");
            cases += 1;
        }
    }
    assert_eq!(cases, 12);
}

#[test]
fn issue_eng_08_chains_past_the_stage_bound_stay_correct_and_do_not_overflow_the_stack() {
    let mut cases = 0;
    for stages in [MAX_FUSED_STAGES + 1, 200] {
        let source = chain_source(stages);
        let (_, program) = program_of(&source);
        let serial = mir_eval::observe(&program, 10_000_000);
        let run = schedule::run_source(source.as_bytes(), &program, 10_000_000, 2).unwrap();
        assert_eq!(run.observation.result, serial.result, "stages={stages}");
        assert_eq!(run.observation.result, Ok(eval::Value::I32(0)));
        cases += 1;
    }
    assert_eq!(cases, 2);
}

#[test]
fn issue_eng_08_deleted_loop_caps_no_longer_exist() {
    let source = include_str!("../src/loops.rs");
    let mut cases = 0;
    for deleted in [
        "MAX_CHAIN",
        "nodes: 64",
        "map_ops: 16",
        "array_len: 1_000_000",
    ] {
        assert!(!source.contains(deleted), "{deleted}");
        cases += 1;
    }
    let chain = include_str!("../src/loops/chain.rs");
    assert!(!chain.contains("2..=16"));
    cases += 1;
    assert_eq!(cases, 5);
}
