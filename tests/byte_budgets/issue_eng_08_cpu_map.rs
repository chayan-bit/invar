//! #57 part B: the CPU map pass is bounded by host bytes, not by the old one million element
//! cap. A map one element past the old cap runs on `cpu-parallel` at every worker count with
//! results, steps and cleanup identical to the serial MIR engine and the reference engine.
use invar::{check::check, eval, front::parse, mir, mir_eval, plan::PlanKind, schedule};

const PAST_OLD_CAP: usize = 1_000_001;
const FUEL: u64 = 400_000_000;

/// `stages` chained maps over `len` elements; the program returns 0 when the last element is right.
fn chain(len: usize, stages: usize) -> String {
    let mut source = format!("fn main() -> i32 {{ let xs: Array<u32> = filled_u32({len}, 5u32); ");
    for stage in 0..stages {
        let input = if stage == 0 {
            "xs".to_owned()
        } else {
            format!("stable{}", stage - 1)
        };
        source.push_str(&format!(
            "var out{stage}: Array<u32> = filled_u32(len({input}), 0u32); \
             for j{stage} in 0u32 .. len({input}) {{ out{stage}[j{stage}] = wrapping_add({input}[j{stage}], 7u32); }} "
        ));
        if stage + 1 < stages {
            source.push_str(&format!(
                "let stable{stage}: Array<u32> = move out{stage}; "
            ));
        }
    }
    source.push_str(&format!(
        "if out{}[{}u32] != {}u32 {{ return 1; }} return 0; }}",
        stages - 1,
        len - 1,
        5 + 7 * stages
    ));
    source
}

fn check_len(len: usize, stages: usize, expected: PlanKind) -> usize {
    let source = chain(len, stages);
    let checked = check(parse(source.as_bytes()).unwrap(), true).unwrap();
    let program = mir::lower(&checked).unwrap();
    let reference = eval::observe(&checked, FUEL);
    assert_eq!(reference.result, Ok(eval::Value::I32(0)), "len={len}");
    let serial = mir_eval::observe(&program, FUEL);
    assert_eq!(serial.result, reference.result, "len={len}");
    let mut cases = 0;
    for workers in [1, 2, 7, 64] {
        let run = schedule::run_source(source.as_bytes(), &program, FUEL, workers).unwrap();
        assert_eq!(run.observation.result, serial.result, "workers={workers}");
        assert_eq!(run.observation.cleanup_trace, serial.cleanup_trace);
        assert_eq!(run.plan, expected, "len={len} workers={workers}");
        assert_eq!(run.fallback_reason, None, "len={len} workers={workers}");
        assert_eq!(run.executed.len(), stages);
        assert!(run.joined > 0 || workers == 1, "workers={workers}");
        cases += 1;
    }
    cases
}

#[test]
fn issue_eng_08_cpu_map_past_the_old_element_cap_runs_in_parallel() {
    let cases = check_len(PAST_OLD_CAP, 1, PlanKind::Map);
    assert_eq!(cases, 4);
}

#[test]
fn issue_eng_08_cpu_fused_chain_past_the_old_element_cap_runs_in_parallel() {
    let cases = check_len(PAST_OLD_CAP, 3, PlanKind::FusedMap);
    assert_eq!(cases, 4);
}
