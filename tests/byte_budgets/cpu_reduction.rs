//! #368: CPU reductions, pair reductions and scans follow the live element policy, not the
//! old one million element cap. One element past the old cap each loop is published from CPU
//! workers at every worker count, with result, steps and cleanup identical to serial MIR and
//! the reference engine.
use invar::{check::check, eval, front::parse, mir, mir_eval, plan::PlanKind, schedule};

const PAST_OLD_CAP: u32 = 1_000_001;
const FUEL: u64 = 400_000_000;

fn affine(len: u32) -> u32 {
    (0..len).fold(17u32, |s, _| s.wrapping_mul(3).wrapping_add(1))
}

fn check_published(source: &str, expected: PlanKind) {
    let checked = check(parse(source.as_bytes()).unwrap(), true).unwrap();
    let program = mir::lower(&checked).unwrap();
    let reference = eval::observe(&checked, FUEL);
    assert_eq!(reference.result, Ok(eval::Value::I32(0)), "{source}");
    let serial = mir_eval::observe(&program, FUEL);
    assert_eq!(serial.result, reference.result);
    assert_eq!(serial.steps_used, reference.steps_used);
    for workers in [1, 2, 7, 64] {
        let run = schedule::run_source(source.as_bytes(), &program, FUEL, workers).unwrap();
        assert_eq!(run.observation.result, serial.result, "workers={workers}");
        assert_eq!(
            run.observation.steps_used, serial.steps_used,
            "workers={workers}"
        );
        assert_eq!(run.observation.cleanup_trace, serial.cleanup_trace);
        assert_eq!(run.plan, expected, "workers={workers}");
        assert_eq!(run.fallback_reason, None, "workers={workers}");
        assert_eq!(run.executed.len(), 1, "workers={workers}");
        assert!(run.joined > 0 || workers == 1, "workers={workers}");
    }
}

#[test]
fn cpu_reduction_past_the_old_element_cap_runs_in_parallel() {
    let sum = PAST_OLD_CAP.wrapping_mul(5);
    check_published(
        &format!(
            "fn main() -> i32 {{ let xs: Array<u32> = filled_u32({PAST_OLD_CAP}u32, 5u32); var acc: u32 = 0u32; \
             for i in 0u32 .. len(xs) {{ acc = wrapping_add(acc, xs[i]); }} if acc == {sum}u32 {{ return 0; }} return 1; }}"
        ),
        PlanKind::Reduction,
    );
}

#[test]
fn cpu_pair_reduction_past_the_old_element_cap_runs_in_parallel() {
    let (sa, sb) = (
        3u32.wrapping_pow(PAST_OLD_CAP).wrapping_mul(5),
        affine(PAST_OLD_CAP),
    );
    check_published(
        &format!(
            "fn main() -> i32 {{ let aa: Array<u32> = filled_u32({PAST_OLD_CAP}u32, 3u32); let bb: Array<u32> = filled_u32({PAST_OLD_CAP}u32, 1u32); \
             var sa: u32 = 5u32; var sb: u32 = 17u32; for i in 0u32 .. len(aa) {{ sa = wrapping_mul(aa[i], sa); sb = wrapping_add(wrapping_mul(aa[i], sb), bb[i]); }} \
             if sa == {sa}u32 and sb == {sb}u32 {{ return 0; }} return 1; }}"
        ),
        PlanKind::Reduction,
    );
}

#[test]
fn cpu_affine_scan_past_the_old_element_cap_runs_in_parallel() {
    let (last, first) = (affine(PAST_OLD_CAP), affine(1));
    check_published(
        &format!(
            "fn main() -> i32 {{ let aa: Array<u32> = filled_u32({PAST_OLD_CAP}u32, 3u32); let bb: Array<u32> = filled_u32({PAST_OLD_CAP}u32, 1u32); \
             var state: u32 = 17u32; var out: Array<u32> = filled_u32(len(aa), 0u32); \
             for i in 0u32 .. len(aa) {{ state = wrapping_add(wrapping_mul(aa[i], state), bb[i]); out[i] = state; }} \
             if out[0u32] != {first}u32 {{ return 1; }} if out[{}u32] != {last}u32 {{ return 1; }} return 0; }}",
            PAST_OLD_CAP - 1
        ),
        PlanKind::Scan,
    );
}
