use super::*;
use invar::summary_source::host;

fn same(actual: &eval::Observation, expected: &eval::Observation) {
    assert_eq!(actual.result, expected.result);
    assert_eq!(actual.steps_used, expected.steps_used);
    assert_eq!(actual.cleanup_trace, expected.cleanup_trace);
}

#[test]
fn issue_m4_07_reached_summary_cpu_publication_matches_source_all_output_modes() {
    let mut fixtures = 0;
    let mut completed = 0;
    for family in families() {
        for prefix in [false, true] {
            for steps in 0..=7 {
                let f = source(family, prefix);
                let (coeff, seed) = data(family, steps);
                let all = independent(family, &coeff, &seed);
                let expected = if prefix {
                    all.iter().flatten().copied().collect::<Vec<_>>()
                } else {
                    all.last().unwrap_or(&seed).clone()
                };
                let s = caller(&f, family, steps, &coeff, &seed, Some(&expected));
                let reference = eval::observe(&checked(&s), 100_000);
                let run = host::run_source(s.as_bytes(), 100_000, budget()).unwrap();
                same(&run.observation, &reference);
                assert_eq!(
                    run.completed,
                    usize::from(steps != 0),
                    "{family:?} {prefix} {steps}: {:?}",
                    run.refusals
                );
                assert!(run.refusals.is_empty());
                fixtures += 1;
                completed += run.completed;
            }
        }
    }
    assert_eq!(fixtures, 96);
    assert_eq!(completed, 84);
    println!("reached_source_summary_fixtures={fixtures} actual_completed_regions={completed}");
}

#[test]
fn issue_m4_07_reached_summary_every_source_fuel_prefix_retains_errors_and_cleanup() {
    let mut prefixes = 0;
    let mut admitted = 0;
    for family in [Family::Table(2), Family::Matrix(2)] {
        for prefix in [false, true] {
            let (coeff, seed) = data(family, 3);
            let s = caller(&source(family, prefix), family, 3, &coeff, &seed, None);
            let p = checked(&s);
            let full = eval::observe(&p, 100_000);
            for fuel in 1..=full.steps_used + 1 {
                let reference = eval::observe(&p, fuel);
                let run = host::run_source(s.as_bytes(), fuel, budget()).unwrap();
                same(&run.observation, &reference);
                prefixes += 1;
                admitted += run.completed;
            }
        }
    }
    assert!(prefixes > 500);
    assert!(admitted > 0);
    println!("reached_source_summary_fuel_prefixes={prefixes} actual_completed_regions={admitted}");
}

#[test]
fn issue_m4_07_reached_summary_guard_refusals_keep_original_source_behavior() {
    let f = source(Family::Table(2), true);
    let (coeff, seed) = data(Family::Table(2), 3);
    let valid = caller(&f, Family::Table(2), 3, &coeff, &seed, None);
    for b in [
        Budget {
            max_steps: 0,
            ..budget()
        },
        Budget {
            max_bytes: 0,
            ..budget()
        },
        Budget {
            max_composition_ops: 0,
            ..budget()
        },
    ] {
        let run = host::run_source(valid.as_bytes(), 100_000, b).unwrap();
        same(&run.observation, &eval::observe(&checked(&valid), 100_000));
        assert_eq!(run.completed, 0);
        assert!(run.refusals.contains(&"summary_admission"));
    }
    for s in [
        caller(
            &source(Family::Table(2), false),
            Family::Table(2),
            1,
            &[2, 0],
            &[0],
            None,
        ),
        caller(
            &source(Family::Table(2), false),
            Family::Table(2),
            2,
            &[2, 0, 0, 1],
            &[0],
            None,
        ),
        caller(
            &source(Family::Table(2), false),
            Family::Table(2),
            1,
            &[],
            &[0],
            None,
        ),
        caller(
            &source(Family::Table(2), false),
            Family::Table(2),
            1,
            &[0, 1],
            &[2],
            None,
        ),
        format!(
            "{} fn main() -> i32 {{ let shared: Array<u32> = [3u32]; let answer = run(shared,shared,1u32); return 7; }}",
            source(Family::Matrix(1), false)
        ),
        format!(
            "{} fn main() -> i32 {{ if false {{ let input: Array<u32> = [0u32,1u32]; let answer = run(input,0u32,1u32); }} return 7; }}",
            source(Family::Table(2), false)
        ),
        format!(
            "{} fn main() -> i32 {{ let earlier = 1 / 0; let input: Array<u32> = [0u32,1u32]; let answer = run(input,0u32,1u32); return 7; }}",
            source(Family::Table(2), false)
        ),
    ] {
        let p = checked(&s);
        let run = host::run_source(s.as_bytes(), 100_000, budget()).unwrap();
        same(&run.observation, &eval::observe(&p, 100_000));
        assert_eq!(run.completed, 0);
    }
    let invalid = valid.replace("state = coeff[i * 2u32 + state]", "state = true");
    assert!(matches!(
        host::run_source(invalid.as_bytes(), 100_000, budget()),
        Err(Error::Source(_))
    ));
}

#[test]
fn issue_m4_07_reached_summary_repeated_calls_use_current_inputs_and_seeds() {
    let s = format!(
        "{} fn main() -> i32 {{ var input: Array<u32> = [1u32,0u32,0u32,1u32]; let a = run(input,0u32,2u32); input[0u32] = 0u32; let b = run(input,0u32,2u32); let c = run(input,1u32,2u32); if a != 1u32 {{ return 0; }} if b != 0u32 {{ return 0; }} if c != 0u32 {{ return 0; }} return 7; }}",
        source(Family::Table(2), false)
    );
    let reference = eval::observe(&checked(&s), 100_000);
    assert_eq!(reference.result, Ok(eval::Value::I32(7)));
    let run = host::run_source(s.as_bytes(), 100_000, budget()).unwrap();
    same(&run.observation, &reference);
    assert_eq!(run.completed, 3);
    assert!(run.refusals.is_empty());
}
