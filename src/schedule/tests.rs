//! Attribution of offered steps when MIR refuses one partway through a run.
use super::*;
use crate::{check, front};

const TWO_LOOPS: &str = "fn main() -> i32 { var a = filled_u32(4u32, 0u32); for i in 0u32..4u32 { a[i] = i; } var b = filled_u32(4u32, 0u32); for j in 0u32..4u32 { b[j] = j + 1u32; } if a[3u32] * 10u32 + b[3u32] == 34u32 { return 7; } return 0; }";

fn plan(iterations: usize, owner: &str, values: Vec<u32>) -> LoopOverride {
    LoopOverride {
        iterations,
        array: Some((owner.into(), values)),
        ..Default::default()
    }
}

#[test]
fn a_mid_run_fallback_keeps_each_later_publication_label() {
    let checked = check::check(front::parse(TWO_LOOPS.as_bytes()).unwrap(), false).unwrap();
    let program = mir::lower(&checked).unwrap();
    let mut offered = Vec::new();
    let mut provider = |live: &LiveLoop<'_>| -> Result<Option<LoopOverride>, EvalError> {
        // The first loop's plan claims a wrong trip count, so MIR refuses it and
        // runs that loop serially; the second loop's plan is exact.
        let plan = if offered.is_empty() {
            plan(live.iterations + 1, "a", vec![0, 1, 2, 3, 4])
        } else {
            plan(live.iterations, "b", vec![1, 2, 3, 4])
        };
        offered.push(Executed {
            function: live.function.clone(),
            span: live.span,
            kind: PlanKind::Map,
            published_array: plan.published_array(),
            implementation: None,
        });
        Ok(Some(plan))
    };
    let (observation, published) = mir_eval::observe_offers(&program, 10_000, Some(&mut provider));
    assert_eq!(published, [false, true]);
    let serial = mir_eval::observe(&program, 10_000);
    assert_eq!(observation.result, Ok(crate::eval::Value::I32(7)));
    assert_eq!(observation.result, serial.result);
    assert_eq!(observation.steps_used, serial.steps_used);
    assert_eq!(observation.cleanup_trace, serial.cleanup_trace);
    let second = offered[1].clone();
    assert_ne!(offered[0].span, second.span);
    let executed = keep_published(offered, &published);
    assert_eq!(executed, [second]);
    assert_eq!(executed[0].published_array.as_deref(), Some("b"));
}
