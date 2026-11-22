//! #362: a pair (#262) no longer costs its function's other loops their
//! parallel admission. The region tree tracks a pair local as its two scalar
//! components, so the loops keep their checked form. Value, cleanup trace and
//! fuel prefix equal the reference evaluator and MIR at every worker count.
use invar::{check::check, eval, front::parse, mir, mir_eval, schedule};

const FUEL: u64 = 1_000_000;

/// A pair before the loop and read after it (the issue's program).
const BEFORE: &str = "fn main() -> i32 { let p: (u32, u32) = (3u32, 4u32); let xs: Array<u32> = filled_u32(4096u32, 7u32); var acc: u32 = 0u32; for i in 0u32 .. len(xs) { acc = wrapping_add(acc, xs[i]); } if acc == 28672u32 and p.0 == 3u32 { return 1; } return 0; }";
/// A map loop in a function whose pair is stored and copied.
const STORED: &str = "fn main() -> i32 { var p: (u32, u32) = (3u32, 4u32); p = (p.1, wrapping_add(p.1, 4294967295u32)); let r: (u32, u32) = p; let xs: Array<u32> = filled_u32(4096u32, 7u32); var ys: Array<u32> = filled_u32(4096u32, 0u32); for i in 0u32 .. len(xs) { ys[i] = wrapping_add(xs[i], 1u32); } if ys[4095u32] == 8u32 and r.0 == 4u32 and r.1 == 3u32 { return 1; } return 0; }";
/// A pair inside a sibling loop: that loop stays serial, the other one does not.
const SIBLING: &str = "fn main() -> i32 { let xs: Array<u32> = filled_u32(4096u32, 7u32); var acc: u32 = 0u32; for i in 0u32 .. len(xs) { acc = wrapping_add(acc, xs[i]); } var s: u32 = 0u32; for j in 0u32 .. 8u32 { let q: (u32, u32) = (j, wrapping_mul(j, 2u32)); s = wrapping_add(s, q.1); } if acc == 28672u32 and s == 56u32 { return 1; } return 0; }";
/// A loop that reads a pair component as a live-in.
const LIVE_IN: &str = "fn main() -> i32 { let p: (u32, u32) = (3u32, 4u32); let xs: Array<u32> = filled_u32(4096u32, 7u32); var acc: u32 = 0u32; for i in 0u32 .. len(xs) { acc = wrapping_add(acc, wrapping_add(xs[i], p.1)); } if acc == 45056u32 { return 1; } return 0; }";
/// A swap store: its second component reads `p.0` after the first is stored, so
/// it has no tree form and the function keeps the old serial frame.
const SWAP: &str = "fn main() -> i32 { var p: (u32, u32) = (3u32, 4u32); p = (p.1, p.0); let xs: Array<u32> = filled_u32(64u32, 7u32); var acc: u32 = 0u32; for i in 0u32 .. len(xs) { acc = wrapping_add(acc, xs[i]); } if acc == 448u32 and p.0 == 4u32 and p.1 == 3u32 { return 1; } return 0; }";
/// A pair crossing a call boundary still has no tree form; values still agree.
const BOUNDARY: &str = "fn swap(p: (u32, u32)) -> (u32, u32) { return (p.1, p.0); } fn main() -> i32 { let p: (u32, u32) = swap((3u32, 4u32)); let xs: Array<u32> = filled_u32(64u32, 7u32); var acc: u32 = 0u32; for i in 0u32 .. len(xs) { acc = wrapping_add(acc, xs[i]); } if acc == 448u32 and p.0 == 4u32 { return 1; } return 0; }";

fn agree(source: &str) -> Vec<schedule::Run> {
    let checked = check(parse(source.as_bytes()).unwrap(), true).unwrap();
    let program = mir::lower(&checked).unwrap();
    let reference = eval::observe(&checked, FUEL);
    let serial = mir_eval::observe(&program, FUEL);
    assert_eq!(serial.result, reference.result, "{source}");
    assert_eq!(serial.steps_used, reference.steps_used, "{source}");
    assert_eq!(format!("{:?}", reference.result), "Ok(I32(1))", "{source}");
    [1, 2, 4, 7]
        .into_iter()
        .map(|workers| {
            let run = schedule::run_source(source.as_bytes(), &program, FUEL, workers).unwrap();
            assert_eq!(run.observation.result, serial.result, "{source}");
            assert_eq!(run.observation.steps_used, serial.steps_used, "{source}");
            assert_eq!(
                run.observation.cleanup_trace, serial.cleanup_trace,
                "{source}"
            );
            run
        })
        .collect()
}

/// `(status, chosen_plan)` of every region `explain` reports, in source order.
fn receipt(source: &str) -> Vec<(String, String)> {
    let program = mir::lower(&check(parse(source.as_bytes()).unwrap(), true).unwrap()).unwrap();
    let report = schedule::regions(source.as_bytes(), &program)
        .unwrap()
        .to_json();
    let regions = report["regions"].as_array().unwrap().iter();
    regions
        .map(|r| {
            (
                r["status"].as_str().unwrap().to_owned(),
                r["chosen_plan"].as_str().unwrap().to_owned(),
            )
        })
        .collect()
}

fn admitted(plan: &str) -> (String, String) {
    ("admitted".to_owned(), plan.to_owned())
}

#[test]
fn issue_362_a_pair_outside_the_loop_keeps_its_admission() {
    assert_eq!(receipt(BEFORE), [admitted("wrapping-reduction-if-guard")]);
    for run in agree(BEFORE).iter().skip(1) {
        assert_eq!(run.plan, schedule::PlanKind::Reduction);
        assert_eq!(run.executed.len(), 1);
    }
    assert_eq!(receipt(STORED), [admitted("wrapping-map-candidate-if-guard")]);
    for run in agree(STORED).iter().skip(1) {
        assert_ne!(run.plan, schedule::PlanKind::Serial);
        assert_eq!(run.executed.len(), 1);
    }
}

#[test]
fn issue_362_a_pair_in_a_sibling_loop_demotes_only_that_loop() {
    let receipt = receipt(SIBLING);
    assert_eq!(receipt[0], admitted("wrapping-reduction-if-guard"));
    assert_eq!(receipt[1].0, "rejected");
    for run in agree(SIBLING).iter().skip(1) {
        assert_eq!(run.executed.len(), 1);
    }
}

#[test]
fn issue_362_pair_live_ins_and_boundaries_agree_on_every_engine() {
    agree(LIVE_IN);
    agree(BOUNDARY);
    agree(SWAP);
    // A pair parameter or result, or a swap store, keeps the function without a
    // tree form, as before.
    assert_eq!(receipt(BOUNDARY)[0].0, "rejected");
    assert_eq!(receipt(SWAP)[0].0, "rejected");
}
