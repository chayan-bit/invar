//! Ordinary indexed source routing. Hardware cases must be explicitly requested.

use a1_indexed_common::{CASES, FUEL, equal, lower, static_refusals};
use invar::{mir_eval, schedule};

#[path = "../support/a1_indexed_common.rs"]
mod a1_indexed_common;

#[test]
fn a1_indexed_cpu_keeps_serial_source_semantics() {
    for s in CASES {
        let p = lower(s);
        let serial = mir_eval::observe(&p, FUEL);
        let run = schedule::run_source(s, &p, FUEL, 2).unwrap();
        equal(&run.observation, &serial);
        assert!(
            !run.executed
                .iter()
                .any(|r| r.kind == schedule::PlanKind::Indexed)
        );
    }
    for s in static_refusals() {
        let p = lower(s.as_bytes());
        let run = schedule::run_source(s.as_bytes(), &p, FUEL, 2).unwrap();
        equal(&run.observation, &mir_eval::observe(&p, FUEL));
        assert!(run.executed.is_empty());
    }
}
