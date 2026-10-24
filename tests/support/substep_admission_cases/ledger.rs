//! Observation, budget and independent ledger verifier tests.

use super::*;

#[test]
fn observations_bounds_contacts_and_signed_zero_fail_exactly() {
    let b = binding(7);
    let home = State::home(b);
    for index in 0..b.joint_bounds.len() {
        let mut state = home.clone();
        state.q[index] = b.joint_bounds[index][1].next_up();
        assert_eq!(b.gate(&state), Err(Fault::Bounds));
        let mut state = home.clone();
        state.q[index] = b.joint_bounds[index][0].next_down();
        assert_eq!(b.gate(&state), Err(Fault::Bounds));
        let mut state = home.clone();
        state.dq[index] = b.speeds[index].next_up();
        assert_eq!(b.gate(&state), Err(Fault::Bounds));
        let mut state = home.clone();
        state.dq[index] = -b.speeds[index].next_up();
        assert_eq!(b.gate(&state), Err(Fault::Bounds));
    }
    for index in 0..b.force_bounds.len() {
        let mut state = home.clone();
        state.forces[index] = b.force_bounds[index][1].next_up();
        assert_eq!(b.gate(&state), Err(Fault::Bounds));
    }
    let robot = b.geoms.iter().position(|c| *c == GeomClass::Robot).unwrap();
    for class in [GeomClass::Robot, GeomClass::Table, GeomClass::Environment] {
        let other = b.geoms.iter().position(|c| *c == class).unwrap();
        for pair in [(robot, other), (other, robot)] {
            let mut s = home.clone();
            s.contacts.push(pair);
            assert_eq!(b.gate(&s), Err(Fault::Contact));
        }
    }
    let object = b
        .geoms
        .iter()
        .position(|c| *c == GeomClass::Object)
        .unwrap();
    let mut s = home.clone();
    s.contacts.push((robot, object));
    assert!(b.gate(&s).is_ok());
    s.contacts.push((MAX_GEOMS, object));
    assert_eq!(b.gate(&s), Err(Fault::Capacity));
    let mut s = home.clone();
    s.contacts = vec![(0, object); MAX_CONTACTS + 1];
    assert_eq!(b.gate(&s), Err(Fault::Capacity));
    let mut s = home.clone();
    s.q.pop();
    assert_eq!(b.gate(&s), Err(Fault::Capacity));
    for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        let mut s = home.clone();
        s.forces[0] = bad;
        assert_eq!(b.gate(&s), Err(Fault::Nonfinite));
    }
    let mut run = new_run(b, 1, 260);
    let mut p = Predictor::new(Behavior::Pass, 1);
    let t = run.prepare(request(&run), &mut p).unwrap();
    let mut actual = t.predicted.clone();
    actual.q[0] = -0.0;
    assert_eq!(
        run.commit(t, actual).unwrap_err(),
        Fault::PredictionMismatch
    );
    assert_eq!(run.live.step, 1);
    assert_eq!(run.commands, 1);
    assert!(run.credits_consistent());
}

#[test]
fn failed_prediction_no_advance_and_stale_request_order_are_typed() {
    let b = binding(7);
    for behavior in [Behavior::Nonfinite, Behavior::WrongTransition] {
        let mut run = new_run(b, 1, 260);
        let mut p = Predictor::new(behavior, 256);
        let before = run.live.clone();
        assert!(run.prepare(request(&run), &mut p).is_err());
        assert_eq!(run.live, before);
        assert_eq!(p.inputs.len(), 1);
        assert!(run.credits_consistent());
    }
    let mut run = new_run(b, 2, 260);
    let mut p = Predictor::new(Behavior::Pass, 2);
    let first = request(&run);
    let t = run.prepare(first.clone(), &mut p).unwrap();
    let actual = t.predicted.clone();
    run.commit(t, actual).unwrap();
    drain(&mut run);
    assert_eq!(run.prepare(first, &mut p).unwrap_err(), Fault::Order);
    assert_eq!(run.live.step, 1);
    let mut run = new_run(b, 2, 260);
    let mut p = Predictor::new(Behavior::Pass, 2);
    let t = run.prepare(request(&run), &mut p).unwrap();
    let actual = t.predicted.clone();
    run.commit(t, actual).unwrap();
    drain(&mut run);
    let mut r = request(&run);
    if let Arm::Seven(a) = &mut r.arm {
        a[0] = Rad(0.1)
    }
    assert_eq!(run.prepare(r, &mut p).unwrap_err(), Fault::Order);
}

#[test]
fn completion_requires_exact_counts_final_capture_and_drain() {
    let b = binding(6);
    for count in [0, 1, 999] {
        let mut run = new_run(b, 1000, 260);
        let mut p = Predictor::new(Behavior::Pass, 1000);
        for _ in 0..count {
            drain(&mut run);
            let t = run.prepare(request(&run), &mut p).unwrap();
            let actual = t.predicted.clone();
            run.commit(t, actual).unwrap();
        }
        run.quiesce(run.live.clone()).unwrap();
        drain(&mut run);
        assert_eq!(run.finish().unwrap_err(), Fault::Incomplete);
    }
    let mut run = new_run(b, 1, 260);
    let mut p = Predictor::new(Behavior::Pass, 1);
    let t = run.prepare(request(&run), &mut p).unwrap();
    let actual = t.predicted.clone();
    run.commit(t, actual).unwrap();
    assert_eq!(run.finish().unwrap_err(), Fault::Lifecycle);
    run.quiesce(run.live.clone()).unwrap();
    assert_eq!(run.finish().unwrap_err(), Fault::Credits);
    let front = run.queue.front().unwrap().sequence;
    run.acknowledge(front).unwrap();
    assert_eq!(run.finish().unwrap_err(), Fault::Credits);
    assert_eq!(run.acknowledge(front).unwrap_err(), Fault::Order);
    assert_eq!(
        run.acknowledge(run.last_record + 1).unwrap_err(),
        Fault::Order
    );
    drain(&mut run);
    let done = run.finish().unwrap();
    assert_eq!((done.commands, done.snapshots), (1, 2));
    assert!(Run::new(b, RunId([1; 16]), State::home(b), 0, 260).is_err());
    assert!(Run::new(b, RunId([1; 16]), State::home(b), MAX_STEPS + 1, 260).is_err());
}

#[test]
fn acknowledgement_corruption_overflow_and_inflight_stop_keep_credits() {
    let b = binding(7);
    let mut run = new_run(b, 1, 260);
    let mut p = Predictor::new(Behavior::Pass, 1);
    let t = run.prepare(request(&run), &mut p).unwrap();
    let actual = t.predicted.clone();
    run.commit(t, actual).unwrap();
    let available = run.available;
    let len = run.queue.len();
    run.queue.get_mut(1).unwrap().sequence = 99;
    assert_eq!(run.acknowledge(run.last_record).unwrap_err(), Fault::Order);
    assert_eq!(run.available, available);
    assert_eq!(run.queue.len(), len);
    let mut run = new_run(b, 1, 260);
    run.last_record = u64::MAX;
    let mut p = Predictor::new(Behavior::Pass, 1);
    assert_eq!(
        run.prepare(request(&run), &mut p).unwrap_err(),
        Fault::Overflow
    );
    assert!(p.inputs.is_empty());
    assert_eq!(run.live.step, 0);
    assert!(run.credits_consistent());
    let mut run = new_run(b, 1, 260);
    let mut p = Predictor::new(Behavior::Pass, 1);
    let _ticket = run.prepare(request(&run), &mut p).unwrap();
    assert_eq!(run.quiesce(run.live.clone()).unwrap_err(), Fault::Lifecycle);
    assert_eq!(run.live.step, 0);
    assert_eq!(run.held, 0);
    assert!(run.credits_consistent());
}

#[test]
fn cumulative_candidate_budget_is_monotonic_and_never_refunded() {
    let b = binding(7);
    let mut run = new_run(b, 20_000, 260);
    let mut p = Predictor::new(Behavior::RejectAbove(252), 30_001);
    for step in 1..=15_000 {
        drain(&mut run);
        let t = run.prepare(request(&run), &mut p).unwrap();
        let actual = t.predicted.clone();
        run.commit(t, actual).unwrap();
        assert_eq!(run.live.step, step);
        assert_eq!(run.candidate_evaluations, step * 2);
        assert!(run.credits_consistent());
    }
    drain(&mut run);
    assert_eq!(run.candidate_evaluations, MAX_RUN_CANDIDATES);
    let before = run.live.clone();
    let calls = p.calls;
    assert_eq!(
        run.prepare(request(&run), &mut p).unwrap_err(),
        Fault::CandidateBudget
    );
    assert_eq!(p.calls, calls);
    assert_eq!(run.live, before);
    assert_eq!(run.candidate_evaluations, MAX_RUN_CANDIDATES);
    assert!(run.credits_consistent());
}

#[test]
fn independent_ledger_verifier_rejects_truncation_aliases_and_mutations() {
    let b = binding(7);
    let mut run = new_run(b, 2, 260);
    let mut p = Predictor::new(Behavior::RejectAbove(252), 4);
    let mut ledger = Vec::new();
    for _ in 0..2 {
        ledger.extend(run.queue.iter().cloned());
        drain(&mut run);
        let t = run.prepare(request(&run), &mut p).unwrap();
        let actual = t.predicted.clone();
        run.commit(t, actual).unwrap();
    }
    run.quiesce(run.live.clone()).unwrap();
    ledger.extend(run.queue.iter().cloned());
    verify_complete_trace(&ledger, b, run.id, 2).unwrap();
    assert!(verify_complete_trace(&ledger, b, run.id, 1000).is_err());
    for length in 0..ledger.len() {
        assert!(verify_complete_trace(&ledger[..length], b, run.id, 2).is_err());
    }
    for mutation in 0..17 {
        let mut changed = ledger.clone();
        match mutation {
            0 => {
                changed.remove(3);
            }
            1 => {
                changed.insert(3, changed[3].clone());
            }
            2 => changed.swap(1, 2),
            3 => changed[1].sequence = 0,
            4..=9 => {
                let Kind::Candidate {
                    evaluation,
                    rank,
                    prospective,
                    predicted,
                    result,
                    controls,
                    ..
                } = &mut changed[1].kind
                else {
                    unreachable!()
                };
                match mutation {
                    4 => *evaluation += 1,
                    5 => *rank += 1,
                    6 => prospective.latent[3] += 1,
                    7 => predicted.q[0] = f64::NAN,
                    8 => *result = Ok(()),
                    _ => {
                        if let Controls::Panda { tendon, .. } = controls {
                            *tendon = 0;
                        }
                    }
                }
            }
            10..=13 => {
                let record = changed
                    .iter_mut()
                    .find(|r| matches!(r.kind, Kind::Command { .. }))
                    .unwrap();
                let Kind::Command {
                    request, controls, ..
                } = &mut record.kind
                else {
                    unreachable!()
                };
                match mutation {
                    10 => request.run = RunId([8; 16]),
                    11 => request.identity.source = Digest([8; 32]),
                    12 => request.equality.attached = true,
                    _ => {
                        if let Controls::Panda { tendon, .. } = controls {
                            *tendon = 0;
                        }
                    }
                }
            }
            14..=15 => {
                let Kind::Final { step, witness, .. } = &mut changed.last_mut().unwrap().kind
                else {
                    unreachable!()
                };
                if mutation == 14 {
                    *step = 1000;
                } else {
                    *witness = Digest([8; 32]);
                }
            }
            _ => {
                changed.last_mut().unwrap().kind = Kind::Halt {
                    step: 2,
                    reason: Fault::CandidateBudget,
                    evaluations: 4,
                }
            }
        }
        assert!(
            verify_complete_trace(&changed, b, run.id, 2).is_err(),
            "mutation {mutation}"
        );
    }
    // A source-bound role alias passes every role/classification check but
    // changes the rendered model, so the qualified identity pin refuses it.
    let base = source(7);
    let renamed = base
        .replace("table", "bench_alias")
        .replace("cube", "payload_alias");
    let error = compile(&renamed).unwrap_err();
    assert_eq!(error.code, "S1504");
    assert_eq!(
        error.message,
        "substep policy rendered model or vendored assets differ from the qualified identity"
    );
}

#[test]
fn terminal_regression_completed_prefix_cannot_hide_a_later_halt() {
    let b = binding(7);
    let mut run = new_run(b, 1, 260);
    let mut p = Predictor::new(Behavior::Pass, 2);
    let ticket = run.prepare(request(&run), &mut p).unwrap();
    let actual = ticket.predicted.clone();
    run.commit(ticket, actual).unwrap();
    assert_eq!(run.held, 0);
    assert_eq!(
        run.prepare(request(&run), &mut p).unwrap_err(),
        Fault::Order
    );
    assert!(
        !run.queue
            .iter()
            .any(|r| matches!(r.kind, Kind::Halt { .. }))
    );
    run.quiesce(run.live.clone()).unwrap();
    let ledger = run.queue.iter().cloned().collect::<Vec<_>>();
    drain(&mut run);
    assert_eq!(run.finish().unwrap_err(), Fault::Order);
    assert_eq!(
        verify_complete_trace(&ledger, b, run.id, 1),
        Err(Fault::Order)
    );
}

#[test]
fn terminal_regression_rejected_candidate_cannot_change_the_frozen_arm() {
    let b = binding(7);
    let mut run = new_run(b, 1, 260);
    let mut p = Predictor::new(Behavior::RejectAbove(252), 2);
    let ticket = run.prepare(request(&run), &mut p).unwrap();
    let actual = ticket.predicted.clone();
    run.commit(ticket, actual).unwrap();
    run.quiesce(run.live.clone()).unwrap();
    let mut ledger = run.queue.iter().cloned().collect::<Vec<_>>();
    verify_complete_trace(&ledger, b, run.id, 1).unwrap();
    let Kind::Candidate {
        controls,
        predicted,
        ..
    } = &mut ledger[1].kind
    else {
        unreachable!()
    };
    for value in [controls, &mut predicted.applied] {
        let Controls::Panda { arm, .. } = value else {
            unreachable!()
        };
        arm[0] = Rad(0.1);
    }
    assert_eq!(
        verify_complete_trace(&ledger, b, run.id, 1),
        Err(Fault::PredictionMismatch)
    );
}
