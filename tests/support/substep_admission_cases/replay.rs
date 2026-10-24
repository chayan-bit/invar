//! Captured replay, frozen-arm and request-mutation tests.

use super::*;

#[test]
/// Recaptured with `capture_panda_home.py` after the 2026-10-13 decisions: at
/// the source-declared hold 253 every first candidate is admitted (the prior
/// capture at 255 needed five rejections).
fn captured_home_data_replays_1000_commands_1001_states_and_all_rejections() {
    let b = binding(7);
    let fixture: Value = serde_json::from_str(include_str!(
        "../../fixtures/substep_admission/panda-home-diagnostic.json"
    ))
    .unwrap();
    let actual_hex = b
        .identity
        .model_xml
        .0
        .iter()
        .map(|v| format!("{v:02x}"))
        .collect::<String>();
    assert_eq!(actual_hex, fixture["model_xml_sha256"].as_str().unwrap());
    let records = fixture["candidates"]
        .as_array()
        .unwrap()
        .iter()
        .map(|r| {
            let state = capture(&r["state"], b);
            Recorded {
                step: r["step"].as_u64().unwrap(),
                controls: state.applied.clone(),
                state,
            }
        })
        .collect();
    let initial = capture(&fixture["initial"], b);
    let mut run = Run::new(b, RunId([2; 16]), initial, 1000, 260).unwrap();
    let mut predictor = Predictor::new(Behavior::Recorded(records), 1000);
    let mut ledger = Vec::new();
    for actual in fixture["executed"].as_array().unwrap().iter().skip(1) {
        ledger.extend(run.queue.iter().cloned());
        drain(&mut run);
        let ticket = run.prepare(request(&run), &mut predictor).unwrap();
        run.commit(ticket, capture(actual, b)).unwrap();
        assert!(run.credits_consistent());
    }
    run.quiesce(run.live.clone()).unwrap();
    assert_eq!(run.finish().unwrap_err(), Fault::Credits);
    ledger.extend(run.queue.iter().cloned());
    drain(&mut run);
    let done = run.finish().unwrap();
    assert_eq!(
        (done.commands, done.snapshots, done.completed_steps),
        (1000, 1001, 1000)
    );
    assert_eq!(done.final_sequence, ledger.len() as u64);
    assert_eq!(predictor.calls, 1000);
    assert_eq!(run.candidate_evaluations, 1000);
    let selected = ledger
        .iter()
        .filter_map(|r| match &r.kind {
            Kind::Command { step, controls, .. } => Some((*step, controls.tendon().unwrap())),
            _ => None,
        })
        .collect::<BTreeMap<_, _>>();
    assert_eq!(selected.len(), 1000);
    assert!(selected.values().all(|control| *control == 253));
    let mut per_step = BTreeMap::<u64, u32>::new();
    for r in &ledger {
        if let Kind::Candidate { step, .. } = r.kind {
            *per_step.entry(step).or_default() += 1;
        }
    }
    assert_eq!(per_step.values().copied().max(), Some(1)); // observed hold, not worst-case WCET
    assert_eq!(
        ledger
            .iter()
            .filter(|r| matches!(r.kind, Kind::Candidate { result: Err(_), .. }))
            .count(),
        0
    );
    let steps = ledger
        .iter()
        .filter_map(|r| match &r.kind {
            Kind::Snapshot(s) => Some(s.step),
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(steps, (0..=1000).collect::<Vec<_>>());
    let command_steps = ledger
        .iter()
        .filter_map(|r| match &r.kind {
            Kind::Command { step, .. } => Some(*step),
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(command_steps, (1..=1000).collect::<Vec<_>>());
    assert_eq!(run.live.time.to_bits(), 2.0000000000000013_f64.to_bits());
    verify_complete_trace(&ledger, b, RunId([2; 16]), 1000).unwrap();
    assert_eq!(
        ledger
            .iter()
            .filter(|r| matches!(r.kind, Kind::Final { .. }))
            .count(),
        1
    );
    assert!(run.finish().is_err());
}

#[test]
fn moving_arm_and_equality_are_frozen_separately_model_only() {
    for n in [6, 7] {
        let b = binding(n);
        let mut run = new_run(b, 2, 260);
        let mut p = Predictor::new(Behavior::Pass, 2);
        for step in 1..=2 {
            drain(&mut run);
            let mut r = request(&run);
            r.arm_sequence = step;
            r.equality = Equality {
                sequence: step,
                attached: step == 1,
            };
            match &mut r.arm {
                Arm::Seven(a) => a[0] = Rad(0.01 * step as f64),
                Arm::Six(a) => a[0] = Rad(0.01 * step as f64),
            }
            let frozen = r.arm.clone();
            let t = run.prepare(r, &mut p).unwrap();
            assert_eq!(t.controls.arm(), frozen);
            assert_eq!(run.live.step, step - 1);
            assert_eq!(t.predicted.attached, step == 1);
            let actual = t.predicted.clone();
            run.commit(t, actual).unwrap();
        }
        assert_eq!(p.calls, 2); // six-arm has exactly one candidate per step
        run.quiesce(run.live.clone()).unwrap();
        drain(&mut run);
        assert!(run.finish().is_ok());
    }
}

#[test]
fn ticket_and_request_mutations_cannot_rewrite_a_frozen_command() {
    let b = binding(7);
    for variant in 0..5 {
        let mut run = new_run(b, 1, 260);
        let mut p = Predictor::new(Behavior::Pass, 1);
        let mut t = run.prepare(request(&run), &mut p).unwrap();
        let actual = t.predicted.clone();
        match variant {
            0 => t.request.arm_sequence += 1,
            1 => t.request.equality.attached = true,
            2 => t.request.run = RunId([9; 16]),
            3 => t.request.identity.model_xml = Digest([9; 32]),
            _ => match &mut t.request.arm {
                Arm::Seven(a) => a[0] = Rad(0.1),
                Arm::Six(_) => unreachable!(),
            },
        }
        assert_eq!(run.commit(t, actual).unwrap_err(), Fault::Identity);
        assert_eq!(run.live.step, 0);
        assert!(run.credits_consistent());
    }
    for variant in 0..5 {
        let mut run = new_run(b, 1, 260);
        let mut p = Predictor::new(Behavior::Pass, 1);
        let mut r = request(&run);
        match variant {
            0 => r.run = RunId([9; 16]),
            1 => r.identity.controls = Digest([9; 32]),
            2 => r.next_step = 2,
            3 => r.arm = Arm::Six([Rad(0.0); 6]),
            _ => {
                if let Arm::Seven(a) = &mut r.arm {
                    a[0] = Rad(f64::NAN)
                }
            }
        }
        assert!(run.prepare(r, &mut p).is_err());
        assert!(p.inputs.is_empty());
        assert_eq!(run.live.step, 0);
        assert!(run.credits_consistent());
    }
}
