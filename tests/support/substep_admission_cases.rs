//! Private source-binding and sealed-model tests. No native/ROS qualification.
use super::model::*;
use super::*;
use crate::front;
use std::path::Path;
use std::sync::OnceLock;

fn source(n: usize) -> String {
    let original = if n == 7 {
        include_str!("../../examples/cells/reach_place/cell.inv")
    } else {
        include_str!("../../examples/cells/reach_place/cell-ur5e.inv")
    };
    let start = original.find("component mujoco_sim {").unwrap();
    let end = start + original[start..].find("\n}").unwrap() + 2;
    let component = original[start..end].replace(
        "    impl invar run;",
        include_str!("../fixtures/substep_admission/policy-clause.inv").trim_end(),
    );
    format!("{}{}{}", &original[..start], component, &original[end..])
}
fn compile(source: &str) -> Result<SystemIr, crate::front::FrontError> {
    super::super::compile_at(
        front::parse_system(source.as_bytes())?,
        &Path::new(env!("CARGO_MANIFEST_DIR")).join("examples/cells/reach_place"),
    )
}
fn binding(n: usize) -> &'static Binding {
    static SIX: OnceLock<Binding> = OnceLock::new();
    static SEVEN: OnceLock<Binding> = OnceLock::new();
    let storage = if n == 6 { &SIX } else { &SEVEN };
    storage.get_or_init(|| {
        bind(&compile(&source(n)).expect("ordinary source policy binding"))
            .unwrap()
            .unwrap()
    })
}
fn request(run: &Run<'_>) -> Request {
    Request {
        run: run.id,
        identity: run.binding.identity,
        next_step: run.live.step + 1,
        arm_sequence: 1,
        arm: run.live.applied.arm(),
        equality: Equality {
            sequence: 1,
            attached: run.live.attached,
        },
    }
}
fn new_run(binding: &Binding, required: u64, capacity: u32) -> Run<'_> {
    Run::new(
        binding,
        RunId([1; 16]),
        State::home(binding),
        required,
        capacity,
    )
    .unwrap()
}
fn drain(run: &mut Run<'_>) {
    if !run.queue.is_empty() {
        run.acknowledge(run.last_record).unwrap();
    }
    assert!(run.credits_consistent());
}
fn capture(value: &Value, binding: &Binding) -> State {
    let floats = |key: &str| {
        value[key]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap())
            .collect::<Vec<_>>()
    };
    let witness = value["witness"].as_str().unwrap();
    let mut bytes = [0; 32];
    for (index, byte) in bytes.iter_mut().enumerate() {
        *byte = u8::from_str_radix(&witness[index * 2..index * 2 + 2], 16).unwrap();
    }
    State {
        step: value["step"].as_u64().unwrap(),
        time: value["time"].as_f64().unwrap(),
        q: floats("q"),
        dq: floats("dq"),
        forces: floats("forces"),
        contacts: value["contacts"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| {
                (
                    v[0].as_u64().unwrap() as usize,
                    v[1].as_u64().unwrap() as usize,
                )
            })
            .collect(),
        applied: Controls::from_raw(binding.layout, &floats("controls")).unwrap(),
        attached: false,
        latent: [0; 8],
        witness: Digest(bytes),
    }
}

fn same_floats(a: &[f64], b: &[f64]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits())
}
fn same_arm(a: &Arm, b: &Arm) -> bool {
    match (a, b) {
        (Arm::Seven(a), Arm::Seven(b)) => {
            a.iter().zip(b).all(|(x, y)| x.0.to_bits() == y.0.to_bits())
        }
        (Arm::Six(a), Arm::Six(b)) => a.iter().zip(b).all(|(x, y)| x.0.to_bits() == y.0.to_bits()),
        _ => false,
    }
}
fn same_state(a: &State, b: &State) -> bool {
    a.step == b.step
        && a.time.to_bits() == b.time.to_bits()
        && same_floats(&a.q, &b.q)
        && same_floats(&a.dq, &b.dq)
        && same_floats(&a.forces, &b.forces)
        && same_floats(&a.applied.raw(), &b.applied.raw())
        && a.contacts == b.contacts
        && a.attached == b.attached
        && a.latent == b.latent
        && a.witness == b.witness
}

// Independently walk persisted records, without consulting Run::finish or its
// counters/credits. This checks model evidence only, not native host attestation.
fn verify_complete_trace(
    records: &[Record],
    binding: &Binding,
    id: RunId,
    required: u64,
) -> Result<(), Fault> {
    if required == 0 || required > MAX_STEPS {
        return Err(Fault::Incomplete);
    }
    let Some(Record {
        kind: Kind::Snapshot(initial),
        ..
    }) = records.first()
    else {
        return Err(Fault::Incomplete);
    };
    if initial.step != 0 || initial.time.to_bits() != 0.0_f64.to_bits() {
        return Err(Fault::Order);
    }
    binding.gate(initial)?;
    let mut current = initial.clone();
    let mut evaluations = 0;
    let mut rank_expected = 0;
    let mut frozen: Option<State> = None;
    let mut frozen_arm: Option<Arm> = None;
    let mut accepted: Option<(Controls, State)> = None;
    let mut command: Option<Request> = None;
    let mut previous: Option<Request> = None;
    let mut finalized = false;
    for (index, record) in records.iter().enumerate() {
        if record.sequence != index as u64 + 1 || finalized {
            return Err(Fault::Order);
        }
        if index == 0 {
            continue;
        }
        match &record.kind {
            Kind::Candidate {
                step,
                evaluation,
                rank,
                controls,
                prospective,
                predicted,
                result,
            } => {
                if accepted.is_some()
                    || command.is_some()
                    || *step != current.step + 1
                    || *step > required
                    || *rank != rank_expected
                    || *rank >= binding.candidates()
                    || *evaluation != evaluations + 1
                    || *evaluation > MAX_RUN_CANDIDATES
                {
                    return Err(Fault::Order);
                }
                let expected_tendon = match binding.layout {
                    Layout::SevenArmTendon => binding
                        .tendon_start()
                        .and_then(|start| start.checked_sub(u8::try_from(*rank).ok()?)),
                    Layout::SixArm => None,
                };
                if controls.tendon() != expected_tendon {
                    return Err(Fault::Order);
                }
                let mut expected_input = current.clone();
                expected_input.attached = prospective.attached;
                if !same_state(prospective, &expected_input)
                    || frozen.as_ref().is_some_and(|v| !same_state(v, prospective))
                {
                    return Err(Fault::PredictionMismatch);
                }
                frozen.get_or_insert_with(|| prospective.clone());
                let arm = controls.arm();
                if frozen_arm
                    .as_ref()
                    .is_some_and(|first| !same_arm(first, &arm))
                {
                    return Err(Fault::PredictionMismatch);
                }
                frozen_arm.get_or_insert(arm);
                if predicted.step != *step
                    || predicted.time.to_bits() != (current.time + DT).to_bits()
                    || !same_floats(&predicted.applied.raw(), &controls.raw())
                    || predicted.attached != prospective.attached
                {
                    return Err(Fault::PredictionMismatch);
                }
                let observed = binding.gate(predicted);
                if observed != *result {
                    return Err(Fault::PredictionMismatch);
                }
                match observed {
                    Ok(()) => accepted = Some((controls.clone(), predicted.as_ref().clone())),
                    Err(Fault::Bounds | Fault::Contact) => (),
                    Err(fault) => return Err(fault),
                }
                rank_expected += 1;
                evaluations += 1;
            }
            Kind::Command {
                step,
                request,
                controls,
            } => {
                let (selected, _) = accepted.as_ref().ok_or(Fault::Order)?;
                if command.is_some()
                    || *step != current.step + 1
                    || request.next_step != *step
                    || request.run != id
                    || request.identity != binding.identity
                {
                    return Err(Fault::Identity);
                }
                if !same_floats(&selected.raw(), &controls.raw())
                    || !same_arm(&selected.arm(), &request.arm)
                    || frozen
                        .as_ref()
                        .is_none_or(|s| s.attached != request.equality.attached)
                {
                    return Err(Fault::PredictionMismatch);
                }
                if previous.as_ref().is_some_and(|last| {
                    request.arm_sequence < last.arm_sequence
                        || (request.arm_sequence == last.arm_sequence
                            && !same_arm(&request.arm, &last.arm))
                        || request.equality.sequence < last.equality.sequence
                        || (request.equality.sequence == last.equality.sequence
                            && request.equality.attached != last.equality.attached)
                }) {
                    return Err(Fault::Order);
                }
                command = Some(request.clone());
            }
            Kind::Snapshot(state) => {
                let (_, predicted) = accepted.take().ok_or(Fault::Order)?;
                previous = Some(command.take().ok_or(Fault::Order)?);
                if state.step != current.step + 1 || !same_state(state, &predicted) {
                    return Err(Fault::PredictionMismatch);
                }
                binding.gate(state)?;
                current = state.clone();
                rank_expected = 0;
                frozen = None;
                frozen_arm = None;
            }
            Kind::Halt {
                step,
                reason,
                evaluations: charged,
            } => {
                if *step != current.step || *charged < evaluations || *charged > MAX_RUN_CANDIDATES
                {
                    return Err(Fault::Order);
                }
                return Err(*reason);
            }
            Kind::Final {
                step,
                witness,
                outcome,
            } => {
                (*outcome)?;
                if accepted.is_some()
                    || command.is_some()
                    || frozen.is_some()
                    || *step != required
                    || current.step != required
                    || *witness != current.witness
                {
                    return Err(Fault::Incomplete);
                }
                finalized = true;
            }
        }
    }
    if !finalized {
        return Err(Fault::Incomplete);
    }
    Ok(())
}

#[test]
fn real_source_binds_both_structural_layouts_without_runtime_export() {
    for n in [6, 7] {
        let ir = compile(&source(n)).unwrap();
        let b = bind(&ir).unwrap().unwrap();
        assert_eq!(b.layout.arm_count(), n);
        // Gripper candidates step down from the source-declared hold 253.
        assert_eq!(b.tendon_start(), if n == 7 { Some(253) } else { None });
        assert_eq!(b.candidates(), if n == 7 { 254 } else { 1 });
        assert_eq!(b.identity.source, digest(ir.canonical_json().as_bytes()));
        assert_ne!(b.identity.assets, Digest([0; 32]));
        assert_ne!(b.identity.controls, b.identity.joints);
        assert_eq!(b.memory_bytes, 512 * 1024 * 1024);
        assert_eq!(b.profile.version, [3, 12, 0]);
        assert_eq!(b.profile.timestep_s, 0.002);
        assert_eq!(
            (b.profile.integrator, b.profile.solver, b.profile.jacobian),
            ("implicitfast", "Newton", "dense")
        );
        assert!(ir.checked_document().is_ok());
        // Never Debug-print an `Ok` export: it holds every generated file.
        let Err(error) = super::super::ros2::generate(&ir) else {
            panic!("ros2::generate arm {n}: policy model was exported instead of refused");
        };
        assert_eq!(error.code, "S1504");
        assert!(error.message.contains("policy is not active"));
        let serialized = super::super::SystemIr::from_json(ir.canonical_json().as_bytes()).unwrap();
        assert_eq!(bind(&serialized).unwrap_err(), Fault::UncheckedSource);
    }
}

#[test]
fn legacy_and_other_manifest_resolution_remain_ordinary() {
    for n in [6, 7] {
        let base = source(n);
        let start = base.find("    param physics_dt:").unwrap();
        let end = start + base[start..].find("    impl extern").unwrap();
        let legacy = format!(
            "{}    impl invar run;{}",
            &base[..start],
            &base[end..].split_once(';').unwrap().1
        );
        let ir = compile(&legacy).unwrap();
        assert!(bind(&ir).unwrap().is_none());
        assert!(super::super::generate_mjcf(&ir).is_ok());
        assert!(super::super::generate_ros2_mjcf(&ir).is_ok());
        assert!(super::super::ros2::generate(&ir).is_ok());
        let position = legacy.find("component mujoco_sim {").unwrap();
        let updated = format!(
            "{}{}",
            &legacy[..position],
            legacy[position..].replacen("impl invar run;", "impl extern pinocchio_fk;", 1)
        );
        assert!(
            compile(&updated).is_ok(),
            "existing actual manifest remains resolved"
        );
        for unknown in [
            "invar::mujoco_substep_admission_v2",
            "invar::mujoco_substep_admission_v1_extra",
            "unregistered_manifest",
        ] {
            let error = compile(&base.replace(TARGET, unknown)).unwrap_err();
            assert_eq!(error.code, "S1001");
            assert!(error.message.contains("extern component manifest"));
        }
    }
}

#[test]
fn source_parameters_fail_closed_without_rounding_or_defaults() {
    let base = source(7);
    for (old, new) in [
        (
            "0.002 s range 0.002 s .. 0.002 s",
            "0.0021 s range 0.0021 s .. 0.0021 s",
        ),
        (
            "256 one range 256 one .. 256 one",
            "255 one range 255 one .. 255 one",
        ),
        ("1 one range 1 one .. 1 one", "0 one range 0 one .. 0 one"),
        ("0.002 s range 0.002 s .. 0.002 s", "0.002 s"),
        (
            "256 one range 256 one .. 256 one",
            "256.00000000000000001 one range 256.00000000000000001 one .. 256.00000000000000001 one",
        ),
        (
            "    param physics_dt: Quantity<dec,s> = 0.002 s range 0.002 s .. 0.002 s;\n",
            "",
        ),
        (
            "    impl extern invar::mujoco_substep_admission_v1;",
            "    param surprise: Quantity<dec,one> = 1 one;\n    impl extern invar::mujoco_substep_admission_v1;",
        ),
        (
            "resource cpu_cores = 1, memory = 512 MiB;",
            "resource cpu_cores = 1, memory = 256 MiB;",
        ),
    ] {
        assert!(base.contains(old));
        let error = compile(&base.replace(old, new)).unwrap_err();
        assert_eq!(error.code, "S1504", "{new}: {}", error.message);
    }
    let duplicate = base.replace(
        "    controller: joint_trajectory_controller;",
        "    duplicate_hardware: mujoco_sim;\n    controller: joint_trajectory_controller;",
    ).replace(
        "connect controller.cmd -> hardware.cmd policy sample-and-hold;",
        "connect controller.cmd -> hardware.cmd policy sample-and-hold;\nconnect controller.cmd -> duplicate_hardware.cmd policy sample-and-hold;",
    );
    assert_eq!(compile(&duplicate).unwrap_err().code, "S1504");
    let duplicate_param=base.replace("    param maximum_candidates:","    param physics_dt: Quantity<dec,s> = 0.002 s range 0.002 s .. 0.002 s;\n    param maximum_candidates:");
    assert_eq!(compile(&duplicate_param).unwrap_err().code, "S1000");
}

#[test]
fn source_layout_bounds_and_control_order_are_not_fixture_name_authority() {
    for n in [6, 7] {
        let base = source(n);
        let renamed = base
            .replace("mujoco_sim", "simulation_alias")
            .replace("hardware:", "hardware_alias:")
            .replace("hardware.cmd", "hardware_alias.cmd");
        assert!(compile(&renamed).is_ok());
        let changed = if n == 7 {
            base.replacen(
                "position -2.8973 rad .. 2.8973 rad;",
                "position -2.89730000000000001 rad .. 2.8973 rad;",
                1,
            )
        } else {
            base.replacen(
                "velocity 3.14159 rad/s;",
                "velocity 3.14159000000000001 rad/s;",
                1,
            )
        };
        assert!(compile(&changed).is_err());
    }
    let base = source(7);
    let swapped = base
        .replacen("joint joint1;", "joint TEMP;", 1)
        .replacen("joint joint2;", "joint joint1;", 1)
        .replacen("joint TEMP;", "joint joint2;", 1);
    assert!(compile(&swapped).is_err());
    assert!(
        compile(&base.replace(
            "        grasp pose xyz(0 m, 0 m, 0.025 m) quat(0, 1, 0, 0);\n",
            ""
        ))
        .is_err()
    );
}

#[test]
fn finite_order_rejects_without_live_state_or_clone_contamination() {
    let b = binding(7);
    let mut run = new_run(b, 1, 260);
    let before = run.live.clone();
    // Candidates start at the source-declared hold 253.
    let mut predictor = Predictor::new(Behavior::RejectAbove(251), 254);
    let ticket = run.prepare(request(&run), &mut predictor).unwrap();
    assert_eq!(ticket.controls.tendon(), Some(251));
    assert_eq!(predictor.inputs.len(), 3);
    assert_eq!(run.live, before);
    assert_eq!(run.commands, 0);
    assert_eq!(run.snapshots, 1);
    for input in &predictor.inputs {
        assert_eq!(input, &before);
        assert_ne!(input.q.as_ptr(), run.live.q.as_ptr());
    }
    let order = run
        .queue
        .iter()
        .filter_map(|r| match &r.kind {
            Kind::Candidate {
                rank,
                controls,
                result,
                ..
            } => Some((*rank, controls.tendon().unwrap(), result.is_ok())),
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(order, [(0, 253, false), (1, 252, false), (2, 251, true)]);
    assert_eq!(run.held, 3);
    assert!(run.credits_consistent());
    let actual = ticket.predicted.clone();
    run.commit(ticket, actual).unwrap();
    assert_eq!(run.live.latent[0], 1);
    assert_eq!(run.live.latent[1], 0);
    assert_eq!(run.held, 0);
    assert!(run.credits_consistent());
}

#[test]
fn every_fuel_prefix_releases_reservation_and_retains_all_rejections() {
    let b = binding(7);
    for fuel in 0..=254 {
        let mut run = new_run(b, 1, 260);
        drain(&mut run);
        let before = run.live.clone();
        let mut p = Predictor::new(Behavior::Exhaust, fuel);
        let fault = run.prepare(request(&run), &mut p).unwrap_err();
        assert_eq!(
            fault,
            if fuel == 254 {
                Fault::NoAdmissibleFiniteGrid
            } else {
                Fault::Fuel
            }
        );
        assert_eq!(run.live, before);
        assert_eq!(run.held, 0);
        assert!(run.credits_consistent());
        assert_eq!(p.inputs.len(), fuel as usize);
        assert_eq!(
            run.queue
                .iter()
                .filter(|r| matches!(r.kind, Kind::Candidate { .. }))
                .count(),
            fuel as usize
        );
        run.quiesce(run.live.clone()).unwrap();
        drain(&mut run);
        assert!(run.finish().is_err());
    }
}

#[test]
fn queue_credit_boundary_has_no_prediction_or_advance_on_failure() {
    let b = binding(7);
    // One held snapshot plus the worst-case reservation (candidates + 3).
    let need = b.candidates() + 4;
    assert_eq!(need, 258);
    for cap in [1, need - 2, need - 1, need] {
        let mut run = new_run(b, 1, cap);
        let before = run.live.clone();
        let mut p = Predictor::new(Behavior::Pass, 1);
        let outcome = run.prepare(request(&run), &mut p);
        if cap < need {
            assert_eq!(outcome.unwrap_err(), Fault::Credits);
            assert!(p.inputs.is_empty());
            assert_eq!(run.live, before);
        } else {
            let ticket = outcome.unwrap();
            let actual = ticket.predicted.clone();
            run.commit(ticket, actual).unwrap();
        }
        assert!(run.credits_consistent());
    }
    assert!(Run::new(b, RunId([1; 16]), State::home(b), 1, u32::MAX).is_err());
}

#[test]
/// Recaptured with `capture_panda_home.py` after the 2026-10-13 decisions: at
/// the source-declared hold 253 every first candidate is admitted (the prior
/// capture at 255 needed five rejections).
fn captured_home_data_replays_1000_commands_1001_states_and_all_rejections() {
    let b = binding(7);
    let fixture: Value = serde_json::from_str(include_str!(
        "../fixtures/substep_admission/panda-home-diagnostic.json"
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

// ---- Review3 policy fixes (P1-P4) ----

/// P1: both MJCF APIs reach `generate_mjcf_target`, which now refuses a policy
/// model with S1504 before any schema check, asset read or output; URDF keeps
/// its S1607 refusal; ordinary (non-policy) cells still generate.
#[test]
fn p1_mjcf_apis_refuse_policy_models_and_leave_ordinary_cells_unchanged() {
    for n in [6, 7] {
        let ir = compile(&source(n)).unwrap();
        for (api, result) in [
            ("generate_mjcf", super::super::generate_mjcf(&ir)),
            ("generate_ros2_mjcf", super::super::generate_ros2_mjcf(&ir)),
        ] {
            // Never Debug-print an `Ok` export: it holds every generated file.
            let Err(error) = result else {
                panic!("{api} arm {n}: policy model was exported instead of refused");
            };
            assert_eq!(error.code, "S1504", "{api} arm {n}");
            assert!(
                error
                    .message
                    .contains("native host adapter unavailable; policy is not active"),
                "{api} arm {n}: {}",
                error.message
            );
        }
        let Err(error) = super::super::generate_urdf(&ir) else {
            panic!("generate_urdf arm {n}: policy model was exported instead of refused");
        };
        assert_eq!(error.code, "S1607", "generate_urdf arm {n}");
    }
    for original in [
        include_str!("../../examples/cells/reach_place/cell.inv"),
        include_str!("../../examples/cells/reach_place/cell-ur5e.inv"),
    ] {
        let ir = compile(original).unwrap();
        assert!(super::super::generate_mjcf(&ir).is_ok());
        assert!(super::super::generate_ros2_mjcf(&ir).is_ok());
    }
}

fn vendored(name: &str) -> String {
    std::fs::read_to_string(
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("examples/cells/reach_place/assets/menagerie")
            .join(name),
    )
    .unwrap()
}
fn rendered(xml: &str) -> Result<(), Fault> {
    check_rendered(&roxmltree::Document::parse(xml).unwrap())
}

/// P2: only the approved `<option>` attributes are admitted and `<option>` may
/// have no children, so a `<flag>` cannot disable contact (or anything else).
#[test]
fn p2_option_attributes_are_allowlisted_and_flags_refused() {
    const ORIGINAL: &str = r#"<option integrator="implicitfast"/>"#;
    for model in [
        "franka_emika_panda/panda.xml",
        "universal_robots_ur5e/ur5e.xml",
    ] {
        let xml = vendored(model);
        assert_eq!(xml.matches(ORIGINAL).count(), 1, "{model}");
        assert_eq!(rendered(&xml), Ok(()), "{model}");
        let with = |option: &str| xml.replace(ORIGINAL, option);
        assert_eq!(
            rendered(&with(
                r#"<option integrator="implicitfast" solver="Newton" jacobian="dense" timestep="0.002"/>"#
            )),
            Ok(()),
            "{model}"
        );
        for (name, option) in [
            (
                "gravity",
                r#"<option integrator="implicitfast" gravity="0 0 0"/>"#,
            ),
            (
                "cone",
                r#"<option integrator="implicitfast" cone="elliptic"/>"#,
            ),
            (
                "impratio",
                r#"<option integrator="implicitfast" impratio="10"/>"#,
            ),
            (
                "tolerance",
                r#"<option integrator="implicitfast" tolerance="1e-6"/>"#,
            ),
            (
                "iterations",
                r#"<option integrator="implicitfast" iterations="1"/>"#,
            ),
            (
                "noslip",
                r#"<option integrator="implicitfast" noslip_iterations="3"/>"#,
            ),
            (
                "flag contact",
                r#"<option integrator="implicitfast"><flag contact="disable"/></option>"#,
            ),
            (
                "flag warmstart",
                r#"<option integrator="implicitfast"><flag warmstart="disable"/></option>"#,
            ),
            (
                "empty flag",
                r#"<option integrator="implicitfast"><flag/></option>"#,
            ),
            ("integrator", r#"<option integrator="Euler"/>"#),
            (
                "solver",
                r#"<option integrator="implicitfast" solver="CG"/>"#,
            ),
            (
                "jacobian",
                r#"<option integrator="implicitfast" jacobian="sparse"/>"#,
            ),
            (
                "timestep",
                r#"<option integrator="implicitfast" timestep="0.001"/>"#,
            ),
            (
                "two options",
                r#"<option integrator="implicitfast"/><option/>"#,
            ),
        ] {
            assert_eq!(
                rendered(&with(option)),
                Err(Fault::UnsupportedLayout),
                "{model}: {name}"
            );
        }
    }
}

/// P3: `<frame>`/`<attach>` nesting is refused, and any geometry the
/// classification walk does not reach fails closed instead of silently
/// shifting native geom indices.
#[test]
fn p3_unsupported_nesting_fails_closed_without_geom_index_shift() {
    let xml = vendored("franka_emika_panda/panda.xml");
    let at = xml.find("<worldbody>").unwrap() + "<worldbody>".len();
    for (name, inserted) in [
        ("frame", r#"<frame pos="0 0 1"><geom size="0.1"/></frame>"#),
        ("attach", r#"<attach model="other" body="b" prefix="x"/>"#),
    ] {
        let mut forged = xml.clone();
        forged.insert_str(at, inserted);
        assert_eq!(rendered(&forged), Err(Fault::UnsupportedLayout), "{name}");
    }
    let roles = BTreeMap::from([("robot", GeomClass::Robot)]);
    let classes = |body: &str| {
        let xml = format!(
            r#"<mujoco><worldbody><geom/><body name="robot">{body}</body></worldbody></mujoco>"#
        );
        geometry_classes(&roxmltree::Document::parse(&xml).unwrap(), &roles)
    };
    assert_eq!(
        classes(r#"<geom/><body name="link"><geom/></body>"#),
        Ok(vec![
            GeomClass::Environment,
            GeomClass::Robot,
            GeomClass::Robot
        ])
    );
    for (name, body) in [
        ("frame", r#"<geom/><frame><geom/></frame>"#),
        (
            "replicate",
            r#"<geom/><replicate count="2"><geom/></replicate>"#,
        ),
        ("unknown container", r#"<geom/><group><geom/></group>"#),
    ] {
        assert_eq!(classes(body), Err(Fault::UnsupportedLayout), "{name}");
    }
}

/// P4: every step charges at least one evaluation, so the 30,000-evaluation
/// run cap admits zero rejected candidates over 30,000 steps, and r rejected
/// candidates cap the run at exactly 30,000 - r steps (cap not raised).
#[test]
fn p4_rejected_candidates_reduce_reachable_steps() {
    assert_eq!(MAX_RUN_CANDIDATES, MAX_STEPS);
    let b = binding(7);
    for rejected in [0u64, 5] {
        let mut run = new_run(b, MAX_STEPS, 260);
        let mut p = Predictor::new(Behavior::RejectAbove(252), 30_001);
        let mut steps = 0u64;
        let stop = loop {
            if steps == rejected {
                p.behavior = Behavior::Pass;
            }
            if steps == MAX_STEPS {
                break None;
            }
            drain(&mut run);
            match run.prepare(request(&run), &mut p) {
                Ok(t) => {
                    let actual = t.predicted.clone();
                    run.commit(t, actual).unwrap();
                    steps += 1;
                }
                Err(fault) => break Some(fault),
            }
        };
        assert_eq!(steps, MAX_STEPS - rejected, "rejected {rejected}");
        assert_eq!(run.candidate_evaluations, MAX_RUN_CANDIDATES);
        assert_eq!(
            stop,
            (rejected > 0).then_some(Fault::CandidateBudget),
            "rejected {rejected}"
        );
        assert!(run.credits_consistent());
    }
}

type Forge = Option<Box<dyn Fn(&str) -> String>>;
thread_local! {
    static FORGE: std::cell::RefCell<Forge> = const { std::cell::RefCell::new(None) };
}
/// Test-only seam: replaces the binder's own rendered model text, so P5
/// forgeries reach the real `bind` path with every other input unchanged.
pub(super) fn forge_rendered(xml: String) -> String {
    FORGE.with(|forge| match forge.borrow().as_ref() {
        Some(forge) => forge(&xml),
        None => xml,
    })
}
fn bind_forged(ir: &SystemIr, forge: impl Fn(&str) -> String + 'static) -> Result<(), Fault> {
    FORGE.with(|f| *f.borrow_mut() = Some(Box::new(forge)));
    let result = bind(ir).map(|_| ());
    FORGE.with(|f| *f.borrow_mut() = None);
    result
}
fn rendered_model(ir: &SystemIr) -> String {
    let captured = std::rc::Rc::new(std::cell::RefCell::new(String::new()));
    let sink = captured.clone();
    bind_forged(ir, move |xml| {
        *sink.borrow_mut() = xml.to_owned();
        xml.to_owned()
    })
    .unwrap();
    captured.take()
}
fn in_body(node: roxmltree::Node<'_, '_>, name: &str) -> bool {
    node.ancestors()
        .any(|a| a.has_tag_name("body") && a.attribute("name") == Some(name))
}
/// Rewrites the attributes of the first selected `<geom>` (always childless).
fn retag(
    xml: &str,
    select: impl Fn(roxmltree::Node<'_, '_>) -> bool,
    changes: &[(&str, &str)],
) -> String {
    let doc = roxmltree::Document::parse(xml).unwrap();
    let geom = doc
        .descendants()
        .find(|n| n.has_tag_name("geom") && select(*n))
        .expect("selected geom");
    assert!(geom.children().next().is_none());
    let mut attrs: Vec<(String, String)> = geom
        .attributes()
        .map(|a| (a.name().to_owned(), a.value().to_owned()))
        .collect();
    for (name, value) in changes {
        match attrs.iter_mut().find(|(n, _)| n == name) {
            Some(entry) => entry.1 = (*value).to_owned(),
            None => attrs.push(((*name).to_owned(), (*value).to_owned())),
        }
    }
    let tag: String = attrs.iter().map(|(k, v)| format!(" {k}=\"{v}\"")).collect();
    let range = geom.range();
    format!("{}<geom{tag}/>{}", &xml[..range.start], &xml[range.end..])
}
/// Independent effective-mask resolution of a vendored model: nested
/// `<default>` classes (top level is `main`), `class`, inherited `childclass`
/// and MuJoCo's built-in 1, in binder walk order from `root`.
fn vendored_roles(xml: &str, root: &str) -> String {
    type Masks = (Option<String>, Option<String>);
    fn defaults(
        node: roxmltree::Node<'_, '_>,
        inherited: &Masks,
        out: &mut BTreeMap<String, Masks>,
    ) {
        let mut masks = inherited.clone();
        for geom in node.children().filter(|n| n.has_tag_name("geom")) {
            if let Some(v) = geom.attribute("contype") {
                masks.0 = Some(v.to_owned());
            }
            if let Some(v) = geom.attribute("conaffinity") {
                masks.1 = Some(v.to_owned());
            }
        }
        out.insert(
            node.attribute("class").unwrap_or("main").to_owned(),
            masks.clone(),
        );
        for child in node.children().filter(|n| n.has_tag_name("default")) {
            defaults(child, &masks, out);
        }
    }
    fn walk(
        body: roxmltree::Node<'_, '_>,
        active: &str,
        classes: &BTreeMap<String, Masks>,
        out: &mut String,
    ) {
        let active = body.attribute("childclass").unwrap_or(active);
        for geom in body.children().filter(|n| n.has_tag_name("geom")) {
            let class = &classes[geom.attribute("class").unwrap_or(active)];
            let contype = geom
                .attribute("contype")
                .or(class.0.as_deref())
                .unwrap_or("1");
            let conaffinity = geom
                .attribute("conaffinity")
                .or(class.1.as_deref())
                .unwrap_or("1");
            out.push(match (contype, conaffinity) {
                ("1", "1") => 'c',
                ("0", "0") => 'v',
                _ => '?',
            });
        }
        for child in body.children().filter(|n| n.has_tag_name("body")) {
            walk(child, active, classes, out);
        }
    }
    let doc = roxmltree::Document::parse(xml).unwrap();
    let mut classes = BTreeMap::new();
    for top in doc
        .root_element()
        .children()
        .filter(|n| n.has_tag_name("default"))
    {
        defaults(top, &(None, None), &mut classes);
    }
    let body = doc
        .descendants()
        .find(|n| n.has_tag_name("body") && n.attribute("name") == Some(root))
        .unwrap();
    let mut out = String::new();
    walk(body, "main", &classes, &mut out);
    out
}

/// P5: the pinned robot roles are the vendored models' effective masks, and
/// the real Panda/UR5e policy sources bind with every robot geom colliding and
/// Panda's link0/link1 self exclusion.
#[test]
fn p5_vendored_collision_masks_are_resolved_and_pinned() {
    for (model, root, layout) in [
        (
            "franka_emika_panda/panda.xml",
            "link0",
            Layout::SevenArmTendon,
        ),
        ("universal_robots_ur5e/ur5e.xml", "base", Layout::SixArm),
    ] {
        let xml = vendored(model);
        assert_eq!(vendored_roles(&xml, root), layout.robot_geoms(), "{model}");
        // A default that zeroes the collision class changes the effective
        // roles, so a model edited that way cannot match the pin.
        const COLLISION: &str = r#"<default class="collision">"#;
        assert_eq!(xml.matches(COLLISION).count(), 1, "{model}");
        let zeroed = xml.replace(
            COLLISION,
            r#"<default class="collision"><geom contype="0" conaffinity="0"/>"#,
        );
        let roles = vendored_roles(&zeroed, root);
        assert!(
            !roles.contains('c') && roles != layout.robot_geoms(),
            "{model}"
        );
    }
    for n in [6, 7] {
        let ir = compile(&source(n)).unwrap();
        assert!(bind(&ir).unwrap().is_some(), "arm {n}");
        let xml = rendered_model(&ir);
        let doc = roxmltree::Document::parse(&xml).unwrap();
        let excludes: Vec<_> = doc
            .descendants()
            .filter(|e| e.has_tag_name("exclude"))
            .map(|e| (e.attribute("body1").unwrap(), e.attribute("body2").unwrap()))
            .collect();
        assert_eq!(
            excludes,
            if n == 7 {
                vec![("link0", "link1")]
            } else {
                // The cell's checked mount exclusions for the UR5e.
                vec![("table", "base"), ("table", "shoulder_link")]
            }
        );
        assert!(doc.descendants().any(|g| g.has_tag_name("geom")
            && g.attribute("contype") == Some("0")
            && g.attribute("conaffinity") == Some("0")));
    }
}

/// P5: direct or default-driven zero masks, partial or disjoint masks, role
/// flips, and exclusions that could hide robot contact are all refused.
#[test]
fn p5_contact_masks_defaults_and_exclusions_fail_closed() {
    const MESSAGE: &str =
        "substep policy collision masks, defaults or contact exclusions unsupported";
    for n in [6, 7] {
        let ir = compile(&source(n)).unwrap();
        let xml = rendered_model(&ir);
        let doc = roxmltree::Document::parse(&xml).unwrap();
        let root = if n == 7 { "link0" } else { "base" };
        let robot = doc
            .descendants()
            .find(|b| b.has_tag_name("body") && b.attribute("name") == Some(root))
            .unwrap();
        let mut chain = std::iter::successors(Some(robot), |b| {
            b.children().find(|c| c.has_tag_name("body"))
        })
        .map(|b| b.attribute("name").unwrap().to_owned());
        let (r0, r1, r2) = (
            chain.next().unwrap(),
            chain.next().unwrap(),
            chain.next().unwrap(),
        );
        let robot_collision =
            |g: roxmltree::Node<'_, '_>| in_body(g, root) && !g.has_attribute("contype");
        let table = |g: roxmltree::Node<'_, '_>| in_body(g, "table");
        let cube_collision =
            |g: roxmltree::Node<'_, '_>| in_body(g, "cube") && !g.has_attribute("contype");
        let floor =
            |g: roxmltree::Node<'_, '_>| g.parent().is_some_and(|p| p.has_tag_name("worldbody"));
        let contact = |body: &str| {
            xml.replacen(
                "</mujoco>",
                &format!("<contact>{body}</contact></mujoco>"),
                1,
            )
        };
        let open = xml.find("<mujoco").unwrap();
        let open = open + xml[open..].find('>').unwrap() + 1;
        let default =
            |inner: &str| format!("{}<default>{inner}</default>{}", &xml[..open], &xml[open..]);
        let refused = [
            (
                "robot contype 0",
                retag(&xml, robot_collision, &[("contype", "0")]),
            ),
            (
                "robot conaffinity 0",
                retag(&xml, robot_collision, &[("conaffinity", "0")]),
            ),
            (
                "robot made visual",
                retag(
                    &xml,
                    robot_collision,
                    &[("contype", "0"), ("conaffinity", "0")],
                ),
            ),
            (
                "robot disjoint mask 2",
                retag(
                    &xml,
                    robot_collision,
                    &[("contype", "2"), ("conaffinity", "2")],
                ),
            ),
            (
                "robot mask spelling",
                retag(&xml, robot_collision, &[("contype", "01")]),
            ),
            (
                "table masks 0",
                retag(&xml, table, &[("contype", "0"), ("conaffinity", "0")]),
            ),
            (
                "object conaffinity 0",
                retag(&xml, cube_collision, &[("conaffinity", "0")]),
            ),
            (
                "floor masks 0",
                retag(&xml, floor, &[("contype", "0"), ("conaffinity", "0")]),
            ),
            (
                "default main mask 0",
                default(r#"<geom contype="0" conaffinity="0"/>"#),
            ),
            (
                "default class",
                default(r#"<default class="quiet"><geom contype="0"/></default>"#),
            ),
            (
                "geom class",
                retag(&xml, robot_collision, &[("class", "quiet")]),
            ),
            ("body childclass", {
                // Insert after `<body`, whatever attributes the generator
                // renders before `name` (gravcomp since decision D1).
                let doc = roxmltree::Document::parse(&xml).unwrap();
                let at = doc
                    .descendants()
                    .find(|b| b.has_tag_name("body") && b.attribute("name") == Some(&r1))
                    .expect("second robot body")
                    .range()
                    .start
                    + "<body".len();
                format!("{} childclass=\"quiet\"{}", &xml[..at], &xml[at..])
            }),
            (
                "exclude robot-object",
                contact(&format!(r#"<exclude body1="{r2}" body2="cube"/>"#)),
            ),
            (
                "exclude robot-environment",
                contact(&format!(r#"<exclude body1="{r1}" body2="place_target"/>"#)),
            ),
            (
                "exclude robot-world",
                contact(&format!(r#"<exclude body1="world" body2="{r0}"/>"#)),
            ),
            (
                "exclude table-object",
                contact(r#"<exclude body1="table" body2="cube"/>"#),
            ),
            (
                "exclude robot non-adjacent",
                contact(&format!(r#"<exclude body1="{r0}" body2="{r2}"/>"#)),
            ),
            (
                "exclude robot self",
                contact(&format!(r#"<exclude body1="{r1}" body2="{r1}"/>"#)),
            ),
            (
                "exclude unknown body",
                contact(&format!(r#"<exclude body1="{r1}" body2="ghost"/>"#)),
            ),
            (
                "exclude extra attribute",
                contact(&format!(
                    r#"<exclude body1="{r1}" body2="{r2}" margin="1"/>"#
                )),
            ),
            ("contact pair", contact(r#"<pair geom1="a" geom2="b"/>"#)),
        ];
        for (name, forged) in refused {
            assert_ne!(forged, xml, "arm {n} {name}: forgery not applied");
            assert_eq!(
                bind_forged(&ir, move |_| forged.clone()),
                Err(Fault::ContactVisibility),
                "arm {n} {name}"
            );
        }
        let message = {
            let forged = retag(&xml, table, &[("contype", "0"), ("conaffinity", "0")]);
            FORGE.with(|f| *f.borrow_mut() = Some(Box::new(move |_| forged.clone())));
            let message = validate(&ir);
            FORGE.with(|f| *f.borrow_mut() = None);
            message
        };
        assert_eq!(message, Err(MESSAGE), "arm {n}");
        // Robot-adjacent self exclusions, in either order, pass every contact
        // rule; they change the rendered model, so only the qualified
        // identity pin (P5 residuals 1 and 3) refuses them.
        for (a, b) in [(&r1, &r2), (&r2, &r1), (&r0, &r1)] {
            let forged = contact(&format!(
                r#"<exclude name="adjacent" body1="{a}" body2="{b}"/>"#
            ));
            assert_eq!(
                bind_forged(&ir, move |_| forged.clone()),
                Err(Fault::Identity),
                "arm {n} {a}/{b}"
            );
        }
        // The cell's checked mount exclusions (UR5e base and shoulder on the table)
        // pass the contact rules in either order; forging them again changes the
        // rendered model, so only the identity pin refuses. Any other robot-table
        // exclusion, including every Panda one, is a contact-visibility refusal.
        for (body, declared) in [(&r0, n == 6), (&r1, n == 6), (&r2, false)] {
            for (a, b) in [(body.as_str(), "table"), ("table", body.as_str())] {
                let forged = contact(&format!(r#"<exclude body1="{a}" body2="{b}"/>"#));
                let expected = if declared {
                    Fault::Identity
                } else {
                    Fault::ContactVisibility
                };
                assert_eq!(
                    bind_forged(&ir, move |_| forged.clone()),
                    Err(expected),
                    "arm {n} {a}/{b}"
                );
            }
        }
        assert_eq!(bind_forged(&ir, |xml| xml.to_owned()), Ok(()), "arm {n}");
    }
}

/// P5 residuals 1 and 3: the binding identity is the qualified rendered model
/// and the qualified vendored asset set. Contact parameters, equality/tendon
/// additions and mesh edits that pass every structural rule are refused.
#[test]
fn p5_residual_model_and_assets_are_pinned_to_qualified_identity() {
    for n in [6, 7] {
        let mut ir = compile(&source(n)).unwrap();
        let b = bind(&ir).unwrap().unwrap();
        assert_eq!(
            [b.identity.model_xml.hex(), b.identity.assets.hex()],
            b.layout.qualified_identity(),
            "arm {n}"
        );
        let xml = rendered_model(&ir);
        let root = if n == 7 { "link0" } else { "base" };
        let robot = |g: roxmltree::Node<'_, '_>| in_body(g, root) && !g.has_attribute("contype");
        let table = |g: roxmltree::Node<'_, '_>| in_body(g, "table");
        let cube = |g: roxmltree::Node<'_, '_>| in_body(g, "cube") && !g.has_attribute("contype");
        let append = |inner: &str| xml.replacen("</mujoco>", &format!("{inner}</mujoco>"), 1);
        let forged = [
            ("robot margin", retag(&xml, robot, &[("margin", "0.01")])),
            ("robot gap", retag(&xml, robot, &[("gap", "0.01")])),
            ("table solref", retag(&xml, table, &[("solref", "0.5 1")])),
            (
                "table solimp",
                retag(&xml, table, &[("solimp", "0.1 0.2 0.001")]),
            ),
            ("cube margin", retag(&xml, cube, &[("margin", "0.02")])),
            (
                "extra equality",
                append(r#"<equality><weld name="pin" body1="cube" body2="table"/></equality>"#),
            ),
            (
                "extra spatial tendon",
                append(
                    r#"<tendon><spatial name="pull"><site site="invar_tool"/><site site="cube_grasp"/></spatial></tendon>"#,
                ),
            ),
        ];
        for (name, forged) in forged {
            assert_ne!(forged, xml, "arm {n} {name}: forgery not applied");
            assert_eq!(
                bind_forged(&ir, move |_| forged.clone()),
                Err(Fault::Identity),
                "arm {n} {name}"
            );
        }
        let robot_id = selected_robot_id(&ir);
        let files = ir.assets.get_mut(&robot_id).unwrap();
        // The UR5e model has no mesh asset, so its license file stands in.
        let mesh = files
            .keys()
            .find(|name| name.ends_with(".stl") || name.ends_with(".obj"))
            .or_else(|| files.keys().next())
            .cloned()
            .unwrap();
        let original = files[&mesh].clone();
        *files.get_mut(&mesh).unwrap().last_mut().unwrap() ^= 1;
        assert_eq!(bind(&ir).unwrap_err(), Fault::Identity, "arm {n} mesh byte");
        let files = ir.assets.get_mut(&robot_id).unwrap();
        files.insert(mesh.clone(), original.clone());
        assert!(bind(&ir).is_ok(), "arm {n} restored mesh");
        let files = ir.assets.get_mut(&robot_id).unwrap();
        files.remove(&mesh);
        files.insert(format!("{mesh}.renamed"), original);
        assert_eq!(bind(&ir).unwrap_err(), Fault::Identity, "arm {n} mesh name");
    }
}
fn selected_robot_id(ir: &SystemIr) -> String {
    let document = ir.checked_document().unwrap();
    let product = &document["products"][0]["robot"];
    document["robots"]
        .as_array()
        .unwrap()
        .iter()
        .find(|r| r["id"] == *product)
        .unwrap()["id"]
        .as_str()
        .unwrap()
        .to_owned()
}

/// P5 residual 2: the robot-table contact gate is qualified only for a cell
/// with exactly one table-class (collision fixture) body; it is not inferred.
#[test]
fn p5_residual_cells_without_exactly_one_table_body_are_refused() {
    const MESSAGE: &str =
        "substep policy collision masks, defaults or contact exclusions unsupported";
    const TABLE: &str = "    fixture table {\n        pose xyz (0.4 m, 0 m, -0.05 m) quat (1, 0, 0, 0);\n        collision box (0.5, 0.5, 0.05) m;\n    }\n";
    // The mount relation names the table; without it the substep gate is what refuses.
    const MOUNT: &str = " on table\n        exclude (menagerie::universal_robots_ur5e::base, menagerie::universal_robots_ur5e::shoulder_link)";
    for n in [6, 7] {
        let base = source(n).replace(MOUNT, "");
        assert!(base.contains(TABLE), "arm {n}");
        let second = TABLE
            .replace("fixture table", "fixture side_table")
            .replace("0.4 m, 0 m, -0.05 m", "0.4 m, 1.5 m, -0.05 m");
        for (name, changed) in [
            ("no table", base.replace(TABLE, "")),
            (
                "visual-only table",
                base.replace(TABLE, &TABLE.replace("collision box", "visual box")),
            ),
            (
                "two tables",
                base.replace(TABLE, &format!("{TABLE}{second}")),
            ),
        ] {
            let error = compile(&changed).unwrap_err();
            assert_eq!(error.code, "S1504", "arm {n} {name}: {}", error.message);
            assert!(
                error.message.contains(MESSAGE),
                "arm {n} {name}: {}",
                error.message
            );
        }
    }
}
