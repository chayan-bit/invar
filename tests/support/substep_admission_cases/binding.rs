//! Source-binding and layout tests.

use super::*;

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
        let Err(error) = super::super::super::ros2::generate(&ir) else {
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
        assert!(super::super::super::generate_mjcf(&ir).is_ok());
        assert!(super::super::super::generate_ros2_mjcf(&ir).is_ok());
        assert!(super::super::super::ros2::generate(&ir).is_ok());
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
