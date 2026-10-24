//! P1 to P4 tests: policy-model refusal, option attributes, nesting and rejected candidates.

use super::*;

// ---- Review3 policy fixes (P1-P4) ----

/// P1: both MJCF APIs reach `generate_mjcf_target`, which now refuses a policy
/// model with S1504 before any schema check, asset read or output; URDF keeps
/// its S1607 refusal; ordinary (non-policy) cells still generate.
#[test]
fn p1_mjcf_apis_refuse_policy_models_and_leave_ordinary_cells_unchanged() {
    for n in [6, 7] {
        let ir = compile(&source(n)).unwrap();
        for (api, result) in [
            ("generate_mjcf", super::super::super::generate_mjcf(&ir)),
            (
                "generate_ros2_mjcf",
                super::super::super::generate_ros2_mjcf(&ir),
            ),
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
        let Err(error) = super::super::super::generate_urdf(&ir) else {
            panic!("generate_urdf arm {n}: policy model was exported instead of refused");
        };
        assert_eq!(error.code, "S1607", "generate_urdf arm {n}");
    }
    for original in [
        include_str!("../../../examples/cells/reach_place/cell.inv"),
        include_str!("../../../examples/cells/reach_place/cell-ur5e.inv"),
    ] {
        let ir = compile(original).unwrap();
        assert!(super::super::super::generate_mjcf(&ir).is_ok());
        assert!(super::super::super::generate_ros2_mjcf(&ir).is_ok());
    }
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
