//! CPU preparation only. Synthetic coherent telemetry and supplied endpoint FK
//! exercise the task core through a real checked C lifecycle, not live acceptance.
//! Issue #29 (S1-06).
use invar::system::{
    sequencer::{Action, Config, Error, MAX_SIM_NS, Move, Phase, Sequencer, Snapshot},
    tool_projection::Pose,
};
use invar_foreign::component::planner::{
    Config as PlannerConfig, Planner, Request, Trajectory, phase as world,
};
use std::{fs, path::PathBuf, process::Command};

fn pose(x: f64, y: f64, z: f64) -> Pose {
    Pose {
        translation_m: [x, y, z],
        ..Pose::IDENTITY
    }
}
fn config<const N: usize>() -> Config<N> {
    Config {
        home_q_rad: [0.0; N],
        home_tool: Pose::IDENTITY,
        pregrasp_tool: pose(0.3, 0.0, 0.1),
        grasp_tool: pose(0.3, 0.0, 0.0),
        final_cube: pose(0.5, 0.2, 0.0),
        joint_limits_rad: [[-2.0, 2.0]; N],
        velocity_limits_rad_s: [2.0; N],
    }
}
fn sample<const N: usize>(ns: u64, tool: Pose, cube: Pose, attached: bool) -> Snapshot<N> {
    let mut q = [0.0; N];
    q[0] = tool.translation_m[0];
    Snapshot {
        simulation_ns: ns,
        q_rad: q,
        dq_rad_s: [0.0; N],
        tool_base: tool,
        cube_base: cube,
        attached,
        forbidden_contacts: 0,
    }
}
fn request<const N: usize>(action: Option<Action<N>>) -> Request<N> {
    let Some(Action::Plan(request)) = action else {
        panic!("expected plan");
    };
    request
}
fn as_pose(p: invar_foreign::component::Pose) -> Pose {
    Pose {
        translation_m: p.translation_m,
        quaternion_xyzw: p.quaternion_xyzw,
    }
}

/// Fields drop in order: the planner unloads its library before the root is
/// removed, since Windows cannot delete a loaded library.
struct Fixture<const N: usize> {
    planner: Planner<N>,
    _root: Root,
}
struct Root(PathBuf);
impl Drop for Root {
    fn drop(&mut self) {
        fs::remove_dir_all(&self.0).unwrap();
    }
}
impl<const N: usize> Fixture<N> {
    fn new(label: &str) -> Self {
        let root = std::env::temp_dir().join(format!(
            "invar-sequencer-{}-{N}-{label}",
            std::process::id()
        ));
        fs::create_dir(&root).unwrap();
        fs::write(
            root.join("wire.h"),
            invar::robotics::planner::header(N as u32).unwrap(),
        )
        .unwrap();
        fs::write(
            root.join("fixture.c"),
            include_str!("../fixtures/s1_06/sequencer_planner_cpu.c.in")
                .replace("@N@", &N.to_string()),
        )
        .unwrap();
        let path = root.join(format!("probe.{}", std::env::consts::DLL_EXTENSION));
        let result = Command::new("cc")
            .args([
                "-std=c11",
                "-Wall",
                "-Wextra",
                "-Werror",
                "-pedantic",
                "-fPIC",
            ])
            .arg(if cfg!(target_os = "macos") {
                "-dynamiclib"
            } else {
                "-shared"
            })
            .arg(root.join("fixture.c"))
            .arg("-o")
            .arg(&path)
            .output()
            .unwrap();
        crate::child_deadline::assert_success(&result);
        fs::write(root.join("child"), "not executed: CPU double").unwrap();
        fs::write(root.join("robot.yml"), "CPU double").unwrap();
        let mut planner = Planner::<N>::open(&path, "probe").unwrap();
        planner.init(&root.join("child")).unwrap();
        let names: Vec<_> = (0..N).map(|i| format!("joint{i}")).collect();
        let names: Vec<_> = names.iter().map(String::as_str).collect();
        let limits = [[-2.0, 2.0]; N];
        let speeds = [2.0; N];
        planner
            .configure(&PlannerConfig {
                model_sha256: [1; 32],
                robot_config_sha256: [2; 32],
                scene_sha256: [3; 32],
                joint_limits_rad: &limits,
                velocity_limits_rad_s: &speeds,
                table_pose: invar_foreign::component::Pose {
                    translation_m: [0.0; 3],
                    quaternion_xyzw: [0.0, 0.0, 0.0, 1.0],
                },
                table_half_extents_m: [0.5, 0.5, 0.05],
                cube_edge_m: 0.05,
                base_frame: "base",
                tool_frame: "tool",
                joint_names: &names,
                robot_config_path: &root.join("robot.yml"),
                scene_config_path: &root.join("robot.yml"),
            })
            .unwrap();
        Self {
            planner,
            _root: Root(root),
        }
    }
    fn gate(&mut self, core: &mut Sequencer<N>, request: &Request<N>, ns: u64) {
        let trajectory = self.planner.plan(request).unwrap();
        assert_eq!(
            trajectory.waypoints().len(),
            if request.target_base_from_tool.translation_m[2] == 0.1 {
                256
            } else {
                2
            }
        );
        core.planned(&trajectory, as_pose(request.target_base_from_tool), ns)
            .unwrap();
    }
}
impl<const N: usize> Drop for Fixture<N> {
    fn drop(&mut self) {
        self.planner.shutdown().unwrap();
    }
}

const S: u64 = 1_000_000_000;
const MS: u64 = 1_000_000;

fn shifted(p: Pose, dx: f64) -> Pose {
    let mut q = p;
    q.translation_m[0] += dx;
    q
}

/// Drive one task from home through every planned move to the home request.
/// Returns the requests in order (the home request last) with the core in
/// `Planning(Home)`.
fn to_home<const N: usize>(fixture: &mut Fixture<N>, core: &mut Sequencer<N>) -> Vec<Request<N>> {
    let cube = pose(0.3, 0.0, 0.0);
    let goal = pose(0.5, 0.2, 0.0);
    let pregrasp = pose(0.3, 0.0, 0.1);
    let preplace = pose(0.5, 0.2, 0.1);
    let mut requests = Vec::new();
    let mut plan = |core: &mut Sequencer<N>, s: Snapshot<N>, phase: u32, target: Pose| {
        let r = request(core.observe(s).unwrap());
        assert_eq!(r.phase, phase);
        assert_eq!(as_pose(r.target_base_from_tool), target);
        assert_eq!(
            r.goal_q_rad.map(f64::to_bits),
            if phase == world::HOME {
                [0.0; N]
            } else {
                r.start_q_rad
            }
            .map(f64::to_bits)
        );
        fixture.gate(core, &r, s.simulation_ns);
        requests.push(r);
        r
    };
    let first = plan(
        core,
        sample(0, Pose::IDENTITY, cube, false),
        world::FREE,
        pregrasp,
    );
    assert_eq!(as_pose(first.cube_base_pose), cube);
    assert!(
        core.observe(sample(110 * MS, pregrasp, cube, false))
            .unwrap()
            .is_none()
    );
    assert_eq!(core.phase(), Phase::NeedPlan(Move::Descend));
    assert!(
        core.observe(sample(5 * S - 1, pregrasp, cube, false))
            .unwrap()
            .is_none()
    );
    plan(
        core,
        sample(5 * S, pregrasp, cube, false),
        world::APPROACH,
        cube,
    );
    assert!(matches!(
        core.observe(sample(5 * S + 100 * MS, cube, cube, false))
            .unwrap(),
        Some(Action::SetAttachment(true))
    ));
    // Queue response is not applied physics. No progression until both ack
    // and a later coherent observed attachment state are available.
    assert!(
        core.observe(sample(5 * S + 110 * MS, cube, cube, true))
            .unwrap()
            .is_none()
    );
    assert_eq!(core.phase(), Phase::Attachment(true));
    core.attachment_response(true, 5 * S + 120 * MS).unwrap();
    assert!(
        core.observe(sample(5 * S + 120 * MS, cube, cube, true))
            .unwrap()
            .is_none()
    );
    assert_eq!(core.phase(), Phase::Attachment(true));
    // Settle (decision D3): the relation must hold for the whole window; a
    // 0.2 mm slip restarts it, and so does the return.
    let settle = [
        (130, cube),
        (300, cube),
        (320, shifted(cube, 0.0002)),
        (360, cube),
        (559, cube),
    ];
    for (ms, observed) in settle {
        assert!(
            core.observe(sample(5 * S + ms * MS, cube, observed, true))
                .unwrap()
                .is_none()
        );
        assert_eq!(core.phase(), Phase::Settling, "{ms}");
    }
    assert!(
        core.observe(sample(5 * S + 560 * MS, cube, cube, true))
            .unwrap()
            .is_none()
    );
    assert_eq!(core.phase(), Phase::NeedPlan(Move::Lift));
    let lift = plan(
        core,
        sample(10 * S, cube, cube, true),
        world::HELD_CONTACT,
        pregrasp,
    );
    assert_eq!(as_pose(lift.tool_from_cube), Pose::IDENTITY);
    assert!(
        core.observe(sample(10 * S + 100 * MS, pregrasp, pregrasp, true))
            .unwrap()
            .is_none()
    );
    plan(
        core,
        sample(15 * S, pregrasp, pregrasp, true),
        world::HELD,
        preplace,
    );
    assert!(
        core.observe(sample(15 * S + 100 * MS, preplace, preplace, true))
            .unwrap()
            .is_none()
    );
    plan(
        core,
        sample(20 * S, preplace, preplace, true),
        world::HELD_CONTACT,
        goal,
    );
    let mut wrong_cube = goal;
    wrong_cube.translation_m[1] += 0.006;
    assert!(
        core.observe(sample(20 * S + 100 * MS, goal, wrong_cube, true))
            .unwrap()
            .is_none()
    );
    assert!(matches!(
        core.observe(sample(20 * S + 110 * MS, goal, goal, true))
            .unwrap(),
        Some(Action::SetAttachment(false))
    ));
    core.attachment_response(true, 20 * S + 120 * MS).unwrap();
    assert!(
        core.observe(sample(20 * S + 130 * MS, goal, goal, false))
            .unwrap()
            .is_none()
    );
    assert_eq!(core.phase(), Phase::NeedPlan(Move::Retreat));
    plan(
        core,
        sample(25 * S, goal, goal, false),
        world::APPROACH,
        preplace,
    );
    assert!(
        core.observe(sample(25 * S + 100 * MS, preplace, goal, false))
            .unwrap()
            .is_none()
    );
    let home = request(core.observe(sample(30 * S, preplace, goal, false)).unwrap());
    assert_eq!(home.phase, world::HOME);
    assert_eq!(home.goal_q_rad, [0.0; N]);
    assert_eq!(core.phase(), Phase::Planning(Move::Home));
    assert_eq!(core.phase().label(), "home/planning/world=home");
    requests.push(home);
    requests
}

fn flow<const N: usize>() {
    let mut fixture = Fixture::<N>::new("flow");
    let mut traces = Vec::new();
    for _ in 0..2 {
        let mut core = Sequencer::new(config::<N>()).unwrap();
        let requests = to_home(&mut fixture, &mut core);
        let home = *requests.last().unwrap();
        let goal = pose(0.5, 0.2, 0.0);
        fixture.gate(&mut core, &home, 30 * S);
        let mut wrong_home = sample(30 * S + 100 * MS, Pose::IDENTITY, goal, false);
        wrong_home.q_rad[N - 1] = 0.01;
        assert!(core.observe(wrong_home).unwrap().is_none());
        assert!(matches!(
            core.observe(sample(30 * S + 110 * MS, Pose::IDENTITY, goal, false))
                .unwrap(),
            Some(Action::Complete)
        ));
        assert_eq!(core.phase(), Phase::Complete);
        assert!(matches!(
            core.observe(sample(30 * S + 120 * MS, Pose::IDENTITY, goal, false)),
            Err(Error::Lifecycle)
        ));
        traces.push(
            requests
                .iter()
                .map(|r| {
                    (
                        r.start_q_rad,
                        r.goal_q_rad,
                        r.target_base_from_tool.translation_m,
                        r.target_base_from_tool.quaternion_xyzw,
                        r.phase,
                    )
                })
                .collect::<Vec<_>>(),
        );
    }
    assert_eq!(traces[0].len(), 7);
    assert_eq!(traces[0], traces[1], "synthetic replay determinism only");
}

/// Decision D4: a home plan whose last joints miss the exact home joints by more
/// than 1e-6 rad fails, even with its endpoint FK on the home tool pose.
fn home_joint_gate<const N: usize>() {
    let mut fixture = Fixture::<N>::new("home");
    for (offset, expected) in [(1e-6, Ok(())), (2e-6, Err(Error::Endpoint))] {
        let mut core = Sequencer::new(config::<N>()).unwrap();
        let home = *to_home(&mut fixture, &mut core).last().unwrap();
        let planned = fixture.planner.plan(&home).unwrap();
        let mut points = planned.waypoints().to_vec();
        points.last_mut().unwrap().q_rad[N - 1] += offset;
        let trajectory = Trajectory::checked(
            planned.dt_s(),
            &points,
            &home.start_q_rad,
            &[[-2.0, 2.0]; N],
            &[2.0; N],
        )
        .unwrap();
        assert_eq!(core.planned(&trajectory, Pose::IDENTITY, 30 * S), expected);
    }
}

/// Decision D3: a held cube that never holds still for the window within 2 s of
/// the attach acknowledgement is the typed failure `Error::Settle`.
fn settle_timeout<const N: usize>() {
    let mut fixture = Fixture::<N>::new("settle");
    let cube = pose(0.3, 0.0, 0.0);
    let pregrasp = pose(0.3, 0.0, 0.1);
    let mut core = Sequencer::new(config::<N>()).unwrap();
    let first = request(
        core.observe(sample(0, Pose::IDENTITY, cube, false))
            .unwrap(),
    );
    fixture.gate(&mut core, &first, 0);
    core.observe(sample(100 * MS, pregrasp, cube, false))
        .unwrap();
    let descend = request(core.observe(sample(5 * S, pregrasp, cube, false)).unwrap());
    fixture.gate(&mut core, &descend, 5 * S);
    core.observe(sample(5 * S + 100 * MS, cube, cube, false))
        .unwrap();
    core.attachment_response(true, 5 * S + 110 * MS).unwrap();
    // Slips by 0.15 mm every 150 ms: always outside the window's first sample.
    let mut ns = 5 * S + 120 * MS;
    let mut slip = 0.0;
    while ns <= 5 * S + 110 * MS + 2 * S {
        assert!(
            core.observe(sample(ns, cube, shifted(cube, slip), true))
                .unwrap()
                .is_none()
        );
        assert_eq!(core.phase(), Phase::Settling);
        ns += 150 * MS;
        slip = if slip == 0.0 { 0.00015 } else { 0.0 };
    }
    assert_eq!(
        core.observe(sample(ns, cube, shifted(cube, slip), true))
            .map(|_| ()),
        Err(Error::Settle)
    );
    assert_eq!(core.phase(), Phase::Failed(Error::Settle));
}

fn failures<const N: usize>() {
    let mut fixture = Fixture::<N>::new("faults");
    let cube = pose(0.3, 0.0, 0.0);
    for expected in [
        Error::Clock,
        Error::Deadline,
        Error::Input,
        Error::Contact,
        Error::Planner,
        Error::Endpoint,
        Error::Plan,
        Error::Lifecycle,
        Error::Attachment,
    ] {
        let mut core = Sequencer::new(config::<N>()).unwrap();
        let req = request(
            core.observe(sample(100, Pose::IDENTITY, cube, false))
                .unwrap(),
        );
        let result = match expected {
            Error::Clock => {
                let trajectory = fixture.planner.plan(&req).unwrap();
                core.planned(&trajectory, as_pose(req.target_base_from_tool), 200)
                    .unwrap();
                core.observe(sample(199, Pose::IDENTITY, cube, false))
                    .map(|_| ())
            }
            Error::Deadline => core
                .observe(sample(MAX_SIM_NS + 101, Pose::IDENTITY, cube, false))
                .map(|_| ()),
            Error::Input => {
                let mut s = sample(101, Pose::IDENTITY, cube, false);
                s.dq_rad_s[0] = f64::NAN;
                core.observe(s).map(|_| ())
            }
            Error::Contact => {
                let mut s = sample(101, Pose::IDENTITY, cube, false);
                s.forbidden_contacts = 1;
                core.observe(s).map(|_| ())
            }
            Error::Planner => core.planner_failed(),
            Error::Endpoint => {
                let trajectory = fixture.planner.plan(&req).unwrap();
                let mut wrong = as_pose(req.target_base_from_tool);
                wrong.translation_m[0] += 0.002001;
                core.planned(&trajectory, wrong, 101)
            }
            Error::Plan => {
                let mut drift = sample(101, Pose::IDENTITY, cube, false);
                drift.q_rad[0] = 0.000002;
                core.observe(drift).unwrap();
                let trajectory = fixture.planner.plan(&req).unwrap();
                core.planned(&trajectory, as_pose(req.target_base_from_tool), 102)
            }
            Error::Lifecycle => core.attachment_response(true, 101),
            Error::Attachment => core
                .observe(sample(101, Pose::IDENTITY, cube, true))
                .map(|_| ()),
            _ => unreachable!(),
        };
        assert_eq!(result, Err(expected));
        assert_eq!(core.phase(), Phase::Failed(expected));
        assert!(
            matches!(core.observe(sample(102,Pose::IDENTITY,cube,false)),Err(e) if e==expected)
        );
        assert_eq!(core.planner_failed(), Err(expected));
    }
    let mut core = Sequencer::new(config::<N>()).unwrap();
    core.observe(sample(0, Pose::IDENTITY, cube, false))
        .unwrap();
    assert!(
        core.observe(sample(MAX_SIM_NS, Pose::IDENTITY, cube, false))
            .unwrap()
            .is_none()
    );
    assert!(matches!(
        core.observe(sample(MAX_SIM_NS + 1, Pose::IDENTITY, cube, false)),
        Err(Error::Deadline)
    ));
    let mut core = Sequencer::new(config::<N>()).unwrap();
    let snapshot = sample(0, Pose::IDENTITY, cube, false);
    core.observe(snapshot).unwrap();
    assert!(core.observe(snapshot).unwrap().is_none());
    let mut changed = snapshot;
    changed.q_rad[0] = 1e-8;
    assert!(matches!(core.observe(changed), Err(Error::Input)));
    let mut invalid = config::<N>();
    invalid.home_q_rad[0] = 3.0;
    assert!(matches!(Sequencer::new(invalid), Err(Error::Config)));
    assert!(matches!(Sequencer::new(config::<5>()), Err(Error::Config)));

    // Refuse callback clock overflow and execution that would overrun the
    // existing inclusive 60 simulated-second deadline.
    for (start, receipt, expected) in [
        (u64::MAX - 1, u64::MAX, Error::Clock),
        (0, MAX_SIM_NS, Error::Deadline),
    ] {
        let mut core = Sequencer::new(config::<N>()).unwrap();
        let req = request(
            core.observe(sample(start, Pose::IDENTITY, cube, false))
                .unwrap(),
        );
        let trajectory = fixture.planner.plan(&req).unwrap();
        assert_eq!(
            core.planned(&trajectory, as_pose(req.target_base_from_tool), receipt),
            Err(expected)
        );
        assert_eq!(core.phase(), Phase::Failed(expected));
    }
    for duplicate in [false, true] {
        let mut core = Sequencer::new(config::<N>()).unwrap();
        let req = request(
            core.observe(sample(0, Pose::IDENTITY, cube, false))
                .unwrap(),
        );
        fixture.gate(&mut core, &req, 0);
        core.observe(sample(
            100_000_000,
            as_pose(req.target_base_from_tool),
            cube,
            false,
        ))
        .unwrap();
        let req = request(
            core.observe(sample(
                5_000_000_000,
                as_pose(req.target_base_from_tool),
                cube,
                false,
            ))
            .unwrap(),
        );
        fixture.gate(&mut core, &req, 5_000_000_000);
        assert!(matches!(
            core.observe(sample(5_100_000_000, cube, cube, false))
                .unwrap(),
            Some(Action::SetAttachment(true))
        ));
        let expected = if duplicate {
            core.attachment_response(true, 5_110_000_000).unwrap();
            Error::Lifecycle
        } else {
            Error::Attachment
        };
        assert_eq!(
            core.attachment_response(duplicate, 5_120_000_000),
            Err(expected)
        );
        assert_eq!(core.phase(), Phase::Failed(expected));
    }
    // A unit quaternion with excessive orientation error must fail even when
    // translation is exact. Antipodal identity must succeed.
    for angle in [0.0, 1.001_f64.to_radians()] {
        let mut core = Sequencer::new(config::<N>()).unwrap();
        let req = request(
            core.observe(sample(0, Pose::IDENTITY, cube, false))
                .unwrap(),
        );
        let trajectory = fixture.planner.plan(&req).unwrap();
        let mut fk = as_pose(req.target_base_from_tool);
        fk.quaternion_xyzw = [0.0, 0.0, -(angle / 2.0).sin(), -(angle / 2.0).cos()];
        let result = core.planned(&trajectory, fk, 0);
        assert_eq!(
            result,
            if angle == 0.0 {
                Ok(())
            } else {
                Err(Error::Endpoint)
            }
        );
    }
}

#[test]
fn sequencer_preparation_six_joint_ordering() {
    flow::<6>();
}
#[test]
fn sequencer_preparation_seven_joint_ordering() {
    flow::<7>();
}
#[test]
fn sequencer_preparation_home_is_planned_to_the_exact_home_joints() {
    home_joint_gate::<6>();
    home_joint_gate::<7>();
}
#[test]
fn sequencer_preparation_held_cube_settle_timeout_is_typed() {
    settle_timeout::<6>();
    settle_timeout::<7>();
}
#[test]
fn sequencer_preparation_six_joint_failure_latch() {
    failures::<6>();
}
#[test]
fn sequencer_preparation_seven_joint_failure_latch() {
    failures::<7>();
}
