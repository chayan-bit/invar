//! Bounded S1-06 task core. Public source/ROS binding is deliberately separate.
//! Driver inputs must be coherent, source-framed measurements, not asynchronous
//! mixtures. Logical completion here is not recorded/live S1 acceptance.
use super::tool_projection::Pose;
pub mod runtime;
use invar_foreign::component::{
    Pose as ForeignPose,
    planner::{Request, Trajectory, phase as world},
};

pub const MAX_SIM_NS: u64 = 60_000_000_000;
pub const REQUEST_PERIOD_NS: u64 = 5_000_000_000; // Existing declared planner rate:0.2Hz.
const START_RAD: f64 = 1e-6;
/// Clearance above the grasp and the place pose, along base z (the binding requires
/// an identity mount, so base z is world z), for the pregrasp and preplace poses.
pub const CLEARANCE_Z_M: f64 = 0.1;
/// Held-cube settle (decision D3): after the attach acknowledgement, the measured
/// tool-from-cube relation must stay within SETTLE_TRANSLATION_M and
/// SETTLE_ROTATION_RAD of the window's first sample for SETTLE_WINDOW_NS of
/// simulation time. The settled relation is the attachment pose. Not settling within
/// SETTLE_TIMEOUT_NS of the acknowledgement is the typed failure `Error::Settle`.
pub const SETTLE_WINDOW_NS: u64 = 200_000_000;
pub const SETTLE_TIMEOUT_NS: u64 = 2_000_000_000;
pub const SETTLE_TRANSLATION_M: f64 = 1e-4;
pub const SETTLE_ROTATION_RAD: f64 = std::f64::consts::PI / 1800.0;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Error {
    Config,
    Lifecycle,
    Clock,
    Deadline,
    Input,
    Contact,
    Planner,
    Plan,
    Endpoint,
    Attachment,
    Settle,
    Home,
}
/// Every move is planned; its planner world phase is fixed by the move (decision D2).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Move {
    /// Home to above the cube; the cube is an obstacle.
    Pregrasp,
    /// Final approach to the grasp; the cube obstacle is removed.
    Descend,
    /// Held cube off the table; only its table-contact slab is uncovered.
    Lift,
    /// Held cube to above the place pose with its full cover.
    Transfer,
    /// Held cube down onto the table; only its table-contact slab is uncovered.
    PlaceDescend,
    /// Away from the released cube; the cube obstacle is removed.
    Retreat,
    /// Joint-space plan to the exact home joints; the cube is an obstacle.
    Home,
}
impl Move {
    pub fn world_phase(self) -> u32 {
        match self {
            Move::Pregrasp => world::FREE,
            Move::Descend | Move::Retreat => world::APPROACH,
            Move::Transfer => world::HELD,
            Move::Lift | Move::PlaceDescend => world::HELD_CONTACT,
            Move::Home => world::HOME,
        }
    }
    fn name(self) -> &'static str {
        match self {
            Move::Pregrasp => "pregrasp",
            Move::Descend => "descend",
            Move::Lift => "lift",
            Move::Transfer => "transfer",
            Move::PlaceDescend => "place_descend",
            Move::Retreat => "retreat",
            Move::Home => "home",
        }
    }
}
fn world_name(phase: u32) -> &'static str {
    match phase {
        world::FREE => "free",
        world::APPROACH => "approach",
        world::HELD => "held",
        world::HELD_CONTACT => "held_contact",
        _ => "home",
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Phase {
    Ready,
    NeedPlan(Move),
    Planning(Move),
    Executing(Move),
    Attachment(bool),
    /// Attached; waiting for the held cube to settle before measuring the grasp.
    Settling,
    Complete,
    Failed(Error),
}
impl Phase {
    /// Stable task-log/MCAP name: `move/state/world=<planner world phase>`.
    pub fn label(self) -> String {
        let moving = |m: Move, state: &str| {
            format!("{}/{state}/world={}", m.name(), world_name(m.world_phase()))
        };
        match self {
            Phase::Ready => "ready".into(),
            Phase::NeedPlan(m) => moving(m, "need_plan"),
            Phase::Planning(m) => moving(m, "planning"),
            Phase::Executing(m) => moving(m, "executing"),
            Phase::Attachment(true) => "attach".into(),
            Phase::Attachment(false) => "release".into(),
            Phase::Settling => "settle".into(),
            Phase::Complete => "complete".into(),
            Phase::Failed(e) => format!("failed/{e:?}"),
        }
    }
}

/// Supplied task poses are candidate inputs, not a reachability/collision proof.
pub struct Config<const N: usize> {
    pub home_q_rad: [f64; N],
    pub home_tool: Pose,
    pub pregrasp_tool: Pose,
    pub grasp_tool: Pose,
    pub final_cube: Pose,
    pub joint_limits_rad: [[f64; 2]; N],
    pub velocity_limits_rad_s: [f64; N],
}
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Snapshot<const N: usize> {
    pub simulation_ns: u64,
    pub q_rad: [f64; N],
    pub dq_rad_s: [f64; N],
    pub tool_base: Pose,
    pub cube_base: Pose,
    pub attached: bool,
    /// Driver must classify actual recorded contacts using the pinned model.
    pub forbidden_contacts: u32,
}
#[derive(Debug)]
pub enum Action<const N: usize> {
    Plan(Request<N>),
    SetAttachment(bool),
    Complete,
}

fn foreign(p: Pose) -> ForeignPose {
    ForeignPose {
        translation_m: p.translation_m,
        quaternion_xyzw: p.quaternion_xyzw,
    }
}
fn error(a: Pose, b: Pose) -> Option<(f64, f64)> {
    if !a.valid() || !b.valid() {
        return None;
    }
    let distance = a
        .translation_m
        .iter()
        .zip(b.translation_m)
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f64>()
        .sqrt();
    let same = a
        .quaternion_xyzw
        .iter()
        .zip(b.quaternion_xyzw)
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f64>()
        .sqrt();
    let opposite = a
        .quaternion_xyzw
        .iter()
        .zip(b.quaternion_xyzw)
        .map(|(x, y)| (x + y).powi(2))
        .sum::<f64>()
        .sqrt();
    let angle = 4.0 * same.min(opposite).atan2(same.max(opposite));
    if distance.is_finite() && angle.is_finite() {
        Some((distance, angle))
    } else {
        None
    }
}
fn within(a: Pose, b: Pose, m: f64, r: f64) -> bool {
    error(a, b).is_some_and(|(p, q)| p <= m && q <= r)
}
fn q_near<const N: usize>(a: &[f64; N], b: &[f64; N]) -> bool {
    a.iter()
        .zip(b)
        .all(|(a, b)| a.is_finite() && b.is_finite() && (a - b).abs() <= START_RAD)
}

/// CPU task preparation engine; no driver/source binding is adopted here.
/// Storage is fixed by joint count and no trajectory is retained. The driver
/// must bind independent last-q FK, telemetry identity, contact classification,
/// watchdogs and recording. All active failures latch the first typed error.
pub struct Sequencer<const N: usize> {
    config: Config<N>,
    phase: Phase,
    started_ns: Option<u64>,
    last: Option<Snapshot<N>>,
    last_event_ns: Option<u64>,
    last_request_ns: Option<u64>,
    requested_start: [f64; N],
    requested_goal: Pose,
    end_ns: u64,
    attachment_pose: Pose,
    attachment_ack: bool,
    attachment_ack_ns: u64,
    attachment_command_ns: u64,
    settle_reference: Pose,
    settle_since_ns: u64,
}
impl<const N: usize> Sequencer<N> {
    /// Reject unsupported counts, invalid poses/limits or an out-of-range home.
    pub fn new(config: Config<N>) -> Result<Self, Error> {
        if !matches!(N, 6 | 7)
            || [
                config.home_tool,
                config.pregrasp_tool,
                config.grasp_tool,
                config.final_cube,
            ]
            .iter()
            .any(|p| !p.valid())
            || config
                .joint_limits_rad
                .iter()
                .any(|[a, b]| !a.is_finite() || !b.is_finite() || a >= b)
            || config
                .velocity_limits_rad_s
                .iter()
                .any(|x| !x.is_finite() || *x <= 0.0)
            || config
                .home_q_rad
                .iter()
                .zip(&config.joint_limits_rad)
                .any(|(q, [a, b])| !q.is_finite() || q < a || q > b)
        {
            return Err(Error::Config);
        }
        Ok(Self {
            config,
            phase: Phase::Ready,
            started_ns: None,
            last: None,
            last_event_ns: None,
            last_request_ns: None,
            requested_start: [0.0; N],
            requested_goal: Pose::IDENTITY,
            end_ns: 0,
            attachment_pose: Pose::IDENTITY,
            attachment_ack: false,
            attachment_ack_ns: 0,
            attachment_command_ns: 0,
            settle_reference: Pose::IDENTITY,
            settle_since_ns: 0,
        })
    }
    pub fn phase(&self) -> Phase {
        self.phase
    }
    fn fail<T>(&mut self, error: Error) -> Result<T, Error> {
        self.phase = Phase::Failed(error);
        Err(error)
    }
    fn active(&self) -> Result<(), Error> {
        match self.phase {
            Phase::Failed(e) => Err(e),
            Phase::Complete => Err(Error::Lifecycle),
            _ => Ok(()),
        }
    }
    fn time(&mut self, now: u64) -> Result<(), Error> {
        self.active()?;
        if self.last_event_ns.is_some_and(|last| now < last) {
            return self.fail(Error::Clock);
        }
        if self.started_ns.is_some_and(|start| {
            now.checked_sub(start)
                .is_none_or(|elapsed| elapsed > MAX_SIM_NS)
        }) {
            return self.fail(Error::Deadline);
        }
        self.last_event_ns = Some(now);
        Ok(())
    }
    /// Missing telemetry/child failure must become this terminal driver event.
    pub fn planner_failed(&mut self) -> Result<(), Error> {
        self.active()?;
        self.fail(Error::Planner)
    }
    /// A queued service acknowledgement requires strictly later observed
    /// physics telemetry before another move. Duplicate/failed replies latch.
    pub fn attachment_response(&mut self, success: bool, receipt_ns: u64) -> Result<(), Error> {
        self.time(receipt_ns)?;
        if !matches!(self.phase, Phase::Attachment(_)) || self.attachment_ack {
            return self.fail(Error::Lifecycle);
        }
        if !success {
            return self.fail(Error::Attachment);
        }
        self.attachment_ack = true;
        self.attachment_ack_ns = receipt_ns;
        Ok(())
    }
    /// Gate a checked planner trajectory using independent FK on its last q.
    /// Driver publishes to the controller only after this succeeds.
    pub fn planned(
        &mut self,
        trajectory: &Trajectory<N>,
        last_fk: Pose,
        receipt_ns: u64,
    ) -> Result<(), Error> {
        self.time(receipt_ns)?;
        let Phase::Planning(movement) = self.phase else {
            return self.fail(Error::Lifecycle);
        };
        let points = trajectory.waypoints();
        let (Some(first), Some(last)) = (points.first(), points.last()) else {
            return self.fail(Error::Plan);
        };
        if !(2..=256).contains(&points.len())
            || !q_near(&first.q_rad, &self.requested_start)
            || self
                .last
                .is_none_or(|sample| !q_near(&sample.q_rad, &self.requested_start))
            || points.iter().any(|p| {
                p.q_rad
                    .iter()
                    .zip(&self.config.joint_limits_rad)
                    .any(|(q, [a, b])| !q.is_finite() || q < a || q > b)
                    || p.dq_rad_s
                        .iter()
                        .zip(&self.config.velocity_limits_rad_s)
                        .any(|(v, m)| !v.is_finite() || v.abs() > *m)
            })
        {
            return self.fail(Error::Plan);
        }
        // Home is planned in joint space to the exact home joints (decision D4).
        if !within(
            last_fk,
            self.requested_goal,
            0.002,
            std::f64::consts::PI / 180.0,
        ) || (movement == Move::Home && !q_near(&last.q_rad, &self.config.home_q_rad))
        {
            return self.fail(Error::Endpoint);
        }
        let duration = (last.time_from_start_s * 1e9).ceil();
        if !duration.is_finite() || duration <= 0.0 || duration > MAX_SIM_NS as f64 {
            return self.fail(Error::Plan);
        }
        let Some(end) = receipt_ns.checked_add(duration as u64) else {
            return self.fail(Error::Clock);
        };
        if self.started_ns.is_some_and(|start| {
            end.checked_sub(start)
                .is_none_or(|elapsed| elapsed > MAX_SIM_NS)
        }) {
            return self.fail(Error::Deadline);
        }
        self.end_ns = end;
        self.phase = Phase::Executing(movement);
        Ok(())
    }
    /// Accept one coherent source-frame snapshot. At most one action is emitted;
    /// duplicate snapshots are inert, conflicting/superseded stamps fail.
    pub fn observe(&mut self, sample: Snapshot<N>) -> Result<Option<Action<N>>, Error> {
        self.time(sample.simulation_ns)?;
        if sample
            .q_rad
            .iter()
            .zip(&self.config.joint_limits_rad)
            .any(|(q, [a, b])| !q.is_finite() || q < a || q > b)
            || sample
                .dq_rad_s
                .iter()
                .zip(&self.config.velocity_limits_rad_s)
                .any(|(v, m)| !v.is_finite() || v.abs() > *m)
            || !sample.tool_base.valid()
            || !sample.cube_base.valid()
        {
            return self.fail(Error::Input);
        }
        if sample.forbidden_contacts != 0 {
            return self.fail(Error::Contact);
        }
        if let Some(previous) = self.last
            && previous.simulation_ns == sample.simulation_ns
        {
            return if previous == sample {
                Ok(None)
            } else {
                self.fail(Error::Input)
            };
        }
        self.last = Some(sample);
        let expected_attached = match self.phase {
            Phase::NeedPlan(m) | Phase::Planning(m) | Phase::Executing(m) => {
                world::held(m.world_phase())
            }
            phase => phase == Phase::Settling,
        };
        if !matches!(self.phase, Phase::Attachment(_)) && sample.attached != expected_attached {
            return self.fail(Error::Attachment);
        }
        match self.phase {
            Phase::Ready => {
                if !q_near(&sample.q_rad, &self.config.home_q_rad)
                    || !within(
                        sample.tool_base,
                        self.config.home_tool,
                        0.002,
                        std::f64::consts::PI / 180.0,
                    )
                {
                    return self.fail(Error::Home);
                }
                self.started_ns = Some(sample.simulation_ns);
                self.phase = Phase::NeedPlan(Move::Pregrasp);
            }
            Phase::Executing(movement)
                if sample.simulation_ns >= self.end_ns
                    && within(
                        sample.tool_base,
                        self.requested_goal,
                        0.002,
                        std::f64::consts::PI / 180.0,
                    ) =>
            {
                match movement {
                    Move::Pregrasp => self.phase = Phase::NeedPlan(Move::Descend),
                    Move::Descend => {
                        self.phase = Phase::Attachment(true);
                        self.attachment_ack = false;
                        self.attachment_command_ns = sample.simulation_ns;
                        return Ok(Some(Action::SetAttachment(true)));
                    }
                    Move::Lift => self.phase = Phase::NeedPlan(Move::Transfer),
                    Move::Transfer => self.phase = Phase::NeedPlan(Move::PlaceDescend),
                    Move::Retreat => self.phase = Phase::NeedPlan(Move::Home),
                    Move::PlaceDescend => {
                        if !within(
                            sample.cube_base,
                            self.config.final_cube,
                            0.005,
                            std::f64::consts::PI / 90.0,
                        ) {
                            return Ok(None);
                        }
                        self.phase = Phase::Attachment(false);
                        self.attachment_ack = false;
                        self.attachment_command_ns = sample.simulation_ns;
                        return Ok(Some(Action::SetAttachment(false)));
                    }
                    Move::Home => {
                        // Conservative preparation control: same configured home
                        // joints, not merely another IK solution for its pose.
                        if !q_near(&sample.q_rad, &self.config.home_q_rad)
                            || !within(
                                sample.cube_base,
                                self.config.final_cube,
                                0.005,
                                std::f64::consts::PI / 90.0,
                            )
                        {
                            return Ok(None);
                        }
                        self.phase = Phase::Complete;
                        return Ok(Some(Action::Complete));
                    }
                }
            }
            Phase::Attachment(requested)
                if self.attachment_ack
                    && sample.simulation_ns
                        > self.attachment_command_ns.max(self.attachment_ack_ns)
                    && sample.attached == requested =>
            {
                if requested {
                    let Ok(pose) = relation(&sample) else {
                        return self.fail(Error::Input);
                    };
                    self.settle_reference = pose;
                    self.settle_since_ns = sample.simulation_ns;
                    self.phase = Phase::Settling;
                } else {
                    self.phase = Phase::NeedPlan(Move::Retreat);
                }
            }
            Phase::Settling => {
                let Ok(pose) = relation(&sample) else {
                    return self.fail(Error::Input);
                };
                if !within(
                    pose,
                    self.settle_reference,
                    SETTLE_TRANSLATION_M,
                    SETTLE_ROTATION_RAD,
                ) {
                    self.settle_reference = pose;
                    self.settle_since_ns = sample.simulation_ns;
                } else if sample.simulation_ns - self.settle_since_ns >= SETTLE_WINDOW_NS {
                    self.attachment_pose = pose;
                    self.phase = Phase::NeedPlan(Move::Lift);
                }
                if self.phase == Phase::Settling
                    && sample.simulation_ns - self.attachment_ack_ns > SETTLE_TIMEOUT_NS
                {
                    return self.fail(Error::Settle);
                }
            }
            _ => {}
        }
        let Phase::NeedPlan(movement) = self.phase else {
            return Ok(None);
        };
        if self
            .last_request_ns
            .is_some_and(|last| sample.simulation_ns - last < REQUEST_PERIOD_NS)
        {
            return Ok(None);
        }
        let Ok(place_tool) = self
            .attachment_pose
            .inverse()
            .and_then(|p| self.config.final_cube.compose(p))
        else {
            return self.fail(Error::Input);
        };
        let goal = match movement {
            Move::Pregrasp | Move::Lift => self.config.pregrasp_tool,
            Move::Descend => self.config.grasp_tool,
            Move::Transfer | Move::Retreat => above(place_tool),
            Move::PlaceDescend => place_tool,
            Move::Home => self.config.home_tool,
        };
        let phase = movement.world_phase();
        let tool_from_cube = if world::held(phase) {
            self.attachment_pose
        } else {
            let Ok(pose) = relation(&sample) else {
                return self.fail(Error::Input);
            };
            pose
        };
        self.requested_start = sample.q_rad;
        self.requested_goal = goal;
        self.last_request_ns = Some(sample.simulation_ns);
        self.phase = Phase::Planning(movement);
        Ok(Some(Action::Plan(Request {
            start_q_rad: sample.q_rad,
            goal_q_rad: if movement == Move::Home {
                self.config.home_q_rad
            } else {
                sample.q_rad
            },
            target_base_from_tool: foreign(goal),
            cube_base_pose: foreign(sample.cube_base),
            tool_from_cube: foreign(tool_from_cube),
            phase,
        })))
    }
}
/// Measured tool-from-cube relation of one coherent snapshot.
fn relation<const N: usize>(sample: &Snapshot<N>) -> Result<Pose, &'static str> {
    sample
        .tool_base
        .inverse()
        .and_then(|p| p.compose(sample.cube_base))
}
fn above(pose: Pose) -> Pose {
    let mut raised = pose;
    raised.translation_m[2] += CLEARANCE_Z_M;
    raised
}
