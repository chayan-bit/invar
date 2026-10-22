//! Typed planner-v1 lifecycle boundary. Unsafe calls remain in this boundary crate.
//! A selected library must uphold the generated C signatures and pointer lifetimes.
use std::{collections::BTreeSet, path::Path};

use super::{Component, ComponentError, Pose, State, path_bytes};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PlannerState {
    Loaded,
    Initialized,
    Configured,
    Failed,
    Shutdown,
}

/// Planner world phase of one request (planner-component-v2). The child keeps
/// one world model per phase; nothing is inferred from the request poses.
pub mod phase {
    /// Cube is a world obstacle, nothing is held; pose goal.
    pub const FREE: u32 = 0;
    /// Cube obstacle disabled for the final approach or the retreat; pose goal.
    pub const APPROACH: u32 = 1;
    /// Cube attached to the tool with a full sphere cover; pose goal.
    pub const HELD: u32 = 2;
    /// Cube attached; only its table-contact slab is left uncovered; pose goal.
    pub const HELD_CONTACT: u32 = 3;
    /// Cube is a world obstacle, nothing is held; joint goal `goal_q_rad`.
    pub const HOME: u32 = 4;
    pub fn held(phase: u32) -> bool {
        matches!(phase, HELD | HELD_CONTACT)
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct Request<const N: usize> {
    pub start_q_rad: [f64; N],
    /// Joint goal for `phase::HOME`; every other phase repeats `start_q_rad`.
    pub goal_q_rad: [f64; N],
    pub target_base_from_tool: Pose,
    pub cube_base_pose: Pose,
    pub tool_from_cube: Pose,
    pub phase: u32,
}

impl<const N: usize> Request<N> {
    /// Phase, goal-joint and pose admission shared by every planner boundary.
    pub fn valid(&self, limits: &[[f64; 2]; N]) -> bool {
        let inside = |q: &[f64; N]| {
            q.iter()
                .zip(limits)
                .all(|(q, [lo, hi])| q.is_finite() && q >= lo && q <= hi)
        };
        self.phase <= phase::HOME
            && inside(&self.start_q_rad)
            && inside(&self.goal_q_rad)
            && (self.phase == phase::HOME
                || self
                    .goal_q_rad
                    .iter()
                    .zip(&self.start_q_rad)
                    .all(|(a, b)| a.to_bits() == b.to_bits()))
            && valid_pose(self.target_base_from_tool)
            && valid_pose(self.cube_base_pose)
            && valid_pose(self.tool_from_cube)
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct Waypoint<const N: usize> {
    pub time_from_start_s: f64,
    pub q_rad: [f64; N],
    pub dq_rad_s: [f64; N],
}

#[repr(C)]
pub struct Trajectory<const N: usize> {
    dt_s: f64,
    count: u32,
    waypoints: [Waypoint<N>; 256],
}

fn valid_points<const N: usize>(
    dt_s: f64,
    points: &[Waypoint<N>],
    start: &[f64; N],
    limits: &[[f64; 2]; N],
    speeds: &[f64; N],
) -> bool {
    if !matches!(N, 6 | 7)
        || !(2..=256).contains(&points.len())
        || !dt_s.is_finite()
        || dt_s <= 0.0
        || limits
            .iter()
            .any(|[lo, hi]| !lo.is_finite() || !hi.is_finite() || lo >= hi)
        || speeds.iter().any(|x| !x.is_finite() || *x <= 0.0)
        || start
            .iter()
            .zip(limits)
            .any(|(q, [lo, hi])| !q.is_finite() || q < lo || q > hi)
        || points[0]
            .q_rad
            .iter()
            .zip(start)
            .any(|(q, s)| !q.is_finite() || (q - s).abs() > 1e-6)
    {
        return false;
    }
    let mut previous = -1.0;
    for p in points {
        if !p.time_from_start_s.is_finite()
            || p.time_from_start_s < 0.0
            || p.time_from_start_s <= previous
            || p.q_rad
                .iter()
                .zip(limits)
                .any(|(q, [lo, hi])| !q.is_finite() || q < lo || q > hi)
            || p.dq_rad_s
                .iter()
                .zip(speeds)
                .any(|(v, m)| !v.is_finite() || v.abs() > *m)
        {
            return false;
        }
        previous = p.time_from_start_s;
    }
    true
}

impl<const N: usize> Trajectory<N> {
    /// Checked ingress for a copied ROS/private-protocol trajectory. Uses the
    /// same point/start contract as the C boundary, without publishing partial
    /// output. The caller supplies its checked robot limits and request start.
    pub fn checked(
        dt_s: f64,
        points: &[Waypoint<N>],
        start: &[f64; N],
        limits: &[[f64; 2]; N],
        speeds: &[f64; N],
    ) -> Result<Self, ComponentError> {
        if !valid_points(dt_s, points, start, limits, speeds) {
            return Err(ComponentError::Output);
        }
        let mut output = Self {
            dt_s,
            count: points.len() as u32,
            waypoints: [Waypoint {
                time_from_start_s: 0.0,
                q_rad: [0.0; N],
                dq_rad_s: [0.0; N],
            }; 256],
        };
        output.waypoints[..points.len()].copy_from_slice(points);
        Ok(output)
    }
    pub fn dt_s(&self) -> f64 {
        self.dt_s
    }
    pub fn waypoints(&self) -> &[Waypoint<N>] {
        // Construction is private; plan validates count before publishing Self.
        &self.waypoints[..self.count as usize]
    }
}

pub struct Config<'a> {
    pub model_sha256: [u8; 32],
    pub robot_config_sha256: [u8; 32],
    pub scene_sha256: [u8; 32],
    pub joint_limits_rad: &'a [[f64; 2]],
    pub velocity_limits_rad_s: &'a [f64],
    pub table_pose: Pose,
    pub table_half_extents_m: [f64; 3],
    pub cube_edge_m: f64,
    pub base_frame: &'a str,
    pub tool_frame: &'a str,
    pub joint_names: &'a [&'a str],
    pub robot_config_path: &'a Path,
    pub scene_config_path: &'a Path,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct Text {
    bytes: *const u8,
    length: u16,
}

#[repr(C)]
struct ConfigWire {
    robot_kind: u32,
    joint_count: u32,
    model_sha256: [u8; 32],
    robot_config_sha256: [u8; 32],
    scene_sha256: [u8; 32],
    joint_limits_rad: [[f64; 2]; 7],
    table_pose: Pose,
    table_half_extents_m: [f64; 3],
    cube_edge_m: f64,
    base_frame: Text,
    tool_frame: Text,
    joint_names: [Text; 7],
    robot_config_path: Text,
    scene_config_path: Text,
}

const _: () = {
    assert!(std::mem::size_of::<Text>() == 16);
    assert!(std::mem::size_of::<ConfigWire>() == 480);
    assert!(std::mem::offset_of!(ConfigWire, joint_limits_rad) == 104);
    assert!(std::mem::offset_of!(ConfigWire, table_pose) == 216);
    assert!(std::mem::offset_of!(ConfigWire, joint_names) == 336);
    assert!(std::mem::offset_of!(ConfigWire, robot_config_path) == 448);
    assert!(std::mem::size_of::<Request<6>>() == 272);
    assert!(std::mem::size_of::<Request<7>>() == 288);
    assert!(std::mem::offset_of!(Request<7>, target_base_from_tool) == 112);
    assert!(std::mem::offset_of!(Request<7>, phase) == 280);
    assert!(std::mem::size_of::<Trajectory<6>>() == 26640);
    assert!(std::mem::size_of::<Trajectory<7>>() == 30736);
    assert!(std::mem::offset_of!(Trajectory<6>, waypoints) == 16);
    assert!(std::mem::offset_of!(Trajectory<7>, waypoints) == 16);
};

fn text(bytes: &[u8], limit: usize) -> Result<Text, ComponentError> {
    if bytes.is_empty()
        || bytes.len() > limit
        || bytes.len() > u16::MAX as usize
        || bytes.contains(&0)
    {
        return Err(ComponentError::Input);
    }
    Ok(Text {
        bytes: bytes.as_ptr(),
        length: bytes.len() as u16,
    })
}

fn valid_pose(pose: Pose) -> bool {
    let norm: f64 = pose.quaternion_xyzw.iter().map(|x| x * x).sum();
    pose.translation_m
        .iter()
        .chain(&pose.quaternion_xyzw)
        .all(|x| x.is_finite())
        && norm.is_finite()
        && (norm.sqrt() - 1.0).abs() <= 1e-12
}

pub struct Planner<const N: usize> {
    core: Component,
    state: PlannerState,
    limits: [[f64; 2]; N],
    velocities: [f64; N],
}

impl<const N: usize> Planner<N> {
    /// Only explicitly selected libraries with the exact joint-count ABI marker
    /// are loaded. The FK component loader rejects this planner-specific marker.
    pub fn open(path: &Path, prefix: &str) -> Result<Self, ComponentError> {
        if !matches!(N, 6 | 7) {
            return Err(ComponentError::Input);
        }
        let core =
            Component::open_expected(path, prefix, N, &format!("invar-planner-component-v2/{N}/"))?;
        Ok(Self {
            core,
            state: PlannerState::Loaded,
            limits: [[0.0; 2]; N],
            velocities: [0.0; N],
        })
    }

    pub fn state(&self) -> PlannerState {
        self.state
    }
    pub fn version(&self) -> &str {
        self.core.version()
    }

    pub fn init(&mut self, child: &Path) -> Result<(), ComponentError> {
        if self.state != PlannerState::Loaded {
            return Err(ComponentError::Lifecycle);
        }
        let result = self.core.init(Some(child));
        self.state = if result.is_ok() {
            PlannerState::Initialized
        } else {
            PlannerState::Failed
        };
        result
    }

    pub fn configure(&mut self, config: &Config<'_>) -> Result<(), ComponentError> {
        if self.state != PlannerState::Initialized {
            return Err(ComponentError::Lifecycle);
        }
        if !matches!(N, 6 | 7)
            || config.joint_limits_rad.len() != N
            || config.velocity_limits_rad_s.len() != N
            || config.joint_names.len() != N
            || !valid_pose(config.table_pose)
            || config.cube_edge_m != 0.05
            || config
                .table_half_extents_m
                .iter()
                .any(|x| !x.is_finite() || *x <= 0.0)
            || config
                .joint_limits_rad
                .iter()
                .any(|[lo, hi]| !lo.is_finite() || !hi.is_finite() || lo >= hi)
            || config
                .velocity_limits_rad_s
                .iter()
                .any(|x| !x.is_finite() || *x <= 0.0)
            || config
                .joint_names
                .iter()
                .copied()
                .collect::<BTreeSet<_>>()
                .len()
                != N
        {
            return Err(ComponentError::Input);
        }
        let robot_path = path_bytes(config.robot_config_path)?;
        let scene_path = path_bytes(config.scene_config_path)?;
        let mut wire = ConfigWire {
            robot_kind: N as u32,
            joint_count: N as u32,
            model_sha256: config.model_sha256,
            robot_config_sha256: config.robot_config_sha256,
            scene_sha256: config.scene_sha256,
            joint_limits_rad: [[0.0; 2]; 7],
            table_pose: config.table_pose,
            table_half_extents_m: config.table_half_extents_m,
            cube_edge_m: config.cube_edge_m,
            base_frame: text(config.base_frame.as_bytes(), 64)?,
            tool_frame: text(config.tool_frame.as_bytes(), 64)?,
            joint_names: [Text {
                bytes: std::ptr::null(),
                length: 0,
            }; 7],
            robot_config_path: text(&robot_path, 4096)?,
            scene_config_path: text(&scene_path, 4096)?,
        };
        for i in 0..N {
            wire.joint_limits_rad[i] = config.joint_limits_rad[i];
            wire.joint_names[i] = text(config.joint_names[i].as_bytes(), 64)?;
        }
        // SAFETY: open verified the planner/N ABI marker; ConfigWire has its fixed
        // 480-byte C layout. All borrowed paths/names live through this call only.
        let status =
            unsafe { (self.core.configure_fn)(self.core.context, (&raw const wire).cast()) };
        if status != 0 {
            self.state = PlannerState::Failed;
            return Err(ComponentError::Status(status));
        }
        self.limits.copy_from_slice(config.joint_limits_rad);
        self.velocities
            .copy_from_slice(config.velocity_limits_rad_s);
        self.core.state = State::Configured;
        self.state = PlannerState::Configured;
        Ok(())
    }

    pub fn plan(&mut self, request: &Request<N>) -> Result<Trajectory<N>, ComponentError> {
        if self.state != PlannerState::Configured {
            return Err(ComponentError::Lifecycle);
        }
        if !request.valid(&self.limits) {
            return Err(ComponentError::Input);
        }
        let mut result = Trajectory {
            dt_s: f64::NAN,
            count: 0,
            waypoints: [Waypoint {
                time_from_start_s: f64::NAN,
                q_rad: [f64::NAN; N],
                dq_rad_s: [f64::NAN; N],
            }; 256],
        };
        // SAFETY: Request<N>/Trajectory<N> match the exact marker verified on
        // open. Both pointers stay live and exclusive as required; C may not
        // retain them. The temporary is never published before full validation.
        let status = unsafe {
            (self.core.step_fn)(
                self.core.context,
                (&raw const *request).cast(),
                (&raw mut result).cast(),
            )
        };
        if status != 0 {
            self.state = PlannerState::Failed;
            return Err(ComponentError::Status(status));
        }
        let valid = self.valid_trajectory(&result, request);
        if !valid {
            self.state = PlannerState::Failed;
            return Err(ComponentError::Output);
        }
        Ok(result)
    }

    fn valid_trajectory(&self, result: &Trajectory<N>, request: &Request<N>) -> bool {
        (2..=256).contains(&result.count)
            && valid_points(
                result.dt_s,
                &result.waypoints[..result.count as usize],
                &request.start_q_rad,
                &self.limits,
                &self.velocities,
            )
    }

    pub fn shutdown(&mut self) -> Result<(), ComponentError> {
        if !matches!(
            self.state,
            PlannerState::Initialized | PlannerState::Configured | PlannerState::Failed
        ) {
            return Err(ComponentError::Lifecycle);
        }
        if self.core.context.is_null() && self.state == PlannerState::Failed {
            self.core.state = State::Shutdown;
            self.state = PlannerState::Shutdown;
            return Ok(());
        }
        let result = self.core.shutdown();
        self.state = if self.core.state == State::Shutdown {
            PlannerState::Shutdown
        } else {
            PlannerState::Failed
        };
        result
    }
}
