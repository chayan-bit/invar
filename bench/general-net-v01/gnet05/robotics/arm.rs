//! Binary32 tables derived once from a loaded model. The Invar programs and
//! the handwritten baselines consume exactly these values.
use super::model::{Axis, Model};

#[derive(Clone, Debug, PartialEq)]
pub struct Arm {
    pub name: &'static str,
    pub bodies: usize,
    pub joints: usize,
    /// Parent body per body (0 for the base).
    pub parent: Vec<u32>,
    /// Fixed parent-to-joint frame per body: row-major R then p (12 each).
    pub fixed: Vec<f32>,
    /// Hinge axis per body: 0 none, 1 x, 2 y, 3 z.
    pub axis: Vec<u32>,
    /// Joint index per body (0 when the body has no hinge).
    pub joint: Vec<u32>,
    /// Per joint: (body, axis code).
    pub joint_body: Vec<(u32, u32)>,
    pub lo: Vec<f32>,
    pub hi: Vec<f32>,
    /// `hi - lo` rounded once from binary64.
    pub span: Vec<f32>,
    pub armature: Vec<f32>,
    pub tool_body: u32,
    /// Tool point offset in the tool body frame.
    pub tool: [f32; 3],
    /// Per body: mass, com (3), body-axes inertia about the com (9, row-major).
    pub inertial: Vec<f32>,
    /// Per body and joint: 1 when the joint moves the body.
    pub moves: Vec<u32>,
    /// Per geom: body, kind (0 capsule, 1 cylinder).
    pub geom_index: Vec<u32>,
    /// Per geom: radius, half length, 2 * half length, 1 / (2 h)^2,
    /// centre (3) and unit axis (3) in the body frame.
    pub geom: Vec<f32>,
    pub geom_names: Vec<String>,
}

pub const GEOM_WORDS: usize = 10;

impl Arm {
    pub fn of(m: &Model) -> Self {
        let f = |x: f64| x as f32;
        let mut a = Arm {
            name: m.name,
            bodies: m.bodies.len(),
            joints: 0,
            parent: Vec::new(),
            fixed: Vec::new(),
            axis: Vec::new(),
            joint: Vec::new(),
            joint_body: Vec::new(),
            lo: Vec::new(),
            hi: Vec::new(),
            span: Vec::new(),
            armature: Vec::new(),
            tool_body: m.tool.0 as u32,
            tool: m.tool.1.p.map(f),
            inertial: Vec::new(),
            moves: Vec::new(),
            geom_index: Vec::new(),
            geom: Vec::new(),
            geom_names: Vec::new(),
        };
        for (i, b) in m.bodies.iter().enumerate() {
            a.parent.push(b.parent.unwrap_or(0) as u32);
            a.fixed.extend(b.fixed.r.iter().flatten().map(|&x| f(x)));
            a.fixed.extend(b.fixed.p.map(f));
            match &b.hinge {
                Some((_, axis, lo, hi, armature)) => {
                    let code = match axis {
                        Axis::X => 1,
                        Axis::Y => 2,
                        Axis::Z => 3,
                    };
                    a.axis.push(code);
                    a.joint.push(a.joints as u32);
                    a.joint_body.push((i as u32, code));
                    a.lo.push(f(*lo));
                    a.hi.push(f(*hi));
                    a.span.push(f(hi - lo));
                    a.armature.push(f(*armature));
                    a.joints += 1;
                }
                None => {
                    a.axis.push(0);
                    a.joint.push(0);
                }
            }
            a.inertial.push(f(b.mass));
            a.inertial.extend(b.com.map(f));
            a.inertial.extend(b.inertia.iter().flatten().map(|&x| f(x)));
        }
        for b in 0..a.bodies {
            for &(jb, _) in &a.joint_body {
                a.moves.push(u32::from(m.moves(jb as usize, b)));
            }
        }
        for (k, g) in m.geoms.iter().enumerate() {
            a.geom_index.extend([g.body as u32, u32::from(!g.capsule)]);
            let h2 = 2.0 * g.half_length;
            a.geom
                .extend([f(g.radius), f(g.half_length), f(h2), f(1.0 / (h2 * h2))]);
            a.geom.extend(g.center.map(f));
            a.geom.extend(g.axis.map(f));
            a.geom_names.push(format!("{}#{k}", m.bodies[g.body].name));
        }
        a
    }
    pub fn geoms(&self) -> usize {
        self.geom_index.len() / 2
    }
}
