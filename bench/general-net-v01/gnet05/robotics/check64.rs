//! Independent binary64 checks: kinematics, Jacobian, mass and exact
//! primitive distances straight from the MJCF model (no binary32 tables),
//! plus exact binary32 replays of the seeded input streams.
use super::model::{Axis, Frame, Mat, Model, Vec3, apply, mul};

/// World frame of every body at joint angles `q` (radians, active joints).
pub fn fk(m: &Model, q: &[f64]) -> Vec<Frame> {
    let mut out: Vec<Frame> = Vec::with_capacity(m.bodies.len());
    let mut j = 0;
    for b in &m.bodies {
        let mut f = match b.parent {
            None => b.fixed,
            Some(p) => Frame {
                r: mul(&out[p].r, &b.fixed.r),
                p: add(&apply(&out[p].r, &b.fixed.p), &out[p].p),
            },
        };
        if let Some((_, axis, ..)) = &b.hinge {
            let (s, c) = q[j].sin_cos();
            let rot: Mat = match axis {
                Axis::X => [[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]],
                Axis::Y => [[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]],
                Axis::Z => [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]],
            };
            f.r = mul(&f.r, &rot);
            j += 1;
        }
        out.push(f);
    }
    out
}

fn add(a: &Vec3, b: &Vec3) -> Vec3 {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}
fn sub(a: &Vec3, b: &Vec3) -> Vec3 {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}
fn cross(a: &Vec3, b: &Vec3) -> Vec3 {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}
fn dot(a: &Vec3, b: &Vec3) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// Active joints as (body, world axis, world origin).
fn joints(m: &Model, frames: &[Frame]) -> Vec<(usize, Vec3, Vec3)> {
    m.active()
        .into_iter()
        .map(|b| {
            let k = match m.bodies[b].hinge.as_ref().unwrap().1 {
                Axis::X => 0,
                Axis::Y => 1,
                Axis::Z => 2,
            };
            let r = &frames[b].r;
            (b, [r[0][k], r[1][k], r[2][k]], frames[b].p)
        })
        .collect()
}

pub fn tool_point(m: &Model, frames: &[Frame]) -> Vec3 {
    let f = &frames[m.tool.0];
    add(&apply(&f.r, &m.tool.1.p), &f.p)
}

/// 6xN tool Jacobian (rows vx vy vz wx wy wz), base axes.
pub fn jacobian(m: &Model, frames: &[Frame]) -> Vec<Vec<f64>> {
    let p = tool_point(m, frames);
    let js = joints(m, frames);
    let mut out = vec![vec![0.0; js.len()]; 6];
    for (j, (_, z, o)) in js.iter().enumerate() {
        let v = cross(z, &sub(&p, o));
        for k in 0..3 {
            out[k][j] = v[k];
            out[3 + k][j] = z[k];
        }
    }
    out
}

/// Joint-space mass matrix including armature.
pub fn mass(m: &Model, frames: &[Frame]) -> Vec<Vec<f64>> {
    let js = joints(m, frames);
    let n = js.len();
    let mut out = vec![vec![0.0; n]; n];
    for (b, body) in m.bodies.iter().enumerate() {
        let f = &frames[b];
        let com = add(&apply(&f.r, &body.com), &f.p);
        let inertia = mul(&mul(&f.r, &body.inertia), &transpose(&f.r));
        let cols: Vec<(Vec3, Vec3)> = js
            .iter()
            .map(|(jb, z, o)| {
                if m.moves(*jb, b) {
                    (cross(z, &sub(&com, o)), *z)
                } else {
                    ([0.0; 3], [0.0; 3])
                }
            })
            .collect();
        for i in 0..n {
            for j in 0..n {
                out[i][j] += body.mass * dot(&cols[i].0, &cols[j].0)
                    + dot(&cols[i].1, &apply(&inertia, &cols[j].1));
            }
        }
    }
    for (j, (b, ..)) in js.iter().enumerate() {
        out[j][j] += m.bodies[*b].hinge.as_ref().unwrap().4;
    }
    out
}

fn transpose(a: &Mat) -> Mat {
    [0, 1, 2].map(|i| [0, 1, 2].map(|j| a[j][i]))
}

/// Signed distance from a sphere to geom `g` (exact primitive) at `frames`.
pub fn distance(m: &Model, frames: &[Frame], g: usize, centre: &Vec3, radius: f64) -> f64 {
    let geom = &m.geoms[g];
    let f = &frames[geom.body];
    let c = add(&apply(&f.r, &geom.center), &f.p);
    let a = apply(&f.r, &geom.axis);
    let v = sub(centre, &c);
    let along = dot(&v, &a);
    if geom.capsule {
        let t = along.clamp(-geom.half_length, geom.half_length);
        let e = sub(&v, &a.map(|x| x * t));
        return dot(&e, &e).sqrt() - geom.radius - radius;
    }
    let radial = sub(&v, &a.map(|x| x * along));
    let qx = dot(&radial, &radial).sqrt() - geom.radius;
    let qy = along.abs() - geom.half_length;
    qx.max(0.0).hypot(qy.max(0.0)) + qx.max(qy).min(0.0) - radius
}

/// Exact binary32 replays of the in-source streams.
pub fn lcg(x: u32) -> u32 {
    x.wrapping_mul(1_664_525).wrapping_add(1_013_904_223)
}
pub fn unit(x: u32) -> f32 {
    (x / 256) as f32 * super::emit::constant("unit")
}
pub fn stream(seed: u32, i: u32) -> u32 {
    lcg(i.wrapping_mul(2_654_435_761).wrapping_add(seed))
}
/// Angles of configuration `b` as drawn in source (binary32 then widened),
/// or the rational control's equivalent angle `2 atan(u)`.
pub fn angles(lo: &[f32], span: &[f32], trig: bool, seed: u32, b: u32) -> Vec<f64> {
    let mut x = stream(seed, b);
    (0..lo.len())
        .map(|j| {
            x = lcg(x);
            if trig {
                f64::from(lo[j] + span[j] * unit(x))
            } else {
                2.0 * f64::from(2.0 * unit(x) - 1.0).atan()
            }
        })
        .collect()
}

/// Exact binary32 replay of the in-source `sincos` (cos, sin).
pub fn sincos32(x: f32) -> (f32, f32) {
    let k32 = super::emit::constant;
    let mut k = 0.0f32;
    let mut m = 0u32;
    while x * k32("two_pi") - k > 0.5 {
        k += 1.0;
        m += 1;
    }
    while x * k32("two_pi") - k < -0.5 {
        k -= 1.0;
        m += 3;
    }
    let r = x - k * k32("dp1") - k * k32("dp2") - k * k32("dp3");
    let z = r * r;
    let s = ((k32("s0") * z + k32("s1")) * z - k32("s2")) * z * r + r;
    let c = ((k32("c0") * z - k32("c1")) * z + k32("c2")) * z * z - 0.5 * z + 1.0;
    match m % 4 {
        0 => (c, s),
        1 => (-s, c),
        2 => (-c, -s),
        _ => (s, -c),
    }
}

/// Exact binary32 replay of the rational control map: (cos, sin).
pub fn rational32(u: f32) -> (f32, f32) {
    let d = 1.0 + u * u;
    ((1.0 - u * u) / d, (u + u) / d)
}
