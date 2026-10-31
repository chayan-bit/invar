//! Independent checks for the GNET-05 robotics corpus: binary64 model math
//! per output word, exact binary32 replays of the seeded inputs and of the
//! integer/compaction logic, and the committed Pinocchio references.
#![allow(dead_code)]
use crate::robotics_corpus::{TOL_COST, TOL_DISTANCE, TOL_FRAME, TOL_JACOBIAN, TOL_MASS};
use crate::robotics::{
    arm::Arm,
    check64::{self, lcg, rational32, sincos32, stream, unit},
    corpus::Program,
    families,
    model::{self, Model},
};
use serde_json::Value as Json;
use std::path::Path;

pub struct Models {
    pub panda: (Model, Arm),
    pub ur5e: (Model, Arm),
    reference: Json,
    all_link: Json,
}

fn num(v: &Json) -> f64 {
    v.as_str().unwrap().parse().unwrap()
}

fn max_err(acc: &mut f64, got: f32, want: f64, tol: f64, context: &str) {
    let e = (f64::from(got) - want).abs();
    assert!(e <= tol, "{context}: |{got} - {want}| = {e} > {tol}");
    *acc = acc.max(e);
}

impl Models {
    pub fn load(repo: &Path) -> Self {
        let load = |name| {
            let m = model::load(repo, name);
            let a = Arm::of(&m);
            (m, a)
        };
        let fixtures = repo.join("tests/fixtures/foreign-components/gpu-parity-oracles");
        let json =
            |f: &str| serde_json::from_slice(&std::fs::read(fixtures.join(f)).unwrap()).unwrap();
        Models {
            panda: load("panda"),
            ur5e: load("ur5e"),
            reference: json("reference.json"),
            all_link: json("all_link_reference.json"),
        }
    }
    fn arm(&self, name: &str) -> &(Model, Arm) {
        match name {
            "panda" => &self.panda,
            _ => &self.ur5e,
        }
    }

    /// Checks one output against independent math; returns the max error.
    pub fn check(&self, p: &Program, input: &[u32], out: &[f32], context: &str) -> f64 {
        let (seed, n) = (input[0], input[1] as usize);
        let mut err = 0.0;
        if p.case.family == "pointcloud" {
            let want = pointcloud(p.case.trig, seed, input[1], input[2]);
            assert!(
                want.len() == out.len()
                    && want
                        .iter()
                        .zip(out)
                        .all(|(a, b)| a.to_bits() == b.to_bits()),
                "{context}: exact binary32 replay of transform/filter/compaction/neighbours"
            );
            return 0.0;
        }
        let (m, a) = self.arm(p.case.arm);
        let nj = a.joints;
        let angles = |b: usize| check64::angles(&a.lo, &a.span, p.case.trig, seed, b as u32);
        match p.case.family {
            "fk" | "fkpre" => {
                assert_eq!(out.len(), n * a.bodies * 12, "{context}");
                for b in 0..n {
                    let frames = check64::fk(m, &angles(b));
                    for (i, f) in frames.iter().enumerate() {
                        let want = f.r.iter().flatten().chain(f.p.iter());
                        for (k, w) in want.enumerate() {
                            let got = out[(b * a.bodies + i) * 12 + k];
                            max_err(&mut err, got, *w, TOL_FRAME, context);
                        }
                    }
                }
            }
            "jacobian" => {
                assert_eq!(out.len(), n * 6 * nj, "{context}");
                for b in 0..n {
                    let j = check64::jacobian(m, &check64::fk(m, &angles(b)));
                    for r in 0..6 {
                        for c in 0..nj {
                            max_err(
                                &mut err,
                                out[b * 6 * nj + r * nj + c],
                                j[r][c],
                                TOL_JACOBIAN,
                                context,
                            );
                        }
                    }
                }
            }
            "mass" => {
                assert_eq!(out.len(), n * nj * nj, "{context}");
                for b in 0..n {
                    let mm = check64::mass(m, &check64::fk(m, &angles(b)));
                    for r in 0..nj {
                        for c in 0..nj {
                            max_err(
                                &mut err,
                                out[b * nj * nj + r * nj + c],
                                mm[r][c],
                                TOL_MASS,
                                context,
                            );
                        }
                    }
                }
            }
            "collision" => {
                let obstacles = input[2] as usize;
                let ng = a.geoms();
                assert_eq!(out.len(), n * ng * obstacles, "{context}");
                let obs = spheres(seed, obstacles);
                for b in 0..n {
                    let frames = check64::fk(m, &angles(b));
                    for g in 0..ng {
                        for (k, o) in obs.iter().enumerate() {
                            let c = [o[0], o[1], o[2]].map(f64::from);
                            let d = check64::distance(m, &frames, g, &c, f64::from(o[3]));
                            max_err(
                                &mut err,
                                out[(b * ng + g) * obstacles + k],
                                d,
                                TOL_DISTANCE,
                                context,
                            );
                        }
                    }
                }
            }
            "rollout" => err = rollout(m, a, seed, n, input[2] as usize, out, context),
            "pipeline" => {
                let ng = a.geoms();
                assert_eq!(out.len(), n * (ng + 1), "{context}");
                let (pts, kept) =
                    cloud(p.case.trig, seed, input[2], [1.0, 1.0, 1.2], CROP_PIPELINE);
                let pts = &pts[..3 * kept];
                for b in 0..n {
                    let frames = check64::fk(m, &angles(b));
                    let mut cost = 0.0;
                    for g in 0..ng {
                        let best = pts.chunks(3).fold(1000.0f64, |best, q| {
                            let c = [q[0], q[1], q[2]].map(f64::from);
                            best.min(check64::distance(m, &frames, g, &c, 0.0))
                        });
                        max_err(&mut err, out[b * (ng + 1) + g], best, TOL_DISTANCE, context);
                        let gap = f64::from(families::CLEARANCE) - best;
                        if gap > 0.0 {
                            cost += gap * gap;
                        }
                    }
                    max_err(&mut err, out[b * (ng + 1) + ng], cost, TOL_COST, context);
                }
            }
            other => panic!("family {other}"),
        }
        err
    }

    /// Binary64 model math against Pinocchio for all committed cases.
    pub fn model_matches_pinocchio(&self, tol: f64) {
        for (name, (m, _)) in [("panda", &self.panda), ("ur5e", &self.ur5e)] {
            for case in self.cases(&self.reference, name) {
                let q: Vec<f64> = case["q_rad"].as_array().unwrap().iter().map(num).collect();
                let frames = check64::fk(m, &q);
                let j = check64::jacobian(m, &frames);
                let mm = check64::mass(m, &frames);
                for (r, row) in case["jacobian_pinocchio"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .enumerate()
                {
                    for (c, v) in row.as_array().unwrap().iter().enumerate() {
                        assert!((j[r][c] - num(v)).abs() <= tol, "{name} J[{r}][{c}]");
                    }
                }
                for (r, row) in case["mass_pinocchio"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .enumerate()
                {
                    for (c, v) in row.as_array().unwrap().iter().enumerate() {
                        assert!((mm[r][c] - num(v)).abs() <= tol, "{name} M[{r}][{c}]");
                    }
                }
            }
            for case in self.cases(&self.all_link, name) {
                let q: Vec<f64> = case["q_rad"].as_array().unwrap().iter().map(num).collect();
                let frames = check64::fk(m, &q);
                for (b, body) in case["bodies"].as_array().unwrap().iter().enumerate() {
                    assert_eq!(body["name"], m.bodies[b].name.as_str(), "body order");
                    let (rot, p) = pinocchio_frame(body);
                    for (k, want) in rot.iter().enumerate() {
                        assert!((frames[b].r[k / 3][k % 3] - want).abs() <= tol, "{name} R");
                    }
                    for (got, want) in frames[b].p.iter().zip(p) {
                        assert!((got - want).abs() <= tol, "{name} p");
                    }
                }
            }
        }
    }

    fn cases<'a>(&self, doc: &'a Json, name: &str) -> Vec<&'a Json> {
        let model = doc["models"]
            .as_array()
            .unwrap()
            .iter()
            .find(|m| m["name"] == name)
            .unwrap();
        model["cases"].as_array().unwrap().iter().collect()
    }

    /// Runs `probe` at every committed case; returns the number of cases.
    pub fn probe(&self, p: &Program, probe: &dyn Fn(&[f32]) -> Vec<f32>) -> usize {
        let (_, a) = self.arm(p.case.arm);
        let doc = if p.case.family == "fk" {
            &self.all_link
        } else {
            &self.reference
        };
        let mut count = 0;
        for case in self.cases(doc, p.case.arm) {
            let q: Vec<f32> = case["q_rad"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| num(v) as f32)
                .collect();
            let out = probe(&q);
            let context = format!("{} probe {}", p.id, case["name"]);
            let mut err = 0.0;
            match p.case.family {
                "fk" => {
                    for (b, body) in case["bodies"].as_array().unwrap().iter().enumerate() {
                        let (rot, pos) = pinocchio_frame(body);
                        for k in 0..9 {
                            max_err(&mut err, out[12 * b + k], rot[k], TOL_FRAME, &context);
                        }
                        for k in 0..3 {
                            max_err(&mut err, out[12 * b + 9 + k], pos[k], TOL_FRAME, &context);
                        }
                    }
                }
                family => {
                    let (key, tol) = if family == "mass" {
                        ("mass_pinocchio", TOL_MASS)
                    } else {
                        ("jacobian_pinocchio", TOL_JACOBIAN)
                    };
                    let rows = case[key].as_array().unwrap();
                    let cols = a.joints;
                    for (r, row) in rows.iter().enumerate() {
                        for (c, v) in row.as_array().unwrap().iter().enumerate() {
                            max_err(&mut err, out[r * cols + c], num(v), tol, &context);
                        }
                    }
                }
            }
            println!("gnet05r {context} max_abs_error={err:e}");
            count += 1;
        }
        count
    }
}

fn pinocchio_frame(body: &Json) -> (Vec<f64>, Vec<f64>) {
    let rot = body["pinocchio_rotation_row_major"]
        .as_array()
        .unwrap()
        .iter()
        .flat_map(|r| r.as_array().unwrap().iter().map(num))
        .collect();
    let p = body["pinocchio_translation_m"]
        .as_array()
        .unwrap()
        .iter()
        .map(num)
        .collect();
    (rot, p)
}

/// Obstacle spheres (x, y, z, r) as drawn in source.
fn spheres(seed: u32, count: usize) -> Vec<[f32; 4]> {
    let mut x = stream(seed.wrapping_mul(747_796_405), 0);
    (0..count)
        .map(|_| {
            let mut draw = |scale: f32, offset: f32| {
                x = lcg(x);
                scale * unit(x) + offset
            };
            let cx = draw(1.8, -0.9);
            let cy = draw(1.8, -0.9);
            let cz = draw(1.2, -0.1);
            let r = draw(0.1, 0.02);
            [cx, cy, cz, r]
        })
        .collect()
}

/// Rollout: joint states replayed exactly in binary32; step costs against
/// binary64 kinematics at the recorded states.
fn rollout(m: &Model, a: &Arm, seed: u32, n: usize, h: usize, out: &[f32], context: &str) -> f64 {
    let nj = a.joints;
    assert_eq!(out.len(), n * h * (nj + 1), "{context}");
    let mut x = stream(seed.wrapping_mul(747_796_405), 0);
    let mut draw = |scale: f32, offset: f32| {
        x = lcg(x);
        scale * unit(x) + offset
    };
    let target = [draw(1.2, -0.6), draw(1.2, -0.6), draw(0.8, 0.1)].map(f64::from);
    let (dt, umax, w) = (
        families::ROLLOUT_DT,
        families::ROLLOUT_UMAX,
        families::ROLLOUT_WEIGHT,
    );
    let mut err = 0.0;
    for b in 0..n {
        let mut y = stream(seed, b as u32);
        let mut q: Vec<f32> = (0..nj)
            .map(|j| {
                y = lcg(y);
                a.lo[j] + a.span[j] * unit(y)
            })
            .collect();
        let mut v = vec![0.0f32; nj];
        for k in 0..h {
            let mut uu = 0.0f32;
            for j in 0..nj {
                y = lcg(y);
                let u = (2.0 * unit(y) - 1.0) * umax;
                uu += u * u;
                v[j] += dt * u;
                let mut p = q[j] + dt * v[j];
                if p < a.lo[j] {
                    p = a.lo[j];
                    v[j] = 0.0;
                }
                if p > a.hi[j] {
                    p = a.hi[j];
                    v[j] = 0.0;
                }
                q[j] = p;
            }
            let o = (b * h + k) * (nj + 1);
            for j in 0..nj {
                assert_eq!(
                    out[o + j].to_bits(),
                    q[j].to_bits(),
                    "{context} b={b} k={k} q{j}"
                );
            }
            let qs: Vec<f64> = q.iter().map(|&x| f64::from(x)).collect();
            let tool = check64::tool_point(m, &check64::fk(m, &qs));
            let e: f64 = (0..3).map(|i| (tool[i] - target[i]).powi(2)).sum();
            let want = e + f64::from(w) * f64::from(uu);
            max_err(&mut err, out[o + nj], want, TOL_COST, context);
        }
    }
    err
}

/// Crop boxes: (x, y, z min, z max, r^2 min, r^2 max).
const CROP_CLOUD: [f32; 6] = [1.5, 1.5, 0.05, 1.8, 0.04, 4.0];
const CROP_PIPELINE: [f32; 6] = [0.9, 0.9, 0.0, 1.1, f32::NEG_INFINITY, f32::INFINITY];

/// Sensor frame (cos, sin, tx, ty, tz) as drawn in source.
fn frame(trig: bool, seed: u32) -> [f32; 5] {
    let mut x = stream(seed.wrapping_mul(747_796_405), 0);
    x = lcg(x);
    let (c, s) = if trig {
        sincos32(std::f32::consts::TAU * unit(x) - std::f32::consts::PI)
    } else {
        rational32(2.0 * unit(x) - 1.0)
    };
    let mut draw = |scale: f32, offset: f32| {
        x = lcg(x);
        scale * unit(x) + offset
    };
    [c, s, draw(0.5, -0.25), draw(0.5, -0.25), draw(0.2, 0.0)]
}

/// Transformed points (all n, then compacted in input order) and kept count.
fn cloud(trig: bool, seed: u32, n: u32, extent: [f32; 3], crop: [f32; 6]) -> (Vec<f32>, usize) {
    let fr = frame(trig, seed);
    let mut kept = Vec::new();
    for i in 0..n {
        let mut y = stream(seed, i);
        let mut draw = |scale: f32, offset: f32| {
            y = lcg(y);
            scale * unit(y) + offset
        };
        let px = draw(2.0 * extent[0], -extent[0]);
        let py = draw(2.0 * extent[1], -extent[1]);
        let pz = draw(extent[2], 0.0);
        let w = [
            fr[0] * px - fr[1] * py + fr[2],
            fr[1] * px + fr[0] * py + fr[3],
            pz + fr[4],
        ];
        let r2 = w[0] * w[0] + w[1] * w[1] + w[2] * w[2];
        if w[0] >= -crop[0]
            && w[0] <= crop[0]
            && w[1] >= -crop[1]
            && w[1] <= crop[1]
            && w[2] >= crop[2]
            && w[2] <= crop[3]
            && r2 >= crop[4]
            && r2 <= crop[5]
        {
            kept.extend(w);
        }
    }
    let k = kept.len() / 3;
    (kept, k)
}

fn pointcloud(trig: bool, seed: u32, n: u32, m: u32) -> Vec<f32> {
    let (pts, kept) = cloud(trig, seed, n, [2.0, 2.0, 2.0], CROP_CLOUD);
    let mut out = vec![kept as f32];
    out.extend(&pts);
    let r2 = families::NEIGHBOUR_RADIUS2;
    for a in 0..m as usize {
        if a >= kept {
            out.push(0.0);
            continue;
        }
        let count = (0..kept)
            .filter(|&k| {
                let d: Vec<f32> = (0..3).map(|c| pts[3 * k + c] - pts[3 * a + c]).collect();
                d[0] * d[0] + d[1] * d[1] + d[2] * d[2] <= r2
            })
            .count();
        out.push(count as f32);
    }
    out
}

/// Emitted code with the rename prefix and source spans removed.
pub fn without_names_and_spans(code: &str) -> String {
    crate::guard::without_spans(&code.replace(crate::guard::RENAME_PREFIX, ""))
}

/// Variant name of a typed error.
pub fn class(e: &impl std::fmt::Debug) -> String {
    format!("{e:?}")
        .split(|c: char| !c.is_alphanumeric() && c != '_')
        .next()
        .unwrap_or_default()
        .to_owned()
}
