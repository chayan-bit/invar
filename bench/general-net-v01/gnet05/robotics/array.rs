//! Array-input variants of the robotics families (`gnet05-robotics-array-gen-v1`).
//!
//! Same cases, chains, model tables and in-source math as the seed programs,
//! but the bulk inputs (angles or (cos, sin) pairs, obstacles, controls and
//! target, sensor frame and raw points) arrive as `read Array<f32>` host
//! buffers and the output is written into a host `mut Array<f32>`; `run`
//! returns a `u32` count. `expand` builds those buffers on the host from the
//! same `(seed, sizes)` inputs by exact binary32 replays of the seed
//! programs' streams, so on the same input an array program's output equals
//! its seed program's output bit for bit (point cloud: the prefix the seed
//! program returns; the rest of the zeroed host buffer stays zero).
//! The entry needs host-array parameters (ABI13), so these programs run only
//! on a stack that contains it.
use super::arm::Arm;
use super::check64::{lcg, rational32, stream, unit};
use super::corpus::{self, Program};
use super::emit::{self, Chain};
use super::families::{self, Case};
use crate::generate::{HeldOut, hex, normalized_sha256};
use sha2::{Digest, Sha256};
use std::fmt::Write;
use std::path::Path;

pub const GENERATOR: &str = "gnet05-robotics-array-gen-v1";
pub const REQUIRES: &str = "abi13-host-arrays";

/// One entry argument: a host buffer or a scalar.
#[derive(Clone, Debug, PartialEq)]
pub enum Arg {
    Read(Vec<f32>),
    U32(u32),
    /// The zero-filled output buffer of this length.
    Out(usize),
}

/// Entry parameters of `run`, in order.
pub fn signature(case: &Case) -> &'static str {
    match case.family {
        "collision" => "a: read Array<f32>, obs: read Array<f32>, out: mut Array<f32>",
        "rollout" => {
            "q0: read Array<f32>, u: read Array<f32>, target: read Array<f32>, out: mut Array<f32>"
        }
        "pointcloud" => "fr: read Array<f32>, raw: read Array<f32>, m: u32, out: mut Array<f32>",
        "pipeline" => {
            "a: read Array<f32>, fr: read Array<f32>, raw: read Array<f32>, out: mut Array<f32>"
        }
        _ => "a: read Array<f32>, out: mut Array<f32>",
    }
}

/// The seed programs' prelude without the seeded streams.
fn prelude(trig: bool, root: bool) -> String {
    let p = emit::prelude(trig, root);
    let at = ["fn sincos", "fn root"]
        .iter()
        .filter_map(|f| p.find(f))
        .min()
        .unwrap_or(p.len());
    p[at..].to_owned()
}

/// Per-configuration fill of `cs` from the input `a`: angles through the
/// in-source `sincos`, or the precomputed (cos, sin) pairs copied.
fn fill(trig: bool, nj: usize) -> String {
    if trig {
        format!(
            "for jf in 0u32..{nj}u32 {{
            sincos(a[b * {nj}u32 + jf], cs, jf);
        }}"
        )
    } else {
        format!(
            "for jf in 0u32..{}u32 {{
            cs[jf] = a[b * {}u32 + jf];
        }}",
            2 * nj,
            2 * nj
        )
    }
}

/// Sensor frame `[cos, sin, tx, ty, tz]` from the input `fr`: a yaw through
/// `sincos` then the translation, or the precomputed frame copied.
fn frame_fn(trig: bool) -> &'static str {
    if trig {
        "fn pose(f: mut Array<f32>, fr: read Array<f32>) -> Unit {
    sincos(fr[0u32], f, 0u32);
    f[2u32] = fr[1u32];
    f[3u32] = fr[2u32];
    f[4u32] = fr[3u32];
}
"
    } else {
        "fn pose(f: mut Array<f32>, fr: read Array<f32>) -> Unit {
    for e in 0u32..5u32 {
        f[e] = fr[e];
    }
}
"
    }
}

/// Full program source for one case.
pub fn source(case: &Case, arm: Option<&Arm>) -> String {
    let mut s = format!(
        "// GNET-05 robotics corpus, host-array inputs: {} ({}, {}, {}).\n",
        case.family,
        case.arm,
        if case.trig {
            "trigonometric"
        } else {
            "precomputed transforms"
        },
        families::variant_label(case)
    );
    let trig = case.trig || case.family == "rollout";
    let root = matches!(case.family, "collision" | "pipeline");
    s.push_str(&prelude(trig, root));
    if case.family == "pointcloud" {
        s.push_str(&pointcloud(case));
    } else {
        s.push_str(&arm_program(case, arm.expect("arm")));
    }
    s.push_str("fn main() -> i32 {\n    return 0;\n}\n");
    s
}

fn arm_program(case: &Case, arm: &Arm) -> String {
    let mut s = String::new();
    let chain = emit::chain(arm, Chain::ALL[case.variant]);
    s.push_str(&chain.functions);
    s.push_str(&emit::tool(arm));
    let nj = arm.joints;
    let nb12 = arm.bodies * 12;
    let per = if case.trig { nj } else { 2 * nj };
    let fill = fill(case.trig, nj);
    let x = &chain.extra_args;
    let mut tables = chain.tables.clone();
    let head = format!(
        "    let n = len(a) / {per}u32;
    var cs = filled_f32({}u32, 0.0);
",
        2 * nj
    );
    let body = match case.family {
        "fk" | "fkpre" => format!(
            "{head}    for b in 0u32..n {{
        {fill}
        fk(out, b * {nb12}u32, cs{x});
    }}
    return n;
"
        ),
        "jacobian" => {
            s.push_str(emit::JACOBIAN);
            tables.push_str(&emit::joint_table(arm));
            format!(
                "{head}    var t = filled_f32({nb12}u32, 0.0);
    for b in 0u32..n {{
        {fill}
        fk(t, 0u32, cs{x});
        jacobian(out, b * {}u32, t, jx);
    }}
    return n;
",
                6 * nj
            )
        }
        "mass" => {
            s.push_str(emit::MASS);
            tables.push_str(&emit::joint_table(arm));
            tables.push_str(&emit::mass_tables(arm));
            format!(
                "{head}    var t = filled_f32({nb12}u32, 0.0);
    var sv = filled_f32({}u32, 0.0);
    for b in 0u32..n {{
        {fill}
        fk(t, 0u32, cs{x});
        mass(out, b * {}u32, t, sv, jx, mv, bt, ar);
    }}
    return n;
",
                9 * nj,
                nj * nj
            )
        }
        "collision" => {
            s.push_str(emit::PLACE);
            s.push_str(emit::DISTANCE);
            tables.push_str(&emit::geom_tables(arm));
            let ng = arm.geoms();
            format!(
                    "{head}    let m = len(obs) / 4u32;
    var t = filled_f32({nb12}u32, 0.0);
    var gw = filled_f32({}u32, 0.0);
    for b in 0u32..n {{
        {fill}
        fk(t, 0u32, cs{x});
        place(gw, t, gi, gt);
        for g in 0u32..{ng}u32 {{
            for k in 0u32..m {{
                out[(b * {ng}u32 + g) * m + k] = distance(gw, g, gi, gt, obs[4u32 * k], obs[4u32 * k + 1u32], obs[4u32 * k + 2u32], obs[4u32 * k + 3u32]);
            }}
        }}
    }}
    return n;
",
                    6 * ng
                )
        }
        "rollout" => {
            let limits = emit::limit_tables(arm);
            tables.push_str(limits.lines().next().unwrap());
            tables.push('\n');
            tables.push_str(&emit::hi_table(arm));
            let (dt, wt) = (
                emit::lit(families::ROLLOUT_DT),
                emit::lit(families::ROLLOUT_WEIGHT),
            );
            format!(
                "    if len(q0) == 0u32 {{
        return 0u32;
    }}
    let n = len(q0) / {nj}u32;
    let h = len(u) / len(q0);
    var q = filled_f32({nj}u32, 0.0);
    var v = filled_f32({nj}u32, 0.0);
    var cs = filled_f32({}u32, 0.0);
    var t = filled_f32({nb12}u32, 0.0);
    for b in 0u32..n {{
        for j0 in 0u32..{nj}u32 {{
            q[j0] = q0[b * {nj}u32 + j0];
            v[j0] = 0.0;
        }}
        for k in 0u32..h {{
            var uu = 0.0;
            for j in 0u32..{nj}u32 {{
                let uj = u[(b * h + k) * {nj}u32 + j];
                uu = uu + uj * uj;
                v[j] = v[j] + {dt} * uj;
                var p = q[j] + {dt} * v[j];
                if p < lo[j] {{
                    p = lo[j];
                    v[j] = 0.0;
                }}
                if p > hi[j] {{
                    p = hi[j];
                    v[j] = 0.0;
                }}
                q[j] = p;
                sincos(p, cs, j);
            }}
            fk(t, 0u32, cs{x});
            let ex = tool(t, 0u32) - target[0u32];
            let ey = tool(t, 1u32) - target[1u32];
            let ez = tool(t, 2u32) - target[2u32];
            let o = (b * h + k) * {}u32;
            for j2 in 0u32..{nj}u32 {{
                out[o + j2] = q[j2];
            }}
            out[o + {nj}u32] = ex * ex + ey * ey + ez * ez + {wt} * uu;
        }}
    }}
    return n;
",
                2 * nj,
                nj + 1
            )
        }
        "pipeline" => {
            s.push_str(emit::PLACE);
            s.push_str(emit::DISTANCE);
            s.push_str(frame_fn(case.trig));
            tables.push_str(&emit::geom_tables(arm));
            let ng = arm.geoms();
            let cl = emit::lit(families::CLEARANCE);
            format!(
                    "    var fw = filled_f32(5u32, 0.0);
    pose(fw, fr);
    let m = len(raw) / 3u32;
    var pts = filled_f32(3u32 * m, 0.0);
    var kept = 0u32;
    for i in 0u32..m {{
        let px = raw[3u32 * i];
        let py = raw[3u32 * i + 1u32];
        let pz = raw[3u32 * i + 2u32];
        let wx = fw[0u32] * px - fw[1u32] * py + fw[2u32];
        let wy = fw[1u32] * px + fw[0u32] * py + fw[3u32];
        let wz = pz + fw[4u32];
        if wx >= (-0.9) and wx <= 0.9 and wy >= (-0.9) and wy <= 0.9 and wz >= 0.0 and wz <= 1.1 {{
            pts[3u32 * kept] = wx;
            pts[3u32 * kept + 1u32] = wy;
            pts[3u32 * kept + 2u32] = wz;
            kept = kept + 1u32;
        }}
    }}
{head}    var t = filled_f32({nb12}u32, 0.0);
    var gw = filled_f32({}u32, 0.0);
    for b in 0u32..n {{
        {fill}
        fk(t, 0u32, cs{x});
        place(gw, t, gi, gt);
        var cost = 0.0;
        for g in 0u32..{ng}u32 {{
            var best = 1000.0;
            for k in 0u32..kept {{
                let d = distance(gw, g, gi, gt, pts[3u32 * k], pts[3u32 * k + 1u32], pts[3u32 * k + 2u32], 0.0);
                if d < best {{
                    best = d;
                }}
            }}
            out[b * {}u32 + g] = best;
            if best < {cl} {{
                cost = cost + ({cl} - best) * ({cl} - best);
            }}
        }}
        out[b * {}u32 + {ng}u32] = cost;
    }}
    return n;
",
                    6 * ng,
                    ng + 1,
                    ng + 1
                )
        }
        other => panic!("family {other}"),
    };
    write!(
        s,
        "fn run({}) -> u32 {{\n{tables}{body}}}\n",
        signature(case)
    )
    .unwrap();
    s
}

/// Point cloud with the host frame and raw points; the seed program's
/// filter, compaction and neighbour code, writing into the host buffer.
fn pointcloud(case: &Case) -> String {
    let mut s = frame_fn(case.trig).to_owned();
    s.push_str(
        "fn sample(pts: mut Array<f32>, raw: read Array<f32>, i: u32, fr: read Array<f32>) -> Unit {
    let px = raw[3u32 * i];
    let py = raw[3u32 * i + 1u32];
    let pz = raw[3u32 * i + 2u32];
    pts[3u32 * i] = fr[0u32] * px - fr[1u32] * py + fr[2u32];
    pts[3u32 * i + 1u32] = fr[1u32] * px + fr[0u32] * py + fr[3u32];
    pts[3u32 * i + 2u32] = pz + fr[4u32];
}
",
    );
    s.push_str(families::KEEP);
    writeln!(s, "{}", families::neighbours()).unwrap();
    write!(
        s,
        "fn run({}) -> u32 {{
    var fw = filled_f32(5u32, 0.0);
    pose(fw, fr);
    let n = len(raw) / 3u32;
    var pts = filled_f32(3u32 * n, 0.0);
",
        signature(case)
    )
    .unwrap();
    let (sample, alloc) = (
        "sample(pts, seed, i, fr);",
        "    var out = filled_f32(1u32 + 3u32 * kept + m, 0.0);\n",
    );
    let middle = families::cloud_middle(case.variant);
    assert!(middle.matches(sample).count() == 1 && middle.matches(alloc).count() == 1);
    s.push_str(
        &middle
            .replace(sample, "sample(pts, raw, i, fw);")
            .replace(alloc, ""),
    );
    s.push_str(&families::CLOUD_TAIL.replace("    return out;\n", "    return kept;\n"));
    s
}

/// Host buffers of one `(seed, sizes)` input: exact binary32 replays of
/// the seed program's streams, in entry parameter order.
pub fn expand(case: &Case, arm: Option<&Arm>, input: &[u32]) -> Vec<Arg> {
    let seed = input[0];
    let configs = |n: u32| -> Vec<f32> {
        let arm = arm.expect("arm");
        let mut a = Vec::new();
        for b in 0..n {
            let mut x = stream(seed, b);
            for j in 0..arm.joints {
                x = lcg(x);
                if case.trig {
                    a.push(arm.lo[j] + arm.span[j] * unit(x));
                } else {
                    let (c, s) = rational32(2.0 * unit(x) - 1.0);
                    a.extend([c, s]);
                }
            }
        }
        a
    };
    let frame = || -> Vec<f32> {
        let mut x = stream(seed.wrapping_mul(747_796_405), 0);
        x = lcg(x);
        let mut f = if case.trig {
            // The seed program's 6.2831855 and 3.1415927 are these constants.
            vec![std::f32::consts::TAU * unit(x) - std::f32::consts::PI]
        } else {
            let (c, s) = rational32(2.0 * unit(x) - 1.0);
            vec![c, s]
        };
        for (scale, offset) in [(0.5f32, 0.25f32), (0.5, 0.25), (0.2, 0.0)] {
            x = lcg(x);
            f.push(if offset == 0.0 {
                scale * unit(x)
            } else {
                scale * unit(x) - offset
            });
        }
        f
    };
    let raw = |count: u32, xy: f32, z: f32| -> Vec<f32> {
        let mut r = Vec::new();
        for i in 0..count {
            let mut y = stream(seed, i);
            y = lcg(y);
            r.push(2.0 * xy * unit(y) - xy);
            y = lcg(y);
            r.push(2.0 * xy * unit(y) - xy);
            y = lcg(y);
            r.push(z * unit(y));
        }
        r
    };
    let words = |n: u32, per: usize| Arg::Out(n as usize * per);
    match case.family {
        "fk" | "fkpre" => {
            let a = arm.unwrap();
            vec![Arg::Read(configs(input[1])), words(input[1], 12 * a.bodies)]
        }
        "jacobian" => vec![
            Arg::Read(configs(input[1])),
            words(input[1], 6 * arm.unwrap().joints),
        ],
        "mass" => {
            let nj = arm.unwrap().joints;
            vec![Arg::Read(configs(input[1])), words(input[1], nj * nj)]
        }
        "collision" => {
            let (n, m) = (input[1], input[2]);
            let mut x = stream(seed.wrapping_mul(747_796_405), 0);
            let mut obs = Vec::new();
            for _ in 0..m {
                for (scale, offset) in [(1.8f32, 0.9f32), (1.8, 0.9), (1.2, 0.1)] {
                    x = lcg(x);
                    obs.push(scale * unit(x) - offset);
                }
                x = lcg(x);
                obs.push(0.1 * unit(x) + 0.02);
            }
            let ng = arm.unwrap().geoms();
            vec![
                Arg::Read(configs(n)),
                Arg::Read(obs),
                Arg::Out(n as usize * ng * m as usize),
            ]
        }
        "rollout" => {
            let arm = arm.unwrap();
            let (n, h) = (input[1], input[2]);
            let mut x = stream(seed.wrapping_mul(747_796_405), 0);
            let mut target = Vec::new();
            for (scale, offset) in [(1.2f32, 0.6f32), (1.2, 0.6)] {
                x = lcg(x);
                target.push(scale * unit(x) - offset);
            }
            x = lcg(x);
            target.push(0.8 * unit(x) + 0.1);
            let (mut q0, mut u) = (Vec::new(), Vec::new());
            for b in 0..n {
                let mut y = stream(seed, b);
                for j in 0..arm.joints {
                    y = lcg(y);
                    q0.push(arm.lo[j] + arm.span[j] * unit(y));
                }
                for _ in 0..h * arm.joints as u32 {
                    y = lcg(y);
                    u.push((2.0 * unit(y) - 1.0) * families::ROLLOUT_UMAX);
                }
            }
            vec![
                Arg::Read(q0),
                Arg::Read(u),
                Arg::Read(target),
                Arg::Out((n * h) as usize * (arm.joints + 1)),
            ]
        }
        "pointcloud" => {
            let (n, m) = (input[1], input[2]);
            vec![
                Arg::Read(frame()),
                Arg::Read(raw(n, 2.0, 2.0)),
                Arg::U32(m),
                Arg::Out(1 + 3 * n as usize + m as usize),
            ]
        }
        "pipeline" => {
            let (n, m) = (input[1], input[2]);
            let ng = arm.unwrap().geoms();
            vec![
                Arg::Read(configs(n)),
                Arg::Read(frame()),
                Arg::Read(raw(m, 1.0, 1.2)),
                Arg::Out(n as usize * (ng + 1)),
            ]
        }
        other => panic!("family {other}"),
    }
}

/// The array programs: one per seed program, same ID suffix, seed, split
/// and `(seed, sizes)` inputs.
pub fn generate(root: &Path) -> Vec<Program> {
    let (panda, ur5e) = corpus::arms(root);
    corpus::generate(root)
        .into_iter()
        .map(|p| {
            let arm = match p.case.arm {
                "panda" => Some(&panda),
                "ur5e" => Some(&ur5e),
                _ => None,
            };
            Program {
                id: p.id.replacen("gnet05r-", "gnet05ra-", 1),
                source: source(&p.case, arm),
                ..p
            }
        })
        .collect()
}

pub fn held_out(programs: &[Program]) -> HeldOut {
    corpus::held_out(programs)
}

pub fn manifest(root: &Path, programs: &[Program]) -> serde_json::Value {
    let held = held_out(programs);
    serde_json::json!({
        "schema": "invar-gnet05-robotics-array-v1",
        "generator": {"name": GENERATOR, "base": corpus::GENERATOR,
            "per_family": corpus::PER_FAMILY, "held_out_per_family": corpus::HELD_OUT_PER_FAMILY},
        "requires": REQUIRES,
        "requires_note": "host-array entry parameters (read and mut Array<f32>) on the reference, the CPU net oracle and the device (ABI13, feat/v01-gnet-f32-entry); the files and the held-out guard are checked on every stack, execution only on a stack that contains it",
        "seed_manifest": "../manifest.json",
        "host_expansion": "expand() in array.rs: exact binary32 replays of the seed programs' streams for the same (seed, sizes) input; outputs equal the seed program's output bit for bit (point cloud: the returned prefix, zero after)",
        "models": corpus::manifest(root, &corpus::generate(root))["models"].clone(),
        "held_out": {"count": held.len(), "normalized_identity_sha256": hex(&held.digest())},
        "entry": "run",
        "programs": programs.iter().map(|p| serde_json::json!({
            "id": p.id,
            "seed_program": p.id.replacen("gnet05ra-", "gnet05r-", 1),
            "family": p.case.family,
            "arm": p.case.arm,
            "transforms": if p.case.trig { "trigonometric" } else { "precomputed" },
            "variant": families::variant_label(&p.case),
            "ordinal": p.ordinal,
            "split": p.split.label(),
            "requires": REQUIRES,
            "file": format!("programs/{}.inv", p.id),
            "source_sha256": hex(&Sha256::digest(p.source.as_bytes())),
            "normalized_sha256": hex(&normalized_sha256(p.source.as_bytes())),
            "signature": signature(&p.case),
            "returns": if p.case.family == "pointcloud" { "kept point count" } else { "configurations n" },
            "layout": p.layout,
            "oracle_inputs": p.oracle_inputs,
            "scored_inputs": p.scored_inputs,
        })).collect::<Vec<_>>(),
    })
}
