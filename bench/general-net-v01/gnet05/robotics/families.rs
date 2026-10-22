//! Whole-program sources per robotics family. Every entry is
//! `run(seed: u32, n: u32[, m: u32]) -> Array<f32>`: the runtime inputs, not
//! the source, choose the batch, horizon, obstacle and point counts, and all
//! bulk inputs are expanded in source from `seed` by the frozen LCG streams.
use super::arm::Arm;
use super::emit::{self, Chain};
use std::fmt::Write;

pub const FAMILIES: [&str; 8] = [
    "fk",
    "fkpre",
    "jacobian",
    "mass",
    "collision",
    "rollout",
    "pointcloud",
    "pipeline",
];

/// Frozen workload constants (binary32 literals in source).
pub const ROLLOUT_DT: f32 = 0.01;
pub const ROLLOUT_UMAX: f32 = 2.0;
pub const ROLLOUT_WEIGHT: f32 = 0.001;
pub const NEIGHBOUR_RADIUS2: f32 = 0.0225;
pub const CLEARANCE: f32 = 0.05;

/// One corpus case: family, arm, trigonometric or precomputed transforms,
/// and structural variant (0..3).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Case {
    pub family: &'static str,
    pub arm: &'static str,
    pub trig: bool,
    pub variant: usize,
}

/// The six cases of a family, in ordinal order.
pub fn cases(family: &'static str) -> Vec<Case> {
    let axes: [(&'static str, bool); 2] = match family {
        "fk" | "jacobian" | "mass" | "rollout" => [("panda", true), ("ur5e", true)],
        "fkpre" => [("panda", false), ("ur5e", false)],
        "collision" | "pipeline" => [("ur5e", true), ("ur5e", false)],
        "pointcloud" => [("none", true), ("none", false)],
        other => panic!("family {other}"),
    };
    axes.iter()
        .flat_map(|&(arm, trig)| {
            (0..3).map(move |variant| Case {
                family,
                arm,
                trig,
                variant,
            })
        })
        .collect()
}

pub fn variant_label(case: &Case) -> &'static str {
    if case.family == "pointcloud" {
        ["fused", "staged", "scanned"][case.variant]
    } else {
        Chain::ALL[case.variant].label()
    }
}

/// Output layout, frozen in the manifest.
pub fn layout(case: &Case, arm: Option<&Arm>) -> String {
    let (nb, nj, ng) = arm.map_or((0, 0, 0), |a| (a.bodies, a.joints, a.geoms()));
    match case.family {
        "fk" | "fkpre" => format!(
            "n x {nb} bodies x [R row-major 9, p 3], base frame, bodies in source order"
        ),
        "jacobian" => format!(
            "n x 6 x {nj} row-major [vx vy vz wx wy wz] tool Jacobian in base axes at the tool point"
        ),
        "mass" => format!("n x {nj} x {nj} joint-space mass matrix including source armature"),
        "collision" => format!(
            "n x {ng} geoms x m obstacles signed distance (m), geoms in source order"
        ),
        "rollout" => format!("n x h x [{nj} joint positions, step cost]"),
        "pointcloud" => {
            "[kept count k, kept xyz in input order (3k), m neighbour-count slots: counts within the radius (self included) for the first min(m, k) kept points, zero after]".into()
        }
        "pipeline" => format!(
            "n x [{ng} per-geom minimum distance to kept points (m), clearance cost]"
        ),
        other => panic!("family {other}"),
    }
}

fn chain_parts(case: &Case, arm: &Arm) -> emit::ChainCode {
    emit::chain(arm, Chain::ALL[case.variant])
}

/// Full program source for one case.
pub fn source(case: &Case, arm: Option<&Arm>) -> String {
    let mut s = format!(
        "// GNET-05 robotics corpus: {} ({}, {}, {}).\n",
        case.family,
        case.arm,
        if case.trig {
            "trigonometric"
        } else {
            "precomputed transforms"
        },
        variant_label(case)
    );
    let trig = case.trig || case.family == "rollout";
    let root = matches!(case.family, "collision" | "pipeline");
    s.push_str(&emit::prelude(trig, root));
    if case.family == "pointcloud" {
        s.push_str(&pointcloud(case));
        return s;
    }
    let arm = arm.expect("arm");
    let chain = chain_parts(case, arm);
    s.push_str(emit::fill_fn(case.trig));
    s.push_str(&chain.functions);
    s.push_str(&emit::tool(arm));
    let fill = emit::fill(case.trig, arm.joints);
    let fk_call = |w: &str| format!("fk({w}, 0u32, cs{});", chain.extra_args);
    let nb12 = arm.bodies * 12;
    let nj = arm.joints;
    let mut tables = emit::limit_tables(arm) + &chain.tables;
    let body = match case.family {
        "fk" | "fkpre" => {
            format!(
                "    var out = filled_f32(n * {nb12}u32, 0.0);
    var cs = filled_f32({}u32, 0.0);
    for b in 0u32..n {{
        {fill}
        fk(out, b * {nb12}u32, cs{});
    }}
    return out;
",
                2 * nj,
                chain.extra_args
            )
        }
        "jacobian" => {
            s.push_str(emit::JACOBIAN);
            tables.push_str(&emit::joint_table(arm));
            format!(
                "    var out = filled_f32(n * {}u32, 0.0);
    var cs = filled_f32({}u32, 0.0);
    var t = filled_f32({nb12}u32, 0.0);
    for b in 0u32..n {{
        {fill}
        {}
        jacobian(out, b * {}u32, t, jx);
    }}
    return out;
",
                6 * nj,
                2 * nj,
                fk_call("t"),
                6 * nj
            )
        }
        "mass" => {
            s.push_str(emit::MASS);
            tables.push_str(&emit::joint_table(arm));
            tables.push_str(&emit::mass_tables(arm));
            format!(
                "    var out = filled_f32(n * {}u32, 0.0);
    var cs = filled_f32({}u32, 0.0);
    var t = filled_f32({nb12}u32, 0.0);
    var sv = filled_f32({}u32, 0.0);
    for b in 0u32..n {{
        {fill}
        {}
        mass(out, b * {}u32, t, sv, jx, mv, bt, ar);
    }}
    return out;
",
                nj * nj,
                2 * nj,
                9 * nj,
                fk_call("t"),
                nj * nj
            )
        }
        "collision" => {
            s.push_str(emit::PLACE);
            s.push_str(emit::DISTANCE);
            tables.push_str(&emit::geom_tables(arm));
            let ng = arm.geoms();
            format!(
                "    var obs = filled_f32(4u32 * m, 0.0);
    var x = stream(wrapping_mul(seed, 747796405u32), 0u32);
    for ob in 0u32..m {{
        x = lcg(x);
        obs[4u32 * ob] = 1.8 * unit(x) - 0.9;
        x = lcg(x);
        obs[4u32 * ob + 1u32] = 1.8 * unit(x) - 0.9;
        x = lcg(x);
        obs[4u32 * ob + 2u32] = 1.2 * unit(x) - 0.1;
        x = lcg(x);
        obs[4u32 * ob + 3u32] = 0.1 * unit(x) + 0.02;
    }}
    var out = filled_f32(n * {ng}u32 * m, 0.0);
    var cs = filled_f32({}u32, 0.0);
    var t = filled_f32({nb12}u32, 0.0);
    var gw = filled_f32({}u32, 0.0);
    for b in 0u32..n {{
        {fill}
        {}
        place(gw, t, gi, gt);
        for g in 0u32..{ng}u32 {{
            for k in 0u32..m {{
                out[(b * {ng}u32 + g) * m + k] = distance(gw, g, gi, gt, obs[4u32 * k], obs[4u32 * k + 1u32], obs[4u32 * k + 2u32], obs[4u32 * k + 3u32]);
            }}
        }}
    }}
    return out;
",
                2 * nj,
                6 * ng,
                fk_call("t")
            )
        }
        "rollout" => {
            tables.push_str(&emit::hi_table(arm));
            let (dt, umax, wt) = (
                emit::lit(ROLLOUT_DT),
                emit::lit(ROLLOUT_UMAX),
                emit::lit(ROLLOUT_WEIGHT),
            );
            format!(
                "    var x = stream(wrapping_mul(seed, 747796405u32), 0u32);
    x = lcg(x);
    let tx = 1.2 * unit(x) - 0.6;
    x = lcg(x);
    let ty = 1.2 * unit(x) - 0.6;
    x = lcg(x);
    let tz = 0.8 * unit(x) + 0.1;
    var out = filled_f32(n * h * {}u32, 0.0);
    var q = filled_f32({nj}u32, 0.0);
    var v = filled_f32({nj}u32, 0.0);
    var cs = filled_f32({}u32, 0.0);
    var t = filled_f32({nb12}u32, 0.0);
    for b in 0u32..n {{
        var y = stream(seed, b);
        for j0 in 0u32..{nj}u32 {{
            y = lcg(y);
            q[j0] = lo[j0] + span[j0] * unit(y);
            v[j0] = 0.0;
        }}
        for k in 0u32..h {{
            var uu = 0.0;
            for j in 0u32..{nj}u32 {{
                y = lcg(y);
                let u = (2.0 * unit(y) - 1.0) * {umax};
                uu = uu + u * u;
                v[j] = v[j] + {dt} * u;
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
            {}
            let ex = tool(t, 0u32) - tx;
            let ey = tool(t, 1u32) - ty;
            let ez = tool(t, 2u32) - tz;
            let o = (b * h + k) * {}u32;
            for j2 in 0u32..{nj}u32 {{
                out[o + j2] = q[j2];
            }}
            out[o + {nj}u32] = ex * ex + ey * ey + ez * ez + {wt} * uu;
        }}
    }}
    return out;
",
                nj + 1,
                2 * nj,
                fk_call("t"),
                nj + 1
            )
        }
        "pipeline" => {
            s.push_str(emit::PLACE);
            s.push_str(emit::DISTANCE);
            s.push_str(&cloud_frame(case.trig));
            tables.push_str(&emit::geom_tables(arm));
            let ng = arm.geoms();
            let cl = emit::lit(CLEARANCE);
            format!(
                "    var fr = filled_f32(5u32, 0.0);
    sensor(fr, seed);
    var pts = filled_f32(3u32 * m, 0.0);
    var kept = 0u32;
    for i in 0u32..m {{
        var y = stream(seed, i);
        y = lcg(y);
        let px = 2.0 * unit(y) - 1.0;
        y = lcg(y);
        let py = 2.0 * unit(y) - 1.0;
        y = lcg(y);
        let pz = 1.2 * unit(y);
        let wx = fr[0u32] * px - fr[1u32] * py + fr[2u32];
        let wy = fr[1u32] * px + fr[0u32] * py + fr[3u32];
        let wz = pz + fr[4u32];
        if wx >= (-0.9) and wx <= 0.9 and wy >= (-0.9) and wy <= 0.9 and wz >= 0.0 and wz <= 1.1 {{
            pts[3u32 * kept] = wx;
            pts[3u32 * kept + 1u32] = wy;
            pts[3u32 * kept + 2u32] = wz;
            kept = kept + 1u32;
        }}
    }}
    var out = filled_f32(n * {}u32, 0.0);
    var cs = filled_f32({}u32, 0.0);
    var t = filled_f32({nb12}u32, 0.0);
    var gw = filled_f32({}u32, 0.0);
    for b in 0u32..n {{
        {fill}
        {}
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
    return out;
",
                ng + 1,
                2 * nj,
                6 * ng,
                fk_call("t"),
                ng + 1,
                ng + 1
            )
        }
        other => panic!("family {other}"),
    };
    let params = match case.family {
        "collision" | "pipeline" => "seed: u32, n: u32, m: u32",
        "rollout" => "seed: u32, n: u32, h: u32",
        _ => "seed: u32, n: u32",
    };
    write!(s, "fn run({params}) -> Array<f32> {{\n{tables}{body}}}\n").unwrap();
    if let Some(probe) = probe(case, arm, &chain, &tables) {
        s.push_str(&probe);
    }
    s
}

/// `probe(q0..)`: the same chain at given angles, for the Pinocchio check.
fn probe(case: &Case, arm: &Arm, chain: &emit::ChainCode, tables: &str) -> Option<String> {
    let tail = match case.family {
        "fk" => "    return t;\n".to_owned(),
        "jacobian" => format!(
            "    var out = filled_f32({}u32, 0.0);\n    jacobian(out, 0u32, t, jx);\n    return out;\n",
            6 * arm.joints
        ),
        "mass" => format!(
            "    var out = filled_f32({}u32, 0.0);\n    var sv = filled_f32({}u32, 0.0);\n    mass(out, 0u32, t, sv, jx, mv, bt, ar);\n    return out;\n",
            arm.joints * arm.joints,
            9 * arm.joints
        ),
        _ => return None,
    };
    let params: Vec<String> = (0..arm.joints).map(|j| format!("q{j}: f32")).collect();
    let mut s = format!(
        "fn probe({}) -> Array<f32> {{\n{tables}    var cs = filled_f32({}u32, 0.0);\n",
        params.join(", "),
        2 * arm.joints
    );
    for j in 0..arm.joints {
        writeln!(s, "    sincos(q{j}, cs, {j}u32);").unwrap();
    }
    write!(
        s,
        "    var t = filled_f32({}u32, 0.0);\n    fk(t, 0u32, cs{});\n{tail}}}\n",
        12 * arm.bodies,
        chain.extra_args
    )
    .unwrap();
    Some(s)
}

/// Sensor frame from the seed: (cos, sin) of a yaw in `[-pi, pi)` (or the
/// precomputed control), then a translation.
fn cloud_frame(trig: bool) -> String {
    let rotation = if trig {
        "    x = lcg(x);
    var cs = filled_f32(2u32, 0.0);
    sincos(6.2831855 * unit(x) - 3.1415927, cs, 0u32);
    fr[0u32] = cs[0u32];
    fr[1u32] = cs[1u32];
"
    } else {
        "    x = lcg(x);
    let u = 2.0 * unit(x) - 1.0;
    let d = 1.0 + u * u;
    fr[0u32] = (1.0 - u * u) / d;
    fr[1u32] = (u + u) / d;
"
    };
    format!(
        "fn sensor(fr: mut Array<f32>, seed: u32) -> Unit {{
    var x = stream(wrapping_mul(seed, 747796405u32), 0u32);
{rotation}    x = lcg(x);
    fr[2u32] = 0.5 * unit(x) - 0.25;
    x = lcg(x);
    fr[3u32] = 0.5 * unit(x) - 0.25;
    x = lcg(x);
    fr[4u32] = 0.2 * unit(x);
}}
"
    )
}

/// Point-cloud crop and range filter of transformed point `i`.
pub const KEEP: &str = "\
fn keep(pts: read Array<f32>, i: u32) -> Bool {
    let wx = pts[3u32 * i];
    let wy = pts[3u32 * i + 1u32];
    let wz = pts[3u32 * i + 2u32];
    let r2 = wx * wx + wy * wy + wz * wz;
    return wx >= (-1.5) and wx <= 1.5 and wy >= (-1.5) and wy <= 1.5 and wz >= 0.05 and wz <= 1.8 and r2 >= 0.04 and r2 <= 4.0;
}
";

/// Fixed-radius neighbour count of kept point `a` (self included).
pub fn neighbours() -> String {
    let r2 = emit::lit(NEIGHBOUR_RADIUS2);
    format!(
        "fn neighbours(out: read Array<f32>, kept: u32, a: u32) -> f32 {{
    let ax = out[1u32 + 3u32 * a];
    let ay = out[2u32 + 3u32 * a];
    let az = out[3u32 + 3u32 * a];
    var count = 0u32;
    for k in 0u32..kept {{
        let dx = out[1u32 + 3u32 * k] - ax;
        let dy = out[2u32 + 3u32 * k] - ay;
        let dz = out[3u32 + 3u32 * k] - az;
        if dx * dx + dy * dy + dz * dz <= {r2} {{
            count = count + 1u32;
        }}
    }}
    return f32_from_u32(count);
}}"
    )
}

/// Neighbour counts of the first `min(m, kept)` kept points.
pub const CLOUD_TAIL: &str = "    var q = m;
    if kept < q {
        q = kept;
    }
    for a in 0u32..q {
        let count = neighbours(out, kept, a);
        out[1u32 + 3u32 * kept + a] = count;
    }
    return out;
}
";

/// Mark, count and compact: the three structural point-cloud variants.
pub fn cloud_middle(variant: usize) -> &'static str {
    match variant {
        // One pass marks and counts, a second pass compacts in order.
        0 => {
            "    var flag = filled_u32(n, 0u32);
    var kept = 0u32;
    for i in 0u32..n {
        sample(pts, seed, i, fr);
        if keep(pts, i) {
            flag[i] = 1u32;
            kept = kept + 1u32;
        }
    }
    var out = filled_f32(1u32 + 3u32 * kept + m, 0.0);
    out[0u32] = f32_from_u32(kept);
    var w = 0u32;
    for i2 in 0u32..n {
        if flag[i2] == 1u32 {
            for c in 0u32..3u32 {
                out[1u32 + 3u32 * w + c] = pts[3u32 * i2 + c];
            }
            w = w + 1u32;
        }
    }
"
        }
        // Separate transform and filter stages before compaction.
        1 => {
            "    for i in 0u32..n {
        sample(pts, seed, i, fr);
    }
    var flag = filled_u32(n, 0u32);
    for i2 in 0u32..n {
        if keep(pts, i2) {
            flag[i2] = 1u32;
        }
    }
    var kept = 0u32;
    for i3 in 0u32..n {
        kept = kept + flag[i3];
    }
    var out = filled_f32(1u32 + 3u32 * kept + m, 0.0);
    out[0u32] = f32_from_u32(kept);
    var w = 0u32;
    for i4 in 0u32..n {
        if flag[i4] == 1u32 {
            out[1u32 + 3u32 * w] = pts[3u32 * i4];
            out[2u32 + 3u32 * w] = pts[3u32 * i4 + 1u32];
            out[3u32 + 3u32 * w] = pts[3u32 * i4 + 2u32];
            w = w + 1u32;
        }
    }
"
        }
        // Exclusive prefix offsets, then an independent scatter per point.
        _ => {
            "    var pos = filled_u32(n + 1u32, 0u32);
    for i in 0u32..n {
        sample(pts, seed, i, fr);
        var f = 0u32;
        if keep(pts, i) {
            f = 1u32;
        }
        pos[i + 1u32] = pos[i] + f;
    }
    let kept = pos[n];
    var out = filled_f32(1u32 + 3u32 * kept + m, 0.0);
    out[0u32] = f32_from_u32(kept);
    for i2 in 0u32..n {
        if pos[i2 + 1u32] != pos[i2] {
            let w = pos[i2];
            for c in 0u32..3u32 {
                out[1u32 + 3u32 * w + c] = pts[3u32 * i2 + c];
            }
        }
    }
"
        }
    }
}

/// Point cloud: transform, crop/range filter, stable compaction and
/// fixed-radius neighbour counts. Three structural variants.
fn pointcloud(case: &Case) -> String {
    let mut s = cloud_frame(case.trig);
    s.push_str(
        "fn sample(pts: mut Array<f32>, seed: u32, i: u32, fr: read Array<f32>) -> Unit {
    var y = stream(seed, i);
    y = lcg(y);
    let px = 4.0 * unit(y) - 2.0;
    y = lcg(y);
    let py = 4.0 * unit(y) - 2.0;
    y = lcg(y);
    let pz = 2.0 * unit(y);
    pts[3u32 * i] = fr[0u32] * px - fr[1u32] * py + fr[2u32];
    pts[3u32 * i + 1u32] = fr[1u32] * px + fr[0u32] * py + fr[3u32];
    pts[3u32 * i + 2u32] = pz + fr[4u32];
}
",
    );
    s.push_str(KEEP);
    writeln!(s, "{}", neighbours()).unwrap();
    s.push_str(
        "fn run(seed: u32, n: u32, m: u32) -> Array<f32> {
    var fr = filled_f32(5u32, 0.0);
    sensor(fr, seed);
    var pts = filled_f32(3u32 * n, 0.0);
",
    );
    s.push_str(cloud_middle(case.variant));
    s.push_str(CLOUD_TAIL);
    s
}
