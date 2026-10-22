//! Invar source emitters for the robotics corpus. All math is written in
//! source (no trig, sqrt or matrix builtins exist), so every program charges
//! its own trigonometry, square roots and geometry. Expressions are emitted
//! in one fixed evaluation order that the handwritten baselines mirror.
use super::arm::{Arm, GEOM_WORDS};
use std::fmt::Write;

/// Structural variant of the kinematic chain; all three compute the same
/// operations in the same order and therefore the same bits.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Chain {
    /// One loop over bodies reading literal parent/frame/axis tables.
    Table,
    /// Straight-line code with every frame constant inline.
    Unrolled,
    /// One function per body segment, called in order.
    Segments,
}
impl Chain {
    pub const ALL: [Chain; 3] = [Chain::Table, Chain::Unrolled, Chain::Segments];
    pub fn label(self) -> &'static str {
        match self {
            Chain::Table => "table",
            Chain::Unrolled => "unrolled",
            Chain::Segments => "segments",
        }
    }
}

/// Binary32 literal: shortest round-trip decimal, never an exponent;
/// negatives are parenthesised unary minus.
pub fn lit(x: f32) -> String {
    assert!(x.is_finite());
    let mut s = format!("{}", x.abs());
    if !s.contains('.') {
        s.push_str(".0");
    }
    if x.is_sign_negative() {
        format!("(-{s})")
    } else {
        s
    }
}
fn f32s(v: &[f32]) -> String {
    format!(
        "[{}]",
        v.iter().map(|&x| lit(x)).collect::<Vec<_>>().join(", ")
    )
}
fn u32s(v: &[u32]) -> String {
    format!(
        "[{}]",
        v.iter()
            .map(|x| format!("{x}u32"))
            .collect::<Vec<_>>()
            .join(", ")
    )
}

/// Frozen binary32 constants, rounded once from their decimal sources:
/// 2^-24, 2/pi, the Cephes three-part pi/2 split, the Cephes sinf/cosf
/// minimax coefficients and the Newton seed `7/6 - m/6`.
pub const CONSTANTS: [(&str, &str); 13] = [
    ("unit", "0.000000059604644775390625"),
    ("two_pi", "0.636619772367581343"),
    ("dp1", "1.5703125"),
    ("dp2", "0.0004837512969970703125"),
    ("dp3", "0.0000000754978995489188216"),
    ("s0", "-0.00019515295891"),
    ("s1", "0.0083321608736"),
    ("s2", "0.16666654611"),
    ("c0", "0.00002443315711809948"),
    ("c1", "0.001388731625493765"),
    ("c2", "0.04166664568298827"),
    ("y0", "1.16666666666666667"),
    ("y1", "0.16666666666666667"),
];
pub fn constant(name: &str) -> f32 {
    let text = CONSTANTS.iter().find(|(n, _)| *n == name).unwrap().1;
    text.parse::<f32>().unwrap()
}

/// Seeded streams, Cephes single-precision sin/cos with a three-part
/// Cody-Waite reduction by pi/2, and a division-free Newton square root.
pub fn prelude(trig: bool, root: bool) -> String {
    let c = |n: &str| lit(constant(n));
    let all = format!(
        "fn lcg(x: u32) -> u32 {{
    return wrapping_add(wrapping_mul(x, 1664525u32), 1013904223u32);
}}
fn unit(x: u32) -> f32 {{
    return f32_from_u32(x / 256u32) * {unit};
}}
fn stream(seed: u32, i: u32) -> u32 {{
    return lcg(wrapping_add(wrapping_mul(i, 2654435761u32), seed));
}}
fn sincos(x: f32, cs: mut Array<f32>, j: u32) -> Unit {{
    var k = 0.0;
    var m = 0u32;
    while x * {two_pi} - k > 0.5 {{
        k = k + 1.0;
        m = m + 1u32;
    }}
    while x * {two_pi} - k < (-0.5) {{
        k = k - 1.0;
        m = m + 3u32;
    }}
    let r = x - k * {dp1} - k * {dp2} - k * {dp3};
    let z = r * r;
    let s = (({s0} * z + {s1}) * z - {s2}) * z * r + r;
    let c = (({c0} * z - {c1}) * z + {c2}) * z * z - 0.5 * z + 1.0;
    let q = m % 4u32;
    var sn = s;
    var cn = c;
    if q == 1u32 {{
        sn = c;
        cn = -s;
    }} else if q == 2u32 {{
        sn = -s;
        cn = -c;
    }} else if q == 3u32 {{
        sn = -c;
        cn = s;
    }}
    cs[2u32 * j] = cn;
    cs[2u32 * j + 1u32] = sn;
}}
fn root(x: f32) -> f32 {{
    if x <= 0.0 {{
        return 0.0;
    }}
    var m = x;
    var g = 1.0;
    while m >= 4.0 {{
        m = m * 0.25;
        g = g * 2.0;
    }}
    while m < 1.0 {{
        m = m * 4.0;
        g = g * 0.5;
    }}
    var y = {y0} - {y1} * m;
    for i in 0u32..5u32 {{
        y = y * (1.5 - 0.5 * m * y * y);
    }}
    return m * y * g;
}}
",
        unit = c("unit"),
        two_pi = c("two_pi"),
        dp1 = c("dp1"),
        dp2 = c("dp2"),
        dp3 = c("dp3"),
        s0 = c("s0"),
        s1 = c("s1"),
        s2 = c("s2"),
        c0 = c("c0"),
        c1 = c("c1"),
        c2 = c("c2"),
        y0 = c("y0"),
        y1 = c("y1"),
    );
    let at_sincos = all.find("fn sincos").unwrap();
    let at_root = all.find("fn root").unwrap();
    let mut out = all[..at_sincos].to_owned();
    if trig {
        out.push_str(&all[at_sincos..at_root]);
    }
    if root {
        out.push_str(&all[at_root..]);
    }
    out
}

/// Joint angles drawn in `[lo, lo + span)`; sin/cos charged in source.
pub const ANGLES: &str = "\
fn angles(cs: mut Array<f32>, seed: u32, b: u32, lo: read Array<f32>, span: read Array<f32>) -> Unit {
    var x = stream(seed, b);
    for j in 0u32..len(lo) {
        x = lcg(x);
        sincos(lo[j] + span[j] * unit(x), cs, j);
    }
}
";

/// Precomputed-transform control: (cos, sin) of `2 atan(u)` for `u` in
/// `[-1, 1)` by the rational half-angle map; no trigonometry.
pub const RATIONAL: &str = "\
fn rational(cs: mut Array<f32>, seed: u32, b: u32, nj: u32) -> Unit {
    var x = stream(seed, b);
    for j in 0u32..nj {
        x = lcg(x);
        let u = 2.0 * unit(x) - 1.0;
        let d = 1.0 + u * u;
        cs[2u32 * j] = (1.0 - u * u) / d;
        cs[2u32 * j + 1u32] = (u + u) / d;
    }
}
";

/// The chain functions and the argument list a call appends after
/// `(w, o, cs`, plus table declarations `run` makes once.
pub struct ChainCode {
    pub functions: String,
    pub tables: String,
    pub extra_args: String,
}

/// The joint-local matrix `JM = A * Rot(axis, q)` entry `(k, col)` of body
/// `i`, as an expression (the cuRobo association: rotate the fixed frame
/// first, then compose with the parent).
fn joint_matrix(arm: &Arm, i: usize, k: usize, col: usize, c: &str, s: &str) -> String {
    let a = &arm.fixed[12 * i..12 * i + 12];
    let x = |col: usize| lit(a[3 * k + col]);
    match (arm.axis[i], col) {
        (3, 0) => format!("{c} * {} + {s} * {}", x(0), x(1)),
        (3, 1) => format!("{c} * {} - {s} * {}", x(1), x(0)),
        (2, 0) => format!("{c} * {} - {s} * {}", x(0), x(2)),
        (2, 2) => format!("{s} * {} + {c} * {}", x(0), x(2)),
        (1, 1) => format!("{c} * {} + {s} * {}", x(1), x(2)),
        (1, 2) => format!("{c} * {} - {s} * {}", x(2), x(1)),
        _ => x(col),
    }
}

/// Statements for body `i` (> 0): `JM = A * Rot`, then `R = R_parent * JM`
/// and `p = R_parent * a + p_parent`. `at(b, k)` renders an index into `w`
/// for frame word `k` of body `b`.
fn segment(arm: &Arm, i: usize, at: &dyn Fn(usize, usize) -> String, c: &str, s: &str) -> String {
    let p = arm.parent[i] as usize;
    let a = &arm.fixed[12 * i..12 * i + 12];
    let mut out = String::new();
    let hinge = arm.axis[i] != 0;
    let jm = |k: usize, col: usize| {
        if hinge {
            format!("j{i}_{k}{col}")
        } else {
            lit(a[3 * k + col])
        }
    };
    if hinge {
        for k in 0..3 {
            for col in 0..3 {
                writeln!(
                    out,
                    "    let {} = {};",
                    jm(k, col),
                    joint_matrix(arm, i, k, col, c, s)
                )
                .unwrap();
            }
        }
    }
    for r in 0..3 {
        for col in 0..3 {
            writeln!(
                out,
                "    w[{}] = w[{}] * {} + w[{}] * {} + w[{}] * {};",
                at(i, 3 * r + col),
                at(p, 3 * r),
                jm(0, col),
                at(p, 3 * r + 1),
                jm(1, col),
                at(p, 3 * r + 2),
                jm(2, col)
            )
            .unwrap();
        }
        writeln!(
            out,
            "    w[{}] = w[{}] * {} + w[{}] * {} + w[{}] * {} + w[{}];",
            at(i, 9 + r),
            at(p, 3 * r),
            lit(a[9]),
            at(p, 3 * r + 1),
            lit(a[10]),
            at(p, 3 * r + 2),
            lit(a[11]),
            at(p, 9 + r)
        )
        .unwrap();
    }
    out
}

fn identity_base(at: &dyn Fn(usize, usize) -> String, arm: &Arm) -> String {
    (0..12)
        .map(|k| format!("    w[{}] = {};\n", at(0, k), lit(arm.fixed[k])))
        .collect()
}

pub fn chain(arm: &Arm, variant: Chain) -> ChainCode {
    let header = "fn fk(w: mut Array<f32>, o: u32, cs: read Array<f32>";
    match variant {
        Chain::Table => ChainCode {
            functions: format!(
                "{header}, par: read Array<u32>, fix: read Array<f32>, ax: read Array<u32>, jt: read Array<u32>, jm: mut Array<f32>) -> Unit {{
    for e in 0u32..12u32 {{
        w[o + e] = fix[e];
    }}
    for i in 1u32..len(par) {{
        let so = o + i * 12u32;
        let po = o + par[i] * 12u32;
        let ao = i * 12u32;
        let a = ax[i];
        let c = cs[2u32 * jt[i]];
        let s = cs[2u32 * jt[i] + 1u32];
        for k in 0u32..3u32 {{
            let f = ao + 3u32 * k;
            let x = fix[f];
            let y = fix[f + 1u32];
            let z = fix[f + 2u32];
            jm[3u32 * k] = x;
            jm[3u32 * k + 1u32] = y;
            jm[3u32 * k + 2u32] = z;
            if a == 3u32 {{
                jm[3u32 * k] = c * x + s * y;
                jm[3u32 * k + 1u32] = c * y - s * x;
            }} else if a == 2u32 {{
                jm[3u32 * k] = c * x - s * z;
                jm[3u32 * k + 2u32] = s * x + c * z;
            }} else if a == 1u32 {{
                jm[3u32 * k + 1u32] = c * y + s * z;
                jm[3u32 * k + 2u32] = c * z - s * y;
            }}
        }}
        for r in 0u32..3u32 {{
            let pr = po + r * 3u32;
            for col in 0u32..3u32 {{
                w[so + r * 3u32 + col] = w[pr] * jm[col] + w[pr + 1u32] * jm[3u32 + col] + w[pr + 2u32] * jm[6u32 + col];
            }}
            w[so + 9u32 + r] = w[pr] * fix[ao + 9u32] + w[pr + 1u32] * fix[ao + 10u32] + w[pr + 2u32] * fix[ao + 11u32] + w[po + 9u32 + r];
        }}
    }}
}}
"
            ),
            tables: format!(
                "    let par: Array<u32> = {};\n    let fix: Array<f32> = {};\n    let ax: Array<u32> = {};\n    let jt: Array<u32> = {};\n    var jm = filled_f32(9u32, 0.0);\n",
                u32s(&arm.parent),
                f32s(&arm.fixed),
                u32s(&arm.axis),
                u32s(&arm.joint)
            ),
            extra_args: ", par, fix, ax, jt, jm".into(),
        },
        Chain::Unrolled => {
            let at = |b: usize, k: usize| format!("o + {}u32", 12 * b + k);
            let mut body = identity_base(&at, arm);
            for i in 1..arm.bodies {
                let (c, s) = (format!("c{i}"), format!("s{i}"));
                if arm.axis[i] != 0 {
                    let j = arm.joint[i];
                    writeln!(body, "    let {c} = cs[{}u32];", 2 * j).unwrap();
                    writeln!(body, "    let {s} = cs[{}u32];", 2 * j + 1).unwrap();
                }
                body.push_str(&segment(arm, i, &at, &c, &s));
            }
            ChainCode {
                functions: format!("{header}) -> Unit {{\n{body}}}\n"),
                tables: String::new(),
                extra_args: String::new(),
            }
        }
        Chain::Segments => {
            let at = |b: usize, k: usize| format!("o + {}u32", 12 * b + k);
            let mut functions = String::new();
            let mut calls = identity_base(&at, arm);
            for i in 1..arm.bodies {
                let hinge = arm.axis[i] != 0;
                let params = if hinge { ", c: f32, s: f32" } else { "" };
                write!(
                    functions,
                    "fn link{i}(w: mut Array<f32>, o: u32{params}) -> Unit {{\n{}}}\n",
                    segment(arm, i, &at, "c", "s")
                )
                .unwrap();
                if hinge {
                    let j = arm.joint[i];
                    writeln!(
                        calls,
                        "    link{i}(w, o, cs[{}u32], cs[{}u32]);",
                        2 * j,
                        2 * j + 1
                    )
                    .unwrap();
                } else {
                    writeln!(calls, "    link{i}(w, o);").unwrap();
                }
            }
            write!(functions, "{header}) -> Unit {{\n{calls}}}\n").unwrap();
            ChainCode {
                functions,
                tables: String::new(),
                extra_args: String::new(),
            }
        }
    }
}

/// Tool point coordinate `i` from frames in `t` at offset 0.
pub fn tool(arm: &Arm) -> String {
    let tb = 12 * arm.tool_body;
    let [x, y, z] = arm.tool.map(lit);
    format!(
        "fn tool(t: read Array<f32>, i: u32) -> f32 {{
    let r = {tb}u32 + 3u32 * i;
    return t[r] * {x} + t[r + 1u32] * {y} + t[r + 2u32] * {z} + t[{tb}u32 + 9u32 + i];
}}
"
    )
}

/// Fill step for configuration `b`: trigonometric or precomputed control.
pub fn fill(trig: bool, nj: usize) -> String {
    if trig {
        "angles(cs, seed, b, lo, span);".into()
    } else {
        format!("rational(cs, seed, b, {nj}u32);")
    }
}
pub fn fill_fn(trig: bool) -> &'static str {
    if trig { ANGLES } else { RATIONAL }
}
pub fn limit_tables(arm: &Arm) -> String {
    format!(
        "    let lo: Array<f32> = {};\n    let span: Array<f32> = {};\n",
        f32s(&arm.lo),
        f32s(&arm.span)
    )
}

/// Full 6xN tool Jacobian (rows vx vy vz wx wy wz) in base axes.
pub const JACOBIAN: &str = "\
fn jacobian(out: mut Array<f32>, o: u32, t: read Array<f32>, jx: read Array<u32>) -> Unit {
    let px = tool(t, 0u32);
    let py = tool(t, 1u32);
    let pz = tool(t, 2u32);
    let n = len(jx) / 2u32;
    for j in 0u32..n {
        let bo = jx[2u32 * j] * 12u32;
        let a = jx[2u32 * j + 1u32] - 1u32;
        let zx = t[bo + a];
        let zy = t[bo + 3u32 + a];
        let zz = t[bo + 6u32 + a];
        let dx = px - t[bo + 9u32];
        let dy = py - t[bo + 10u32];
        let dz = pz - t[bo + 11u32];
        out[o + j] = zy * dz - zz * dy;
        out[o + n + j] = zz * dx - zx * dz;
        out[o + 2u32 * n + j] = zx * dy - zy * dx;
        out[o + 3u32 * n + j] = zx;
        out[o + 4u32 * n + j] = zy;
        out[o + 5u32 * n + j] = zz;
    }
}
";

/// Joint-space mass matrix: sum over bodies of `m Jv^T Jv + Jw^T I Jw`
/// (body-frame angular columns) plus source armature on the diagonal.
pub const MASS: &str = "\
fn mass(out: mut Array<f32>, o: u32, t: read Array<f32>, sv: mut Array<f32>, jx: read Array<u32>, mv: read Array<u32>, bt: read Array<f32>, ar: read Array<f32>) -> Unit {
    let n = len(ar);
    for idx in 0u32..n * n {
        out[o + idx] = 0.0;
    }
    for b in 1u32..len(bt) / 13u32 {
        let bo = b * 12u32;
        let k = b * 13u32;
        let cx = t[bo] * bt[k + 1u32] + t[bo + 1u32] * bt[k + 2u32] + t[bo + 2u32] * bt[k + 3u32] + t[bo + 9u32];
        let cy = t[bo + 3u32] * bt[k + 1u32] + t[bo + 4u32] * bt[k + 2u32] + t[bo + 5u32] * bt[k + 3u32] + t[bo + 10u32];
        let cz = t[bo + 6u32] * bt[k + 1u32] + t[bo + 7u32] * bt[k + 2u32] + t[bo + 8u32] * bt[k + 3u32] + t[bo + 11u32];
        for j in 0u32..n {
            let s = 9u32 * j;
            if mv[b * n + j] == 1u32 {
                let jo = jx[2u32 * j] * 12u32;
                let axis = jx[2u32 * j + 1u32] - 1u32;
                let zx = t[jo + axis];
                let zy = t[jo + 3u32 + axis];
                let zz = t[jo + 6u32 + axis];
                let dx = cx - t[jo + 9u32];
                let dy = cy - t[jo + 10u32];
                let dz = cz - t[jo + 11u32];
                let wx = t[bo] * zx + t[bo + 3u32] * zy + t[bo + 6u32] * zz;
                let wy = t[bo + 1u32] * zx + t[bo + 4u32] * zy + t[bo + 7u32] * zz;
                let wz = t[bo + 2u32] * zx + t[bo + 5u32] * zy + t[bo + 8u32] * zz;
                sv[s] = zy * dz - zz * dy;
                sv[s + 1u32] = zz * dx - zx * dz;
                sv[s + 2u32] = zx * dy - zy * dx;
                sv[s + 3u32] = wx;
                sv[s + 4u32] = wy;
                sv[s + 5u32] = wz;
                sv[s + 6u32] = bt[k + 4u32] * wx + bt[k + 5u32] * wy + bt[k + 6u32] * wz;
                sv[s + 7u32] = bt[k + 7u32] * wx + bt[k + 8u32] * wy + bt[k + 9u32] * wz;
                sv[s + 8u32] = bt[k + 10u32] * wx + bt[k + 11u32] * wy + bt[k + 12u32] * wz;
            } else {
                for e in 0u32..9u32 {
                    sv[s + e] = 0.0;
                }
            }
        }
        let m = bt[k];
        for i in 0u32..n {
            let si = 9u32 * i;
            for jc in 0u32..n {
                let sj = 9u32 * jc;
                let lin = sv[si] * sv[sj] + sv[si + 1u32] * sv[sj + 1u32] + sv[si + 2u32] * sv[sj + 2u32];
                let ang = sv[si + 3u32] * sv[sj + 6u32] + sv[si + 4u32] * sv[sj + 7u32] + sv[si + 5u32] * sv[sj + 8u32];
                out[o + i * n + jc] = out[o + i * n + jc] + (m * lin + ang);
            }
        }
    }
    for d in 0u32..n {
        out[o + d * n + d] = out[o + d * n + d] + ar[d];
    }
}
";

/// World centre and axis of every collision geom from frames in `t`.
pub const PLACE: &str = "\
fn place(gw: mut Array<f32>, t: read Array<f32>, gi: read Array<u32>, gt: read Array<f32>) -> Unit {
    for g in 0u32..len(gi) / 2u32 {
        let bo = gi[2u32 * g] * 12u32;
        let k = g * 10u32;
        for r in 0u32..3u32 {
            let br = bo + 3u32 * r;
            gw[6u32 * g + r] = t[br] * gt[k + 4u32] + t[br + 1u32] * gt[k + 5u32] + t[br + 2u32] * gt[k + 6u32] + t[bo + 9u32 + r];
            gw[6u32 * g + 3u32 + r] = t[br] * gt[k + 7u32] + t[br + 1u32] * gt[k + 8u32] + t[br + 2u32] * gt[k + 9u32];
        }
    }
}
";

/// Exact signed distance from a sphere (centre, radius) to an original
/// primitive: capsule by segment projection, capped cylinder by its SDF.
pub const DISTANCE: &str = "\
fn distance(gw: read Array<f32>, g: u32, gi: read Array<u32>, gt: read Array<f32>, ox: f32, oy: f32, oz: f32, orad: f32) -> f32 {
    let k = g * 10u32;
    let w = 6u32 * g;
    let cx = gw[w];
    let cy = gw[w + 1u32];
    let cz = gw[w + 2u32];
    let ax = gw[w + 3u32];
    let ay = gw[w + 4u32];
    let az = gw[w + 5u32];
    let r = gt[k];
    let h = gt[k + 1u32];
    if gi[2u32 * g + 1u32] == 0u32 {
        let p0x = cx - h * ax;
        let p0y = cy - h * ay;
        let p0z = cz - h * az;
        let dx = gt[k + 2u32] * ax;
        let dy = gt[k + 2u32] * ay;
        let dz = gt[k + 2u32] * az;
        var u = ((ox - p0x) * dx + (oy - p0y) * dy + (oz - p0z) * dz) * gt[k + 3u32];
        if u < 0.0 {
            u = 0.0;
        }
        if u > 1.0 {
            u = 1.0;
        }
        let ex = ox - (p0x + u * dx);
        let ey = oy - (p0y + u * dy);
        let ez = oz - (p0z + u * dz);
        return root(ex * ex + ey * ey + ez * ez) - (r + orad);
    }
    let vx = ox - cx;
    let vy = oy - cy;
    let vz = oz - cz;
    let along = vx * ax + vy * ay + vz * az;
    let rx = vx - along * ax;
    let ry = vy - along * ay;
    let rz = vz - along * az;
    let radial = root(rx * rx + ry * ry + rz * rz);
    var aa = along;
    if aa < 0.0 {
        aa = -aa;
    }
    let qx = radial - r;
    let qy = aa - h;
    var px = qx;
    if px < 0.0 {
        px = 0.0;
    }
    var py = qy;
    if py < 0.0 {
        py = 0.0;
    }
    var inside = qx;
    if qy > inside {
        inside = qy;
    }
    if inside > 0.0 {
        inside = 0.0;
    }
    return root(px * px + py * py) + inside - orad;
}
";

pub fn geom_tables(arm: &Arm) -> String {
    assert_eq!(arm.geom.len(), GEOM_WORDS * arm.geoms());
    format!(
        "    let gi: Array<u32> = {};\n    let gt: Array<f32> = {};\n",
        u32s(&arm.geom_index),
        f32s(&arm.geom)
    )
}
pub fn joint_table(arm: &Arm) -> String {
    let jx: Vec<u32> = arm.joint_body.iter().flat_map(|&(b, a)| [b, a]).collect();
    format!("    let jx: Array<u32> = {};\n", u32s(&jx))
}
pub fn mass_tables(arm: &Arm) -> String {
    format!(
        "    let mv: Array<u32> = {};\n    let bt: Array<f32> = {};\n    let ar: Array<f32> = {};\n",
        u32s(&arm.moves),
        f32s(&arm.inertial),
        f32s(&arm.armature)
    )
}
pub fn hi_table(arm: &Arm) -> String {
    format!("    let hi: Array<f32> = {};\n", f32s(&arm.hi))
}
