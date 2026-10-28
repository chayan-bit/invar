//! Rust oracles for the domain suites. Plain f32 and u32 arithmetic in the
//! same operation order as the Invar kernels (strict binary32, no fusing, no
//! reassociation); no Invar evaluator or engine is involved. The shared
//! pieces are the library functions the language contract defines: `exp` is
//! the correctly rounded `invar-math-cr-v1` result (one ulp of `f64` libm,
//! checked in `tools/tests/bench_tooling/issue_eng_10_domain_suites.rs`), and so are `sin` and
//! `cos`.
use invar::{
    math::{self, MathFn},
    numeric_contract::F32Bits,
};

/// The input recurrence of the generated `main`: values in [-1, 1] in steps
/// of 0.001, from a 32-bit linear congruential generator.
pub fn fill(n: u32, seed: u32) -> Vec<f32> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(1664525).wrapping_add(1013904223);
            ((s / 65536) % 2001) as f32 - 1000.0
        })
        .map(|x| x * 0.001)
        .collect()
}

fn cr(f: MathFn, x: f32) -> f32 {
    f32::from_bits(math::eval_f32(f, &[F32Bits::from_bits(x.to_bits())]).to_bits())
}

pub fn exp(x: f32) -> f32 {
    cr(MathFn::Exp, x)
}

pub fn softmax_rows(x: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; rows * cols];
    for r in 0..rows {
        let base = r * cols;
        let mut peak = x[base];
        for c in 1..cols {
            if x[base + c] > peak {
                peak = x[base + c];
            }
        }
        let mut total = 0.0f32;
        for c in 0..cols {
            let e = exp(x[base + c] - peak);
            out[base + c] = e;
            total += e;
        }
        for c in 0..cols {
            out[base + c] /= total;
        }
    }
    out
}

pub fn rmsnorm_rows(x: &[f32], w: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; rows * cols];
    let inv_n = 1.0f32 / cols as f32;
    for r in 0..rows {
        let base = r * cols;
        let mut ss = 0.0f32;
        for c in 0..cols {
            ss += x[base + c] * x[base + c];
        }
        let scale = 1.0f32 / (ss * inv_n + 0.00001f32).sqrt();
        for c in 0..cols {
            out[base + c] = x[base + c] * scale * w[c];
        }
    }
    out
}

pub fn layernorm_rows(
    x: &[f32],
    gamma: &[f32],
    beta: &[f32],
    rows: usize,
    cols: usize,
) -> Vec<f32> {
    let mut out = vec![0.0f32; rows * cols];
    let n = cols as f32;
    for r in 0..rows {
        let base = r * cols;
        let mut sum = 0.0f32;
        for c in 0..cols {
            sum += x[base + c];
        }
        let mean = sum / n;
        let mut spread = 0.0f32;
        for c in 0..cols {
            let d = x[base + c] - mean;
            spread += d * d;
        }
        let inv = 1.0f32 / (spread / n + 0.00001f32).sqrt();
        for c in 0..cols {
            out[base + c] = (x[base + c] - mean) * inv * gamma[c] + beta[c];
        }
    }
    out
}

pub fn gemm_f32(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0.0f32;
            for p in 0..k {
                acc += a[i * k + p] * b[p * n + j];
            }
            out[i * n + j] = acc;
        }
    }
    out
}

/// `value` rounded to binary16 (nearest, ties to even), as an exact integer
/// multiple of 2^-24. Inputs here are finite and inside the binary16 range.
fn f16_units(value: f32) -> i128 {
    let wide = f64::from(value).abs();
    if wide == 0.0 {
        return 0;
    }
    let exponent = ((wide.to_bits() >> 52) & 0x7ff) as i32 - 1023;
    // Binary16 spacing: 2^(e-10) for normals, 2^-24 below 2^-14.
    let step = f64::powi(2.0, (exponent - 10).max(-24) + 24);
    let units = (wide * f64::powi(2.0, 24) / step).round_ties_even() * step;
    assert!(
        units <= 65504.0 * f64::powi(2.0, 24),
        "{value} is outside binary16"
    );
    let units = units as i128;
    if value < 0.0 { -units } else { units }
}

/// GEMM under the `tc-f16-f32-v1` contract of `tc_mul_add`: every input is
/// rounded to binary16 and a cell is the exact sum of the products, rounded
/// once to binary32. Exact integer arithmetic; no floating-point accumulation.
pub fn gemm_tc(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let exact: i128 = (0..k)
                .map(|p| f16_units(a[i * k + p]) * f16_units(b[p * n + j]))
                .sum();
            // |exact| < 2^53: the f64 holds it exactly, `as f32` rounds once.
            assert!(exact.unsigned_abs() < 1 << 53);
            out[i * n + j] = (exact as f64 * f64::powi(2.0, -48)) as f32;
        }
    }
    out
}

/// Single-head attention without a mask: `softmax(Q K^T / sqrt(dim)) V`.
pub fn attention(q: &[f32], k: &[f32], v: &[f32], seq: usize, dim: usize) -> Vec<f32> {
    let scale = 1.0f32 / (dim as f32).sqrt();
    let mut scores = vec![0.0f32; seq * seq];
    for i in 0..seq {
        for j in 0..seq {
            let mut acc = 0.0f32;
            for d in 0..dim {
                acc += q[i * dim + d] * k[j * dim + d];
            }
            scores[i * seq + j] = acc * scale;
        }
    }
    let weights = softmax_rows(&scores, seq, seq);
    gemm_f32(&weights, v, seq, seq, dim)
}

pub fn particle_integrate(p0: &[f32], v0: &[f32], n: usize, steps: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; n * 3];
    for i in 0..n {
        let b = i * 3;
        let (mut x, mut y, mut z) = (p0[b], p0[b + 1] + 1.0, p0[b + 2]);
        let (vx, mut vy, vz) = (v0[b], v0[b + 1], v0[b + 2]);
        for _ in 0..steps {
            vy -= 0.0981;
            x += vx * 0.01;
            y += vy * 0.01;
            z += vz * 0.01;
            if y < 0.0 {
                y = 0.0 - y;
                vy = 0.0 - vy * 0.5;
            }
        }
        out[b..b + 3].copy_from_slice(&[x, y, z]);
    }
    out
}

fn cell_of(v: f32, g: u32) -> u32 {
    let c = ((v + 1.0) * 0.5 * g as f32) as u32;
    if c >= g { g - 1 } else { c }
}

/// Per particle, the others within `sqrt(r2)` found in the 3x3 cells around it.
pub fn grid_broadphase(pos: &[f32], n: usize, g: u32, r2: f32) -> Vec<u32> {
    let mut hits = vec![0u32; n];
    for i in 0..n {
        let (cx, cy) = (cell_of(pos[i * 2], g), cell_of(pos[i * 2 + 1], g));
        for j in 0..n {
            if j == i {
                continue;
            }
            let (jx, jy) = (cell_of(pos[j * 2], g), cell_of(pos[j * 2 + 1], g));
            if jx.abs_diff(cx) > 1 || jy.abs_diff(cy) > 1 {
                continue;
            }
            let dx = pos[j * 2] - pos[i * 2];
            let dy = pos[j * 2 + 1] - pos[i * 2 + 1];
            if dx * dx + dy * dy < r2 {
                hits[i] += 1;
            }
        }
    }
    hits
}

pub fn skinning(verts: &[f32], weights: &[f32], bones: &[f32], n: usize, nb: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; n * 3];
    for i in 0..n {
        let (x, y, z) = (verts[i * 3], verts[i * 3 + 1], verts[i * 3 + 2]);
        for r in 0..3 {
            let mut acc = 0.0f32;
            for k in 0..4 {
                let m = ((i + k) % nb) * 12 + r * 4;
                let t = bones[m] * x + bones[m + 1] * y + bones[m + 2] * z + bones[m + 3];
                acc += weights[i * 4 + k] * t;
            }
            out[i * 3 + r] = acc;
        }
    }
    out
}

const PLANES: [f32; 24] = [
    0.894, 0.0, 0.447, 0.9, -0.894, 0.0, 0.447, 0.9, 0.0, 0.894, 0.447, 0.9, 0.0, -0.894, 0.447,
    0.9, 0.0, 0.0, 1.0, 0.5, 0.0, 0.0, -1.0, 1.5,
];

pub fn frustum_cull(spheres: &[f32], n: usize) -> Vec<u32> {
    (0..n)
        .map(|i| {
            let (x, y, z) = (spheres[i * 4], spheres[i * 4 + 1], spheres[i * 4 + 2]);
            let r = (spheres[i * 4 + 3] + 1.0) * 0.1;
            let inside = (0..6).all(|p| {
                let d = PLANES[p * 4] * x
                    + PLANES[p * 4 + 1] * y
                    + PLANES[p * 4 + 2] * z
                    + PLANES[p * 4 + 3];
                d >= 0.0 - r
            });
            u32::from(inside)
        })
        .collect()
}

/// Breadth-first distances on an `n` by `n` grid from cell 0; a cell is a
/// wall when its value exceeds 0.4, and walls and unreachable cells are `u32::MAX`.
pub fn grid_bfs(walls: &[f32], n: usize) -> Vec<u32> {
    let mut dist = vec![u32::MAX; n * n];
    dist[0] = 0;
    let mut frontier = vec![0usize];
    let mut level = 0;
    while !frontier.is_empty() {
        let mut next = Vec::new();
        for &cell in &frontier {
            let (cx, cy) = (cell % n, cell / n);
            let neighbours = [
                (cx + 1 < n).then(|| cy * n + cx + 1),
                (cx > 0).then(|| cy * n + cx - 1),
                (cy + 1 < n).then(|| (cy + 1) * n + cx),
                (cy > 0).then(|| (cy - 1) * n + cx),
            ];
            for at in neighbours.into_iter().flatten() {
                if walls[at] <= 0.4 && dist[at] == u32::MAX {
                    dist[at] = level + 1;
                    next.push(at);
                }
            }
        }
        frontier = next;
        level += 1;
    }
    dist
}

pub fn sin(x: f32) -> f32 {
    cr(MathFn::Sin, x)
}

pub fn cos(x: f32) -> f32 {
    cr(MathFn::Cos, x)
}

pub fn batched_fk(q: &[f32], n: usize, links: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; n * 3];
    for i in 0..n {
        let (mut x, mut y, mut a) = (0.0f32, 0.0f32, 0.0f32);
        for j in 0..links {
            a += q[i * links + j];
            x += 0.5 * cos(a);
            y += 0.5 * sin(a);
        }
        out[i * 3..i * 3 + 3].copy_from_slice(&[x, y, a]);
    }
    out
}

/// One damped least-squares step (lambda squared 0.01) of a planar three-link
/// arm with link length 0.5 toward `target`.
pub fn dls_ik_step(q: &[f32], target: &[f32], n: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; n * 3];
    for i in 0..n {
        let a0 = q[i * 3];
        let a1 = a0 + q[i * 3 + 1];
        let a2 = a1 + q[i * 3 + 2];
        let (s0, s1, s2) = (0.5 * sin(a0), 0.5 * sin(a1), 0.5 * sin(a2));
        let (c0, c1, c2) = (0.5 * cos(a0), 0.5 * cos(a1), 0.5 * cos(a2));
        let ex = target[i * 2] - (c0 + c1 + c2);
        let ey = target[i * 2 + 1] - (s0 + s1 + s2);
        let sx1 = s2 + s1;
        let sx0 = sx1 + s0;
        let cx1 = c2 + c1;
        let cx0 = cx1 + c0;
        let (j00, j01, j02) = (0.0 - sx0, 0.0 - sx1, 0.0 - s2);
        let (j10, j11, j12) = (cx0, cx1, c2);
        let m00 = j00 * j00 + j01 * j01 + j02 * j02 + 0.01;
        let m01 = j00 * j10 + j01 * j11 + j02 * j12;
        let m11 = j10 * j10 + j11 * j11 + j12 * j12 + 0.01;
        let det = m00 * m11 - m01 * m01;
        let w0 = (m11 * ex - m01 * ey) / det;
        let w1 = (m00 * ey - m01 * ex) / det;
        out[i * 3] = q[i * 3] + (j00 * w0 + j10 * w1);
        out[i * 3 + 1] = q[i * 3 + 1] + (j01 * w0 + j11 * w1);
        out[i * 3 + 2] = q[i * 3 + 2] + (j02 * w0 + j12 * w1);
    }
    out
}

pub fn sphere_collision(a: &[f32], b: &[f32], na: usize, nb: usize) -> Vec<u32> {
    (0..na)
        .map(|i| {
            let ra = (a[i * 4 + 3] + 1.0) * 0.15;
            (0..nb)
                .filter(|&j| {
                    let rb = (b[j * 4 + 3] + 1.0) * 0.15;
                    let dx = a[i * 4] - b[j * 4];
                    let dy = a[i * 4 + 1] - b[j * 4 + 1];
                    let dz = a[i * 4 + 2] - b[j * 4 + 2];
                    let reach = ra + rb;
                    dx * dx + dy * dy + dz * dz < reach * reach
                })
                .count() as u32
        })
        .collect()
}

/// Unicycle rollouts (dt 0.1) of `k` control sequences toward the goal (1, 0.5).
pub fn mpc_rollout(controls: &[f32], k: usize, horizon: usize) -> Vec<f32> {
    (0..k)
        .map(|c| {
            let (mut x, mut y, mut th, mut total) = (0.0f32, 0.0f32, 0.0f32, 0.0f32);
            for h in 0..horizon {
                let v = controls[(c * horizon + h) * 2];
                let w = controls[(c * horizon + h) * 2 + 1];
                th += w * 0.1;
                x += v * cos(th) * 0.1;
                y += v * sin(th) * 0.1;
                let (dx, dy) = (x - 1.0, y - 0.5);
                total += dx * dx + dy * dy + 0.01 * (v * v + w * w);
            }
            total
        })
        .collect()
}

pub fn voxel_filter(points: &[f32], n: usize, g: u32) -> Vec<f32> {
    let cells = (g * g * g) as usize;
    let mut sums = vec![0.0f32; cells * 3];
    let mut counts = vec![0u32; cells];
    for i in 0..n {
        let at = |d: usize| cell_of(points[i * 3 + d], g);
        let cell = ((at(2) * g + at(1)) * g + at(0)) as usize;
        for d in 0..3 {
            sums[cell * 3 + d] += points[i * 3 + d];
        }
        counts[cell] += 1;
    }
    let mut out = vec![0.0f32; cells * 3];
    for c in 0..cells {
        if counts[c] > 0 {
            let inv = 1.0f32 / counts[c] as f32;
            for d in 0..3 {
                out[c * 3 + d] = sums[c * 3 + d] * inv;
            }
        }
    }
    out
}

pub fn radius_filter(points: &[f32], n: usize, r2: f32) -> Vec<u32> {
    (0..n)
        .map(|i| {
            (0..n)
                .filter(|&j| {
                    let dx = points[i * 3] - points[j * 3];
                    let dy = points[i * 3 + 1] - points[j * 3 + 1];
                    let dz = points[i * 3 + 2] - points[j * 3 + 2];
                    j != i && dx * dx + dy * dy + dz * dz < r2
                })
                .count() as u32
        })
        .collect()
}
