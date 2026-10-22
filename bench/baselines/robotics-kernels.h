// External handwritten robotics baselines (GNET-05 comparison D), not Invar-generated code.
// One source for Metal (compiled at runtime by metal-robotics-handwritten.mm) and CUDA
// (included by cuda-robotics-handwritten.cu). Every binary32 operation mirrors the frozen
// Invar source of bench/general-net-v01/gnet05/robotics in the same order, without
// contraction, so outputs are bit-identical to the reference evaluator.
//
// Prior art ported, not copied: the chain uses the cuRobo v0.8.0 (NVlabs/curobo @ 4ea77366,
// Apache-2.0) kinematics_fused_kernel decomposition, four lanes per configuration (three
// rotation columns and the translation column) with the joint-local matrix JM = A * Rot(axis)
// formed per column (xyz_rot_fn) and composed with the parent's cumulative frame
// (apply_joint_transform). Dynamics follow the batched-dynamics layout of GRiD and the
// Pinocchio CRBA definition (M = sum_b m Jv^T Jv + Jw^T I Jw), computed per configuration.
// Trigonometry is Cephes sinf/cosf; collision uses the exact original capsule/cylinder
// primitives of the UR5e (no sphere envelopes).
#pragma once

#if defined(__METAL_VERSION__)
#include <metal_stdlib>
using namespace metal;
#pragma METAL fp contract(off)
#define RK_DEV device
#define RK_TG threadgroup
#define RK_FN static inline
#define RK_BARRIER() threadgroup_barrier(mem_flags::mem_threadgroup)
#define RK_DIV(a, b) precise::divide((a), (b))
#define RK_F(x) as_type<float>(x)
typedef uint u32;
#else
#define RK_DEV
#define RK_TG
#define RK_FN static __device__ __forceinline__
#define RK_BARRIER() __syncthreads()
#define RK_DIV(a, b) __fdiv_rn((a), (b))
#define RK_F(x) __uint_as_float(x)
typedef unsigned int u32;
#endif

// Configurations per threadgroup and lanes per configuration.
#define RK_CONFIGS 16
#define RK_LANES 4
#define RK_THREADS (RK_CONFIGS * RK_LANES)
// Threadgroup words per configuration: frames (<= 11 bodies), cos/sin, scratch.
#define RK_MAX_BODIES 11
#define RK_MAX_JOINTS 7
#define RK_MAX_GEOMS 9
#define RK_FRAME_WORDS (RK_MAX_BODIES * 12)
#define RK_SCRATCH_WORDS 64
#define RK_CLOUD_BLOCK 256

// Word offsets of each model table in the `model` buffer (see robotics-tables.h), the
// runtime inputs and the cloud parameters. Four-byte fields only.
struct RkParams {
  u32 seed, n, m, trig;
  u32 nb, nj, ng, tb;
  u32 par, fix, ax, jt, jx, lo, span, hi, mv, bt, ar, gi, gt, tool;
  u32 cloud_range, blocks, points, pad1;
  float sx, ox, sy, oy, sz;
  float crop_x, crop_y, crop_z0, crop_z1, crop_r0, crop_r1;
};

struct RkCs { float c, s; };

RK_FN u32 rk_lcg(u32 x) { return x * 1664525u + 1013904223u; }
RK_FN float rk_unit(u32 x) { return (float)(x / 256u) * 0.000000059604645f; }
RK_FN u32 rk_stream(u32 seed, u32 i) { return rk_lcg(i * 2654435761u + seed); }

// Cephes sinf/cosf after a three-part Cody-Waite reduction by pi/2.
RK_FN RkCs rk_sincos(float x) {
  float k = 0.0f;
  u32 m = 0u;
  while (x * 0.63661975f - k > 0.5f) { k = k + 1.0f; m = m + 1u; }
  while (x * 0.63661975f - k < -0.5f) { k = k - 1.0f; m = m + 3u; }
  float r = x - k * 1.5703125f - k * 0.0004837513f - k * 0.0000000754979f;
  float z = r * r;
  float s = ((-0.00019515296f * z + 0.008332161f) * z - 0.16666655f) * z * r + r;
  float c = ((0.000024433157f * z - 0.0013887316f) * z + 0.041666646f) * z * z - 0.5f * z + 1.0f;
  u32 q = m % 4u;
  RkCs o;
  o.c = c; o.s = s;
  if (q == 1u) { o.s = c; o.c = -s; }
  else if (q == 2u) { o.s = -s; o.c = -c; }
  else if (q == 3u) { o.s = -c; o.c = s; }
  return o;
}

RK_FN RkCs rk_rational(float u) {
  float d = 1.0f + u * u;
  RkCs o;
  o.c = RK_DIV(1.0f - u * u, d);
  o.s = RK_DIV(u + u, d);
  return o;
}

// Division-free Newton square root.
RK_FN float rk_root(float x) {
  if (x <= 0.0f) return 0.0f;
  float m = x, g = 1.0f;
  while (m >= 4.0f) { m = m * 0.25f; g = g * 2.0f; }
  while (m < 1.0f) { m = m * 4.0f; g = g * 0.5f; }
  float y = 1.1666666f - 0.16666667f * m;
  for (int i = 0; i < 5; i++) y = y * (1.5f - 0.5f * m * y * y);
  return m * y * g;
}

#define RK_W(o) RK_F(model[(o)])
#define RK_U(o) (model[(o)])

// cos/sin of every joint of configuration b, joints split across lanes.
RK_FN void rk_fill(RK_TG float* cs, RK_DEV const u32* model, RkParams p, u32 b, u32 lane) {
  u32 x = rk_stream(p.seed, b);
  for (u32 j = 0u; j < p.nj; j++) {
    x = rk_lcg(x);
    if (j % RK_LANES != lane) continue;
    RkCs v;
    if (p.trig != 0u) {
      v = rk_sincos(RK_W(p.lo + j) + RK_W(p.span + j) * rk_unit(x));
    } else {
      v = rk_rational(2.0f * rk_unit(x) - 1.0f);
    }
    cs[2u * j] = v.c;
    cs[2u * j + 1u] = v.s;
  }
}

// Joint-local matrix entry JM[k][col] = (A * Rot(axis, q))[k][col].
RK_FN float rk_jm(RK_DEV const u32* model, u32 ao, u32 a, u32 k, u32 col, float c, float s) {
  float x = RK_W(ao + 3u * k), y = RK_W(ao + 3u * k + 1u), z = RK_W(ao + 3u * k + 2u);
  if (col == 0u) {
    if (a == 3u) return c * x + s * y;
    if (a == 2u) return c * x - s * z;
    return x;
  }
  if (col == 1u) {
    if (a == 3u) return c * y - s * x;
    if (a == 1u) return c * y + s * z;
    return y;
  }
  if (a == 2u) return s * x + c * z;
  if (a == 1u) return c * z - s * y;
  return z;
}

// cuRobo-style chain: lane 0..2 computes rotation column `lane`, lane 3 the translation.
// Contains barriers: every thread of the group must call it.
RK_FN void rk_fk(RK_TG float* w, RK_TG const float* cs, RK_DEV const u32* model, RkParams p,
                 u32 lane) {
  for (u32 e = lane; e < 12u; e += RK_LANES) w[e] = RK_W(p.fix + e);
  RK_BARRIER();
  for (u32 i = 1u; i < p.nb; i++) {
    u32 so = 12u * i, po = 12u * RK_U(p.par + i), ao = p.fix + 12u * i;
    u32 a = RK_U(p.ax + i);
    u32 jt = RK_U(p.jt + i);
    float c = cs[2u * jt], s = cs[2u * jt + 1u];
    if (lane < 3u) {
      float j0 = rk_jm(model, ao, a, 0u, lane, c, s);
      float j1 = rk_jm(model, ao, a, 1u, lane, c, s);
      float j2 = rk_jm(model, ao, a, 2u, lane, c, s);
      for (u32 r = 0u; r < 3u; r++) {
        u32 pr = po + 3u * r;
        w[so + 3u * r + lane] = w[pr] * j0 + w[pr + 1u] * j1 + w[pr + 2u] * j2;
      }
    } else {
      for (u32 r = 0u; r < 3u; r++) {
        u32 pr = po + 3u * r;
        w[so + 9u + r] = w[pr] * RK_W(ao + 9u) + w[pr + 1u] * RK_W(ao + 10u) +
                         w[pr + 2u] * RK_W(ao + 11u) + w[po + 9u + r];
      }
    }
    RK_BARRIER();
  }
}

RK_FN float rk_tool(RK_TG const float* t, RK_DEV const u32* model, RkParams p, u32 i) {
  u32 r = 12u * p.tb + 3u * i;
  return t[r] * RK_W(p.tool) + t[r + 1u] * RK_W(p.tool + 1u) + t[r + 2u] * RK_W(p.tool + 2u) +
         t[12u * p.tb + 9u + i];
}

// Exact signed distance from a sphere to geom g (capsule or capped cylinder).
RK_FN float rk_distance(RK_TG const float* gw, RK_DEV const u32* model, RkParams p, u32 g,
                        float ox, float oy, float oz, float orad) {
  u32 k = p.gt + 10u * g, w = 6u * g;
  float cx = gw[w], cy = gw[w + 1u], cz = gw[w + 2u];
  float ax = gw[w + 3u], ay = gw[w + 4u], az = gw[w + 5u];
  float r = RK_W(k), h = RK_W(k + 1u);
  if (RK_U(p.gi + 2u * g + 1u) == 0u) {
    float p0x = cx - h * ax, p0y = cy - h * ay, p0z = cz - h * az;
    float dx = RK_W(k + 2u) * ax, dy = RK_W(k + 2u) * ay, dz = RK_W(k + 2u) * az;
    float u = ((ox - p0x) * dx + (oy - p0y) * dy + (oz - p0z) * dz) * RK_W(k + 3u);
    if (u < 0.0f) u = 0.0f;
    if (u > 1.0f) u = 1.0f;
    float ex = ox - (p0x + u * dx), ey = oy - (p0y + u * dy), ez = oz - (p0z + u * dz);
    return rk_root(ex * ex + ey * ey + ez * ez) - (r + orad);
  }
  float vx = ox - cx, vy = oy - cy, vz = oz - cz;
  float along = vx * ax + vy * ay + vz * az;
  float rx = vx - along * ax, ry = vy - along * ay, rz = vz - along * az;
  float radial = rk_root(rx * rx + ry * ry + rz * rz);
  float aa = along;
  if (aa < 0.0f) aa = -aa;
  float qx = radial - r, qy = aa - h;
  float px = qx;
  if (px < 0.0f) px = 0.0f;
  float py = qy;
  if (py < 0.0f) py = 0.0f;
  float inside = qx;
  if (qy > inside) inside = qy;
  if (inside > 0.0f) inside = 0.0f;
  return rk_root(px * px + py * py) + inside - orad;
}

// World centre and axis of geoms g = lane, lane + 4, ...
RK_FN void rk_place(RK_TG float* gw, RK_TG const float* t, RK_DEV const u32* model, RkParams p,
                    u32 lane) {
  for (u32 g = lane; g < p.ng; g += RK_LANES) {
    u32 bo = 12u * RK_U(p.gi + 2u * g), k = p.gt + 10u * g;
    for (u32 r = 0u; r < 3u; r++) {
      u32 br = bo + 3u * r;
      gw[6u * g + r] = t[br] * RK_W(k + 4u) + t[br + 1u] * RK_W(k + 5u) +
                       t[br + 2u] * RK_W(k + 6u) + t[bo + 9u + r];
      gw[6u * g + 3u + r] = t[br] * RK_W(k + 7u) + t[br + 1u] * RK_W(k + 8u) +
                            t[br + 2u] * RK_W(k + 9u);
    }
  }
}

// Per-configuration threadgroup slices.
#define RK_SLICE (RK_FRAME_WORDS + 2 * RK_MAX_JOINTS + 9 * RK_MAX_JOINTS + 6 * RK_MAX_GEOMS + 16)

// Families fk, fkpre, jacobian, mass, collision, pipeline: one configuration per 4 lanes.
// `obs` holds collision spheres (4 per obstacle) or the pipeline's compacted points, with
// the kept count in `count[0]`.
RK_FN void rk_config(u32 family, RK_TG float* slice, RK_DEV const u32* model, RkParams p,
                     RK_DEV float* out, RK_DEV const float* obs, RK_DEV const u32* count,
                     u32 b, u32 lane) {
  RK_TG float* t = slice;
  RK_TG float* cs = slice + RK_FRAME_WORDS;
  RK_TG float* sv = cs + 2 * RK_MAX_JOINTS;
  RK_TG float* gw = sv + 9 * RK_MAX_JOINTS;
  RK_TG float* best = gw + 6 * RK_MAX_GEOMS;
  bool live = b < p.n;
  rk_fill(cs, model, p, b, lane);
  RK_BARRIER();
  rk_fk(t, cs, model, p, lane);
  u32 nj = p.nj;
  if (family == 0u) {  // fk, fkpre: all frames
    if (live)
      for (u32 e = lane; e < 12u * p.nb; e += RK_LANES) out[b * 12u * p.nb + e] = t[e];
  } else if (family == 1u) {  // jacobian
    float px = rk_tool(t, model, p, 0u), py = rk_tool(t, model, p, 1u),
          pz = rk_tool(t, model, p, 2u);
    u32 o = b * 6u * nj;
    for (u32 j = lane; j < nj && live; j += RK_LANES) {
      u32 bo = 12u * RK_U(p.jx + 2u * j), a = RK_U(p.jx + 2u * j + 1u) - 1u;
      float zx = t[bo + a], zy = t[bo + 3u + a], zz = t[bo + 6u + a];
      float dx = px - t[bo + 9u], dy = py - t[bo + 10u], dz = pz - t[bo + 11u];
      out[o + j] = zy * dz - zz * dy;
      out[o + nj + j] = zz * dx - zx * dz;
      out[o + 2u * nj + j] = zx * dy - zy * dx;
      out[o + 3u * nj + j] = zx;
      out[o + 4u * nj + j] = zy;
      out[o + 5u * nj + j] = zz;
    }
  } else if (family == 2u) {  // mass: rows i = lane, lane + 4, ... owned by this lane
    u32 o = b * nj * nj;
    for (u32 i = lane; i < nj && live; i += RK_LANES)
      for (u32 jc = 0u; jc < nj; jc++) out[o + i * nj + jc] = 0.0f;
    for (u32 bi = 1u; bi < p.nb; bi++) {
      u32 bo = 12u * bi, k = p.bt + 13u * bi;
      float cx = t[bo] * RK_W(k + 1u) + t[bo + 1u] * RK_W(k + 2u) + t[bo + 2u] * RK_W(k + 3u) + t[bo + 9u];
      float cy = t[bo + 3u] * RK_W(k + 1u) + t[bo + 4u] * RK_W(k + 2u) + t[bo + 5u] * RK_W(k + 3u) + t[bo + 10u];
      float cz = t[bo + 6u] * RK_W(k + 1u) + t[bo + 7u] * RK_W(k + 2u) + t[bo + 8u] * RK_W(k + 3u) + t[bo + 11u];
      for (u32 j = lane; j < nj; j += RK_LANES) {
        u32 s = 9u * j;
        if (RK_U(p.mv + bi * nj + j) == 1u) {
          u32 jo = 12u * RK_U(p.jx + 2u * j), axis = RK_U(p.jx + 2u * j + 1u) - 1u;
          float zx = t[jo + axis], zy = t[jo + 3u + axis], zz = t[jo + 6u + axis];
          float dx = cx - t[jo + 9u], dy = cy - t[jo + 10u], dz = cz - t[jo + 11u];
          float wx = t[bo] * zx + t[bo + 3u] * zy + t[bo + 6u] * zz;
          float wy = t[bo + 1u] * zx + t[bo + 4u] * zy + t[bo + 7u] * zz;
          float wz = t[bo + 2u] * zx + t[bo + 5u] * zy + t[bo + 8u] * zz;
          sv[s] = zy * dz - zz * dy;
          sv[s + 1u] = zz * dx - zx * dz;
          sv[s + 2u] = zx * dy - zy * dx;
          sv[s + 3u] = wx;
          sv[s + 4u] = wy;
          sv[s + 5u] = wz;
          sv[s + 6u] = RK_W(k + 4u) * wx + RK_W(k + 5u) * wy + RK_W(k + 6u) * wz;
          sv[s + 7u] = RK_W(k + 7u) * wx + RK_W(k + 8u) * wy + RK_W(k + 9u) * wz;
          sv[s + 8u] = RK_W(k + 10u) * wx + RK_W(k + 11u) * wy + RK_W(k + 12u) * wz;
        } else {
          for (u32 e = 0u; e < 9u; e++) sv[s + e] = 0.0f;
        }
      }
      RK_BARRIER();
      float m = RK_W(k);
      for (u32 i = lane; i < nj && live; i += RK_LANES) {
        u32 si = 9u * i;
        for (u32 jc = 0u; jc < nj; jc++) {
          u32 sj = 9u * jc;
          float lin = sv[si] * sv[sj] + sv[si + 1u] * sv[sj + 1u] + sv[si + 2u] * sv[sj + 2u];
          float ang = sv[si + 3u] * sv[sj + 6u] + sv[si + 4u] * sv[sj + 7u] + sv[si + 5u] * sv[sj + 8u];
          out[o + i * nj + jc] = out[o + i * nj + jc] + (m * lin + ang);
        }
      }
      RK_BARRIER();
    }
    for (u32 d = lane; d < nj && live; d += RK_LANES)
      out[o + d * nj + d] = out[o + d * nj + d] + RK_W(p.ar + d);
  } else if (family == 3u) {  // collision: (geom, obstacle) pairs split across lanes
    rk_place(gw, t, model, p, lane);
    RK_BARRIER();
    for (u32 e = lane; e < p.ng * p.m && live; e += RK_LANES) {
      u32 g = e / p.m, k = e % p.m;
      out[(b * p.ng + g) * p.m + k] =
          rk_distance(gw, model, p, g, obs[4u * k], obs[4u * k + 1u], obs[4u * k + 2u], obs[4u * k + 3u]);
    }
  } else {  // pipeline: per-geom minimum over kept points, then the ordered clearance cost
    rk_place(gw, t, model, p, lane);
    RK_BARRIER();
    u32 kept = count[0];
    for (u32 g = lane; g < p.ng; g += RK_LANES) {
      float bst = 1000.0f;
      for (u32 k = 0u; k < kept; k++) {
        float d = rk_distance(gw, model, p, g, obs[3u * k], obs[3u * k + 1u], obs[3u * k + 2u], 0.0f);
        if (d < bst) bst = d;
      }
      best[g] = bst;
      if (live) out[b * (p.ng + 1u) + g] = bst;
    }
    RK_BARRIER();
    if (lane == 0u && live) {
      float cost = 0.0f;
      for (u32 g = 0u; g < p.ng; g++)
        if (best[g] < 0.05f) cost = cost + (0.05f - best[g]) * (0.05f - best[g]);
      out[b * (p.ng + 1u) + p.ng] = cost;
    }
  }
}

// Rollout: one trajectory per 4 lanes; joints integrated by their lane, every lane replays
// the control stream to keep the ordered control-energy sum.
RK_FN void rk_rollout(RK_TG float* slice, RK_DEV const u32* model, RkParams p, RK_DEV float* out,
                      u32 b, u32 lane) {
  RK_TG float* t = slice;
  RK_TG float* cs = slice + RK_FRAME_WORDS;
  RK_TG float* qv = cs + 2 * RK_MAX_JOINTS;  // q then v
  bool live = b < p.n;
  u32 nj = p.nj, h = p.m;
  u32 x = rk_stream(p.seed * 747796405u, 0u);
  x = rk_lcg(x);
  float tx = 1.2f * rk_unit(x) - 0.6f;
  x = rk_lcg(x);
  float ty = 1.2f * rk_unit(x) - 0.6f;
  x = rk_lcg(x);
  float tz = 0.8f * rk_unit(x) + 0.1f;
  u32 y = rk_stream(p.seed, b);
  for (u32 j = 0u; j < nj; j++) {
    y = rk_lcg(y);
    if (j % RK_LANES == lane) {
      qv[j] = RK_W(p.lo + j) + RK_W(p.span + j) * rk_unit(y);
      qv[nj + j] = 0.0f;
    }
  }
  for (u32 k = 0u; k < h; k++) {
    float uu = 0.0f;
    for (u32 j = 0u; j < nj; j++) {
      y = rk_lcg(y);
      float u = (2.0f * rk_unit(y) - 1.0f) * 2.0f;
      uu = uu + u * u;
      if (j % RK_LANES != lane) continue;
      qv[nj + j] = qv[nj + j] + 0.01f * u;
      float q = qv[j] + 0.01f * qv[nj + j];
      if (q < RK_W(p.lo + j)) { q = RK_W(p.lo + j); qv[nj + j] = 0.0f; }
      if (q > RK_W(p.hi + j)) { q = RK_W(p.hi + j); qv[nj + j] = 0.0f; }
      qv[j] = q;
      RkCs v = rk_sincos(q);
      cs[2u * j] = v.c;
      cs[2u * j + 1u] = v.s;
    }
    RK_BARRIER();
    rk_fk(t, cs, model, p, lane);
    u32 o = (b * h + k) * (nj + 1u);
    for (u32 j = lane; j < nj && live; j += RK_LANES) out[o + j] = qv[j];
    if (lane == 0u && live) {
      float ex = rk_tool(t, model, p, 0u) - tx;
      float ey = rk_tool(t, model, p, 1u) - ty;
      float ez = rk_tool(t, model, p, 2u) - tz;
      out[o + nj] = ex * ex + ey * ey + ez * ez + 0.001f * uu;
    }
    RK_BARRIER();
  }
}

// Sensor frame (cos, sin, tx, ty, tz) from the seed; one thread.
RK_FN void rk_sensor(RK_DEV float* fr, RkParams p) {
  u32 x = rk_stream(p.seed * 747796405u, 0u);
  x = rk_lcg(x);
  RkCs v;
  if (p.trig != 0u) {
    v = rk_sincos(6.2831855f * rk_unit(x) - 3.1415927f);
  } else {
    v = rk_rational(2.0f * rk_unit(x) - 1.0f);
  }
  fr[0] = v.c;
  fr[1] = v.s;
  x = rk_lcg(x);
  fr[2] = 0.5f * rk_unit(x) - 0.25f;
  x = rk_lcg(x);
  fr[3] = 0.5f * rk_unit(x) - 0.25f;
  x = rk_lcg(x);
  fr[4] = 0.2f * rk_unit(x);
}

// Collision spheres from the seed; one thread.
RK_FN void rk_spheres(RK_DEV float* obs, RkParams p) {
  u32 x = rk_stream(p.seed * 747796405u, 0u);
  for (u32 k = 0u; k < p.m; k++) {
    x = rk_lcg(x);
    obs[4u * k] = 1.8f * rk_unit(x) - 0.9f;
    x = rk_lcg(x);
    obs[4u * k + 1u] = 1.8f * rk_unit(x) - 0.9f;
    x = rk_lcg(x);
    obs[4u * k + 2u] = 1.2f * rk_unit(x) - 0.1f;
    x = rk_lcg(x);
    obs[4u * k + 3u] = 0.1f * rk_unit(x) + 0.02f;
  }
}

// Cloud point i: sample, transform and filter; returns the keep flag.
RK_FN u32 rk_sample(RK_DEV float* pts, RK_DEV const float* fr, RkParams p, u32 i) {
  u32 y = rk_stream(p.seed, i);
  y = rk_lcg(y);
  float px = p.sx * rk_unit(y) - p.ox;
  y = rk_lcg(y);
  float py = p.sy * rk_unit(y) - p.oy;
  y = rk_lcg(y);
  float pz = p.sz * rk_unit(y);
  float wx = fr[0] * px - fr[1] * py + fr[2];
  float wy = fr[1] * px + fr[0] * py + fr[3];
  float wz = pz + fr[4];
  pts[3u * i] = wx;
  pts[3u * i + 1u] = wy;
  pts[3u * i + 2u] = wz;
  bool keep = wx >= -p.crop_x && wx <= p.crop_x && wy >= -p.crop_y && wy <= p.crop_y &&
              wz >= p.crop_z0 && wz <= p.crop_z1;
  if (p.cloud_range != 0u) {
    float r2 = wx * wx + wy * wy + wz * wz;
    keep = keep && r2 >= p.crop_r0 && r2 <= p.crop_r1;
  }
  return keep ? 1u : 0u;
}

// Neighbour count of kept point a among `kept` compacted points at `pts` (3 per point).
RK_FN float rk_neighbours(RK_DEV const float* pts, u32 kept, u32 a) {
  float ax = pts[3u * a], ay = pts[3u * a + 1u], az = pts[3u * a + 2u];
  u32 count = 0u;
  for (u32 k = 0u; k < kept; k++) {
    float dx = pts[3u * k] - ax, dy = pts[3u * k + 1u] - ay, dz = pts[3u * k + 2u] - az;
    if (dx * dx + dy * dy + dz * dz <= 0.0225f) count = count + 1u;
  }
  return (float)count;
}
