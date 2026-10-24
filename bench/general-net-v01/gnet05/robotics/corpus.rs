//! GNET-05 robotics corpus generator (`gnet05-robotics-gen-v1`).
//!
//! Eight families, six programs each: two-arm FK (angle-driven), its
//! precomputed-transform control, full tool Jacobian, joint-space mass,
//! UR5e exact-primitive collision distance, rollout/cost, point-cloud
//! transform/filter/compact/neighbourhood, and a composed UR5e pipeline.
//! A seeded shuffle holds out three programs per family under the same
//! normalized-identity guard as the dynamic corpus.
use super::arm::Arm;
use super::families::{self, Case};
use super::model;
use crate::generate::{HeldOut, Rng, Split, hex, normalized_sha256};
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;
use std::fmt::Write;
use std::path::Path;

pub const GENERATOR: &str = "gnet05-robotics-gen-v1";
pub const ROOT_SEED: u64 = 0x474e_4554_3035_5242;
pub const PER_FAMILY: usize = 6;
pub const HELD_OUT_PER_FAMILY: usize = 3;

pub const LIMITATIONS: [&str; 7] = [
    "Panda collision geometry is meshes and fingertip pads; the vendor sphere sets are not in the repo and under-cover the meshes, so collision and pipeline cases use the UR5e only, on its nine exact original capsule/cylinder primitives (no sphere envelopes).",
    "The reference and oracle entry ABI admits scalar inputs only, so bulk inputs (angles, obstacles, controls, points) are expanded in source from a seed by a frozen 32-bit LCG; that expansion is charged to every compared path.",
    "The point cloud is a synthetic seeded cloud, not a public dataset; frame, crop box, range band and neighbour radius are frozen here.",
    "Rollout is a cost-only joint double integrator with clamped limits and a tool-target cost; it has no collision term and is not a planner.",
    "Precomputed-transform controls draw (cos, sin) by the rational half-angle map of u in [-1, 1), i.e. angles in [-pi/2, pi/2), independent of joint limits.",
    "The mass matrix is the Jacobian-sum form accumulated body by body in binary32; it is compared with Pinocchio CRBA within a frozen tolerance, never bitwise.",
    "Panda finger slides are fixed at 0.04 m and folded into the finger frames (contract fixed_inactive_joints).",
];

#[derive(Clone, Debug, PartialEq)]
pub struct Program {
    pub id: String,
    pub ordinal: usize,
    pub seed: u64,
    pub split: Split,
    pub case: Case,
    pub source: String,
    pub layout: String,
    /// Small inputs checked against the CPU net oracle on the full output.
    pub oracle_inputs: Vec<Vec<u32>>,
    /// Scored sizes; the reference output digest is frozen for each.
    pub scored_inputs: Vec<Vec<u32>>,
}

pub fn arms(root: &Path) -> (Arm, Arm) {
    (
        Arm::of(&model::load(root, "panda")),
        Arm::of(&model::load(root, "ur5e")),
    )
}

/// Scalar parameter names of `run` after `seed`.
pub fn params(family: &str) -> &'static [&'static str] {
    match family {
        "collision" | "pipeline" | "pointcloud" => &["seed", "n", "m"],
        "rollout" => &["seed", "n", "h"],
        _ => &["seed", "n"],
    }
}

/// (oracle sizes, scored sizes) per family, after the seed.
fn sizes(family: &str) -> (Vec<Vec<u32>>, Vec<Vec<u32>>) {
    let v = |rows: &[&[u32]]| rows.iter().map(|r| r.to_vec()).collect::<Vec<_>>();
    match family {
        "fk" | "fkpre" | "jacobian" | "mass" => {
            (v(&[&[1], &[2]]), v(&[&[1], &[32], &[256], &[4096]]))
        }
        "collision" => (
            v(&[&[1, 2], &[2, 3]]),
            v(&[&[1, 16], &[64, 16], &[1024, 16]]),
        ),
        "rollout" => (
            v(&[&[1, 2], &[2, 3]]),
            v(&[
                &[1, 64],
                &[32, 64],
                &[256, 64],
                &[1, 256],
                &[32, 256],
                &[256, 256],
            ]),
        ),
        "pointcloud" => (v(&[&[16, 4], &[64, 8]]), v(&[&[1024, 64], &[65536, 256]])),
        "pipeline" => (
            v(&[&[1, 16], &[2, 24]]),
            v(&[&[1, 256], &[64, 256], &[1024, 256]]),
        ),
        other => panic!("family {other}"),
    }
}

pub fn program_seed(family_index: usize, ordinal: usize) -> u64 {
    Rng::new(ROOT_SEED ^ ((family_index as u64) << 32) ^ ordinal as u64).next()
}

pub fn held_out_ordinals(family_index: usize) -> BTreeSet<usize> {
    let mut r = Rng::new(ROOT_SEED.rotate_left(17) ^ family_index as u64);
    let mut order: Vec<usize> = (0..PER_FAMILY).collect();
    for i in (1..order.len()).rev() {
        let j = (r.next() % (i as u64 + 1)) as usize;
        order.swap(i, j);
    }
    order.into_iter().take(HELD_OUT_PER_FAMILY).collect()
}

pub fn generate(root: &Path) -> Vec<Program> {
    let (panda, ur5e) = arms(root);
    let mut out = Vec::new();
    for (fi, family) in families::FAMILIES.iter().enumerate() {
        let held = held_out_ordinals(fi);
        let cases = families::cases(family);
        assert_eq!(cases.len(), PER_FAMILY);
        for (ordinal, case) in cases.into_iter().enumerate() {
            let arm = match case.arm {
                "panda" => Some(&panda),
                "ur5e" => Some(&ur5e),
                _ => None,
            };
            let seed = program_seed(fi, ordinal);
            let mut r = Rng::new(seed);
            let (oracle, scored) = sizes(family);
            let mut with_seed = |rows: Vec<Vec<u32>>| -> Vec<Vec<u32>> {
                rows.into_iter()
                    .map(|row| {
                        let mut input = vec![r.next() as u32];
                        input.extend(row);
                        input
                    })
                    .collect()
            };
            let oracle_inputs = with_seed(oracle);
            let scored_inputs = with_seed(scored);
            // A `main` makes each program runnable as is, on the first
            // oracle input; it returns 1 when the output is non-empty.
            let mut source = families::source(&case, arm);
            let call: Vec<String> = oracle_inputs[0].iter().map(|x| format!("{x}u32")).collect();
            writeln!(
                source,
                "fn main() -> i32 {{\n    let out = run({});\n    if len(out) > 0u32 {{\n        return 1;\n    }}\n    return 0;\n}}",
                call.join(", ")
            )
            .unwrap();
            out.push(Program {
                id: format!("gnet05r-{family}-{ordinal:02}"),
                ordinal,
                seed,
                split: if held.contains(&ordinal) {
                    Split::HeldOut
                } else {
                    Split::Tuning
                },
                source,
                layout: families::layout(&case, arm),
                case,
                oracle_inputs,
                scored_inputs,
            });
        }
    }
    out
}

pub fn held_out(programs: &[Program]) -> HeldOut {
    HeldOut::of_sources(
        programs
            .iter()
            .filter(|p| p.split == Split::HeldOut)
            .map(|p| p.source.as_str()),
    )
}

/// The only robotics corpus view a tuning path may consume.
pub fn tuning_programs<'a>(programs: &'a [Program], held: &HeldOut) -> Vec<&'a Program> {
    programs
        .iter()
        .filter(|p| p.split == Split::Tuning)
        .filter(|p| held.admit_for_tuning(p.source.as_bytes()).is_ok())
        .collect()
}

pub fn manifest(root: &Path, programs: &[Program]) -> serde_json::Value {
    let held = held_out(programs);
    let (panda, ur5e) = arms(root);
    let arm_json = |a: &Arm, file: &str, sha: &str, tool: &str| {
        serde_json::json!({"file": file, "sha256": sha, "bodies": a.bodies, "joints": a.joints,
            "tool": tool, "collision_geoms": a.geom_names})
    };
    serde_json::json!({
        "schema": "invar-gnet05-robotics-v1",
        "generator": {"name": GENERATOR, "root_seed": format!("{ROOT_SEED:#018x}"), "rng": "splitmix64",
            "per_family": PER_FAMILY, "held_out_per_family": HELD_OUT_PER_FAMILY},
        "families": families::FAMILIES,
        "models": {
            "panda": arm_json(&panda, model::PANDA_MJCF, model::PANDA_SHA256, "hand origin"),
            "ur5e": arm_json(&ur5e, model::UR5E_MJCF, model::UR5E_SHA256, "attachment_site"),
        },
        "math": {
            "constants": super::emit::CONSTANTS.iter().map(|(name, text)| serde_json::json!({
                "name": name, "decimal": text,
                "f32_bits": format!("{:#010x}", super::emit::constant(name).to_bits())})).collect::<Vec<_>>(),
            "trig": "Cephes sinf/cosf polynomials after Cody-Waite reduction by pi/2 (three-part split)",
            "sqrt": "scale into [1, 4) by powers of 4, 5 Newton reciprocal-root steps, no division",
            "stream": "lcg(x) = x * 1664525 + 1013904223 mod 2^32; stream(seed, i) = lcg(i * 2654435761 + seed); unit(x) = (x / 256) * 2^-24",
            "rollout": {"dt": families::ROLLOUT_DT, "umax": families::ROLLOUT_UMAX, "control_weight": families::ROLLOUT_WEIGHT},
            "neighbour_radius_squared": families::NEIGHBOUR_RADIUS2,
            "pipeline_clearance": families::CLEARANCE,
        },
        "limitations": LIMITATIONS,
        "held_out": {"count": held.len(), "normalized_identity_sha256": hex(&held.digest())},
        "entry": "run",
        "programs": programs.iter().map(|p| serde_json::json!({
            "id": p.id,
            "family": p.case.family,
            "arm": p.case.arm,
            "transforms": if p.case.trig { "trigonometric" } else { "precomputed" },
            "variant": families::variant_label(&p.case),
            "ordinal": p.ordinal,
            "seed": format!("{:#018x}", p.seed),
            "split": p.split.label(),
            "file": format!("programs/{}.inv", p.id),
            "source_sha256": hex(&Sha256::digest(p.source.as_bytes())),
            "normalized_sha256": hex(&normalized_sha256(p.source.as_bytes())),
            "params": params(p.case.family),
            "layout": p.layout,
            "oracle_inputs": p.oracle_inputs,
            "scored_inputs": p.scored_inputs,
        })).collect::<Vec<_>>(),
    })
}

/// `tools/bench/baselines/robotics-tables.h`: the binary32 model tables as bit
/// patterns, for the handwritten baselines (generated data, not code).
pub fn baseline_tables(root: &Path) -> String {
    let (panda, ur5e) = arms(root);
    let mut s = String::from(
        "// Generated by tests/gnet05_corpus/issue_gnet_05_robotics.rs (INVAR_GNET05_WRITE=1) from the pinned MJCF\n\
         // models; do not edit. Binary32 tables as bit patterns, word offsets per table.\n\
         #pragma once\n#include <stdint.h>\n\n\
         struct RoboticsModel {\n  const char* name;\n  const uint32_t* words;\n  uint32_t count;\n  \
         uint32_t nb, nj, ng, tb;\n  uint32_t par, fix, ax, jt, jx, lo, span, hi, mv, bt, ar, gi, gt, tool;\n};\n",
    );
    for a in [&panda, &ur5e] {
        let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<u32>>();
        let jx: Vec<u32> = a.joint_body.iter().flat_map(|&(b, x)| [b, x]).collect();
        let tables: [(&str, Vec<u32>); 14] = [
            ("par", a.parent.clone()),
            ("fix", bits(&a.fixed)),
            ("ax", a.axis.clone()),
            ("jt", a.joint.clone()),
            ("jx", jx),
            ("lo", bits(&a.lo)),
            ("span", bits(&a.span)),
            ("hi", bits(&a.hi)),
            ("mv", a.moves.clone()),
            ("bt", bits(&a.inertial)),
            ("ar", bits(&a.armature)),
            ("gi", a.geom_index.clone()),
            ("gt", bits(&a.geom)),
            ("tool", bits(&a.tool)),
        ];
        let upper = a.name.to_uppercase();
        let mut words: Vec<u32> = Vec::new();
        let mut offsets = Vec::new();
        for (_, t) in &tables {
            offsets.push(words.len());
            words.extend(t);
        }
        writeln!(
            s,
            "\nstatic const uint32_t {upper}_WORDS[{}] = {{",
            words.len().max(1)
        )
        .unwrap();
        for chunk in words.chunks(8) {
            let row: Vec<String> = chunk.iter().map(|w| format!("0x{w:08x}u")).collect();
            writeln!(s, "  {},", row.join(", ")).unwrap();
        }
        writeln!(
            s,
            "}};\nstatic const RoboticsModel {upper} = {{\"{}\", {upper}_WORDS, {}, {}, {}, {}, {}, {}}};",
            a.name,
            words.len(),
            a.bodies,
            a.joints,
            a.geoms(),
            a.tool_body,
            offsets.iter().map(|o| o.to_string()).collect::<Vec<_>>().join(", ")
        )
        .unwrap();
    }
    s
}
